#!/usr/bin/env python
"""把任意几个 NAM 存档放到**同一个时间窗**上评，用于跨窗口可比。

为什么必须有它：`rank_ic_holdout` 是各自窗口的后 40%，不同配方的 holdout 落在
**不同日历期**上，直接对比等于拿两个不同难度的考卷比分数。T122（生产模型训至最新日）
的 holdout 是 2026 年那 145 天、T115 的是 2022 上半年 —— 差 0.062 里有多少是
「模型变差」、多少是「时段变难」，只有把它们放同一张卷子上才知道。

诚实性说明：把 T115（训到 2020-12-18）放到 2025~2026 段上评，对它是**干净样本外**
（领先 4~5 年）；把 T122（训到 2025-01-20）放上去只领先 0~19 个月。
所以 T122 在**时效上占优**，这个比较对它有利；若它仍不赢，问题就是真的。

用法：
  python scripts/exp/eval_models_on_window.py \
      --models models/nam_gate/T115_idxrel_s42,models/nam_gate/T122_prod9y_s42 \
      --years 9 --end 2026-08-10 --folds 0.8:1.0 --select-holdout 0.4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from config import DATABASE_PATH
from config.factor_config import TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.nam_gate_model import NAMGateModel
from core.factors.regime_features import build_regime_matrix
from core.factors.train_ml_model import MLModelTrainer
from scripts.exp.exp_nam_gate import _day_slices, _prepare_fold, _rank_ic_columns


def _score_days(model, X, M, day_slices):
    """逐日打分（走 forward_day，与训练/回测同一路径）。"""
    dev = torch.device(model.device)
    mean = None if model.input_mean is None else np.asarray(model.input_mean, dtype=np.float32)
    std = None if model.input_std is None else np.asarray(model.input_std, dtype=np.float32)
    model.net.eval()
    out = []
    with torch.no_grad():
        for s, e in day_slices:
            xt = torch.as_tensor(np.asarray(X[s:e], dtype=np.float32), device=dev)
            if mean is not None:
                xt = (xt - torch.as_tensor(mean, device=dev)) / torch.as_tensor(std, device=dev)
            mt = torch.as_tensor(np.asarray(M[s], dtype=np.float32), device=dev)
            sc, _, _ = model.net.forward_day(xt, mt)
            out.append(sc.cpu().numpy().astype(np.float64))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True, help='逗号分隔的存档目录')
    ap.add_argument('--cache-dir', default='database/system_data/factors_cache_2026-08-18-idxrel')
    ap.add_argument('--stocks', type=int, default=6000)
    ap.add_argument('--years', type=int, default=9)
    ap.add_argument('--end', default='2026-08-10')
    ap.add_argument('--train-fraction', type=float, default=0.8)
    ap.add_argument('--select-holdout', type=float, default=0.4)
    ap.add_argument('--out', default='diagnose_output/T123_cross_window_eval.json')
    a = ap.parse_args()

    dirs = [d.strip() for d in a.models.split(',') if d.strip()]
    models = {}
    feats0 = None
    for d in dirs:
        m = NAMGateModel()
        m.load_model(os.path.join(d, 'nam_gate_factor_model.pkl'))
        if feats0 is None:
            feats0 = list(m.feature_names)
        elif list(m.feature_names) != feats0:
            raise SystemExit(f'{d} 的特征名与首个存档不一致，无法同窗比较')
        models[os.path.basename(d)] = m
    print(f'待评存档 {len(models)} 个，面板 {len(feats0)} 列')

    end_dt = datetime.strptime(a.end, '%Y-%m-%d')
    start = (end_dt - timedelta(days=365 * a.years)).strftime('%Y-%m-%d')
    TrainingConfig.FUTURE_DAYS = 7
    trainer = MLModelTrainer(db_path=DATABASE_PATH, cache_dir=a.cache_dir)
    mgr = BaostockDataManager()
    codes = mgr.get_stock_list_from_db()['code'].tolist()[:a.stocks]
    mgr.close()
    sd = trainer.load_label_data(codes, start, a.end)
    ds = trainer.prepare_dataset(
        sd, train_start_date=start, train_end_date=a.end,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        n_jobs=4, target_features=None, use_factor_cache_only=True)
    regime = build_regime_matrix(DATABASE_PATH)
    fold = _prepare_fold(trainer, ds, feats0, a.train_fraction, 1.0, regime, target='returns')
    del sd, ds

    va = _day_slices(fold['d_val'])
    n_sel = max(1, int(round(len(va) * (1.0 - a.select_holdout))))
    print(f'验证段 {len(va)} 日（起 {fold["split_date"]}）→ 选型 {n_sel} / holdout {len(va)-n_sel}')

    # 基准涨跌日：当日全池等权收益符号
    day_ret = np.array([float(np.nanmean(fold['ret_val'][s:e])) for s, e in va])

    res = {'window': {'start': start, 'end': a.end, 'split_date': str(fold['split_date']),
                      'n_val_days': len(va), 'n_holdout_days': len(va) - n_sel},
           # 逐日市场方向存盘：跌日否决门要能**只在 holdout 上**分层，
           # 而 'down'/'up' 是全验证段的（含各存档自己的选型段，会互相污染）。
           'day_ret': [float(v) for v in day_ret],
           'n_select_days': n_sel, 'models': {}}
    print(f'\n{"存档":24s} {"全段IC":>9s} {"holdoutIC":>10s} {"跌日IC":>9s} {"涨日IC":>9s}')
    for name, m in models.items():
        sc = _score_days(m, fold['X_val'], fold['M_val'], va)
        ics = np.array([_rank_ic_columns(sc[i][:, None],
                                        np.asarray(fold['ret_val'][s:e], dtype=np.float64))[0]
                        for i, (s, e) in enumerate(va)])
        ho = ics[n_sel:]
        dn = ics[day_ret < 0]
        up = ics[day_ret >= 0]
        row = {'all': float(np.nanmean(ics)), 'holdout': float(np.nanmean(ho)),
               'down': float(np.nanmean(dn)), 'up': float(np.nanmean(up)),
               'down_holdout': float(np.nanmean(ics[n_sel:][day_ret[n_sel:] < 0])),
               'up_holdout': float(np.nanmean(ics[n_sel:][day_ret[n_sel:] >= 0])),
               'n_down': int((day_ret < 0).sum()), 'n_up': int((day_ret >= 0).sum()),
               'holdout_daily': [None if np.isnan(v) else float(v) for v in ho]}
        res['models'][name] = row
        print(f'{name:24s} {row["all"]:+9.5f} {row["holdout"]:+10.5f} '
              f'{row["down"]:+9.5f} {row["up"]:+9.5f}')

    # 同窗配对：任意两个存档在 holdout 上的逐日配对 t
    names = list(models)
    if len(names) > 1:
        print(f'\n同窗 holdout 逐日配对（n={len(va)-n_sel} 日）：')
        base = names[0]
        b = np.array(res['models'][base]['holdout_daily'], dtype=float)
        for n in names[1:]:
            c = np.array(res['models'][n]['holdout_daily'], dtype=float)
            d = c - b
            d = d[~np.isnan(d)]
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 2 else np.nan
            t = d.mean() / se if se and se > 0 else np.nan
            print(f'  {n} − {base}: Δ均值 {d.mean():+.5f}  t={t:+.2f}  '
                  f'胜日 {(d>0).mean()*100:.1f}%')
            res.setdefault('paired', {})[f'{n}-{base}'] = {
                'mean': float(d.mean()), 't': float(t), 'win_ratio': float((d > 0).mean())}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
