#!/usr/bin/env python
"""T127：门控敏感度 |a_k| 为什么随族规模递减？—— 直接测「族日度 IC 的噪声」这条解释。

背景（T125 之后的实测规律）
--------------------------
四个门控臂（手工 12 族 / 数据驱动 6 簇 × 开关 group-norm）全部满足
    |a_k| ∝ n_k^(−1.45)     spearman(n, |a|) = −0.92 ~ −1.00，r² 0.75~0.84
其中 a 是 scalar 门控的**全部**参数（`w = K·softmax(a·s)`，s = macro_m1m2_gap）。
**开不开 group-norm 都一样** ⇒ 驱动 a 的量与「族输出幅度」无关。
而 rank IC 是尺度不变的，所以嫌疑落在**族日度 IC 序列的统计性质**上。

要检验的因果链
--------------
    族越小 → 族日度 IC 序列越抖（成员少，平均不掉噪声）
          → 在有限个独立宏观区块上与标量 s 的**样本相关**越容易偏离 0
          → 门为拟合这个偶然相关而给该族更大的 |a|
若成立，|a_k| 应当与 |corr(IC_k(day), s(day))| 强相关，**且该相关本身随 n 递减**；
并且用**验证段**重算同一个 corr 时应大幅衰减（偶然相关不外推）。
若不成立（|a| 与 corr 无关），则「小族噪声」这条解释被证伪，据实记录。

⚠ 这是**观测性**测量：只做前向，不训练，不改变任何已有结论。
   T125 已经把门控轴 2×2 填满并封口（见台账），本脚本只回答「为什么」。

用法：
  python scripts/exp/diag_gate_sensitivity_vs_size.py --model models/nam_gate/T116_macro_gate_s42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from scipy.stats import linregress, spearmanr

from config import DATABASE_PATH
from config.factor_config import TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.nam_gate_model import NAMGateModel
from core.factors.regime_features import build_regime_matrix
from core.factors.train_ml_model import MLModelTrainer
from scripts.exp.exp_nam_gate import _day_slices, _prepare_fold, _rank_ic_columns


def group_daily_ic(model, X, ret, dates, M, max_days=0):
    """[n_days, K] 族日度贡献 IC + 每日的门控标量 s。

    族输出取**归一化前**的 group_sums：rank IC 尺度不变，所以这样与
    group-norm 与否无关，正是本测量要的「尺度无关量」。
    """
    dev = torch.device(model.device)
    mean = None if model.input_mean is None else np.asarray(model.input_mean, dtype=np.float32)
    std = None if model.input_std is None else np.asarray(model.input_std, dtype=np.float32)
    sl = _day_slices(dates)
    if max_days and len(sl) > max_days:
        keep = np.unique(np.linspace(0, len(sl) - 1, max_days).round().astype(int))
        sl = [sl[i] for i in keep]
    K = len(model.group_names)
    out = np.full((len(sl), K), np.nan, dtype=np.float64)
    s_col = np.full(len(sl), np.nan, dtype=np.float64)
    si = model.regime_cols.index(model.gate_scalar_col)
    model.net.eval()
    with torch.no_grad():
        for i, (a, b) in enumerate(sl):
            if b - a < 20:
                continue
            xt = torch.as_tensor(np.asarray(X[a:b], dtype=np.float32), device=dev)
            if mean is not None:
                xt = (xt - torch.as_tensor(mean, device=dev)) / torch.as_tensor(std, device=dev)
            c = model.net.experts(xt)
            gs = (c @ model.net.group_onehot).cpu().numpy().astype(np.float64)
            out[i] = _rank_ic_columns(gs, np.asarray(ret[a:b], dtype=np.float64))
            s_col[i] = float(M[a][si])
    ok = ~np.all(np.isnan(out), axis=1)
    return out[ok], s_col[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/nam_gate/T116_macro_gate_s42')
    ap.add_argument('--cache-dir', default='database/system_data/factors_cache_2026-08-18-idxrel')
    ap.add_argument('--stocks', type=int, default=6000)
    ap.add_argument('--years', type=int, default=13)
    ap.add_argument('--end', default='2022-09-05')
    ap.add_argument('--train-days', type=int, default=900)
    ap.add_argument('--out', default='diagnose_output/T127_gate_sensitivity.json')
    a = ap.parse_args()

    m = NAMGateModel()
    m.load_model(os.path.join(a.model, 'nam_gate_factor_model.pkl'))
    if getattr(m.net.gate, 'a', None) is None:
        raise SystemExit('该存档不是 scalar 门控，没有 a 参数')
    avec = np.abs(m.net.gate.a.detach().cpu().numpy())
    gid = np.asarray(m.group_ids)
    n_k = np.array([(gid == k).sum() for k in range(len(m.group_names))], dtype=float)
    print(f'{a.model}: {len(m.feature_names)} 列 / {len(m.group_names)} 族，'
          f'门控标量 = {m.gate_scalar_col}')

    end_dt = datetime.strptime(a.end, '%Y-%m-%d')
    start = (end_dt - timedelta(days=365 * a.years)).strftime('%Y-%m-%d')
    TrainingConfig.FUTURE_DAYS = 7
    tr = MLModelTrainer(db_path=DATABASE_PATH, cache_dir=a.cache_dir)
    mgr = BaostockDataManager()
    codes = mgr.get_stock_list_from_db()['code'].tolist()[:a.stocks]
    mgr.close()
    sd = tr.load_label_data(codes, start, a.end)
    ds = tr.prepare_dataset(sd, train_start_date=start, train_end_date=a.end,
                            include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
                            n_jobs=4, target_features=None, use_factor_cache_only=True)
    regime = build_regime_matrix(DATABASE_PATH)
    fold = _prepare_fold(tr, ds, list(m.feature_names), 0.8, 1.0, regime, target='returns')
    del sd, ds

    res = {'model': a.model, 'groups': list(m.group_names),
           'n_k': n_k.tolist(), 'abs_a': avec.tolist()}
    seg = {}
    for tag, (X, r, d, M, md) in {
        'train': (fold['X_train'], fold['ret_train'], fold['d_train'], fold['M_train'], a.train_days),
        'val':   (fold['X_val'],   fold['ret_val'],   fold['d_val'],   fold['M_val'],   0),
    }.items():
        ic, s = group_daily_ic(m, X, r, d, M, max_days=md)
        sd_ic = np.nanstd(ic, axis=0)
        corr = np.array([np.corrcoef(ic[:, k][~np.isnan(ic[:, k])],
                                     s[~np.isnan(ic[:, k])])[0, 1]
                         for k in range(ic.shape[1])])
        seg[tag] = {'n_days': int(ic.shape[0]), 'ic_std': sd_ic.tolist(),
                    'corr_ic_s': corr.tolist(), 'ic_mean': np.nanmean(ic, axis=0).tolist()}
        print(f'\n=== {tag} 段（{ic.shape[0]} 日）===')
        print(f'{"族":12s} {"n":>4s} {"|a|":>7s} {"IC日度σ":>9s} {"corr(IC,s)":>11s} {"IC均值":>9s}')
        for k in np.argsort(-avec):
            print(f'{m.group_names[k]:12s} {int(n_k[k]):4d} {avec[k]:7.3f} '
                  f'{sd_ic[k]:9.4f} {corr[k]:+11.4f} {np.nanmean(ic[:, k]):+9.4f}')
        for lab, v in [('n_k', n_k), ('IC日度σ', sd_ic), ('|corr(IC,s)|', np.abs(corr))]:
            rr = spearmanr(v, avec)
            print(f'  spearman({lab}, |a|) = {rr.statistic:+.3f} (p={rr.pvalue:.4f})')
        lr = linregress(np.log(n_k), np.log(sd_ic + 1e-12))
        print(f'  log(IC日度σ) ~ log(n) 斜率 = {lr.slope:+.3f} (r²={lr.rvalue ** 2:.3f})'
              f'   [独立成员平均则应为 −0.5]')
    res['segments'] = seg
    # 关键：训练段的 corr 在验证段还剩多少（偶然相关不外推的直接证据）
    ct, cv = np.array(seg['train']['corr_ic_s']), np.array(seg['val']['corr_ic_s'])
    print(f'\n=== corr(IC,s) 的外推性 ===')
    print(f'{"族":12s} {"n":>4s} {"train":>9s} {"val":>9s} {"保留":>8s}')
    for k in np.argsort(-np.abs(ct)):
        keep = cv[k] / ct[k] if abs(ct[k]) > 1e-9 else np.nan
        print(f'{m.group_names[k]:12s} {int(n_k[k]):4d} {ct[k]:+9.4f} {cv[k]:+9.4f} {keep:8.2f}')
    same = float(np.mean(np.sign(ct) == np.sign(cv)))
    print(f'  符号一致率 = {same:.1%}（纯偶然应约 50%）')
    print(f'  |corr| 均值: train {np.abs(ct).mean():.4f} → val {np.abs(cv).mean():.4f}')
    res['sign_agreement'] = same

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
