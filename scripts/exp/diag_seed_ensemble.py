#!/usr/bin/env python
"""T131：**多种子等权集成**作为生产载体（零训练）。

背景
----
[[ensemble-beats-single]] 记着：xgb+lgb 截面排名后**等权**平均，回测 +17% vs 单模型 +5%，
且权重必须 0.5/0.5（按 IR 加权已被证伪）。NAM 侧一直只用单种子（生产 = T115_idxrel_s42），
**多种子集成从未测过** —— 这是"载体"轴上唯一没填的格子。

预注册判据（出结果前写死）
--------------------------
1. **必须赢最幸运的那个单种子**，不是四种子均值。否则"集成的增益"只是
   "我事后挑中了好种子"的另一种说法。晋级线仍是 Δ ≥ +0.005。
2. 两把尺子都报：全截面 holdout rank IC（老协议）**和**头部 top-K 超额
   （T130 的新尺子，见 [[head-ruler-is-blind-in-full-cross-section-ic]]）。
3. **跌日分层否决门优先**（[[regime-stratified-ic-gate]]）：集成相对**在跑的那个**
   种子，跌日 Δ 必须 ≥0。
4. 另报**种子间离散度**。集成的真实价值可能不是"更强"而是"不用抽签" ——
   若单种子间差异远大于集成的增益，那么消掉抽签风险本身就是理由，据实记录。

集成口径与 xgb+lgb 对齐：**逐日截面 rank 后等权平均**（不是平均原始分数 ——
不同种子的分数尺度不可比，NAM 的 bias 更是逐存档不同）。

用法：
  python scripts/exp/diag_seed_ensemble.py \
      --models models/nam_gate/T122_prod9y_s42,...,s37 --years 9 --end 2026-08-10
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
from scipy.stats import rankdata

from config import DATABASE_PATH
from config.factor_config import TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.nam_gate_model import NAMGateModel
from core.factors.regime_features import build_regime_matrix
from core.factors.train_ml_model import MLModelTrainer
from scripts.exp.diag_frozen_expert_gate import precompute_group_sums
from scripts.exp.exp_nam_gate import _day_slices, _prepare_fold


def load_group_sums(dirs, a, feats, gid, K):
    """取数 + 逐存档算冻结族输出；结果缓存到 npz（重跑判读口径可跳过 ~25min）。"""
    if a.cache_npz and os.path.exists(a.cache_npz):
        z = np.load(a.cache_npz, allow_pickle=False)
        if int(z['n_models']) == len(dirs):
            print(f'  已复用缓存 {a.cache_npz}')
            return ([z[f'S_va_{i}'] for i in range(len(dirs))], z['ret_va'],
                    [(int(s), int(e)) for s, e in z['va_days']])
        print(f'  缓存里的存档数 {int(z["n_models"])} ≠ {len(dirs)}，重算')
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
                            n_jobs=a.n_jobs, target_features=None, use_factor_cache_only=True)
    regime = build_regime_matrix(DATABASE_PATH)
    fold = _prepare_fold(tr, ds, feats, a.train_fraction, 1.0, regime, target='returns')
    del sd, ds
    print(f'  切折 split_date={fold["split_date"]}')
    S_va = []
    for d in dirs:
        mi = NAMGateModel()
        mi.load_model(os.path.join(d, 'nam_gate_factor_model.pkl'))
        S_va.append(precompute_group_sums(mi, fold['X_val'], gid, K).cpu().numpy())
        del mi
        torch.cuda.empty_cache()
    ret_va = np.asarray(fold['ret_val'])
    va_days = _day_slices(fold['d_val'])
    if a.cache_npz:
        os.makedirs(os.path.dirname(os.path.abspath(a.cache_npz)), exist_ok=True)
        np.savez(a.cache_npz, n_models=len(dirs), ret_va=ret_va,
                 va_days=np.array(va_days), **{f'S_va_{i}': v for i, v in enumerate(S_va)})
        print(f'  已缓存 {a.cache_npz}')
    return S_va, ret_va, va_days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(
        f'models/nam_gate/T115_idxrel_s{s}' for s in (42, 11, 23, 37)))
    ap.add_argument('--incumbent', default=None,
                    help='当前在跑的那个存档（默认取 --models 第一个）；跌日否决门对它判')
    ap.add_argument('--cache-dir', default='database/system_data/factors_cache_2026-08-18-idxrel')
    ap.add_argument('--cache-npz', default='')
    ap.add_argument('--stocks', type=int, default=6000)
    ap.add_argument('--years', type=int, default=13)
    ap.add_argument('--end', default='2022-09-05')
    ap.add_argument('--train-fraction', type=float, default=0.8)
    ap.add_argument('--select-holdout', type=float, default=0.4)
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--head-n', type=int, default=200)
    ap.add_argument('--out', default='diagnose_output/T131_seed_ensemble.json')
    a = ap.parse_args()

    dirs = [d.strip() for d in a.models.split(',') if d.strip()]
    tags = [d.rstrip('/').split('/')[-1] for d in dirs]
    inc = a.incumbent or dirs[0]
    m0 = NAMGateModel()
    m0.load_model(os.path.join(dirs[0], 'nam_gate_factor_model.pkl'))
    feats, gid, K = list(m0.feature_names), np.asarray(m0.group_ids), len(m0.group_names)
    print(f'{len(dirs)} 个存档 / {len(feats)} 列 / {K} 族，窗口 {a.years}y → {a.end}')

    S_va, ret_va, va_days = load_group_sums(dirs, a, feats, gid, K)
    days = [(s, e) for s, e in va_days if e - s >= 30]
    n_sel = int(round(len(days) * (1.0 - a.select_holdout)))
    print(f'验证 {len(days)} 日，holdout（后 {a.select_holdout:.0%}）= {len(days) - n_sel} 日')

    names = tags + ['ENSEMBLE']
    ic = {k: np.full(len(days), np.nan) for k in names}
    ex = {k: np.full(len(days), np.nan) for k in names}
    mkt = np.full(len(days), np.nan)
    for i, (s, e) in enumerate(days):
        r = np.asarray(ret_va[s:e], dtype=np.float64)
        rr = rankdata(r)
        mkt[i] = r.mean()
        ranks = []
        for j, tg in enumerate(tags):
            sc = S_va[j][s:e].astype(np.float64).sum(1)
            ranks.append(rankdata(sc))
            ic[tg][i] = np.corrcoef(ranks[-1], rr)[0, 1]
            ex[tg][i] = _head_excess(sc, r, a.head_n, a.topk)
        mm = np.mean(ranks, axis=0)          # 截面 rank 后等权平均（与 xgb+lgb 同口径）
        ic['ENSEMBLE'][i] = np.corrcoef(rankdata(mm), rr)[0, 1]
        ex['ENSEMBLE'][i] = _head_excess(mm, r, a.head_n, a.topk)

    sl = slice(n_sel, None)
    up, dn = mkt[sl] > 0, mkt[sl] <= 0
    print(f'\n{"载体":26s} {"holdout IC":>11s} {"IC涨日":>8s} {"IC跌日":>8s} '
          f'{"top%d超额" % a.topk:>10s} {"超额涨日":>9s} {"超额跌日":>9s}')
    res = {}
    for k in names:
        res[k] = {'ic': float(np.nanmean(ic[k][sl])),
                  'ic_up': float(np.nanmean(ic[k][sl][up])),
                  'ic_down': float(np.nanmean(ic[k][sl][dn])),
                  'ex': float(np.nanmean(ex[k][sl])),
                  'ex_up': float(np.nanmean(ex[k][sl][up])),
                  'ex_down': float(np.nanmean(ex[k][sl][dn]))}
        mark = '  ← 集成' if k == 'ENSEMBLE' else ('  ← 在跑' if inc.endswith(k) else '')
        v = res[k]
        print(f'{k:26s} {v["ic"]:+11.5f} {v["ic_up"]:+8.4f} {v["ic_down"]:+8.4f} '
              f'{v["ex"]:+10.4f} {v["ex_up"]:+9.4f} {v["ex_down"]:+9.4f}{mark}')

    best_ic = max(tags, key=lambda k: res[k]['ic'])
    best_ex = max(tags, key=lambda k: res[k]['ex'])
    mean_ic = float(np.mean([res[k]['ic'] for k in tags]))
    sd_ic = float(np.std([res[k]['ic'] for k in tags], ddof=1))
    sd_ex = float(np.std([res[k]['ex'] for k in tags], ddof=1))
    inc_tag = [t for t in tags if inc.endswith(t)]
    inc_tag = inc_tag[0] if inc_tag else tags[0]
    E = res['ENSEMBLE']

    print(f'\n★ 判据 1（必须赢最幸运单种子，线 +0.005）')
    print(f'   IC:   集成 {E["ic"]:+.5f} vs 最幸运 {best_ic} {res[best_ic]["ic"]:+.5f} '
          f'⇒ Δ {E["ic"] - res[best_ic]["ic"]:+.5f}'
          f'   {"过" if E["ic"] - res[best_ic]["ic"] >= 0.005 else "不过"}')
    print(f'   头部: 集成 {E["ex"]:+.4f} vs 最幸运 {best_ex} {res[best_ex]["ex"]:+.4f} '
          f'⇒ Δ {E["ex"] - res[best_ex]["ex"]:+.4f}')
    print(f'★ 判据 2（vs 在跑的 {inc_tag}，这才是换不换的实际比较）')
    print(f'   IC Δ {E["ic"] - res[inc_tag]["ic"]:+.5f}   '
          f'头部 Δ {E["ex"] - res[inc_tag]["ex"]:+.4f}')
    print(f'★ 判据 3（跌日否决门 vs 在跑的）: IC Δ跌日 '
          f'{E["ic_down"] - res[inc_tag]["ic_down"]:+.5f}   超额 Δ跌日 '
          f'{E["ex_down"] - res[inc_tag]["ex_down"]:+.4f}   '
          f'{"过" if E["ex_down"] - res[inc_tag]["ex_down"] >= 0 else "不过"}')
    print(f'★ 判据 4（抽签风险）: 单种子 IC σ={sd_ic:.5f}（极差 '
          f'{max(res[k]["ic"] for k in tags) - min(res[k]["ic"] for k in tags):.5f}）、'
          f'头部超额 σ={sd_ex:.4f}（极差 '
          f'{max(res[k]["ex"] for k in tags) - min(res[k]["ex"] for k in tags):.4f}）')
    print(f'   ⇒ 若极差 >> 集成增益，「不用抽签」本身就是换的理由，别只看点估计。')

    out = {'models': dirs, 'incumbent': inc, 'window': f'{a.years}y→{a.end}',
           'n_holdout_days': len(days) - n_sel, 'per_carrier': res,
           'seed_sd': {'ic': sd_ic, 'ex': sd_ex}, 'seed_mean_ic': mean_ic,
           'luckiest': {'ic': best_ic, 'ex': best_ex}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {a.out}')
    return 0


def _head_excess(score, r, head_n, k):
    """先按分数取前 head_n 候选池，再在池内取前 k —— 与 T130 同口径。"""
    n = len(score)
    pool = np.argpartition(score, -head_n)[-head_n:] if n > head_n else np.arange(n)
    kk = min(k, len(pool))
    top = pool[np.argpartition(score[pool], -kk)[-kk:]]
    return float(r[top].mean() - r.mean())


if __name__ == '__main__':
    sys.exit(main())
