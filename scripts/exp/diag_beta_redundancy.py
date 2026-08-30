#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
beta 家族冗余/共线性校验（不污染共享缓存）
=========================================
目的：IC 诊断显示信号集中在 corr/r2（与市场变量的协同度），而 idio_vol/alpha
大概率是「个股自身波动率 / 短期反转」的代理（已被基线 vol/动量特征覆盖）。

本脚本校验：候选「协同度」特征与基线已有特征的最大 |相关系数|，判断其是否为
可叠加的新信号，还是已有特征的线性近似。

- 候选新特征：所有 msens_*_corr_{20,60} / msens_*_r2_{20,60}（协同度族）。
- 基线对照特征：idx_corr_20/idx_rs_20/idx_idio_vol_20/idx_beta_60（已有 idx_*）、
  return_5d/10d/20d/60d、momentum_5d/20d、atr_14、natr_28、price_volatility_20/60。
- 输出：每个候选特征的 max|corr| 与最相关的基线特征名。

用法
  python scripts/exp/diag_beta_redundancy.py [--pool 800]
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from core.factors import market_sensitivity as ms

DB_DAILY = os.path.join(ROOT, 'database', 'stock_daily.db')
DB_META = os.path.join(ROOT, 'database', 'stock_meta.db')
CACHE = os.path.join(ROOT, 'database', 'system_data', 'factors_cache')

# 基线对照列（从缓存 parquet 取）
EXIST_COLS = [
    'idx_corr_20', 'idx_rs_20', 'idx_idio_vol_20', 'idx_beta_60',
    'return_5d', 'return_10d', 'return_20d', 'return_60d',
    'momentum_5d', 'momentum_20d',
    'atr_14', 'natr_28', 'price_volatility_20', 'price_volatility_60',
]

# 候选新特征：协同度族（corr / r2）
CAND_PREFIX = ('msens_idxret', 'msens_idxvol', 'msens_breadth_up', 'msens_breadth_ma',
               'msens_marginflow', 'msens_finbuyflow', 'msens_basis_if',
               'msens_shibor3m', 'msens_cn10y')
CAND_SUFFIX = ('corr_20', 'corr_60', 'r2_20', 'r2_60')


def pick_pool(pool, start, end):
    conn = sqlite3.connect(DB_DAILY, timeout=60.0)
    try:
        rows = conn.execute(
            """SELECT code FROM daily_data WHERE date>=? AND date<?
               GROUP BY code HAVING COUNT(*)>=400 ORDER BY COUNT(*) DESC LIMIT ?""",
            (start, end, pool)).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=800)
    ap.add_argument('--start', default='2019-01-01')
    ap.add_argument('--end', default='2022-01-01')
    a = ap.parse_args()

    codes = pick_pool(a.pool, a.start, a.end)
    idx = ms.load_index_returns(DB_DAILY)
    glob = ms.load_market_series(DB_META)
    idx_vol = idx[ms.MARKET_INDEX].rolling(20, min_periods=13).std() if idx else None

    # 收集 (date, stock) 长表
    rows = []
    for code in codes:
        # 基线特征
        f = glob_cache = os.path.join(CACHE, f'{code}_factors.parquet')
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f, columns=['date'] + EXIST_COLS)
        df['date'] = df['date'].astype(str)
        df = df.drop_duplicates('date').set_index('date').sort_index()
        if len(df) < 200:
            continue
        r_i = pd.to_numeric(
            pd.read_sql_query('SELECT date, pctChg FROM daily_data WHERE code=? '
                              'AND date>=? AND date<? ORDER BY date ASC',
                              sqlite3.connect(DB_DAILY),
                              params=(code, a.start, a.end)).set_index('date')['pctChg'],
            errors='coerce') / 100.0
        r_i = r_i.reindex(df.index)
        # 新特征
        for entry in ms.SENS_REGISTRY:
            prefix = entry['prefix']
            if not prefix.startswith(CAND_PREFIX):
                continue
            if entry['kind'] == 'index_ret':
                x = ms.board_series(code, idx)
            elif entry['kind'] == 'index_vol':
                x = idx_vol
            else:
                if entry['src'] not in glob:
                    continue
                x = glob[entry['src']]
            if entry['diff']:
                x = x.diff()
            sub = ms.compute_sensitivity(r_i, x)
            for col in sub.columns:
                if col in CAND_SUFFIX:
                    full = f'{prefix}_{col}'
                    df[full] = sub[col].reindex(df.index).values
        df = df.reset_index().rename(columns={'index': 'date'})
        df['code'] = code
        rows.append(df)
    big = pd.concat(rows, ignore_index=True)
    print(f'[load] {len(big)} 行 (date×stock)，候选特征 '
          f'{sum(1 for c in big.columns if c.startswith(CAND_PREFIX) and c.split("_")[-1] in ("20","60") and ("corr" in c or "r2" in c))}')

    # 仅保留 corr/r2 候选 + 基线
    cand_cols = [c for c in big.columns if c.startswith(CAND_PREFIX)
                 and c.split('_')[-1] in ('20', '60')
                 and ('corr' in c or 'r2' in c)]
    use = [c for c in EXIST_COLS if c in big.columns] + cand_cols
    mat = big[use].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    finite = np.isfinite(mat)
    col_means = np.array([mat[finite[:, j], j].mean() if finite[:, j].any() else 0.0
                          for j in range(mat.shape[1])])
    mat = np.where(finite, mat, col_means)
    corr = np.corrcoef(mat, rowvar=False)
    names = use

    print(f'\n=== 候选协同度特征 vs 基线：max|corr| 与最相关基线特征 ===')
    print(f'{"candidate":32s} {"max|corr|":>9s} {"vs_feature":20s} {"direction":>10s}')
    recs = []
    for c in cand_cols:
        ci = names.index(c)
        best, bj = 0.0, -1
        for j, nj in enumerate(names):
            if nj in EXIST_COLS:
                v = corr[ci, j]
                if abs(v) > abs(best):
                    best, bj = v, j
        recs.append((c, best, names[bj] if bj >= 0 else ''))
    for c, best, bj in sorted(recs, key=lambda r: abs(r[1])):
        flag = '  <-- 近冗余' if abs(best) > 0.7 else ('  ~中等' if abs(best) > 0.4 else '')
        print(f'{c:32s} {best:>+9.3f} {bj:20s} {("+" if best>0 else "-"):>10s}{flag}')

    # 汇总
    near = [r for r in recs if abs(r[1]) > 0.7]
    mid = [r for r in recs if 0.4 < abs(r[1]) <= 0.7]
    low = [r for r in recs if abs(r[1]) <= 0.4]
    print(f'\n近冗余(|corr|>0.7): {len(near)} | 中等(0.4-0.7): {len(mid)} | 低相关(<0.4, 可叠加): {len(low)}')
    if low:
        print('低相关候选（建议物化的新信号）：')
        for c, best, bj in sorted(low, key=lambda r: r[0]):
            print(f'  {c:32s} max|corr|={best:+.3f}(vs {bj})')


if __name__ == '__main__':
    main()
