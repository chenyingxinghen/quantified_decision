#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
beta 族「时序形态派生」冗余/共线性校验（不污染共享缓存）
=====================================================

目的：diag_beta_morph 显示斜率/稳定性/变化量携带强信号（22/54 显著，top |t| 8~14）。
本脚本用池化相关矩阵确认每个形态派生特征与基线已有 247 列的最大 |相关系数|，
判定其是否为可叠加新信号，还是被基线动量/波动率特征覆盖。

- 候选：9 变量 × {slope5_corr, slope10_corr, slope5_r2, stab20_corr, change_corr, change_r2} = 54。
- 基线：factors_cache 全部 247 列。
- 方法：每股票对齐日期，堆叠成 (date×stock) 长表，整体 corrcoef，报候选 vs 基线 max|corr| 及最相关列。

用法
  python scripts/exp/diag_beta_morph_redundancy.py [--pool 800] [--start 2019-01-01] [--end 2022-01-01]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from core.factors import market_sensitivity as ms

DB_DAILY = os.path.join(ROOT, 'database', 'stock_daily.db')
CACHE = os.path.join(ROOT, 'database', 'system_data', 'factors_cache')


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


def load_stock(code, start, end):
    conn = sqlite3.connect(DB_DAILY, timeout=60.0)
    try:
        d = pd.read_sql_query(
            'SELECT date, pctChg FROM daily_data WHERE code=? '
            'AND date>=? AND date<? ORDER BY date ASC',
            conn, params=(code, start, end))
    finally:
        conn.close()
    if d.empty:
        return None
    d['date'] = d['date'].astype(str)
    d = d.drop_duplicates('date').set_index('date').sort_index()
    return pd.to_numeric(d['pctChg'], errors='coerce') / 100.0


def rolling_slope(y: pd.Series, L: int) -> pd.Series:
    arr = y.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n < L:
        return pd.Series(out, index=y.index)
    w = sliding_window_view(arr, L)
    t = np.arange(L, dtype=float)
    St = t.sum()
    Stt = (t * t).sum()
    D = L * Stt - St * St
    valid = np.sum(~np.isnan(w), axis=1) >= max(3, L // 2)
    sy = np.nansum(w, axis=1)
    sty = np.nansum(w * t, axis=1)
    slope = (L * sty - St * sy) / D
    out[L - 1:] = np.where(valid, slope, np.nan)
    return pd.Series(out, index=y.index)


def derive_morph(corr20, corr60, r2_20, r2_60, prefix: str) -> Dict[str, pd.Series]:
    out = {}
    out[f'{prefix}_slope5_corr'] = rolling_slope(corr20, 5)
    out[f'{prefix}_slope10_corr'] = rolling_slope(corr20, 10)
    out[f'{prefix}_slope5_r2'] = rolling_slope(r2_20, 5)
    out[f'{prefix}_stab20_corr'] = corr20.rolling(20, min_periods=10).std()
    out[f'{prefix}_change_corr'] = corr20 - corr60
    out[f'{prefix}_change_r2'] = r2_20 - r2_60
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=800)
    ap.add_argument('--start', default='2019-01-01')
    ap.add_argument('--end', default='2022-01-01')
    a = ap.parse_args()

    t0 = time.time()
    codes = pick_pool(a.pool, a.start, a.end)
    idx = ms.load_index_returns(DB_DAILY)
    glob = ms.load_market_series(
        os.path.join(ROOT, 'database', 'stock_meta.db'))
    idx_vol = idx[ms.MARKET_INDEX].rolling(20, min_periods=13).std() if idx else None

    rows = []
    for code in codes:
        r_i = load_stock(code, a.start, a.end)
        if r_i is None or len(r_i) < 200:
            continue
        f = os.path.join(CACHE, f'{code}_factors.parquet')
        if not os.path.exists(f):
            continue
        base = pd.read_parquet(f)
        if 'date' in base.columns:
            base['date'] = base['date'].astype(str)
            base = base.drop_duplicates('date').set_index('date').sort_index()
        base = base.reindex(r_i.index)
        common = base.index.intersection(r_i.index)
        if len(common) < 200:
            continue
        base = base.loc[common]

        morph_frames = []
        for entry in ms.SENS_REGISTRY:
            prefix = entry['prefix']
            if entry['kind'] == 'index_ret':
                if idx is None:
                    continue
                x = ms.board_series(code, idx)
            elif entry['kind'] == 'index_vol':
                if idx_vol is None:
                    continue
                x = idx_vol
            else:
                if glob is None or entry['src'] not in glob:
                    continue
                x = glob[entry['src']]
            if entry['diff']:
                x = x.diff()
            sub = ms.compute_sensitivity(r_i, x)
            if sub.empty or 'corr_20' not in sub.columns:
                continue
            m = derive_morph(sub['corr_20'], sub['corr_60'],
                             sub['r2_20'], sub['r2_60'], prefix)
            mf = pd.DataFrame(m, index=sub.index)
            mf = mf.reindex(common)
            morph_frames.append(mf)
        if not morph_frames:
            continue
        morph = pd.concat(morph_frames, axis=1)
        merged = pd.concat([morph, base], axis=1)
        merged['code'] = code
        rows.append(merged.reset_index())

    big = pd.concat(rows, ignore_index=True)
    cand_cols = list(morph.columns)
    base_cols = [c for c in base.columns if c in big.columns]
    print(f'[load] {len(big)} 行；形态候选 {len(cand_cols)} 个，基线 {len(base_cols)} 列，'
          f'用时 {time.time()-t0:.1f}s')

    use = base_cols + cand_cols
    mat = big[use].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    finite = np.isfinite(mat)
    col_means = np.array([mat[finite[:, j], j].mean() if finite[:, j].any() else 0.0
                          for j in range(mat.shape[1])])
    mat = np.where(finite, mat, col_means)
    corr = np.corrcoef(mat, rowvar=False)
    names = use

    print(f'\n=== beta 形态派生 vs 基线247列：max|corr| 与最相关基线特征 ===')
    print(f'{"candidate":36s} {"max|corr|":>9s} {"vs_feature":22s} {"dir":>5s}')
    recs = []
    for c in cand_cols:
        ci = names.index(c)
        best, bj = 0.0, -1
        for j, nj in enumerate(names):
            if nj in base_cols:
                v = corr[ci, j]
                if abs(v) > abs(best):
                    best, bj = v, j
        recs.append((c, best, names[bj] if bj >= 0 else ''))

    near = [r for r in recs if abs(r[1]) > 0.7]
    mid = [r for r in recs if 0.4 < abs(r[1]) <= 0.7]
    low = [r for r in recs if abs(r[1]) <= 0.4]
    print(f'\n近冗余(|corr|>0.7): {len(near)} | 中等(0.4-0.7): {len(mid)} | 低相关(<0.4, 可叠加): {len(low)}')
    print(f'\n--- 低相关候选（可能的新信号，按 max|corr| 升序）---')
    for c, best, bj in sorted(low, key=lambda r: abs(r[1])):
        print(f'  {c:36s} max|corr|={best:+.3f}(vs {bj})')
    if mid:
        print(f'\n--- 中等相关（需警惕，可能部分冗余）---')
        for c, best, bj in sorted(mid, key=lambda r: abs(r[1]), reverse=True):
            print(f'  {c:36s} max|corr|={best:+.3f}(vs {bj})')
    if near:
        print(f'\n--- 近冗余（不物化）---')
        for c, best, bj in sorted(near, key=lambda r: abs(r[1]), reverse=True):
            print(f'  {c:36s} max|corr|={best:+.3f}(vs {bj})')


if __name__ == '__main__':
    main()
