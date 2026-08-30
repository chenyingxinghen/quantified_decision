#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量价背离/估值动量候选 冗余校验（vs 基线 247 列）
=================================================

diag_volume_value_ic 显示 8/8 显著，但估值动量（val_mom_*）本质是
「价格变化/盈利变化」——盈利季度更新，peTTM 20 日变化 ≈ 价格 20 日动量，
大概率与基线 return_20d/momentum_20d 伪冗余。本脚本验证。

用法
  python scripts/exp/diag_volume_value_redun.py [--pool 800] [--start 2019-01-01] [--end 2022-01-01]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

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


def load_stock_full(code, start, end):
    conn = sqlite3.connect(DB_DAILY, timeout=60.0)
    try:
        d = pd.read_sql_query(
            'SELECT date, close, pctChg, volume, turnover_rate, peTTM, pbMRQ, psTTM '
            'FROM daily_data WHERE code=? AND date>=? AND date<? ORDER BY date ASC',
            conn, params=(code, start, end))
    finally:
        conn.close()
    if d.empty:
        return None
    d['date'] = d['date'].astype(str)
    d = d.drop_duplicates('date').set_index('date').sort_index()
    return d


def derive(d: pd.DataFrame) -> pd.DataFrame:
    cl = d['close'].astype(float)
    vol = d['volume'].astype(float)
    pc = d['pctChg'].astype(float)
    to = d['turnover_rate'].astype(float)
    pe = pd.to_numeric(d['peTTM'], errors='coerce')
    pb = pd.to_numeric(d['pbMRQ'], errors='coerce')
    ps = pd.to_numeric(d['psTTM'], errors='coerce')
    out = pd.DataFrame(index=d.index)
    with np.errstate(invalid='ignore', divide='ignore'):
        out['val_mom_pe20'] = pe.pct_change(20)
        out['val_mom_pb20'] = pb.pct_change(20)
        out['val_mom_ps20'] = ps.pct_change(20)
        out['val_accel_pe5'] = pe.pct_change(5).diff()
        out['val_pe_turn_corr20'] = pe.rolling(20, min_periods=10).corr(to)
        r20 = cl.pct_change(20)
        v20 = vol.pct_change(20)
        out['pv_div20'] = r20.rank(pct=True) - v20.rank(pct=True)
        out['vol_skew20'] = vol.rolling(20, min_periods=10).skew()
        out['vol_corr_ret20'] = vol.rolling(20, min_periods=10).corr(pc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=800)
    ap.add_argument('--start', default='2019-01-01')
    ap.add_argument('--end', default='2022-01-01')
    a = ap.parse_args()

    t0 = time.time()
    codes = pick_pool(a.pool, a.start, a.end)
    print(f'[pool] {len(codes)} 只股票，窗口 {a.start}→{a.end}，'
          f'用时 {time.time()-t0:.1f}s')

    rows = []
    cand_cols_all = []
    for i, code in enumerate(codes):
        d = load_stock_full(code, a.start, a.end)
        if d is None or len(d) < 200:
            continue
        cand = derive(d)
        cand_cols_all = list(cand.columns)
        fp = os.path.join(CACHE, f'{code}_factors.parquet')
        if not os.path.exists(fp):
            continue
        base = pd.read_parquet(fp)
        if 'date' in base.columns:
            base['date'] = base['date'].astype(str)
            base = base.drop_duplicates('date').set_index('date').sort_index()
        common = base.index.intersection(d.index)
        if len(common) < 200:
            continue
        merged = pd.concat([cand.loc[common], base.loc[common]], axis=1)
        merged['code'] = code
        rows.append(merged.reset_index())
        if (i + 1) % 200 == 0:
            print(f'  [{i+1}/{len(codes)}] 用时 {time.time()-t0:.1f}s')

    big = pd.concat(rows, ignore_index=True)
    cand_cols = [c for c in cand_cols_all if c in big.columns]
    base_cols = [c for c in big.columns if c not in cand_cols
                 and c not in ('date', 'code')]
    print(f'[load] {len(big)} 行；候选 {len(cand_cols)}，基线 {len(base_cols)} 列，'
          f'用时 {time.time()-t0:.1f}s')

    use = base_cols + cand_cols
    mat = big[use].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    finite = np.isfinite(mat)
    col_means = np.array([mat[finite[:, j], j].mean() if finite[:, j].any() else 0.0
                          for j in range(mat.shape[1])])
    mat = np.where(finite, mat, col_means)
    corr = np.corrcoef(mat, rowvar=False)
    names = use

    print(f'\n=== 量价/估值候选 vs 基线：max|corr| 与最相关基线特征 ===')
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
    print(f'\n近冗余(|corr|>0.7): {len(near)} | 中等(0.4-0.7): {len(mid)} | 低相关(<0.4): {len(low)}')
    for tag, group in (('低相关', low), ('中等', mid), ('近冗余', near)):
        if group:
            print(f'\n--- {tag} ---')
            for c, best, bj in sorted(group, key=lambda r: abs(r[1])):
                print(f'  {c:22s} max|corr|={best:+.3f}(vs {bj})')


if __name__ == '__main__':
    main()
