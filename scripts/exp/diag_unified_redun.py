#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
行业变换 + 时序曲率 单因子候选 统一冗余校验（vs 基线 247 列）
================================================================

背景
----
- diag_industry_ic：行业变换（indrank/indneu/indz）IC 与全局 rank 相当（t 略高），
  疑似「行业级信息在全局 rank 下被吸收」→ 需确认是否与基线 ret/vol 等冗余。
- diag_temporal_morph_ic：spread/slope 显著（9/15）但 accel2 零信号，疑似与基线
  amount_change_rate/acceleration/roc 冗余。

本脚本把两类候选统一与 factors_cache 全部基线列做池化相关矩阵，报每个候选
vs 基线的 max|corr| 及最相关列。判定近冗余(>0.7)/中等(0.4-0.7)/低相关(<0.4 可叠加)。

候选：
  时序曲率: spread_ret5_20 / spread_vol5_20 / spread_hl5_20 / spread_amt5_20 /
            slope5_amt20 / slope5_hl20 / slope5_ret20 / slope5_vol20
  行业变换: indneu_{ret20,vol20,hl20,amt20} / indrank_ret20（行业内 rank）

用法
  python scripts/exp/diag_unified_redun.py [--pool 800] [--start 2019-01-01] [--end 2022-01-01]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

DB_DAILY = os.path.join(ROOT, 'database', 'stock_daily.db')
DB_META = os.path.join(ROOT, 'database', 'stock_meta.db')
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


def load_industry() -> Dict[str, str]:
    conn = sqlite3.connect(DB_META, timeout=60.0)
    try:
        df = pd.read_sql_query("SELECT code, industry FROM stock_industry", conn)
    finally:
        conn.close()
    return dict(zip(df['code'], df['industry']))


def load_stock_full(code, start, end):
    conn = sqlite3.connect(DB_DAILY, timeout=60.0)
    try:
        d = pd.read_sql_query(
            'SELECT date, open, high, low, close, pctChg, amount FROM daily_data '
            'WHERE code=? AND date>=? AND date<? ORDER BY date ASC',
            conn, params=(code, start, end))
    finally:
        conn.close()
    if d.empty:
        return None
    d['date'] = d['date'].astype(str)
    d = d.drop_duplicates('date').set_index('date').sort_index()
    return d


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=800)
    ap.add_argument('--start', default='2019-01-01')
    ap.add_argument('--end', default='2022-01-01')
    a = ap.parse_args()

    t0 = time.time()
    codes = pick_pool(a.pool, a.start, a.end)
    ind_all = load_industry()
    print(f'[pool] {len(codes)} 只股票，窗口 {a.start}→{a.end}，'
          f'用时 {time.time()-t0:.1f}s')

    # 第一遍：加载全部股票基础水平
    levels = {}
    for code in codes:
        d = load_stock_full(code, a.start, a.end)
        if d is None or len(d) < 200:
            continue
        cl = d['close'].astype(float)
        pc = d['pctChg'].astype(float)
        amt = d['amount'].astype(float)
        hl = (d['high'].astype(float) - d['low'].astype(float))
        with np.errstate(invalid='ignore', divide='ignore'):
            lev = pd.DataFrame(index=d.index)
            lev['ret_5d'] = cl.pct_change(5)
            lev['ret_20d'] = cl.pct_change(20)
            lev['vol_5d'] = pc.rolling(5, min_periods=3).std()
            lev['vol_20d'] = pc.rolling(20, min_periods=10).std()
            lev['hl_ma5'] = (hl / cl.replace(0, np.nan)).rolling(5, min_periods=3).mean()
            lev['hl_ma20'] = (hl / cl.replace(0, np.nan)).rolling(20, min_periods=10).mean()
            lev['amount_ma5'] = amt.rolling(5, min_periods=3).mean()
            lev['amount_ma20'] = amt.rolling(20, min_periods=10).mean()
        levels[code] = lev
    print(f'[load] 有效股票 {len(levels)}，用时 {time.time()-t0:.1f}s')

    # 行业均值（面板法：对每个行业逐日均值）
    ind_of = {c: ind_all.get(c, np.nan) for c in levels}
    ind_members: Dict[str, List[str]] = {}
    for c, v in ind_of.items():
        if not pd.isna(v):
            ind_members.setdefault(v, []).append(c)

    def panel_mean(field: str):
        """返回 dict: iname -> Series(date) 该行业 field 逐日均值。"""
        out = {}
        for iname, members in ind_members.items():
            if len(members) < 5:
                continue
            s = pd.DataFrame({c: levels[c][field] for c in members
                              if c in levels})
            if s.shape[1] >= 5:
                out[iname] = s.mean(axis=1)
        return out

    mean_ret20 = panel_mean('ret_20d')
    mean_vol20 = panel_mean('vol_20d')
    mean_hl20 = panel_mean('hl_ma20')
    mean_amt20 = panel_mean('amount_ma20')
    print(f'[industry] 行业均值序列就绪，用时 {time.time()-t0:.1f}s')

    # 第二遍：逐股组装候选 + 读基线，合并长表
    rows = []
    cand_cols_all: List[str] = []
    for i, (code, lev) in enumerate(levels.items()):
        cand = pd.DataFrame(index=lev.index)
        cand['spread_ret5_20'] = lev['ret_5d'] - lev['ret_20d']
        cand['spread_vol5_20'] = lev['vol_5d'] - lev['vol_20d']
        cand['spread_hl5_20'] = lev['hl_ma5'] - lev['hl_ma20']
        cand['spread_amt5_20'] = lev['amount_ma5'] - lev['amount_ma20']
        cand['slope5_amt20'] = rolling_slope(lev['amount_ma20'], 5)
        cand['slope5_hl20'] = rolling_slope(lev['hl_ma20'], 5)
        cand['slope5_ret20'] = rolling_slope(lev['ret_20d'], 5)
        cand['slope5_vol20'] = rolling_slope(lev['vol_20d'], 5)
        # 行业中性化（个股 − 行业均值）
        iname = ind_of.get(code, np.nan)
        if not pd.isna(iname):
            if iname in mean_ret20:
                cand['indneu_ret20'] = lev['ret_20d'] - mean_ret20[iname].reindex(lev.index)
            if iname in mean_vol20:
                cand['indneu_vol20'] = lev['vol_20d'] - mean_vol20[iname].reindex(lev.index)
            if iname in mean_hl20:
                cand['indneu_hl20'] = lev['hl_ma20'] - mean_hl20[iname].reindex(lev.index)
            if iname in mean_amt20:
                cand['indneu_amt20'] = lev['amount_ma20'] - mean_amt20[iname].reindex(lev.index)
        cand_cols_all = list(cand.columns)

        fp = os.path.join(CACHE, f'{code}_factors.parquet')
        if not os.path.exists(fp):
            continue
        base = pd.read_parquet(fp)
        if 'date' in base.columns:
            base['date'] = base['date'].astype(str)
            base = base.drop_duplicates('date').set_index('date').sort_index()
        common = base.index.intersection(lev.index)
        if len(common) < 200:
            continue
        merged = pd.concat([cand.loc[common], base.loc[common]], axis=1)
        merged['code'] = code
        rows.append(merged.reset_index())
        if (i + 1) % 200 == 0:
            print(f'  [{i+1}/{len(levels)}] 用时 {time.time()-t0:.1f}s')

    big = pd.concat(rows, ignore_index=True)
    cand_cols = [c for c in cand_cols_all if c in big.columns]
    base_cols = [c for c in big.columns
                 if c not in cand_cols and c not in ('date', 'code')]
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

    print(f'\n=== 行业/时序曲率候选 vs 基线：max|corr| 与最相关基线特征 ===')
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
