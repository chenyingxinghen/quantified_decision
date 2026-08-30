#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
K线形态候选特征 冗余/共线性校验（不污染共享缓存）
================================================
目的：diag_candle_ic 显示信号集中在 OHLC 形态量（range_rel/intraday_ret/close_pos/gap）
与形态计数，且本质疑似量价波动/反转，基线 247 列可能已覆盖。本脚本用池化相关矩阵
确认每个候选 candle 特征与基线已有特征的最大 |相关系数|，判定其是否为可叠加新信号。

- 候选：OHLC 标量(intraday_ret/gap/close_pos/range_rel) + 其 {5,10,20} 滚动均值；
       形态计数(black/white/doji/hanging_man/marubozu _cnt{5,10,20})；
       模块连续量(candle_body_ratio/upper_shadow_ratio/lower_shadow_ratio/pattern_strength/pattern_confirmation)。
- 基线：factors_cache 全部 247 列（含 atr/natr/hl_range_*/oc_ratio_*/intraday_drawdown_*/return_*/momentum_* 等）。
- 方法：每股票对齐日期，堆叠成 (date×stock) 长表，整体 corrcoef，报候选 vs 基线 max|corr| 及最相关列。

用法
  python scripts/exp/diag_candle_redundancy.py [--pool 800] [--start 2019-01-01] [--end 2022-01-01] [--load-start 2015-01-01]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from core.factors.candlestick_pattern_factors import CandlestickPatternFactors

DB_DAILY = os.path.join(ROOT, 'database', 'stock_daily.db')
CACHE = os.path.join(ROOT, 'database', 'system_data', 'factors_cache')
ROLL = [5, 10, 20]
CNT_PATTERNS = ['black_candle', 'white_candle', 'doji', 'hammer', 'hanging_man',
                'marubozu', 'bullish_engulfing', 'bearish_engulfing', 'harami',
                'shooting_star', 'spinning_top', 'morning_star', 'evening_star',
                'three_white_soldiers', 'three_black_crows', 'dark_cloud_cover', 'piercing_line']
CONTINUOUS = ['candle_body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
              'pattern_strength', 'pattern_confirmation']


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


def load_ohlc(code, load_start, end):
    conn = sqlite3.connect(DB_DAILY, timeout=60.0)
    try:
        d = pd.read_sql_query(
            'SELECT date, open, high, low, close, volume FROM daily_data '
            'WHERE code=? AND date>=? AND date<? ORDER BY date ASC',
            conn, params=(code, load_start, end))
    finally:
        conn.close()
    if d.empty:
        return None
    d['date'] = d['date'].astype(str)
    d = d.drop_duplicates('date').set_index('date').sort_index()
    for c in ['open', 'high', 'low', 'close', 'volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d


def compute_candidates(d: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = d['open'], d['high'], d['low'], d['close']
    allf = CandlestickPatternFactors().calculate_all_candlestick_patterns(d[['open', 'high', 'low', 'close']])
    allf.index = d.index
    feats = pd.DataFrame(index=d.index)
    # OHLC 标量 + 滚动均值
    oc = c / o - 1.0
    prev_c = c.shift(1)
    gap = o / prev_c - 1.0
    rng = (h - l).replace(0, np.nan)
    close_pos = (c - l) / rng
    range_rel = (h - l) / prev_c
    sup = {'intraday_ret': oc, 'gap': gap, 'close_pos': close_pos, 'range_rel': range_rel}
    for name, ser in sup.items():
        feats[name] = ser
        for w in ROLL:
            feats[f'{name}_ma{w}'] = ser.rolling(w, min_periods=3).mean()
    # 形态计数
    for col in allf.columns:
        if col in CONTINUOUS:
            feats[col] = allf[col]
            continue
        if col not in CNT_PATTERNS:
            continue
        for w in ROLL:
            feats[f'{col}_cnt{w}'] = (allf[col] > 0).astype(float).rolling(w, min_periods=3).mean()
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=800)
    ap.add_argument('--start', default='2019-01-01')
    ap.add_argument('--end', default='2022-01-01')
    ap.add_argument('--load-start', default='2015-01-01')
    a = ap.parse_args()

    codes = pick_pool(a.pool, a.start, a.end)
    rows = []
    for code in codes:
        d = load_ohlc(code, a.load_start, a.end)
        if d is None or len(d) < 200:
            continue
        cand = compute_candidates(d)
        f = os.path.join(CACHE, f'{code}_factors.parquet')
        if not os.path.exists(f):
            continue
        base = pd.read_parquet(f)
        if 'date' in base.columns:
            base['date'] = base['date'].astype(str)
            base = base.drop_duplicates('date').set_index('date').sort_index()
        base = base.reindex(cand.index)
        # 对齐
        idx = cand.index.intersection(base.index)
        if len(idx) < 200:
            continue
        merged = pd.concat([cand.loc[idx], base.loc[idx]], axis=1)
        merged['code'] = code
        rows.append(merged.reset_index())
    big = pd.concat(rows, ignore_index=True)
    cand_cols = [c for c in big.columns if c not in ('index', 'date', 'code') and c not in base.columns]
    base_cols = [c for c in base.columns if c in big.columns]
    print(f'[load] {len(big)} 行；候选 {len(cand_cols)} 个，基线 {len(base_cols)} 列')

    use = base_cols + cand_cols
    mat = big[use].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    finite = np.isfinite(mat)
    col_means = np.array([mat[finite[:, j], j].mean() if finite[:, j].any() else 0.0
                          for j in range(mat.shape[1])])
    mat = np.where(finite, mat, col_means)
    corr = np.corrcoef(mat, rowvar=False)
    names = use

    print(f'\n=== 候选 candle 特征 vs 基线247列：max|corr| 与最相关基线特征 ===')
    print(f'{"candidate":34s} {"max|corr|":>9s} {"vs_feature":22s} {"dir":>5s}')
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
    for c, best, bj in sorted(recs, key=lambda r: abs(r[1]), reverse=True):
        flag = '  <-- 近冗余' if abs(best) > 0.7 else ('  ~中等' if abs(best) > 0.4 else '')
        print(f'{c:34s} {best:>+9.3f} {bj:22s} {("+" if best>0 else "-"):>5s}{flag}')

    near = [r for r in recs if abs(r[1]) > 0.7]
    mid = [r for r in recs if 0.4 < abs(r[1]) <= 0.7]
    low = [r for r in recs if abs(r[1]) <= 0.4]
    print(f'\n近冗余(|corr|>0.7): {len(near)} | 中等(0.4-0.7): {len(mid)} | 低相关(<0.4, 可叠加): {len(low)}')
    if low:
        print('低相关候选（可能的新信号）：')
        for c, best, bj in sorted(low, key=lambda r: abs(r[1]), reverse=True):
            print(f'  {c:34s} max|corr|={best:+.3f}(vs {bj})')


if __name__ == '__main__':
    main()
