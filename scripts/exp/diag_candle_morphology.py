#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
K线 时序形态派生特征 IC 诊断（主动构造，非滞后平移）
====================================================
目的：diag_candle_temporal 仅对原特征做滞后平移（lag0..lag10），发现是单调衰减
（level+decay），无时序顺序/形态结构。本脚本**主动构造**真正刻画"形态如何演化"
的派生特征，检验是否存在超越标量+计数的新信号：

  D1 导数(速度):   dM_1 = M[t]-M[t-1],  dM_5 = M[t]-M[t-5]   —— 烛形量在升还是在降（方向敏感）
  D2 斜率(趋势):   slope5/10 = 滚动OLS斜率(M over 5/10d)     —— 形态的线性演化方向
  D3 加速度(曲率): acc2 = M[t]-2M[t-1]+M[t-2]                —— 形态演化的二阶变化
  D4 有序转移对:   tr_XY = (P_X@t) AND (P_Y@t-1)             —— 顺序依赖：今日X且昨日Y
  L  水平参考:     M[t] 本身（标量，已在 temporal 测过，此处重算作对照）

M = 8 个连续烛形量：candle_body_ratio / upper_shadow_ratio / lower_shadow_ratio /
    pattern_strength / m_intraday_ret / m_gap / m_close_pos / m_range_rel
P(转移) = doji/hammer/hanging_man/marubozu/bullish_engulfing/bearish_engulfing (6)

方法：800股×2019-2022（OHLC 2015起预热）；逐日横截面 rank-IC vs 7d/15d 前瞻收益。
重点看 D1/D2/D3/D4 是否出现 |ic7|>0.01 且 |t|>2.8 的稳定信号（超越 level 对照）。

用法
  python scripts/exp/diag_candle_morphology.py [--pool 800] [--end 2022-01-01]
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

from core.factors.candlestick_pattern_factors import CandlestickPatternFactors

DB_DAILY = os.path.join(ROOT, 'database', 'stock_daily.db')
METRICS = ['candle_body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
           'pattern_strength', 'm_intraday_ret', 'm_gap', 'm_close_pos', 'm_range_rel']
MORPH_PATTERNS = ['doji', 'hammer', 'hanging_man', 'marubozu',
                  'bullish_engulfing', 'bearish_engulfing']


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


def compute_base(d):
    o, h, l, c = d['open'], d['high'], d['low'], d['close']
    allf = CandlestickPatternFactors().calculate_all_candlestick_patterns(d[['open', 'high', 'low', 'close']])
    allf.index = d.index
    oc = c / o - 1.0
    prev_c = c.shift(1)
    rng = (h - l).replace(0, np.nan)
    metrics = pd.DataFrame({
        'candle_body_ratio': allf['candle_body_ratio'],
        'upper_shadow_ratio': allf['upper_shadow_ratio'],
        'lower_shadow_ratio': allf['lower_shadow_ratio'],
        'pattern_strength': allf['pattern_strength'],
        'm_intraday_ret': oc,
        'm_gap': o / prev_c - 1.0,
        'm_close_pos': (c - l) / rng,
        'm_range_rel': (h - l) / prev_c,
    }, index=d.index)
    skip = ('candle_body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
            'pattern_strength', 'pattern_confirmation')
    flags = allf[[col for col in allf.columns if col not in skip]]
    return metrics, flags


def ols_slope(y):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom <= 0:
        return np.nan
    return float(((x - xm) * (y - ym)).sum() / denom)


def derive(metrics, flags):
    out = pd.DataFrame(index=metrics.index)
    # D1 导数
    for m in METRICS:
        out[f'd1_{m}'] = metrics[m] - metrics[m].shift(1)
        out[f'd5_{m}'] = metrics[m] - metrics[m].shift(5)
    # D2 斜率
    for w in (5, 10):
        for m in METRICS:
            out[f'slope{w}_{m}'] = metrics[m].rolling(w, min_periods=max(3, w // 2)).apply(ols_slope, raw=True)
    # D3 加速度
    for m in METRICS:
        out[f'acc2_{m}'] = metrics[m] - 2 * metrics[m].shift(1) + metrics[m].shift(2)
    # D4 有序转移对
    for X in MORPH_PATTERNS:
        for Y in MORPH_PATTERNS:
            out[f'tr_{X}__{Y}'] = flags[X].astype(float) * flags[Y].shift(1).astype(float)
    # L 水平参考
    for m in METRICS:
        out[f'L_{m}'] = metrics[m]
    return out


def fwd_returns(close, horizons=(7, 15)):
    c = close.astype(float).to_numpy()
    out = {}
    for h in horizons:
        fc = np.concatenate([c[h:], np.full(min(h, len(c)), np.nan)])[:len(c)]
        with np.errstate(invalid='ignore', divide='ignore'):
            out[h] = pd.Series(fc / np.where(c == 0, np.nan, c) - 1.0, index=close.index)
    return out


def rank_ic_panel(panel, yret):
    fr = panel.rank(axis=1)
    yr = yret.rank(axis=1)
    ics = []
    for d in panel.index:
        fv = fr.loc[d].to_numpy(dtype=float)
        yv = yr.loc[d].to_numpy(dtype=float)
        m = np.isfinite(fv) & np.isfinite(yv)
        if m.sum() < 50:
            continue
        fv, yv = fv[m], yv[m]
        fv = fv - fv.mean()
        yv = yv - yv.mean()
        denom = np.sqrt((fv * fv).sum() * (yv * yv).sum())
        if denom <= 0:
            continue
        ics.append(float((fv * yv).sum() / denom))
    ics = np.array(ics, dtype=float)
    if len(ics) < 10:
        return (np.nan, np.nan, np.nan)
    mean = ics.mean()
    sd = ics.std(ddof=1)
    n = len(ics)
    t = mean / (sd / np.sqrt(n)) if sd > 0 else np.nan
    icir = mean / sd if sd > 0 else np.nan
    return (mean, t, icir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=800)
    ap.add_argument('--end', default='2022-01-01')
    ap.add_argument('--load-start', default='2015-01-01')
    a = ap.parse_args()
    ic_start = '2019-01-01'

    print(f"[{time.strftime('%H:%M:%S')}] loading codes ...")
    codes = pick_pool(a.pool, ic_start, a.end)
    print(f"  {len(codes)} codes")

    # 前向收益面板
    print(f"[{time.strftime('%H:%M:%S')}] building forward-return panels ...")
    ypan = {h: {} for h in (7, 15)}
    for code in codes:
        d = load_ohlc(code, ic_start, a.end)
        if d is None:
            continue
        fr = fwd_returns(d['close'])
        for h in (7, 15):
            ypan[h][code] = fr[h]
    Y = {h: pd.DataFrame(ypan[h]) for h in (7, 15)}
    ic_dates = Y[7].dropna(how='all').index
    for h in (7, 15):
        Y[h] = Y[h].loc[ic_dates]
    print(f"  ic_dates = {len(ic_dates)}")

    # 派生特征
    print(f"[{time.strftime('%H:%M:%S')}] deriving morphology features ...")
    feat_series = {}  # fname -> {code: Series}
    for ci, code in enumerate(codes):
        d = load_ohlc(code, a.load_start, a.end)
        if d is None or len(d) < 200:
            continue
        metrics, flags = compute_base(d)
        der = derive(metrics, flags)
        for col in der.columns:
            feat_series.setdefault(col, {})[code] = der[col]
        if (ci + 1) % 200 == 0:
            print(f"  {ci+1}/{len(codes)} codes done")

    # IC
    print(f"[{time.strftime('%H:%M:%S')}] computing IC ...")
    results = []
    for fname, series in feat_series.items():
        panel = pd.DataFrame(series).loc[ic_dates]
        for h in (7, 15):
            mean, t, icir = rank_ic_panel(panel, Y[h])
            results.append((fname, h, mean, t, icir))
    res = pd.DataFrame(results, columns=['f', 'h', 'ic', 't', 'icir'])

    # 输出：分组，按 |ic7| 降序
    print(f"\n{'='*118}")
    print(f"K线 时序形态派生特征 IC（{len(codes)}股，2019-2022，7d/15d 前瞻收益）")
    print(f"{'='*118}")
    d7 = res[res['h'] == 7].copy()
    groups = {'D1_deriv': 'd[15]_', 'D2_slope': 'slope', 'D3_accel': 'acc2',
              'D4_trans': 'tr_', 'L_level': 'L_'}
    for gname, prefix in groups.items():
        sub = d7[d7['f'].str.startswith(prefix)].copy()
        sub = sub.reindex(sub['ic'].abs().sort_values(ascending=False).index)
        print(f"\n--- {gname} ({len(sub)} 特征) ---")
        print(f"{'feature':40s} {'ic7':>8} {'t7':>8} {'ic15':>8} {'t15':>8}")
        for _, r in sub.iterrows():
            r15 = d7[d7['f'] == r['f']]
            ic15 = d7[(d7['f'] == r['f']) & (d7['h'] == 15)]['ic']
            t15 = d7[(d7['f'] == r['f']) & (d7['h'] == 15)]['t']
            mark = ' *' if abs(r['t']) > 2.8 and abs(r['ic']) > 0.01 else ''
            print(f"{r['f']:40s} {r['ic']:>+8.4f} {r['t']:>+8.2f} "
                  f"{ic15.values[0] if len(ic15) else float('nan'):>+8.4f} "
                  f"{t15.values[0] if len(t15) else float('nan'):>+8.2f}{mark}")

    # 汇总：各组的显著数
    print(f"\n{'='*118}")
    for gname, prefix in groups.items():
        sub = d7[d7['f'].str.startswith(prefix)]
        n_sig = ((sub['t'].abs() > 2.8) & (sub['ic'].abs() > 0.01)).sum()
        print(f"{gname:12s}: 显著特征 {n_sig}/{len(sub)}")
    # 全局最强
    top = d7.reindex(d7['ic'].abs().sort_values(ascending=False).index).head(10)
    print(f"\n全局 |ic7| Top10:")
    for _, r in top.iterrows():
        print(f"  {r['f']:40s} ic7={r['ic']:+.4f} t7={r['t']:+.2f}")
    print(f"[{time.strftime('%H:%M:%S')}] done")


if __name__ == '__main__':
    main()
