#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
K线形态因子 raw IC 诊断（小池 800×3y，不污染共享缓存）
====================================================

目的
----
验证 ``core/factors/candlestick_pattern_factors.py`` 产出的 K 线形态因子是否携带
可外推的截面预测信号。这些因子**当前未进入 NAM 训练缓存（factors_cache 247 列
无 candle/pattern 列）**，所以是否值得物化进生产是个待决问题。

方法
----
1. 取小池 800 只股票，OHLC 历史从 2015-01-01 起载入（给滚动上下文做热身），
   仅 2019-01-01→2022-01-01 窗口做 IC 聚合。
2. 复用 ``CandlestickPatterns.calculate_all_candlestick_patterns`` 产 23 列：
   - 19 个**离散**形态（doji/hammer/engulfing/star/harami…）→ 转成 {5,10,20}
     窗口内的**出现计数**（quasi-continuous），避免纯二值 rank 退化，并捕捉
     「形态簇」信号（单根噪声大、簇集才有信息）。
   - 5 个连续量（candle_body_ratio / upper_shadow_ratio / lower_shadow_ratio /
     pattern_strength / pattern_confirmation）。
3. 补充 4 个模块未覆盖但属 K 线形态的连续量：日内收益 intraday_ret、跳空 gap、
   收盘在区间位置 close_pos、真实波幅占比 range_rel，各含 {5,10,20} 滚动均值。
4. 前瞻收益标签：7 日 / 15 日（close.shift(-h)/close-1，对齐生产口径）。
5. 逐日横截面 rank-IC 聚合 mean / t / ICIR。

输出
----
按 |mean IC(7d)| 降序的特征表，并分组标注：模块连续量 / 模块形态计数 /
补充 OHLC 形态量；最后汇总显著特征数。

用法
  python scripts/exp/diag_candle_ic.py [--pool 800] [--ic-start 2019-01-01] \
      [--end 2022-01-01] [--load-start 2015-01-01] [--topn 50]
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

# 模块产出中的连续量（非离散形态）
CONTINUOUS = {'candle_body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
              'pattern_strength', 'pattern_confirmation'}
COUNT_WINDOWS = [5, 10, 20]
ROLL_WINDOWS = [5, 10, 20]


def pick_pool(pool: int, start: str, end: str):
    conn = sqlite3.connect(DB_DAILY, timeout=60.0)
    try:
        rows = conn.execute(
            """SELECT code, COUNT(*) n FROM daily_data
               WHERE date>=? AND date<? GROUP BY code HAVING n>=400
               ORDER BY n DESC LIMIT ?""",
            (start, end, pool)).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def load_ohlc(code: str, load_start: str, end: str):
    conn = sqlite3.connect(DB_DAILY, timeout=60.0)
    try:
        d = pd.read_sql_query(
            'SELECT date, open, high, low, close, pctChg FROM daily_data '
            'WHERE code=? AND date>=? AND date<? ORDER BY date ASC',
            conn, params=(code, load_start, end))
    finally:
        conn.close()
    if d.empty:
        return None
    d['date'] = d['date'].astype(str)
    d = d.drop_duplicates('date').set_index('date').sort_index()
    for c in ['open', 'high', 'low', 'close', 'pctChg']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d


def fwd_returns(close: pd.Series, horizons=(7, 15)):
    c = close.astype(float).to_numpy()
    out = {}
    for h in horizons:
        fc = np.concatenate([c[h:], np.full(min(h, len(c)), np.nan)])[:len(c)]
        with np.errstate(invalid='ignore', divide='ignore'):
            out[h] = pd.Series(fc / np.where(c == 0, np.nan, c) - 1.0,
                               index=close.index)
    return out


def per_day_rank_ic(fmat: pd.DataFrame, ymat: pd.DataFrame):
    fr = fmat.rank(axis=1)
    yr = ymat.rank(axis=1)
    ic_days = []
    for d in fr.index:
        fv = fr.loc[d].to_numpy(dtype=float)
        yv = yr.loc[d].to_numpy(dtype=float)
        m = np.isfinite(fv) & np.isfinite(yv)
        if m.sum() < 30:
            continue
        fv, yv = fv[m], yv[m]
        fv = fv - fv.mean()
        yv = yv - yv.mean()
        denom = np.sqrt((fv * fv).sum() * (yv * yv).sum())
        if denom <= 0:
            continue
        ic_days.append(float((fv * yv).sum() / denom))
    return np.array(ic_days, dtype=float)


def compute_features(d: pd.DataFrame):
    """返回 (date-indexed DataFrame of all candidate features, group-label dict)."""
    data = pd.DataFrame({
        'open': d['open'], 'high': d['high'],
        'low': d['low'], 'close': d['close'],
    })
    cp = CandlestickPatternFactors()
    allf = cp.calculate_all_candlestick_patterns(data)  # 23 列

    feats = pd.DataFrame(index=data.index)
    group = {}

    # (A) 模块连续量
    for col in allf.columns:
        if col in CONTINUOUS:
            feats[col] = allf[col]
            group[col] = 'A_module_continuous'

    # (B) 模块离散形态 → 窗口计数
    binary = [c for c in allf.columns if c not in CONTINUOUS]
    for col in binary:
        for w in COUNT_WINDOWS:
            name = f'{col}_cnt{w}'
            feats[name] = allf[col].rolling(w, min_periods=3).sum()
            group[name] = 'B_module_pattern_count'

    # (C) 补充 OHLC 形态连续量（模块未覆盖）
    o, h, l, c = d['open'], d['high'], d['low'], d['close']
    extra = {}
    extra['intraday_ret'] = c / o - 1.0
    prev_c = c.shift(1)
    extra['gap'] = o / prev_c - 1.0
    rng = (h - l).replace(0, np.nan)
    extra['close_pos'] = (c - l) / rng
    extra['range_rel'] = (h - l) / prev_c
    for base, ser in extra.items():
        feats[base] = ser
        group[base] = 'C_supplementary_ohlc'
        for w in ROLL_WINDOWS:
            name = f'{base}_ma{w}'
            feats[name] = ser.rolling(w, min_periods=3).mean()
            group[name] = 'C_supplementary_ohlc'

    return feats.replace([np.inf, -np.inf], np.nan), group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=800)
    ap.add_argument('--ic-start', default='2019-01-01')
    ap.add_argument('--end', default='2022-01-01')
    ap.add_argument('--load-start', default='2015-01-01')
    ap.add_argument('--topn', type=int, default=50)
    a = ap.parse_args()

    t0 = time.time()
    codes = pick_pool(a.pool, a.ic_start, a.end)
    print(f'[pool] {len(codes)} 只股票，IC 窗口 {a.ic_start}→{a.end}，'
          f'载入自 {a.load_start}，用时 {time.time()-t0:.1f}s')

    # 逐股：载入 OHLC → 计算特征 → 前瞻收益
    stock_feats = {}   # code -> (dates, feats_df)
    y7 = {}
    y15 = {}
    for code in codes:
        d = load_ohlc(code, a.load_start, a.end)
        if d is None or len(d) < 200:
            continue
        feats, _ = compute_features(d)
        fwd = fwd_returns(d['close'].astype(float))
        stock_feats[code] = (feats.index, feats)
        y7[code] = (feats.index, fwd[7].reindex(feats.index))
        y15[code] = (feats.index, fwd[15].reindex(feats.index))
    print(f'[load] 有效股票 {len(stock_feats)}，用时 {time.time()-t0:.1f}s')

    # 统一日期网格（仅 IC 窗口内）
    all_dates = sorted(set().union(
        *[set(idx) for idx, _ in stock_feats.values()]))
    all_dates = [dt for dt in all_dates if a.ic_start <= dt < a.end]
    date_idx = pd.Index(all_dates)
    print(f'[grid] IC 窗口交易日 {len(date_idx)}')

    # 收集特征名与分组（从首个有效股一次性取得列名与 group 映射）
    feat_names = list(next(iter(stock_feats.values()))[1].columns)
    _, group = compute_features(load_ohlc(codes[0], a.load_start, a.end))

    panels = {f: pd.DataFrame(index=date_idx, columns=codes, dtype=float)
              for f in feat_names}
    Y7 = pd.DataFrame(index=date_idx, columns=codes, dtype=float)
    Y15 = pd.DataFrame(index=date_idx, columns=codes, dtype=float)
    for code in codes:
        if code not in stock_feats:
            continue
        idx, feats = stock_feats[code]
        _, f7 = y7[code]
        _, f15 = y15[code]
        m = (idx >= a.ic_start) & (idx < a.end)
        idx_w = idx[m]
        for f in feat_names:
            panels[f].loc[idx_w, code] = feats[f].reindex(idx_w).values
        Y7.loc[idx_w, code] = f7.reindex(idx_w).values
        Y15.loc[idx_w, code] = f15.reindex(idx_w).values
    print(f'[compute] 候选特征 {len(panels)}，用时 {time.time()-t0:.1f}s')

    # IC 聚合
    records = []
    for feat, mat in panels.items():
        ic7 = per_day_rank_ic(mat, Y7)
        ic15 = per_day_rank_ic(mat, Y15)
        if len(ic7) == 0:
            continue
        m7, s7 = ic7.mean(), ic7.std()
        m15, s15 = ic15.mean(), ic15.std()
        records.append({
            'feature': feat,
            'group': group.get(feat, '?'),
            'ic7_mean': m7,
            'ic7_t': m7 / (s7 / np.sqrt(len(ic7))) if s7 > 0 else np.nan,
            'ic7_icir': m7 / s7 if s7 > 0 else np.nan,
            'ic15_mean': m15,
            'ic15_t': m15 / (s15 / np.sqrt(len(ic15))) if s15 > 0 else np.nan,
            'n_days': len(ic7),
        })
    df = pd.DataFrame(records).sort_values(
        'ic7_mean', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_rows', 300)

    print(f'\n=== K线形态因子 raw rank-IC（{a.pool}股 × '
          f'{a.ic_start[:4]}-{a.end[:4]}），按 |IC7| 降序 ===')
    cols = ['feature', 'group', 'ic7_mean', 'ic7_t', 'ic7_icir',
            'ic15_mean', 'ic15_t', 'n_days']
    with pd.option_context('display.float_format', lambda v: f'{v:+.4f}'):
        print(df[cols].head(a.topn).to_string(index=False))

    sig = df[(df['ic7_t'].abs() > 2) & (df['ic7_mean'].abs() > 0.01)]
    print(f'\n显著特征 (|ic7_t|>2 且 |ic7_mean|>0.01): {len(sig)} / {len(df)}')
    print(f'  正向(预测涨): {(sig["ic7_mean"]>0).sum()}，'
          f'负向: {(sig["ic7_mean"]<0).sum()}')
    for g in ['A_module_continuous', 'B_module_pattern_count',
              'C_supplementary_ohlc']:
        gs = sig[sig['group'] == g]
        print(f'  组 {g}: 显著 {len(gs)} 个')
        for _, r in gs.iterrows():
            print(f"    {r['feature']:36s} ic7={r['ic7_mean']:+.4f} "
                  f"t={r['ic7_t']:+.2f} icir={r['ic7_icir']:+.3f}")

    print(f'\n[done] 总用时 {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
