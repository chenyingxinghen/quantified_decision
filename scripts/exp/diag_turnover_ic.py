#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
换手率(turnover)纯个股时序族 + 估值动量补充 raw IC 诊断（小池 800×3y）
=====================================================================

背景
----
用户问「msens 是全局×个股交互；纯个股级时序特征有构造吗」。核查发现：
- 基线已有大量纯个股时序族（动量16/波动13/量能29/技术73列），其"再加工"
  （时序曲率/跨期差）已被 diag_temporal_morph_ic 证明全冗余。
- 但 daily_data 自带 **turnover_rate（换手率）97.6% 覆盖，基线 0 列** ——
  最纯的个股级时序变量（流动性/筹码交换速度），完全未利用。
- 估值列（peTTM/pbMRQ/psTTM）也未进基线，val_mom_pb20 已验证可叠加。

本脚本构造 turnover 纯个股时序族 + 补全估值动量，统一 raw IC 诊断：
  turnover 族（全部单指标时序派生，不引用任何市场序列）:
    turn_ma5/20       : 换手率 5/20 日均值（流动性水平）
    turn_chg20        : 换手率 20 日变化率（流动性变化）
    turn_std20        : 换手率 20 日波动（换手稳定性）
    turn_slope5       : 换手率最近 5 日滚动斜率（换手加速/衰减）
    turn_vol_corr20   : 换手率与成交量的 20 日相关（换手-量协同）
    turn_price_corr20 : 换手率与收益的 20 日相关（放量方向）
 估值动量（补充 diag_volume_value_ic 已验证族，一并复核）:
    val_mom_pb20 / val_mom_pe20 / val_pe_turn_corr20 / val_accel_pe5

方法（对齐 diag_beta_family_ic）：800 股 × 2019-2022，逐日横截面 rank-IC vs 7d/15d。

用法
  python scripts/exp/diag_turnover_ic.py [--pool 800] [--start 2019-01-01] [--end 2022-01-01]
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


def pick_pool(pool, start, end):
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


def load_stock_full(code, start, end):
    conn = sqlite3.connect(DB_DAILY, timeout=60.0)
    try:
        d = pd.read_sql_query(
            'SELECT date, close, pctChg, volume, turnover_rate, peTTM, pbMRQ '
            'FROM daily_data WHERE code=? AND date>=? AND date<? ORDER BY date ASC',
            conn, params=(code, start, end))
    finally:
        conn.close()
    if d.empty:
        return None
    d['date'] = d['date'].astype(str)
    d = d.drop_duplicates('date').set_index('date').sort_index()
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


def derive(d: pd.DataFrame) -> Dict[str, pd.Series]:
    out = {}
    cl = d['close'].astype(float)
    pc = d['pctChg'].astype(float)
    vol = d['volume'].astype(float)
    to = pd.to_numeric(d['turnover_rate'], errors='coerce')
    pe = pd.to_numeric(d['peTTM'], errors='coerce')
    pb = pd.to_numeric(d['pbMRQ'], errors='coerce')
    with np.errstate(invalid='ignore', divide='ignore'):
        # turnover 纯个股时序族
        out['turn_ma5'] = to.rolling(5, min_periods=3).mean()
        out['turn_ma20'] = to.rolling(20, min_periods=10).mean()
        out['turn_chg20'] = to.pct_change(20)
        out['turn_std20'] = to.rolling(20, min_periods=10).std()
        out['turn_slope5'] = rolling_slope(to, 5)
        out['turn_vol_corr20'] = to.rolling(20, min_periods=10).corr(vol)
        out['turn_price_corr20'] = to.rolling(20, min_periods=10).corr(pc)
        # 估值动量（复核）
        out['val_mom_pb20'] = pb.pct_change(20)
        out['val_mom_pe20'] = pe.pct_change(20)
        out['val_pe_turn_corr20'] = pe.rolling(20, min_periods=10).corr(to)
        out['val_accel_pe5'] = pe.pct_change(5).diff()
    return out


def per_day_rank_ic(fmat: pd.DataFrame, ymat: pd.DataFrame) -> np.ndarray:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=800)
    ap.add_argument('--start', default='2019-01-01')
    ap.add_argument('--end', default='2022-01-01')
    ap.add_argument('--topn', type=int, default=50)
    a = ap.parse_args()

    t0 = time.time()
    codes = pick_pool(a.pool, a.start, a.end)
    print(f'[pool] {len(codes)} 只股票，窗口 {a.start}→{a.end}，'
          f'用时 {time.time()-t0:.1f}s')

    stock_data = {}
    for code in codes:
        d = load_stock_full(code, a.start, a.end)
        if d is None or len(d) < 200:
            continue
        fwd = fwd_returns(d['close'].astype(float))
        stock_data[code] = (d, fwd[7], fwd[15])
    print(f'[load] 有效股票 {len(stock_data)}，用时 {time.time()-t0:.1f}s')

    all_dates = sorted(set().union(*[set(s[0].index) for s in stock_data.values()]))
    date_idx = pd.Index(all_dates)

    probe = derive(next(iter(stock_data.values()))[0])
    feat_names = list(probe.keys())
    panels = {name: pd.DataFrame(index=date_idx, columns=codes, dtype=float)
              for name in feat_names}
    y7 = pd.DataFrame(index=date_idx, columns=codes, dtype=float)
    y15 = pd.DataFrame(index=date_idx, columns=codes, dtype=float)

    for code in codes:
        if code not in stock_data:
            continue
        d, f7, f15 = stock_data[code]
        idx = d.index
        y7.loc[idx, code] = f7.reindex(idx).values
        y15.loc[idx, code] = f15.reindex(idx).values
        der = derive(d)
        for name in feat_names:
            if name in der:
                panels[name].loc[idx, code] = der[name].reindex(idx).values
    print(f'[compute] 候选 {len(panels)}，用时 {time.time()-t0:.1f}s')

    records = []
    for feat, mat in panels.items():
        ic7 = per_day_rank_ic(mat, y7)
        ic15 = per_day_rank_ic(mat, y15)
        if len(ic7) == 0:
            continue
        m7, s7 = ic7.mean(), ic7.std()
        m15, s15 = ic15.mean(), ic15.std()
        records.append({
            'feature': feat,
            'ic7_mean': m7,
            'ic7_t': m7 / (s7 / np.sqrt(len(ic7))) if s7 > 0 else np.nan,
            'ic7_icir': m7 / s7 if s7 > 0 else np.nan,
            'ic15_mean': m15,
            'ic15_t': m15 / (s15 / np.sqrt(len(ic15))) if s15 > 0 else np.nan,
            'n_days': len(ic7),
        })
    df = pd.DataFrame(records).sort_values('ic7_mean', key=lambda s: s.abs(),
                                           ascending=False).reset_index(drop=True)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_rows', 300)

    print(f'\n=== 换手率纯个股时序族 + 估值动量 raw rank-IC（{a.pool}股 × '
          f'{a.start[:4]}-{a.end[:4]}），按 |IC7| 降序 ===')
    cols = ['feature', 'ic7_mean', 'ic7_t', 'ic7_icir', 'ic15_mean', 'ic15_t', 'n_days']
    with pd.option_context('display.float_format', lambda v: f'{v:+.4f}'):
        print(df[cols].head(a.topn).to_string(index=False))

    sig = df[(df['ic7_t'].abs() > 2) & (df['ic7_mean'].abs() > 0.01)]
    print(f'\n显著特征 (|ic7_t|>2 且 |ic7_mean|>0.01): {len(sig)} / {len(df)}')
    for kind in ('turn_', 'val_'):
        sub = df[df['feature'].str.startswith(kind)]
        nsig = ((sub['ic7_t'].abs() > 2) & (sub['ic7_mean'].abs() > 0.01)).sum()
        print(f'  [{kind:6s}] 候选 {len(sub):2d}，显著 {nsig}')
    if len(sig):
        for _, r in sig.iterrows():
            print(f"    {r['feature']:26s} ic7={r['ic7_mean']:+.4f} t={r['ic7_t']:+.2f}")

    print(f'\n[done] 总用时 {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
