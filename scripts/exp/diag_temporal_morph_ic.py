#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
广义指标「时序曲率/跨期差」单因子族 raw IC 诊断（小池 800×3y）
================================================================

背景
----
特征审计发现：247 列基线中**时序形态（斜率/曲率/跨期差）近乎空白**——
仅 acceleration_5d/10d、ma_slope_14、amount/volume_change_rate 12 列。
而 beta 族证明「同一指标序列的演化量」携带强信号（msens slope/change/stab）。

本脚本对基线常见**基础单指标**（收益/波动/振幅/量能/乖离）做三类**单指标时序派生**
（不涉及任何指标×指标交叉，全部是同一序列的变换）：

    spread_{f}    : f_short − f_long（短窗 vs 长窗水平差 = 跨期差，形态代理）
    slope5_{f}    : f 最近 5 日滚动 OLS 斜率（上升/下降速度）
    accel2_{f}    : f 的二阶差分（加速度/曲率）

基础单指标 f（从 daily_data 直接计算，不读共享缓存）：
    ret_5d / ret_20d      收益水平（短/长）
    vol_20d               波动
    hl_ma20               振幅
    amount_ma20           量能
    rsi_12                乖离

派生组合：
    spread: (ret_5d−ret_20d), (vol_20d−vol_5d), (amount_ma20−amount_ma5),
            (hl_ma20−hl_ma5), (rsi_12−rsi_6)
    slope5: 对 ret_20d / vol_20d / amount_ma20 / hl_ma20 / rsi_12
    accel2: 对 ret_20d / vol_20d / amount_ma20 / hl_ma20 / rsi_12

方法（对齐 diag_beta_family_ic / diag_industry_ic）
----
800 股 × 2019-2022（预热 2015 起），逐日横截面 rank-IC vs 7d/15d 前瞻收益。
输出按 |IC7| 排序；再按派生类别汇总显著数。

用法
  python scripts/exp/diag_temporal_morph_ic.py [--pool 800] [--start 2019-01-01] [--end 2022-01-01]
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


def pick_pool(pool: int, start: str, end: str) -> List[str]:
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


def load_stock_full(code: str, start: str, end: str):
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
    """最近 L 日滚动 OLS 斜率。与 diag_beta_morph 逐位一致。"""
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


def base_levels(d: pd.DataFrame) -> Dict[str, pd.Series]:
    """基础单指标水平序列（含短窗用于跨期差）。"""
    cl = d['close'].astype(float)
    pc = d['pctChg'].astype(float)
    amt = d['amount'].astype(float)
    hl = (d['high'].astype(float) - d['low'].astype(float))
    out = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        out['ret_5d'] = cl.pct_change(5)
        out['ret_20d'] = cl.pct_change(20)
        out['vol_5d'] = pc.rolling(5, min_periods=3).std()
        out['vol_20d'] = pc.rolling(20, min_periods=10).std()
        out['hl_ma5'] = (hl / cl.replace(0, np.nan)).rolling(5, min_periods=3).mean()
        out['hl_ma20'] = (hl / cl.replace(0, np.nan)).rolling(20, min_periods=10).mean()
        out['amount_ma5'] = amt.rolling(5, min_periods=3).mean()
        out['amount_ma20'] = amt.rolling(20, min_periods=10).mean()
        # RSI(6)/RSI(12)
        up = pc.clip(lower=0)
        dn = (-pc).clip(lower=0)
        for n in (6, 12):
            au = up.rolling(n, min_periods=3).mean()
            ad_ = dn.rolling(n, min_periods=3).mean()
            rs = au / ad_.replace(0, np.nan)
            out[f'rsi_{n}'] = 100 - 100 / (1 + rs)
    return out


def derive(f: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    """从基础水平派生时序形态量（全部单指标变换）。"""
    out = {}
    # 跨期差 spread
    spread_pairs = [('ret_5d', 'ret_20d'), ('vol_5d', 'vol_20d'),
                    ('hl_ma5', 'hl_ma20'), ('amount_ma5', 'amount_ma20'),
                    ('rsi_6', 'rsi_12')]
    for short, long in spread_pairs:
        if short in f and long in f:
            out[f'spread_{short}_x_{long}'] = f[short] - f[long]
    # 斜率 slope5（对长窗水平）
    for name in ('ret_20d', 'vol_20d', 'hl_ma20', 'amount_ma20', 'rsi_12'):
        if name in f:
            out[f'slope5_{name}'] = rolling_slope(f[name], 5)
    # 加速度 accel2（二阶差分，对长窗水平）
    for name in ('ret_20d', 'vol_20d', 'hl_ma20', 'amount_ma20', 'rsi_12'):
        if name in f:
            out[f'accel2_{name}'] = f[name].diff().diff()
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
        stock_data[code] = (base_levels(d), fwd[7], fwd[15])
    print(f'[load] 有效股票 {len(stock_data)}，用时 {time.time()-t0:.1f}s')

    all_dates = sorted(set().union(*[set(s[0]['ret_20d'].index) for s in stock_data.values()]))
    date_idx = pd.Index(all_dates)

    # 先算一次派生确定列名集合
    probe = derive(next(iter(stock_data.values()))[0])
    feat_names = list(probe.keys())
    panels: Dict[str, pd.DataFrame] = {
        name: pd.DataFrame(index=date_idx, columns=codes, dtype=float)
        for name in feat_names}
    y7 = pd.DataFrame(index=date_idx, columns=codes, dtype=float)
    y15 = pd.DataFrame(index=date_idx, columns=codes, dtype=float)

    for code in codes:
        if code not in stock_data:
            continue
        lev, f7, f15 = stock_data[code]
        idx = lev['ret_20d'].index
        y7.loc[idx, code] = f7.reindex(idx).values
        y15.loc[idx, code] = f15.reindex(idx).values
        der = derive(lev)
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

    print(f'\n=== 广义指标时序曲率/跨期差 raw rank-IC（{a.pool}股 × '
          f'{a.start[:4]}-{a.end[:4]}），按 |IC7| 降序 ===')
    cols = ['feature', 'ic7_mean', 'ic7_t', 'ic7_icir', 'ic15_mean', 'ic15_t', 'n_days']
    with pd.option_context('display.float_format', lambda v: f'{v:+.4f}'):
        print(df[cols].head(a.topn).to_string(index=False))

    sig = df[(df['ic7_t'].abs() > 2) & (df['ic7_mean'].abs() > 0.01)]
    print(f'\n显著特征 (|ic7_t|>2 且 |ic7_mean|>0.01): {len(sig)} / {len(df)}')
    for kind in ('spread_', 'slope5_', 'accel2_'):
        sub = df[df['feature'].str.startswith(kind)]
        nsig = ((sub['ic7_t'].abs() > 2) & (sub['ic7_mean'].abs() > 0.01)).sum()
        print(f'  [{kind:8s}] 候选 {len(sub):2d}，显著 {nsig}')
    if len(sig):
        print('  显著特征列表:')
        for _, r in sig.iterrows():
            print(f"    {r['feature']:28s} ic7={r['ic7_mean']:+.4f} t={r['ic7_t']:+.2f}")

    print(f'\n[done] 总用时 {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
