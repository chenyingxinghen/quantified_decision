#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
beta 族「时序形态派生」raw IC 诊断（小池 800×3y，不污染共享缓存）
====================================================================

背景
----
beta 协同度族（msens_*_corr/r2）在 diag_beta_family_ic 里已证明是基线未覆盖的新信号
（尤其广度协同度）。但那些是「水平」——每条 corr/r2 序列是随时间变的标量，只答
「最近跟市场变量协同得如何」，没有「协同关系正在怎么演化」。

本脚本在 corr/r2 序列上派生**时序形态量**，检验「协同关系的演化」是否携带超越水平的
新截面信号（廉价、逐日标量、直接进专家网络走横截面 rank，不重新引入失败的序列学习）：

    slope5 / slope10 : corr 序列在最近 5/10 日的滚动 OLS 斜率（协同在升还是降）
    stab20           : corr_20 序列的 20 日滚动 std（关系稳定 vs 漂移）
    change           : corr_20 − corr_60（短期 vs 长期协动发散，形态代理）

同样对 r2 做 slope5 / change 各一列做对照。共 9 变量 × 6 = 54 候选。

方法（对齐 diag_beta_family_ic）
----
取小池 800 股 × 2019-2022，每只股对 9 个市场变量做滚动回归得到 corr/r2 序列，
派生形态量后逐日横截面 rank-IC vs 7d/15d 前瞻收益。仅内存计算。

用法
  python scripts/exp/diag_beta_morph.py [--pool 800] [--start 2019-01-01] [--end 2022-01-01] [--topn 50]
"""
from __future__ import annotations

import argparse
import glob
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
DB_META = os.path.join(ROOT, 'database', 'stock_meta.db')

# 仅 corr / r2 两统计量参与形态派生
USE_STATS = ('corr', 'r2')


def pick_pool(pool: int, start: str, end: str) -> list:
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


def load_stock(code: str, start: str, end: str):
    conn = sqlite3.connect(DB_DAILY, timeout=60.0)
    try:
        d = pd.read_sql_query(
            'SELECT date, pctChg, close FROM daily_data WHERE code=? '
            'AND date>=? AND date<? ORDER BY date ASC',
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
    """corr 序列在最近 L 日的滚动 OLS 斜率（协同上升/下降）。NaN 安全。"""
    arr = y.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n < L:
        return pd.Series(out, index=y.index)
    w = sliding_window_view(arr, L)            # (n-L+1, L)
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


def derive_morph(corr20, corr60, r2_20, r2_60, prefix: str) -> Dict[str, pd.Series]:
    """从单股 corr/r2 序列派生时序形态量。返回 feat_name -> Series。"""
    out = {}
    # slope
    out[f'{prefix}_slope5_corr'] = rolling_slope(corr20, 5)
    out[f'{prefix}_slope10_corr'] = rolling_slope(corr20, 10)
    out[f'{prefix}_slope5_r2'] = rolling_slope(r2_20, 5)
    # stability
    out[f'{prefix}_stab20_corr'] = corr20.rolling(20, min_periods=10).std()
    # change (short vs long divergence)
    out[f'{prefix}_change_corr'] = corr20 - corr60
    out[f'{prefix}_change_r2'] = r2_20 - r2_60
    return out


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

    idx = ms.load_index_returns(DB_DAILY)
    glob = ms.load_market_series(DB_META)
    if glob:
        print(f'[market] 全局序列: {list(glob.keys())}')
    idx_vol = idx[ms.MARKET_INDEX].rolling(20, min_periods=13).std() if idx else None

    stock_data = {}
    for code in codes:
        d = load_stock(code, a.start, a.end)
        if d is None or len(d) < 200:
            continue
        r_i = pd.to_numeric(d['pctChg'], errors='coerce') / 100.0
        fwd = fwd_returns(d['close'].astype(float))
        stock_data[code] = (r_i, fwd[7], fwd[15])
    print(f'[load] 有效股票 {len(stock_data)}，用时 {time.time()-t0:.1f}s')

    all_dates = sorted(set().union(*[set(s[0].index) for s in stock_data.values()]))
    date_idx = pd.Index(all_dates)
    print(f'[grid] 统一交易日 {len(date_idx)}')

    # 逐股计算 corr/r2 序列 -> 派生形态量 -> 装入面板
    panels: Dict[str, pd.DataFrame] = {}
    y7 = pd.DataFrame(index=date_idx, columns=codes, dtype=float)
    y15 = pd.DataFrame(index=date_idx, columns=codes, dtype=float)

    for code in codes:
        if code not in stock_data:
            continue
        r_i, f7, f15 = stock_data[code]
        y7.loc[r_i.index, code] = f7.reindex(r_i.index).values
        y15.loc[r_i.index, code] = f15.reindex(r_i.index).values

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
            if sub.empty:
                continue
            # 取需要的统计量序列
            series = {}
            for st in USE_STATS:
                for w in (20, 60):
                    key = f'{st}_{w}'
                    if key in sub.columns:
                        series[key] = sub[key]
            if 'corr_20' not in series or 'corr_60' not in series \
                    or 'r2_20' not in series or 'r2_60' not in series:
                continue
            morph = derive_morph(series['corr_20'], series['corr_60'],
                                 series['r2_20'], series['r2_60'], prefix)
            for fname, fseries in morph.items():
                if fname not in panels:
                    panels[fname] = pd.DataFrame(index=date_idx, columns=codes,
                                                dtype=float)
                panels[fname].loc[r_i.index, code] = fseries.reindex(r_i.index).values

    print(f'[compute] 时序形态候选 {len(panels)}，用时 {time.time()-t0:.1f}s')

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
    df = pd.DataFrame(records).sort_values(
        'ic7_mean', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_rows', 200)

    print(f'\n=== beta 族时序形态派生 raw rank-IC（{a.pool}股 × '
          f'{a.start[:4]}-{a.end[:4]}），按 |IC7| 降序 ===')
    cols = ['feature', 'ic7_mean', 'ic7_t', 'ic7_icir', 'ic15_mean', 'ic15_t', 'n_days']
    with pd.option_context('display.float_format', lambda v: f'{v:+.4f}'):
        print(df[cols].head(a.topn).to_string(index=False))

    sig = df[(df['ic7_t'].abs() > 2) & (df['ic7_mean'].abs() > 0.01)]
    print(f'\n显著特征 (|ic7_t|>2 且 |ic7_mean|>0.01): {len(sig)} / {len(df)}')
    print(f'  正向: {(sig["ic7_mean"]>0).sum()}，负向: {(sig["ic7_mean"]<0).sum()}')
    # 按形态类别分组统计
    for kind in ('slope5_corr', 'slope10_corr', 'slope5_r2', 'stab20_corr',
                 'change_corr', 'change_r2'):
        sub = df[df['feature'].str.endswith(kind)]
        nsig = ((sub['ic7_t'].abs() > 2) & (sub['ic7_mean'].abs() > 0.01)).sum()
        print(f'  [{kind:12s}] 候选 {len(sub):2d}，显著 {nsig}')
    if len(sig):
        print('  显著特征列表:')
        for _, r in sig.iterrows():
            print(f"    {r['feature']:36s} ic7={r['ic7_mean']:+.4f} t={r['ic7_t']:+.2f}")

    print(f'\n[done] 总用时 {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
