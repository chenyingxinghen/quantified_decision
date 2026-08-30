#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
beta 特征族 raw IC 诊断（小池 800×3y，不污染共享缓存）
=====================================================

目的
----
在投入「物化到独立缓存 + 配对重训」之前，先用最低成本判断 beta 家族是否携带
可外推的截面预测信号。仅内存计算，不读写 ``factors_cache``。

方法
----
1. 取小池 800 只股票 × 3 年窗口（训练期内，2019-01-01→2022-01-01）。
2. 每只股票：日收益（daily_data.pctChg/100）对 9 个市场变量做滚动回归
   （market_sensitivity.compute_sensitivity），产出 beta/alpha/r2/idio_vol/corr
   × 窗口{20,60}。
3. 前瞻收益标签：7 日 / 15 日（对齐生产「7 日前向 rank 标签」口径，close.shift(-h)/close-1）。
4. 逐日横截面 rank-IC：每天对 (特征, 前瞻收益) 做截面 spearman，跨日聚合
   mean / t / ICIR。

输出
----
按 |mean IC(7d)| 降序的特征表，并标注现有 idx_beta_60 等价族（msens_idxret）作为 sanity。

用法
  python scripts/exp/diag_beta_family_ic.py [--pool 800] [--start 2019-01-01] [--end 2022-01-01] [--topn 40]
"""
from __future__ import annotations

import argparse
import glob
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

from core.factors import market_sensitivity as ms

DB_DAILY = os.path.join(ROOT, 'database', 'stock_daily.db')
DB_META = os.path.join(ROOT, 'database', 'stock_meta.db')
CACHE = os.path.join(ROOT, 'database', 'system_data', 'factors_cache')

MARKET_REL_FILES = glob.glob(os.path.join(CACHE, '*.parquet'))


def pick_pool(pool: int, start: str, end: str) -> List[str]:
    """取窗口内交易日最多的前 pool 只股票。"""
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
    """close.shift(-h)/close - 1，返回 dict h→Series(与 close 同索引)。"""
    c = close.astype(float).to_numpy()
    out = {}
    for h in horizons:
        fc = np.concatenate([c[h:], np.full(min(h, len(c)), np.nan)])[:len(c)]
        with np.errstate(invalid='ignore', divide='ignore'):
            out[h] = pd.Series(fc / np.where(c == 0, np.nan, c) - 1.0,
                               index=close.index)
    return out


def per_day_rank_ic(fmat: pd.DataFrame, ymat: pd.DataFrame):
    """逐日横截面 rank-IC。

    fmat / ymat：index=date, columns=stock, values=feature / forward return。
    返回 per-day IC 的 numpy 数组（已剔除不足 30 只股票的日期）。
    """
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
    ap.add_argument('--topn', type=int, default=40)
    a = ap.parse_args()

    t0 = time.time()
    codes = pick_pool(a.pool, a.start, a.end)
    print(f'[pool] {len(codes)} 只股票，窗口 {a.start}→{a.end}，'
          f'用时 {time.time()-t0:.1f}s')

    idx = ms.load_index_returns(DB_DAILY)
    glob = ms.load_market_series(DB_META)
    if glob:
        print(f'[market] 全局序列: {list(glob.keys())}')
    else:
        print('[WARN] market_macro_daily / market_sentiment 缺失，仅指数相关变量可用')

    # 预先载入指数波动序列
    idx_vol = idx[ms.MARKET_INDEX].rolling(20, min_periods=13).std() if idx else None

    # 收集：code -> (dates, r_i, fwd7, fwd15)
    stock_data = {}
    for code in codes:
        d = load_stock(code, a.start, a.end)
        if d is None or len(d) < 200:
            continue
        r_i = pd.to_numeric(d['pctChg'], errors='coerce') / 100.0
        fwd = fwd_returns(d['close'].astype(float))
        stock_data[code] = (r_i, fwd[7], fwd[15])
    print(f'[load] 有效股票 {len(stock_data)}，用时 {time.time()-t0:.1f}s')

    # 统一日期网格（取所有股票日期并集）
    all_dates = sorted(set().union(*[set(s[0].index) for s in stock_data.values()]))
    date_idx = pd.Index(all_dates)
    print(f'[grid] 统一交易日 {len(date_idx)}')

    # 逐变量计算敏感度，并装入 (date × stock) 面板
    panels = {}  # feat_name -> DataFrame(date, stock)
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
            sub = ms.compute_sensitivity(r_i, x)  # index=stock dates
            for col in sub.columns:
                full = f'{prefix}_{col}'
                if full not in panels:
                    panels[full] = pd.DataFrame(index=date_idx, columns=codes, dtype=float)
                panels[full].loc[r_i.index, code] = sub[col].reindex(r_i.index).values

    print(f'[compute] 候选特征 {len(panels)}，用时 {time.time()-t0:.1f}s')

    # IC 聚合
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
            'ic7_mean': m7, 'ic7_t': m7 / (s7 / np.sqrt(len(ic7))) if s7 > 0 else np.nan,
            'ic7_icir': m7 / s7 if s7 > 0 else np.nan,
            'ic15_mean': m15, 'ic15_t': m15 / (s15 / np.sqrt(len(ic15))) if s15 > 0 else np.nan,
            'n_days': len(ic7),
        })
    df = pd.DataFrame(records).sort_values('ic7_mean', key=lambda s: s.abs(),
                                           ascending=False).reset_index(drop=True)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 200)

    print(f'\n=== beta 家族 raw rank-IC（{a.pool}股 × {a.start[:4]}-{a.end[:4]}），'
          f'按 |IC7| 降序 ===')
    cols = ['feature', 'ic7_mean', 'ic7_t', 'ic7_icir', 'ic15_mean', 'ic15_t', 'n_days']
    with pd.option_context('display.float_format', lambda v: f'{v:+.4f}'):
        print(df[cols].head(a.topn).to_string(index=False))

    # 摘要：显著特征数（|t|>2 且 |mean|>0.01）
    sig = df[(df['ic7_t'].abs() > 2) & (df['ic7_mean'].abs() > 0.01)]
    print(f'\n显著特征 (|ic7_t|>2 且 |ic7_mean|>0.01): {len(sig)} / {len(df)}')
    print(f'  正向(预测涨): {(sig["ic7_mean"]>0).sum()}，负向: {(sig["ic7_mean"]<0).sum()}')
    if len(sig):
        print('  显著特征列表:')
        for _, r in sig.iterrows():
            print(f"    {r['feature']:32s} ic7={r['ic7_mean']:+.4f} t={r['ic7_t']:+.2f}")

    print(f'\n[done] 总用时 {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
