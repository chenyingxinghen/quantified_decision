#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
行业/板块单因子族 raw IC 诊断（小池 800×3y，不污染共享缓存）
================================================================

背景
----
特征审计发现：247 列缓存中**行业/板块维度整体缺失**（无 industry_*/sector_*）。
行业信息是单因子结构可利用的重要上下文——「个股相对行业的强弱」天然带横截面
判别力（行业内相对位置），且不引入因子×因子交叉。

本脚本验证三类**行业单因子变换**（都是把基础单因子映射到行业上下文，无交互）：

    ind_rank_{f}       : f 在行业内当日横截面 rank（行业内相对位置）
    ind_neutral_{f}    : f − 行业当日均值（剥离行业共同成分）
    ind_z_{f}          : (f − 行业均值) / 行业 std（行业内 z-score）

基础单因子 f 从 daily_data 直接计算（不读共享缓存）：
    ret_5d / ret_20d           收益动量
    vol_20d                    pctChg 20 日波动
    hl_ma20                    (high-low)/close 20 日均值（振幅）
    amount_ma20                amount 20 日均值（量能）
    ind_rel_ret_5/20           个股收益 − 行业等权收益（行业相对强弱，本身即单因子）

对照：每个 f 的「全局 rank」版本（现状基线的横截面 rank）作为基线，
对比行业内版本的 IC 是否有增量。

方法（对齐 diag_beta_family_ic）
----
800 股 × 2019-2022（预热 2015 起），逐日横截面 rank-IC vs 7d/15d 前瞻收益。
行业内 rank 用「当日该行业 ≥5 只股票」才计算，不足则 NaN。

用法
  python scripts/exp/diag_industry_ic.py [--pool 800] [--start 2019-01-01] [--end 2022-01-01]
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

DB_DAILY = os.path.join(ROOT, 'database', 'stock_daily.db')
DB_META = os.path.join(ROOT, 'database', 'stock_meta.db')


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


def load_industry() -> Dict[str, str]:
    conn = sqlite3.connect(DB_META, timeout=60.0)
    try:
        df = pd.read_sql_query(
            "SELECT code, industry FROM stock_industry", conn)
    finally:
        conn.close()
    return dict(zip(df['code'], df['industry']))


def load_stock_full(code: str, start: str, end: str):
    """2015 起加载以预热 MA/滚动量，只保留窗口内日期返回。"""
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


def base_features(d: pd.DataFrame) -> pd.DataFrame:
    """从 OHLCV 计算基础单因子（date 索引 DataFrame）。"""
    cl = d['close'].astype(float)
    out = pd.DataFrame(index=d.index)
    with np.errstate(invalid='ignore', divide='ignore'):
        out['ret_5d'] = cl.pct_change(5)
        out['ret_20d'] = cl.pct_change(20)
        out['vol_20d'] = d['pctChg'].astype(float).rolling(20, min_periods=10).std()
        hl = (d['high'].astype(float) - d['low'].astype(float))
        out['hl_ma20'] = (hl / cl.replace(0, np.nan)).rolling(20, min_periods=10).mean()
        out['amount_ma20'] = d['amount'].astype(float).rolling(20, min_periods=10).mean()
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


def industry_panel_transform(fmat: pd.DataFrame, ind: pd.Series, mode: str) -> pd.DataFrame:
    """面板级行业变换。

    fmat: (date × stock)；ind: stock→industry（缺失行业 → NaN，不参与）。
    mode: 'rank' | 'neutral' | 'z'
    行业内股票数 <5 的日期置 NaN（rank/z 无意义）。
    """
    out = pd.DataFrame(index=fmat.index, columns=fmat.columns, dtype=float)
    codes = fmat.columns
    ind_of = pd.Series(ind).reindex(codes)
    for ind_name, members in ind_of.groupby(ind_of):
        if pd.isna(ind_name) or len(members) < 5:
            continue
        sub = fmat[members.index]
        if mode == 'rank':
            transformed = sub.rank(axis=1, pct=True)
        elif mode == 'neutral':
            transformed = sub.sub(sub.mean(axis=1), axis=0)
        else:  # z
            m = sub.mean(axis=1)
            s = sub.std(axis=1).replace(0, np.nan)
            transformed = sub.sub(m, axis=0).div(s, axis=0)
        out.loc[:, members.index] = transformed
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

    ind_all = load_industry()
    ind = {c: ind_all.get(c) for c in codes}
    n_ind = len(set(v for v in ind.values() if v))
    print(f'[industry] 覆盖 {sum(1 for v in ind.values() if v)}/{len(codes)} '
          f'只股票，{n_ind} 个行业')

    # 加载全部股票基础因子与前瞻收益
    stock_feats = {}
    for code in codes:
        d = load_stock_full(code, a.start, a.end)
        if d is None or len(d) < 200:
            continue
        feats = base_features(d)
        fwd = fwd_returns(d['close'].astype(float))
        stock_feats[code] = (feats, fwd[7], fwd[15])
    print(f'[load] 有效股票 {len(stock_feats)}，用时 {time.time()-t0:.1f}s')

    all_dates = sorted(set().union(*[set(s[0].index) for s in stock_feats.values()]))
    date_idx = pd.Index(all_dates)
    print(f'[grid] 统一交易日 {len(date_idx)}')

    # 行业等权收益（用于 ind_rel_ret）
    ind_ret = pd.DataFrame(index=date_idx, columns=sorted(set(ind.values())), dtype=float)
    ret5 = pd.DataFrame(index=date_idx, columns=codes, dtype=float)
    for code in codes:
        if code not in stock_feats:
            continue
        feats, f7, f15 = stock_feats[code]
        ret5.loc[feats.index, code] = feats['ret_5d'].reindex(feats.index).values
        iname = ind.get(code)
        if iname:
            ind_ret.loc[feats.index, iname] = feats['ret_5d'].reindex(feats.index).values
    ind_mean5 = ind_ret.mean(axis=1)

    # 组装各基础因子的面板 + 前瞻收益面板
    panels_raw: Dict[str, pd.DataFrame] = {}
    y7 = pd.DataFrame(index=date_idx, columns=codes, dtype=float)
    y15 = pd.DataFrame(index=date_idx, columns=codes, dtype=float)
    base_names = ['ret_5d', 'ret_20d', 'vol_20d', 'hl_ma20', 'amount_ma20']
    for name in base_names:
        panels_raw[name] = pd.DataFrame(index=date_idx, columns=codes, dtype=float)
    for code in codes:
        if code not in stock_feats:
            continue
        feats, f7, f15 = stock_feats[code]
        y7.loc[feats.index, code] = f7.reindex(feats.index).values
        y15.loc[feats.index, code] = f15.reindex(feats.index).values
        for name in base_names:
            panels_raw[name].loc[feats.index, code] = \
                feats[name].reindex(feats.index).values
    print(f'[panel] 基础因子面板就绪，用时 {time.time()-t0:.1f}s')

    # 行业相对收益（个股 − 行业等权，ind_rel_ret_5）
    panels_raw['ind_rel_ret_5'] = ret5.sub(ind_mean5, axis=0)

    # 生成全部候选：原始(全局)、rank、neutral、z
    ind_series = pd.Series(ind, dtype=object)
    candidates: Dict[str, pd.DataFrame] = {}
    for name in base_names + ['ind_rel_ret_5']:
        candidates[f'g_{name}'] = panels_raw[name]  # 全局原始（对照）
        candidates[f'grank_{name}'] = panels_raw[name].rank(axis=1, pct=True)
        for mode, tag in (('rank', 'indrank'), ('neutral', 'indneu'), ('z', 'indz')):
            candidates[f'{tag}_{name}'] = industry_panel_transform(
                panels_raw[name], ind_series, mode)
    print(f'[candidates] {len(candidates)} 个候选，用时 {time.time()-t0:.1f}s')

    # IC 聚合
    records = []
    for feat, mat in candidates.items():
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

    print(f'\n=== 行业单因子族 raw rank-IC（{a.pool}股 × {a.start[:4]}-{a.end[:4]}），'
          f'按 |IC7| 降序 ===')
    cols = ['feature', 'ic7_mean', 'ic7_t', 'ic7_icir', 'ic15_mean', 'ic15_t', 'n_days']
    with pd.option_context('display.float_format', lambda v: f'{v:+.4f}'):
        print(df[cols].head(a.topn).to_string(index=False))

    sig = df[(df['ic7_t'].abs() > 2) & (df['ic7_mean'].abs() > 0.01)]
    print(f'\n显著特征 (|ic7_t|>2 且 |ic7_mean|>0.01): {len(sig)} / {len(df)}')
    if len(sig):
        for _, r in sig.iterrows():
            print(f"    {r['feature']:24s} ic7={r['ic7_mean']:+.4f} t={r['ic7_t']:+.2f}")

    # 对比：同一基础因子，全局 vs 行业变换 的 IC
    print('\n=== 全局 vs 行业变换 对比（同一基础因子）===')
    for name in base_names + ['ind_rel_ret_5']:
        row = {}
        for tag in ('g_', 'grank_', 'indrank_', 'indneu_', 'indz_'):
            key = f'{tag}{name}'
            r = df[df['feature'] == key]
            if len(r):
                row[tag.rstrip('_')] = f"{r.iloc[0]['ic7_mean']:+.4f}(t{r.iloc[0]['ic7_t']:+.1f})"
        print(f"  {name:16s} " + "  ".join(f"{k}={v}" for k, v in row.items()))

    print(f'\n[done] 总用时 {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
