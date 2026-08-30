#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量价背离 / 估值动量 单因子族 raw IC 诊断（小池 800×3y，不污染共享缓存）
=====================================================================

背景
----
特征审计发现：基线 37 个量能列但**无估值动量**（peTTM/pbMRQ 是水平值，无
时序变化率），也缺「量-价-估值」三角关系的时序形态。daily_data 本身带
volume/amount/turnover_rate/peTTM/pbMRQ/psTTM/pcfNcfTTM，可直接构造。

候选（全部**单因子结构**——单指标时序派生或单对序列关系统计，无因子×因子交叉）：

  估值动量（基线缺口）:
    val_mom_pe20       : peTTM 20 日 pct_change（估值膨胀/收缩）
    val_mom_pb20       : pbMRQ 20 日 pct_change
    val_mom_ps20       : psTTM 20 日 pct_change
    val_accel_pe5      : peTTM 5 日变化的一阶差（估值加速度）
    val_pe_turn_corr20 : peTTM 与 turnover_rate 的 20 日滚动相关（估值×换手协同）

  量价背离（补强，测是否被基线覆盖）:
    pv_div20           : 20 日收益 rank − 20 日量能变化 rank（价强量弱=负背离）
    vol_skew20         : volume 20 日偏度（量能分布形态）
    vol_corr_ret20     : volume 与 pctChg 的 20 日滚动相关（当日量价协同）

方法（对齐 diag_beta_family_ic）
----
800 股 × 2019-2022，逐日横截面 rank-IC vs 7d/15d 前瞻收益。
输出按 |IC7| 排序，再按类别汇总。

用法
  python scripts/exp/diag_volume_value_ic.py [--pool 800] [--start 2019-01-01] [--end 2022-01-01]
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
            'SELECT date, open, high, low, close, pctChg, volume, amount, '
            'turnover_rate, peTTM, pbMRQ, psTTM, pcfNcfTTM FROM daily_data '
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


def derive(d: pd.DataFrame) -> Dict[str, pd.Series]:
    out = {}
    cl = d['close'].astype(float)
    vol = d['volume'].astype(float)
    pc = d['pctChg'].astype(float)
    to = d['turnover_rate'].astype(float)
    pe = pd.to_numeric(d['peTTM'], errors='coerce')
    pb = pd.to_numeric(d['pbMRQ'], errors='coerce')
    ps = pd.to_numeric(d['psTTM'], errors='coerce')
    with np.errstate(invalid='ignore', divide='ignore'):
        # 估值动量
        out['val_mom_pe20'] = pe.pct_change(20)
        out['val_mom_pb20'] = pb.pct_change(20)
        out['val_mom_ps20'] = ps.pct_change(20)
        out['val_accel_pe5'] = pe.pct_change(5).diff()
        out['val_pe_turn_corr20'] = pe.rolling(20, min_periods=10).corr(to)
        # 量价背离
        r20 = cl.pct_change(20)
        v20 = vol.pct_change(20)
        out['pv_div20'] = r20.rank(pct=True) - v20.rank(pct=True)
        out['vol_skew20'] = vol.rolling(20, min_periods=10).skew()
        out['vol_corr_ret20'] = vol.rolling(20, min_periods=10).corr(pc)
    return {k: v for k, v in out.items() if v is not None}


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

    # 探测列名
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

    print(f'\n=== 量价背离/估值动量 raw rank-IC（{a.pool}股 × '
          f'{a.start[:4]}-{a.end[:4]}），按 |IC7| 降序 ===')
    cols = ['feature', 'ic7_mean', 'ic7_t', 'ic7_icir', 'ic15_mean', 'ic15_t', 'n_days']
    with pd.option_context('display.float_format', lambda v: f'{v:+.4f}'):
        print(df[cols].head(a.topn).to_string(index=False))

    sig = df[(df['ic7_t'].abs() > 2) & (df['ic7_mean'].abs() > 0.01)]
    print(f'\n显著特征 (|ic7_t|>2 且 |ic7_mean|>0.01): {len(sig)} / {len(df)}')
    for kind in ('val_', 'pv_', 'vol_'):
        sub = df[df['feature'].str.startswith(kind)]
        nsig = ((sub['ic7_t'].abs() > 2) & (sub['ic7_mean'].abs() > 0.01)).sum()
        print(f'  [{kind:6s}] 候选 {len(sub):2d}，显著 {nsig}')
    if len(sig):
        for _, r in sig.iterrows():
            print(f"    {r['feature']:26s} ic7={r['ic7_mean']:+.4f} t={r['ic7_t']:+.2f}")

    print(f'\n[done] 总用时 {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
