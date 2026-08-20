#!/usr/bin/env python
"""回测的**日频配对**分析：把两条净值曲线按日对齐做配对差，并按基准涨跌日分层。

为什么需要它：终值收益在 n=1 种子下分辨率约 80~100pp（[[backtest-mde-is-80pp]]），
四个不同模型的牛市超额铺开 100pp 与同一模型跨种子铺开 82pp 无法区分。日频配对
有 400+ 个观测，能回答「差异是系统性的还是几天造成的」，并给出跌日分解 ——
后者对应 IC 侧的分层否决门（[[regime-stratified-ic-gate]]）。

⚠ 分辨率的边界要说清：日频配对提高的是「这两个**模型实例**是否不同」的分辨率，
  **不是**「这条轴是否有用」。n=1 种子下轴效应与种子特异性不可分离，
  所以显著的日频差异**仍不能**作为晋级证据。

用法：
    python scripts/exp/diag_backtest_daily_paired.py --tag T120 --base T115_idxrel_s42
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

WINDOWS = {'2022-09-05': '熊 22-09→24-08', '2024-08-05': '牛 24-08→26-08'}


def _find_runs(tag):
    runs = {}
    for eq in glob.glob(os.path.join('backtest_result', '**', 'backtest_equity_curve.csv'),
                        recursive=True):
        d = os.path.dirname(eq)
        base = os.path.basename(d)
        if f'_{tag}_' not in base:
            continue
        m = re.search(r'_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})$', base)
        if not m:
            continue
        arm = base.split(f'_{tag}_', 1)[1].split('_minp')[0]
        runs[(arm, m.group(1))] = d
    return runs


def _daily(d):
    eq = pd.read_csv(os.path.join(d, 'backtest_equity_curve.csv'))
    eq['date'] = pd.to_datetime(eq['date'])
    s = eq.set_index('date')['equity'].astype(float)
    return s.pct_change().dropna()


def _bench(d):
    p = os.path.join(d, 'benchmark_daily_return_rebal.csv')
    if not os.path.exists(p):
        return None
    b = pd.read_csv(p)
    dc = [c for c in b.columns if 'date' in c.lower()][0]
    vc = [c for c in b.columns if c != dc][0]
    b[dc] = pd.to_datetime(b[dc])
    return b.set_index(dc)[vc].astype(float)


def _t(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float('nan'), float('nan')
    se = x.std(ddof=1) / np.sqrt(n)
    return (x.mean() / se if se > 0 else float('nan')), se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='T120')
    ap.add_argument('--base', required=True, help='作为配对基线的臂名')
    a = ap.parse_args()

    runs = _find_runs(a.tag)
    arms = sorted({k[0] for k in runs})
    wins = sorted({k[1] for k in runs})
    if a.base not in arms:
        raise SystemExit(f'基线 {a.base} 不在 {arms}')

    for w in wins:
        bd = runs.get((a.base, w))
        if bd is None:
            continue
        rb = _daily(bd)
        bench = _bench(bd)
        print(f'===== {WINDOWS.get(w, w)}  基线={a.base}  n={len(rb)} 交易日 =====')
        print(f'{"候选":24s} {"Δ日均bp":>9s} {"t":>7s} {"胜日占比":>9s} '
              f'{"跌日Δbp":>9s} {"跌日t":>7s} {"涨日Δbp":>9s} {"最大单日|Δ|贡献":>15s}')
        for arm in arms:
            if arm == a.base:
                continue
            cd = runs.get((arm, w))
            if cd is None:
                continue
            rc = _daily(cd)
            idx = rb.index.intersection(rc.index)
            d = (rc.reindex(idx) - rb.reindex(idx)).dropna()
            t, _ = _t(d)
            top = np.abs(d).max() / np.abs(d).sum() * 100 if len(d) else float('nan')
            if bench is not None:
                bi = bench.reindex(d.index)
                dn = d[bi < 0]
                up = d[bi >= 0]
                tdn, _ = _t(dn)
                s_dn = f'{dn.mean()*1e4:9.2f}'
                s_tdn = f'{tdn:7.2f}'
                s_up = f'{up.mean()*1e4:9.2f}'
            else:
                s_dn = s_tdn = s_up = '        -'
            print(f'{arm:24s} {d.mean()*1e4:9.2f} {t:7.2f} '
                  f'{(d > 0).mean()*100:8.1f}% {s_dn} {s_tdn} {s_up} {top:14.1f}%')
        print()
    print('Δ 单位 bp = 万分之一/日。t 为配对 t 统计量（|t|>2 约等于 5% 水平显著）。')
    print('「最大单日|Δ|贡献」= 单日最大绝对差占全部绝对差之和的比例，越高说明差异越像少数几天造成。')
    print('⚠ n=1 种子：显著只说明这两个模型实例不同，不能推断这条轴有用。')


if __name__ == '__main__':
    main()
