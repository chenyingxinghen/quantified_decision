#!/usr/bin/env python
"""汇总 T120 回测矩阵（以及任意 tag）的两窗读数，输出可直接进台账的表。

用法：
    python scripts/exp/collect_backtest_matrix.py --tag T120 [--also T099]

只读 backtest_result/**/backtest_{metrics,benchmark}.json，不重算任何东西。
基准口径固定取「全市场等权(日频再平衡)」—— 与 T099/T103 那批台账读数同口径。
"""
import argparse
import glob
import json
import os
import re

BENCH_KEY = '全市场等权(日频再平衡)'
WINDOWS = {'2022-09-05': '熊 22-09→24-08', '2024-08-05': '牛 24-08→26-08'}


def collect(tag):
    rows = []
    pat = os.path.join('backtest_result', '**', 'backtest_benchmark.json')
    for bf in glob.glob(pat, recursive=True):
        d = os.path.dirname(bf)
        if f'_{tag}_' not in os.path.basename(d):
            continue
        m = re.search(r'_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})$', os.path.basename(d))
        if not m:
            continue
        bench = json.load(open(bf, encoding='utf-8')).get(BENCH_KEY, {})
        if not bench.get('available'):
            continue
        mf = os.path.join(d, 'backtest_metrics.json')
        met = json.load(open(mf, encoding='utf-8')) if os.path.exists(mf) else {}
        arm = os.path.basename(d).split(f'_{tag}_', 1)[1].split('_minp')[0]
        rows.append({
            'arm': arm,
            'win': WINDOWS.get(m.group(1), m.group(1)),
            'ret': bench['strategy']['total_return_pct'],
            'bench': bench['benchmark']['total_return_pct'],
            'excess': bench['excess_total_pct'],
            'beta': bench['beta'],
            'alpha': bench['alpha_annual_pct'],
            'ir': bench['information_ratio'],
            'mdd': bench['strategy']['max_drawdown_pct'],
            'sharpe': bench['strategy']['sharpe'],
            'winrate': met.get('win_rate'),
            'trades': met.get('total_trades'),
        })
    return rows


def show(rows, title):
    if not rows:
        print(f'[{title}] 无产物')
        return
    print(f'===== {title} =====')
    hdr = (f'{"臂":24s} {"窗口":16s} {"收益%":>9s} {"基准%":>9s} {"超额pp":>8s} '
           f'{"β":>6s} {"α年化%":>8s} {"IR":>6s} {"回撤%":>8s} {"夏普":>6s} {"胜率%":>7s} {"笔数":>5s}')
    print(hdr)
    for r in sorted(rows, key=lambda x: (x['win'], x['arm'])):
        wr = f"{r['winrate']:.2f}" if r['winrate'] is not None else '-'
        tr = f"{r['trades']}" if r['trades'] is not None else '-'
        print(f'{r["arm"]:24s} {r["win"]:16s} {r["ret"]:9.2f} {r["bench"]:9.2f} '
              f'{r["excess"]:8.2f} {r["beta"]:6.3f} {r["alpha"]:8.2f} {r["ir"]:6.3f} '
              f'{r["mdd"]:8.2f} {r["sharpe"]:6.3f} {wr:>7s} {tr:>5s}')
    print()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='T120')
    ap.add_argument('--also', action='append', default=[])
    a = ap.parse_args()
    show(collect(a.tag), a.tag)
    for t in a.also:
        show(collect(t), t)
