"""E5：实现持有期分布测量（改 FUTURE_DAYS 之前必须先做的测量）。

动机
----
标签是 7 日前向收益，但曾有推测认为「两年约 84 次换手 × 20 持仓」意味着
实际平均持有期远短于 7 日，从而标签与执行错配。这个推测必须先用回测
交易记录证实或证伪，再决定是否要动 FUTURE_DAYS —— 盲扫 5/10 是浪费。

做法
----
直接从 backtest_trades.csv 统计 (sell_date - buy_date) 的交易日跨度分布，
而非自然日；并按分位数与直方图给出形状，同时区分正常到期卖出与
止损/止盈等提前卖出。

用法
----
  python scripts/archive/e_series/diag_holding_period.py --trades <path> [--trades <path> ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


def analyze(path: str, label: str):
    df = pd.read_csv(path)
    bcol = _pick(df, 'buy_date', 'entry_date', 'open_date')
    scol = _pick(df, 'sell_date', 'exit_date', 'close_date')
    if bcol is None or scol is None:
        raise ValueError(f'{path} 缺少买/卖日期列，实有列: {list(df.columns)}')

    b = pd.to_datetime(df[bcol])
    s = pd.to_datetime(df[scol])

    # 交易日跨度：用全体出现过的日期构成交易日历，避免自然日高估
    cal = pd.Index(sorted(set(b.dropna()) | set(s.dropna())))
    pos = {d: i for i, d in enumerate(cal)}
    hold = np.array([pos.get(y, np.nan) - pos.get(x, np.nan)
                     for x, y in zip(b, s)], dtype=float)
    hold = hold[np.isfinite(hold)]

    rcol = _pick(df, 'sell_reason', 'exit_reason', 'reason')
    reasons = (df[rcol].value_counts().to_dict() if rcol else {})

    q = {f'p{p}': float(np.percentile(hold, p)) for p in (5, 25, 50, 75, 95)}
    out = {
        'label': label, 'trades_file': path, 'n_trades': int(len(hold)),
        'mean': float(hold.mean()), 'median': float(np.median(hold)),
        'std': float(hold.std(ddof=1)), 'min': float(hold.min()),
        'max': float(hold.max()), 'quantiles': q,
        'hist': {str(int(k)): int(v) for k, v in
                 zip(*np.unique(np.clip(hold, 0, 20), return_counts=True))},
        'sell_reasons': reasons,
    }

    print('=' * 72)
    print(f'{label}   n={out["n_trades"]}')
    print(f'  均值 {out["mean"]:.2f} 交易日 | 中位 {out["median"]:.1f} | '
          f'std {out["std"]:.2f} | 范围 [{out["min"]:.0f}, {out["max"]:.0f}]')
    print('  分位: ' + '  '.join(f'{k}={v:.0f}' for k, v in q.items()))
    print('  直方图(交易日 -> 笔数):')
    for k, v in sorted(out['hist'].items(), key=lambda kv: int(kv[0])):
        bar = '#' * max(1, int(60 * v / max(out['hist'].values())))
        print(f'    {int(k):>3} | {v:>5}  {bar}')
    if reasons:
        print('  卖出原因: ' + ', '.join(f'{k}={v}' for k, v in reasons.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trades', action='append', required=True,
                    help='backtest_trades.csv 路径，可重复')
    ap.add_argument('--label', action='append', default=[])
    ap.add_argument('--out', default='diagnose_output/holding_period.json')
    args = ap.parse_args()

    results = []
    for i, p in enumerate(args.trades):
        lab = args.label[i] if i < len(args.label) else os.path.basename(os.path.dirname(p))
        results.append(analyze(p, lab))

    print('\n' + '=' * 72)
    print('结论口径：标签为 7 日前向收益。若实现持有期中位/均值 ≈ 7，则标签与')
    print('执行对齐，horizon 轴无需改动；若显著小于 7，才有理由动 FUTURE_DAYS。')
    print('=' * 72)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'已保存: {args.out}')


if __name__ == '__main__':
    main()
