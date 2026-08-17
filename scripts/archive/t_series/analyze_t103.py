"""T103 判决：**全量池树模型 vs 全量池 NAM 的回测对照** —— 头部选股能力的直接口径。

**为什么用回测而不是 IC**（2026-08-17 用户第二次指出，采纳）：lambdarank 优化 ndcg
头部截断，整截面 Rank IC 度量全体排序质量，两者不是一回事。回测的持仓就是头部 20 只，
是唯一直接检验头部能力的尺子。T101 那个「全量 lgb IC 0.091 < 全量 NAM IC 0.111」
**不能**推出「树的头部更差」。

⚠ n=1 种子没有配对功率（[[backtest-mde-is-80pp]]：n=4 时 MDE 约 80pp，n=1 更差）。
所以本脚本**只报量级与方向，不做晋级判定**。判读规则写死在输出里，避免事后挑窗口。

对照臂：T099_bt_full_s*（全量池 NAM，4 种子，同两个窗口、同一 preclose 复权口径）。
NAM 侧给出 4 种子的范围，让「树赢没赢」这件事能对着噪声带看。

用法：python -u scripts/archive/t_series/analyze_t103.py
"""

import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT)
DIRS = (os.path.join(ROOT, 'diagnose_output'),
        os.path.join(ROOT, 'diagnose_output', 'iter_output'))
WINDOWS = (('2022-09-05', '熊'), ('2024-08-05', '牛'))
TREE_ARMS = ('lgb', 'xgb', 'blend')
NAM_SEEDS = (42, 11, 23, 37)

_RE = {'ret': re.compile(r'总收益率:\s*(-?[\d.]+)%'),
       'dd': re.compile(r'最大回撤:\s*(-?[\d.]+)%'),
       'sharpe': re.compile(r'夏普比率:\s*(-?[\d.]+)'),
       'beta': re.compile(r'beta\s*:\s*(-?[\d.]+)'),
       'win': re.compile(r'盈利交易:\s*\d+\s*\(([\d.]+)%\)'),
       'trades': re.compile(r'总交易次数:\s*(\d+)')}
_RE_BENCH = re.compile(r'全市场等权\(日频再平衡\)\s+(-?[\d.]+)%')


def _read(name):
    for d in DIRS:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return open(p, encoding='utf-8', errors='replace').read()
    return None


def _parse(txt):
    o = {}
    for k, r in _RE.items():
        m = r.search(txt)
        o[k] = float(m.group(1)) if m else np.nan
    m = _RE_BENCH.search(txt)
    o['bench'] = float(m.group(1)) if m else np.nan
    o['excess'] = o['ret'] - o['bench']
    return o


def main():
    for win, wname in WINDOWS:
        print(f'\n{"="*76}\n=== {wname}市窗口 {win} ===\n{"="*76}')
        print(f"{'臂':>16} {'超额':>10} {'总收益':>9} {'β':>7} {'最大回撤':>9} "
              f"{'夏普':>7} {'胜率':>7} {'笔数':>5}")
        nam = []
        for s in NAM_SEEDS:
            t = _read(f'T099_bt_full_s{s}_{win}.log')
            if t:
                nam.append(_parse(t))
        for i, r in enumerate(nam):
            print(f"{'NAM全量 s'+str(NAM_SEEDS[i]):>16} {r['excess']:>9.2f}pp "
                  f"{r['ret']:>8.2f}% {r['beta']:>7.3f} {r['dd']:>8.2f}% "
                  f"{r['sharpe']:>7.3f} {r['win']:>6.1f}% {r['trades']:>5.0f}")
        if nam:
            ex = np.array([r['excess'] for r in nam])
            print(f"{'  → NAM 均值':>16} {ex.mean():>9.2f}pp   "
                  f"（4 种子范围 {ex.min():.2f} ~ {ex.max():.2f}pp，σ {ex.std(ddof=1):.2f}pp）")
        print()
        for arm in TREE_ARMS:
            t = _read(f'T103_bt_{arm}_s42_{win}.log')
            if not t:
                print(f'{arm+"全量 s42":>16}   （未完成）')
                continue
            r = _parse(t)
            print(f"{arm+'全量 s42':>16} {r['excess']:>9.2f}pp "
                  f"{r['ret']:>8.2f}% {r['beta']:>7.3f} {r['dd']:>8.2f}% "
                  f"{r['sharpe']:>7.3f} {r['win']:>6.1f}% {r['trades']:>5.0f}")
            if nam:
                z = (r['excess'] - ex.mean()) / ex.std(ddof=1) if ex.std(ddof=1) > 0 else np.nan
                pos = int((ex < r['excess']).sum())
                print(f"{'':>16}   vs NAM: 超过 {pos}/{len(ex)} 个种子，"
                      f"距均值 {r['excess']-ex.mean():+.2f}pp = {z:+.2f}σ_seed")

    print(f'\n{"="*76}\n判读规则（预注册，防止事后挑窗口）\n{"="*76}')
    print('  · n=1 种子，NAM 侧 σ_seed 就是噪声带宽。树臂落在 NAM 4 个种子的范围内 ⇒ '
          '无证据，不动生产。')
    print('  · 树臂在**两个窗口都**超过 NAM 全部 4 个种子，且距均值 > 2σ_seed ⇒ '
          '强证据，扩到 4 种子做正式配对判定。')
    print('  · 只在一个窗口赢 ⇒ regime 债（[[regime-stratified-ic-gate]] 的回测层版本），'
          '按 T095 门1 出局。')
    print('  · 另看胜率与笔数：头部能力若真更强，应体现在**胜率**上而不只是收益 —— '
          '收益可能来自少数暴涨股的运气。')


if __name__ == '__main__':
    main()
