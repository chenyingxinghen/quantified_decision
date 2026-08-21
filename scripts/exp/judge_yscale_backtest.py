#!/usr/bin/env python
"""T134 判定：用回测检验「y_scale = 静态风险偏好旋钮」这个**机制说法**。

⚠ 这个脚本不产生晋级结论。y4/y8 已在 T133 被跌日否决门拒掉，回测只能否决不能晋级
  （[[alpha-not-rules-ic-first-protocol]]）。这里只回答一个问题：**机制说法对不对。**

待检验假设 H（T133 之后我给出的解释）
------------------------------------
  T133 观察到 y_scale 拧大时"涨日头部略好、跌日头部塌"，我据此说
  y_scale 是一个**静态风险偏好旋钮**。若为真，可证伪预测：
    H1  **β 随 y_scale 单调上升**  ← 核心预测，也最容易测准
    H2  牛市窗总收益随 y_scale 上升
    H3  熊市窗总收益随 y_scale 下降
    H4  上行捕获与下行捕获**同向变大**（= 暴露变大，不是选股变好）
  反过来若 β 平坦/非单调，则"风险偏好"这个说法**被证伪**，
  T133 的涨跌日不对称就得换解释（最可能是有效样本量塌缩带来的纯噪声）。

为什么判据挂在 β 上而不是收益上
--------------------------------
  n=2 种子的终值回测 MDE 约 80pp（[[backtest-mde-is-80pp]]），H2/H3 基本判不动，
  只能看方向是否一致、不看幅度。而 β 由 ~460 个交易日的日频回归估出，
  标准误小两个数量级 —— H1 才是真正可判的那条。

  β 的判据（预注册）：
    · 两个种子**都**给出同一形状（单调增），且 y8 与 y1 的 β 差 ≥ 0.10
      ⇒ 机制成立
    · 两个种子形状不一致，或极差 < 0.10 ⇒ 机制**被证伪**
  0.10 这条线的来历：T120 那批同口径回测里，不同存档间 β 的散布约 0.1~0.2，
  低于这个量级就分不清"剂量效应"和"存档之间本来就有的差异"。

用法
  python scripts/exp/judge_yscale_backtest.py --tag T134 --arms 1,2,4,8 --seeds 42,11
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BENCH_KEY_HINT = '日频再平衡'      # 与 T120 一致：用日频再平衡的全市场等权基准


def find_run(archive, start, tag):
    """定位 backtest_result/<archive>/nam/<tag...>_<start>_to_*/ 。"""
    pat = os.path.join('backtest_result', archive, 'nam', f'*{tag}*{start}_to_*')
    hits = sorted(glob.glob(pat))
    return hits[-1] if hits else None


def load_run(d):
    """取回测指标 + 基准对照（β/α/捕获率）。"""
    out = {}
    mp = os.path.join(d, 'backtest_metrics.json')
    if os.path.exists(mp):
        m = json.load(open(mp, encoding='utf-8'))
        out.update(ret_pct=m.get('total_return_pct'), sharpe=m.get('sharpe_ratio'),
                   mdd=m.get('max_drawdown'), win=m.get('win_rate'),
                   trades=m.get('total_trades'), hold=m.get('avg_holding_days'))
    bp = os.path.join(d, 'backtest_benchmark.json')
    if os.path.exists(bp):
        b = json.load(open(bp, encoding='utf-8'))
        key = next((k for k in b if BENCH_KEY_HINT in k), None) or next(iter(b), None)
        if key:
            v = b[key]
            out.update(beta=v.get('beta'), corr=v.get('corr'),
                       alpha=v.get('alpha_annual_pct'),
                       bench_ret=v.get('benchmark', {}).get('total_return_pct'),
                       vol=v.get('strategy', {}).get('annual_vol_pct'),
                       excess=v.get('excess_total_pct'), bench_key=key)
    # 上下行捕获：用逐日资金曲线与基准日收益自算（产物里没有现成字段）
    ep = os.path.join(d, 'backtest_equity_curve.csv')
    bpp = os.path.join(d, 'benchmark_daily_return_rebal.csv')
    if os.path.exists(ep) and os.path.exists(bpp):
        import csv
        eq = {}
        with open(ep, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                eq[row['date']] = float(row['equity'])
        bd = {}
        with open(bpp, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                bd[row['date']] = float(row['benchmark_ret'])
        ds = sorted(set(eq) & set(bd))
        if len(ds) > 20:
            e = np.array([eq[d_] for d_ in ds])
            sr = np.diff(e) / e[:-1]
            br = np.array([bd[d_] for d_ in ds])[1:]
            up, dn = br > 0, br < 0
            out['cap_up'] = float(sr[up].mean() / br[up].mean()) if up.sum() > 5 else None
            out['cap_dn'] = float(sr[dn].mean() / br[dn].mean()) if dn.sum() > 5 else None
            out['n_days'] = len(sr)
    return out


def shape_of(vals):
    d = np.sign(np.diff(vals))
    d = d[d != 0]
    if len(d) == 0:
        return 'flat'
    turns = int((np.diff(d) != 0).sum()) if len(d) > 1 else 0
    if turns == 0:
        return 'mono_up' if d[0] > 0 else 'mono_down'
    return 'peak' if turns == 1 and d[0] > 0 else ('valley' if turns == 1 else 'zigzag')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='T134')
    ap.add_argument('--arms', default='1,2,4,8')
    ap.add_argument('--seeds', default='42,11')
    ap.add_argument('--archive-fmt', default='T133_yscale{arm}_s{seed}')
    ap.add_argument('--bear-start', default='2022-09-05')
    ap.add_argument('--bull-start', default='2024-08-05')
    ap.add_argument('--beta-spread-line', type=float, default=0.10,
                    help='β 极差需 ≥ 此值才算"剂量效应可分辨"（T120 存档间 β 散布约 0.1~0.2）')
    ap.add_argument('--out', default='diagnose_output/T134_yscale_backtest.json')
    a = ap.parse_args()

    arms = [x.strip() for x in a.arms.split(',') if x.strip()]
    seeds = [x.strip() for x in a.seeds.split(',') if x.strip()]
    wins = [('熊 %s' % a.bear_start, a.bear_start), ('牛 %s' % a.bull_start, a.bull_start)]

    data, missing = {}, []
    for wname, wstart in wins:
        for arm in arms:
            for sd in seeds:
                arc = a.archive_fmt.format(arm=arm, seed=sd)
                d = find_run(arc, wstart, a.tag)
                if d is None:
                    missing.append((wname, arc))
                    continue
                data[(wname, arm, sd)] = load_run(d)
    if missing:
        print('缺以下回测产物（先跑 run_t134_backtest_yscale.sh）:')
        for w, arc in missing:
            print('  %s  %s' % (w, arc))
        if len(missing) == len(wins) * len(arms) * len(seeds):
            return 2

    res = {}
    for wname, _ in wins:
        print('\n' + '=' * 96)
        print('窗口 %s' % wname)
        print('%-22s %8s %8s %8s %8s %8s %8s %8s %8s' % (
            'archive', 'ret%', 'bench%', 'beta', 'alpha%', 'sharpe', 'mdd', 'capUp', 'capDn'))
        for arm in arms:
            for sd in seeds:
                r = data.get((wname, arm, sd))
                if not r:
                    continue
                arc = a.archive_fmt.format(arm=arm, seed=sd)
                res['%s|y%s|s%s' % (wname, arm, sd)] = r

                def f(k, w=8, p=2):
                    v = r.get(k)
                    return ('%*.*f' % (w, p, v)) if isinstance(v, (int, float)) else '%*s' % (w, '-')
                print('%-22s %s %s %s %s %s %s %s %s' % (
                    arc, f('ret_pct'), f('bench_ret'), f('beta', 8, 3), f('alpha'),
                    f('sharpe'), f('mdd', 8, 3), f('cap_up', 8, 3), f('cap_dn', 8, 3)))

    # ── H1：β 的剂量-反应 ──────────────────────────────────────────────────
    print('\n' + '=' * 96)
    print('★ H1（核心）：β 是否随 y_scale 单调上升')
    h1 = {}
    for wname, _ in wins:
        for sd in seeds:
            vals, got = [], []
            for arm in arms:
                r = data.get((wname, arm, sd))
                if r and isinstance(r.get('beta'), (int, float)):
                    vals.append(r['beta'])
                    got.append(arm)
            if len(vals) < 3:
                continue
            sh = shape_of(vals)
            spread = max(vals) - min(vals)
            h1['%s|s%s' % (wname, sd)] = {'arms': got, 'beta': vals,
                                          'shape': sh, 'spread': spread}
            print('  %-12s s%-3s  ' % (wname, sd)
                  + '  '.join('y%s=%.3f' % (k, v) for k, v in zip(got, vals))
                  + '   形状=%-9s 极差=%.3f' % (sh, spread))
    shapes = {k: v['shape'] for k, v in h1.items()}
    spreads = [v['spread'] for v in h1.values()]
    # ⚠ 「数据不足」必须与「被证伪」严格区分 —— 缺产物时读成"证伪"会得出反向结论。
    n_expect = len(wins) * len(seeds)
    enough = len(h1) == n_expect and len(arms) >= 3
    mono_up = bool(shapes) and all(s == 'mono_up' for s in shapes.values())
    big = bool(spreads) and all(s >= a.beta_spread_line for s in spreads)
    verdict_h1 = 'H1_HOLDS' if (enough and mono_up and big) else (
        'INSUFFICIENT' if not enough else 'H1_FALSIFIED')
    if verdict_h1 == 'INSUFFICIENT':
        print('  ⇒ **数据不足，无法判定**：需要 %d 条 (窗口×种子) β 序列且剂量点 ≥3，'
              '实际 %d 条 / %d 个剂量点。' % (n_expect, len(h1), len(arms)))
    else:
        print('  ⇒ 全部单调增=%s   极差全部 ≥%.2f=%s   ⇒ H1 %s' % (
            mono_up, a.beta_spread_line, big,
            '成立' if verdict_h1 == 'H1_HOLDS' else '**不成立**'))
        if verdict_h1 == 'H1_FALSIFIED':
            print('     形状: %s' % shapes)
            print('     极差: %s' % ['%.3f' % s for s in spreads])

    # ── H2/H3：收益方向（仅方向，MDE 太大不看幅度）────────────────────────
    print('\n★ H2/H3（辅证，MDE≈80pp，**只看方向不看幅度**）')
    h23 = {}
    for wname, _ in wins:
        for sd in seeds:
            vals, got = [], []
            for arm in arms:
                r = data.get((wname, arm, sd))
                if r and isinstance(r.get('ret_pct'), (int, float)):
                    vals.append(r['ret_pct'])
                    got.append(arm)
            if len(vals) < 3:
                continue
            h23['%s|s%s' % (wname, sd)] = {'arms': got, 'ret': vals, 'shape': shape_of(vals)}
            print('  %-12s s%-3s  ' % (wname, sd)
                  + '  '.join('y%s=%+.1f%%' % (k, v) for k, v in zip(got, vals))
                  + '   形状=%s' % shape_of(vals))
    print('  预测: 牛市应单调增(H2)、熊市应单调减(H3)。')

    # ── H4：上下行捕获同向变大 ─────────────────────────────────────────────
    print('\n★ H4：上行/下行捕获是否**同向变大**（暴露变大而非选股变好）')
    for wname, _ in wins:
        for sd in seeds:
            cu, cd, got = [], [], []
            for arm in arms:
                r = data.get((wname, arm, sd))
                if r and isinstance(r.get('cap_up'), (int, float)) \
                        and isinstance(r.get('cap_dn'), (int, float)):
                    cu.append(r['cap_up'])
                    cd.append(r['cap_dn'])
                    got.append(arm)
            if len(cu) < 3:
                continue
            print('  %-12s s%-3s  上行 ' % (wname, sd)
                  + ' '.join('y%s=%.2f' % (k, v) for k, v in zip(got, cu))
                  + '  [%s]' % shape_of(cu) + '   下行 '
                  + ' '.join('y%s=%.2f' % (k, v) for k, v in zip(got, cd))
                  + '  [%s]' % shape_of(cd))

    print('\n' + '=' * 96)
    print('★ 机制裁决')
    if verdict_h1 == 'INSUFFICIENT':
        print('  **无法判定** —— 回测产物不全。这既不是"机制成立"也不是"被证伪"，')
        print('     先把 run_t134_backtest_yscale.sh 跑完再判。')
    elif verdict_h1 == 'H1_HOLDS':
        print('  H1 成立 ⇒ 「y_scale = 静态风险偏好旋钮」这个说法**得到支持**。')
        print('     但注意：这只解释了 T133 的涨跌日不对称，**不改变 y4/y8 不晋级**')
        print('     —— 静态拧大暴露是 β 不是 α（[[bull-window-lead-means-beta-not-alpha]]）。')
    else:
        print('  H1 不成立 ⇒ 「风险偏好旋钮」这个说法**被证伪**。')
        print('     那么 T133 里"涨日头部略好、跌日头部塌"更可能是**有效样本量塌缩**')
        print('     带来的噪声，而不是一个可解释的风险暴露机制。我先前那个解释应作废。')

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({'runs': res, 'h1_beta': h1, 'h23_ret': h23,
               'h1_verdict': verdict_h1, 'beta_spread_line': a.beta_spread_line,
               'n_beta_series': len(h1), 'n_expected': n_expect},
              open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n已保存: %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
