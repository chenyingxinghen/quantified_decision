"""T096：读 `rank_ic_holdout`（选型不可见的报告段）横比多个 HPO 臂。

**为什么不用 rank_ic**：那一栏是「选型集上最优 epoch 处的整验证折 IC」，仍带
选型偏差。修好评估器之后，唯一能跨配置横比的是 `rank_ic_holdout`。
两栏都打出来，差值就是该配置的**选型偏差实测值** —— 顺便验证修复是否生效。

判据仍是 4/4 同向；但 holdout 段只有验证折的 40%，MDE 约为原来的 1/√0.4≈1.6 倍，
所以只做粗筛，不追 0.001 量级的差异。

用法：python -u scripts/exp/analyze_holdout.py --prefix T096 --arms base,ys4,eh32,lr1e3
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
SEEDS = (42, 11, 23, 37)
DIRS = (os.path.join(ROOT, 'diagnose_output'),
        os.path.join(ROOT, 'diagnose_output', 'iter_output'))


def _load(prefix, arm, seed):
    for d in DIRS:
        p = os.path.join(d, f'{prefix}_{arm}_s{seed}.json')
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as f:
                return json.load(f)
    return None


def _arm_table(prefix, arm):
    """返回 {seed: (折均 rank_ic, 折均 rank_ic_holdout)}，缺的种子不入表。"""
    out = {}
    for s in SEEDS:
        d = _load(prefix, arm, s)
        if not d:
            continue
        ic, ho = [], []
        for row in d['folds'].values():
            r = row.get('nam_gate') or {}
            if 'rank_ic' in r:
                ic.append(float(r['rank_ic']))
            if r.get('rank_ic_holdout') is not None:
                ho.append(float(r['rank_ic_holdout']))
        if ic:
            out[s] = (float(np.mean(ic)), float(np.mean(ho)) if ho else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='T096')
    ap.add_argument('--arms', required=True, help='逗号分隔，第一个当基线')
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]

    tables = {a: _arm_table(args.prefix, a) for a in arms}
    print(f"{'臂':>8} {'n':>3} {'折均IC(带偏)':>13} {'holdout(无偏)':>14} {'选型偏差':>10} {'σ_seed':>9}")
    for a in arms:
        t = tables[a]
        if not t:
            print(f'{a:>8}   -  （尚无结果）')
            continue
        ic = np.array([v[0] for v in t.values()])
        ho = np.array([v[1] for v in t.values()])
        bias = np.nanmean(ic - ho)
        print(f'{a:>8} {len(t):>3} {ic.mean():>13.5f} {np.nanmean(ho):>14.5f} '
              f'{bias:>+10.5f} {np.nanstd(ho, ddof=1) if len(t) > 1 else np.nan:>9.5f}')

    base = arms[0]
    bt = tables[base]
    if not bt:
        return
    for a in arms[1:]:
        t = tables[a]
        common = sorted(set(t) & set(bt))
        if not common:
            continue
        d = np.array([t[s][1] - bt[s][1] for s in common])
        print(f'\n=== 配对 Δ holdout: {a} − {base} ===')
        for s, v in zip(common, d):
            print(f'  s{s}: {bt[s][1]:.5f} → {t[s][1]:.5f}   Δ {v:+.5f}')
        mde = 2.0 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        print(f'  {int((d > 0).sum())}/{len(d)} 上升，Δ 中位 {np.median(d):+.5f}，'
              f'配对 MDE ≈ {mde:.5f}  → '
              f'{"晋级" if (d > 0).all() and abs(np.median(d)) > mde else "不晋级"}')


if __name__ == '__main__':
    main()
