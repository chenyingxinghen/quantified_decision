"""T100 树 HPO 判决：**主判据 top-20 超额，副判据整截面 Rank IC**。

为什么主副这样排：lambdarank 的 ndcg 有头部截断与特有优化，整截面 IC 会低估它 ——
一个只把前 20 名排对、中后段乱排的模型，IC 平平但正是生产要的。所以横比看
`top20_excess_holdout`；`rank_ic_holdout` 只作副证。**两者背离时以 top20 为准，
但必须打印出来** —— 那正是「头部强但整截面平」的直接证据。

两栏都是**无偏读数**（`--select-holdout 0.4`，报告段对早停不可见）。
带偏栏也一并打出来，差值就是该配置的选型偏差实测值。
注意树的偏差方向可能是负的：lgb 的 ndcg@20 早停在 109~161 棵树，远没到 IC 最优点。

判据：4/4 同向 + Δ 中位过配对 MDE（2σ_paired/√n）。

用法：python -u scripts/archive/t_series/analyze_tree_hpo.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT)
SEEDS = (42, 11, 23, 37)
DIRS = (os.path.join(ROOT, 'diagnose_output'),
        os.path.join(ROOT, 'diagnose_output', 'iter_output'))
GROUPS = {'lig': ('base', 'lr03', 'deep', 'nd40'), 'xgb': ('base', 'lr03', 'deep')}


def _rows(model, arm, seed):
    for d in DIRS:
        p = os.path.join(d, f'T100_{model}_{arm}_s{seed}.json')
        if os.path.isfile(p):
            f = json.load(open(p, encoding='utf-8'))['folds']
            return [v['nam_gate'] for v in f.values() if isinstance(v, dict) and 'nam_gate' in v]
    return None


def _agg(model, arm):
    """{seed: (top20_holdout, ic_holdout, ic_biased, best_iter)}，按折平均。"""
    out = {}
    for s in SEEDS:
        rs = _rows(model, arm, s)
        if not rs:
            continue
        g = lambda k: float(np.mean([r[k] for r in rs if r.get(k) is not None]))
        out[s] = (g('top20_excess_holdout'), g('rank_ic_holdout'),
                  g('rank_ic'), float(np.mean([r['best_iteration'] for r in rs])))
    return out


def _paired(t, b, i):
    common = sorted(set(t) & set(b))
    if len(common) < 2:
        return None
    d = np.array([t[s][i] - b[s][i] for s in common])
    return d, common, 2.0 * d.std(ddof=1) / np.sqrt(len(d))


def main():
    for model, arms in GROUPS.items():
        tabs = {a: _agg(model, a) for a in arms}
        if not any(tabs.values()):
            print(f'\n### {model}：尚无结果')
            continue
        print(f"\n{'='*78}\n### {model}\n{'='*78}")
        print(f"{'臂':>6} {'n':>3} {'top20超额(主)':>14} {'IC_holdout(副)':>15} "
              f"{'IC带偏':>9} {'选型偏差':>10} {'best_iter':>10}")
        for a in arms:
            t = tabs[a]
            if not t:
                print(f'{a:>6}   -  （尚无结果）')
                continue
            arr = np.array([v for v in t.values()])
            print(f'{a:>6} {len(t):>3} {arr[:,0].mean():>14.5f} {arr[:,1].mean():>15.5f} '
                  f'{arr[:,2].mean():>9.5f} {arr[:,2].mean()-arr[:,1].mean():>+10.5f} '
                  f'{arr[:,3].mean():>10.0f}')

        base = tabs['base']
        if not base:
            continue
        for a in arms[1:]:
            if not tabs[a]:
                continue
            print(f'\n--- {a} − base ---')
            verdict = {}
            for i, (name, w) in enumerate(((f'top20超额(主)', 0), ('IC_holdout(副)', 1))):
                r = _paired(tabs[a], base, w)
                if r is None:
                    continue
                d, common, mde = r
                ok = bool((d > 0).all() or (d < 0).all()) and abs(np.median(d)) > mde
                verdict[name] = (np.median(d), ok, int((d > 0).sum()), len(d))
                print(f'  {name:>16}: {int((d>0).sum())}/{len(d)} 上升，'
                      f'Δ 中位 {np.median(d):+.6f}，MDE {mde:.6f} → '
                      f'{"过门" if ok else "不过门"}')
            if len(verdict) == 2:
                (m_d, m_ok, _, _), (i_d, i_ok, _, _) = verdict.values()
                if np.sign(m_d) != np.sign(i_d):
                    print(f'  ⚠ **主副背离**：top20 {m_d:+.6f} 而 IC {i_d:+.6f} —— '
                          f'这正是「头部能力与整截面 IC 不是一回事」的证据，'
                          f'按预注册以 top20 为准。')
                print(f'  → **{"晋级" if m_ok and m_d > 0 else "不晋级"}**')


if __name__ == '__main__':
    main()
