"""T102 判决：NAM 加交互结构（门控 / 显式交互列）能否追上树。

单种子，所以**没有配对功率** —— 只看方向与量级，判据写在 run_t102 脚本头：
Δ ≥ +0.005（≈ 树在 800 只上领先量的 2/3）才值得扩种子；|Δ| < 0.002 直接关轴。

同时打印门控健康度：熵塌陷（gate_H → 0、max_share → 1）说明门控退化成
「只用一个因子族」，那不是学到 regime 路由，判定时必须把这种臂标出来。

用法：python -u scripts/archive/t_series/analyze_t102.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT)
FOLDS = ('60-80%', '70-90%', '80-100%')
ARMS = (('base', 'T096_base_s42'), ('gate', 'T102_gate_s42'),
        ('xint', 'T102_xint_s42'), ('both', 'T102_both_s42'))


def _load(stem):
    for d in (os.path.join(ROOT, 'diagnose_output'),
              os.path.join(ROOT, 'diagnose_output', 'iter_output')):
        p = os.path.join(d, f'{stem}.json')
        if os.path.isfile(p):
            f = json.load(open(p, encoding='utf-8'))['folds']
            return {k: v['nam_gate'] for k, v in f.items()
                    if isinstance(v, dict) and 'nam_gate' in v}
    return None


def main():
    tabs = {name: _load(stem) for name, stem in ARMS}
    base = tabs['base']
    if not base:
        print('缺 T096_base_s42 基线')
        return

    print('单种子 42 / 800 只 / 同折 / 无偏 holdout（select-holdout 0.4）\n')
    print(f"{'臂':>6} {'折数':>4} {'holdout 折均':>13} {'Δ vs base':>11} "
          f"{'跌日 IC':>9} {'门控熵':>8} {'最大族占比':>10}")
    for name, _ in ARMS:
        t = tabs[name]
        if not t:
            print(f'{name:>6}   -  （尚无结果）')
            continue
        common = [f for f in FOLDS if f in t and f in base]
        ho = np.mean([t[f]['rank_ic_holdout'] for f in common])
        dn = np.mean([t[f]['regime']['down_days']['rank_ic'] for f in common])
        bh = np.mean([base[f]['rank_ic_holdout'] for f in common])
        ent = np.mean([t[f].get('gate_entropy', np.nan) for f in common])
        sh = np.mean([t[f].get('gate_max_share', np.nan) for f in common])
        d = '' if name == 'base' else f'{ho - bh:+11.5f}'
        print(f'{name:>6} {len(common):>4} {ho:>13.5f} {d:>11} {dn:>9.5f} '
              f'{ent:>8.3f} {sh:>10.3f}')

    print(f"\n门控熵上界 ln(11) = {np.log(11):.3f}；均匀分布时 max_share = 1/11 = 0.091")
    print('熵 → 0 且 max_share → 1 = 门控塌陷（退化成只用一个因子族），'
          '不是学到 regime 路由。\n')

    for name, _ in ARMS[1:]:
        t = tabs[name]
        if not t:
            continue
        common = [f for f in FOLDS if f in t and f in base]
        d = np.array([t[f]['rank_ic_holdout'] - base[f]['rank_ic_holdout'] for f in common])
        med = float(np.median(d))
        verdict = ('值得扩种子' if med >= 0.005 else
                   '关轴' if abs(med) < 0.002 else '灰区：方向可疑，量级不足')
        print(f'  {name} − base: 逐折 ' + ' '.join(f'{x:+.5f}' for x in d) +
              f'  中位 {med:+.5f} → **{verdict}**')


if __name__ == '__main__':
    main()
