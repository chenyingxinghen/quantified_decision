"""T081 / E11 回测确认判定：多持有期标签 vs 基线，K=40 双窗口同种子配对。

基线零假设：``diagnose_output/random_null_T069_s{seed}_k40_{win}.json``。
熊市为主判（MDE 1.64 z），牛市 MDE 1.13 z。晋级到生产需熊市 Δz 中位 > MDE
且不是 1/4 正向；若熊市不劣化但也不显著，按「IC 已晋级、回测中性」记录并保留。
"""
from __future__ import annotations

import json
import os
import statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SEEDS = [42, 11, 23, 37]
MDE = {'bear': 1.64, 'bull': 1.13}


def _load(tag):
    p = os.path.join(ROOT, 'diagnose_output', f'random_null_{tag}.json')
    if not os.path.isfile(p):
        return None
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    z = d.get('z_score', d.get('z'))
    # diag_random_null 存的是 actual/null_mean 两个原始百分比，没有现成的超额列
    # （老版本这里找 excess_return_pp 永远取不到，Δ超额那一列一直是空的）。
    ex = d.get('excess_return_pp', d.get('excess_pp'))
    if ex is None and d.get('actual_return_pct') is not None \
            and d.get('null_mean_pct') is not None:
        ex = float(d['actual_return_pct']) - float(d['null_mean_pct'])
    return (None if z is None else float(z), None if ex is None else float(ex))


def main():
    for win in ('bear', 'bull'):
        print(f'\n=== {win}（MDE {MDE[win]} z）===')
        print(f'{"seed":>5} {"base_z":>8} {"mh_z":>8} {"Δz":>8} '
              f'{"base_ex":>9} {"mh_ex":>9} {"Δex_pp":>8}')
        dz, dex = [], []
        for s in SEEDS:
            b = _load(f'T069_s{s}_k40_{win}')
            m = _load(f'T081_mh_s{s}_k40_{win}')
            if not b or not m:
                print(f'{s:>5}  缺数据 (base={"有" if b else "无"}, mh={"有" if m else "无"})')
                continue
            dz.append(m[0] - b[0])
            if b[1] is not None and m[1] is not None:
                dex.append(m[1] - b[1])
                print(f'{s:>5} {b[0]:8.3f} {m[0]:8.3f} {m[0] - b[0]:+8.3f} '
                      f'{b[1]:9.2f} {m[1]:9.2f} {m[1] - b[1]:+8.2f}')
            else:
                print(f'{s:>5} {b[0]:8.3f} {m[0]:8.3f} {m[0] - b[0]:+8.3f}')
        if dz:
            pos = sum(1 for v in dz if v > 0)
            med = statistics.median(dz)
            print(f'  Δz: {pos}/{len(dz)} 正向，中位 {med:+.3f}'
                  + (f'  |  Δ超额中位 {statistics.median(dex):+.2f} pp' if dex else ''))
            if len(dz) == 4:
                if med > MDE[win] and pos >= 2:
                    print(f'  → {win} 显著改善')
                elif med >= 0:
                    print(f'  → {win} 不劣化但未超 MDE')
                else:
                    print(f'  → {win} 劣化')


if __name__ == '__main__':
    main()
