"""T073 / E6 判定：标量条件化门控（11 个门控参数）是否有增益。

口径
----
1. **验证 IC 这次是可比的**：与基线唯一差别是门控开关，标签完全相同（不像 E3）。
2. **主判仍是回测**：同种子配对，K=40 筛选口径（跨种子收益 std 只有 K=20 的 1/2.7），
   基线为 T069_s{seed}_k40_{win}，熊市为主判，熊市 MDE 1.64 z。
3. 门禁：熊市 Δz 中位 > +1.64 且正向 ≥3/4 才算通过、才值得回 K=20 确认。

用法: python scripts/archive/e_series/analyze_e6.py
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'exp'))

from null_metrics import read_null  # noqa: E402

DIAG = os.path.join(ROOT, 'diagnose_output')

def _diag(name):
    """读取路径：跑完的轮次会被归档进 iter_output/，两处都找（与 exp/ 的在用工具同口径）。"""
    for d in (DIAG, os.path.join(DIAG, 'iter_output')):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return os.path.join(DIAG, name)

SEEDS = [42, 11, 23, 37]
BASE_IC = {42: 0.09685, 11: 0.08868, 23: 0.09384, 37: 0.09225}  # T045 / T068_seed*
MDE = {'bear': 1.64, 'bull': 1.13}


def main():
    print('=== 验证 Rank IC（同标签，可直接比）===')
    ic_d = []
    for s in SEEDS:
        p = _diag(f'T073_e6_scalar_s{s}.json')
        d = json.load(open(p, encoding='utf-8'))
        n = list(d['folds'].values())[0]['nam_gate']
        ic = n['rank_ic']
        ic_d.append(ic - BASE_IC[s])
        print(f'  s{s}: 基线 {BASE_IC[s]:.5f} -> scalar {ic:.5f}  '
              f'Δ={ic - BASE_IC[s]:+.5f}  gate_max_share={n["gate_max_share"]:.3f}')
    print(f'  Δ中位={st.median(ic_d):+.5f}  正向 {sum(1 for x in ic_d if x > 0)}/{len(ic_d)}')

    out = {'ic_delta': ic_d, 'windows': {}}
    for win in ('bear', 'bull'):
        tag = '熊市（主判）' if win == 'bear' else '牛市（参考）'
        print(f'\n=== {tag} K=40 同种子配对 ===')
        dz, dex, dret = [], [], []
        for s in SEEDS:
            a = read_null(_diag(f'random_null_T073_e6_s{s}_k40_{win}.json'))
            b = read_null(_diag(f'random_null_T069_s{s}_k40_{win}.json'))
            if not a or not b:
                print(f'  s{s}: 数据缺失，跳过')
                continue
            dz.append(a['z'] - b['z'])
            dex.append(a['excess'] - b['excess'])
            dret.append(a['ret'] - b['ret'])
            print(f'  s{s}: z {b["z"]:+.3f} -> {a["z"]:+.3f} (Δ{dz[-1]:+.3f}) | '
                  f'超额 {b["excess"]:+.2f} -> {a["excess"]:+.2f} (Δ{dex[-1]:+.2f}) | '
                  f'收益 {b["ret"]:+.2f} -> {a["ret"]:+.2f}')
        if not dz:
            continue
        pos = sum(1 for x in dz if x > 0)
        print(f'  Δz 中位={st.median(dz):+.3f} 均值={st.mean(dz):+.3f} 正向 {pos}/{len(dz)}'
              f' | Δ超额 中位={st.median(dex):+.2f}pp | Δ收益 中位={st.median(dret):+.2f}pp')
        verdict = '通过' if (st.median(dz) > MDE[win] and pos >= 3) else '未通过'
        print(f'  门禁（Δz中位 > {MDE[win]} 且正向≥3/4）: **{verdict}**')
        out['windows'][win] = {'dz': dz, 'dex': dex, 'dret': dret,
                               'dz_median': st.median(dz), 'pos': pos, 'verdict': verdict}

    p = os.path.join(DIAG, 'e6_scalar.json')
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n已保存: {p}')


if __name__ == '__main__':
    main()
