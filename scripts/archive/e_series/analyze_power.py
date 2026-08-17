"""评估器功效分析：用 E1b 测出的 σ_seed 反推「多少种子才测得动多大效应」。

为什么必须先算这个
------------------
E1b 得到 σ_seed（牛 27.7pp / 熊 16.3pp，随机分位口径）。任何"配置 A 比 B 好"
的结论，其可检出的最小效应量由 σ_seed 和样本数决定。过去 T053–T067 全部是
**每臂 n=1**，MDE 远大于任何真实效应 —— 所以那些"晋级/否决"结论在统计上
根本不成立，既不能说好也不能说坏。

口径
----
双侧 α=0.05、power=0.8 → 常数 ≈ 2.80（(z_{0.975}+z_{0.8}) = 1.96+0.84）。

- 非配对（两臂独立换种子）：MDE = 2.80 · σ_seed · √(2/n)
- 配对（同种子、只换被测处理）：MDE = 2.80 · σ_diff / √n
  其中 σ_diff = σ_seed · √(2(1−ρ))，ρ 是两臂同种子结果的相关。
  ρ 越高配对越省样本；E3 / E4 会给出第一个 ρ 的实测值。

用法
----
  python scripts/archive/e_series/analyze_power.py
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DIAG = os.path.join(ROOT, 'diagnose_output')

def _diag(name):
    """读取路径：跑完的轮次会被归档进 iter_output/，两处都找（与 exp/ 的在用工具同口径）。"""
    for d in (DIAG, os.path.join(DIAG, 'iter_output')):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return os.path.join(DIAG, name)

Z = 2.80  # z_{0.975} + z_{0.80}


def main():
    sj = json.load(open(_diag('sigma_seed_full.json'), 'r', encoding='utf-8'))
    out = {}
    for win, cn in (('bull', '牛市'), ('bear', '熊市（主判）')):
        ss = sj[win]['sigma_seed']
        sd = float(ss['std'])
        med = float(ss['median'])
        print(f'\n=== {cn}  σ_seed={sd:.2f}pp  中位={med:.1f}分位  n={ss["n"]} ===')
        print('  最小可检出效应 MDE（随机分位 pp，α=.05 双侧，power=.8）')
        print(f'  {"n/臂":>5} {"非配对":>9} {"配对ρ=.5":>10} {"配对ρ=.8":>10} {"配对ρ=.95":>10}')
        rows = {}
        for n in (1, 2, 4, 6, 10, 25):
            unp = Z * sd * np.sqrt(2.0 / n)
            r = {'unpaired': float(unp)}
            cells = f'  {n:>5} {unp:>9.1f}'
            for rho in (0.5, 0.8, 0.95):
                sdiff = sd * np.sqrt(2 * (1 - rho))
                mde = Z * sdiff / np.sqrt(n)
                r[f'paired_rho{rho}'] = float(mde)
                cells += f' {mde:>10.1f}'
            rows[n] = r
            print(cells)
        out[win] = {'sigma_seed': sd, 'median': med, 'mde': rows}

        # 参照：分位是 0~100 的有界量，中位到上界只有这么多空间
        head = 100.0 - med
        print(f'  参照：中位 {med:.1f} 到上界 100 只剩 {head:.1f}pp 空间；'
              f'非配对 n=10 的 MDE={Z*sd*np.sqrt(0.2):.1f}pp'
              + ('  → **该窗口非配对设计几乎不可能测出任何东西**' if Z*sd*np.sqrt(0.2) > head else ''))

    print('\n结论：')
    print('  1. 每臂 n=1 的 MDE 大到没有意义 —— T053~T067 的"晋级/否决"在统计上全部无效。')
    print('  2. 牛市窗口即使 n=10 非配对也测不动，只能作稳健性参考，不得作判据。')
    print('  3. 唯一可行路线是**配对设计**（同种子只换处理）+ 熊市主判；')
    print('     若两臂相关 ρ≥0.8，熊市 n=4 就能测出约 15pp 的效应。')
    print('  4. 降低 σ_seed 本身（如加大持仓 K）与提高效应量同等重要，甚至更重要。')

    p = os.path.join(DIAG, 'evaluator_power.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {p}')


if __name__ == '__main__':
    main()
