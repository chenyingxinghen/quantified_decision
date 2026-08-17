"""E3 判定：标签残差化 (beta_size) 是否真的有增益。

判定口径（严格遵守 E1b + 功效分析的结论）
------------------------------------------
1. **不看 Rank IC**。`_prepare_fold` 对 y_train 与 y_val 同时残差化，
   E3 的 val rank_ic(≈0.071) 与 T045 的 0.0968 是两个不同任务的难度，
   数值不可比，IC 下降不构成否决理由。
2. **配对比较**：同一 seed 的 beta_size 与 none 做差，n=4 对。
   种子彩票被差分掉，剩下的才是处理效应。非配对设计在本评估器上
   即使 n=10 也测不动（见 scripts/archive/e_series/analyze_power.py）。
3. **三口径并列**：分位在熊市已触顶（10 种子中位 97.2，离上界只剩 2.8pp），
   只能测劣化测不出改进，所以主效应量改用 **z 值**（无上界），
   分位仅作历史可比，超额 pp 作经济口径。
4. **熊市为主判**（E1b：熊市 z 中位 1.951；牛市 z 中位仅 0.631，本身就在随机附近）。

用法
----
  python scripts/archive/e_series/analyze_e3.py --out diagnose_output/e3_beta_size.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'exp'))

from null_metrics import METRICS, METRIC_LABEL, SIGMA_SEED, read_null  # noqa: E402

DIAG = os.path.join(ROOT, 'diagnose_output')
SEEDS = [42, 11, 23, 37]

# 同种子 none 基线（E1b 已产出）
NONE_BASE = {
    'bull': {
        42: 'random_null_T045.json',
        11: 'random_null_T068_s11_bull.json',
        23: 'random_null_T068_s23_bull.json',
        37: 'random_null_T068_s37_bull.json',
    },
    'bear': {
        42: 'random_null_final_s42_bear.json',
        11: 'random_null_T068_s11_bear.json',
        23: 'random_null_T068_s23_bear.json',
        37: 'random_null_T068_s37_bear.json',
    },
}
E3_FILE = {w: {s: f'random_null_T070_e3_s{s}_{w}.json' for s in SEEDS}
           for w in ('bull', 'bear')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(DIAG, 'e3_beta_size.json'))
    args = ap.parse_args()

    result = {'seeds': SEEDS, 'sigma_seed': SIGMA_SEED, 'windows': {}}

    for win in ('bull', 'bear'):
        title = '牛市 2024-08→2026-08（仅稳健性参考）' if win == 'bull' \
            else '熊市 2022-09→2024-08（主判窗口）'
        print(f'\n=== {title} ===')

        rows = {}
        for s in SEEDS:
            rows[s] = {
                'e3': read_null(os.path.join(DIAG, E3_FILE[win][s])),
                'none': read_null(os.path.join(DIAG, NONE_BASE[win][s])),
            }

        print(f'{"seed":>5} | {"none分位":>8} {"E3分位":>7} {"Δ":>7} '
              f'| {"none_z":>7} {"E3_z":>6} {"Δz":>7} '
              f'| {"none超额":>8} {"E3超额":>7} {"Δ超额":>8} '
              f'| {"none收益":>8} {"E3收益":>8}')
        for s in SEEDS:
            a, b = rows[s]['e3'], rows[s]['none']
            def c(v, w, nd=1):
                return f'{v:>{w}.{nd}f}' if v is not None else f'{"--":>{w}}'
            def d(k):
                return (a[k] - b[k]) if (a and b) else None
            print(f'{s:>5} | {c(b["pct"] if b else None,8)} {c(a["pct"] if a else None,7)} '
                  f'{c(d("pct"),7)} | {c(b["z"] if b else None,7,3)} {c(a["z"] if a else None,6,3)} '
                  f'{c(d("z"),7,3)} | {c(b["excess"] if b else None,8,2)} '
                  f'{c(a["excess"] if a else None,7,2)} {c(d("excess"),8,2)} '
                  f'| {c(b["ret"] if b else None,8,2)} {c(a["ret"] if a else None,8,2)}')

        wnode = {'rows': {str(s): rows[s] for s in SEEDS}, 'paired': {}}
        for m in METRICS:
            diffs = [rows[s]['e3'][m] - rows[s]['none'][m]
                     for s in SEEDS if rows[s]['e3'] and rows[s]['none']]
            if not diffs:
                continue
            a = np.array(diffs, dtype=float)
            e = {'n': int(a.size), 'mean': float(a.mean()),
                 'median': float(np.median(a)), 'n_positive': int((a > 0).sum())}
            if a.size > 1:
                se = a.std(ddof=1) / np.sqrt(a.size)
                e['sd'] = float(a.std(ddof=1))
                e['se'] = float(se)
                e['t'] = float(a.mean() / se) if se > 0 else None
                # 配对相关：两臂同种子结果的相关，决定配对设计能省多少样本
                xa = np.array([rows[s]['e3'][m] for s in SEEDS if rows[s]['e3'] and rows[s]['none']])
                xb = np.array([rows[s]['none'][m] for s in SEEDS if rows[s]['e3'] and rows[s]['none']])
                if xa.size > 2 and xa.std() > 0 and xb.std() > 0:
                    e['rho_pair'] = float(np.corrcoef(xa, xb)[0, 1])
            sd_seed = SIGMA_SEED[win][m]
            e['sigma_seed'] = sd_seed
            e['over_sigma'] = float(a.mean() / sd_seed)
            wnode['paired'][m] = e

            msg = (f'  [{METRIC_LABEL[m]}] Δ均值={e["mean"]:+.3f} 中位={e["median"]:+.3f} '
                   f'正向 {e["n_positive"]}/{e["n"]}')
            if 'se' in e:
                msg += f' 配对SD={e["sd"]:.3f} SE={e["se"]:.3f} t={e["t"]:+.2f}'
            msg += f' | σ_seed={sd_seed:.3f} Δ/σ_seed={e["over_sigma"]:+.2f}'
            if 'rho_pair' in e:
                msg += f' | ρ配对={e["rho_pair"]:+.2f}'
            print(msg)

        # 判定（只在熊市下结论）
        if win == 'bear' and 'z' in wnode['paired']:
            e = wnode['paired']['z']
            if e.get('se'):
                if abs(e['t']) < 2.0:
                    verdict = (f'未测出（|t|={abs(e["t"]):.2f} < 2）。'
                               f'以配对SD={e["sd"]:.3f} 计，n=4 的 MDE≈{2.80*e["sd"]/2:.2f} z，'
                               f'当前效应 {e["mean"]:+.3f} z 落在噪声内。'
                               f'不得写"有效"，也不得写"无效"。')
                elif e['mean'] > 0:
                    verdict = f'有增益（Δz={e["mean"]:+.3f}, t={e["t"]:+.2f}）'
                else:
                    verdict = f'有劣化（Δz={e["mean"]:+.3f}, t={e["t"]:+.2f}）'
                print(f'\n  >>> 熊市判定：{verdict}')
                wnode['verdict'] = verdict

        result['windows'][win] = wnode

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {args.out}')


if __name__ == '__main__':
    main()
