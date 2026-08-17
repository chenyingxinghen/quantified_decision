"""E4 判定：持仓档位 K ∈ {20, 40, 60}。

判定口径
--------
K 不涉及重训练，同一 seed 下 K=20/40/60 用的是同一条信号，
因此**同种子内是严格配对**，种子彩票被差分掉。但「K 更大更好」
能否跨种子成立必须验证，所以在 σ_seed 种子池的 4 个种子上各扫一遍。

三口径并列（分位在熊市已触顶，主效应量用 z）：见 scripts/exp/null_metrics.py。

**本实验最重要的产出不是收益，而是跨种子离散度。**
E1b 测出 σ_seed（熊市 z 口径 1.264）里有很大一块来自「只持 20 只」的
组合特异性风险。若 K 增大能把跨种子 std 压下去，那 K 的价值是
**把评估器从 variance-limited 里拉出来** —— 按 analyze_power.py，
σ 降一半，同样样本量下 MDE 也降一半，等于所有后续实验的信噪比翻倍。

用法
----
  python scripts/archive/e_series/analyze_k_sweep.py --out diagnose_output/k_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'exp'))

from null_metrics import METRICS, METRIC_LABEL, SIGMA_SEED, read_null  # noqa: E402

DIAG = os.path.join(ROOT, 'diagnose_output')
SEEDS = [42, 11, 23, 37]
KS = [40]  # 2026-08-12 缩减：K=60 在 s42 上已被 K=40 支配，不再扫（见台账 E4 节）

# K -> (实验前缀)。K=10 是 T072（E4b 反方向），K=40/60 是 T069。
K_PREFIX = {10: 'T072', 40: 'T069', 60: 'T069'}

K20 = {
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(DIAG, 'k_sweep.json'))
    ap.add_argument('--ks', default=None,
                    help='逗号分隔的 K 列表，覆盖默认（例：--ks 10 判定 E4b）')
    args = ap.parse_args()
    global KS
    if args.ks:
        KS = [int(x) for x in args.ks.split(',') if x.strip()]

    result = {'seeds': SEEDS, 'ks': KS, 'sigma_seed': SIGMA_SEED, 'windows': {}}

    for win in ('bull', 'bear'):
        title = '牛市 2024-08→2026-08（仅稳健性参考）' if win == 'bull' \
            else '熊市 2022-09→2024-08（主判窗口）'
        print(f'\n=== {title} ===')

        data = {}
        for s in SEEDS:
            data[s] = {20: read_null(os.path.join(DIAG, K20[win][s]))}
            for k in KS:
                data[s][k] = read_null(
                    os.path.join(DIAG,
                                 f'random_null_{K_PREFIX.get(k, "T069")}_s{s}_k{k}_{win}.json'))

        hdr = f'{"seed":>5} |'
        for k in [20] + KS:
            hdr += f' {"K"+str(k)+"分位":>8} {"K"+str(k)+"_z":>7} {"K"+str(k)+"收益":>8} |'
        print(hdr)
        for s in SEEDS:
            line = f'{s:>5} |'
            for k in [20] + KS:
                v = data[s][k]
                if v is None:
                    line += f' {"--":>8} {"--":>7} {"--":>8} |'
                else:
                    line += f' {v["pct"]:>8.1f} {v["z"]:>7.3f} {v["ret"]:>8.2f} |'
            print(line)

        wnode = {'rows': {str(s): {str(k): data[s][k] for k in [20] + KS} for s in SEEDS},
                 'paired': {}, 'dispersion': {}}

        # --- 配对差 (K) - (K=20) ---
        for k in KS:
            wnode['paired'][str(k)] = {}
            msgs = []
            for m in METRICS:
                diffs = [data[s][k][m] - data[s][20][m]
                         for s in SEEDS if data[s][k] and data[s][20]]
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
                e['sigma_seed'] = SIGMA_SEED[win][m]
                e['over_sigma'] = float(a.mean() / SIGMA_SEED[win][m])
                wnode['paired'][str(k)][m] = e
                t = f' t={e["t"]:+.2f}' if '0t' not in e and e.get('t') is not None else ''
                msgs.append(f'{METRIC_LABEL[m]} Δ={e["mean"]:+.3f}(中位{e["median"]:+.3f},'
                            f'正向{e["n_positive"]}/{e["n"]}{t})')
            if msgs:
                print(f'  K={k} vs K=20 配对：' + ' | '.join(msgs))

        # --- 跨种子离散度：K 能否压掉噪声地板 ---
        for k in [20] + KS:
            vals = {m: [data[s][k][m] for s in SEEDS if data[s][k]] for m in METRICS}
            rets = [data[s][k]['ret'] for s in SEEDS if data[s][k]]
            if len(rets) > 1:
                node = {'n': len(rets),
                        'ret_median': float(np.median(rets)),
                        'ret_std': float(np.std(rets, ddof=1))}
                for m in METRICS:
                    node[f'{m}_median'] = float(np.median(vals[m]))
                    node[f'{m}_std'] = float(np.std(vals[m], ddof=1))
                wnode['dispersion'][str(k)] = node
        if len(wnode['dispersion']) > 1:
            print('  --- 跨种子离散度（K 能否压掉 σ_seed 噪声地板）---')
            base = wnode['dispersion'].get('20')
            for k, v in wnode['dispersion'].items():
                shrink = ''
                if base and k != '20' and base['z_std'] > 0:
                    r = v['z_std'] / base['z_std']
                    mde = 2.80 * v['z_std'] * np.sqrt(2 / 4)
                    shrink = (f'  → z_std 为 K20 的 {r*100:.0f}%'
                              f'；非配对 n=4 的 MDE {mde:.2f} z')
                print(f'    K={k:>2} n={v["n"]}  z中位={v["z_median"]:+.3f} '
                      f'z_std={v["z_std"]:.3f}  收益中位={v["ret_median"]:+.2f}% '
                      f'收益std={v["ret_std"]:.2f}pp{shrink}')

        result['windows'][win] = wnode

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {args.out}')


if __name__ == '__main__':
    main()
