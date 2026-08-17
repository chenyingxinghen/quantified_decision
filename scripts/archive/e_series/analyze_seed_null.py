"""E1b/E2：种子噪声地板 σ_seed 与「用正确基线重判」。

问题
----
台账此前把 T045(seed42) 的随机分位（牛 94.5% / 熊 98.9%）当作晋级门槛。
但 T047 已经测出四种子随机分位中位只有 76.4%，seed42 是高分位离群抽样，
且种子间同日新买 Jaccard 仅 1.5%~6.8%（几乎是不同策略）。于是过去
「某配置分位 < 94.5% → 否决」这一判断，主要反映的是 seed42 的幸运，
而非配置本身的性质。

本脚本做两件事
--------------
1. 汇总一组**同配置不同种子**的随机分位，给出零效应分布：
   中位、均值、std（= σ_seed）、min/max、以及经验分位带。
   这就是评估器的测量误差地板。
2. 用这个分布重判任意候选配置 / 集成：报告
   - 候选分位 vs 种子中位的差值
   - 候选分位在种子零效应分布中的经验位置
   - 该差值是否超出 σ_seed

用法
----
  python scripts/archive/e_series/analyze_seed_null.py --out diagnose_output/sigma_seed.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DIAG = os.path.join(ROOT, 'diagnose_output')

# 同配置种子池（架构/超参/数据完全一致，仅训练种子不同）
# 牛市窗口 2024-08-05~2026-08-05；熊市窗口 2022-09-05~2024-08-05
SEED_POOL = {
    'bull': {
        42: 'random_null_T045.json',
        7: 'random_null_T047_s7.json',
        123: 'random_null_T047_s123.json',
        2024: 'random_null_T047_s2024.json',
        11: 'random_null_T068_s11_bull.json',
        23: 'random_null_T068_s23_bull.json',
        37: 'random_null_T068_s37_bull.json',
        53: 'random_null_T068_s53_bull.json',
        67: 'random_null_T068_s67_bull.json',
        89: 'random_null_T068_s89_bull.json',
    },
    'bear': {
        42: 'random_null_final_s42_bear.json',
        7: 'random_null_T047_s7_bear.json',
        123: 'random_null_T047_s123_bear.json',
        2024: 'random_null_T047_s2024_bear.json',
        11: 'random_null_T068_s11_bear.json',
        23: 'random_null_T068_s23_bear.json',
        37: 'random_null_T068_s37_bear.json',
        53: 'random_null_T068_s53_bear.json',
        67: 'random_null_T068_s67_bear.json',
        89: 'random_null_T068_s89_bear.json',
    },
}

# 待重判的候选（非同配置，是被否决/待判的实验）
CANDIDATES = {
    'bull': {
        'T051_ens4': 'random_null_T051_ens4_bull.json',
        'T068_ens7': 'random_null_T068_ens7_bull.json',
        'T053_downside_v2': 'random_null_t053_bull.json',
        'T056_top20sel': 'random_null_T056_bull.json',
        'T058_hidden8': 'random_null_T058_bull.json',
        'T059_hidden32': 'random_null_T059_bull.json',
        'T060_yscale3': 'random_null_T060_bull.json',
        'T061_yscale15': 'random_null_T061_bull.json',
        'T062_lr1e3': 'random_null_T062_bull.json',
        'T063_lr4e3': 'random_null_T063_bull.json',
        'T064_wd1e4': 'random_null_T064_bull.json',
        'T065_wd0': 'random_null_T065_bull.json',
        'T066_accum2': 'random_null_T066_bull.json',
        'T067_accum8': 'random_null_T067_bull.json',
    },
    'bear': {
        'T068_ens7': 'random_null_T068_ens7_bear.json',
        'T053_downside_v2': 'random_null_t053_bear.json',
        'T056_top20sel': 'random_null_T056_bear.json',
        'T058_hidden8': 'random_null_T058_bear.json',
        'T059_hidden32': 'random_null_T059_bear.json',
        'T060_yscale3': 'random_null_T060_bear.json',
        'T061_yscale15': 'random_null_T061_bear.json',
        'T062_lr1e3': 'random_null_T062_bear.json',
        'T063_lr4e3': 'random_null_T063_bear.json',
        'T064_wd1e4': 'random_null_T064_bear.json',
        'T065_wd0': 'random_null_T065_bear.json',
        'T066_accum2': 'random_null_T066_bear.json',
        'T067_accum8': 'random_null_T067_bear.json',
    },
}


def _read(fname):
    p = os.path.join(DIAG, fname)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    return {
        'pct': float(d['percentile_of_actual']),
        'ret': float(d['actual_return_pct']),
        'p': float(d['p_value_one_sided']),
        'file': fname,
    }


def collect(window):
    got, miss = {}, []
    for seed, fname in SEED_POOL[window].items():
        r = _read(fname)
        if r is None:
            miss.append((seed, fname))
        else:
            got[seed] = r
    return got, miss


def describe(pcts):
    a = np.asarray(sorted(pcts), dtype=float)
    return {
        'n': int(len(a)),
        'median': float(np.median(a)),
        'mean': float(a.mean()),
        'std': float(a.std(ddof=1)) if len(a) > 1 else float('nan'),
        'min': float(a.min()), 'max': float(a.max()),
        'p10': float(np.percentile(a, 10)), 'p90': float(np.percentile(a, 90)),
        'iqr': float(np.percentile(a, 75) - np.percentile(a, 25)),
        'values': a.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='diagnose_output/sigma_seed.json')
    ap.add_argument('--min-seeds', type=int, default=4,
                    help='少于该数量的种子不给出 σ_seed 判定')
    args = ap.parse_args()

    payload = {}
    for window in ('bull', 'bear'):
        print('\n' + '=' * 78)
        print(f'【{window.upper()}】同配置种子零效应分布')
        print('=' * 78)
        got, miss = collect(window)
        if miss:
            print(f'  缺失 {len(miss)} 个: ' + ', '.join(f's{s}' for s, _ in miss))
        if not got:
            print('  无可用样本，跳过')
            continue

        print(f'{"seed":>6} {"分位%":>9} {"收益%":>10} {"p":>8}')
        for s in sorted(got, key=lambda k: got[k]['pct']):
            r = got[s]
            print(f'{s:>6} {r["pct"]:>9.1f} {r["ret"]:>10.2f} {r["p"]:>8.4f}')

        st = describe([r['pct'] for r in got.values()])
        print('-' * 78)
        print(f'  n={st["n"]}  中位={st["median"]:.1f}%  均值={st["mean"]:.1f}%  '
              f'σ_seed={st["std"]:.1f}pp')
        print(f'  范围=[{st["min"]:.1f}%, {st["max"]:.1f}%]  '
              f'IQR={st["iqr"]:.1f}pp  P10-P90=[{st["p10"]:.1f}%, {st["p90"]:.1f}%]')

        ok = st['n'] >= args.min_seeds and np.isfinite(st['std'])
        if ok:
            print(f'  ▎可测量下限：单变量轴的效应量若 < σ_seed = {st["std"]:.1f}pp，不予预注册。')

        payload[window] = {'seeds': {str(k): v for k, v in got.items()},
                           'sigma_seed': st, 'missing': [f's{s}' for s, _ in miss]}

        # ---- 用正确基线重判候选 ----
        print('-' * 78)
        print(f'  候选重判（对照：{st["n"]} 种子中位 {st["median"]:.1f}%，'
              f'而非 seed42）')
        print(f'  {"配置":<20} {"分位%":>8} {"Δ中位":>9} {"|Δ|/σ":>8}  判定')
        rows = {}
        for name, fname in CANDIDATES[window].items():
            r = _read(fname)
            if r is None:
                continue
            delta = r['pct'] - st['median']
            z = abs(delta) / st['std'] if ok and st['std'] > 0 else float('nan')
            if not ok:
                verdict = '样本不足'
            elif z < 1.0:
                verdict = '★ 不可测量（落在种子噪声内）'
            elif delta > 0:
                verdict = '优于种子中位'
            else:
                verdict = '劣于种子中位'
            print(f'  {name:<20} {r["pct"]:>8.1f} {delta:>+9.1f} {z:>8.2f}  {verdict}')
            rows[name] = {'pct': r['pct'], 'delta_vs_median': delta,
                          'z_vs_sigma_seed': z, 'verdict': verdict}
        payload[window]['candidates'] = rows

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {args.out}')


if __name__ == '__main__':
    main()
