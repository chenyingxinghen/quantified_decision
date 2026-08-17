"""T076 / E7 判定：行业内相对分位是否真的让模型**排得更准**。

只看验证 Rank IC（同标签、同折 80-100%、同 target='returns'，与基线严格可比）。
零回测。判定规则（预注册）：
  - 晋级条件：4/4 种子 ΔRankIC > 0 且中位 Δ ≥ +0.005（≈基线 IC 的 5%，
    是跨种子 σ_seed≈0.0035 的 1.4 倍以上，见台账 E4 离散度表）。
  - 只要 ≤2/4 为正，直接按「无效」关闭，不进回测。
"""
from __future__ import annotations

import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BASE_IC = {42: 0.09685, 11: 0.08868, 23: 0.09384, 37: 0.09225}
FOLD = '80-100%'
KEY = 'best_val_rank_ic_on_label'


def _ic(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    row = d['folds'][FOLD]['nam_gate']
    return float(row.get(KEY, row['rank_ic'])), float(row['rank_ic_ir']), int(row['features_used'])


def main():
    rows, deltas = [], []
    for seed, base in BASE_IC.items():
        p = os.path.join(ROOT, 'diagnose_output', f'T076_indrel_s{seed}.json')
        if not os.path.isfile(p):
            print(f'  seed={seed} 尚无结果 ({os.path.basename(p)})')
            continue
        ic, ir, nfeat = _ic(p)
        rows.append((seed, base, ic, ic - base, ir, nfeat))
        deltas.append(ic - base)

    print(f'\n{"seed":>5} {"base_IC":>9} {"E7_IC":>9} {"ΔIC":>9} {"IC_IR":>7} {"n_feat":>7}')
    for seed, base, ic, d, ir, nfeat in rows:
        print(f'{seed:>5} {base:9.5f} {ic:9.5f} {d:+9.5f} {ir:7.3f} {nfeat:>7}')

    if len(deltas) < 2:
        print('\n样本不足，等训练完成。')
        return
    pos = sum(1 for d in deltas if d > 0)
    med = statistics.median(deltas)
    print(f'\n  {pos}/{len(deltas)} 种子 IC 上升，Δ 中位 {med:+.5f}')
    if len(deltas) == 4 and pos == 4 and med >= 0.005:
        print('  判定：晋级 → 进 K=40 双窗口回测确认。')
    elif len(deltas) == 4 and pos <= 2:
        print('  判定：无效，按预注册规则关闭 E7 轴，不进回测。')
    else:
        print('  判定：效应存在但未达晋级门槛（或样本未满），按台账规则记录后不进回测。')


if __name__ == '__main__':
    main()
