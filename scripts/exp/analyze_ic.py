"""通用 IC 判定器：把任意一组 `diagnose_output/<PREFIX>_s{seed}.json` 与基线
验证 Rank IC 做同种子配对比较。打分层轴的**唯一**筛选口径（见台账「方向修正」）。

用法:
  python scripts/exp/analyze_ic.py --prefix T077_nocf --label "E8-A 去 cross_fund"
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 基线 = T045/T068_seed*（--disable-gate, target=returns, 折 80-100%），同标签可比
BASE_IC = {42: 0.09685, 11: 0.08868, 23: 0.09384, 37: 0.09225}
FOLD = '80-100%'
PROMOTE_MEDIAN = 0.005          # ≈ σ_seed(0.0035) 的 1.4 倍


def _row(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    r = d['folds'][FOLD]['nam_gate']
    return (float(r.get('best_val_rank_ic_on_label', r['rank_ic'])),
            float(r['rank_ic_ir']), int(r['features_used']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True, help='结果文件前缀，如 T077_nocf')
    ap.add_argument('--label', default='')
    a = ap.parse_args()

    rows, deltas = [], []
    for seed, base in BASE_IC.items():
        p = os.path.join(ROOT, 'diagnose_output', f'{a.prefix}_s{seed}.json')
        if not os.path.isfile(p):
            print(f'  seed={seed} 尚无结果')
            continue
        ic, ir, nf = _row(p)
        rows.append((seed, base, ic, ic - base, ir, nf))
        deltas.append(ic - base)

    print(f'\n=== {a.label or a.prefix} ===')
    print(f'{"seed":>5} {"base_IC":>9} {"IC":>9} {"ΔIC":>9} {"IC_IR":>7} {"n_feat":>7}')
    for seed, base, ic, d, ir, nf in rows:
        print(f'{seed:>5} {base:9.5f} {ic:9.5f} {d:+9.5f} {ir:7.3f} {nf:>7}')
    if not deltas:
        return
    pos, med = sum(1 for d in deltas if d > 0), statistics.median(deltas)
    print(f'  {pos}/{len(deltas)} 上升，Δ 中位 {med:+.5f}')
    if len(deltas) < 4:
        print('  样本未满，暂不判定。')
    elif pos == 4 and med >= PROMOTE_MEDIAN:
        print('  判定：晋级 → 进 K=40 双窗口回测确认。')
    elif pos <= 2:
        print('  判定：无效，按预注册规则关闭该轴，不进回测。')
    else:
        print('  判定：未达晋级门槛，记录后不进回测。')


if __name__ == '__main__':
    main()
