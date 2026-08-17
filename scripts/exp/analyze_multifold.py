"""多折验证 IC 标定/判定器。

- 单臂（只给 --prefix）：打印每折 IC、每种子折均 IC、跨种子 σ_seed，
  并给出新的 IC 判定 MDE ≈ 2·σ_seed/√n（n=4 配对时的可检测下限量级）。
- 双臂（再给 --base-prefix）：同种子配对比较折均 IC，**并做 T082/E12 的行情分层门槛**：
  合并 IC 只是必要条件，还要看下跌日 ΔIC 的符号一致性（见 `_regime_gate`）。

用法:
  python scripts/exp/analyze_multifold.py --prefix T078_mf_base
  python scripts/exp/analyze_multifold.py --prefix T079_xxx --base-prefix T078_mf_base
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEDS = [42, 11, 23, 37]
KEY = 'best_val_rank_ic_on_label'
# 结果 JSON 的搜索顺序：先看在跑的 diagnose_output/，再看归档的 iter_output/。
# 跑完的轮次会被移进 iter_output 归档，但历史基线（T090_base 等）仍要能当
# --base-prefix 用，所以两处都找。
_SEARCH_DIRS = (os.path.join(ROOT, 'diagnose_output'),
                os.path.join(ROOT, 'diagnose_output', 'iter_output'))


def _find(prefix, seed):
    """在搜索路径里定位 {prefix}_s{seed}.json；找不到返回 None。"""
    for d in _SEARCH_DIRS:
        p = os.path.join(d, f'{prefix}_s{seed}.json')
        if os.path.isfile(p):
            return p
    return None


def _folds(prefix, seed):
    p = _find(prefix, seed)
    if p is None:
        return None
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    out = {}
    for tag, row in d['folds'].items():
        r = row.get('nam_gate')
        if r:
            out[tag] = float(r.get(KEY, r['rank_ic']))
    return out


def _regime(prefix, seed):
    """取每折的行情分层指标（老结果没有这一段，返回空）。"""
    p = _find(prefix, seed)
    if p is None:
        return {}
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    out = {}
    for tag, row in d['folds'].items():
        r = (row.get('nam_gate') or {}).get('regime')
        if r:
            out[tag] = r
    return out


def _table(prefix):
    per_seed = {s: _folds(prefix, s) for s in SEEDS}
    have = {s: v for s, v in per_seed.items() if v}
    if not have:
        print(f'  {prefix}: 尚无结果')
        return None
    tags = sorted({t for v in have.values() for t in v})
    print(f'\n=== {prefix} ===')
    print(f'{"seed":>5} ' + ' '.join(f'{t:>10}' for t in tags) + f' {"折均":>9}')
    means = {}
    for s, v in have.items():
        vals = [v[t] for t in tags if t in v]
        means[s] = statistics.fmean(vals)
        print(f'{s:>5} ' + ' '.join(f'{v.get(t, float("nan")):10.5f}' for t in tags)
              + f' {means[s]:9.5f}')
    for t in tags:
        vals = [v[t] for v in have.values() if t in v]
        if len(vals) > 1:
            print(f'  折 {t}: 跨种子 mean {statistics.fmean(vals):.5f} '
                  f'std {statistics.stdev(vals):.5f}')
    if len(means) > 1:
        sd = statistics.stdev(means.values())
        print(f'  折均 IC: mean {statistics.fmean(means.values()):.5f} '
              f'σ_seed {sd:.5f}  → 配对 n={len(means)} 的 MDE ≈ '
              f'{2 * sd / math.sqrt(len(means)):.5f}')
    return means


def _regime_gate(cur_prefix, base_prefix):
    """T082/E12 行情分层门槛：下跌日 ΔIC 的符号一致性。

    E11 的教训是合并 IC 会把「上涨日赢、下跌日输」平均成净正。下跌日的日数少、
    单点噪声大（σ_seed ≈ 0.013，比合并口径大一个量级），所以这里**不做幅度检验**，
    只做符号检验：把所有 (折 × 种子) 的下跌日 ΔIC 汇总，≤1/3 为正即判定为
    regime 倾斜，直接关轴，不给回测。
    """
    rows = []
    for s in SEEDS:
        c, b = _regime(cur_prefix, s), _regime(base_prefix, s)
        for tag in sorted(set(c) & set(b)):
            for stratum in ('up_days', 'down_days'):
                cv, bv = c[tag].get(stratum, {}), b[tag].get(stratum, {})
                if cv.get('rank_ic') is None or bv.get('rank_ic') is None:
                    continue
                rows.append((stratum, s, tag, cv['rank_ic'] - bv['rank_ic'],
                             (cv.get('topk_excess') or 0) - (bv.get('topk_excess') or 0)))
    if not rows:
        print('\n[行情分层] 两臂都没有 regime 段（T082 之前的老结果），跳过。'
              '重跑训练即可获得该门槛。')
        return
    print(f'\n=== 行情分层 Δ（T082/E12 门槛）: {cur_prefix} − {base_prefix} ===')
    for stratum in ('up_days', 'down_days'):
        d = [r[3] for r in rows if r[0] == stratum]
        x = [r[4] for r in rows if r[0] == stratum]
        if not d:
            continue
        pos = sum(1 for v in d if v > 0)
        print(f'  {stratum:<10} ΔIC {pos}/{len(d)} 正向，中位 {statistics.median(d):+.5f}'
              f'   |  Δtop-K 超额中位 {100 * statistics.median(x):+.3f} pp')
    dn = [r[3] for r in rows if r[0] == 'down_days']
    if dn:
        pos = sum(1 for v in dn if v > 0)
        if pos * 3 <= len(dn):
            print(f'  → **不通过**：下跌日只有 {pos}/{len(dn)} 正向，是 regime 倾斜不是 alpha。'
                  f'（E11 当时是 2/12）')
        elif statistics.median(dn) >= 0:
            print('  → 通过：下跌日不劣化。')
        else:
            print('  → 存疑：下跌日符号未一边倒但中位为负，需扩种子再判。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--base-prefix', default=None)
    a = ap.parse_args()

    cur = _table(a.prefix)
    if not a.base_prefix:
        return
    base = _table(a.base_prefix)
    if not cur or not base:
        return
    common = [s for s in SEEDS if s in cur and s in base]
    print(f'\n=== 配对 Δ折均IC: {a.prefix} − {a.base_prefix} ===')
    deltas = []
    for s in common:
        d = cur[s] - base[s]
        deltas.append(d)
        print(f'  s{s} {base[s]:.5f} → {cur[s]:.5f}   Δ {d:+.5f}')
    if deltas:
        pos = sum(1 for d in deltas if d > 0)
        print(f'  {pos}/{len(deltas)} 上升，Δ 中位 {statistics.median(deltas):+.5f}')
    _regime_gate(a.prefix, a.base_prefix)


if __name__ == '__main__':
    main()
