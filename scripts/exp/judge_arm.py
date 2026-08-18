"""串行协议的单臂判定器：候选 vs 当前基线，一次打全三道门的读数。

为什么不复用 `analyze_holdout.py`
--------------------------------
那个是 T096 时代的工具：同一个 prefix 下横比多个臂，且只看整截面 holdout IC。
本轮（T113~T116）是**串行协议** —— 每轮的基线是上一轮存活下来的面板，
所以要跨 prefix 配对（如 `T113_fp16` vs `T098_full`）；而且
[[regime-stratified-ic-gate]] 已经把**跌日 ΔIC 符号**立成主判否决门，
整截面 IC 会把「涨日赢跌日输」平均成净正，单看它会重蹈 E11 的覆辙。

三道门（判据在各 run_t11x_*.sh 头部预注册，本工具只出读数与机械判读）
----------------------------------------------------------------
  门0 通道校准（--gate calib）：|Δ| 应在 dtype 效应量级（~0.001）；
       任一种子 |Δ|>0.01 ⇒ 通道异常，不得用作基线。
  门1 晋级（--gate promote）：holdout 配对 4/4 为正 **且** Δ中位 ≥ 0.005。
  门2 无害（--gate noharm）：不得「0/4 且 |Δ中位| > MDE」（决定性劣化）。
  门3 分层否决（所有 gate 都查）：跌日 ΔIC ≤1/3 种子为正 ⇒ 关轴/回退，
       不论整截面多好看。

注意 `regime` 块算在整个验证折上（带选型偏差），但配对两臂偏差结构相同，
Δ 仍可读 —— T108/T111 终审用的就是这个口径。

用法:
  python scripts/exp/judge_arm.py --base T098_full --cand T113_fp16 --gate calib
  python scripts/exp/judge_arm.py --base T113_fp16 --cand T114_slim --gate noharm
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEDS = (42, 11, 23, 37)
DIRS = (os.path.join(ROOT, 'diagnose_output'),
        os.path.join(ROOT, 'diagnose_output', 'iter_output'))


def _load(tag: str, seed: int):
    for d in DIRS:
        p = os.path.join(d, f'{tag}_s{seed}.json')
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as f:
                return json.load(f)
    return None


def _metrics(tag: str):
    """{seed: dict(holdout, ic, up, down, n_feat)}；缺的种子不入表。"""
    out = {}
    for s in SEEDS:
        d = _load(tag, s)
        if not d:
            continue
        ho, ic, up, dn = [], [], [], []
        for row in d['folds'].values():
            r = row.get('nam_gate') if isinstance(row, dict) else None
            if not r:
                continue
            if r.get('rank_ic_holdout') is not None:
                ho.append(float(r['rank_ic_holdout']))
            if r.get('rank_ic') is not None:
                ic.append(float(r['rank_ic']))
            reg = r.get('regime') or {}
            for key, acc in (('up_days', up), ('down_days', dn)):
                v = (reg.get(key) or {}).get('rank_ic')
                if v is not None:
                    acc.append(float(v))
        if not ho:
            continue
        out[s] = {
            'holdout': float(np.mean(ho)),
            'ic': float(np.mean(ic)) if ic else np.nan,
            'up': float(np.mean(up)) if up else np.nan,
            'down': float(np.mean(dn)) if dn else np.nan,
            'n_feat': d.get('metadata', {}).get('nam_features'),
        }
    return out


def _paired(cand: dict, base: dict, key: str):
    common = sorted(set(cand) & set(base))
    d = np.array([cand[s][key] - base[s][key] for s in common], dtype=float)
    return common, d


def _mde(d: np.ndarray) -> float:
    return 2.0 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='基线 run tag，如 T113_fp16')
    ap.add_argument('--cand', required=True, help='候选 run tag，如 T114_slim')
    ap.add_argument('--gate', default='promote',
                    choices=['calib', 'promote', 'noharm'],
                    help='预注册的门类型：calib=通道校准 / promote=晋级 / noharm=无害')
    ap.add_argument('--promote-delta', type=float, default=0.005)
    a = ap.parse_args()

    base, cand = _metrics(a.base), _metrics(a.cand)
    if not base or not cand:
        print(f'[缺数据] base={len(base)} 个种子, cand={len(cand)} 个种子')
        return 2

    nb = {v['n_feat'] for v in base.values()}
    nc = {v['n_feat'] for v in cand.values()}
    print(f'基线 {a.base}: n={len(base)} 种子, 特征 {nb}')
    print(f'候选 {a.cand}: n={len(cand)} 种子, 特征 {nc}')
    print()

    print(f"{'seed':>5} {'base_ho':>9} {'cand_ho':>9} {'Δho':>9} "
          f"{'base_dn':>9} {'cand_dn':>9} {'Δ跌日':>9} {'Δ涨日':>9}")
    common, dh = _paired(cand, base, 'holdout')
    _, dd = _paired(cand, base, 'down')
    _, du = _paired(cand, base, 'up')
    for i, s in enumerate(common):
        print(f'{s:>5} {base[s]["holdout"]:>9.5f} {cand[s]["holdout"]:>9.5f} {dh[i]:>+9.5f} '
              f'{base[s]["down"]:>9.5f} {cand[s]["down"]:>9.5f} {dd[i]:>+9.5f} {du[i]:>+9.5f}')

    mde = _mde(dh)
    pos, n = int((dh > 0).sum()), len(dh)
    dn_pos = int((dd > 0).sum())
    print()
    print(f'holdout 配对: {pos}/{n} 为正, Δ中位 {np.median(dh):+.5f}, '
          f'Δ均值 {dh.mean():+.5f}, σ {dh.std(ddof=1):.5f}, MDE≈{mde:.5f}')
    print(f'跌日 ΔIC   : {dn_pos}/{n} 为正, Δ中位 {np.median(dd):+.5f}')
    print(f'涨日 ΔIC   : {int((du > 0).sum())}/{n} 为正, Δ中位 {np.median(du):+.5f}')
    print()

    # --- 门3：分层否决（所有 gate 都先查这个）---
    # 两个条件同时满足才否决：①计数 ≤1/3 为正；②跌日劣化**幅度**超过跌日自身
    # 配对 MDE。只数符号会在噪声区误伤：T113 通道校准里跌日 Δ 中位仅 −0.0025、
    # 均值 −0.0033 < MDE 0.0045（不显著），却拿到 1/4 的计数。作为对照，
    # T108/T111 真正该关轴时跌日 Δ 是 −0.031/−0.090/−0.051/−0.051 —— 量级差
    # 一个数量级，加幅度条件后那批依然会被正确否决。
    dd_mde = _mde(dd)
    dd_material = abs(dd.mean()) > dd_mde if np.isfinite(dd_mde) else True
    veto = (dn_pos * 3 <= n) and dd_material
    print('【门3 分层否决】跌日 ΔIC %d/%d 为正, 均值 %+.5f vs 跌日MDE %.5f → %s'
          % (dn_pos, n, dd.mean(), dd_mde,
             '触发否决（关轴/回退）' if veto
             else ('计数够但幅度不显著，不否决' if dn_pos * 3 <= n else '未触发')))

    # --- 主门 ---
    if a.gate == 'calib':
        worst = float(np.max(np.abs(dh)))
        ok = worst <= 0.01
        print(f'【门0 通道校准】最大 |Δ| = {worst:.5f} → '
              f'{"通道等价，可作基线" if ok else "通道异常，不得作基线"}')
        verdict = ok
    elif a.gate == 'promote':
        ok = (pos == n) and (np.median(dh) >= a.promote_delta)
        print(f'【门1 晋级】需 {n}/{n} 为正且 Δ中位 ≥ {a.promote_delta} → '
              f'{"晋级" if ok else "不晋级"}')
        verdict = ok
    else:
        decisive_bad = (pos == 0) and (abs(np.median(dh)) > mde)
        ok = not decisive_bad
        print(f'【门2 无害】决定性劣化(0/{n} 且 |Δ中位|>MDE) = {decisive_bad} → '
              f'{"无害，可接受" if ok else "劣化，回退"}')
        verdict = ok

    final = verdict and not veto
    print()
    print(f'>>> 综合判读: {"通过" if final else "不通过"}'
          f'{"（分层门否决优先）" if verdict and veto else ""}')
    return 0 if final else 1


if __name__ == '__main__':
    sys.exit(main())
