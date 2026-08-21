#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T137b —— 对 T137 里**唯一过门**的那一格（score 的逐日截面**离散度 std**）做自查。

为什么必须自查：T137 的判据 (3) 只看「幅度 ≥0.5pp 且 4/4 同号」，
**没有显著性检验**，而且那 4 个存档看的是**同一批 161 天、同一族配方**，
分数之间高度相关 ⇒「4/4 一致」几乎不提供独立证据（这是我判据设计的弱点）。
再叠加 7 日重叠收益的自相关 0.86，低五分位的 32 天很可能只是**少数几段行情**。

本脚本查四件事：
  ① 聚集度：低五分位的日子是几段连续区间？（段数 ≈ 有效样本量）
  ② 显著性：移动块自助（block=10）给低五分位效应一个 p 值。
  ③ 段外一致性：选型段（前 242 日）上同样的效应还在吗？
     —— 逐日**水平**从未被选型用过，所以这一段对本问题是干净的追加样本。
  ④ 增量：score-std 相对两个平凡基线还有没有增量 ——
     (a) 已知的**过去 7 日市场收益**（动量），(b) **过去实现波动**代理。
     若 score-std 只是波动率的影子，那它不是「模型原始输出带的信息」，
     而是 regime 特征里早就有、且 T135 已判过的东西。
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import rankdata, spearmanr


def runs_of(mask):
    """布尔序列里连续 True 段的段数。"""
    m = np.asarray(mask, bool).astype(int)
    return int(((np.diff(np.concatenate([[0], m, [0]])) == 1).sum()))


def block_bootstrap_p(v, y, q, n_boot=5000, block=10, seed=0):
    """低五分位效应的移动块自助 p 值（单尾：效应为负）。

    对 (v, y) **成对**做块重采样，保留时间依赖结构，每次重算分位与效应。
    """
    rng = np.random.default_rng(seed)
    n = len(v)
    obs = float(y[v <= np.quantile(v, q)].mean() - y.mean())
    nb = int(np.ceil(n / block))
    starts_pool = np.arange(0, n - block + 1)
    cnt = 0
    for _ in range(n_boot):
        st = rng.choice(starts_pool, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        vb, yb = v[idx], y[idx]
        eb = float(yb[vb <= np.quantile(vb, q)].mean() - yb.mean())
        if eb <= obs:
            cnt += 1
    return obs, float(cnt / n_boot)


def ols_t(X, y):
    """多元 OLS，返回各系数的普通 t（仅作方向参考；重叠收益下会虚高）。"""
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(e @ e) / dof
    V = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(V))
    return beta[1:], beta[1:] / se[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-npz', default='diagnose_output/T133_groupsums.npz')
    ap.add_argument('--model-idx', default='0,1,2,3')
    ap.add_argument('--model-names', default='y1_s42,y1_s11,y2_s42,y2_s11')
    ap.add_argument('--select-holdout', type=float, default=0.4)
    ap.add_argument('--min-stocks', type=int, default=30)
    ap.add_argument('--horizon', type=int, default=7)
    ap.add_argument('--quantile', type=float, default=0.2)
    ap.add_argument('--block', type=int, default=10)
    ap.add_argument('--vol-window', type=int, default=20)
    ap.add_argument('--out', default='diagnose_output/T137b_score_std_selfcheck.json')
    a = ap.parse_args()

    idx = [int(v) for v in a.model_idx.split(',')]
    names = [v.strip() for v in a.model_names.split(',')]

    z = np.load(a.cache_npz, allow_pickle=False)
    ret_va = z['ret_va']
    days = [(int(s), int(e)) for s, e in z['va_days'] if int(e) - int(s) >= a.min_stocks]
    n_sel = int(round(len(days) * (1.0 - a.select_holdout)))
    print(f'验证 {len(days)} 日，选型段 {n_sel} 日，holdout {len(days) - n_sel} 日')

    mkt = np.array([float(np.asarray(ret_va[s:e], float).mean()) for s, e in days])
    H = a.horizon
    # 过去 7 日市场收益：mkt[i-H] 覆盖 [i-H, i]，在 i 日**已知**（不含未来）
    past = np.full(len(days), np.nan)
    past[H:] = mkt[:-H]
    # 过去实现波动代理：用已知的 past 序列滚动 std（每 H 日取一个不重叠点以减重叠）
    vol = np.full(len(days), np.nan)
    for i in range(len(days)):
        lo = i - a.vol_window
        if lo >= H:
            vol[i] = float(np.nanstd(past[lo:i]))

    S = {nm: z[f'S_va_{j}'].astype(np.float32) for j, nm in zip(idx, names)}
    std = {nm: np.array([float(S[nm][s:e].astype(np.float64).sum(1).std())
                         for s, e in days]) for nm in names}
    std['MEAN4'] = np.mean([std[nm] for nm in names], axis=0)
    all_names = names + ['MEAN4']

    segs = {'holdout': slice(n_sel, None), 'select': slice(0, n_sel),
            'all': slice(0, None)}

    print(f'\n===== ① 聚集度 + ② 显著性 + ③ 段外一致性 =====')
    print(f'{"存档":8s} {"段":9s} {"n":>4s} {"低分位n":>7s} {"连续段数":>8s} '
          f'{"效应":>9s} {"块自助p":>9s} {"Spearman":>9s}')
    out = {'window': {'n_days': len(days), 'n_select': n_sel,
                      'n_holdout': len(days) - n_sel, 'horizon': H,
                      'quantile': a.quantile, 'block': a.block}, 'std': {}}
    for nm in all_names:
        out['std'][nm] = {}
        for sname, sl in segs.items():
            v, y = std[nm][sl], mkt[sl]
            low = v <= np.quantile(v, a.quantile)
            eff, p = block_bootstrap_p(v, y, a.quantile, block=a.block)
            rho = float(spearmanr(v, y).correlation)
            out['std'][nm][sname] = {'n': int(len(v)), 'n_low': int(low.sum()),
                                     'runs': runs_of(low), 'effect': eff,
                                     'boot_p': p, 'spearman': rho}
            print(f'{nm:8s} {sname:9s} {len(v):4d} {int(low.sum()):7d} '
                  f'{runs_of(low):8d} {eff:+9.4%} {p:9.3f} {rho:+9.4f}')

    print(f'\n===== ④ 增量：score-std vs 两个平凡基线（holdout）=====')
    sl = segs['holdout']
    y = mkt[sl]
    pb, vb = past[sl], vol[sl]
    ok = np.isfinite(pb) & np.isfinite(vb)
    print(f'  可用日数 {int(ok.sum())}/{len(y)}（前 {a.vol_window + H} 日无基线）')
    base_res = {}
    for bn, bv in (('past_ret', pb), ('past_vol', vb)):
        rho = float(spearmanr(bv[ok], y[ok]).correlation)
        low = bv[ok] <= np.nanquantile(bv[ok], a.quantile)
        base_res[bn] = {'spearman': rho,
                        'low_q_effect': float(y[ok][low].mean() - y[ok].mean())}
        print(f'  基线 {bn:9s} Spearman {rho:+.4f}   低五分位效应 '
              f'{base_res[bn]["low_q_effect"]:+.4%}')
    print(f'\n{"存档":8s} {"与past_vol相关":>14s} {"单变量t":>9s} {"控制后t":>9s} {"控制后β符号":>12s}')
    for nm in all_names:
        v = std[nm][sl][ok]
        corr_vol = float(spearmanr(v, vb[ok]).correlation)
        _, t1 = ols_t([rankdata(v)], rankdata(y[ok]))
        b2, t2 = ols_t([rankdata(v), rankdata(pb[ok]), rankdata(vb[ok])],
                       rankdata(y[ok]))
        out['std'][nm]['incremental'] = {
            'corr_with_past_vol': corr_vol, 'univar_t': float(t1[0]),
            'ctrl_t': float(t2[0]), 'ctrl_beta': float(b2[0])}
        print(f'{nm:8s} {corr_vol:+14.4f} {float(t1[0]):+9.2f} '
              f'{float(t2[0]):+9.2f} {"+" if b2[0] > 0 else "-":>12s}')
    out['baselines'] = base_res

    # ---------- 结论判据 ----------
    m = out['std']['MEAN4']
    survives = bool(m['holdout']['boot_p'] < 0.05 and m['select']['effect'] < 0
                    and m['holdout']['runs'] >= 5
                    and abs(m['incremental']['ctrl_t']) >= 2)
    print(f'\n★ score-std 是否经得起自查：{"是" if survives else "否"}')
    print(f'   holdout 块自助 p = {m["holdout"]["boot_p"]:.3f}（要 <0.05）')
    print(f'   选型段效应 = {m["select"]["effect"]:+.4%}（要同为负）')
    print(f'   低分位连续段数 = {m["holdout"]["runs"]}（要 ≥5，否则只是几段行情）')
    print(f'   控制动量+波动后 t = {m["incremental"]["ctrl_t"]:+.2f}（要 |t|≥2）')
    out['survives'] = survives
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n已写 {a.out}')


if __name__ == '__main__':
    main()
