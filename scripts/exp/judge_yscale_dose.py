#!/usr/bin/env python
"""T133 判定：**y_scale 剂量-反应扫描**，用能看见头部的尺子。

为什么要单独一个脚本
--------------------
`diag_seed_ensemble.py` 报的是逐载体的 holdout 均值，够判"换不换载体"，
但剂量扫描要判的是**趋势形状**和**逐日配对显著性**：
  · 每臂只有 2 种子（用户指定：4 个剂量点 × 2 种子，换取剂量-反应曲线），
    原先「4/4 为正」那条门用不了 —— 2/2 为正在纯噪声下就有 25% 概率。
  · 补偿手段是判**跨剂量的趋势一致性**：8 个点连成的单调/单峰曲线，
    比 2 个点各 4 种子更难被噪声伪造。锯齿是噪声的形状。
  · 头部超额的量级只有 IC 的十分之一，点估计之间的差必须配 t 才知道分不分得开。

主尺子 = **全池 top-K 超额**（当日按候选分数取前 K，减当日全池等权收益）。
  为什么不是"前 200 候选池内取前 20"：用**同一个分数**既定池又在池内排序时，
  池子大小是**恒等空转** —— 4700 只取前 200、再在这 200 里取前 20，
  就等于 4700 只里取前 20（已 1200 次合成检验，改 head_n 零次改变结果）。
  T130 的池子有意义是因为那里有**两个分数**：池由冻结的生产分数定、
  池内按候选方案的族权重重排。本实验比的是两个独立训练出来的模型，
  各自用自己的分数选股，没有两阶段，所以全池 top-K 才是它真正在做的事。
  ⚠ 这只是**改正标签**，判据数值未动 —— 那个数一直是全池 top-K 超额。

  副记录（**不参与晋级**）：跨池重排 —— 池由**同种子基线臂**的分数定、池内按候选臂排。
  它回答"给定在跑配方的候选名单，候选臂能不能排得更好"，把"重排得更好"
  与"名单本身不同"分开。注意池子大小在这把尺子上是**在基线排序与候选排序之间插值**
  的旋钮（head_n=K 就是纯基线，head_n=n 就是纯候选），所以它只能事前定、不能挑。

全截面 rank IC 对只动头部的候选**结构性失明**
  （见 [[head-ruler-is-blind-in-full-cross-section-ic]]），这里只作副判据记录。

预注册判据（写在 scripts/exp/run_t133_head_loss.sh 头部，出结果前定死）
----------------------------------------------------------------------
  主判据：逐种子配对 vs **同种子**的基线臂（y2），三条同时成立才晋级
    (a) 2/2 种子 Δ 为正
    (b) Δ 中位 ≥ +0.002
    (c) 跨剂量趋势一致：两个种子给出的曲线形状同为单调或单峰，且峰位一致
  否决门（优先于主判据）：**跌日** Δ 必须 2/2 为正
  证伪读数：y1 vs y2。y1 若不劣于 y2 ⇒ 赢的不是"对准头部"而是"动一下就有"，整轴关闭。

用法
  python scripts/exp/judge_yscale_dose.py \
      --models models/nam_gate/T133_yscale1_s42,...,models/nam_gate/T133_yscale8_s11 \
      --years 13 --end 2022-09-05 --topk 20 --extra-topk 5,50 \
      --cache-npz diagnose_output/T133_groupsums.npz
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scipy.stats import rankdata

from core.factors.nam_gate_model import NAMGateModel
from scripts.exp.diag_seed_ensemble import load_group_sums

TAG_RE = re.compile(r'yscale(?P<arm>[\d.]+)_s(?P<seed>\d+)$')


def parse_tag(d):
    """从存档目录名解析 (剂量, 种子)。解析不出来就整名当一臂，避免静默错配。"""
    base = d.rstrip('/\\').replace('\\', '/').split('/')[-1]
    m = TAG_RE.search(base)
    if not m:
        return base, base, base
    return base, float(m.group('arm')), m.group('seed')


def shape_of(vals):
    """给一条剂量-反应序列判形状：单调增/单调减/单峰/单谷/锯齿。

    单峰 = 先升后降且只有一个转折。锯齿 = 转折 ≥2 次，那是噪声的形状。
    """
    d = np.diff(vals)
    s = np.sign(d)
    s = s[s != 0]
    if len(s) == 0:
        return 'flat'
    turns = int((np.diff(s) != 0).sum())
    if turns == 0:
        return 'mono_up' if s[0] > 0 else 'mono_down'
    if turns == 1:
        return 'peak' if s[0] > 0 else 'valley'
    return 'zigzag'


def paired(a, b):
    """逐日配对差的均值/配对 t/胜日率。a、b 已对齐到同一批日期。"""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return float('nan'), float('nan'), float('nan'), 0
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else float('nan')
    return float(d.mean()), float(t), float((d > 0).mean()), len(d)


def top_k_excess(score, r, k):
    """全池 top-K 超额：当日按分数取前 K，减当日全池等权收益。

    这是主尺子。**不带 head_n** —— 同一分数既定池又排序时池子恒等空转，
    带上只会让人误以为在测"池内重排"（那需要两个分数，见 cross_pool_excess）。
    """
    kk = min(k, len(score))
    top = np.argpartition(score, -kk)[-kk:]
    return float(r[top].mean() - r.mean())


def cross_pool_excess(pool_score, rank_score, r, head_n, k):
    """跨池重排：池由 pool_score 定（前 head_n），池内按 rank_score 取前 k。

    ⚠ head_n 是**在 pool_score 排序与 rank_score 排序之间插值**的旋钮：
      head_n=k  ⇒ 完全等于 pool_score 的前 k（rank_score 无腾挪空间）
      head_n=n  ⇒ 完全等于 rank_score 的前 k
    所以它必须事前定死；按超额最高去挑 head_n 就是在验证集上挑模型。
    """
    n = len(pool_score)
    pool = np.argpartition(pool_score, -head_n)[-head_n:] if n > head_n else np.arange(n)
    kk = min(k, len(pool))
    top = pool[np.argpartition(rank_score[pool], -kk)[-kk:]]
    return float(r[top].mean() - r.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True, help='逗号分隔的存档目录（名字需含 yscale<X>_s<seed>）')
    ap.add_argument('--baseline-arm', type=float, default=2.0, help='配对基线剂量（默认 y2 = 在跑配方）')
    ap.add_argument('--cache-dir', default='database/system_data/factors_cache_2026-08-18-idxrel')
    ap.add_argument('--cache-npz', default='')
    ap.add_argument('--stocks', type=int, default=6000)
    ap.add_argument('--years', type=int, default=13)
    ap.add_argument('--end', default='2022-09-05')
    ap.add_argument('--train-fraction', type=float, default=0.8)
    ap.add_argument('--select-holdout', type=float, default=0.4)
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--topk', type=int, default=20, help='主尺子的 K（全池 top-K 超额）')
    ap.add_argument('--extra-topk', default='5,50',
                    help='附带报告的其它 K（逗号分隔），看结论是否只在某个 K 上成立')
    ap.add_argument('--cross-pool-n', type=int, default=200,
                    help='副记录用：池由同种子基线臂定的池大小；事前定死，不参与晋级')
    ap.add_argument('--min-rows', type=int, default=30)
    ap.add_argument('--out', default='diagnose_output/T133_yscale_dose.json')
    a = ap.parse_args()


    dirs = [d.strip() for d in a.models.split(',') if d.strip()]
    meta = [parse_tag(d) for d in dirs]
    tags = [m[0] for m in meta]
    arms = sorted({m[1] for m in meta})
    seeds = sorted({m[2] for m in meta})
    print(f'{len(dirs)} 个存档 → 剂量 {arms} × 种子 {seeds}')
    if a.baseline_arm not in arms:
        raise SystemExit(f'基线剂量 {a.baseline_arm} 不在 {arms} 里，无法配对')

    m0 = NAMGateModel()
    m0.load_model(os.path.join(dirs[0], 'nam_gate_factor_model.pkl'))
    feats, gid, K = list(m0.feature_names), np.asarray(m0.group_ids), len(m0.group_names)
    print(f'面板 {len(feats)} 列 / {K} 族，窗口 {a.years}y → {a.end}')

    S_va, ret_va, va_days = load_group_sums(dirs, a, feats, gid, K)
    days = [(s, e) for s, e in va_days if e - s >= a.min_rows]
    n_sel = int(round(len(days) * (1.0 - a.select_holdout)))
    ho = days[n_sel:]
    print(f'验证 {len(days)} 日 → 选型 {n_sel} / holdout {len(ho)} 日'
          f'（只在 holdout 上判，选型段已被早停用掉，见 [[checkpoint-selection-inflates-ic]]）')

    # 逐日：每个存档的 rank IC 与全池 top-K 超额；市场方向用当日全池等权收益
    extra_ks = [int(x) for x in a.extra_topk.split(',') if x.strip()]
    ic = {t: np.full(len(ho), np.nan) for t in tags}
    ex = {t: np.full(len(ho), np.nan) for t in tags}
    exk = {(t, k): np.full(len(ho), np.nan) for t in tags for k in extra_ks}
    xp = {t: np.full(len(ho), np.nan) for t in tags}          # 跨池重排（副记录）
    mkt = np.full(len(ho), np.nan)
    by = {(m[1], m[2]): m[0] for m in meta}
    tag2col = {t: j for j, t in enumerate(tags)}
    for i, (s, e) in enumerate(ho):
        r = np.asarray(ret_va[s:e], dtype=np.float64)
        rr = rankdata(r)
        mkt[i] = r.mean()
        sc_cache = {}
        for j, t in enumerate(tags):
            sc = S_va[j][s:e].astype(np.float64).sum(1)
            sc_cache[t] = sc
            ic[t][i] = np.corrcoef(rankdata(sc), rr)[0, 1]
            ex[t][i] = top_k_excess(sc, r, a.topk)
            for k in extra_ks:
                exk[(t, k)][i] = top_k_excess(sc, r, k)
        # 副记录：池由同种子基线臂的分数定，池内按候选臂排
        for arm, sd in by:
            t = by[(arm, sd)]
            tb = by.get((a.baseline_arm, sd))
            if tb is None or t not in sc_cache or tb not in sc_cache:
                continue
            xp[t][i] = cross_pool_excess(sc_cache[tb], sc_cache[t], r,
                                         a.cross_pool_n, a.topk)
    dn = mkt <= 0
    print(f'holdout 涨日 {int((~dn).sum())} / 跌日 {int(dn.sum())}')

    # ── 剂量-反应表 ─────────────────────────────────────────────────────────
    print(f'\n{"存档":26s} {"top%d超额" % a.topk:>10s} {"超额涨日":>9s} {"超额跌日":>9s} '
          f'{"全截面IC":>10s} {"IC涨日":>8s} {"IC跌日":>8s}')
    res = {}
    for t in tags:
        res[t] = {'ex': float(np.nanmean(ex[t])), 'ex_up': float(np.nanmean(ex[t][~dn])),
                  'ex_down': float(np.nanmean(ex[t][dn])), 'ic': float(np.nanmean(ic[t])),
                  'ic_up': float(np.nanmean(ic[t][~dn])), 'ic_down': float(np.nanmean(ic[t][dn])),
                  'cross_pool_ex': float(np.nanmean(xp[t])),
                  'ex_other_k': {str(k): float(np.nanmean(exk[(t, k)])) for k in extra_ks}}
        v = res[t]
        print(f'{t:26s} {v["ex"]:+10.4f} {v["ex_up"]:+9.4f} {v["ex_down"]:+9.4f} '
              f'{v["ic"]:+10.5f} {v["ic_up"]:+8.4f} {v["ic_down"]:+8.4f}')

    print(f'\n其它 K（看结论是否只在 K={a.topk} 上成立）'
          f'  以及跨池重排（池由同种子 y{a.baseline_arm:g} 定，前 {a.cross_pool_n}；**副记录不参与晋级**）')
    hdr = ''.join(f'{"top%d" % k:>10s}' for k in extra_ks)
    print(f'{"存档":26s}{hdr}{"跨池top%d" % a.topk:>12s}')
    for t in tags:
        cells = ''.join(f'{res[t]["ex_other_k"][str(k)]:+10.4f}' for k in extra_ks)
        print(f'{t:26s}{cells}{res[t]["cross_pool_ex"]:+12.4f}')

    # ── 逐种子配对 vs 同种子基线臂 ──────────────────────────────────────────
    print(f'\n★ 逐种子配对 vs y{a.baseline_arm:g}（同种子，n={len(ho)} 日）')
    print(f'{"":10s} {"种子":>6s} {"Δ头部":>9s} {"配对t":>7s} {"胜日":>6s} '
          f'{"Δ头部跌日":>11s} {"Δ全截面IC":>11s}')
    pair = {}
    for arm in arms:
        if arm == a.baseline_arm:
            continue
        for sd in seeds:
            if (arm, sd) not in by or (a.baseline_arm, sd) not in by:
                continue
            ta, tb = by[(arm, sd)], by[(a.baseline_arm, sd)]
            dm, tt, wr, n = paired(ex[ta], ex[tb])
            dd, _, _, _ = paired(ex[ta][dn], ex[tb][dn])
            di, ti, _, _ = paired(ic[ta], ic[tb])
            pair[f'y{arm:g}_s{sd}'] = {'d_ex': dm, 't_ex': tt, 'win': wr,
                                       'd_ex_down': dd, 'd_ic': di, 't_ic': ti, 'n': n}
            print(f'y{arm:<9g} {sd:>6s} {dm:+9.4f} {tt:+7.2f} {wr * 100:5.1f}% '
                  f'{dd:+11.4f} {di:+11.5f}')

    # ── 趋势形状 ────────────────────────────────────────────────────────────
    print(f'\n★ 剂量-反应形状（头部超额，逐种子）')
    shapes, curves = {}, {}
    for sd in seeds:
        vals = [res[by[(arm, sd)]]['ex'] for arm in arms if (arm, sd) in by]
        got = [arm for arm in arms if (arm, sd) in by]
        if len(vals) < 3:
            continue
        sh = shape_of(vals)
        shapes[sd] = sh
        curves[sd] = {f'y{k:g}': v for k, v in zip(got, vals)}
        peak = got[int(np.argmax(vals))]
        print(f'  s{sd}: ' + '  '.join(f'y{k:g}={v:+.4f}' for k, v in zip(got, vals))
              + f'   形状={sh}  峰位=y{peak:g}')
    # 形状一致 = 两个种子给出同一种形状，且都不是锯齿（锯齿是噪声的形状）
    trend_ok = len(shapes) >= 2 and len(set(shapes.values())) == 1 \
        and 'zigzag' not in set(shapes.values()) and 'flat' not in set(shapes.values())
    peaks = {sd: max(curves[sd], key=curves[sd].get) for sd in curves}
    peak_ok = len(set(peaks.values())) == 1
    print(f'  ⇒ 形状一致={trend_ok}（{shapes}）  峰位一致={peak_ok}（{peaks}）')

    # ── 预注册裁决 ──────────────────────────────────────────────────────────
    print(f'\n★ 预注册裁决（判据见 run_t133_head_loss.sh）')
    verdict = {}
    for arm in arms:
        if arm == a.baseline_arm:
            continue
        ds = [pair[k]['d_ex'] for k in pair if k.startswith(f'y{arm:g}_')]
        dd = [pair[k]['d_ex_down'] for k in pair if k.startswith(f'y{arm:g}_')]
        if not ds:
            continue
        a_ok = all(x > 0 for x in ds)
        b_ok = float(np.median(ds)) >= 0.002
        veto_ok = all(x > 0 for x in dd)
        ok = veto_ok and a_ok and b_ok and trend_ok and peak_ok
        verdict[f'y{arm:g}'] = {'n_pos': int(sum(x > 0 for x in ds)), 'n': len(ds),
                                'median_d_ex': float(np.median(ds)),
                                'down_pos': int(sum(x > 0 for x in dd)),
                                'a_all_pos': a_ok, 'b_median': b_ok,
                                'veto_down': veto_ok, 'trend': trend_ok and peak_ok,
                                'promote': bool(ok)}
        print(f'  y{arm:g}: (a) {sum(x > 0 for x in ds)}/{len(ds)} 为正 {"✓" if a_ok else "✗"}   '
              f'(b) Δ中位 {np.median(ds):+.4f} {"✓" if b_ok else "✗"}   '
              f'(c) 趋势 {"✓" if trend_ok and peak_ok else "✗"}   '
              f'否决门跌日 {sum(x > 0 for x in dd)}/{len(dd)} {"✓" if veto_ok else "✗"}'
              f'   ⇒ {"晋级" if ok else "不晋级"}')

    # 证伪读数：y1 不劣于 y2 ⇒ 赢的不是"对准头部"
    fals = None
    if 1.0 in arms and a.baseline_arm == 2.0:
        d1 = [pair[k]['d_ex'] for k in pair if k.startswith('y1_')]
        if d1:
            worse = all(x < 0 for x in d1)
            fals = {'d_ex': d1, 'y1_worse_than_y2': bool(worse)}
            print(f'\n★ 证伪读数 y1 vs y2（头部尺子）: Δ = '
                  + ', '.join(f'{x:+.4f}' for x in d1))
            print(f'   y1 劣于 y2 = {worse} —— '
                  + ('机制方向自洽，可继续看 y4/y8。'
                     if worse else
                     '**y1 不劣 ⇒ 赢的不是"对准头部"而是扰动噪声，整轴应关闭。**'))

    out = {'models': dirs, 'window': f'{a.years}y→{a.end}', 'topk': a.topk,
           'extra_topk': extra_ks, 'cross_pool_n': a.cross_pool_n,
           'n_holdout_days': len(ho),
           'n_up': int((~dn).sum()), 'n_down': int(dn.sum()),
           'per_archive': res, 'paired_vs_baseline': pair,
           'baseline_arm': a.baseline_arm, 'shapes': shapes, 'curves': curves,
           'trend_consistent': bool(trend_ok and peak_ok), 'verdict': verdict,
           'falsification_y1': fals}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
