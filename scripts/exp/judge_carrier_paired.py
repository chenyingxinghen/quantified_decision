#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T136 判定 —— 生产载体 T115_idxrel → T122_prod9y 换不换。

同卷（9y→2026-08-10，holdout 145 日），**逐种子配对**。
eval_models_on_window.py 自带的配对一律以 names[0] 为基准，
那会把「T122_s23 − T115_s42」这种跨种子差也算进来 —— 种子噪声（本卷 T115
内部 s11−s42 就有 +0.0157）会淹没载体效应。这里只做同种子配对。

预注册判据（见 run_t136_carrier_t115_vs_t122.sh 头部）：
  否决门：T122 跌日 holdout IC 不劣于 T115，≥3/4 种子。不过则直接不换。
  晋级 ：(a) 4/4 种子配对 Δ 为正   (b) Δ 中位数 ≥ +0.005
⚠ 本卷对 T122 有利（T115 训到 2020-12-18 是 4~5 年后的纯样本外；
  T122 训到 2025-01-20，holdout 紧邻其训练末端且与其选型段重叠）。
  ⇒ T122 赢要扣掉这两项才算数；T122 不赢则结论很硬。
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='diagnose_output/T136_T115_vs_T122_same_window.json')
    ap.add_argument('--incumbent', default='T115_idxrel')
    ap.add_argument('--challenger', default='T122_prod9y')
    ap.add_argument('--seeds', default='42,11,23,37')
    ap.add_argument('--line', type=float, default=0.005)
    a = ap.parse_args()

    d = json.load(open(a.json, encoding='utf-8'))
    seeds = [s.strip() for s in a.seeds.split(',')]
    w, M = d['window'], d['models']
    day_ret = np.asarray(d['day_ret'], float)
    n_sel = int(d['n_select_days'])
    dr = day_ret[n_sel:]
    dn, up = dr < 0, dr >= 0
    print(f'窗口 {w["start"]} → {w["end"]}，切点 {w["split_date"]}，'
          f'验证 {w["n_val_days"]} 日 → holdout {w["n_holdout_days"]} 日')
    print(f'holdout 跌日 {int(dn.sum())} / 涨日 {int(up.sum())}\n')

    print(f'{"种子":>6s} {"在跑 holdout":>13s} {"候选 holdout":>13s} {"Δ":>9s} '
          f'{"胜日":>7s} {"t":>7s} | {"在跑跌日":>9s} {"候选跌日":>9s} {"Δ跌日":>9s}')
    deltas, dn_deltas, rows = [], [], {}
    for s in seeds:
        ia, ib = f'{a.incumbent}_s{s}', f'{a.challenger}_s{s}'
        da = np.asarray(M[ia]['holdout_daily'], float)
        db = np.asarray(M[ib]['holdout_daily'], float)
        ok = np.isfinite(da) & np.isfinite(db)
        diff = db[ok] - da[ok]
        t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
        dda = float(np.nanmean(da[dn])) if dn.any() else float('nan')
        ddb = float(np.nanmean(db[dn])) if dn.any() else float('nan')
        deltas.append(float(diff.mean()))
        dn_deltas.append(ddb - dda)
        rows[s] = {'inc': float(np.nanmean(da)), 'chal': float(np.nanmean(db)),
                   'delta': float(diff.mean()), 'win_rate': float((diff > 0).mean()),
                   't': t, 'inc_down': dda, 'chal_down': ddb, 'delta_down': ddb - dda}
        r = rows[s]
        print(f'{s:>6s} {r["inc"]:+13.5f} {r["chal"]:+13.5f} {r["delta"]:+9.5f} '
              f'{r["win_rate"]:7.1%} {t:+7.2f} | {dda:+9.5f} {ddb:+9.5f} '
              f'{r["delta_down"]:+9.5f}')

    deltas = np.array(deltas)
    dn_deltas = np.array(dn_deltas)
    n_pos = int((deltas > 0).sum())
    med = float(np.median(deltas))
    n_dn_ok = int((dn_deltas >= 0).sum())

    print(f'\n★ 否决门（跌日 IC 不劣，≥3/4）：{n_dn_ok}/4 不劣，'
          f'Δ跌日中位 {float(np.median(dn_deltas)):+.5f} '
          f'⇒ {"过" if n_dn_ok >= 3 else "不过 —— 直接不换"}')
    veto_ok = n_dn_ok >= 3
    print(f'★ 判据 (a) 4/4 配对为正：{n_pos}/4 ⇒ {"过" if n_pos == 4 else "不过"}')
    print(f'★ 判据 (b) Δ 中位数 ≥ +{a.line:.3f}：{med:+.5f} '
          f'⇒ {"过" if med >= a.line else "不过"}')
    promote = bool(veto_ok and n_pos == 4 and med >= a.line)
    print(f'\n═══ 终判：{"换载体到 " + a.challenger if promote else "不换，保持 " + a.incumbent} ═══')
    if not promote:
        print('  ⚠ 而且这张卷子本来就偏向 T122（它训到 2025-01-20，holdout 紧邻其训练末端'
              '且与其选型段重叠；T115 训到 2020-12-18，是 4~5 年后的纯样本外）。')
        print('  在对候选最有利的卷子上都不过门 ⇒ 结论硬，不必再补实验。')
    out = {'window': w, 'rows': rows,
           'n_pos': n_pos, 'median_delta': med,
           'n_down_ok': n_dn_ok, 'median_delta_down': float(np.median(dn_deltas)),
           'veto_ok': veto_ok, 'promote': promote}
    p = a.json.replace('.json', '_verdict.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n已写 {p}')


if __name__ == '__main__':
    main()
