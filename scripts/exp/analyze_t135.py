# -*- coding: utf-8 -*-
"""
T135 配对分析：gate（不冻结端到端） vs base（disable-gate），同数据全量 5480 只
============================================================================
T069 协议：同种子配对 + 熊市主判 + 随机零假设分位/z 口径；牛市作稳健性参考。
用法：python scripts/exp/analyze_t135.py [--window bear|bull]
"""
import argparse
import glob
import json
import os

SEEDS = ["42", "11", "23", "37"]
RESULT_ROOT = "backtest_result"


def find_metrics(tag):
    out = {}
    for arm in ("gate", "base"):
        out[arm] = {}
        for s in SEEDS:
            d = os.path.join(RESULT_ROOT, f"T135_full_{arm}_s{s}", "nam")
            hits = glob.glob(os.path.join(d, "*", "backtest_metrics.json"))
            hits = [h for h in hits if tag in h]
            if not hits:
                print(f"  [warn] {d} 无 tag={tag} 产物")
                continue
            latest = max(hits, key=os.path.getmtime)
            m = json.load(open(latest, encoding="utf-8"))
            nullp = os.path.join(os.path.dirname(latest), "random_null.json")
            null = {}
            if os.path.exists(nullp):
                null = json.load(open(nullp, encoding="utf-8"))
            out[arm][s] = {
                "ret": m.get("total_return_pct"),
                "dd": m.get("max_drawdown"),
                "sharpe": m.get("sharpe_ratio"),
                "trades": m.get("total_trades"),
                "pct": null.get("percentile_of_actual"),
                "z": null.get("z_score"),
                "p": null.get("p_value_one_sided"),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="bear", choices=["bear", "bull"])
    args = ap.parse_args()
    tag = f"t135_{args.window}"
    print(f"=== T135 配对分析（窗口 {args.window}，tag={tag}）===")
    r = find_metrics(tag)

    import numpy as np
    print(f"\n{'seed':<6} | {'base ret':>8} {'gate ret':>8} {'Δret':>7} | "
          f"{'base pct':>8} {'gate pct':>8} {'Δpct':>7} | {'base z':>6} {'gate z':>6} {'Δz':>6}")
    print("-" * 88)
    rets_b, rets_g, dds_b, dds_g, pcts_b, pcts_g, zs_b, zs_g = (
        [], [], [], [], [], [], [], [])
    for s in SEEDS:
        if s not in r["base"] or s not in r["gate"]:
            print(f"  {s}: 缺产物")
            continue
        b, g = r["base"][s], r["gate"][s]
        dr = (g["ret"] or 0) - (b["ret"] or 0)
        dp = (g["pct"] or 0) - (b["pct"] or 0)
        dz = (g["z"] or 0) - (b["z"] or 0)
        rets_b.append(b["ret"]); rets_g.append(g["ret"])
        dds_b.append(b["dd"]); dds_g.append(g["dd"])
        pcts_b.append(b["pct"]); pcts_g.append(g["pct"])
        zs_b.append(b["z"]); zs_g.append(g["z"])
        print(f"{s:<6} | {b['ret']:>7.2f}% {g['ret']:>7.2f}% {dr:>+6.2f} | "
              f"{b['pct']:>7.1f} {g['pct']:>7.1f} {dp:>+6.1f} | "
              f"{b['z']:>6.2f} {g['z']:>6.2f} {dz:>+6.2f}")
    if len(rets_b) == len(SEEDS):
        a, b = np.array, np.array
        rb, rg, db, dg, pb, pg, zb, zg = map(a, (rets_b, rets_g, dds_b, dds_g,
                                                  pcts_b, pcts_g, zs_b, zs_g))
        print("-" * 88)
        print(f"收益 Δ 均值 {np.mean(rg - rb):+.2f}pp, 改善 {int((rg > rb).sum())}/{len(SEEDS)}")
        print(f"回撤 Δ 均值 {np.mean(dg - db):+.2f}pp, 改善 {int((dg > db).sum())}/{len(SEEDS)}")
        print(f"分位 Δ 均值 {np.mean(pg - pb):+.1f},  改善 {int((pg > pb).sum())}/{len(SEEDS)}")
        print(f"z    Δ 均值 {np.mean(zg - zb):+.2f},  改善 {int((zg > zb).sum())}/{len(SEEDS)}")
        print(f"\nσ_seed（收益 pp）: base {np.std(rb):.2f} → gate {np.std(rg):.2f}")
        print(f"σ_seed（分位）  : base {np.std(pb):.2f} → gate {np.std(pg):.2f}")
        print(f"σ_seed（z）     : base {np.std(zb):.2f} → gate {np.std(zg):.2f}")


if __name__ == "__main__":
    main()
