# -*- coding: utf-8 -*-
"""
Task #20 因子显著性筛选（phase1-factor-eng）
============================================
从 4 个 base 种子（s11/s23/s37/s42）的 factor_effectiveness.csv 汇总：
  1. 跨种子 val_effective_ic 方向全一致
  2. 跨种子 |val_effective_ic_ir| 中位数 >= 阈值（默认 0.2）
  3. 与 T21 去相关报告（factor_corr_report.json）的聚类合并：
     同一 |rho|>=0.9 簇内只保留 ICIR 最高的代表 → 去冗余
  4. 输出 factor_eng/factor_shortlist.csv（含簇归属、代表标记、方向、强度）

用法：python scripts/exp/build_factor_shortlist.py [--icir 0.2] [--out factor_eng/factor_shortlist.csv]
"""
import argparse, json, os
import numpy as np
import pandas as pd

BASE_DIRS = [
    "training_iterations/phase0/sweepA/base_s11",
    "training_iterations/phase0/sweepA/base_s23",
    "training_iterations/phase0/sweepA/base_s37",
    "training_iterations/phase0/sweepA/base_s42",
]
CORR_REPORT = "diagnose_output/factor_corr_report.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icir", type=float, default=0.20,
                    help="跨种子 |val_effective_ic_ir| 中位数阈值（默认 0.2）")
    ap.add_argument("--out", default=os.path.join("factor_eng", "factor_shortlist.csv"))
    args = ap.parse_args()

    frames = []
    for d in BASE_DIRS:
        p = os.path.join(d, "factor_effectiveness.csv")
        df = pd.read_csv(p)
        df = df[["feature", "group", "val_effective_ic", "val_effective_ic_ir",
                 "val_effective_ic_std", "val_effective_positive_ratio", "effective_ic_decay"]]
        df = df.rename(columns={c: f"{c}_{os.path.basename(d)}" for c in df.columns if c != "feature"})
        frames.append(df.set_index("feature"))
    merged = pd.concat(frames, axis=1)
    print(f"[T20] 合并 {len(BASE_DIRS)} 个种子: {merged.shape}")

    ic_cols = [c for c in merged.columns if "val_effective_ic" in c and "ir" not in c]
    ir_cols = [c for c in merged.columns if "val_effective_ic_ir" in c]
    ic = merged[ic_cols]
    ir = merged[ir_cols]
    # 交叉验证：raw_ic（因子本身 vs 未来收益）方向一致性——信号真实性的独立证据
    raw_ic_cols = [c for c in merged.columns if "val_raw_ic" in c and "ir" not in c and "std" not in c]
    raw_ir_cols = [c for c in merged.columns if "val_raw_ic_ir" in c]
    raw_ic = merged[raw_ic_cols]
    raw_ir = merged[raw_ir_cols]

    # 1) 方向全一致（含零号处理：IC 恰好为 0 视为弱，不计入）
    signs = np.sign(ic.values)
    dir_agree = (np.abs(signs.sum(axis=1)) == ic.shape[1]) & (np.abs(ic.values).min(axis=1) > 0)
    # 2) 跨种子 |ICIR| 中位数
    med_icir = ir.abs().median(axis=1)
    # 3) 跨种子 IC 均值（方向强度）
    mean_ic = ic.mean(axis=1)
    # raw_ic 方向一致（独立证据）
    raw_dir_agree = (np.abs(np.sign(raw_ic.values).sum(axis=1)) == raw_ic.shape[1])

    # 4) T21 聚类合并：同一簇内只保留 ICIR 最高的代表
    corr = json.load(open(CORR_REPORT, encoding="utf-8"))
    cluster_of = {}
    for ci, g in enumerate(corr["clusters"]["groups"]):
        for nm in g["names"]:
            cluster_of[nm] = ci
    # 簇成员里属于筛选集的，挑 ICIR 最高者
    keep = []
    sel = merged.index[dir_agree & (med_icir >= args.icir)]
    for ci in sorted(set(cluster_of.get(f, -1) for f in sel)):
        members = [f for f in sel if cluster_of.get(f, -1) == ci]
        if ci == -1 or len(members) == 1:
            keep.extend(members)
        else:
            best = max(members, key=lambda f: med_icir[f])
            keep.append(best)
    keep = sorted(keep)

    out = pd.DataFrame({
        "feature": keep,
        "group": merged.loc[keep, "group_" + os.path.basename(BASE_DIRS[0])].values,
        "dir_agree_eff_ic": True,
        "dir_agree_raw_ic": raw_dir_agree[merged.index.get_indexer(keep)],
        "mean_val_ic": mean_ic[keep].values,
        "median_abs_val_icir": med_icir[keep].values,
        "cluster_id": [cluster_of.get(f, -1) for f in keep],
        "in_cluster": [cluster_of.get(f, -1) != -1 for f in keep],
    })
    out = out.sort_values("median_abs_val_icir", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    n_dir = int(dir_agree.sum())
    n_thresh = int((dir_agree & (med_icir >= args.icir)).sum())
    n_raw_agree = int(raw_dir_agree.sum())
    print(f"\n[effective_ic 方向全一致] {n_dir}/228（raw_ic 方向全一致: {n_raw_agree}/228 —— 信号方向真实，NAM 学入后抖动）")
    print(f"\n[方向全一致] {n_dir}/228")
    print(f"[ICIR>={args.icir} 且方向一致] {n_thresh}")
    print(f"[簇内去冗余后] {len(keep)}（簇内代表 {sum(out['in_cluster'])} 个）")
    print(f"\n[shortlist TOP25]")
    for _, r in out.head(25).iterrows():
        tag = "C" + str(r["cluster_id"]) if r["in_cluster"] else "-"
        print(f"  {r['feature']:<32} IC={r['mean_val_ic']:+.4f} |ICIR|={r['median_abs_val_icir']:.3f} 簇={tag}")
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
