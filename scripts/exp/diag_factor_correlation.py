# -*- coding: utf-8 -*-
"""
Task #21 因子去相关分析（phase1-factor-eng）
============================================
量化 228 个模型因子（feature_names.json 顺序）的真实独立维度：
  1. 横截面 rank 归一化后的相关矩阵（Spearman）
  2. 特征值谱 + 累计方差解释率（多少维解释 90%/95%）
  3. 高相关对 |rho|>0.9 的连通聚类分组（union-find）
  4. 每聚类输出"代表因子"（该组内与组外总相关度最低者）
  5. 恒定/近恒定列检测

抽样：800 只股票 × 近 N 年（默认 3 年），训练段横截面 rank 后 pooling 算相关。
用法：python scripts/exp/diag_factor_correlation.py [--years 3] [--n-stocks 800] [--out diagnose_output/factor_corr_report.json]
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CACHE = os.path.join("database", "system_data", "factors_cache")
FEATURES_JSON = os.path.join("models", "nam_gate", "sweepA_base_s42", "feature_names.json")
OUT_DEFAULT = os.path.join("diagnose_output", "factor_corr_report.json")


def load_sample(n_stocks: int, years: int):
    """读 n_stocks 只股票近 years 年的因子列，返回 (dates, X) 全样本堆叠。"""
    files = sorted(os.listdir(CACHE))
    files = [f for f in files if f.endswith("_factors.parquet")]
    if n_stocks and len(files) > n_stocks:
        # 均匀抽样代码段，避免只抽到前段上市的老股票
        files = files[:: max(1, len(files) // n_stocks)][:n_stocks]
    feats = json.load(open(FEATURES_JSON, encoding="utf-8"))
    use_cols = ["date"] + feats
    # 统一读出全部（内存可控：~800 股 × 4133 日 × 229 列 float32）
    frames, min_date = [], None
    t0 = time.time()
    for i, f in enumerate(files):
        df = pd.read_parquet(os.path.join(CACHE, f), columns=use_cols)
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)
        if min_date is None or df["date"].min() > min_date:
            min_date = df["date"].min()
        if (i + 1) % 200 == 0:
            print(f"  [load] {i+1}/{len(files)} 只, 已用 {time.time()-t0:.0f}s", flush=True)
    print(f"  [load] 读入 {len(frames)} 只, 已用 {time.time()-t0:.0f}s", flush=True)
    all_df = pd.concat(frames, ignore_index=True)
    del frames
    cutoff = all_df["date"].max() - pd.DateOffset(years=years)
    all_df = all_df[all_df["date"] >= cutoff]
    print(f"  [load] 裁剪至 {cutoff.date()} 之后: {len(all_df)} 行", flush=True)
    dates = all_df["date"].values
    X = all_df[feats].to_numpy(dtype=np.float32)
    n_stocks_used = len(files)
    del all_df
    return dates, X, feats, n_stocks_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--n-stocks", type=int, default=800)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--rho-thresh", type=float, default=0.90)
    args = ap.parse_args()

    print(f"[T21] 加载抽样: {args.n_stocks} 只 × {args.years} 年")
    dates, X, feats, n_stocks_used = load_sample(args.n_stocks, args.years)

    # ---- 横截面 rank 归一化（与训练端一致：对当日横截面 rank）----
    from scipy.stats import rankdata
    print("  [rank] 逐日横截面 rank...", flush=True)
    order = np.argsort(dates, kind="stable")
    dates_sorted = dates[order]
    Xs = X[order]
    # 按日分块
    split_at = np.where(np.diff(dates_sorted.astype("datetime64[D]").view("int64")) != 0)[0] + 1
    blocks = np.split(Xs, split_at)
    Xr = np.empty_like(Xs)
    pos = 0
    for b in blocks:
        r = rankdata(b, axis=0) / (b.shape[0] + 1)
        Xr[pos:pos + b.shape[0]] = r
        pos += b.shape[0]
    n_days = len(blocks)
    del blocks
    print(f"  [rank] 完成: {n_days} 个交易日", flush=True)

    # 剔除恒定列（近恒定: 横截面 rank 标准差 < 1e-6 视为无信息）
    std = Xr.std(axis=0)
    const_idx = np.where(std < 1e-6)[0]
    if len(const_idx):
        print(f"  [warn] {len(const_idx)} 个近恒定列: {[feats[i] for i in const_idx]}")

    # ---- 相关矩阵（Spearman ≈ 对 rank 的 Pearson）----
    print("  [corr] 计算相关矩阵...", flush=True)
    C = np.corrcoef(Xr.T)          # [n_feat, n_feat]
    np.fill_diagonal(C, 0.0)
    del Xr
    n = C.shape[0]

    # 1) 特征值谱
    evals = np.linalg.eigvalsh(C)
    evals = evals[::-1]
    evals = np.clip(evals, 0, None)
    total = evals.sum()
    cum = np.cumsum(evals) / total
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    n99 = int(np.searchsorted(cum, 0.99) + 1)
    npos = int((evals > 1e-9).sum())
    print(f"\n[谱] 有效特征值(>0): {npos}/{n}")
    print(f"[谱] 累计解释 90%: {n90} 维 | 95%: {n95} 维 | 99%: {n99} 维")

    # 2) 高相关对 + 连通聚类
    tri = np.triu(np.abs(C) >= args.rho_thresh, k=1)
    pairs = np.argwhere(tri)
    # union-find
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for a, b in pairs:
        union(a, b)
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    clusters = sorted(comps.values(), key=len, reverse=True)
    clusters = [c for c in clusters if len(c) > 1]
    n_clusters = len(clusters)
    n_in_cluster = sum(len(c) for c in clusters)
    print(f"\n[聚类] |rho|>={args.rho_thresh} 的连通组: {n_clusters} 组, 涉及 {n_in_cluster}/{n} 个因子")
    for ci, c in enumerate(clusters[:15]):
        names = [feats[i] for i in c]
        print(f"  C{ci} (n={len(c)}): {names[:6]}{'...' if len(c) > 6 else ''}")

    # 3) 每聚类代表因子：组内|rho|均值最低者（最独立）
    reps = {}
    for ci, c in enumerate(clusters):
        sub = C[np.ix_(c, c)]
        within = np.abs(sub).mean(axis=1)
        reps[ci] = feats[c[int(np.argmin(within))]]
    print("\n[代表因子] " + ", ".join(f"C{ci}:{r}" for ci, r in list(reps.items())[:15]))

    # 4) 汇总统计
    absC = np.abs(C)
    n_hi = int((absC > args.rho_thresh).sum() // 2)
    print(f"\n[汇总] |rho|>{args.rho_thresh} 的因子对数: {n_hi}")
    print(f"[汇总] 平均 |rho|: {absC.mean():.4f} | 中位 |rho|: {np.median(absC):.4f}")
    # 最强的 20 对
    flat = [(absC[i, j], i, j) for i in range(n) for j in range(i + 1, n)]
    flat.sort(reverse=True)
    print("\n[最强相关对 TOP20]")
    for v, i, j in flat[:20]:
        print(f"  {feats[i]} ~ {feats[j]}: rho={C[i, j]:+.3f}")

    # ---- 保存报告 ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    report = {
        "n_stocks": int(n_stocks_used),
        "n_days": int(n_days),
        "n_features": int(n),
        "years": args.years,
        "eigen": {
            "n_pos": int(npos),
            "cum_var": {f"top{k}": float(cum[k - 1]) for k in (1, 5, 10, 20, 40, 60, 80, 100, n90, n95, n99)},
            "n90": int(n90), "n95": int(n95), "n99": int(n99),
        },
        "clusters": {
            "n_clusters": int(n_clusters),
            "n_involved": int(n_in_cluster),
            "groups": [{"n": len(c), "names": [feats[i] for i in c],
                        "representative": reps[ci]} for ci, c in enumerate(clusters)],
        },
        "summary": {
            "n_pairs_hi": int(n_hi),
            "mean_abs_rho": float(absC.mean()),
            "median_abs_rho": float(np.median(absC)),
            "top20_pairs": [{"a": feats[i], "b": feats[j], "rho": float(C[i, j])} for _, i, j in flat[:20]],
        },
        "constant_cols": [feats[i] for i in const_idx],
        "feature_names": feats,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n[out] 报告已写: {args.out}")


if __name__ == "__main__":
    main()
