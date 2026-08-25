# -*- coding: utf-8 -*-
"""
sweep A 配对分析: 同种子 base vs schemeB, 熊市主判, 测 Δ回撤/Δ收益/Δ夏普/Δ随机分位。

配对口径 (T069): 每个种子 S 仅变 treatment (gate 开/关), 其余全同 -> Δ 即 gate 的因果效应估计。
指标方向: 最大回撤为负值, Δdrawdown = sb - base > 0 表示回撤更小(改善);
           Δreturn/Δsharpe/Δpct 同样 > 0 表示 schemeB 更优。
符号检验: 对每指标统计 n 个种子中"改善"的个数 k, 算精确二项两尾 p (H0: k~Binom(n,0.5))。
"""
import os
import sys
import json
import glob
import math

ROOT = r"G:/ai_proj/quantified_decision"
DIAG = os.path.join(ROOT, "diagnose_output", "sweepA")
SEEDS = [42, 11, 23, 37]


def binom_two_sided(k, n):
    """精确二项两尾 p: 偏离 0.5 的最极端尾巴之和。"""
    if n == 0:
        return float("nan")
    p = 0.5
    # 单点概率
    def pmf(x):
        return math.comb(n, x) * (p ** x) * ((1 - p) ** (n - x))
    obs = pmf(k)
    total = 0.0
    for x in range(n + 1):
        if pmf(x) <= obs + 1e-12:
            total += pmf(x)
    return min(total, 1.0)


def main():
    files = sorted(glob.glob(os.path.join(DIAG, "sweepA_results_*.json")))
    if not files:
        print("未找到 sweepA_results_*.json, 先跑 run_sweep_a.py"); sys.exit(1)
    path = files[-1]
    with open(path) as f:
        R = json.load(f)
    print(f"载入: {os.path.basename(path)}\n")

    rows = []
    deltas = {"drawdown": [], "return": [], "sharpe": [], "pct": [], "z": []}
    for s in SEEDS:
        b = R.get(f"base_s{s}", {})
        t = R.get(f"schemeB_s{s}", {})
        bm = b.get("metrics"); tm = t.get("metrics")
        bn = b.get("random_null"); tn = t.get("random_null")
        if not (bm and tm):
            print(f"[缺失] seed {s} metrics 不全, 跳过配对"); continue
        ddd = tm["max_drawdown"] - bm["max_drawdown"]
        dret = tm["total_return_pct"] - bm["total_return_pct"]
        dsh = tm["sharpe_ratio"] - bm["sharpe_ratio"]
        dpct = (tn["percentile_of_actual"] - bn["percentile_of_actual"]
                if (tn and bn) else None)
        dz = (tn["z_score"] - bn["z_score"] if (tn and bn) else None)
        rows.append((s, ddd, dret, dsh, dpct, dz))
        deltas["drawdown"].append(ddd); deltas["return"].append(dret)
        deltas["sharpe"].append(dsh)
        if dpct is not None: deltas["pct"].append(dpct)
        if dz is not None: deltas["z"].append(dz)

    n = len(rows)
    print(f"{'seed':>5} | {'Δ回撤':>8} | {'Δ收益':>8} | {'Δ夏普':>8} | {'Δ分位':>8} | {'Δz':>7}")
    print("-" * 60)
    for r in rows:
        s, ddd, dret, dsh, dpct, dz = r
        print(f"{s:>5} | {ddd:>+8.2f} | {dret:>+8.2f} | {dsh:>+8.3f} | "
              f"{(dpct if dpct is not None else float('nan')):>+8.2f} | "
              f"{(dz if dz is not None else float('nan')):>+7.2f}")

    print("\n=== 配对聚合 (n=%d) ===" % n)
    for name, key in [("回撤", "drawdown"), ("收益", "return"),
                      ("夏普", "sharpe"), ("随机分位", "pct"), ("z", "z")]:
        vals = deltas[key]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        k = sum(1 for v in vals if v > 0)   # 改善的种子数
        p = binom_two_sided(k, len(vals))
        verdict = "schemeB 更优" if k > len(vals) / 2 else "base 更优/持平"
        print(f"{name:>6}: 均值 {mean:>+.3f} | 改善 {k}/{len(vals)} 种子 "
              f"| 符号检验 p={p:.3f} -> {verdict}")

    # 综合判定
    if deltas["drawdown"]:
        kdd = sum(1 for v in deltas["drawdown"] if v > 0)
        n = len(deltas["drawdown"])
        mean_ret = sum(deltas["return"]) / n
        # 随机分位是 T069 主判口径，回撤只是辅助：交叉检查分位方向
        kpct = sum(1 for v in deltas["pct"] if v > 0)
        mean_pct = sum(deltas["pct"]) / n if deltas["pct"] else 0.0
        # 灾难反例：任何种子收益崩 >10pp 或分位崩 >30pp（gate 把好模型毁掉）
        worst_ret = min(deltas["return"])
        worst_pct = min(deltas["pct"]) if deltas["pct"] else 0.0
        has_disaster = (worst_ret < -10.0) or (worst_pct < -30.0)
        print("\n=== 综合判定 (T069 配对) ===")
        print(f"  回撤改善 {kdd}/{n} | 分位改善 {kpct}/{n} (均值 {mean_pct:+.2f}pp) "
              f"| 最差种子 Δ收益 {worst_ret:+.2f}pp / Δ分位 {worst_pct:+.2f}pp")
        if has_disaster:
            print(f"  存在灾难反例 (Δ分位 < -30pp 或 Δ收益 < -10pp) -> gate 可在个别种子"
                  f"上把好模型毁到随机以下，'最坏=base' 设计承诺被打破，不升档")
        elif kdd >= max(3, math.ceil(n * 0.75)) and mean_ret > -3.0 and kpct >= 2:
            print(f"  回撤改善 {kdd}/{n} + 分位改善 {kpct}/{n} + 收益未系统性恶化"
                  f"(均值 {mean_ret:+.2f}pp) -> scheme B 回撤增益为 SYSTEMATIC 防御变体, "
                  f"建议升'防御档'候选")
        elif kdd >= n / 2:
            print(f"  回撤改善 {kdd}/{n} 但分位仅 {kpct}/{n} 改善 -> 增益部分存在但"
                  f"证据弱 (符号检验不显著), 维持候选不晋级")
        else:
            print(f"  回撤仅 {kdd}/{n} 种子改善 -> 增益为噪声/单种子运气, 不晋级, 考虑路径 D")


if __name__ == "__main__":
    main()
