# -*- coding: utf-8 -*-
"""
T133 配对分析：T133_pca30（PCA 正交压缩） vs T045 base（228 因子）
=====================================================================
T069 协议：同种子配对 + 熊市主判窗 + 随机零假设分位/z 口径。
核心目标：
  1. 熊市随机分位是否保持（base 中位 97.2，T132 不能掉）
  2. σ_seed 是否下降（T21/T20 立论：去冗余 → 等价解空间小 → 种子抖动小）
  3. 收益/回撤/夏普配对 Δ 方向

用法：python scripts/exp/analyze_t132.py
"""
import glob
import json
import os

RESULT_ROOT = "backtest_result"
WINDOW = "2022-09-05_to_2024-08-05"
SEEDS = ["42", "11", "23", "37"]

BASE_DIRS = {s: f"backtest_result/sweepA_base_s{s}/nam" for s in SEEDS}
T133_DIRS = {s: f"backtest_result/T133_pca30_s{s}/nam" for s in SEEDS}


def find_metrics(dirs, tag=None):
    out = {}
    for s in SEEDS:
        base = dirs[s]
        hits = glob.glob(os.path.join(base, "*", "backtest_metrics.json"))
        # 排除投票共识（ens4）等 ensemble 产物，避免拿错目录
        hits = [h for h in hits if "ens4" not in h and "ens-" not in h]
        if tag:
            hits = [h for h in hits if tag in h]
        if not hits:
            print(f"  [warn] {base} 无回测产物")
            continue
        # 优先 r4verify（与 sweepA 数值逐位一致，且同为批量回测产物）
        r4 = [h for h in hits if "r4verify" in h]
        latest = max(r4 if r4 else hits, key=os.path.getmtime)
        m = json.load(open(latest, encoding="utf-8"))
        nullp = os.path.join(os.path.dirname(latest), "random_null.json")
        null = json.load(open(nullp, encoding="utf-8")) if os.path.exists(nullp) else {}
        out[s] = {
            "dir": os.path.dirname(latest),
            "ret_pct": m.get("total_return_pct"),
            "dd_pct": m.get("max_drawdown"),
            "sharpe": m.get("sharpe_ratio"),
            "win_rate": m.get("win_rate"),
            "trades": m.get("total_trades"),
            "null_pct": null.get("percentile_of_actual"),
            "null_z": null.get("z_score"),
            "null_p": null.get("p_value_one_sided"),
        }
    return out


def main():
    print("=== 寻找回测产物 ===")
    base = find_metrics(BASE_DIRS)
    t133 = find_metrics(T133_DIRS)

    print(f"\n{'seed':<6} | {'base ret':>9} {'t133 ret':>9} {'Δret':>8} | "
          f"{'base dd':>8} {'t133 dd':>8} {'Δdd':>7} | "
          f"{'base pct':>9} {'t133 pct':>9} {'Δpct':>7} | "
          f"{'base z':>7} {'t133 z':>7}")
    print("-" * 105)
    rets_b, rets_t, dds_b, dds_t, pcts_b, pcts_t, zs_b, zs_t = (
        [], [], [], [], [], [], [], [])
    for s in SEEDS:
        if s not in base or s not in t133:
            print(f"  {s}: 缺产物")
            continue
        b, t = base[s], t133[s]
        dr = (t["ret_pct"] or 0) - (b["ret_pct"] or 0)
        dd = (t["dd_pct"] or 0) - (b["dd_pct"] or 0)
        dp = (t["null_pct"] or 0) - (b["null_pct"] or 0)
        dz = (t["null_z"] or 0) - (b["null_z"] or 0)
        rets_b.append(b["ret_pct"]); rets_t.append(t["ret_pct"])
        dds_b.append(b["dd_pct"]); dds_t.append(t["dd_pct"])
        pcts_b.append(b["null_pct"]); pcts_t.append(t["null_pct"])
        zs_b.append(b["null_z"]); zs_t.append(t["null_z"])
        print(f"{s:<6} | {b['ret_pct']:>8.2f}% {t['ret_pct']:>8.2f}% {dr:>+7.2f} | "
              f"{b['dd_pct']:>7.2f}% {t['dd_pct']:>7.2f}% {dd:>+6.2f} | "
              f"{b['null_pct']:>8.1f} {t['null_pct']:>8.1f} {dp:>+6.1f} | "
              f"{b['null_z']:>6.2f} {t['null_z']:>6.2f}")

    import numpy as np
    if len(rets_b) == len(SEEDS):
        print("-" * 105)
        print(f"收益 Δ 均值 {np.mean(np.array(rets_t)-np.array(rets_b)):+.2f}pp, "
              f"改善 {sum(1 for i in range(len(SEEDS)) if rets_t[i]>rets_b[i])}/{len(SEEDS)}")
        print(f"回撤 Δ 均值 {np.mean(np.array(dds_t)-np.array(dds_b)):+.2f}pp, "
              f"改善 {sum(1 for i in range(len(SEEDS)) if dds_t[i]>dds_b[i])}/{len(SEEDS)}")
        print(f"分位 Δ 均值 {np.mean(np.array(pcts_t)-np.array(pcts_b)):+.1f}, "
              f"改善 {sum(1 for i in range(len(SEEDS)) if pcts_t[i]>pcts_b[i])}/{len(SEEDS)}")
        # σ_seed 对比（熊市收益/分位标准差）
        print(f"\nσ_seed（熊市收益 pp）: base {np.std(rets_b):.2f} → t133 {np.std(rets_t):.2f}")
        print(f"σ_seed（熊市分位）  : base {np.std(pcts_b):.2f} → t133 {np.std(pcts_t):.2f}")
        print(f"σ_seed（熊市 z）    : base {np.std(zs_b):.2f} → t133 {np.std(zs_t):.2f}")


if __name__ == "__main__":
    main()
