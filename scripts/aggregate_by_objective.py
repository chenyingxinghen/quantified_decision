"""Aggregate the 6 per-objective backtests into a single comparison table.

Reads backtest_result/by_objective/<obj>_hold<days>/backtest_metrics.json for each
training objective and prints a comparative summary plus a CSV.
"""
import json
import os

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "backtest_result", "by_objective")

# (objective_key, label, fixed_hold_days, label_horizon_used)
RUNS = [
    ("rank_y_ret_5d",   "收益5日",   5),
    ("rank_y_ret_20d",  "收益20日",  20),
    ("rank_y_ret_60d",  "收益60日",  60),
    ("rank_y_mdd_20d",  "回撤20日",  20),
    ("rank_y_downvol_20d", "下行波动20日", 20),
    ("rank_y_illiq_20d", "非流动性20日", 20),
]


def load(obj, days):
    d = os.path.join(BASE, f"{obj}_hold{days}")
    p = os.path.join(d, "backtest_metrics.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = []
    for obj, label, days in RUNS:
        m = load(obj, days)
        if m is None:
            rows.append((obj, label, days, None))
            continue
        rows.append((obj, label, days, m))

    # Header
    print("=" * 92)
    print("按训练目标分别回测（固定持仓=标签周期，时间止损强制在周期末退出）")
    print("=" * 92)
    hdr = f"{'目标':<16}{'持仓':>5}{'交易':>5}{'胜率%':>8}{'总收益%':>9}{'夏普':>8}{'最大回撤%':>10}{'盈亏比':>8}"
    print(hdr)
    print("-" * 92)

    csv_lines = ["objective,label,hold_days,total_trades,win_rate_pct,total_return_pct,sharpe_ratio,max_drawdown_pct,profit_factor,exit_reasons"]
    for obj, label, days, m in rows:
        if m is None:
            print(f"{label:<16}{days:>5}{'-':>5}  (未生成结果)")
            csv_lines.append(f"{obj},{label},{days},,,,,")
            continue
        # 注意：metrics.json 中 win_rate / max_drawdown / total_return_pct 已是百分比值
        wr = float(m.get("win_rate", 0))
        tr = float(m.get("total_return_pct", 0))
        sh = float(m.get("sharpe_ratio", 0))
        mdd = float(m.get("max_drawdown", 0))
        pf = float(m.get("profit_factor", 0))
        nt = int(m.get("total_trades", 0))
        print(f"{label:<14}{days:>5}{nt:>5}{wr:>8.1f}{tr:>9.2f}{sh:>8.2f}{mdd:>10.1f}{pf:>8.2f}")
        er = m.get("exit_reasons", {})
        er_s = ";".join(f"{k}={v}" for k, v in er.items())
        csv_lines.append(f"{obj},{label},{days},{nt},{wr:.2f},{tr:.2f},{sh:.4f},{mdd:.2f},{pf:.4f},{er_s}")

    print("-" * 92)
    # Exit reason summary
    print("\n退出原因分布：")
    for obj, label, days, m in rows:
        if m is None:
            continue
        er = m.get("exit_reasons", {})
        er_s = ", ".join(f"{k}:{v}" for k, v in er.items())
        print(f"  {label:<14} ({days}d) -> {er_s}")

    # Verdict
    print("\n初步判断（盈亏比 >1 且 总收益>0 视为有独立样本外信号）：")
    for obj, label, days, m in rows:
        if m is None:
            print(f"  {label:<14} -> 无数据")
            continue
        pf = m.get("profit_factor", 0)
        tr = m.get("total_return_pct", 0)
        ok = (pf > 1.0) and (tr > 0)
        print(f"  {label:<14} -> {'有信号' if ok else '弱/无信号'}  (盈亏比={pf:.2f}, 总收益={tr:.1f}%)")

    out_csv = os.path.join(BASE, "comparison.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))
    print(f"\n已写出对比 CSV: {out_csv}")


if __name__ == "__main__":
    main()
