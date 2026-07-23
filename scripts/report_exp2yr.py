"""2年训练实验对比报告 (exp2yr)

对比 回测1 (样本内, 同2年窗口) 与 回测2 (其后1年, 前向样本外)：
  - 集成预测 (combined, 加权多目标) 在两段的表现
  - 6 个分目标预测 (per-objective, 固定持仓=标签周期) 在两段的表现

用途：诊断模型各目标的预测力 / 泛化能力 —— 样本内有效但样本外失效的目标
      说明过拟合，需要在集成权重里下调或剔除。

用法：
  python -u scripts/report_exp2yr.py
"""
import json
import os

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "backtest_result", "exp2yr")

# (objective_key, 中文标签, 固定持仓天数)
RUNS = [
    ("rank_y_ret_5d",      "收益5日",   5),
    ("rank_y_ret_20d",     "收益20日",  20),
    ("rank_y_ret_60d",     "收益60日",  60),
    ("rank_y_mdd_20d",     "回撤20日",  20),
    ("rank_y_downvol_20d", "下行波动20日", 20),
    ("rank_y_illiq_20d",   "非流动性20日", 20),
]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def m_of(m):
    """提取关键指标元组，缺失返回 None 占位。"""
    if m is None:
        return (None, None, None, None, None, None)
    return (
        int(m.get("total_trades", 0)),
        float(m.get("win_rate", 0)),           # 已是百分比
        float(m.get("total_return_pct", 0)),   # 已是百分比
        float(m.get("sharpe_ratio", 0)),
        float(m.get("max_drawdown", 0)),       # 已是百分比(负)
        float(m.get("profit_factor", 0)),
    )


def fmt(v, nd=2, suffix=""):
    if v is None:
        return "   -  "
    return f"{v:.{nd}f}{suffix}"


def main():
    b1_int = load(os.path.join(BASE, "backtest1", "integrated", "backtest_metrics.json"))
    b2_int = load(os.path.join(BASE, "backtest2", "integrated", "backtest_metrics.json"))

    b1_obj = {obj: load(os.path.join(BASE, "backtest1", "by_objective",
                                     f"{obj}_hold{days}", "backtest_metrics.json"))
              for obj, _, days in RUNS}
    b2_obj = {obj: load(os.path.join(BASE, "backtest2", "by_objective",
                                     f"{obj}_hold{days}", "backtest_metrics.json"))
              for obj, _, days in RUNS}

    csv_lines = ["section,key,label,hold_days,"
                 "b1_trades,b1_win_pct,b1_ret_pct,b1_sharpe,b1_mdd_pct,b1_pf,"
                 "b2_trades,b2_win_pct,b2_ret_pct,b2_sharpe,b2_mdd_pct,b2_pf,"
                 "oos_minus_insample_ret_pct,generalizes"]

    def section(title, label, days, m1, m2, key):
        t1, w1, r1, s1, d1, p1 = m_of(m1)
        t2, w2, r2, s2, d2, p2 = m_of(m2)
        print(f"\n{'='*100}")
        print(title)
        print(f"{'':<18}{'交易':>6}{'胜率%':>9}{'总收益%':>10}{'夏普':>8}{'最大回撤%':>11}{'盈亏比':>8}")
        print("-"*100)
        if None in (r1, r2):
            print(f"{'样本内(回测1)':<16}{fmt(t1,0):>6}{fmt(w1):>9}{fmt(r1):>10}{fmt(s1):>8}{fmt(d1):>11}{fmt(p1):>8}")
            print(f"{'样本外(回测2)':<16}{fmt(t2,0):>6}{fmt(w2):>9}{fmt(r2):>10}{fmt(s2):>8}{fmt(d2):>11}{fmt(p2):>8}")
            print("  (其中一段缺数据)")
            return
        print(f"{'样本内(回测1)':<16}{t1:>6}{w1:>9.1f}{r1:>10.2f}{s1:>8.2f}{d1:>11.1f}{p1:>8.2f}")
        print(f"{'样本外(回测2)':<16}{t2:>6}{w2:>9.1f}{r2:>10.2f}{s2:>8.2f}{d2:>11.1f}{p2:>8.2f}")
        delta = r2 - r1
        gen = (p2 > 1.0) and (r2 > 0)
        print(f"{'样本外-样本内':<16}{'':>6}{'':>9}{delta:>+10.2f}")
        print(f"  -> 泛化判断: {'泛化(样本外仍正收益且盈亏比>1) ✓' if gen else '未泛化(样本外失效/过拟合) ✗'}")
        csv_lines.append(",".join(str(x) for x in [
            key, label, days, t1, fmt(w1,2), fmt(r1,2), fmt(s1,4), fmt(d1,2), fmt(p1,4),
            t2, fmt(w2,2), fmt(r2,2), fmt(s2,4), fmt(d2,2), fmt(p2,4),
            f"{delta:.2f}", "yes" if gen else "no"]))

    print("#"*100)
    print("# 2年训练实验对比报告：样本内(回测1) vs 前向样本外(回测2)")
    print("#"*100)

    section("【集成预测 combined】加权多目标模型", "集成", "-", b1_int, b2_int, "combined")

    for obj, label, days in RUNS:
        section(f"【分目标 per-objective】{label} (固定持仓 {days} 日)",
                label, days, b1_obj[obj], b2_obj[obj], obj)

    # 综合结论：哪些目标在样本外仍有效
    print("\n" + "="*100)
    print("泛化汇总（样本外盈亏比>1 且 总收益>0 视为有效）")
    print("-"*100)
    print(f"{'目标':<16}{'样本内收益%':>12}{'样本外收益%':>12}{'样本外盈亏比':>12}{'结论':>8}")
    any_gen = False
    for obj, label, days in RUNS:
        _, _, r1, _, _, p1 = m_of(b1_obj[obj])
        _, _, r2, _, _, p2 = m_of(b2_obj[obj])
        if None in (r2, p2):
            print(f"{label:<16}{fmt(r1):>12}{fmt(r2):>12}{fmt(p2,4):>12}  缺数据")
            continue
        gen = (p2 > 1.0) and (r2 > 0)
        any_gen = any_gen or gen
        print(f"{label:<16}{fmt(r1):>12}{fmt(r2):>12}{fmt(p2,4):>12}  {'有效✓' if gen else '失效✗'}")
    # 集成
    _, _, r1i, _, _, p1i = m_of(b1_int)
    _, _, r2i, _, _, p2i = m_of(b2_int)
    geni = (p2i > 1.0) and (r2i > 0) if None not in (r2i, p2i) else False
    print(f"{'集成':<16}{fmt(r1i):>12}{fmt(r2i):>12}{fmt(p2i,4):>12}  {'有效✓' if geni else '失效✗'}")
    print("-"*100)
    print("结论: " + ("至少部分目标/集成在样本外仍有效，模型具备一定预测力。" if (any_gen or geni)
          else "样本外普遍失效，需重新检视特征/标签/权重或缩短训练窗口。"))

    out_csv = os.path.join(BASE, "exp2yr_comparison.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))
    print(f"\n已写出对比 CSV: {out_csv}")


if __name__ == "__main__":
    main()
