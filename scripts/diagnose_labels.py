"""标签构造诊断脚本：验证正交性与无数据泄露。

用法：
    python scripts/diagnose_labels.py

无需外部数据：内置合成独立收益序列，演示
  1) 嵌套累积收益（旧目标）共线性强；
  2) 非重叠前向收益腿（新）近似正交；
  3) 横截面 Gram-Schmidt 残差化后目标近似正交；
  4) verify_no_lookahead 防泄露闸门通过。

若传入 --parquet 指向含 date/open/high/low/close/volume 的 parquet，
则直接对真实数据做诊断。
"""
from __future__ import annotations

import argparse
import os
import sys

# 允许以脚本方式直接运行：将项目根目录加入模块搜索路径。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core.factors.multi_objective_labels import (
    MultiObjectiveLabelBuilder,
    orthogonalize_labels,
    diagnose_label_orthogonality,
)


def _synth_universe(n_stocks: int = 80, n_days: int = 400, seed: int = 2024):
    rng = np.random.default_rng(seed)
    stocks: dict[str, pd.DataFrame] = {}
    for s in range(n_stocks):
        rets = rng.normal(0.0004, 0.02, size=n_days)
        close = 10.0 * np.cumprod(1.0 + rets)
        open_ = np.concatenate([[10.0], close[:-1]])
        stocks[f"{s:06d}.SZ"] = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n_days).astype(str),
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 10_000, n_days).astype(float),
            "amount": rng.integers(1_000_000, 10_000_000, n_days).astype(float),
        })
    return stocks


def _report(title: str, diag: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"  截面日数 n_groups      : {diag['n_groups']}")
    print(f"  最大非对角相关 |corr|  : {diag['max_abs_offdiag_corr']:.4f}")
    print(f"  相关矩阵条件数 cond    : {diag['condition_number']:.2f}")
    if diag["condition_number"] > 30:
        print("  ⚠ 强共线（条件数 > 30），目标需正交化")
    else:
        print("  ✓ 近似正交（条件数 <= 30）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=None, help="可选：真实 OHLCV parquet 路径")
    args = ap.parse_args()

    if args.parquet:
        df = pd.read_parquet(args.parquet)
        stocks = {str(k): g for k, g in df.groupby("code")} if "code" in df else {"X": df}
    else:
        print("[diagnose] 使用内置合成独立收益数据（无外部依赖）")
        stocks = _synth_universe()

    builder = MultiObjectiveLabelBuilder(
        (5, 20, 60), risk_horizon=20,
        orthogonal_legs=True, use_matching_risk_for_sharpe=True,
    )
    universe = builder.build_universe(stocks)

    # 1) 旧目标：嵌套累积收益
    nested = ["y_ret_5d", "y_ret_20d", "y_ret_60d", "y_mdd_20d"]
    _report("旧目标（嵌套累积收益，作为对照）", diagnose_label_orthogonality(universe, nested))

    # 2) 新目标：非重叠前向收益腿
    legs = ["y_ret_leg_1_5d", "y_ret_leg_6_20d", "y_ret_leg_21_60d"]
    _report("新目标（非重叠前向收益腿）", diagnose_label_orthogonality(universe, legs))

    # 3) Gram-Schmidt 残差化
    obj_cols = ["y_ret_5d", "y_ret_20d", "y_ret_60d", "y_mdd_20d", "y_downvol_20d", "y_illiq_20d"]
    orth = orthogonalize_labels(universe, obj_cols)
    orth_cols = [f"orth_{c}" for c in obj_cols]
    _report("Gram-Schmidt 残差化后（orth_*）", diagnose_label_orthogonality(orth, orth_cols))

    # 4) 防泄露闸门
    sample_code = next(iter(stocks))
    ok = builder.verify_no_lookahead(stocks[sample_code])
    print(f"\n=== 防泄露闸门 verify_no_lookahead ===")
    print(f"  {'✓ 通过' if ok else '✗ 失败'}：标签 t 不依赖 close[t]/open[t]（仅引用 t+1 起未来数据）")

    print("\n[diagnose] 结论：开启 MULTI_OBJECTIVE_ORTHOGONAL_LEGS（默认）即获得正交收益腿；"
          "置 MULTI_OBJECTIVE_ORTHOGONALIZE=True 可对任意目标集做横截面正交化（需重新训练）。")


if __name__ == "__main__":
    main()
