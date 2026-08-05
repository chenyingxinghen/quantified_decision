"""使用时间外推验证搜索旧模型与当前模型的预测 rank 融合权重。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _evaluate(frame: pd.DataFrame, candidate_weight: float) -> dict:
    prediction = (
        candidate_weight * frame["candidate_rank"].to_numpy()
        + (1.0 - candidate_weight) * frame["baseline_rank"].to_numpy()
    )
    future_return = frame["return"].to_numpy()
    dates = frame["date"].to_numpy()
    _, group_starts, group_counts = np.unique(
        dates, return_index=True, return_counts=True
    )
    rank_ics = []
    top5_excess = []
    for start, count in zip(group_starts, group_counts):
        day_prediction = prediction[start:start + count]
        day_return = future_return[start:start + count]
        if len(day_return) < 10:
            continue
        ic = spearmanr(day_prediction, day_return).statistic
        if np.isfinite(ic):
            rank_ics.append(float(ic))
        top_indices = np.argpartition(day_prediction, -5)[-5:]
        top5_excess.append(float(day_return[top_indices].mean() - day_return.mean()))
    rank_ics = np.asarray(rank_ics)
    ic_std = float(rank_ics.std())
    return {
        "candidate_weight": candidate_weight,
        "days": int(len(rank_ics)),
        "rank_ic": float(rank_ics.mean()),
        "rank_ic_std": ic_std,
        "rank_ic_ir": float(rank_ics.mean() / ic_std) if ic_std > 0 else 0.0,
        "positive_ic_ratio": float((rank_ics > 0).mean()),
        "top5_excess": float(np.mean(top5_excess)),
    }


def _select_weight(results: list[dict], ic_tolerance: float, ir_tolerance: float) -> dict:
    candidate = next(row for row in results if row["candidate_weight"] == 1.0)
    eligible = [
        row for row in results
        if row["rank_ic"] >= candidate["rank_ic"] - ic_tolerance
        and row["rank_ic_ir"] >= candidate["rank_ic_ir"] - ir_tolerance
    ]
    return max(eligible, key=lambda row: (row["top5_excess"], row["rank_ic_ir"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="models/diagnostics/candidate_comparison_20240804/prediction_ranks.parquet",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--embargo-days", type=int, default=7)
    parser.add_argument("--ic-tolerance", type=float, default=0.002)
    parser.add_argument("--ir-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    frame = pd.read_parquet(args.predictions)
    frame["date"] = pd.to_datetime(frame["date"])
    required = {"date", "return", "baseline_rank", "candidate_rank"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"预测文件缺少列: {sorted(missing)}")

    dates = np.sort(frame["date"].unique())
    split_index = len(dates) // 2
    select_dates = dates[:split_index]
    confirm_dates = dates[min(split_index + args.embargo_days, len(dates)):]
    select_frame = frame[frame["date"].isin(select_dates)]
    confirm_frame = frame[frame["date"].isin(confirm_dates)]
    weights = np.round(np.arange(0.0, 1.0 + args.step / 2, args.step), 8)

    selection_results = [_evaluate(select_frame, float(weight)) for weight in weights]
    selected = _select_weight(selection_results, args.ic_tolerance, args.ir_tolerance)
    confirmation_results = [_evaluate(confirm_frame, float(weight)) for weight in weights]
    confirmation_selected = next(
        row for row in confirmation_results
        if row["candidate_weight"] == selected["candidate_weight"]
    )
    confirmation_candidate = next(
        row for row in confirmation_results if row["candidate_weight"] == 1.0
    )

    payload = {
        "protocol": {
            "selection_start": str(select_dates[0])[:10],
            "selection_end": str(select_dates[-1])[:10],
            "confirmation_start": str(confirm_dates[0])[:10],
            "confirmation_end": str(confirm_dates[-1])[:10],
            "embargo_days": args.embargo_days,
            "ic_tolerance": args.ic_tolerance,
            "ir_tolerance": args.ir_tolerance,
            "weight_step": args.step,
        },
        "selected_weight": selected["candidate_weight"],
        "selection_results": selection_results,
        "confirmation_selected": confirmation_selected,
        "confirmation_candidate": confirmation_candidate,
        "confirmation_results": confirmation_results,
    }
    output = Path(args.output or Path(args.predictions).with_name("rank_blend_nested.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"选择期: {payload['protocol']['selection_start']} ~ {payload['protocol']['selection_end']}; "
        f"确认期: {payload['protocol']['confirmation_start']} ~ {payload['protocol']['confirmation_end']}"
    )
    print(f"{'w_candidate':>11} | {'Select IC':>9} | {'Select IR':>9} | {'Select Ex':>9} | {'Confirm IC':>10} | {'Confirm IR':>10} | {'Confirm Ex':>10}")
    for select, confirm in zip(selection_results, confirmation_results):
        marker = " *" if select["candidate_weight"] == selected["candidate_weight"] else ""
        print(
            f"{select['candidate_weight']:>11.1f} | {select['rank_ic']:>9.4f} | "
            f"{select['rank_ic_ir']:>9.3f} | {select['top5_excess']:>9.2%} | "
            f"{confirm['rank_ic']:>10.4f} | {confirm['rank_ic_ir']:>10.3f} | "
            f"{confirm['top5_excess']:>10.2%}{marker}"
        )
    print(f"选择权重: candidate={selected['candidate_weight']:.1f}, baseline={1-selected['candidate_weight']:.1f}")
    print(
        f"确认期相对纯 candidate: IC {confirmation_selected['rank_ic'] - confirmation_candidate['rank_ic']:+.4f}, "
        f"ICIR {confirmation_selected['rank_ic_ir'] - confirmation_candidate['rank_ic_ir']:+.3f}, "
        f"Top5Ex {confirmation_selected['top5_excess'] - confirmation_candidate['top5_excess']:+.2%}"
    )


if __name__ == "__main__":
    main()
