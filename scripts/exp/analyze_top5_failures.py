"""Analyze stable point-in-time exposures behind model Top-5 failures."""

import argparse
import json
import os

import numpy as np
import pandas as pd


def _safe_float(value):
    return float(value) if np.isfinite(value) else None


def _period_summary(top: pd.DataFrame, universe: pd.DataFrame) -> dict:
    daily = top.groupby("date", sort=True)["return"].mean().rename("top5_return").to_frame()
    daily["universe_return"] = universe.groupby("date", sort=True)["return"].mean()
    daily["top5_excess"] = daily["top5_return"] - daily["universe_return"]
    result = {
        "days": int(len(daily)),
        "top5_return": _safe_float(daily["top5_return"].mean()),
        "top5_excess": _safe_float(daily["top5_excess"].mean()),
        "positive_excess_ratio": _safe_float((daily["top5_excess"] > 0).mean()),
        "loss_day_ratio": _safe_float((daily["top5_return"] < 0).mean()),
    }
    return result


def _exposure_diagnostics(top: pd.DataFrame, universe: pd.DataFrame) -> dict:
    daily = top.groupby("date", sort=True)["return"].mean().rename("top5_return").to_frame()
    daily["universe_return"] = universe.groupby("date", sort=True)["return"].mean()
    daily["success"] = daily["top5_return"] > daily["universe_return"]
    top = top.join(daily["success"], on="date")

    output = {}
    for column in sorted(c for c in top.columns if c.startswith("rank_")):
        by_day = top.groupby("date", sort=True)[column].mean().to_frame("exposure")
        by_day = by_day.join(daily["success"])
        success = by_day.loc[by_day["success"], "exposure"]
        failure = by_day.loc[~by_day["success"], "exposure"]
        output[column.removeprefix("rank_")] = {
            "success_mean": _safe_float(success.mean()),
            "failure_mean": _safe_float(failure.mean()),
            "failure_minus_success": _safe_float(failure.mean() - success.mean()),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.predictions)
    required = {"date", "code", "return", "candidate_rank"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["top5_order"] = frame.groupby("date", sort=False)["candidate_rank"].rank(
        method="first", ascending=False
    )
    top = frame.loc[frame["top5_order"] <= 5].copy()
    if len(top) != frame["date"].nunique() * 5:
        raise AssertionError("Each validation date must contain exactly five selections")

    dates = np.sort(frame["date"].unique())
    midpoint = len(dates) // 2
    period_masks = {
        "full": np.ones(len(frame), dtype=bool),
        "first_half": frame["date"].isin(dates[:midpoint]).to_numpy(),
        "second_half": frame["date"].isin(dates[midpoint:]).to_numpy(),
    }
    for year in sorted(frame["date"].dt.year.unique()):
        period_masks[f"year_{year}"] = (frame["date"].dt.year == year).to_numpy()
    for quarter in sorted(frame["date"].dt.to_period("Q").unique()):
        period_masks[f"quarter_{quarter}"] = (
            frame["date"].dt.to_period("Q") == quarter
        ).to_numpy()

    periods = {}
    exposures = {}
    for label, mask in period_masks.items():
        universe_part = frame.loc[mask]
        top_part = top[top["date"].isin(universe_part["date"].unique())]
        periods[label] = _period_summary(top_part, universe_part)
        exposures[label] = _exposure_diagnostics(top_part, universe_part)

    daily_top = top.groupby("date", sort=True)["return"].mean()
    tail_threshold = float(top["return"].quantile(0.05))
    tail = top[top["return"] <= tail_threshold]
    code_counts = top["code"].value_counts()
    payload = {
        "metadata": {
            "samples": int(len(frame)),
            "days": int(frame["date"].nunique()),
            "top5_samples": int(len(top)),
            "date_start": str(frame["date"].min().date()),
            "date_end": str(frame["date"].max().date()),
        },
        "periods": periods,
        "exposures": exposures,
        "tail_risk": {
            "top5_return_p05": _safe_float(tail_threshold),
            "tail_sample_share": _safe_float(len(tail) / len(top)),
            "tail_return_contribution": _safe_float(tail["return"].sum() / top["return"].abs().sum()),
            "worst_daily_top5": _safe_float(daily_top.min()),
            "top10_code_selection_share": _safe_float(code_counts.head(10).sum() / len(top)),
            "max_single_code_selection_share": _safe_float(code_counts.iloc[0] / len(top)),
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload["tail_risk"], ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
