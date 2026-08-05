"""Diagnose cross-sectional calibration between path labels and future returns."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import spearmanr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.baostock_config import DATABASE_PATH
from config.factor_config import TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.train_ml_model import MLModelTrainer


def _split_indices(dates):
    raw_split_idx = int(len(dates) * TrainingConfig.TRAIN_TEST_SPLIT)
    split_date = dates[raw_split_idx]
    split_idx = int(np.searchsorted(dates, split_date, side="left"))
    unique_dates = np.unique(dates)
    split_date_idx = int(np.searchsorted(unique_dates, split_date))
    forward_days = TrainingConfig.FUTURE_DAYS
    val_date_idx = min(split_date_idx + forward_days, len(unique_dates) - 1)
    val_start_date = unique_dates[val_date_idx]
    val_start_idx = int(np.searchsorted(dates, val_start_date, side="left"))
    return split_idx, val_start_idx, str(split_date), str(val_start_date)


def _daily_percentiles(scores, dates):
    percentiles = np.empty(len(scores), dtype=np.float64)
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    for start, count in zip(starts, counts):
        values = scores[start:start + count]
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(count, dtype=np.float64)
        ranks[order] = np.arange(1, count + 1, dtype=np.float64)
        percentiles[start:start + count] = ranks / (count + 1.0)
    return percentiles


def _safe_spearman(x, y):
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    value = spearmanr(x, y).statistic
    return None if not np.isfinite(value) else float(value)


def _summarize_mask(mask, returns, raw_scores, dates, limit_groups):
    selected_returns = returns[mask]
    selected_scores = raw_scores[mask]
    selected_dates = dates[mask]
    daily_means = []
    daily_medians = []
    for date in np.unique(selected_dates):
        values = selected_returns[selected_dates == date]
        daily_means.append(float(np.mean(values)))
        daily_medians.append(float(np.median(values)))
    daily_means_arr = np.asarray(daily_means, dtype=np.float64)
    limit_mix = {}
    for value in np.unique(limit_groups[mask]):
        key = f"{float(value):.4f}"
        limit_mix[key] = float(np.mean(np.isclose(limit_groups[mask], value)))
    return {
        "samples": int(mask.sum()),
        "dates": int(len(daily_means)),
        "future_return_mean": float(np.mean(selected_returns)),
        "future_return_median": float(np.median(selected_returns)),
        "positive_return_ratio": float(np.mean(selected_returns > 0)),
        "raw_path_score_mean": float(np.mean(selected_scores)),
        "raw_path_score_median": float(np.median(selected_scores)),
        "daily_return_mean": float(np.mean(daily_means_arr)),
        "daily_return_std": float(np.std(daily_means_arr)),
        "positive_daily_mean_ratio": float(np.mean(daily_means_arr > 0)),
        "daily_median_mean": float(np.mean(daily_medians)),
        "limit_threshold_mix": limit_mix,
    }


def summarize_segment(scores, returns, dates, limit_groups, bins):
    percentiles = _daily_percentiles(scores, dates)
    bucket_ids = np.minimum((percentiles * bins).astype(np.int64), bins - 1)
    bucket_rows = []
    for bucket in range(bins):
        mask = bucket_ids == bucket
        row = _summarize_mask(mask, returns, scores, dates, limit_groups)
        row["bucket"] = bucket + 1
        row["percentile_low"] = bucket / bins
        row["percentile_high"] = (bucket + 1) / bins
        bucket_rows.append(row)

    bucket_numbers = np.arange(1, bins + 1, dtype=np.float64)
    bucket_returns = np.asarray(
        [row["daily_return_mean"] for row in bucket_rows], dtype=np.float64
    )
    adjacent_increases = np.diff(bucket_returns) >= 0
    top_slices = {}
    for fraction in (0.01, 0.05, 0.10):
        mask = percentiles >= 1.0 - fraction
        top_slices[f"top_{int(fraction * 100)}pct"] = _summarize_mask(
            mask, returns, scores, dates, limit_groups
        )

    daily_ic = []
    for date in np.unique(dates):
        mask = dates == date
        value = _safe_spearman(scores[mask], returns[mask])
        if value is not None:
            daily_ic.append(value)
    daily_ic_arr = np.asarray(daily_ic, dtype=np.float64)
    return {
        "samples": int(len(scores)),
        "dates": int(len(np.unique(dates))),
        "daily_label_return_rank_ic_mean": float(np.mean(daily_ic_arr)),
        "daily_label_return_rank_ic_std": float(np.std(daily_ic_arr)),
        "positive_daily_ic_ratio": float(np.mean(daily_ic_arr > 0)),
        "bucket_return_spearman": _safe_spearman(bucket_numbers, bucket_returns),
        "adjacent_bucket_increase_ratio": float(np.mean(adjacent_increases)),
        "top_bucket_minus_universe": float(
            bucket_rows[-1]["daily_return_mean"] - np.mean(returns)
        ),
        "buckets": bucket_rows,
        "top_slices": top_slices,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")

    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
    started = time.time()

    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()["code"].tolist()[:args.stocks]
    manager.close()
    stocks_data = trainer.load_label_data(codes, start, end)
    dataset = trainer.prepare_dataset(
        stocks_data,
        train_start_date=start,
        train_end_date=end,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        n_jobs=args.workers,
        target_features=None,
        use_factor_cache_only=True,
    )
    _, _, returns, factor_names, dates, _, limit_groups, raw_scores, _, _ = dataset
    split_idx, val_start_idx, split_date, val_start_date = _split_indices(dates)

    payload = {
        "metadata": {
            "stocks_requested": args.stocks,
            "stocks_loaded": len(stocks_data),
            "start": start,
            "end": end,
            "samples": len(dates),
            "features": len(factor_names),
            "bins": args.bins,
            "split_date": split_date,
            "validation_start_date": val_start_date,
            "embargo_trading_days": TrainingConfig.FUTURE_DAYS,
            "label": "raw path quality score, bucketed by daily percentile",
            "return": "future close / next open - 1",
        },
        "train": summarize_segment(
            raw_scores[:split_idx], returns[:split_idx], dates[:split_idx],
            limit_groups[:split_idx], args.bins,
        ),
        "validation": summarize_segment(
            raw_scores[val_start_idx:], returns[val_start_idx:], dates[val_start_idx:],
            limit_groups[val_start_idx:], args.bins,
        ),
    }
    payload["metadata"]["elapsed_seconds"] = time.time() - started
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
