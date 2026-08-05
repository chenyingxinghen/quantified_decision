"""Measure fundamental feature stability across one training window."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import rankdata

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.baostock_config import DATABASE_PATH
from config.factor_config import ModelConfig, TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.train_ml_model import MLModelTrainer
from scripts.exp_xgb_feature_group_ablation import classify_feature


def daily_rank_ic(values, returns, dates, minimum_samples=20):
    daily = []
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    for start, count in zip(starts, counts):
        end = start + count
        x = values[start:end]
        y = returns[start:end]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < minimum_samples:
            continue
        x = x[valid]
        y = y[valid]
        if np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
            continue
        corr = np.corrcoef(rankdata(x), rankdata(y))[0, 1]
        if np.isfinite(corr):
            daily.append(float(corr))
    return np.asarray(daily, dtype=np.float64)


def summarize_feature(values, returns, dates):
    finite = np.isfinite(values)
    nonzero = finite & (np.abs(values) > 1e-12)
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    varying_days = 0
    valid_days = 0
    for start, count in zip(starts, counts):
        subset = values[start:start + count]
        subset = subset[np.isfinite(subset)]
        if len(subset) < 20:
            continue
        valid_days += 1
        varying_days += int(np.ptp(subset) > 1e-12)
    ics = daily_rank_ic(values, returns, dates)
    return {
        "finite_ratio": float(finite.mean()),
        "nonzero_ratio": float(nonzero.mean()),
        "varying_day_ratio": float(varying_days / valid_days) if valid_days else 0.0,
        "ic_days": int(len(ics)),
        "rank_ic": float(ics.mean()) if len(ics) else None,
        "rank_ic_std": float(ics.std()) if len(ics) else None,
        "positive_ic_ratio": float((ics > 0).mean()) if len(ics) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", required=True)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    cache_dir = os.path.join(output_dir, "fundamental_stability_cache", end)
    os.makedirs(cache_dir, exist_ok=True)

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
    (X, y, returns, factor_names, dates, unbuyable, limit_groups,
     path_scores, is_st, w_sig) = dataset

    raw_split_idx = int(len(dates) * TrainingConfig.TRAIN_TEST_SPLIT)
    split_date = dates[raw_split_idx]
    split_idx = int(np.searchsorted(dates, split_date, side="left"))
    unique_dates = np.unique(dates)
    split_date_idx = int(np.searchsorted(unique_dates, split_date))
    forward_days = getattr(TrainingConfig, "FUTURE_DAYS", 7)
    val_start_date = unique_dates[min(split_date_idx + forward_days, len(unique_dates) - 1)]
    val_start_idx = int(np.searchsorted(dates, val_start_date, side="left"))

    fundamental_indices = [
        idx for idx, name in enumerate(factor_names) if classify_feature(name) == "fundamental"
    ]
    diagnostics = {}
    for idx in fundamental_indices:
        name = factor_names[idx]
        diagnostics[name] = {
            "train": summarize_feature(
                X[:split_idx, idx], returns[:split_idx], dates[:split_idx]
            ),
            "validation": summarize_feature(
                X[val_start_idx:, idx], returns[val_start_idx:], dates[val_start_idx:]
            ),
        }

    original_save_dir = TrainingConfig.SAVE_DIR
    original_estimators = ModelConfig.XGBOOST_PARAMS["n_estimators"]
    TrainingConfig.SAVE_DIR = cache_dir
    ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators
    started = time.time()
    try:
        results = trainer.train_models(
            X, y, returns, factor_names, dates,
            unbuyable_mask=unbuyable,
            limit_groups=limit_groups,
            path_scores=path_scores,
            is_st_arr=is_st,
            w_sig_arr=w_sig,
            model_types=["xgboost"],
            val_eval_sample_ratio=1.0,
        )
    finally:
        TrainingConfig.SAVE_DIR = original_save_dir
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = original_estimators

    metrics = results["xgboost"]["val_metrics"]
    model = trainer.models["xgboost"]
    selected = set(model.feature_names)
    gain = model.feature_importance
    for name, item in diagnostics.items():
        item["selected"] = name in selected
        item["gain"] = float(gain.get(name, 0.0))

    payload = {
        "metadata": {
            "stocks_requested": args.stocks,
            "stocks_loaded": len(stocks_data),
            "start": start,
            "end": end,
            "samples": len(dates),
            "raw_features": len(factor_names),
            "selected_features": len(model.feature_names),
            "fundamental_features_diagnosed": len(diagnostics),
            "split_date": str(split_date),
            "validation_start_date": str(val_start_date),
            "elapsed_seconds": time.time() - started,
        },
        "model_metrics": {
            "best_iteration": model._get_xgb_best_iteration(model.model),
            "rank_ic": metrics.get("rank_ic"),
            "rank_ic_std": metrics.get("rank_ic_std", metrics.get("ic_std")),
            "rank_ic_ir": metrics.get("rank_ic_ir"),
            "positive_ic_ratio": metrics.get("positive_ic_ratio"),
            "top5_excess": metrics.get("top5_excess_return"),
        },
        "features": diagnostics,
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
