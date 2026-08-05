"""Compare single-fold early stopping with historical-fold iteration selection."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.baostock_config import DATABASE_PATH
from config.factor_config import ModelConfig, TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.train_ml_model import MLModelTrainer
from scripts.exp_xgb_feature_group_ablation import train_variant


def run_training(trainer, dataset, feature_names, variant):
    row = train_variant(trainer, dataset, feature_names, variant)
    row["configured_estimators"] = ModelConfig.XGBOOST_PARAMS["n_estimators"]
    return row


def build_closed_fold(dataset, train_fraction, validation_end_fraction):
    """Return data ending at validation_end_fraction without splitting a date."""
    dates = dataset[4]
    raw_train_idx = int(len(dates) * train_fraction)
    raw_end_idx = int(len(dates) * validation_end_fraction)
    train_date = dates[raw_train_idx]
    end_date = dates[min(raw_end_idx, len(dates) - 1)]
    train_idx = int(np.searchsorted(dates, train_date, side="left"))
    end_idx = int(np.searchsorted(dates, end_date, side="left"))
    if not 0 < train_idx < end_idx:
        raise ValueError("Invalid closed-fold boundaries")

    sliced = []
    for index, value in enumerate(dataset):
        sliced.append(value if index == 3 else value[:end_idx])
    return tuple(sliced), train_idx / end_idx, str(train_date), str(end_date)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", required=True)
    parser.add_argument("--folds", default="0.6,0.7")
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    folds = [float(value) for value in args.folds.split(",")]
    if not folds or any(value <= 0 or value >= 0.8 for value in folds):
        raise ValueError("Historical folds must lie between 0 and the final 0.8 split")
    fold_windows = [(value, round(value + 0.1, 10)) for value in folds]
    if any(end_fraction > 0.8 for _, end_fraction in fold_windows):
        raise ValueError("Historical fold validation must end by the final 0.8 split")

    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    cache_dir = os.path.join(output_dir, "multifold_iteration_cache", end)
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

    original_save_dir = TrainingConfig.SAVE_DIR
    original_split = TrainingConfig.TRAIN_TEST_SPLIT
    original_params = ModelConfig.XGBOOST_PARAMS.copy()
    started = time.time()
    try:
        TrainingConfig.SAVE_DIR = cache_dir
        TrainingConfig.TRAIN_TEST_SPLIT = 0.8
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators
        baseline = run_training(trainer, dataset, list(dataset[3]), "single_fold_early_stopping")
        selected_features = baseline["trained_feature_names"]

        fold_results = []
        for fold, fold_end in fold_windows:
            fold_dataset, fold_split, fold_train_date, fold_end_date = build_closed_fold(
                dataset, fold, fold_end
            )
            TrainingConfig.TRAIN_TEST_SPLIT = fold_split
            ModelConfig.XGBOOST_PARAMS.clear()
            ModelConfig.XGBOOST_PARAMS.update(original_params)
            ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators
            row = run_training(
                trainer, fold_dataset, selected_features,
                f"historical_fold_{fold:.2f}_to_{fold_end:.2f}"
            )
            row["train_test_split"] = fold
            row["validation_end_fraction"] = fold_end
            row["train_boundary_date"] = fold_train_date
            row["validation_end_date_exclusive"] = fold_end_date
            fold_results.append(row)

        best_rounds = [row["best_iteration"] + 1 for row in fold_results]
        selected_rounds = max(1, int(np.median(best_rounds)))

        TrainingConfig.TRAIN_TEST_SPLIT = 0.8
        ModelConfig.XGBOOST_PARAMS.clear()
        ModelConfig.XGBOOST_PARAMS.update(original_params)
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = selected_rounds
        ModelConfig.XGBOOST_PARAMS["early_stopping_rounds"] = None
        candidate = run_training(
            trainer, dataset, selected_features, "multifold_median_fixed_rounds"
        )
        candidate["selected_rounds"] = selected_rounds
        candidate["selection_fold_best_rounds"] = best_rounds
    finally:
        TrainingConfig.SAVE_DIR = original_save_dir
        TrainingConfig.TRAIN_TEST_SPLIT = original_split
        ModelConfig.XGBOOST_PARAMS.clear()
        ModelConfig.XGBOOST_PARAMS.update(original_params)

    payload = {
        "metadata": {
            "stocks_requested": args.stocks,
            "stocks_loaded": len(stocks_data),
            "start": start,
            "end": end,
            "samples": len(dataset[4]),
            "raw_features": len(dataset[3]),
            "selected_features": len(selected_features),
            "historical_folds": [list(window) for window in fold_windows],
            "final_split": 0.8,
            "embargo_trading_days": TrainingConfig.FUTURE_DAYS,
            "max_estimators": args.estimators,
            "round_selection": "median of one-based historical-fold best iterations",
            "elapsed_seconds": time.time() - started,
        },
        "baseline": baseline,
        "selection_folds": fold_results,
        "candidate": candidate,
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
