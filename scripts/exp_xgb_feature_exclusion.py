"""Compare a baseline selected feature set with an explicit exclusion list."""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.baostock_config import DATABASE_PATH
from config.factor_config import ModelConfig, TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.train_ml_model import MLModelTrainer
from scripts.exp_xgb_feature_group_ablation import train_variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", required=True)
    parser.add_argument("--exclude", required=True)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    excluded = [name.strip() for name in args.exclude.split(",") if name.strip()]
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    cache_dir = os.path.join(output_dir, "feature_exclusion_cache", end)
    os.makedirs(cache_dir, exist_ok=True)

    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()["code"].tolist()[:args.stocks]
    manager.close()
    stocks_data = trainer.load_label_data(codes, start, end)
    dataset = trainer.prepare_dataset(
        stocks_data, train_start_date=start, train_end_date=end,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        n_jobs=args.workers, target_features=None, use_factor_cache_only=True,
    )

    original_save_dir = TrainingConfig.SAVE_DIR
    original_estimators = ModelConfig.XGBOOST_PARAMS["n_estimators"]
    TrainingConfig.SAVE_DIR = cache_dir
    ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators
    try:
        baseline = train_variant(trainer, dataset, list(dataset[3]), "baseline")
        selected = baseline["trained_feature_names"]
        absent = sorted(set(excluded) - set(selected))
        if absent:
            raise ValueError(f"Excluded features not selected in this window: {absent}")
        candidate_names = [name for name in selected if name not in set(excluded)]
        candidate = train_variant(trainer, dataset, candidate_names, "exclude_unstable_fundamental")
    finally:
        TrainingConfig.SAVE_DIR = original_save_dir
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = original_estimators

    payload = {
        "metadata": {
            "stocks_requested": args.stocks,
            "stocks_loaded": len(stocks_data),
            "start": start,
            "end": end,
            "samples": len(dataset[4]),
            "excluded_features": excluded,
            "selection_rule": (
                "Selected in all windows; positive train IC and negative validation IC "
                "in both historical end-2022 and end-2023 windows"
            ),
        },
        "results": [baseline, candidate],
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
