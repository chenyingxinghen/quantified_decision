"""Scan conservative one-axis XGBoost regularization changes."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.baostock_config import DATABASE_PATH
from config.factor_config import ModelConfig, TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.train_ml_model import MLModelTrainer
from scripts.exp_xgb_feature_group_ablation import train_variant


VARIANTS = {
    "baseline": {},
    "depth_2": {"max_depth": 2},
    "child_weight_5": {"min_child_weight": 5.0},
    "subsample_07": {"subsample": 0.7},
    "colsample_07": {"colsample_bytree": 0.7},
    "reg_alpha_125": {"reg_alpha": 1.25},
    "reg_alpha_5": {"reg_alpha": 5.0},
    "reg_lambda_125": {"reg_lambda": 1.25},
    "reg_lambda_5": {"reg_lambda": 5.0},
    "lr_003_n700": {"learning_rate": 0.03, "n_estimators": 700},
    "lr_002_n1000": {"learning_rate": 0.02, "n_estimators": 1000},
    "gamma_0": {"gamma": 0.0},
    "gamma_005": {"gamma": 0.05},
}

QUERY_WEIGHT_VARIANTS = {
    "query_dispersion_high": "high",
    "query_dispersion_low": "low",
}

OBJECTIVE_VARIANTS = {
    "objective_pairwise": "rank:pairwise",
}

METRICS = ("rank_ic", "rank_ic_ir", "positive_ic_ratio", "top5_excess")


def compare_with_baseline(row, baseline):
    deltas = {key: row[key] - baseline[key] for key in METRICS}
    # A candidate advances only when it does not trade away any core metric.
    passes = all(deltas[key] >= 0.0 for key in METRICS) and any(
        deltas[key] > 0.0 for key in METRICS
    )
    return deltas, passes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    variants = args.variants.split(",")
    unknown = sorted(
        set(variants) - set(VARIANTS) - set(QUERY_WEIGHT_VARIANTS)
        - set(OBJECTIVE_VARIANTS)
    )
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    if "baseline" not in variants:
        variants.insert(0, "baseline")

    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    cache_dir = os.path.join(output_dir, "regularization_scan_cache", end)
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
    original_params = ModelConfig.XGBOOST_PARAMS.copy()
    original_query_weight_mode = TrainingConfig.QUERY_LABEL_DISPERSION_WEIGHT
    original_objective = TrainingConfig.XGBOOST_RANKING_OBJECTIVE
    started = time.time()
    rows = []
    try:
        TrainingConfig.SAVE_DIR = cache_dir
        baseline_features = list(dataset[3])
        baseline = None
        for variant in variants:
            ModelConfig.XGBOOST_PARAMS.clear()
            ModelConfig.XGBOOST_PARAMS.update(original_params)
            ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators
            ModelConfig.XGBOOST_PARAMS.update(VARIANTS.get(variant, {}))
            TrainingConfig.QUERY_LABEL_DISPERSION_WEIGHT = QUERY_WEIGHT_VARIANTS.get(
                variant, "off"
            )
            TrainingConfig.XGBOOST_RANKING_OBJECTIVE = OBJECTIVE_VARIANTS.get(
                variant, "rank:ndcg"
            )
            effective = {
                key: ModelConfig.XGBOOST_PARAMS[key]
                for key in (
                    "n_estimators", "learning_rate", "max_depth",
                    "min_child_weight", "subsample",
                    "colsample_bytree", "gamma", "reg_alpha", "reg_lambda",
                )
            }
            print(f"\n{'#' * 72}\n# {variant}: {effective}\n{'#' * 72}")
            row = train_variant(trainer, dataset, baseline_features, variant)
            row["parameter_overrides"] = VARIANTS.get(variant, {})
            row["query_label_dispersion_weight"] = (
                TrainingConfig.QUERY_LABEL_DISPERSION_WEIGHT
            )
            row["ranking_objective"] = TrainingConfig.XGBOOST_RANKING_OBJECTIVE
            row["xgboost_label_mode"] = (
                "fixed_15_bins"
                if TrainingConfig.XGBOOST_RANKING_OBJECTIVE == "rank:pairwise"
                else "continuous_rank_power"
            )
            row["effective_params"] = effective
            if variant == "baseline":
                baseline = row
                baseline_features = row["trained_feature_names"]
                row["deltas_vs_baseline"] = {key: 0.0 for key in METRICS}
                row["passes_first_window"] = True
            else:
                deltas, passes = compare_with_baseline(row, baseline)
                row["deltas_vs_baseline"] = deltas
                row["passes_first_window"] = passes
            rows.append(row)
    finally:
        TrainingConfig.SAVE_DIR = original_save_dir
        ModelConfig.XGBOOST_PARAMS.clear()
        ModelConfig.XGBOOST_PARAMS.update(original_params)
        TrainingConfig.QUERY_LABEL_DISPERSION_WEIGHT = original_query_weight_mode
        TrainingConfig.XGBOOST_RANKING_OBJECTIVE = original_objective

    payload = {
        "metadata": {
            "stocks_requested": args.stocks,
            "stocks_loaded": len(stocks_data),
            "start": start,
            "end": end,
            "samples": len(dataset[4]),
            "raw_features": len(dataset[3]),
            "selected_features": len(baseline_features),
            "max_estimators": args.estimators,
            "validation_sample_ratio": 1.0,
            "embargo_trading_days": TrainingConfig.FUTURE_DAYS,
            "promotion_rule": "all four core metric deltas >= 0 and at least one > 0",
            "cache_dir": cache_dir,
            "elapsed_seconds": time.time() - started,
        },
        "results": rows,
        "promoted_variants": [
            row["variant"] for row in rows
            if row["variant"] != "baseline" and row["passes_first_window"]
        ],
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
