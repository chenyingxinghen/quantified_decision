"""Controlled ablation of components in the XGBoost ranking path score."""

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


VARIANTS = {
    "baseline": {},
    "no_path_order": {"PATH_BONUS": 0.0, "PATH_PENALTY": 0.0},
    "no_upside": {"UPSIDE_WEIGHT": 0.0},
    "no_downside": {"DOWNSIDE_WEIGHT": 0.0},
    "final_return_only": {
        "UPSIDE_WEIGHT": 0.0,
        "DOWNSIDE_WEIGHT": 0.0,
        "PATH_BONUS": 0.0,
        "PATH_PENALTY": 0.0,
    },
}

CONFIG_KEYS = (
    "UPSIDE_WEIGHT", "DOWNSIDE_WEIGHT", "FINAL_RETURN_WEIGHT",
    "PATH_BONUS", "PATH_PENALTY", "VOL_BOOSTER_COEF",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", default=None)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    variants = args.variants.split(",")
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    end_dt = datetime.strptime(args.end, "%Y-%m-%d") if args.end else (
        datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)
    )
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")

    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()["code"].tolist()[:args.stocks]
    manager.close()
    stocks_data = trainer.load_label_data(codes, start, end)
    stocks_loaded = len(stocks_data)

    original_config = {key: getattr(TrainingConfig, key) for key in CONFIG_KEYS}
    original_estimators = ModelConfig.XGBOOST_PARAMS["n_estimators"]
    ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators
    rows = []
    try:
        for variant in variants:
            for key, value in original_config.items():
                setattr(TrainingConfig, key, value)
            for key, value in VARIANTS[variant].items():
                setattr(TrainingConfig, key, value)

            effective = {key: getattr(TrainingConfig, key) for key in CONFIG_KEYS}
            print(f"\n{'#' * 72}\n# {variant}: {effective}\n{'#' * 72}")
            started = time.time()
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
            metrics = results["xgboost"]["val_metrics"]
            model = trainer.models["xgboost"]
            rows.append({
                "variant": variant,
                "effective_config": effective,
                "samples": len(dates),
                "features_selected": len(model.feature_names),
                "best_iteration": model._get_xgb_best_iteration(model.model),
                "rank_ic": metrics.get("rank_ic"),
                "rank_ic_std": metrics.get("rank_ic_std", metrics.get("ic_std")),
                "rank_ic_ir": metrics.get("rank_ic_ir"),
                "positive_ic_ratio": metrics.get("positive_ic_ratio"),
                "top5_return": metrics.get("top5_mean_return"),
                "top5_excess": metrics.get("top5_excess_return"),
                "elapsed_seconds": time.time() - started,
            })
    finally:
        for key, value in original_config.items():
            setattr(TrainingConfig, key, value)
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = original_estimators

    payload = {
        "metadata": {
            "stocks_requested": args.stocks,
            "stocks_loaded": stocks_loaded,
            "start": start,
            "end": end,
            "estimators": args.estimators,
        },
        "results": rows,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
