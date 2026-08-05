"""Ablate mutually exclusive feature groups with a fixed selected feature set."""

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


ENGINEERED_MARKERS = (
    "_mul_", "_div_", "_sub_", "_add_", "_x_", "log_", "rank_", "sqrt_",
)
FUNDAMENTAL_MARKERS = (
    "YOY", "MBRevenue", "asset", "Asset", "Equity", "Liability", "Ratio",
    "dupont", "Margin", "eps", "EPS", "roe", "ROE", "liqaShare", "totalShare",
    "dynamic_p", "inv_p", "peg", "sue",
)
STATE_PREFIXES = (
    "mean_return_regime_", "breadth_ma20_regime_", "up_ratio_regime_",
    "industry_", "sector_", "is_", "days_to_", "market_type",
)
ADVANCED_FEATURES = {
    "hl_range_mean", "hl_range_std", "oc_ratio_mean", "oc_ratio_std",
    "price_volatility_20", "price_volatility_60", "price_skewness", "price_kurtosis",
    "high_position", "low_position", "volume_change_rate", "volume_volatility",
    "price_volume_corr", "amount_per_volume", "amount_change_rate", "amount_ma_10",
    "amount_std_30", "return_5d", "return_10d", "return_20d", "return_60d",
    "momentum_5d", "momentum_10d", "momentum_20d", "acceleration_5d",
    "acceleration_10d", "downside_risk", "drawdown", "max_drawdown_20",
    "sharpe_ratio", "return_skewness", "return_kurtosis", "intraday_drawdown_avg_5d",
}


def classify_feature(name):
    """Assign one group only; ordering makes the partition deterministic."""
    if name.startswith(STATE_PREFIXES):
        return "market_state"
    if any(marker in name for marker in ENGINEERED_MARKERS):
        return "engineered"
    if any(marker in name for marker in FUNDAMENTAL_MARKERS):
        return "fundamental"
    if name in ADVANCED_FEATURES:
        return "price_volume_risk"
    return "technical"


def train_variant(trainer, dataset, keep_names, variant):
    (X, y, returns, factor_names, dates, unbuyable, limit_groups,
     path_scores, is_st, w_sig) = dataset
    index = {name: idx for idx, name in enumerate(factor_names)}
    missing = [name for name in keep_names if name not in index]
    if missing:
        raise ValueError(f"Missing requested features: {missing[:5]}")
    keep_indices = [index[name] for name in keep_names]

    started = time.time()
    results = trainer.train_models(
        X[:, keep_indices].copy(), y.copy(), returns, list(keep_names), dates,
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
    return {
        "variant": variant,
        "features_requested": len(keep_names),
        "features_trained": len(model.feature_names),
        "trained_feature_names": list(model.feature_names),
        "best_iteration": model._get_xgb_best_iteration(model.model),
        "rank_ic": metrics.get("rank_ic"),
        "rank_ic_std": metrics.get("rank_ic_std", metrics.get("ic_std")),
        "rank_ic_ir": metrics.get("rank_ic_ir"),
        "positive_ic_ratio": metrics.get("positive_ic_ratio"),
        "top5_return": metrics.get("top5_mean_return"),
        "top5_excess": metrics.get("top5_excess_return"),
        "elapsed_seconds": time.time() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", default=None)
    parser.add_argument("--groups", default="all")
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    end_dt = datetime.strptime(args.end, "%Y-%m-%d") if args.end else (
        datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)
    )
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    cache_dir = os.path.join(output_dir, "feature_ablation_cache", end)
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
    factor_names = list(dataset[3])

    original_save_dir = TrainingConfig.SAVE_DIR
    original_estimators = ModelConfig.XGBOOST_PARAMS["n_estimators"]
    TrainingConfig.SAVE_DIR = cache_dir
    ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators
    try:
        baseline = train_variant(trainer, dataset, factor_names, "baseline")
        selected = baseline["trained_feature_names"]
        grouped = {group: [] for group in (
            "fundamental", "engineered", "market_state", "price_volume_risk", "technical"
        )}
        for name in selected:
            grouped[classify_feature(name)].append(name)

        requested_groups = list(grouped) if args.groups == "all" else args.groups.split(",")
        unknown = sorted(set(requested_groups) - set(grouped))
        if unknown:
            raise ValueError(f"Unknown groups: {unknown}")

        rows = [baseline]
        for group in requested_groups:
            keep = [name for name in selected if name not in set(grouped[group])]
            print(f"\n{'#' * 72}\n# no_{group}: remove {len(grouped[group])}, keep {len(keep)}\n{'#' * 72}")
            rows.append(train_variant(trainer, dataset, keep, f"no_{group}"))
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
            "raw_features": len(factor_names),
            "estimators": args.estimators,
            "cache_dir": cache_dir,
        },
        "feature_groups": grouped,
        "results": rows,
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
