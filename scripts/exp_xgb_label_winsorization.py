"""Controlled experiment for per-date winsorization of ranking path scores."""

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


def winsorize_by_date(scores: np.ndarray, dates: np.ndarray, tail: float) -> np.ndarray:
    if not 0 <= tail < 0.5:
        raise ValueError("tail must be in [0, 0.5)")
    result = np.asarray(scores, dtype=np.float64).copy()
    if tail == 0:
        return result
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    for start, count in zip(starts, counts):
        end = start + count
        lower, upper = np.quantile(result[start:end], [tail, 1.0 - tail])
        np.clip(result[start:end], lower, upper, out=result[start:end])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", default=None)
    parser.add_argument("--tails", default="0,0.005,0.01,0.02")
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tails = [float(value) for value in args.tails.split(",")]
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
    dataset = trainer.prepare_dataset(
        stocks_data,
        train_start_date=start,
        train_end_date=end,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        n_jobs=args.workers,
        target_features=None,
        use_factor_cache_only=True,
    )
    del stocks_data
    (X, y, returns, factor_names, dates, unbuyable, limit_groups,
     path_scores, is_st, w_sig) = dataset

    original_estimators = ModelConfig.XGBOOST_PARAMS["n_estimators"]
    ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators
    rows = []
    try:
        for tail in tails:
            started = time.time()
            clipped_scores = winsorize_by_date(path_scores, dates, tail)
            changed = int(np.count_nonzero(clipped_scores != path_scores))
            results = trainer.train_models(
                X.copy(), y.copy(), returns, list(factor_names), dates,
                unbuyable_mask=unbuyable,
                limit_groups=limit_groups,
                path_scores=clipped_scores,
                is_st_arr=is_st,
                w_sig_arr=w_sig,
                model_types=["xgboost"],
                val_eval_sample_ratio=1.0,
            )
            metrics = results["xgboost"]["val_metrics"]
            model = trainer.models["xgboost"]
            rows.append({
                "tail": tail,
                "changed_samples": changed,
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
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = original_estimators

    payload = {
        "metadata": {
            "stocks_requested": args.stocks,
            "stocks_loaded": stocks_loaded,
            "start": start,
            "end": end,
            "samples": len(dates),
            "features": len(factor_names),
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
