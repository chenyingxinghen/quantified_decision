"""Diagnose whether XGBoost NDCG and daily portfolio metrics select the same round."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.baostock_config import DATABASE_PATH
from config.factor_config import ModelConfig, TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.train_ml_model import MLModelTrainer, _fast_rankdata_1d
from scripts.exp_xgb_multifold_iteration_selection import build_closed_fold


def daily_metrics(predictions, returns, dates):
    rank_ics = []
    top5_excess = []
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    for start, count in zip(starts, counts):
        if count < 5:
            continue
        end = start + count
        pred = predictions[start:end]
        ret = returns[start:end]
        valid = np.isfinite(pred) & np.isfinite(ret)
        if valid.sum() < 5:
            continue
        pred = pred[valid]
        ret = ret[valid]
        pred_rank = _fast_rankdata_1d(pred)
        ret_rank = _fast_rankdata_1d(ret)
        rank_ics.append(float(np.corrcoef(pred_rank, ret_rank)[0, 1]))
        top_count = min(5, len(pred))
        top = np.argpartition(pred, -top_count)[-top_count:]
        top5_excess.append(float(ret[top].mean() - ret.mean()))
    rank_ics = np.asarray(rank_ics, dtype=np.float64)
    top5_excess = np.asarray(top5_excess, dtype=np.float64)
    ic_std = float(rank_ics.std())
    return {
        "rank_ic": float(rank_ics.mean()),
        "rank_ic_std": ic_std,
        "rank_ic_ir": float(rank_ics.mean() / ic_std) if ic_std else None,
        "positive_ic_ratio": float((rank_ics > 0).mean()),
        "top5_excess": float(top5_excess.mean()),
        "validation_days": int(len(rank_ics)),
    }


def train_full_and_scan(trainer, dataset, feature_names, split, checkpoints):
    (X, y, returns, factor_names, dates, unbuyable, limit_groups,
     path_scores, is_st, w_sig) = dataset
    feature_index = {name: index for index, name in enumerate(factor_names)}
    requested_indices = [feature_index[name] for name in feature_names]
    X_work = X[:, requested_indices].copy()

    TrainingConfig.TRAIN_TEST_SPLIT = split
    results = trainer.train_models(
        X_work, y.copy(), returns, list(feature_names), dates,
        unbuyable_mask=unbuyable,
        limit_groups=limit_groups,
        path_scores=path_scores,
        is_st_arr=is_st,
        w_sig_arr=w_sig,
        model_types=["xgboost"],
        val_eval_sample_ratio=1.0,
    )
    model = trainer.models["xgboost"]
    trained_indices = [feature_names.index(name) for name in model.feature_names]
    unique_dates = np.unique(dates)
    split_date = dates[int(len(dates) * split)]
    split_date_index = np.searchsorted(unique_dates, split_date)
    val_start_date = unique_dates[min(
        split_date_index + TrainingConfig.FUTURE_DAYS, len(unique_dates) - 1
    )]
    val_start = int(np.searchsorted(dates, val_start_date, side="left"))
    X_val = X_work[val_start:, trained_indices]
    dates_val = dates[val_start:]
    returns_val = returns[val_start:]
    dval = xgb.DMatrix(X_val, feature_names=model.feature_names)
    booster = model.model

    curve = []
    for rounds in checkpoints:
        predictions = booster.predict(dval, iteration_range=(0, rounds))
        row = daily_metrics(predictions, returns_val, dates_val)
        row["rounds"] = rounds
        curve.append(row)

    validation_curve = model._evals_result["validation"]
    ndcg_name = next(iter(validation_curve))
    ndcg_values = validation_curve[ndcg_name]
    ndcg_best_rounds = int(np.argmax(ndcg_values)) + 1
    return {
        "features_trained": len(model.feature_names),
        "trained_feature_names": list(model.feature_names),
        "split_date": str(split_date),
        "validation_start_date": str(val_start_date),
        "validation_samples": len(dates_val),
        "ndcg_metric": ndcg_name,
        "ndcg_best_rounds": ndcg_best_rounds,
        "ndcg_best_value": float(ndcg_values[ndcg_best_rounds - 1]),
        "curve": curve,
        "trainer_full_round_metrics": results["xgboost"]["val_metrics"],
    }


def row_at_rounds(curve, rounds):
    return next(row for row in curve if row["rounds"] == rounds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", required=True)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.step <= 0 or args.estimators % args.step:
        raise ValueError("estimators must be divisible by a positive step")

    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    cache_dir = os.path.join(output_dir, "iteration_metric_alignment_cache", end)
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
    checkpoints = list(range(args.step, args.estimators + 1, args.step))
    original_save_dir = TrainingConfig.SAVE_DIR
    original_split = TrainingConfig.TRAIN_TEST_SPLIT
    original_params = ModelConfig.XGBOOST_PARAMS.copy()
    started = time.time()
    try:
        TrainingConfig.SAVE_DIR = cache_dir
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators
        ModelConfig.XGBOOST_PARAMS["early_stopping_rounds"] = None

        # Establish one fixed feature set from the final training segment. Its
        # holdout curve is retained but not consulted until historical selection ends.
        final = train_full_and_scan(
            trainer, dataset, list(dataset[3]), 0.8, checkpoints
        )
        selected_features = final["trained_feature_names"]

        closed_dataset, closed_split, _, _ = build_closed_fold(dataset, 0.7, 0.8)
        historical = train_full_and_scan(
            trainer, closed_dataset, selected_features, closed_split, checkpoints
        )
        historical_ic_best = max(
            historical["curve"], key=lambda row: (row["rank_ic"], row["top5_excess"])
        )
        comparison_rounds = sorted(set([
            historical_ic_best["rounds"],
            min(checkpoints, key=lambda value: abs(value - final["ndcg_best_rounds"])),
        ]))
        comparison = [row_at_rounds(final["curve"], rounds) for rounds in comparison_rounds]
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
            "max_estimators": args.estimators,
            "checkpoint_step": args.step,
            "selection_window": "closed 70%-80% fold",
            "confirmation_window": "final 80%-100% holdout",
            "embargo_trading_days": TrainingConfig.FUTURE_DAYS,
            "elapsed_seconds": time.time() - started,
        },
        "historical_selection": historical,
        "historical_rank_ic_selected_rounds": historical_ic_best["rounds"],
        "final_confirmation": final,
        "final_comparison": comparison,
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
