"""Diagnose XGBoost top-label predictability with embargoed time-fold OOF predictions."""

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
from core.factors.train_ml_model import MLModelTrainer


def _daily_percentiles(values, dates):
    result = np.empty(len(values), dtype=np.float64)
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    for start, count in zip(starts, counts):
        order = np.argsort(values[start:start + count], kind="mergesort")
        ranks = np.empty(count, dtype=np.float64)
        ranks[order] = np.arange(1, count + 1, dtype=np.float64)
        result[start:start + count] = ranks / (count + 1.0)
    return result


def _fold_dataset(dataset, train_fraction, validation_end_fraction):
    dates = dataset[4]
    unique_dates = np.unique(dates)
    train_date = dates[int(len(dates) * train_fraction)]
    split_idx = int(np.searchsorted(dates, train_date, side="left"))
    split_date_idx = int(np.searchsorted(unique_dates, train_date))
    val_date_idx = min(split_date_idx + TrainingConfig.FUTURE_DAYS, len(unique_dates) - 1)
    val_start_date = unique_dates[val_date_idx]
    val_start_idx = int(np.searchsorted(dates, val_start_date, side="left"))
    if validation_end_fraction >= 1.0:
        end_idx = len(dates)
        end_date = None
    else:
        end_date = dates[int(len(dates) * validation_end_fraction)]
        end_idx = int(np.searchsorted(dates, end_date, side="left"))
    if not 0 < split_idx < val_start_idx < end_idx:
        raise ValueError("Invalid fold boundaries")
    sliced = [value if index == 3 else value[:end_idx] for index, value in enumerate(dataset[:10])]
    sliced[0] = sliced[0].copy()
    return tuple(sliced), split_idx / end_idx, val_start_idx, end_idx, str(train_date), (
        None if end_date is None else str(end_date)
    )


def _head_metrics(predictions, scores, returns, dates, fraction):
    pred_pct = _daily_percentiles(predictions, dates)
    true_pct = _daily_percentiles(scores, dates)
    pred_top = pred_pct >= 1.0 - fraction
    true_top = true_pct >= 1.0 - fraction
    daily = []
    for date in np.unique(dates):
        mask = dates == date
        p = pred_top[mask]
        t = true_top[mask]
        ret = returns[mask]
        intersection = int(np.sum(p & t))
        predicted = int(np.sum(p))
        actual = int(np.sum(t))
        daily.append({
            "precision": intersection / predicted if predicted else 0.0,
            "recall": intersection / actual if actual else 0.0,
            "jaccard": intersection / int(np.sum(p | t)) if np.any(p | t) else 0.0,
            "predicted_return": float(np.mean(ret[p])),
            "oracle_return": float(np.mean(ret[t])),
            "universe_return": float(np.mean(ret)),
            "predicted_true_percentile": float(np.mean(true_pct[mask][p])),
        })
    result = {
        key: float(np.mean([row[key] for row in daily]))
        for key in daily[0]
    }
    result["days"] = len(daily)
    result["predicted_excess"] = result["predicted_return"] - result["universe_return"]
    result["oracle_excess"] = result["oracle_return"] - result["universe_return"]
    result["oracle_capture_ratio"] = (
        result["predicted_excess"] / result["oracle_excess"]
        if result["oracle_excess"] else None
    )
    return result, pred_top, true_top, pred_pct, true_pct


def _feature_differences(X, feature_names, group_a, group_b, limit=20):
    if not np.any(group_a) or not np.any(group_b):
        return []
    mean_a = np.mean(X[group_a], axis=0)
    mean_b = np.mean(X[group_b], axis=0)
    difference = mean_a - mean_b
    order = np.argsort(np.abs(difference))[::-1][:limit]
    return [{
        "feature": feature_names[index],
        "mean_a": float(mean_a[index]),
        "mean_b": float(mean_b[index]),
        "difference": float(difference[index]),
    } for index in order]


def run_fold(trainer, dataset, feature_names, train_fraction, validation_end, head_fraction):
    fold, split, val_start, end_idx, split_date, end_date = _fold_dataset(
        dataset, train_fraction, validation_end
    )
    X, y, returns, all_names, dates, unbuyable, limits, scores, is_st, w_sig = fold
    feature_index = {name: index for index, name in enumerate(all_names)}
    indices = [feature_index[name] for name in feature_names]
    X = X[:, indices]
    TrainingConfig.TRAIN_TEST_SPLIT = split
    results = trainer.train_models(
        X, y.copy(), returns, list(feature_names), dates,
        unbuyable_mask=unbuyable, limit_groups=limits, path_scores=scores,
        is_st_arr=is_st, w_sig_arr=w_sig, model_types=["xgboost"],
        val_eval_sample_ratio=1.0,
    )
    model = trainer.models["xgboost"]
    trained_indices = [feature_names.index(name) for name in model.feature_names]
    local_val_start = val_start
    X_val = X[local_val_start:, trained_indices]
    dates_val = dates[local_val_start:]
    returns_val = returns[local_val_start:]
    scores_val = scores[local_val_start:]
    dval = xgb.DMatrix(X_val, feature_names=model.feature_names)
    predictions = model._predict_xgb_booster(model.model, dval)
    metrics, pred_top, true_top, pred_pct, true_pct = _head_metrics(
        predictions, scores_val, returns_val, dates_val, head_fraction
    )
    true_positive = pred_top & true_top
    false_positive = pred_top & ~true_top
    false_negative = ~pred_top & true_top
    return {
        "train_fraction": train_fraction,
        "validation_end_fraction": validation_end,
        "split_date": split_date,
        "validation_end_date_exclusive": end_date,
        "validation_start_date": str(dates_val[0]),
        "validation_end_date": str(dates_val[-1]),
        "validation_samples": len(dates_val),
        "validation_days": len(np.unique(dates_val)),
        "features_trained": len(model.feature_names),
        "trained_feature_names": list(model.feature_names),
        "best_iteration": model._get_xgb_best_iteration(model.model),
        "trainer_metrics": results["xgboost"]["val_metrics"],
        "head_metrics": metrics,
        "head_counts": {
            "true_positive": int(np.sum(true_positive)),
            "false_positive": int(np.sum(false_positive)),
            "false_negative": int(np.sum(false_negative)),
        },
        "false_positive_minus_true_positive_features": _feature_differences(
            X_val, model.feature_names, false_positive, true_positive
        ),
        "false_negative_minus_true_positive_features": _feature_differences(
            X_val, model.feature_names, false_negative, true_positive
        ),
        "prediction_label_rank_correlation": float(np.corrcoef(pred_pct, true_pct)[0, 1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", required=True)
    parser.add_argument("--folds", default="0.6:0.7,0.7:0.8,0.8:1.0")
    parser.add_argument("--head-fraction", type=float, default=0.05)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    folds = [tuple(map(float, item.split(":"))) for item in args.folds.split(",")]
    if any(not 0 < start < end <= 1 for start, end in folds):
        raise ValueError("Each fold must satisfy 0 < train < validation end <= 1")

    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    cache_dir = os.path.join(output_dir, "oof_head_cache", end)
    os.makedirs(cache_dir, exist_ok=True)
    started = time.time()

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
    original_split = TrainingConfig.TRAIN_TEST_SPLIT
    original_params = ModelConfig.XGBOOST_PARAMS.copy()
    try:
        TrainingConfig.SAVE_DIR = cache_dir
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators
        rows = []
        selected_features = list(dataset[3])
        for train_fraction, validation_end in folds:
            row = run_fold(
                trainer, dataset, selected_features, train_fraction,
                validation_end, args.head_fraction,
            )
            if len(selected_features) == len(dataset[3]):
                selected_features = row["trained_feature_names"]
            rows.append(row)
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
            "head_fraction": args.head_fraction,
            "folds": folds,
            "embargo_trading_days": TrainingConfig.FUTURE_DAYS,
            "elapsed_seconds": time.time() - started,
        },
        "folds": rows,
    }
    keys = list(rows[0]["head_metrics"])
    payload["aggregate_head_metrics"] = {
        key: float(np.mean([row["head_metrics"][key] for row in rows]))
        for key in keys if key != "days" and all(
            row["head_metrics"][key] is not None for row in rows
        )
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
