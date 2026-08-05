"""Nested test for blending XGBoost prediction rank with head-recall features."""

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
from scripts.diagnose_xgb_oof_head import _daily_percentiles, _fold_dataset


HEAD_FEATURES = [
    "atr_14",
    "hl_range_std",
    "plus_di",
    "atr_14_mul_mfi_15",
    "amount_std_30",
    "bb_width",
    "br_26",
    "ar_26",
    "ma_slope_14",
    "amount_per_volume",
    "oc_ratio_mean",
]


def _daily_metrics(predictions, returns, dates):
    rank_ics = []
    top5_excess = []
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    for start, count in zip(starts, counts):
        if count < 10:
            continue
        end = start + count
        pred = predictions[start:end]
        ret = returns[start:end]
        pred_rank = _fast_rankdata_1d(pred)
        ret_rank = _fast_rankdata_1d(ret)
        ic = np.corrcoef(pred_rank, ret_rank)[0, 1]
        if np.isfinite(ic):
            rank_ics.append(float(ic))
        top = np.argpartition(pred, -min(5, count))[-min(5, count):]
        top5_excess.append(float(ret[top].mean() - ret.mean()))
    rank_ics = np.asarray(rank_ics, dtype=np.float64)
    top5_excess = np.asarray(top5_excess, dtype=np.float64)
    std = float(rank_ics.std())
    return {
        "rank_ic": float(rank_ics.mean()),
        "rank_ic_std": std,
        "rank_ic_ir": float(rank_ics.mean() / std) if std else 0.0,
        "positive_ic_ratio": float(np.mean(rank_ics > 0)),
        "top5_excess": float(np.mean(top5_excess)),
        "days": int(len(rank_ics)),
    }


def _compare(row, baseline):
    keys = ("rank_ic", "rank_ic_ir", "positive_ic_ratio", "top5_excess")
    deltas = {key: row[key] - baseline[key] for key in keys}
    passes = all(value >= 0 for value in deltas.values()) and any(
        value > 0 for value in deltas.values()
    )
    return deltas, passes


def _train_and_predict(trainer, dataset, feature_names, train_fraction, validation_end):
    fold, split, val_start, _, split_date, end_date = _fold_dataset(
        dataset, train_fraction, validation_end
    )
    X, y, returns, all_names, dates, unbuyable, limits, scores, is_st, w_sig = fold
    index = {name: idx for idx, name in enumerate(all_names)}
    selected_idx = [index[name] for name in feature_names]
    X = X[:, selected_idx].copy()
    TrainingConfig.TRAIN_TEST_SPLIT = split
    results = trainer.train_models(
        X, y.copy(), returns, list(feature_names), dates,
        unbuyable_mask=unbuyable,
        limit_groups=limits,
        path_scores=scores,
        is_st_arr=is_st,
        w_sig_arr=w_sig,
        model_types=["xgboost"],
        val_eval_sample_ratio=1.0,
    )
    model = trainer.models["xgboost"]
    trained_idx = [feature_names.index(name) for name in model.feature_names]
    X_val = X[val_start:, :][:, trained_idx]
    dates_val = dates[val_start:]
    returns_val = returns[val_start:]
    dval = xgb.DMatrix(X_val, feature_names=model.feature_names)
    predictions = model._predict_xgb_booster(model.model, dval)
    available = [name for name in HEAD_FEATURES if name in model.feature_names]
    feature_indices = [model.feature_names.index(name) for name in available]
    head_score = np.mean(X_val[:, feature_indices], axis=1) if feature_indices else np.zeros(len(X_val))
    return {
        "split_date": split_date,
        "validation_end_date_exclusive": end_date,
        "validation_start_date": str(dates_val[0]),
        "validation_end_date": str(dates_val[-1]),
        "validation_samples": int(len(dates_val)),
        "trained_feature_names": list(model.feature_names),
        "available_head_features": available,
        "best_iteration": model._get_xgb_best_iteration(model.model),
        "trainer_metrics": results["xgboost"]["val_metrics"],
        "pred_rank": _daily_percentiles(predictions, dates_val),
        "head_rank": _daily_percentiles(head_score, dates_val),
        "returns": returns_val,
        "dates": dates_val,
    }


def _scan(fold_payload, weights):
    rows = []
    for weight in weights:
        pred = (1.0 - weight) * fold_payload["pred_rank"] + weight * fold_payload["head_rank"]
        row = _daily_metrics(pred, fold_payload["returns"], fold_payload["dates"])
        row["head_feature_weight"] = float(weight)
        rows.append(row)
    baseline = next(row for row in rows if row["head_feature_weight"] == 0.0)
    for row in rows:
        deltas, passes = _compare(row, baseline)
        row["deltas_vs_baseline"] = deltas
        row["passes_gate"] = passes
    return rows


def _strip_arrays(payload):
    return {
        key: value for key, value in payload.items()
        if key not in {"pred_rank", "head_rank", "returns", "dates"}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", required=True)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--weights", default="0,0.02,0.05,0.1,0.15,0.2,0.3")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    weights = [float(value) for value in args.weights.split(",")]
    if 0.0 not in weights:
        weights.insert(0, 0.0)

    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    cache_dir = os.path.join(output_dir, "head_feature_blend_cache", end)
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
        selection = _train_and_predict(trainer, dataset, list(dataset[3]), 0.7, 0.8)
        feature_names = selection["trained_feature_names"]
        selection_rows = _scan(selection, weights)
        eligible = [row for row in selection_rows if row["passes_gate"]]
        selected = max(
            eligible or [next(row for row in selection_rows if row["head_feature_weight"] == 0.0)],
            key=lambda row: (row["passes_gate"], row["top5_excess"], row["rank_ic_ir"]),
        )
        confirmation = _train_and_predict(trainer, dataset, feature_names, 0.8, 1.0)
        confirmation_rows = _scan(confirmation, weights)
        confirmation_selected = next(
            row for row in confirmation_rows
            if row["head_feature_weight"] == selected["head_feature_weight"]
        )
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
            "weights": weights,
            "head_features": HEAD_FEATURES,
            "selection_window": "70%-80% closed fold",
            "confirmation_window": "80%-100% final fold",
            "embargo_trading_days": TrainingConfig.FUTURE_DAYS,
            "elapsed_seconds": time.time() - started,
        },
        "selection_fold": _strip_arrays(selection),
        "selection_results": selection_rows,
        "selected": selected,
        "confirmation_fold": _strip_arrays(confirmation),
        "confirmation_results": confirmation_rows,
        "confirmation_selected": confirmation_selected,
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
