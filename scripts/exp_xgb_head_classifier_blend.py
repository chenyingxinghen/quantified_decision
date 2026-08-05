"""Nested test for blending XGBoost ranker score with a Top-5 classifier."""

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
from scripts.diagnose_xgb_oof_head import _daily_percentiles


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


def _top_fraction_labels(scores, dates, fraction=0.05):
    labels = np.zeros(len(scores), dtype=np.int32)
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    for start, count in zip(starts, counts):
        values = scores[start:start + count]
        k = max(1, int(np.ceil(count * fraction)))
        order = np.argsort(values)
        day_labels = labels[start:start + count]
        day_labels[order[-k:]] = 1
        labels[start:start + count] = day_labels
    return labels


def _compare(row, baseline):
    keys = ("rank_ic", "rank_ic_ir", "positive_ic_ratio", "top5_excess")
    deltas = {key: row[key] - baseline[key] for key in keys}
    passes = all(value >= 0 for value in deltas.values()) and any(
        value > 0 for value in deltas.values()
    )
    return deltas, passes


def _train_ranker_and_classifier(trainer, dataset, feature_names, train_fraction, validation_end):
    dates = dataset[4]
    unique_dates = np.unique(dates)
    raw_split_idx = int(len(dates) * train_fraction)
    split_date = dates[raw_split_idx]
    split_idx = int(np.searchsorted(dates, split_date, side="left"))
    split_date_idx = int(np.searchsorted(unique_dates, split_date))
    val_start_date = unique_dates[min(split_date_idx + TrainingConfig.FUTURE_DAYS, len(unique_dates) - 1)]
    val_start_idx = int(np.searchsorted(dates, val_start_date, side="left"))
    if validation_end >= 1.0:
        end_idx = len(dates)
        end_date = None
    else:
        end_date = dates[int(len(dates) * validation_end)]
        end_idx = int(np.searchsorted(dates, end_date, side="left"))
    if not 0 < split_idx < val_start_idx < end_idx:
        raise ValueError("Invalid fold boundaries")

    sliced = [value if index == 3 else value[:end_idx] for index, value in enumerate(dataset[:10])]
    X, y, returns, all_names, dates, unbuyable, limits, scores, is_st, w_sig = sliced
    index = {name: idx for idx, name in enumerate(all_names)}
    selected_idx = [index[name] for name in feature_names]
    X = X[:, selected_idx].copy()

    TrainingConfig.TRAIN_TEST_SPLIT = split_idx / end_idx
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
    ranker = trainer.models["xgboost"]
    trained_idx = [feature_names.index(name) for name in ranker.feature_names]
    X_val = X[val_start_idx:, :][:, trained_idx]
    dates_val = dates[val_start_idx:]
    returns_val = returns[val_start_idx:]
    scores_val = scores[val_start_idx:]

    label_train = _top_fraction_labels(scores[:split_idx], dates[:split_idx], 0.05)
    label_val = _top_fraction_labels(scores[val_start_idx:end_idx], dates[val_start_idx:end_idx], 0.05)
    X_train_cls = X[:split_idx, :][:, trained_idx]
    cls_params = {
        "n_estimators": int(ModelConfig.XGBOOST_PARAMS.get("n_estimators", 500)),
        "learning_rate": float(ModelConfig.XGBOOST_PARAMS.get("learning_rate", 0.04)),
        "max_depth": int(ModelConfig.XGBOOST_PARAMS.get("max_depth", 3)),
        "min_child_weight": float(ModelConfig.XGBOOST_PARAMS.get("min_child_weight", 1.0)),
        "subsample": float(ModelConfig.XGBOOST_PARAMS.get("subsample", 1.0)),
        "colsample_bytree": float(ModelConfig.XGBOOST_PARAMS.get("colsample_bytree", 1.0)),
        "gamma": float(ModelConfig.XGBOOST_PARAMS.get("gamma", 0.0)),
        "reg_alpha": float(ModelConfig.XGBOOST_PARAMS.get("reg_alpha", 0.0)),
        "reg_lambda": float(ModelConfig.XGBOOST_PARAMS.get("reg_lambda", 1.0)),
        "tree_method": ModelConfig.XGBOOST_PARAMS.get("tree_method", "hist"),
        "device": ModelConfig.XGBOOST_PARAMS.get("device", "cuda"),
        "random_state": 42,
        "n_jobs": int(ModelConfig.XGBOOST_PARAMS.get("n_jobs", 15)),
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": float((label_train == 0).sum() / max((label_train == 1).sum(), 1)),
        "max_delta_step": 1,
        "early_stopping_rounds": 200,
    }
    classifier = xgb.XGBClassifier(**cls_params)
    classifier.fit(
        X_train_cls,
        label_train,
        eval_set=[(X_val, label_val)],
        verbose=False,
    )
    rank_dmat = xgb.DMatrix(X_val, feature_names=ranker.feature_names)
    rank_pred = ranker._predict_xgb_booster(ranker.model, rank_dmat)
    cls_pred = classifier.predict_proba(X_val)[:, 1]
    return {
        "split_date": split_date,
        "validation_end_date_exclusive": end_date,
        "validation_start_date": str(dates_val[0]),
        "validation_end_date": str(dates_val[-1]),
        "validation_samples": int(len(dates_val)),
        "trained_feature_names": list(ranker.feature_names),
        "best_iteration": ranker._get_xgb_best_iteration(ranker.model),
        "classifier_best_iteration": getattr(classifier, "best_iteration", None),
        "trainer_metrics": results["xgboost"]["val_metrics"],
        "rank_pred": _daily_percentiles(rank_pred, dates_val),
        "cls_pred": _daily_percentiles(cls_pred, dates_val),
        "returns": returns_val,
        "dates": dates_val,
    }


def _scan(fold_payload, weights):
    rows = []
    for weight in weights:
        pred = (1.0 - weight) * fold_payload["rank_pred"] + weight * fold_payload["cls_pred"]
        row = _daily_metrics(pred, fold_payload["returns"], fold_payload["dates"])
        row["classifier_weight"] = float(weight)
        rows.append(row)
    baseline = next(row for row in rows if row["classifier_weight"] == 0.0)
    for row in rows:
        deltas, passes = _compare(row, baseline)
        row["deltas_vs_baseline"] = deltas
        row["passes_gate"] = passes
    return rows


def _strip_arrays(payload):
    return {key: value for key, value in payload.items() if key not in {"rank_pred", "cls_pred", "returns", "dates"}}


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
    cache_dir = os.path.join(output_dir, "head_classifier_blend_cache", end)
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
        selection = _train_ranker_and_classifier(trainer, dataset, list(dataset[3]), 0.7, 0.8)
        feature_names = selection["trained_feature_names"]
        selection_rows = _scan(selection, weights)
        eligible = [row for row in selection_rows if row["passes_gate"]]
        selected = max(
            eligible or [next(row for row in selection_rows if row["classifier_weight"] == 0.0)],
            key=lambda row: (row["passes_gate"], row["top5_excess"], row["rank_ic_ir"]),
        )
        if selected["passes_gate"]:
            confirmation = _train_ranker_and_classifier(trainer, dataset, feature_names, 0.8, 1.0)
            confirmation_rows = _scan(confirmation, weights)
            confirmation_selected = next(
                row for row in confirmation_rows
                if row["classifier_weight"] == selected["classifier_weight"]
            )
        else:
            confirmation = None
            confirmation_rows = []
            confirmation_selected = None
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
            "selection_window": "70%-80% closed fold",
            "confirmation_window": "80%-100% final fold",
            "embargo_trading_days": TrainingConfig.FUTURE_DAYS,
            "elapsed_seconds": time.time() - started,
        },
        "selection_fold": _strip_arrays(selection),
        "selection_results": selection_rows,
        "selected": selected,
        "confirmation_fold": None if confirmation is None else _strip_arrays(confirmation),
        "confirmation_results": confirmation_rows,
        "confirmation_selected": confirmation_selected,
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
