"""Nested test for Top-5 classifier reranking only inside ranker head buckets."""

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
from scripts.exp_xgb_head_classifier_blend import (
    _compare,
    _daily_metrics,
    _strip_arrays,
    _train_ranker_and_classifier,
)


def _rerank_inside_head(rank_pred, cls_pred, threshold, weight):
    blended = rank_pred.copy()
    mask = rank_pred >= threshold
    local = (1.0 - weight) * rank_pred[mask] + weight * cls_pred[mask]
    blended[mask] = threshold + (1.0 - threshold) * local
    return blended


def _scan(fold_payload, thresholds, weights):
    rows = []
    baseline = _daily_metrics(
        fold_payload["rank_pred"], fold_payload["returns"], fold_payload["dates"]
    )
    baseline.update({
        "mode": "baseline",
        "head_threshold": None,
        "classifier_weight": 0.0,
    })
    rows.append(baseline)
    for threshold in thresholds:
        for weight in weights:
            if weight <= 0:
                continue
            pred = _rerank_inside_head(
                fold_payload["rank_pred"], fold_payload["cls_pred"],
                threshold, weight,
            )
            row = _daily_metrics(pred, fold_payload["returns"], fold_payload["dates"])
            row.update({
                "mode": "head_bucket_rerank",
                "head_threshold": float(threshold),
                "classifier_weight": float(weight),
            })
            rows.append(row)
    for row in rows:
        deltas, passes = _compare(row, baseline)
        row["deltas_vs_baseline"] = deltas
        row["passes_gate"] = passes
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", required=True)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--thresholds", default="0.8,0.9")
    parser.add_argument("--weights", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    thresholds = [float(value) for value in args.thresholds.split(",")]
    weights = [float(value) for value in args.weights.split(",")]
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    cache_dir = os.path.join(output_dir, "head_classifier_rerank_cache", end)
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
        selection = _train_ranker_and_classifier(
            trainer, dataset, list(dataset[3]), 0.7, 0.8
        )
        feature_names = selection["trained_feature_names"]
        selection_rows = _scan(selection, thresholds, weights)
        eligible = [row for row in selection_rows if row["passes_gate"]]
        selected = max(
            eligible or [selection_rows[0]],
            key=lambda row: (row["passes_gate"], row["top5_excess"], row["rank_ic_ir"]),
        )
        if selected["passes_gate"]:
            confirmation = _train_ranker_and_classifier(
                trainer, dataset, feature_names, 0.8, 1.0
            )
            confirmation_rows = _scan(confirmation, thresholds, weights)
            confirmation_selected = next(
                row for row in confirmation_rows
                if row["mode"] == selected["mode"]
                and row["head_threshold"] == selected["head_threshold"]
                and row["classifier_weight"] == selected["classifier_weight"]
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
            "thresholds": thresholds,
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
