"""Nested scan for query-level ATR regime weights in XGBoost ranking."""

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


METRICS = ("rank_ic", "rank_ic_ir", "positive_ic_ratio", "top5_excess")


def _fold_end_index(dates, validation_end):
    if validation_end >= 1.0:
        return len(dates), None
    end_date = dates[int(len(dates) * validation_end)]
    return int(np.searchsorted(dates, end_date, side="left")), str(end_date)


def _train_variant(trainer, dataset, feature_names, metadata, train_fraction, validation_end, mode):
    (X, y, returns, factor_names, dates, unbuyable, limit_groups,
     path_scores, is_st, w_sig) = dataset[:10]
    end_idx, end_date = _fold_end_index(dates, validation_end)
    raw_split_idx = int(len(dates[:end_idx]) * train_fraction / validation_end)
    split_date = dates[raw_split_idx]
    split_idx = int(np.searchsorted(dates[:end_idx], split_date, side="left"))
    if not 0 < split_idx < end_idx:
        raise ValueError("Invalid fold")

    index = {name: idx for idx, name in enumerate(factor_names)}
    keep_idx = [index[name] for name in feature_names]
    fold_dataset = (
        X[:end_idx, :][:, keep_idx].copy(),
        y[:end_idx].copy(),
        returns[:end_idx],
        list(feature_names),
        dates[:end_idx],
        unbuyable[:end_idx],
        limit_groups[:end_idx],
        path_scores[:end_idx],
        is_st[:end_idx],
        w_sig[:end_idx],
    )
    query_regime_values = metadata["atr_rel"].to_numpy(dtype=np.float64)[:end_idx]
    original_split = TrainingConfig.TRAIN_TEST_SPLIT
    original_mode = TrainingConfig.QUERY_ATR_REGIME_WEIGHT
    try:
        TrainingConfig.TRAIN_TEST_SPLIT = split_idx / end_idx
        TrainingConfig.QUERY_ATR_REGIME_WEIGHT = mode
        results = trainer.train_models(
            fold_dataset[0],
            fold_dataset[1],
            fold_dataset[2],
            fold_dataset[3],
            fold_dataset[4],
            unbuyable_mask=fold_dataset[5],
            limit_groups=fold_dataset[6],
            path_scores=fold_dataset[7],
            is_st_arr=fold_dataset[8],
            w_sig_arr=fold_dataset[9],
            query_regime_values=query_regime_values,
            model_types=["xgboost"],
            val_eval_sample_ratio=1.0,
        )
        model = trainer.models["xgboost"]
        metrics = results["xgboost"]["val_metrics"]
        return {
            "mode": mode,
            "split_date": str(split_date),
            "validation_end_date_exclusive": end_date,
            "features_trained": len(model.feature_names),
            "trained_feature_names": list(model.feature_names),
            "best_iteration": model._get_xgb_best_iteration(model.model),
            "rank_ic": metrics.get("rank_ic"),
            "rank_ic_ir": metrics.get("rank_ic_ir"),
            "positive_ic_ratio": metrics.get("positive_ic_ratio"),
            "top5_excess": metrics.get("top5_excess_return"),
            "top5_return": metrics.get("top5_mean_return"),
        }
    finally:
        TrainingConfig.TRAIN_TEST_SPLIT = original_split
        TrainingConfig.QUERY_ATR_REGIME_WEIGHT = original_mode


def _compare(row, baseline):
    deltas = {key: row[key] - baseline[key] for key in METRICS}
    passes = all(value >= 0.0 for value in deltas.values()) and any(
        value > 0.0 for value in deltas.values()
    )
    return deltas, passes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", required=True)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--modes", default="off,high,low")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    modes = args.modes.split(",")
    if "off" not in modes:
        modes.insert(0, "off")

    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    cache_dir = os.path.join(output_dir, "query_atr_regime_weight_cache", end)
    os.makedirs(cache_dir, exist_ok=True)
    started = time.time()

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
        return_sample_metadata=True,
    )
    sample_metadata = dataset[10]

    original_save_dir = TrainingConfig.SAVE_DIR
    original_params = ModelConfig.XGBOOST_PARAMS.copy()
    original_dispersion = TrainingConfig.QUERY_LABEL_DISPERSION_WEIGHT
    original_sample_weight = TrainingConfig.USE_SAMPLE_WEIGHT
    try:
        TrainingConfig.SAVE_DIR = cache_dir
        TrainingConfig.QUERY_LABEL_DISPERSION_WEIGHT = "off"
        TrainingConfig.USE_SAMPLE_WEIGHT = False
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators
        selection_rows = []
        selected_features = list(dataset[3])
        baseline = None
        for mode in modes:
            row = _train_variant(
                trainer, dataset, selected_features, sample_metadata,
                train_fraction=0.7, validation_end=0.8, mode=mode,
            )
            if mode == "off":
                baseline = row
                selected_features = row["trained_feature_names"]
                row["deltas_vs_baseline"] = {key: 0.0 for key in METRICS}
                row["passes_gate"] = False
            else:
                deltas, passes = _compare(row, baseline)
                row["deltas_vs_baseline"] = deltas
                row["passes_gate"] = passes
            selection_rows.append(row)

        eligible = [row for row in selection_rows if row["passes_gate"]]
        selected = max(
            eligible or [baseline],
            key=lambda row: (row["passes_gate"], row["top5_excess"], row["rank_ic_ir"]),
        )
        if selected["passes_gate"]:
            confirmation_baseline = _train_variant(
                trainer, dataset, selected_features, sample_metadata,
                train_fraction=0.8, validation_end=1.0, mode="off",
            )
            confirmation_candidate = _train_variant(
                trainer, dataset, selected_features, sample_metadata,
                train_fraction=0.8, validation_end=1.0, mode=selected["mode"],
            )
            deltas, passes = _compare(confirmation_candidate, confirmation_baseline)
            confirmation_candidate["deltas_vs_baseline"] = deltas
            confirmation_candidate["passes_gate"] = passes
        else:
            confirmation_baseline = None
            confirmation_candidate = None
    finally:
        TrainingConfig.SAVE_DIR = original_save_dir
        TrainingConfig.QUERY_LABEL_DISPERSION_WEIGHT = original_dispersion
        TrainingConfig.USE_SAMPLE_WEIGHT = original_sample_weight
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
            "modes": modes,
            "selection_window": "70%-80% closed fold",
            "confirmation_window": "80%-100% final fold",
            "embargo_trading_days": TrainingConfig.FUTURE_DAYS,
            "elapsed_seconds": time.time() - started,
        },
        "selection_results": selection_rows,
        "selected": selected,
        "confirmation_baseline": confirmation_baseline,
        "confirmation_candidate": confirmation_candidate,
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
