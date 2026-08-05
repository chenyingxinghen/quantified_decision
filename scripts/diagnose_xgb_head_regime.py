"""Diagnose OOF head misses by market regime and sample metadata."""

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


def _bucket_summary(values, masks):
    return [
        {
            "bucket": index + 1,
            "samples": int(mask.sum()),
            "share": float(mask.mean()),
            "hit_rate": float(np.mean(values[mask])) if mask.any() else None,
        }
        for index, mask in enumerate(masks)
    ]


def _quantile_masks(values, cuts):
    edges = np.quantile(values, cuts)
    masks = []
    lower = -np.inf
    for edge in list(edges) + [np.inf]:
        masks.append((values > lower) & (values <= edge))
        lower = edge
    return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-end-fraction", type=float, default=0.8)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    cache_dir = os.path.join(output_dir, "head_regime_cache", end)
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

    (X, y, returns, factor_names, dates, unbuyable, limit_groups,
     path_scores, is_st, w_sig, sample_metadata) = dataset
    dates = np.asarray(dates)
    raw_split_idx = int(len(dates) * args.train_fraction)
    split_date = dates[raw_split_idx]
    split_idx = int(np.searchsorted(dates, split_date, side="left"))
    unique_dates = np.unique(dates)
    split_date_idx = int(np.searchsorted(unique_dates, split_date))
    val_start_date = unique_dates[min(split_date_idx + TrainingConfig.FUTURE_DAYS, len(unique_dates) - 1)]
    val_start_idx = int(np.searchsorted(dates, val_start_date, side="left"))
    if not 0 < split_idx < val_start_idx:
        raise ValueError("Invalid split")

    TrainingConfig.SAVE_DIR = cache_dir
    original_params = ModelConfig.XGBOOST_PARAMS.copy()
    try:
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators
        X_train = X[:split_idx].copy()
        X_val = X[val_start_idx:].copy()
        trainer._apply_cross_sectional_normalization_inplace(X_train, dates[:split_idx], factor_names)
        trainer._apply_cross_sectional_normalization_inplace(X_val, dates[val_start_idx:], factor_names)
        results = trainer.train_models(
            X.copy(), y.copy(), returns, factor_names, dates,
            unbuyable_mask=unbuyable, limit_groups=limit_groups, path_scores=path_scores,
            is_st_arr=is_st, w_sig_arr=w_sig, model_types=["xgboost"], val_eval_sample_ratio=1.0,
        )
        model = trainer.models["xgboost"]
        trained_idx = [factor_names.index(name) for name in model.feature_names]
        X_val = X_val[:, trained_idx]
        dval = xgb.DMatrix(X_val, feature_names=model.feature_names)
        pred = model._predict_xgb_booster(model.model, dval)
        pred_pct = _daily_percentiles(pred, dates[val_start_idx:])
        true_pct = _daily_percentiles(path_scores[val_start_idx:], dates[val_start_idx:])
        pred_top = pred_pct >= 0.95
        true_top = true_pct >= 0.95
        fn = ~pred_top & true_top
        fp = pred_top & ~true_top
        tp = pred_top & true_top
        meta = sample_metadata.iloc[val_start_idx:].reset_index(drop=True)

        regime_columns = ["atr_rel", "intraday_intensity", "volume_ratio", "relative_intensity"]
        regime = {}
        for col in regime_columns:
            values = np.asarray(meta[col], dtype=np.float64)
            regime[col] = {
                "quantile_cuts": [0.2, 0.4, 0.6, 0.8],
                "false_negative": _bucket_summary(fn.astype(float), _quantile_masks(values, [0.2, 0.4, 0.6, 0.8])),
                "false_positive": _bucket_summary(fp.astype(float), _quantile_masks(values, [0.2, 0.4, 0.6, 0.8])),
                "true_positive": _bucket_summary(tp.astype(float), _quantile_masks(values, [0.2, 0.4, 0.6, 0.8])),
            }

        daily_meta = meta.groupby("date", sort=True)[regime_columns].mean(numeric_only=True).reset_index()
        day_values = np.asarray(daily_meta["atr_rel"], dtype=np.float64)
        day_labels = np.asarray(daily_meta["date"], dtype=str)
        day_masks = _quantile_masks(day_values, [0.2, 0.4, 0.6, 0.8])
        day_table = []
        for index, mask in enumerate(day_masks):
            dates_in_bucket = day_labels[mask]
            sample_mask = np.isin(meta["date"].astype(str).to_numpy(), dates_in_bucket)
            day_table.append({
                "bucket": index + 1,
                "dates": int(mask.sum()),
                "samples": int(sample_mask.sum()),
                "fn_rate": float(np.mean(fn[sample_mask])) if sample_mask.any() else None,
                "fp_rate": float(np.mean(fp[sample_mask])) if sample_mask.any() else None,
                "tp_rate": float(np.mean(tp[sample_mask])) if sample_mask.any() else None,
            })

        payload = {
            "metadata": {
                "stocks_requested": args.stocks,
                "stocks_loaded": len(stocks_data),
                "start": start,
                "end": end,
                "samples": len(dates),
                "validation_samples": int(len(dates) - val_start_idx),
                "validation_start_date": str(val_start_date),
                "validation_end_date": str(dates[-1]),
                "split_date": str(split_date),
                "embargo_trading_days": TrainingConfig.FUTURE_DAYS,
                "elapsed_seconds": time.time() - started,
            },
            "metrics": results["xgboost"]["val_metrics"],
            "head_masks": {
                "pred_top_count": int(pred_top.sum()),
                "true_top_count": int(true_top.sum()),
                "tp": int(tp.sum()),
                "fp": int(fp.sum()),
                "fn": int(fn.sum()),
                "precision": float(tp.sum() / pred_top.sum()) if pred_top.sum() else None,
                "recall": float(tp.sum() / true_top.sum()) if true_top.sum() else None,
            },
            "regime": regime,
            "daily_atr_rel_regime": day_table,
        }
    finally:
        ModelConfig.XGBOOST_PARAMS.clear()
        ModelConfig.XGBOOST_PARAMS.update(original_params)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
