"""在同一验证截面上直接比较两个已归档的训练模型。"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.baostock_config import DATABASE_PATH
from config.factor_config import TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.ml_factor_model import MLFactorModel
from core.factors.train_ml_model import MLModelTrainer


DIAGNOSTIC_FEATURES = {
    "market_cap", "liqaShare", "totalShare", "atr_14", "natr_28",
    "price_volatility_20", "price_volatility_60", "volume_change_rate",
    "volume_volatility", "amount_per_volume",
}


def _validation_start(dates: np.ndarray) -> int:
    raw_split_idx = int(len(dates) * TrainingConfig.TRAIN_TEST_SPLIT)
    split_date = dates[raw_split_idx]
    unique_dates = np.unique(dates)
    split_date_idx = np.searchsorted(unique_dates, split_date)
    val_date_idx = min(
        split_date_idx + TrainingConfig.FUTURE_DAYS, len(unique_dates) - 1
    )
    return int(np.searchsorted(dates, unique_dates[val_date_idx], side="left"))


def _load_model(model_dir: str) -> MLFactorModel:
    path = os.path.join(model_dir, "xgboost_factor_model.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    model = MLFactorModel(model_type="xgboost", task="ranking")
    model.load_model(path)
    return model


def _load_norm_stats(model_dir: str) -> dict:
    path = os.path.join(model_dir, "norm_stats.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as file:
        return pickle.load(file)


def _daily_metrics(predictions, returns, dates) -> pd.DataFrame:
    frame = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "prediction": np.asarray(predictions),
        "return": np.asarray(returns),
    })
    rows = []
    for date, group in frame.groupby("date", sort=True):
        if len(group) < 10:
            continue
        prediction = group["prediction"].to_numpy()
        future_return = group["return"].to_numpy()
        ic = spearmanr(prediction, future_return).statistic
        top_indices = np.argpartition(prediction, -min(5, len(group)))[-5:]
        universe_return = float(np.mean(future_return))
        top5_return = float(np.mean(future_return[top_indices]))
        rows.append({
            "date": date,
            "rank_ic": float(ic) if np.isfinite(ic) else np.nan,
            "top5_return": top5_return,
            "universe_return": universe_return,
            "top5_excess": top5_return - universe_return,
            "sample_count": len(group),
        })
    return pd.DataFrame(rows).set_index("date")


def _summarize_daily(daily: pd.DataFrame) -> dict:
    ic = daily["rank_ic"].dropna()
    ic_std = float(ic.std(ddof=0)) if len(ic) else 0.0
    return {
        "days": int(len(daily)),
        "samples": int(daily["sample_count"].sum()),
        "rank_ic": float(ic.mean()) if len(ic) else 0.0,
        "rank_ic_std": ic_std,
        "rank_ic_ir": float(ic.mean() / ic_std) if ic_std > 0 else 0.0,
        "positive_ic_ratio": float((ic > 0).mean()) if len(ic) else 0.0,
        "top5_return": float(daily["top5_return"].mean()),
        "universe_return": float(daily["universe_return"].mean()),
        "top5_excess": float(daily["top5_excess"].mean()),
    }


def _segment_summaries(daily_by_model: dict) -> dict:
    reference = next(iter(daily_by_model.values()))
    market_trend = reference["universe_return"].rolling(20, min_periods=10).mean()
    lower, upper = market_trend.quantile([1 / 3, 2 / 3])
    regime = pd.Series("sideways", index=reference.index)
    regime.loc[market_trend <= lower] = "down"
    regime.loc[market_trend >= upper] = "up"
    regime.loc[market_trend.isna()] = "insufficient_history"

    result = {"regime_thresholds": {"lower": float(lower), "upper": float(upper)}}
    segment_keys = {
        "year": reference.index.year.astype(str),
        "quarter": reference.index.to_period("Q").astype(str),
        "market_regime": regime,
    }
    for segment_name, keys in segment_keys.items():
        result[segment_name] = {}
        for value in pd.unique(keys):
            mask = np.asarray(keys == value)
            result[segment_name][str(value)] = {
                label: _summarize_daily(daily.loc[mask])
                for label, daily in daily_by_model.items()
            }
    return result


def _write_diagnostics(output_dir, daily_by_model, segments, metadata):
    os.makedirs(output_dir, exist_ok=True)
    combined = pd.concat(daily_by_model, names=["model", "date"])
    combined.to_csv(os.path.join(output_dir, "daily_metrics.csv"))
    payload = {"metadata": metadata, "segments": segments}
    with open(os.path.join(output_dir, "segment_summary.json"), "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _cross_sectional_rank(predictions, dates) -> np.ndarray:
    frame = pd.DataFrame({"date": dates, "prediction": predictions})
    return frame.groupby("date", sort=False)["prediction"].rank(
        method="average", pct=True
    ).to_numpy(dtype=np.float32)


def _diagnostic_feature_names(factor_names: list[str]) -> list[str]:
    """Select a compact set of point-in-time size, liquidity, and risk features."""
    return [name for name in factor_names if name in DIAGNOSTIC_FEATURES]


def _print_segment_comparison(segments, segment_name):
    print(f"\n{segment_name} 分层比较")
    print(f"{'分层':>12} | {'Base IC':>8} | {'Cand IC':>8} | {'ΔIC':>8} | {'Base Ex':>8} | {'Cand Ex':>8}")
    for key, models in segments[segment_name].items():
        baseline = models["baseline"]
        candidate = models["candidate"]
        print(
            f"{key:>12} | {baseline['rank_ic']:>8.4f} | {candidate['rank_ic']:>8.4f} | "
            f"{candidate['rank_ic'] - baseline['rank_ic']:>+8.4f} | "
            f"{baseline['top5_excess']:>8.2%} | {candidate['top5_excess']:>8.2%}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="models/latest")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--stocks", type=int, default=6000)
    parser.add_argument("--years", type=int, default=17)
    parser.add_argument("--end", default=None)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output-dir", default="models/diagnostics/candidate_comparison_20240804")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    end = args.end or (
        datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)
    ).strftime("%Y-%m-%d")
    start = (
        datetime.strptime(end, "%Y-%m-%d") - timedelta(days=365 * args.years)
    ).strftime("%Y-%m-%d")

    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()["code"].tolist()[: args.stocks]
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
        return_sample_metadata=args.save_predictions,
    )
    del stocks_data
    X, y, returns, factor_names, dates, *_ = dataset
    sample_metadata = dataset[10] if args.save_predictions else None
    dates = np.asarray(dates)

    val_start = _validation_start(dates)
    split_idx = np.searchsorted(dates, dates[int(len(dates) * TrainingConfig.TRAIN_TEST_SPLIT)], side="left")
    skip_stats = trainer._apply_cross_sectional_normalization_inplace(
        X[:split_idx], dates[:split_idx], factor_names
    )
    trainer._apply_cross_sectional_normalization_inplace(
        X[val_start:], dates[val_start:], factor_names, skip_col_stats=skip_stats
    )
    X_val = pd.DataFrame(X[val_start:], columns=factor_names)
    y_val = y[val_start:]
    returns_val = returns[val_start:]
    dates_val = dates[val_start:]
    metadata_val = sample_metadata.iloc[val_start:].reset_index(drop=True) if sample_metadata is not None else None

    print(f"验证范围: {np.min(dates_val)} ~ {np.max(dates_val)}, {len(X_val)} 样本")
    summary = []
    daily_by_model = {}
    prediction_ranks = {}
    reference_features = None
    for label, model_dir in (("baseline", args.baseline), ("candidate", args.candidate)):
        model = _load_model(model_dir)
        norm_stats = _load_norm_stats(model_dir)
        if list(norm_stats.get("factor_names", [])) != list(factor_names):
            raise ValueError(f"{label} 的归一化特征顺序与重建数据不一致")
        missing = sorted(set(model.feature_names) - set(X_val.columns))
        if missing:
            raise ValueError(f"{label} 缺失特征: {missing[:20]}")
        if reference_features is None:
            reference_features = list(model.feature_names)
        elif list(model.feature_names) != reference_features:
            print(f"警告: {label} 与 baseline 的入模特征集合或顺序不同，将分别对齐评估")
        metrics = model._evaluate(
            X_val[model.feature_names], y_val, label,
            returns=returns_val, dates=dates_val, sample_ratio=1.0,
        )
        summary.append((label, model_dir, metrics))
        predictions = model.predict(X_val[model.feature_names])
        daily_by_model[label] = _daily_metrics(predictions, returns_val, dates_val)
        if args.save_predictions:
            prediction_ranks[label] = _cross_sectional_rank(predictions, dates_val)

    print("\n模型直接比较")
    print(f"{'模型':>10} | {'IC':>8} | {'ICIR':>8} | {'IC+':>8} | {'Top5':>8} | {'Top5Ex':>8}")
    for label, model_dir, m in summary:
        print(
            f"{label:>10} | {m['rank_ic']:>8.4f} | {m['rank_ic_ir']:>8.4f} | "
            f"{m['positive_ic_ratio']:>8.2%} | {m['top5_mean_return']:>8.2%} | "
            f"{m['top5_excess_return']:>8.2%}  {model_dir}"
        )

    segments = _segment_summaries(daily_by_model)
    for segment_name in ("year", "quarter", "market_regime"):
        _print_segment_comparison(segments, segment_name)
    _write_diagnostics(
        args.output_dir,
        daily_by_model,
        segments,
        {
            "baseline": args.baseline,
            "candidate": args.candidate,
            "start": start,
            "end": end,
            "validation_start": str(np.min(dates_val)),
            "validation_end": str(np.max(dates_val)),
            "validation_samples": len(dates_val),
        },
    )
    if args.save_predictions:
        if metadata_val is None or len(metadata_val) != len(dates_val):
            raise AssertionError("逐样本元数据与验证集长度不一致")
        if not np.array_equal(
            pd.to_datetime(metadata_val["date"]).to_numpy(),
            pd.to_datetime(dates_val).to_numpy(),
        ):
            raise AssertionError("逐样本元数据与验证集日期未对齐")
        prediction_frame = metadata_val.copy()
        prediction_frame["return"] = np.asarray(returns_val, dtype=np.float32)
        for name in _diagnostic_feature_names(factor_names):
            values = np.asarray(X[val_start:, factor_names.index(name)], dtype=np.float32)
            prediction_frame[f"feature_{name}"] = values
            prediction_frame[f"rank_{name}"] = _cross_sectional_rank(values, dates_val)
        prediction_frame = prediction_frame.assign(**{
            **{f"{label}_rank": values for label, values in prediction_ranks.items()},
        })
        prediction_frame.to_parquet(
            os.path.join(args.output_dir, "prediction_ranks.parquet"),
            index=False,
        )
    print(f"\n分层诊断已保存: {args.output_dir}")


if __name__ == "__main__":
    main()
