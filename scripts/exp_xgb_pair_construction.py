"""控制变量实验：比较 XGBoost LambdaRank 的 top-k pair 构造范围。"""

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


def _effective_pair_config(model) -> dict:
    config = json.loads(model.model.save_config())
    return config["learner"]["objective"]["lambdarank_param"]


def main():
    stock_count = int(os.environ.get("EXP_STOCKS", "800"))
    years = int(os.environ.get("EXP_YEARS", "8"))
    end = os.environ.get("EXP_END", "2024-08-04")
    estimators = int(os.environ.get("EXP_ESTIMATORS", "500"))
    pair_specs = [
        value.strip()
        for value in os.environ.get(
            "EXP_PAIR_SPECS",
            os.environ.get("EXP_PAIR_LIMITS", "default,topk:5,topk:20,topk:50"),
        ).split(",")
        if value.strip()
    ]
    start = (
        datetime.strptime(end, "%Y-%m-%d") - timedelta(days=365 * years)
    ).strftime("%Y-%m-%d")

    print("=== XGBoost LambdaRank pair 构造实验 ===")
    print(
        f"股票={stock_count}, 窗口={start} ~ {end}, "
        f"pair_specs={pair_specs}, estimators={estimators}"
    )

    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()["code"].tolist()[:stock_count]
    manager.close()
    stocks_data = trainer.load_label_data(codes, start, end)
    dataset = trainer.prepare_dataset(
        stocks_data,
        train_start_date=start,
        train_end_date=end,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        n_jobs=15,
        target_features=None,
        use_factor_cache_only=True,
    )
    del stocks_data
    (
        X, y, returns, factor_names, dates, unbuyable, limit_groups,
        path_scores, is_st, w_sig,
    ) = dataset

    original_params = ModelConfig.XGBOOST_PARAMS.copy()
    summary = []
    try:
        for pair_spec in pair_specs:
            ModelConfig.XGBOOST_PARAMS.clear()
            ModelConfig.XGBOOST_PARAMS.update(original_params)
            ModelConfig.XGBOOST_PARAMS["n_estimators"] = estimators
            if pair_spec == "default":
                ModelConfig.XGBOOST_PARAMS.pop("lambdarank_pair_method", None)
                ModelConfig.XGBOOST_PARAMS.pop("lambdarank_num_pair_per_sample", None)
            else:
                method, pair_count = pair_spec.split(":", maxsplit=1)
                if method not in {"topk", "mean"}:
                    raise ValueError(f"不支持的 pair method: {method}")
                ModelConfig.XGBOOST_PARAMS["lambdarank_pair_method"] = method
                ModelConfig.XGBOOST_PARAMS["lambdarank_num_pair_per_sample"] = int(pair_count)

            print(f"\n{'#' * 72}\n# pair_spec={pair_spec}\n{'#' * 72}")
            started = time.time()
            results = trainer.train_models(
                X.copy(),
                y.copy(),
                returns,
                list(factor_names),
                dates,
                unbuyable_mask=unbuyable,
                limit_groups=limit_groups,
                path_scores=path_scores,
                is_st_arr=is_st,
                w_sig_arr=w_sig,
                model_types=["xgboost"],
                val_eval_sample_ratio=1.0,
            )
            result = results["xgboost"]
            metrics = result["val_metrics"]
            model = trainer.models["xgboost"]
            effective = _effective_pair_config(model)
            summary.append({
                "pair_spec": pair_spec,
                "effective_method": effective["lambdarank_pair_method"],
                "effective_pairs": effective["lambdarank_num_pair_per_sample"],
                "best_iteration": model._get_xgb_best_iteration(model.model),
                "rank_ic": metrics.get("rank_ic", float("nan")),
                "rank_ic_std": metrics.get("rank_ic_std", float("nan")),
                "rank_ic_ir": metrics.get("rank_ic_ir", float("nan")),
                "positive_ic_ratio": metrics.get("positive_ic_ratio", float("nan")),
                "top5_precision": metrics.get("top5_precision", float("nan")),
                "top5_return": metrics.get("top5_mean_return", float("nan")),
                "top5_excess": metrics.get("top5_excess_return", float("nan")),
                "elapsed": time.time() - started,
            })
    finally:
        ModelConfig.XGBOOST_PARAMS.clear()
        ModelConfig.XGBOOST_PARAMS.update(original_params)

    print(f"\n{'=' * 124}")
    print(
        f"{'pair_spec':>10} | {'effective':>12} | {'best':>5} | {'IC':>7} | "
        f"{'ICIR':>6} | {'IC+':>7} | {'Top5Hit':>8} | {'Top5Ret':>8} | "
        f"{'Top5Ex':>8} | {'秒':>7}"
    )
    print("-" * 124)
    for row in summary:
        effective = f"{row['effective_method']}/{row['effective_pairs']}"
        print(
            f"{row['pair_spec']:>10} | {effective:>12} | "
            f"{row['best_iteration']!s:>5} | {row['rank_ic']:>7.4f} | "
            f"{row['rank_ic_ir']:>6.3f} | {row['positive_ic_ratio']:>7.2%} | "
            f"{row['top5_precision']:>8.2%} | {row['top5_return']:>8.2%} | "
            f"{row['top5_excess']:>8.2%} | {row['elapsed']:>7.1f}"
        )
    print("=" * 124)


if __name__ == "__main__":
    main()
