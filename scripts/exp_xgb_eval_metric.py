"""
控制变量实验：比较 XGBoost ranking 的早停 NDCG 截断位。

同一份数据、同一时间切分、同一模型参数，仅改变 eval_metric。

用法:
    EXP_STOCKS=500 EXP_YEARS=8 EXP_METRICS=ndcg,ndcg@20,ndcg@50 \
        python -u scripts/exp_xgb_eval_metric.py
"""
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.baostock_config import DATABASE_PATH
from config.factor_config import ModelConfig, TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.train_ml_model import MLModelTrainer


def main():
    stock_count = int(os.environ.get("EXP_STOCKS", "500"))
    years = int(os.environ.get("EXP_YEARS", "8"))
    metrics = os.environ.get("EXP_METRICS", "ndcg,ndcg@20,ndcg@50").split(",")
    estimators = int(os.environ.get("EXP_ESTIMATORS", "500"))

    end = os.environ.get("EXP_END") or (
        datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)
    ).strftime("%Y-%m-%d")
    start = (
        datetime.strptime(end, "%Y-%m-%d") - timedelta(days=365 * years)
    ).strftime("%Y-%m-%d")

    print("=== XGBoost 早停指标对照实验 ===")
    print(
        f"股票={stock_count}, 窗口={start} ~ {end}, "
        f"metrics={metrics}, estimators={estimators}"
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
        X,
        y,
        returns,
        factor_names,
        dates,
        unbuyable,
        limit_groups,
        path_scores,
        is_st,
        w_sig,
    ) = dataset

    original_metric = ModelConfig.XGBOOST_PARAMS["eval_metric"]
    original_estimators = ModelConfig.XGBOOST_PARAMS["n_estimators"]
    ModelConfig.XGBOOST_PARAMS["n_estimators"] = estimators
    summary = []

    try:
        for metric in metrics:
            metric = metric.strip()
            ModelConfig.XGBOOST_PARAMS["eval_metric"] = metric
            print(f"\n{'#' * 72}\n# eval_metric={metric}\n{'#' * 72}")
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
            val_metrics = result["val_metrics"]
            model = trainer.models["xgboost"]
            booster = model.model
            best_iteration = model._get_xgb_best_iteration(booster)
            summary.append(
                (
                    metric,
                    best_iteration,
                    val_metrics.get("rank_ic", float("nan")),
                    val_metrics.get(
                        "rank_ic_std", val_metrics.get("ic_std", float("nan"))
                    ),
                    val_metrics.get("top1_precision", float("nan")),
                    val_metrics.get("top5_precision", float("nan")),
                    val_metrics.get("rank_ic_ir", float("nan")),
                    val_metrics.get("positive_ic_ratio", float("nan")),
                    val_metrics.get("top5_mean_return", float("nan")),
                    val_metrics.get("top5_excess_return", float("nan")),
                    time.time() - started,
                )
            )
    finally:
        ModelConfig.XGBOOST_PARAMS["eval_metric"] = original_metric
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = original_estimators

    print(f"\n{'=' * 92}")
    print(
        f"{'metric':>12} | {'best_iter':>9} | {'val_ic':>8} | "
        f"{'ic_std':>8} | {'ICIR':>7} | {'IC+':>7} | {'top5':>8} | "
        f"{'top5收益':>9} | {'top5超额':>9} | {'秒':>7}"
    )
    print("-" * 92)
    for metric, best_iter, rank_ic, ic_std, top1, top5, icir, ic_pos, top5_ret, top5_excess, elapsed in summary:
        print(
            f"{metric:>12} | {best_iter!s:>9} | {rank_ic:>8.4f} | "
            f"{ic_std:>8.4f} | {icir:>7.3f} | {ic_pos:>7.2%} | {top5:>8.2%} | "
            f"{top5_ret:>9.2%} | {top5_excess:>9.2%} | {elapsed:>7.1f}"
        )
    print("=" * 92)


if __name__ == "__main__":
    main()
