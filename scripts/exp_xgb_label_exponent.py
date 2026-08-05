"""
控制变量实验：比较 XGBoost ranking 连续标签的头部增益指数。

标签由每日截面 rank 变换为 rank ** exponent。指数越大，LambdaRank 越关注头部。

用法:
    EXP_STOCKS=500 EXP_YEARS=8 EXP_EXPONENTS=1.0,1.1,1.2,1.5 \
        python -u scripts/exp_xgb_label_exponent.py
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
    exponents = [
        float(value)
        for value in os.environ.get("EXP_EXPONENTS", "1.0,1.1,1.2,1.5").split(",")
    ]
    estimators = int(os.environ.get("EXP_ESTIMATORS", "500"))
    val_sample_ratio = float(os.environ.get("EXP_VAL_SAMPLE_RATIO", "1.0"))

    configured_end = os.environ.get("EXP_END")
    end_dt = (
        datetime.strptime(configured_end, "%Y-%m-%d")
        if configured_end
        else datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)
    )
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    print("=== XGBoost 标签指数对照实验 ===")
    print(
        f"股票={stock_count}, 窗口={start} ~ {end}, "
        f"exponents={exponents}, estimators={estimators}, "
        f"val_sample_ratio={val_sample_ratio:g}"
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

    original_enabled = TrainingConfig.LABEL_WEIGHTED_FOR_XGB
    original_exponent = TrainingConfig.LABEL_WEIGHT_EXPONENT
    original_estimators = ModelConfig.XGBOOST_PARAMS["n_estimators"]
    TrainingConfig.LABEL_WEIGHTED_FOR_XGB = True
    ModelConfig.XGBOOST_PARAMS["n_estimators"] = estimators
    summary = []

    try:
        for exponent in exponents:
            TrainingConfig.LABEL_WEIGHT_EXPONENT = exponent
            print(f"\n{'#' * 72}\n# label_exponent={exponent:g}\n{'#' * 72}")
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
                val_eval_sample_ratio=val_sample_ratio,
            )
            result = results["xgboost"]
            val_metrics = result["val_metrics"]
            model = trainer.models["xgboost"]
            best_iteration = model._get_xgb_best_iteration(model.model)
            summary.append(
                (
                    exponent,
                    best_iteration,
                    val_metrics.get("rank_ic", float("nan")),
                    val_metrics.get(
                        "rank_ic_std", val_metrics.get("ic_std", float("nan"))
                    ),
                    val_metrics.get("top1_precision", float("nan")),
                    val_metrics.get("top5_precision", float("nan")),
                    val_metrics.get("win_rate", float("nan")),
                    val_metrics.get("rank_ic_ir", float("nan")),
                    val_metrics.get("positive_ic_ratio", float("nan")),
                    val_metrics.get("top5_mean_return", float("nan")),
                    val_metrics.get("top5_excess_return", float("nan")),
                    time.time() - started,
                )
            )
    finally:
        TrainingConfig.LABEL_WEIGHTED_FOR_XGB = original_enabled
        TrainingConfig.LABEL_WEIGHT_EXPONENT = original_exponent
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = original_estimators

    print(f"\n{'=' * 104}")
    print(
        f"{'exponent':>9} | {'best_iter':>9} | {'val_ic':>8} | {'ic_std':>8} | "
        f"{'top1':>8} | {'top5':>8} | {'win':>8} | {'ICIR':>7} | "
        f"{'IC+':>7} | {'top5ret':>8} | {'top5ex':>8} | {'秒':>7}"
    )
    print("-" * 104)
    for (exponent, best_iter, rank_ic, ic_std, top1, top5, win, icir,
         positive_ic, top5_return, top5_excess, elapsed) in summary:
        print(
            f"{exponent:>9.2f} | {best_iter!s:>9} | {rank_ic:>8.4f} | "
            f"{ic_std:>8.4f} | {top1:>8.2%} | {top5:>8.2%} | "
            f"{win:>8.2%} | {icir:>7.3f} | {positive_ic:>7.2%} | "
            f"{top5_return:>8.2%} | {top5_excess:>8.2%} | {elapsed:>7.1f}"
        )
    print("=" * 104)


if __name__ == "__main__":
    main()
