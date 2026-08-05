"""
控制变量实验：比较训练样本时间衰减对验证集排序质量的影响。

同一份数据、同一时间切分、同一模型参数，仅改变交易日 query 的时间权重。

用法:
    EXP_STOCKS=800 EXP_YEARS=8 EXP_HALF_LIVES=off,2,4,8 \
        python -u scripts/exp_recency_weight.py
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
    stock_count = int(os.environ.get("EXP_STOCKS", "800"))
    years = int(os.environ.get("EXP_YEARS", "8"))
    half_lives = os.environ.get("EXP_HALF_LIVES", "off,2,4,8").split(",")
    estimators = int(os.environ.get("EXP_ESTIMATORS", "500"))

    end = (
        datetime.now() - timedelta(days=365 * TrainingConfig.YEARS_FOR_BACKTEST)
    ).strftime("%Y-%m-%d")
    start = (
        datetime.now()
        - timedelta(days=365 * (TrainingConfig.YEARS_FOR_BACKTEST + years))
    ).strftime("%Y-%m-%d")

    print("=== 时间衰减训练对照实验 ===")
    print(
        f"股票={stock_count}, 窗口={start} ~ {end}, "
        f"half_lives={half_lives}, estimators={estimators}"
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

    original_estimators = ModelConfig.XGBOOST_PARAMS["n_estimators"]
    original_enabled = TrainingConfig.USE_RECENCY_WEIGHT
    original_half_life = TrainingConfig.RECENCY_HALF_LIFE_YEARS
    ModelConfig.XGBOOST_PARAMS["n_estimators"] = estimators
    summary = []

    try:
        for value in half_lives:
            enabled = value.strip().lower() not in {"off", "none", "0", "false"}
            TrainingConfig.USE_RECENCY_WEIGHT = enabled
            if enabled:
                TrainingConfig.RECENCY_HALF_LIFE_YEARS = float(value)

            label = f"{float(value):g}y" if enabled else "off"
            print(f"\n{'#' * 72}\n# recency={label}\n{'#' * 72}")
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
            )
            metrics = results["xgboost"]["val_metrics"]
            summary.append(
                (
                    label,
                    metrics.get("rank_ic", float("nan")),
                    metrics.get("rank_ic_std", metrics.get("ic_std", float("nan"))),
                    metrics.get("top1_precision", float("nan")),
                    metrics.get("top5_precision", float("nan")),
                    time.time() - started,
                )
            )
    finally:
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = original_estimators
        TrainingConfig.USE_RECENCY_WEIGHT = original_enabled
        TrainingConfig.RECENCY_HALF_LIFE_YEARS = original_half_life

    print(f"\n{'=' * 78}")
    print(f"{'衰减':>8} | {'val_ic':>8} | {'ic_std':>8} | {'top1':>8} | {'top5':>8} | {'秒':>7}")
    print("-" * 78)
    for label, rank_ic, ic_std, top1, top5, elapsed in summary:
        print(
            f"{label:>8} | {rank_ic:>8.4f} | {ic_std:>8.4f} | "
            f"{top1:>8.2%} | {top5:>8.2%} | {elapsed:>7.1f}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
