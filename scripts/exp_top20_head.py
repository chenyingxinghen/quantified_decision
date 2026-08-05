"""T037：Top-20 头部诊断（用户反馈：Top-1 太极端，Top-20 更合适）。

受控 OOF 验证：相同 7 日标签 / 特征 / 折切分，仅用生产 XGBoost 模型，
报告预测 Top-1 / Top-5 / Top-20 三档在当日真实收益分布中的平均百分位，
直接检验"头部区间在 Top-20 是否摆脱偶然性、变得可预测"。

四指标（Rank IC/ICIR/正IC/Top-5超额）+ 三档头部分位。
历史折 70%-80% 选择，最终折 80%-100% 确认（样本外）。
"""

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
from scripts.diagnose_xgb_oof_head import _fold_dataset
from scripts.exp_head_features import _train_and_predict
from scripts.exp_horizon_7d_vs_15d import _daily_metrics, _head_percentile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", required=True)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    start = (datetime.strptime(args.end, "%Y-%m-%d")
             - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")

    torch_ok = True
    try:
        import torch  # noqa
    except Exception:
        torch_ok = False

    TrainingConfig.FUTURE_DAYS = 7
    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()["code"].tolist()[:args.stocks]
    manager.close()
    stocks_data = trainer.load_label_data(codes, start, args.end)
    dataset = trainer.prepare_dataset(
        stocks_data, train_start_date=start, train_end_date=args.end,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        n_jobs=args.workers, target_features=None, use_factor_cache_only=True,
        return_sample_metadata=False,
    )
    feature_names = list(dataset[3])
    print(f"  特征数: {len(feature_names)}，样本数: {len(dataset[0])}")

    original_save_dir = TrainingConfig.SAVE_DIR
    original_split = TrainingConfig.TRAIN_TEST_SPLIT
    original_params = ModelConfig.XGBOOST_PARAMS.copy()
    cache_dir = os.path.join("models", "diagnostics", "top5_failure_20240804", "top20_cache")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        TrainingConfig.SAVE_DIR = cache_dir
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators

        # ── 历史折 70%-80% ────────────────────────────────────────────────
        print("\n=== 历史折 70%-80% ===")
        res = _train_and_predict(trainer, dataset, feature_names, 0.7, 0.8)
        b_row = _daily_metrics(res["predictions"], res["returns"], res["dates"])
        b_row["best_iteration"] = res["best_iteration"]
        b_row["head"] = _head_percentile(res["predictions"], res["returns"], res["dates"])
        b_row["features_used"] = len(res["trained_feature_names"])
        print("[XGBoost 7d] 70%-80%:", b_row)

        # ── 最终折 80%-100%（样本外确认）─────────────────────────────────
        print("\n=== 最终折 80%-100% (样本外) ===")
        res_c = _train_and_predict(trainer, dataset, feature_names, 0.8, 1.0)
        c_row = _daily_metrics(res_c["predictions"], res_c["returns"], res_c["dates"])
        c_row["best_iteration"] = res_c["best_iteration"]
        c_row["head"] = _head_percentile(res_c["predictions"], res_c["returns"], res_c["dates"])
        print("[XGBoost 7d] 80%-100%:", c_row)

        payload = {
            "metadata": {
                "experiment": "T037_top20_head",
                "stocks": args.stocks, "years": args.years, "end": args.end,
                "future_days": 7, "folds": {"historical": "70-80%", "confirmation": "80-100%"},
                "torch_available": torch_ok,
                "note": "Top-1 太极端；对比预测 Top-1/Top-5/Top-20 在当日真实收益分布中的平均百分位",
            },
            "baseline_metrics": b_row,        # 历史折
            "confirmation_metrics": c_row,    # 最终折（样本外）
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n已保存: {args.output}")
    finally:
        TrainingConfig.SAVE_DIR = original_save_dir
        TrainingConfig.TRAIN_TEST_SPLIT = original_split
        ModelConfig.XGBOOST_PARAMS.clear()
        ModelConfig.XGBOOST_PARAMS.update(original_params)


if __name__ == "__main__":
    main()
