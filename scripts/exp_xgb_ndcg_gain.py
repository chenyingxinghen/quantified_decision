"""T032: XGBoost NDCG 增益形状（指数增益 vs 线性增益）。

假设：生产配置当前为 `ndcg_exp_gain=False`（线性增益，gain=rel）。
T027-T031 证明瓶颈在头部可预测性，且 T031 用自定义头部加权目标（更聚焦头部）
提升了平均 Rank IC 却牺牲了逐日稳定性与 Top-5 超额。本实验走内置路径测试
"增益形状"这一干净、无自定义目标的头部聚焦杠杆：把 `ndcg_exp_gain` 从 False
切到 True（指数增益，gain=2^rel-1），直接对照 T031——同等"更聚焦头部"方向，
但由 XGBoost 内置 LambdaRank 平滑施加，而非硬 pair 加权。

协议（同 T031）：800 股、8 年窗口、2022-08-04 截止；
先在 70%-80% 历史折上比较 linear（baseline）vs exp（候选），要求 Rank IC /
ICIR / 正 IC 日期 / Top-5 超额四项不降级且至少一项改善；通过的候选再在
80%-100% 最终折确认。
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
from scripts.diagnose_xgb_oof_head import _daily_percentiles, _fold_dataset


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


def _compare(row, baseline):
    keys = ("rank_ic", "rank_ic_ir", "positive_ic_ratio", "top5_excess")
    deltas = {key: row[key] - baseline[key] for key in keys}
    passes = all(value >= 0 for value in deltas.values()) and any(
        value > 0 for value in deltas.values()
    )
    return deltas, passes


def _train_and_predict(trainer, dataset, feature_names, train_fraction, validation_end):
    fold, split, val_start, _, split_date, end_date = _fold_dataset(
        dataset, train_fraction, validation_end
    )
    X, y, returns, all_names, dates, unbuyable, limits, scores, is_st, w_sig = fold
    index = {name: idx for idx, name in enumerate(all_names)}
    selected_idx = [index[name] for name in feature_names]
    X = X[:, selected_idx].copy()
    TrainingConfig.TRAIN_TEST_SPLIT = split
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
    model = trainer.models["xgboost"]
    trained_idx = [feature_names.index(name) for name in model.feature_names]
    X_val = X[val_start:, :][:, trained_idx]
    dates_val = dates[val_start:]
    returns_val = returns[val_start:]
    dval = xgb.DMatrix(X_val, feature_names=model.feature_names)
    predictions = model._predict_xgb_booster(model.model, dval)
    return {
        "split_date": split_date,
        "validation_end_date_exclusive": end_date,
        "validation_start_date": str(dates_val[0]),
        "validation_end_date": str(dates_val[-1]),
        "validation_samples": int(len(dates_val)),
        "trained_feature_names": list(model.feature_names),
        "best_iteration": model._get_xgb_best_iteration(model.model),
        "trainer_metrics": results["xgboost"]["val_metrics"],
        "predictions": predictions,
        "returns": returns_val,
        "dates": dates_val,
    }


def _strip_arrays(payload):
    return {
        key: value for key, value in payload.items()
        if key not in {"predictions", "returns", "dates"}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", required=True)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    cache_dir = os.path.join(output_dir, "ndcg_gain_cache", end)
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
    original_force_discrete = getattr(TrainingConfig, 'XGBOOST_FORCE_DISCRETE_LABEL', False)
    try:
        TrainingConfig.SAVE_DIR = cache_dir
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators
        # ndcg_exp_gain=True 要求整数标签，故两种增益形状都喂离散档位标签，
        # 从而把"增益形状"隔离为唯一变量（XGBoost 连续标签不支持 exp 增益）。
        TrainingConfig.XGBOOST_FORCE_DISCRETE_LABEL = True

        # ── 基线：线性增益（ndcg_exp_gain=False，gain=rel）────────────────
        ModelConfig.XGBOOST_PARAMS["ndcg_exp_gain"] = False
        baseline = _train_and_predict(trainer, dataset, list(dataset[3]), 0.7, 0.8)
        feature_names = baseline["trained_feature_names"]
        baseline_row = _daily_metrics(baseline["predictions"], baseline["returns"], baseline["dates"])
        baseline_row["gain_mode"] = "linear"
        baseline_row["best_iteration"] = baseline["best_iteration"]
        print("\n[基线/linear] 70%-80% 历史折:", baseline_row)

        # ── 候选：指数增益（ndcg_exp_gain=True，gain=2^rel-1，头部聚焦内置路径）─
        ModelConfig.XGBOOST_PARAMS["ndcg_exp_gain"] = True
        cand = _train_and_predict(trainer, dataset, feature_names, 0.7, 0.8)
        row = _daily_metrics(cand["predictions"], cand["returns"], cand["dates"])
        row["gain_mode"] = "exp"
        row["best_iteration"] = cand["best_iteration"]
        deltas, passes = _compare(row, baseline_row)
        row["deltas_vs_baseline"] = deltas
        row["passes_gate"] = passes
        print(f"\n[候选/exp] 70%-80% 历史折:", row)
        candidate_rows = [row]
        selected = row if passes else None

        # ── 最终折确认（仅对通过历史折门槛的候选）────────────────────────
        confirmation_row = None
        confirmation_baseline = None
        if selected is not None:
            ModelConfig.XGBOOST_PARAMS["ndcg_exp_gain"] = False
            conf_base = _train_and_predict(trainer, dataset, feature_names, 0.8, 1.0)
            confirmation_baseline = _daily_metrics(conf_base["predictions"], conf_base["returns"], conf_base["dates"])
            confirmation_baseline["gain_mode"] = "linear"
            confirmation_baseline["best_iteration"] = conf_base["best_iteration"]
            ModelConfig.XGBOOST_PARAMS["ndcg_exp_gain"] = True
            conf_cand = _train_and_predict(trainer, dataset, feature_names, 0.8, 1.0)
            confirmation_row = _daily_metrics(conf_cand["predictions"], conf_cand["returns"], conf_cand["dates"])
            confirmation_row["gain_mode"] = "exp"
            confirmation_row["best_iteration"] = conf_cand["best_iteration"]
            d_c, p_c = _compare(confirmation_row, confirmation_baseline)
            confirmation_row["deltas_vs_baseline"] = d_c
            confirmation_row["passes_gate"] = p_c
            print(f"\n[最终折确认/exp] 80%-100%:", confirmation_row)
    finally:
        TrainingConfig.SAVE_DIR = original_save_dir
        TrainingConfig.TRAIN_TEST_SPLIT = original_split
        TrainingConfig.XGBOOST_FORCE_DISCRETE_LABEL = original_force_discrete
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
            "selection_window": "70%-80% closed fold",
            "confirmation_window": "80%-100% final fold",
            "embargo_trading_days": TrainingConfig.FUTURE_DAYS,
            "elapsed_seconds": time.time() - started,
            "selected_gain_mode": selected["gain_mode"] if selected else None,
        },
        "baseline_selection_fold": _strip_arrays(baseline),
        "baseline_metrics": baseline_row,
        "candidate_rows": candidate_rows,
        "selected_candidate": selected,
        "confirmation_baseline": confirmation_baseline,
        "confirmation_candidate": confirmation_row,
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
