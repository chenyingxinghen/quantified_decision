"""T035：预测周期 A/B —— 7 日 vs 15 日（标签天花板诊断，对应原 6(a) 推荐方向）。

核心假设：T027-T034 证明在当前 XGBoost LambdaRank + 227 维特征下，无论加头部特征
（T033/T034）、调目标聚焦头部（T031/T032）、还是后处理混合（T027-T030），头部可预测性
都卡在"预测头部真实标签分位 ~0.559"（即只挑到上半区，挑不到真头部）。一种可能的根因是
**7 日前向收益在头部噪声过大、信号被反转/噪声淹没**，而非模型或特征本身的天花板。

本实验用受控 A/B 直接检验该假设：相同数据、相同特征集、相同 XGBoost 参数，只切预测周期
（TrainingConfig.SHORT_PREDICTION: True=7日 / False=15日），对比四指标与头部诊断。

- 四指标门槛（原版）：Rank IC / ICIR / 正 IC 日期 / Top-5 超额 不降级且至少一项改善。
- 头部诊断（T027 复算）：预测 Top-K 的真实收益在当日真实收益分布中的分位
  （7 日基线 ~0.559；若 15 日显著更高，证明头部瓶颈是 7 日标签噪声所致）。
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.baostock_config import DATABASE_PATH
from config.factor_config import ModelConfig, TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.train_ml_model import MLModelTrainer, _fast_rankdata_1d
from scripts.diagnose_xgb_oof_head import _fold_dataset


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


def _head_percentile(predictions, returns, dates, top_fracs=(0.01, 0.05, 0.20)):
    """预测 Top-K 的真实收益在当日真实收益分布中的平均百分位（T027 头部诊断）。

    top_fracs 默认含 0.01/0.05/0.20 —— 同时给出 Top-1/Top-5/Top-20 三档，
    用于对比"头部区间"在哪一档开始摆脱偶然性（用户反馈：Top-1 太极端，Top-20 更合适）。
    """
    out = {}
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    for tf in top_fracs:
        percs = []
        for start, count in zip(starts, counts):
            if count < 10:
                continue
            end = start + count
            pred = predictions[start:end]
            ret = returns[start:end]
            k = max(1, int(round(count * tf)))
            top = np.argpartition(pred, -k)[-k:]
            head_ret = ret[top].mean()
            percs.append(float((ret < head_ret).mean()))  # 当日分布中低于头部收益的占比
        out[f"head_pct_top{tf}"] = float(np.mean(percs)) if percs else float("nan")
    return out


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
        "trained_feature_names": list(model.feature_names),
        "best_iteration": model._get_xgb_best_iteration(model.model),
        "predictions": predictions,
        "returns": returns_val,
        "dates": dates_val,
    }


def _strip_arrays(payload):
    return {key: value for key, value in payload.items()
            if key not in {"predictions", "returns", "dates"}}


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
    cache_dir = os.path.join(output_dir, "horizon_cache", end)
    os.makedirs(cache_dir, exist_ok=True)
    started = time.time()

    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()["code"].tolist()[:args.stocks]
    manager.close()
    stocks_data = trainer.load_label_data(codes, start, end)

    # ── 准备两个周期的标签（因子缓存复用，仅标签/合并不同）──────────────
    original_save_dir = TrainingConfig.SAVE_DIR
    original_split = TrainingConfig.TRAIN_TEST_SPLIT
    original_future_days = TrainingConfig.FUTURE_DAYS
    original_params = ModelConfig.XGBOOST_PARAMS.copy()

    def prepare_horizon(future_days: int):
        TrainingConfig.FUTURE_DAYS = future_days
        print(f"\n=== 准备 {future_days}日 数据集 (FUTURE_DAYS={future_days}) ===")
        parts = trainer.prepare_dataset(
            stocks_data, train_start_date=start, train_end_date=end,
            include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
            n_jobs=args.workers, target_features=None, use_factor_cache_only=True,
            return_sample_metadata=False,
        )
        return parts

    ds7 = prepare_horizon(7)
    ds15 = prepare_horizon(15)
    feature_names = list(ds7[3])  # 两周期特征名一致
    print(f"  特征数: {len(feature_names)}，7日样本: {len(ds7[0])}，15日样本: {len(ds15[0])}")

    try:
        TrainingConfig.SAVE_DIR = cache_dir
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators

        def run_horizon(ds, future_days, fold_lo, fold_hi):
            TrainingConfig.FUTURE_DAYS = future_days
            res = _train_and_predict(trainer, ds, feature_names, fold_lo, fold_hi)
            m = _daily_metrics(res["predictions"], res["returns"], res["dates"])
            m["head"] = _head_percentile(res["predictions"], res["returns"], res["dates"])
            m["best_iteration"] = res["best_iteration"]
            m["future_days"] = future_days
            return res, m

        # ── 历史折 70%-80%（选择窗口）────────────────────────────────────
        r7, m7 = run_horizon(ds7, 7, 0.7, 0.8)
        r15, m15 = run_horizon(ds15, 15, 0.7, 0.8)
        print("\n[历史折 70%-80%] 7日:", m7)
        print("[历史折 70%-80%] 15日:", m15)

        deltas, passes = _compare(m15, m7)
        m15["deltas_vs_7d"] = deltas
        m15["passes_gate"] = passes
        head_delta = {k: m15["head"][k] - m7["head"][k] for k in m7["head"]}
        m15["head_delta_vs_7d"] = head_delta
        print(f"[历史折] 四指标 delta(15-7): {deltas}")
        print(f"[历史折] 头部分位 delta(15-7): {head_delta}")

        # ── 最终折 80%-100%（确认窗口，仅当历史折通过门槛）─────────────
        conf7 = conf15 = None
        if passes:
            _, cm7 = run_horizon(ds7, 7, 0.8, 1.0)
            _, cm15 = run_horizon(ds15, 15, 0.8, 1.0)
            d_c, p_c = _compare(cm15, cm7)
            cm15["deltas_vs_7d"] = d_c
            cm15["passes_gate"] = p_c
            cm15["head_delta_vs_7d"] = {k: cm15["head"][k] - cm7["head"][k] for k in cm7["head"]}
            conf7, conf15 = cm7, cm15
            print(f"\n[最终折 80%-100%] 7日:", cm7)
            print(f"[最终折 80%-100%] 15日:", cm15)
    finally:
        TrainingConfig.SAVE_DIR = original_save_dir
        TrainingConfig.TRAIN_TEST_SPLIT = original_split
        TrainingConfig.FUTURE_DAYS = original_future_days
        ModelConfig.XGBOOST_PARAMS.clear()
        ModelConfig.XGBOOST_PARAMS.update(original_params)

    payload = {
        "metadata": {
            "stocks": args.stocks,
            "years": args.years,
            "start": start,
            "end": end,
            "features": len(feature_names),
            "selection_window": "70%-80%",
            "confirmation_window": "80%-100%",
            "embargo_follows_future_days": True,
            "elapsed_seconds": time.time() - started,
            "selected_15d": bool(passes),
        },
        "metrics_7d": _strip_arrays(m7),
        "metrics_15d": _strip_arrays(m15),
        "confirmation_7d": _strip_arrays(conf7) if conf7 else None,
        "confirmation_15d": _strip_arrays(conf15) if conf15 else None,
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
