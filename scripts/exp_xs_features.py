"""T034：截面相对特征（6(a) 真正信息增量形式）。

T033 证明对逐只股票的动量/ATR/偏度做比率/偏度重组被相关性选择剪枝或被模型忽略——
瓶颈不是缺逐只构件，而是模型输入全为"逐只绝对值"，LambdaRank 虽做截面排序但
特征层不含任何"截面相对量"。本实验注入 4 个**截面相对**特征，提供 XGBoost 无法从
逐只特征重构的信息类：

  - hf_xs_mom_pct       ：个股 vol-adjusted 动量在当日全市场截面中的分位（高=今日市场最强动量/波动比）
  - hf_xs_updn_pct      ：个股 上行/下行波动率比 在当日全市场截面中的分位
  - hf_rel_strength_vs_mkt：个股 vol-adjusted 动量 减 当日全市场中位数（市场相对强度，带符号）
  - hf_self_mom_pct_250 ：个股自身过去 250 日 vol-adjusted 动量的滚动分位（高=处于自身历史头部极端区）

对照与隔离同 T033：基线=原始特征，候选=基线选定特征 + 4 截面相对特征；用
return_sample_metadata 逐行 (code,date) 对齐注入；四指标门槛同前。
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
from scripts.diagnose_xgb_oof_head import _daily_percentiles, _fold_dataset


NEW_FEATURE_NAMES = [
    "hf_xs_mom_pct",
    "hf_xs_updn_pct",
    "hf_rel_strength_vs_mkt",
    "hf_self_mom_pct_250",
]


def compute_base_signals(stock_df: pd.DataFrame) -> pd.DataFrame:
    """计算截面相对特征所需的逐只基础信号，索引为字符串日期，无前视。"""
    df = stock_df.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    ret = close.pct_change()
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(14).mean()
    natr = atr / close

    ret20 = close.pct_change(20)
    vol_adj_mom = ret20 / (natr + 1e-6)

    up = ret.clip(lower=0.0)
    dn = (-ret).clip(lower=0.0)
    up_var = up.rolling(20).var()
    dn_var = dn.rolling(20).var()
    updn_ratio = (up_var + 1e-9) / (dn_var + 1e-9)
    updn_ratio = np.log1p(updn_ratio.clip(upper=100.0))

    # 自身历史滚动分位（窗口 250 交易日），rank(pct=True) 给出末位值在区间内分位
    self_mom_pct = vol_adj_mom.rolling(250).rank(pct=True)

    out = pd.DataFrame(
        {
            "vol_adj_mom": vol_adj_mom,
            "updn_ratio": updn_ratio,
            "self_mom_pct": self_mom_pct,
        },
        index=df.index,
    )
    out.index = out.index.strftime("%Y-%m-%d")
    out = out.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return out


def build_xs_features(stocks_data: dict, codes, meta: pd.DataFrame) -> np.ndarray:
    """构建截面相对特征矩阵，按 meta 逐行对齐。

    截面分位在"当日全市场"上计算（所有信号均 ≤ 当日，无未来泄漏）。
    """
    # 1. 逐只基础信号
    base = {}
    for code in codes:
        if code not in stocks_data:
            continue
        try:
            base[code] = compute_base_signals(stocks_data[code])
        except Exception as exc:
            print(f"  [warn] 基础信号失败 {code}: {exc}")
            continue

    # 2. 截面面板（date × code）
    mom_panel = pd.DataFrame({c: b["vol_adj_mom"] for c, b in base.items()})
    updn_panel = pd.DataFrame({c: b["updn_ratio"] for c, b in base.items()})

    # 3. 截面分位（每行=一日，axis=1 跨股票）
    xs_mom_pct = mom_panel.rank(pct=True, axis=1)
    xs_updn_pct = updn_panel.rank(pct=True, axis=1)
    # 4. 市场相对强度（减当日中位数）
    mkt_median = mom_panel.median(axis=1, skipna=True)
    rel_strength = mom_panel.subtract(mkt_median, axis=0)

    # 5. 自身滚动分位（已在基础信号中）
    self_pct = pd.DataFrame({c: b["self_mom_pct"] for c, b in base.items()})

    # 6. 逐行对齐
    n = len(meta)
    arr = np.zeros((n, len(NEW_FEATURE_NAMES)), dtype=np.float32)
    codes_arr = meta["code"].values
    dates_arr = pd.to_datetime(meta["date"].values).strftime("%Y-%m-%d")
    for i in range(n):
        c = codes_arr[i]
        d = dates_arr[i]
        if c not in base or d not in xs_mom_pct.index:
            continue
        try:
            arr[i, 0] = xs_mom_pct.loc[d, c]
            arr[i, 1] = xs_updn_pct.loc[d, c]
            arr[i, 2] = rel_strength.loc[d, c]
            arr[i, 3] = self_pct.loc[d, c]
        except Exception:
            continue
    return arr


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
    cache_dir = os.path.join(output_dir, "xs_features_cache", end)
    os.makedirs(cache_dir, exist_ok=True)
    started = time.time()

    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()["code"].tolist()[:args.stocks]
    manager.close()
    stocks_data = trainer.load_label_data(codes, start, end)
    dataset_parts = trainer.prepare_dataset(
        stocks_data, train_start_date=start, train_end_date=end,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        n_jobs=args.workers, target_features=None, use_factor_cache_only=True,
        return_sample_metadata=True,
    )
    meta = dataset_parts[-1]
    dataset = list(dataset_parts[:-1])
    original_feature_names = list(dataset[3])
    print(f"  原始特征数: {len(original_feature_names)}，样本数: {len(dataset[0])}")

    new_X = build_xs_features(stocks_data, codes, meta)
    nz = float(np.any(new_X != 0, axis=1).mean())
    print(f"  截面特征矩阵: {new_X.shape}，非全零行占比: {nz:.2%}，"
          f"各列均值={dict(zip(NEW_FEATURE_NAMES, new_X.mean(axis=0).round(4).tolist()))}")

    dataset[0] = np.hstack([dataset[0], new_X]).astype(np.float32)
    dataset[3] = original_feature_names + NEW_FEATURE_NAMES
    dataset = tuple(dataset)

    original_save_dir = TrainingConfig.SAVE_DIR
    original_split = TrainingConfig.TRAIN_TEST_SPLIT
    original_params = ModelConfig.XGBOOST_PARAMS.copy()
    try:
        TrainingConfig.SAVE_DIR = cache_dir
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators

        baseline = _train_and_predict(trainer, dataset, original_feature_names, 0.7, 0.8)
        baseline_selected = baseline["trained_feature_names"]
        baseline_row = _daily_metrics(baseline["predictions"], baseline["returns"], baseline["dates"])
        baseline_row["best_iteration"] = baseline["best_iteration"]
        print("\n[基线/原始特征] 70%-80% 历史折:", baseline_row)

        candidate_feature_names = baseline_selected + NEW_FEATURE_NAMES
        cand = _train_and_predict(trainer, dataset, candidate_feature_names, 0.7, 0.8)
        row = _daily_metrics(cand["predictions"], cand["returns"], cand["dates"])
        row["best_iteration"] = cand["best_iteration"]
        row["trained_feature_names"] = cand["trained_feature_names"]
        row["hf_selected"] = [f for f in cand["trained_feature_names"] if f.startswith("hf_")]
        deltas, passes = _compare(row, baseline_row)
        row["deltas_vs_baseline"] = deltas
        row["passes_gate"] = passes
        print(f"\n[候选/+截面相对特征] 70%-80% 历史折:", row)
        selected = row if passes else None

        confirmation_row = None
        confirmation_baseline = None
        if selected is not None:
            conf_base = _train_and_predict(trainer, dataset, original_feature_names, 0.8, 1.0)
            confirmation_baseline = _daily_metrics(conf_base["predictions"], conf_base["returns"], conf_base["dates"])
            confirmation_baseline["best_iteration"] = conf_base["best_iteration"]
            conf_cand = _train_and_predict(trainer, dataset, candidate_feature_names, 0.8, 1.0)
            confirmation_row = _daily_metrics(conf_cand["predictions"], conf_cand["returns"], conf_cand["dates"])
            confirmation_row["best_iteration"] = conf_cand["best_iteration"]
            d_c, p_c = _compare(confirmation_row, confirmation_baseline)
            confirmation_row["deltas_vs_baseline"] = d_c
            confirmation_row["passes_gate"] = p_c
            print(f"\n[最终折确认/+截面相对特征] 80%-100%:", confirmation_row)
    finally:
        TrainingConfig.SAVE_DIR = original_save_dir
        TrainingConfig.TRAIN_TEST_SPLIT = original_split
        ModelConfig.XGBOOST_PARAMS.clear()
        ModelConfig.XGBOOST_PARAMS.update(original_params)

    payload = {
        "metadata": {
            "stocks_requested": args.stocks,
            "stocks_loaded": len(stocks_data),
            "start": start,
            "end": end,
            "samples": len(dataset[4]),
            "original_features": len(original_feature_names),
            "new_features": NEW_FEATURE_NAMES,
            "selection_window": "70%-80% closed fold",
            "confirmation_window": "80%-100% final fold",
            "embargo_trading_days": TrainingConfig.FUTURE_DAYS,
            "elapsed_seconds": time.time() - started,
            "selected": selected is not None,
        },
        "baseline_selection_fold": _strip_arrays(baseline),
        "baseline_metrics": baseline_row,
        "candidate_metrics": row,
        "confirmation_baseline": confirmation_baseline,
        "confirmation_candidate": confirmation_row,
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
