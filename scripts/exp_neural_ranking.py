"""T036：架构升级 —— 神经排序 MLP（ListNet）替换 XGBoost LambdaRank（6(c)）。

假设（用户选定 6(c)）：T027-T035 证明在 XGBoost LambdaRank + 227 维特征下，头部可预测性
卡死——加特征（T033/T034）、调目标聚焦头部（T031/T032）、换预测周期（T035）全部无效或
反而伤头部。剩余未试的模型杠杆是**架构升级**：神经排序网络可学习 GBDT 无法组合出的
头部非线性交互（特征间的深层乘积/门控），可能突破 Rank IC↔Top-5 超额的权衡前沿。

设计（受控 A/B，唯一变量=模型族）：
- 相同数据 / 特征 / 7 日标签 / 折切分（70-80% 历史折选，80-100% 最终折确认）。
- 基线 = XGBoost LambdaRank（复用现有管线，取其选定特征集）。
- 候选 = PyTorch MLP 打分 + ListNet 列表损失（按日分组的 softmax CE，天然偏好头部）。
- 候选输入 = 基线选定特征（隔离"架构"为唯一变量；神经网不惧冗余但为公平用同特征集）。
- 评估：四指标（Rank IC/ICIR/正IC/Top-5超额）+ T027 头部诊断（预测 Top-K 真实收益分位）。
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
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from config.baostock_config import DATABASE_PATH
from config.factor_config import ModelConfig, TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.train_ml_model import MLModelTrainer, _fast_rankdata_1d
from scripts.diagnose_xgb_oof_head import _fold_dataset
from scripts.exp_head_features import _train_and_predict
from scripts.exp_horizon_7d_vs_15d import _daily_metrics, _head_percentile, _compare


class RankMLP(nn.Module):
    def __init__(self, dim, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _listnet_loss(pred, y, temp=1.0, y_scale=10.0):
    """按日分组的 ListNet 列表损失（平均跨日的 softmax 交叉熵）。

    y_scale 放大真实收益分布，使头部标的在目标 softmax 中更突出——更聚焦头部排序。
    """
    p = torch.softmax(pred / temp, dim=0)
    t = torch.softmax(y * y_scale / temp, dim=0)
    return -(t * torch.log(p + 1e-8)).sum()


def _train_mlp(X_train, y_train, dates_train, X_val, y_val, dates_val,
               epochs=40, lr=1e-3, batch_days=None, temp=1.0, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    scaler = StandardScaler().fit(X_train)
    Xtr = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xva = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)

    # 按日分组索引
    def group_idx(dates):
        d = pd.Series(dates)
        return [g.index.values for _, g in d.groupby(d, sort=False)]
    tr_groups = group_idx(dates_train)
    va_groups = group_idx(dates_val)

    model = RankMLP(X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    best_ic = -1.0
    best_state = None
    epochs_since_best = 0
    patience = 12
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(tr_groups))
        total = 0.0
        cnt = 0
        for gi in perm:
            idx = tr_groups[gi]
            if len(idx) < 5:
                continue
            xi = Xtr[idx]
            yi = ytr[idx]
            opt.zero_grad()
            loss = _listnet_loss(model(xi), yi, temp)
            loss.backward()
            opt.step()
            total += loss.item()
            cnt += 1
        # 验证集 Rank IC
        model.eval()
        with torch.no_grad():
            pred_va = model(Xva).numpy()
        m = _daily_metrics(pred_va, y_val, np.asarray(dates_val))
        sched.step(-m["rank_ic"])
        if m["rank_ic"] > best_ic:
            best_ic = m["rank_ic"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"    [MLP epoch {epoch}] ListNet steps={cnt} val_rank_ic={m['rank_ic']:.4f} "
                  f"val_top5_excess={m['top5_excess']:.4f}")
        if epochs_since_best >= patience:
            print(f"    [MLP] 早停 @ epoch {epoch}（验证 Rank IC 连续 {patience} 轮无改善）")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_va = model(Xva).numpy()
    return pred_va, best_ic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--end", required=True)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")
    start = (end_dt - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    cache_dir = os.path.join(output_dir, "neural_cache", end)
    os.makedirs(cache_dir, exist_ok=True)
    started = time.time()

    torch.set_num_threads(min(8, os.cpu_count() or 8))
    TrainingConfig.FUTURE_DAYS = 7  # 与 7 日生产基线对齐
    trainer = MLModelTrainer(db_path=DATABASE_PATH)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()["code"].tolist()[:args.stocks]
    manager.close()
    stocks_data = trainer.load_label_data(codes, start, end)
    dataset = trainer.prepare_dataset(
        stocks_data, train_start_date=start, train_end_date=end,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        n_jobs=args.workers, target_features=None, use_factor_cache_only=True,
        return_sample_metadata=False,
    )
    feature_names = list(dataset[3])
    print(f"  特征数: {len(feature_names)}，样本数: {len(dataset[0])}")

    original_save_dir = TrainingConfig.SAVE_DIR
    original_split = TrainingConfig.TRAIN_TEST_SPLIT
    original_params = ModelConfig.XGBOOST_PARAMS.copy()
    try:
        TrainingConfig.SAVE_DIR = cache_dir
        ModelConfig.XGBOOST_PARAMS["n_estimators"] = args.estimators

        # ── 历史折 70%-80% ────────────────────────────────────────────────
        print("\n=== 历史折 70%-80% ===")
        baseline = _train_and_predict(trainer, dataset, feature_names, 0.7, 0.8)
        base_sel = baseline["trained_feature_names"]
        feat_used = len(base_sel)
        b_row = _daily_metrics(baseline["predictions"], baseline["returns"], baseline["dates"])
        b_row["best_iteration"] = baseline["best_iteration"]
        b_row["head"] = _head_percentile(baseline["predictions"], baseline["returns"], baseline["dates"])
        print("[基线 XGBoost] 70%-80%:", b_row)

        # MLP 候选：相同选定特征
        fold, _, val_start, _, _, _ = _fold_dataset(dataset, 0.7, 0.8)
        X, y, returns, all_names, dates, *_ = fold
        idx = {n: i for i, n in enumerate(all_names)}
        sel_idx = [idx[n] for n in base_sel]
        Xs = X[:, sel_idx]
        X_train, X_val = Xs[:val_start], Xs[val_start:]
        y_train, y_val = y[:val_start], y[val_start:]
        d_train, d_val = dates[:val_start], dates[val_start:]
        print(f"  MLP 输入特征={len(base_sel)}，训练样本={len(X_train)}，验证样本={len(X_val)}")
        pred_va, best_ic = _train_mlp(X_train, y_train, d_train, X_val, y_val, d_val,
                                      epochs=args.epochs)
        m_row = _daily_metrics(pred_va, y_val, np.asarray(d_val))
        m_row["head"] = _head_percentile(pred_va, y_val, np.asarray(d_val))
        m_row["best_rank_ic"] = best_ic
        deltas, passes = _compare(m_row, b_row)
        m_row["deltas_vs_baseline"] = deltas
        m_row["passes_gate"] = passes
        m_row["head_delta_vs_baseline"] = {k: m_row["head"][k] - b_row["head"][k] for k in b_row["head"]}
        print(f"[候选 MLP] 70%-80%:", m_row)
        print(f"[历史折] 四指标 delta(MLP-XGB): {deltas}")
        print(f"[历史折] 头部分位 delta: {m_row['head_delta_vs_baseline']}")

        # ── 最终折 80%-100%（仅当历史折过门槛）─────────────────────────
        conf_base = conf_mlp = None
        if passes:
            print("\n=== 最终折 80%-100% ===")
            cb = _train_and_predict(trainer, dataset, feature_names, 0.8, 1.0)
            cbr = _daily_metrics(cb["predictions"], cb["returns"], cb["dates"])
            cbr["head"] = _head_percentile(cb["predictions"], cb["returns"], cb["dates"])
            conf_base = cbr
            fold2, _, val_start2, _, _, _ = _fold_dataset(dataset, 0.8, 1.0)
            X2, y2, _, an2, d2, *_ = fold2
            idx2 = {n: i for i, n in enumerate(an2)}
            sel2 = [idx2[n] for n in base_sel]
            X2s = X2[:, sel2]
            pred2, _ = _train_mlp(X2s[:val_start2], y2[:val_start2], d2[:val_start2],
                                  X2s[val_start2:], y2[val_start2:], d2[val_start2:],
                                  epochs=args.epochs)
            cmr = _daily_metrics(pred2, y2[val_start2:], np.asarray(d2[val_start2:]))
            cmr["head"] = _head_percentile(pred2, y2[val_start2:], np.asarray(d2[val_start2:]))
            d2c, p2c = _compare(cmr, cbr)
            cmr["deltas_vs_baseline"] = d2c
            cmr["passes_gate"] = p2c
            cmr["head_delta_vs_baseline"] = {k: cmr["head"][k] - cbr["head"][k] for k in cbr["head"]}
            conf_mlp = cmr
            print(f"[候选 MLP] 80%-100%:", cmr)
    finally:
        TrainingConfig.SAVE_DIR = original_save_dir
        TrainingConfig.TRAIN_TEST_SPLIT = original_split
        ModelConfig.XGBOOST_PARAMS.clear()
        ModelConfig.XGBOOST_PARAMS.update(original_params)

    def _strip(p):
        return {k: v for k, v in p.items() if k not in {"predictions", "returns", "dates"}} if p else None

    payload = {
        "metadata": {
            "stocks": args.stocks, "years": args.years, "start": start, "end": end,
            "features_total": len(feature_names), "features_used": feat_used,
            "selection_window": "70%-80%", "confirmation_window": "80%-100%",
            "horizon_days": 7, "elapsed_seconds": time.time() - started,
            "selected_mlp": bool(passes),
        },
        "baseline_metrics": _strip(b_row),
        "mlp_metrics": _strip(m_row),
        "confirmation_baseline": _strip(conf_base),
        "confirmation_mlp": _strip(conf_mlp),
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
