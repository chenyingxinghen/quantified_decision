"""
NeuralTrainer — 神经网络训练器，复用既有数据基础设施。

本模块不重新实现数据管线，而是复用：
- core.factors.train_ml_model.MLModelTrainer
    .load_label_data        读取标签行情
    .prepare_dataset        产出 (X, y, returns, factor_names, dates, ...)
    .build_multiobjective_labels  多目标标签
    ._apply_cross_sectional_normalization_inplace  横截面归一化（与 GBM 一致）
- core.factors.feature_selector.CrossSectionalFeatureSelector  特征选择
- config.factor_config.TrainingConfig / OptimizationConfig      数据/优化参数

只把"训练器"从 LightGBM/XGBoost 替换为 PyTorch MLP（NeuralNetFactorModel）。
"""
from __future__ import annotations

import pickle
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from core.factors.train_ml_model import MLModelTrainer
from core.factors.feature_selector import CrossSectionalFeatureSelector
from config.factor_config import TrainingConfig, OptimizationConfig
from core.neural.nn_models import NeuralNetFactorModel, MultiObjectiveNeuralModel


class NeuralTrainer:
    """封装数据准备与多目标神经网络训练。"""

    def __init__(self, db_path: str, neural_cfg: Optional[Any] = None):
        self.db_path = db_path
        self.neural_cfg = neural_cfg
        self._trainer = MLModelTrainer(db_path=db_path)
        self.norm_stats: Optional[Dict[str, Any]] = None

    # ----------------------------------------------------------- 数据准备
    def build_dataset(
        self,
        trainer_stocks: List[str],
        train_start_date: str,
        train_end_date: str,
        target_features: Optional[List[str]] = None,
        workers: int = 12,
        return_codes: bool = True,
    ) -> Dict[str, Any]:
        """复用 MLModelTrainer 产出特征矩阵与对齐后的多目标标签。

        返回字典包含：
            X            : np.ndarray(float32) 原始（未横截面归一化）特征矩阵
            aligned      : pd.DataFrame         含 rank_<objective> 多目标标签
            factor_names : List[str]
            dates        : np.ndarray
            returns      : np.ndarray
            codes        : np.ndarray
        """
        print(f"\n[NN-Step 1] 读取标签行情数据 ({train_start_date} ~ {train_end_date})...")
        stocks_data = self._trainer.load_label_data(
            trainer_stocks, train_start_date, train_end_date
        )

        print(f"\n[NN-Step 2] 准备特征数据集与多目标标签...")
        full_dataset = self._trainer.prepare_dataset(
            stocks_data,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            include_fundamentals=True,
            n_jobs=workers,
            target_features=target_features,
            use_factor_cache_only=True,
            return_codes=True,
        )
        (X, y, returns, factor_names, dates, unbuyable_mask, limit_groups,
         path_scores, is_st_arr, w_sig_arr, codes) = full_dataset

        print(f"\n[NN-Step 3] 构造多目标标签（风险调整权重，独立于 GBM）...")
        from core.factors.multi_objective_labels import (
            MultiObjectiveLabelBuilder,
            cross_sectional_rank_targets,
        )
        from config import neural_config as _nc
        builder = MultiObjectiveLabelBuilder(
            return_horizons=getattr(
                TrainingConfig, "MULTI_OBJECTIVE_RETURN_HORIZONS", (5, 20, 60)
            ),
            risk_horizon=getattr(TrainingConfig, "MULTI_OBJECTIVE_RISK_HORIZON", 20),
            orthogonal_legs=getattr(TrainingConfig, "MULTI_OBJECTIVE_ORTHOGONAL_LEGS", True),
            use_matching_risk_for_sharpe=getattr(
                TrainingConfig, "MULTI_OBJECTIVE_MATCHING_RISK_SHARPE", True
            ),
        )
        raw_labels = builder.build_universe(stocks_data, train_start_date, train_end_date)
        target_cols = [
            c for c in _nc.NeuralConfig.MULTI_OBJECTIVE_WEIGHTS.keys()
            if c in raw_labels.columns
        ]
        multi_labels = cross_sectional_rank_targets(
            raw_labels, target_cols, risk_cols=_nc.NeuralConfig.RISK_COLS
        )
        del stocks_data
        import gc
        gc.collect()

        keys = pd.DataFrame({
            "date": pd.Series(dates).astype(str).str[:10],
            "code": pd.Series(codes).astype(str),
            "__row_order": np.arange(len(dates)),
        })
        label_frame = multi_labels.copy()
        label_frame["date"] = label_frame["date"].astype(str).str[:10]
        label_frame["code"] = label_frame["code"].astype(str)
        aligned = keys.merge(label_frame, on=["date", "code"], how="left", sort=False)
        aligned = aligned.sort_values("__row_order").reset_index(drop=True)

        return {
            "X": X, "aligned": aligned, "factor_names": factor_names,
            "dates": dates, "returns": returns, "codes": codes,
            "unbuyable_mask": unbuyable_mask, "is_st_arr": is_st_arr,
            "w_sig_arr": w_sig_arr,
        }

    # ------------------------------------------------------------ 时间切分
    @staticmethod
    def _time_split(dates: np.ndarray, forward_days: int, split_ratio: float):
        """与 MLModelTrainer.train_models 一致的时间序列划分 + Embargo 阻隔期。"""
        raw_split_idx = int(len(dates) * split_ratio)
        split_date = dates[raw_split_idx]
        split_idx = int(np.searchsorted(dates, split_date, side="left"))
        unique_dates = np.unique(dates)
        split_date_idx = int(np.searchsorted(unique_dates, split_date))
        val_start_date = unique_dates[min(split_date_idx + forward_days,
                                          len(unique_dates) - 1)]
        val_start_idx = int(np.searchsorted(dates, val_start_date, side="left"))
        train_idx = np.arange(split_idx)
        val_idx = np.arange(val_start_idx, len(dates))
        return train_idx, val_idx, split_date, val_start_date

    # ----------------------------------------------------------- 多目标训练
    def train_multiobjective(
        self,
        dataset: Dict[str, Any],
        objective_weights: Optional[Dict[str, float]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MultiObjectiveNeuralModel, List[str], Dict[str, Any], Dict[str, Any]]:
        """特征选择 + 横截面归一化 + 逐目标训练神经网络。

        参数：
            dataset          : build_dataset 的返回值
            objective_weights: 目标权重（默认 TrainingConfig.MULTI_OBJECTIVE_WEIGHTS）
            model_kwargs     : 传入 NeuralNetFactorModel 的超参覆盖
        返回：
            (multi_model, selected_names, norm_stats, objective_results)
        """
        X = np.asarray(dataset["X"], dtype=np.float32)
        aligned = dataset["aligned"]
        factor_names = dataset["factor_names"]
        dates = np.asarray(dataset["dates"])
        returns = np.asarray(dataset["returns"], dtype=np.float32)

        # 神经网络默认使用 NeuralConfig 的多目标权重（强调风险调整收益），
        # 与 GBM 的 TrainingConfig.MULTI_OBJECTIVE_WEIGHTS 解耦。
        if objective_weights is None:
            from config import neural_config as _nc
            weights = dict(_nc.NeuralConfig.MULTI_OBJECTIVE_WEIGHTS)
        else:
            weights = dict(objective_weights)
        objective_cols = []
        norm_weights = {}
        for raw_name, w in weights.items():
            rank_name = raw_name if raw_name.startswith("rank_") else f"rank_{raw_name}"
            if rank_name in aligned.columns and w > 0:
                objective_cols.append(rank_name)
                norm_weights[rank_name] = float(w)
        if not objective_cols:
            raise ValueError("aligned 中无与 MULTI_OBJECTIVE_WEIGHTS 匹配的 rank 目标")

        y_matrix = aligned[objective_cols].to_numpy(dtype=np.float32)
        complete = np.isfinite(y_matrix).all(axis=1)
        if complete.sum() < 100:
            raise ValueError(f"多目标完整样本不足: {int(complete.sum())}")
        X_fit = X[complete]
        y_fit = y_matrix[complete]
        dates_fit = dates[complete]

        # 与 train_multiobjective_models 相同的时间切分口径
        raw_selection_split = int(len(dates_fit) * TrainingConfig.TRAIN_TEST_SPLIT)
        if raw_selection_split <= 0 or raw_selection_split >= len(dates_fit):
            raise ValueError("多目标训练时间切分后样本不足")
        selection_split_date = dates_fit[raw_selection_split]
        selection_train_mask = dates_fit < selection_split_date
        if selection_train_mask.sum() < 100:
            raise ValueError("多目标特征筛选训练样本不足")

        # 跳过无横截面变化的目标
        variable_mask = np.zeros(y_fit.shape[1], dtype=bool)
        sel_dates = dates_fit[selection_train_mask]
        sel_targets = y_fit[selection_train_mask]
        for d in np.unique(sel_dates):
            day_vals = sel_targets[sel_dates == d]
            if len(day_vals) > 1:
                variable_mask |= (np.nanmax(day_vals, axis=0) -
                                  np.nanmin(day_vals, axis=0)) > 1e-8
        if not variable_mask.all():
            skipped = [n for n, keep in zip(objective_cols, variable_mask) if not keep]
            print(f"  [NN 多目标] 跳过无横截面变化目标: {skipped}")
            objective_cols = [n for n, keep in zip(objective_cols, variable_mask) if keep]
            y_fit = y_fit[:, variable_mask]
            norm_weights = {n: w for n, w in norm_weights.items() if n in objective_cols}
        if not objective_cols:
            raise ValueError("所有多目标标签在训练样本中均无横截面变化")

        # 特征选择（复用与 GBM 完全相同的 selector + 横截面归一化）
        selector = CrossSectionalFeatureSelector(
            max_features=getattr(OptimizationConfig, "N_FEATURES_TO_SELECT", 200),
            min_coverage=0.20,
            corr_threshold=getattr(OptimizationConfig, "CORRELATION_THRESHOLD", 0.95),
        )
        ordered_weights = [norm_weights[n] for n in objective_cols]

        raw_selection_X = X_fit[selection_train_mask]
        feature_coverage = np.isfinite(raw_selection_X).mean(axis=0)
        selection_X = raw_selection_X.copy()
        skip_col_stats = self._trainer._apply_cross_sectional_normalization_inplace(
            selection_X, dates_fit[selection_train_mask], factor_names
        )
        selector.fit(
            selection_X, factor_names, y_fit[selection_train_mask],
            target_weights=ordered_weights, feature_coverage=feature_coverage,
        )
        selected_names = list(selector.report_.selected_features)
        selected_indices = [factor_names.index(n) for n in selected_names]
        X_selected = X_fit[:, selected_indices]
        print(f"  [NN 多目标] 特征选择: {len(factor_names)} -> {len(selected_names)}")

        # 用训练段得到的 skip_col_stats 归一化全量（train+val 一致，无未来泄漏）
        self._trainer._apply_cross_sectional_normalization_inplace(
            X_selected, dates_fit, factor_names[:len(selected_names)],
            skip_col_stats=skip_col_stats,
        )
        self.norm_stats = {
            "skip_col_stats": skip_col_stats,
            "factor_names": selected_names,
        }

        forward_days = getattr(TrainingConfig, "FUTURE_DAYS", 7)
        train_idx, val_idx, split_date, val_start_date = self._time_split(
            dates_fit, forward_days, TrainingConfig.TRAIN_TEST_SPLIT
        )
        print(f"  [NN 多目标] 划分: 训练 {len(train_idx)} / 验证 {len(val_idx)} "
              f"(split={split_date}, val_start={val_start_date})")

        mk = dict(model_kwargs or {})
        trained_models: Dict[str, NeuralNetFactorModel] = {}
        objective_results: Dict[str, Any] = {}

        for idx, objective in enumerate(objective_cols):
            print(f"\n[NN 多目标训练] {objective} ({idx+1}/{len(objective_cols)})")
            y_obj = y_fit[:, idx]
            net = NeuralNetFactorModel(
                input_dim=X_selected.shape[1],
                feature_names=selected_names,
                **mk,
            )
            net.fit(
                X_selected[train_idx], y_obj[train_idx],
                X_selected[val_idx], y_obj[val_idx],
                verbose=True,
            )
            val_pred = net.predict(
                pd.DataFrame(X_selected[val_idx], columns=selected_names)
            )
            val_metrics = self._evaluate_predictions(
                val_pred, y_obj[val_idx],
                dates_fit[val_idx], returns[complete][val_idx],
            )
            objective_results[objective] = {"val_metrics": val_metrics}
            print(f"  [NN {objective}] 验证 Rank IC: "
                  f"{val_metrics.get('rank_ic', 0):.4f} ± "
                  f"{val_metrics.get('rank_ic_std', 0):.4f} | "
                  f"Top-1: {val_metrics.get('top1_precision', 0):.2%}")
            trained_models[objective] = net

        wrapper = MultiObjectiveNeuralModel(
            trained_models,
            {name: norm_weights[name] for name in trained_models},
        )
        return wrapper, selected_names, self.norm_stats, objective_results

    # --------------------------------------------------------------- 评估
    @staticmethod
    def _evaluate_predictions(
        preds: np.ndarray, y_true: np.ndarray,
        dates_val: np.ndarray, returns_val: np.ndarray,
    ) -> Dict[str, float]:
        """组内 Rank IC / Top-N 精度（与 MLFactorModel._evaluate 同口径）。

        preds   : 模型对验证集输出的 [0,1] 分数
        y_true  : 验证集标签（rank 软标签，[0,1]）
        returns_val : 验证集真实收益（用于 Top-N 绝对精度）
        """
        rank_ics, top1_hits, top5_hits = [], [], []
        for d in np.unique(dates_val):
            mask = dates_val == d
            if mask.sum() < 10:
                continue
            g_p = preds[mask]
            g_ref = returns_val[mask] if returns_val is not None else y_true[mask]
            if len(np.unique(g_ref)) > 1 and len(np.unique(g_p)) > 1:
                ic, _ = spearmanr(g_p, g_ref)
                if not np.isnan(ic):
                    rank_ics.append(ic)
            if len(g_ref) >= 10:
                top1_idx = int(np.argmax(g_p))
                top1_hits.append(
                    1.0 if g_ref[top1_idx] >= np.percentile(g_ref, 95) else 0.0
                )
                n_top = min(5, len(g_ref))
                top5_idx = np.argsort(g_p)[-n_top:]
                top5_hits.append(
                    float(np.mean(g_ref[top5_idx] >= np.percentile(g_ref, 80)))
                )
        return {
            "rank_ic": float(np.mean(rank_ics)) if rank_ics else 0.0,
            "rank_ic_std": float(np.std(rank_ics)) if rank_ics else 0.0,
            "top1_precision": float(np.mean(top1_hits)) if top1_hits else 0.0,
            "top5_precision": float(np.mean(top5_hits)) if top5_hits else 0.0,
        }

    # --------------------------------------------------------------- 保存
    def save_artifacts(
        self,
        multi_model: MultiObjectiveNeuralModel,
        selected_names: List[str],
        save_dir: str,
        norm_stats: Dict[str, Any],
    ):
        """保存多目标 NN 模型、norm_stats 与特征摘要。"""
        import os, json, shutil
        from datetime import datetime

        archive_dir = os.path.join(
            save_dir, "neural_multi_objective_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        latest_dir = os.path.join(save_dir, "latest_neural")
        os.makedirs(archive_dir, exist_ok=True)
        os.makedirs(latest_dir, exist_ok=True)

        archive_model = os.path.join(archive_dir, "neural_multi_objective_model.pkl")
        latest_model = os.path.join(latest_dir, "neural_multi_objective_model.pkl")
        multi_model.save_model(archive_model)
        shutil.copy2(archive_model, latest_model)

        for target_dir in (archive_dir, latest_dir):
            with open(os.path.join(target_dir, "norm_stats.pkl"), "wb") as f:
                pickle.dump(norm_stats, f)
            with open(os.path.join(target_dir, "neural_config.json"), "w",
                      encoding="utf-8") as f:
                json.dump({
                    "objectives": list(multi_model.models.keys()),
                    "weights": multi_model.weights,
                    "selected_features": selected_names,
                }, f, ensure_ascii=False, indent=2)
            with open(os.path.join(target_dir, "selected_features.txt"), "w",
                      encoding="utf-8") as f:
                f.write("\n".join(selected_names))

        print(f"\n=== 神经网络多目标模型已保存: {archive_model} ===")
        return archive_dir
