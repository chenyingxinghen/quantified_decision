"""
analysis/shap_analysis.py — SHAP 方法框架与特征贡献解析

覆盖指标:
  - 全局特征重要性:      SHAP 均值(|SHAP|)，按目标权重聚合（无 shap 时回退 gain/排列重要性）
  - 三类特征边际贡献:    财务 / 交易 / 估值 各类 |SHAP| 占比
  - 特征交互效应:        Top 特征 SHAP 交互矩阵
  - 模型间 SHAP 一致性:  各目标 booster 的特征重要性排序 Spearman 相关性

设计: SHAP 优先；若环境未安装 shap 包，自动回退到 LightGBM gain 重要性，保证云端离线可跑。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .common import classify_features, category_of_name


def _get_boosters(model):
    """提取每个目标的 LightGBM Booster（用于 SHAP / gain）。"""
    boosters = {}
    for obj, sub in model.models.items():
        m = getattr(sub, "model", None)
        booster = getattr(m, "booster_", None) or m
        boosters[obj] = booster
    return boosters


def _selected_matrix(Xn: np.ndarray, factor_names, selected, n_sample: int, rng):
    """按 selected 列抽取归一化特征矩阵（可采样）。"""
    idx = [factor_names.index(f) for f in selected]
    Xsub = Xn[:, idx]
    if n_sample and Xsub.shape[0] > n_sample:
        rows = rng.choice(Xsub.shape[0], n_sample, replace=False)
        Xsub = Xsub[rows]
    return pd.DataFrame(Xsub, columns=selected), idx


def global_importance(
    model, Xn: np.ndarray, factor_names, n_sample: int = 5000, seed: int = 0
) -> dict:
    """全局特征重要性。优先 SHAP，回退 gain。

    返回:
        method:        'shap' | 'gain'
        importance:    {feature: value} (mean|SHAP| 或 mean gain，按目标权重聚合)
        per_objective: {objective: {feature: value}}
        selected:      特征名列表（模型所用）
    """
    rng = np.random.default_rng(seed)
    selected = list(model.feature_names)
    weights = model.weights
    Xsel, _ = _selected_matrix(Xn, factor_names, selected, n_sample, rng)

    boosters = _get_boosters(model)
    per_objective: dict = {}
    accum = np.zeros(len(selected), dtype=float)

    try:
        import shap  # 可选依赖
        for obj, b in boosters.items():
            try:
                exp = shap.TreeExplainer(b)
                sv = np.asarray(exp.shap_values(Xsel), dtype=float)
                if sv.ndim == 3:
                    sv = sv.sum(axis=2)  # 多分类保护（本模型为回归/排序，通常 2D）
                mean_abs = np.abs(sv).mean(axis=0)
            except Exception:
                mean_abs = np.asarray(b.feature_importance(importance_type="gain"), dtype=float)
            per_objective[obj] = dict(zip(selected, mean_abs))
            accum += weights.get(obj, 1.0 / len(boosters)) * mean_abs
        method = "shap"
    except Exception:
        # 回退：LightGBM gain 重要性
        for obj, b in boosters.items():
            try:
                gi = np.asarray(b.feature_importance(importance_type="gain"), dtype=float)
            except Exception:
                gi = np.zeros(len(selected))
            per_objective[obj] = dict(zip(selected, gi))
            accum += weights.get(obj, 1.0 / len(boosters)) * gi
        method = "gain"

    importance = dict(zip(selected, accum))
    return {
        "method": method,
        "importance": importance,
        "per_objective": per_objective,
        "selected": selected,
    }


def category_contribution(gi_result: dict) -> dict:
    """三类特征边际贡献：按 |SHAP|/gain 聚合到 财务/交易/估值/其他 并计算占比。"""
    importance = gi_result["importance"]
    cat_map = classify_features(list(importance.keys()))
    contrib = {"财务": 0.0, "交易": 0.0, "估值": 0.0, "其他": 0.0}
    for f, v in importance.items():
        contrib[cat_map.get(f, "其他")] += float(v)
    total = sum(contrib.values()) or 1.0
    share = {k: v / total for k, v in contrib.items()}
    return {
        "contrib": contrib,
        "share": share,
        "total": total,
        "category_map": cat_map,
    }


def interaction_effects(
    model, Xn: np.ndarray, factor_names, top_n: int = 12,
    n_sample: int = 3000, seed: int = 1,
) -> dict:
    """Top 特征的 SHAP 交互效应矩阵（均值 |交互|）。"""
    gi = global_importance(model, Xn, factor_names, n_sample=n_sample, seed=seed)
    selected = gi["selected"]
    order = sorted(selected, key=lambda f: gi["importance"][f], reverse=True)[:top_n]
    rng = np.random.default_rng(seed)
    Xsub, _ = _selected_matrix(Xn, factor_names, order, n_sample, rng)

    boosters = _get_boosters(model)
    obj0 = next(iter(boosters))
    try:
        import shap
        exp = shap.TreeExplainer(boosters[obj0])
        inter = np.asarray(exp.shap_interaction_values(Xsub), dtype=float)
        if inter.ndim == 4:
            inter = inter.sum(axis=2)
        # 取绝对值均值
        mat = np.abs(inter).mean(axis=0)  # (top_n, top_n)
        method = "shap"
    except Exception:
        n = len(order)
        mat = np.zeros((n, n))
        method = "gain_unavailable"
    return {
        "features": order,
        "matrix": mat,
        "method": method,
    }


def cross_model_consistency(
    model, Xn: np.ndarray, factor_names, n_sample: int = 5000, seed: int = 2
) -> dict:
    """模型间 SHAP 一致性：各目标 booster 特征重要性排序的 Spearman 相关性。"""
    gi = global_importance(model, Xn, factor_names, n_sample=n_sample, seed=seed)
    per_obj = gi["per_objective"]
    objectives = list(per_obj.keys())
    features = gi["selected"]
    # 构造 features × objectives 矩阵
    mat = np.array([list(per_obj[o].values()) for o in objectives])  # (obj, feat)
    n_obj = len(objectives)
    pairs = {}
    overall = []
    for i in range(n_obj):
        for j in range(i + 1, n_obj):
            rho, _ = stats.spearmanr(mat[i], mat[j])
            key = f"{objectives[i]}~{objectives[j]}"
            pairs[key] = float(rho)
            if np.isfinite(rho):
                overall.append(rho)
    return {
        "objectives": objectives,
        "per_objective_rank_corr": pairs,
        "mean_consistency": float(np.mean(overall)) if overall else float("nan"),
        "importance_matrix": dict(zip(objectives, [dict(zip(features, r)) for r in mat])),
    }


def run_shap(model, Xn: np.ndarray, factor_names) -> dict:
    """端到端 SHAP 分析。"""
    gi = global_importance(model, Xn, factor_names)
    cat = category_contribution(gi)
    inter = interaction_effects(model, Xn, factor_names)
    consistency = cross_model_consistency(model, Xn, factor_names)
    return {
        "global": gi,
        "category": cat,
        "interaction": inter,
        "consistency": consistency,
    }
