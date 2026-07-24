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


def _get_submodels(model) -> dict:
    """返回 {objective: MLFactorModel 子模型}。"""
    return dict(model.models)


def _booster_object(sub):
    """返回 (booster, kind) 供 SHAP TreeExplainer / gain 提取使用。

    - LightGBM: 取 m.booster_（sklearn 包装下的原生 Booster）
    - XGBoost:  取 m.get_booster()（sklearn 包装下的原生 Booster）
    """
    m = getattr(sub, "model", None)
    booster = getattr(m, "booster_", None)
    if booster is not None:
        return booster, "lgb"
    if m is not None and hasattr(m, "get_booster"):
        try:
            return m.get_booster(), "xgb"
        except Exception:
            return m, "xgb"
    return m, "unknown"


def _gain_importance(sub, selected: list) -> np.ndarray:
    """返回与 selected 顺序对齐的 gain 重要性数组（兼容 LightGBM / XGBoost）。

    修复点：旧实现直接调用 b.feature_importance(importance_type="gain")，
    该 API 仅 LightGBM 提供；XGBoost 的 sklearn 模型无此方法，会抛
    AttributeError 并被外层捕获为全 0，导致三类边际贡献全部为 0.0%。

    兼容以下三种底层形态（本项目 XGB 子模型 pickle 后 sub.model 即 xgb.Booster）：
      - LightGBM Booster:        booster.feature_importance(importance_type="gain")
      - XGBoost Booster (裸):     booster.get_score(importance_type="gain")
      - XGBoost sklearn 包装:     model.get_booster().get_score(...)
    并把 get_score 返回的 {特征名 / f0..索引: gain} 按 selected 顺序对齐
    （f0/f1... 按索引映射，与训练列序一致，逻辑对齐
     MLFactorModel._calculate_feature_importance）。
    """
    m = getattr(sub, "model", None)
    if m is None:
        return np.zeros(len(selected), dtype=float)

    # ── LightGBM: sklearn 包装下的原生 Booster ──
    booster = getattr(m, "booster_", None)
    if booster is not None and hasattr(booster, "feature_importance"):
        try:
            gi = np.asarray(booster.feature_importance(importance_type="gain"),
                            dtype=float)
            if len(gi) == len(selected):
                return gi
        except Exception:
            pass

    # ── XGBoost: 取得 booster（裸 Booster 直接用 m，sklearn 包装用 get_booster）──
    xgb_booster = None
    if hasattr(m, "get_score") and not hasattr(m, "feature_importance"):
        xgb_booster = m  # m 本身就是 xgb.Booster
    elif hasattr(m, "get_booster"):
        try:
            xgb_booster = m.get_booster()
        except Exception:
            xgb_booster = None
    if xgb_booster is not None and hasattr(xgb_booster, "get_score"):
        try:
            score = xgb_booster.get_score(importance_type="gain")
        except Exception:
            score = {}
        if score:
            first_key = next(iter(score))
            # f0, f1, ... 默认索引命名 -> 按位映射回 selected
            if (first_key.startswith("f") and first_key[1:].isdigit()
                    and first_key not in selected):
                gi = np.zeros(len(selected), dtype=float)
                for k, v in score.items():
                    if k.startswith("f") and k[1:].isdigit():
                        idx = int(k[1:])
                        if idx < len(selected):
                            gi[idx] = float(v)
                return gi
            # 实际特征名命名
            return np.asarray([float(score.get(name, 0.0)) for name in selected],
                              dtype=float)

    # ── sklearn-style attribute 兜底（极少触发）──
    if hasattr(m, "feature_importances_"):
        fi = np.asarray(m.feature_importances_, dtype=float)
        if len(fi) == len(selected):
            return fi

    return np.zeros(len(selected), dtype=float)


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
    subs = _get_submodels(model)
    Xsel, _ = _selected_matrix(Xn, factor_names, selected, n_sample, rng)

    per_objective: dict = {}
    accum = np.zeros(len(selected), dtype=float)

    try:
        import shap  # 可选依赖
        for obj, sub in subs.items():
            try:
                b, _ = _booster_object(sub)
                exp = shap.TreeExplainer(b)
                sv = np.asarray(exp.shap_values(Xsel), dtype=float)
                if sv.ndim == 3:
                    sv = sv.sum(axis=2)  # 多分类保护（本模型为回归/排序，通常 2D）
                mean_abs = np.abs(sv).mean(axis=0)
            except Exception:
                mean_abs = _gain_importance(sub, selected)
            per_objective[obj] = dict(zip(selected, mean_abs))
            accum += weights.get(obj, 1.0 / len(subs)) * mean_abs
        method = "shap"
    except Exception:
        # 回退：gain 重要性（LightGBM / XGBoost 均已兼容）
        for obj, sub in subs.items():
            gi = _gain_importance(sub, selected)
            per_objective[obj] = dict(zip(selected, gi))
            accum += weights.get(obj, 1.0 / len(subs)) * gi
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

    subs = _get_submodels(model)
    obj0 = next(iter(subs))
    try:
        import shap
        b0, _ = _booster_object(subs[obj0])
        exp = shap.TreeExplainer(b0)
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
