"""
验证 XGB category_share 修复：
- 复现“旧逻辑”（直接调 b.feature_importance）确认全 0 的 bug
- 用新 _gain_importance 确认 XGB 现在能拿到非零 gain
- 计算 XGB / LGB 的 财务/交易/估值 三类边际贡献，确认非 0 且合理
"""
from __future__ import annotations
import os, sys
import numpy as np

PROJ = r"G:\quantified_decision"
sys.path.insert(0, PROJ)
from analysis.common import load_model
from analysis import shap_analysis as shp
from analysis.common import classify_features

XGB = os.path.join(PROJ, "models", "exp3yr_xgb", "latest", "multi_objective_factor_model.pkl")
LGB = os.path.join(PROJ, "models", "exp3yr_lgb", "latest", "multi_objective_factor_model.pkl")


def compute(model, label):
    selected = list(model.feature_names)
    weights = model.weights
    subs = shp._get_submodels(model)
    accum = np.zeros(len(selected), dtype=float)
    per_obj = {}
    for obj, sub in subs.items():
        gi = shp._gain_importance(sub, selected)
        per_obj[obj] = dict(zip(selected, gi))
        w = weights.get(obj, 1.0 / len(subs))
        accum += w * gi
    importance = dict(zip(selected, accum))

    # 复现“旧的坏逻辑”：直接 b.feature_importance (仅 LGB 支持)
    bad = np.zeros(len(selected))
    for obj, sub in subs.items():
        m = getattr(sub, "model", None)
        booster = getattr(m, "booster_", None) or m
        try:
            gi = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
            if len(gi) == len(selected):
                bad += weights.get(obj, 1.0 / len(subs)) * gi
        except Exception:
            pass  # XGB 在此静默全 0

    cat = shp.category_contribution({"importance": importance})
    total_gain = float(np.sum(accum))
    print(f"\n===== {label} =====")
    print(f"  总 gain 和: {total_gain:.4f}   |Importance 非零特征数: {int(np.sum(accum > 0))}/{len(selected)}")
    print(f"  [旧逻辑 feature_importance] 总 gain 和: {float(np.sum(bad)):.4f}  "
          f"(XGB 应≈0 => 印证 bug)")
    print(f"  三类边际贡献占比: "
          f"财务={cat['share']['财务']*100:.1f}%  "
          f"交易={cat['share']['交易']*100:.1f}%  "
          f"估值={cat['share']['估值']*100:.1f}%  "
          f"其他={cat['share']['其他']*100:.1f}%")
    # Top5
    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top5: " + ", ".join(f"{n}({v:.3f})" for n, v in top))
    return cat


if __name__ == "__main__":
    mx = load_model(XGB)
    ml = load_model(LGB)
    compute(mx, "XGB (exp3yr_xgb)")
    compute(ml, "LGB (exp3yr_lgb)")
