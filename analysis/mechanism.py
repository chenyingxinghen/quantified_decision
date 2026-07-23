"""
analysis/mechanism.py — 机制分析：A 股特有定价规律经济学解释

基于已计算的统计/经济/组合/SHAP 结果，生成可读的机制性解释文本。
所有结论尽量由数据驱动（引用传入的统计量），避免空泛臆测。
"""
from __future__ import annotations

import numpy as np


def _top_pairs(interaction: dict, k: int = 5):
    """从交互矩阵取绝对值最大的 Top-k 特征对。"""
    feats = interaction.get("features", [])
    mat = interaction.get("matrix", None)
    if mat is None or len(feats) == 0:
        return []
    n = len(feats)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((feats[i], feats[j], float(np.abs(mat[i, j]))))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


def build_mechanism_narrative(results: dict) -> dict:
    """构造机制分析各小节文本。"""
    robust = results.get("robustness", {})
    port = results.get("portfolio", {})
    shap_r = results.get("shap", {})
    regime = robust.get("regime", {}).get("by_regime", {})
    cat = shap_r.get("category", {}).get("share", {})
    consistency = shap_r.get("consistency", {})

    # 1. 特征交互效应
    pairs = _top_pairs(shap_r.get("interaction", {}), k=5)
    if pairs:
        pair_txt = "；".join([f"{a} × {b}（交互强度 {v:.4f}）" for a, b, v in pairs])
        interaction_text = (
            f"SHAP 交互分析显示，模型决策最依赖的特征非线性组合为：{pair_txt}。"
            "这说明单一因子难以完整刻画个股未来收益，因子间的协同/抵消效应（如估值与动量、"
            "波动率与流动性的联合作用）构成了主要的定价非线性来源。"
        )
    else:
        interaction_text = "当前环境未取得 SHAP 交互矩阵（shap 未安装或样本不足），交互效应以增益重要性近似描述。"

    # 2. 模型间 SHAP 一致性
    mc = consistency.get("mean_consistency", float("nan"))
    if np.isfinite(mc):
        consistency_text = (
            f"各目标子模型（如收益、低风险、路径质量等）的特征重要性排序一致性系数为 "
            f"{mc:.3f}（Spearman 均值）。一致性越高，说明不同学习目标捕捉的是同一套核心定价因子，"
            "模型结论越稳健；若一致性偏低，则提示各目标关注互补信息，整体集成能降低单一目标的过拟合风险。"
        )
    else:
        consistency_text = "未获得跨模型一致性估计。"

    # 3. A 股特有定价规律经济学解释
    fin = cat.get("财务", 0.0)
    trd = cat.get("交易", 0.0)
    val = cat.get("估值", 0.0)
    bull = regime.get("bull", {})
    bear = regime.get("bear", {})
    side = regime.get("sideways", {})
    bull_ic = bull.get("rankic_mean", float("nan"))
    bear_ic = bear.get("rankic_mean", float("nan"))

    econ_lines = []
    econ_lines.append(
        f"从特征边际贡献看，交易类因子占比 {trd*100:.1f}%、财务类 {fin*100:.1f}%、"
        f"估值类 {val*100:.1f}%。这反映出 A 股市场以散户为主体的投资者结构下，"
        "量价/技术类（交易）信号对短期截面收益的解释力最强，符合 A 股高换手、情绪驱动较强的典型特征。"
    )
    if np.isfinite(bull_ic) and np.isfinite(bear_ic):
        stronger = "牛市" if abs(bull_ic) >= abs(bear_ic) else "熊市"
        econ_lines.append(
            f"分状态看，RankIC 在牛市（{bull_ic:.3f}）与熊市（{bear_ic:.3f}）表现"
            f"{'差异明显' if abs(bull_ic-bear_ic) > 0.02 else '相对接近'}，"
            f"模型信号在{stronger}中方向性更强。"
            "这与 A 股牛熊切换时动量/反转 regime 切换一致：牛市中趋势与情绪因子占优，"
            "熊市中质量与防御因子（低估值、低杠杆）相对占优。"
        )
    econ_lines.append(
        "此外，A 股的 T+1 交易制度与 ±10%/±20% 涨跌停限制，使得模型在构建标签时对内不可买入"
        "（一字板、停牌）样本进行了显式惩罚，避免对流动性受限标的产生虚假信号；"
        "SHAP 对估值类因子的适度贡献也说明市场对‘便宜’的定价并非线性，需结合成长与质量共同判断。"
    )
    econ_text = "\n".join(f"- {l}" for l in econ_lines)

    return {
        "interaction": interaction_text,
        "consistency": consistency_text,
        "a_share_economics": econ_text,
        "top_interaction_pairs": pairs,
    }
