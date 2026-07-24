"""
core/factors/feature_categories.py — 特征归类（财务 / 交易 / 估值 / 其他）

从 analysis/common.py 上移而来，作为 core 层的单一事实来源，避免
analysis 包与 core 包之间的环形依赖（feature_selector 位于 core，
但又需要按类别配额做特征选择，不能反向 import analysis）。

归类规则（与 prepare_dataset 的特征审计一致，并进一步拆分基本面为财务/估值）:
  - 估值: 市盈率/市净率/市值/PEG 等估值类
  - 财务: 盈利/成长/偿债/质量等基本面（非估值）
  - 交易: 技术面、量价、动量、波动、K线形态、市场情绪、状态、特征工程衍生
  - 其他: 以上均未匹配
"""
from __future__ import annotations

from typing import Dict, List

# 估值类特征（显式名单 + 命名模式）
_VALUATION_EXACT = {
    "dynamic_pe", "dynamic_pb", "inv_pe", "inv_pb", "market_cap",
    "peg", "roe_to_pb", "ep_ttm", "bp_ttm", "ps_ttm", "valuation_z",
}
_VALUATION_PREFIX = ("pe_", "pb_", "cap_", "ps_", "ev_", "pcf_")
_VALUATION_SUBSTR = ("_pe", "_pb", "_cap", "_peg", "valuation", "market_cap")

# 基础财务（非估值）显式名单
_FINANCIAL_EXACT = {
    "epsTTM", "roe", "roa", "roic", "liabilityToAsset", "assetToEquity",
    "profit_yoy", "revenue_yoy", "net_profit_yoy", "sue", "eav",
    "gross_margin", "operating_margin", "debt_to_asset", "current_ratio",
    "eps", "bvps", "ocfps", "roe_x_np_growth", "np_growth",
}
_FINANCIAL_PREFIX = ("roe_", "roa_", "eps", "profit", "revenue", "margin_",
                     "asset_", "debt_", "growth_", "sue", "eav", "equity_")


def _is_valuation(name: str) -> bool:
    if name in _VALUATION_EXACT:
        return True
    if any(name.startswith(p) for p in _VALUATION_PREFIX):
        return True
    if any(s in name for s in _VALUATION_SUBSTR):
        return True
    return False


def _is_financial(name: str) -> bool:
    if name in _FINANCIAL_EXACT:
        return True
    if any(name.startswith(p) for p in _FINANCIAL_PREFIX):
        return True
    return False


def classify_features(factor_names: List[str]) -> Dict[str, str]:
    """把每个特征名映射到四大类之一: 财务 / 交易 / 估值 / 其他。"""
    cat: Dict[str, str] = {}
    for name in factor_names:
        if _is_valuation(name):
            cat[name] = "估值"
        elif _is_financial(name):
            cat[name] = "财务"
        else:
            # 交易面特征库（与审计中的 technical/advanced/candle/sentiment/status）
            cat[name] = "交易"
    return cat


def category_of_name(name: str) -> str:
    if _is_valuation(name):
        return "估值"
    if _is_financial(name):
        return "财务"
    return "交易"
