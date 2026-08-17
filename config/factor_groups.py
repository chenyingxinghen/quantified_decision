"""
因子族分组表（NAM + RegimeGate 的门控作用单位）
================================================

为什么要分组
------------
NAM 给每个因子一条独立形状函数 ``f_i(x_i)``，但**门控不能逐因子输出权重**：
227 个因子 → 227 维 softmax，参数爆炸且极易专家坍塌（DeepSeek 研讨结论）。
折中方案是"**逐因子形状函数 + 逐族门控权重**"：

    score = β0 + Σ_k  w_t[k] · Σ_{i ∈ group k} f_i(x_i)

- ``f_i`` 保留单因子可解释性（曲线可画）
- ``w_t[k]`` 由市场状态 ``m_t`` 生成，回答"**什么行情下哪一族因子被放大**"
  —— 这正是原手工 ``{mkt}_regime_{stock}`` 交互项想表达、却表达不好的东西。

分组原则
--------
按**经济含义 + 在不同 regime 下的预期行为**分族，而非按计算方式。
例如动量与反转必须分开：牛市动量有效、震荡市反转有效，这是门控最该学到的切换。

匹配优先级（自上而下，先匹配先生效）
1. ``*_regime_*``      手工市场交互（NAM 线通常剔除，保留分组以兼容旧特征集）
2. 技术 × 基本面交叉    ``*_x_YOY*`` / ``*_div_{估值列}``
3. 技术 × 技术交叉      ``*_mul_*`` / ``*_sub_*``
4. 显式成员集合         动量 / 反转 / 波动 / 量能 / 估值 / 质量 / 成长 / 杠杆 / 状态
5. 关键词兜底           前缀 ``rank_`` / ``log_`` 剥离后重试
6. ``other``            未匹配
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

# 门控输出维度 K = len(FACTOR_GROUPS)
FACTOR_GROUPS: List[str] = [
    'momentum',       # 动量/趋势延续
    'reversal',       # 反转/超买超卖
    'volatility',     # 波动与下行风险
    'volume',         # 量能与资金流
    'value',          # 估值
    'quality',        # 盈利质量
    'growth',         # 成长
    'leverage',       # 杠杆与偿债
    'cross_tech',     # 技术×技术交叉
    'cross_fund',     # 技术×基本面交叉
    'market_regime',  # 手工市场交互（遗留）
    'status',         # 状态位/标的属性
    'indrel',         # 行业内相对位置（``*__ind``，见 E7）
    'forecast',       # 业绩预告（``fc_*``，见 T089/T090 = E18）
    'other',          # 未匹配兜底
]

GROUP_TO_ID: Dict[str, int] = {g: i for i, g in enumerate(FACTOR_GROUPS)}


# --- 显式成员集合（按经济含义手工归类）-------------------------------------

_MOMENTUM = {
    'momentum_5d', 'momentum_10d', 'momentum_20d',
    'return_5d', 'return_10d', 'return_20d', 'return_60d',
    'roc_30', 'mtm_10', 'trix_30',
    'acceleration_5d', 'acceleration_10d',
    'ma_slope_14', 'ma_ratio_50',
    'consecutive_up_days', 'days_above_ma20',
    'macd', 'macd_hist', 'macd_signal',
    'adx_20', 'plus_di', 'minus_di', 'aroon_up', 'aroon_down',
}

_REVERSAL = {
    'rsi_12', 'rsi_21', 'cmo_21',
    'kdj_k', 'kdj_d', 'kdj_j', 'stochrsi_k', 'stochrsi_d',
    'willr_20', 'cci_20', 'bias_18', 'bb_position',
    'psy_30', 'ar_26', 'br_26', 'cr_52',
    'high_position', 'low_position', 'price_percentile_60d',
    'drawdown', 'max_drawdown_20', 'rvi_14',
}

_VOLATILITY = {
    'atr_14', 'natr_28', 'sqrt_atr_14', 'sqrt_natr_28',
    'price_volatility_20', 'price_volatility_60', 'price_var_30',
    'bb_width', 'hl_range_mean', 'hl_range_std',
    'downside_risk', 'ulcer_7', 'sharpe_ratio',
    'return_skewness', 'return_kurtosis', 'price_skewness', 'price_kurtosis',
    'oc_ratio_mean', 'oc_ratio_std', 'intraday_drawdown_avg_5d',
    'volume_volatility',
}

_VOLUME = {
    'vol_ma_14', 'vol_std_30', 'volume_change_rate',
    'vroc_18', 'vrsi_21', 'vr_30', 'vmacd', 'vmacd_signal',
    'obv', 'ad', 'adosc', 'mfi_15', 'eav',
    'amount_change_rate', 'amount_ma_10', 'amount_per_volume', 'amount_std_30',
    'price_volume_corr',
}

_VALUE = {
    'dynamic_pe', 'dynamic_pb', 'inv_pe', 'inv_pb', 'peg',
    'roe_to_pb', 'epsTTM', 'market_cap', 'totalShare', 'liqaShare',
}

_QUALITY = {
    'roeAvg', 'npMargin', 'gpMargin', 'sue',
    'dupontROE', 'dupontAssetTurn', 'dupontAssetStoEquity',
    'dupontEbittogr', 'dupontIntburden', 'dupontNitogr',
    'dupontPnitoni', 'dupontTaxBurden',
}

_GROWTH = {
    'YOYAsset', 'YOYEPSBasic', 'YOYEquity', 'YOYLiability', 'YOYNI', 'YOYPNI',
    'MBRevenue', 'roe_x_np_growth',
}

_LEVERAGE = {
    'assetToEquity', 'liabilityToAsset',
    'currentRatio', 'quickRatio', 'cashRatio',
}

_STATUS = {
    'market_type', 'is_limit_up', 'is_st', 'is_suspended',
}

_EXPLICIT: List[Tuple[str, set]] = [
    ('momentum', _MOMENTUM),
    ('reversal', _REVERSAL),
    ('volatility', _VOLATILITY),
    ('volume', _VOLUME),
    ('value', _VALUE),
    ('quality', _QUALITY),
    ('growth', _GROWTH),
    ('leverage', _LEVERAGE),
    ('status', _STATUS),
]

# 交叉项识别：技术 ÷ 估值类分母 → 技术×基本面
_FUND_DENOMINATORS = ('_div_dynamic_pe', '_div_inv_pb', '_div_npMargin',
                      '_div_peg', '_div_roe_to_pb')
_RE_CROSS_FUND = re.compile(r'_x_YOY|' + '|'.join(re.escape(s) for s in _FUND_DENOMINATORS))
_RE_CROSS_TECH = re.compile(r'_mul_|_sub_|_div_')

# 关键词兜底（显式集合未覆盖的新因子）
_KEYWORD_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ('growth',     ('yoy', 'growth')),
    ('volatility', ('volatility', 'atr', 'std', 'var_', 'risk', 'skew', 'kurt', 'drawdown')),
    ('volume',     ('vol_', 'volume', 'amount', 'turnover', 'obv', 'mfi')),
    ('momentum',   ('momentum', 'return_', 'roc', 'trend', 'slope', 'macd')),
    ('reversal',   ('rsi', 'kdj', 'bias', 'stoch', 'cci', 'willr', 'psy')),
    ('value',      ('_pe', '_pb', 'peg', 'market_cap', 'share')),
    ('quality',    ('roe', 'margin', 'dupont', 'profit')),
    ('leverage',   ('ratio', 'liability', 'equity', 'debt')),
]


def assign_group(name: str) -> str:
    """把单个因子名映射到因子族。未识别时返回 ``'other'``。"""
    if '_regime_' in name:
        return 'market_regime'
    # E7：行业内相对分位（``<base>__ind``）自成一族——它和 base 因子的经济含义不同
    # （剥离了行业共同驱动），门控应该能独立放大/压制它。
    if name.endswith('__ind'):
        return 'indrel'
    # E18：业绩预告自成一族。它既不是事后财务（growth 里的 YOY*）也不是价格派生，
    # 是**前瞻**信息；而且事件驱动、覆盖率只有 19~39%，行为与其他族差别很大，
    # 门控应该能独立控制它的权重。必须放在交叉项判定之前——`fc_chg_log` 含
    # `_log`，落到关键词兜底会被误分。
    if name.startswith('fc_'):
        return 'forecast'
    if _RE_CROSS_FUND.search(name):
        return 'cross_fund'
    if _RE_CROSS_TECH.search(name):
        return 'cross_tech'

    for group, members in _EXPLICIT:
        if name in members:
            return group

    # 剥离 rank_ / log_ / sqrt_ 前缀后重试显式集合
    base = name
    for prefix in ('rank_', 'log_', 'sqrt_'):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    if base != name:
        for group, members in _EXPLICIT:
            if base in members:
                return group

    lowered = base.lower()
    for group, keywords in _KEYWORD_RULES:
        if any(k in lowered for k in keywords):
            return group
    return 'other'


def build_group_index(feature_names: Sequence[str],
                      drop_empty: bool = True) -> Tuple[List[str], np.ndarray]:
    """
    为一组特征名生成分组索引。

    返回 ``(active_groups, group_ids)``：
      - ``active_groups``：实际用到的族名（按 ``FACTOR_GROUPS`` 原顺序）
      - ``group_ids``：``int64[n_features]``，值域 ``[0, len(active_groups))``

    ``drop_empty=True`` 时剔除空族，避免门控为不存在的族浪费一维（也是坍塌来源之一）。
    """
    raw = [assign_group(n) for n in feature_names]
    if drop_empty:
        used = [g for g in FACTOR_GROUPS if g in set(raw)]
    else:
        used = list(FACTOR_GROUPS)
    remap = {g: i for i, g in enumerate(used)}
    return used, np.asarray([remap[g] for g in raw], dtype=np.int64)


def group_summary(feature_names: Sequence[str]) -> Dict[str, List[str]]:
    """按族列出成员，用于人工核对分组是否合理。"""
    out: Dict[str, List[str]] = {}
    for n in feature_names:
        out.setdefault(assign_group(n), []).append(n)
    return {g: out[g] for g in FACTOR_GROUPS if g in out}


if __name__ == '__main__':
    import json
    import os
    import sys

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary_path = os.path.join(_root, 'models', 'latest', 'factor_summary.json')
    if not os.path.exists(summary_path):
        print(f'未找到 {summary_path}')
        sys.exit(0)
    names = json.load(open(summary_path, encoding='utf-8'))['factor_names']
    groups = group_summary(names)
    print(f'共 {len(names)} 个因子 → {len(groups)} 个族\n')
    for g, members in groups.items():
        print(f'[{g}] {len(members)}')
        print('  ' + ', '.join(members[:12]) + (' ...' if len(members) > 12 else ''))
