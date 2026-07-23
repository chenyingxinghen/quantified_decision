"""
量化因子模块

提供基于技术指标的量化因子计算和机器学习模型
"""

from .quantitative_factors import QuantitativeFactors
from .candlestick_pattern_factors import CandlestickPatternFactors
from .ml_factor_model import MLFactorModel
from .ml_strategy import MLFactorStrategy, HybridStrategy
from .multi_objective_labels import (
    MultiObjectiveLabelBuilder,
    cross_sectional_rank_targets,
    orthogonalize_labels,
    diagnose_label_orthogonality,
)

__all__ = [
    'QuantitativeFactors',
    'CandlestickPatternFactors',
    'MLFactorModel',
    'MLFactorStrategy',
    'HybridStrategy',
    'MultiObjectiveLabelBuilder',
    'cross_sectional_rank_targets',
    'orthogonalize_labels',
    'diagnose_label_orthogonality',
]
