"""
配置模块入口
"""
from .baostock_config import *
from .strategy_config import *
from .factor_config import (
    ModelConfig, TrainingConfig, FactorConfig, OptimizationConfig,
    FactorModelConfig
)

__all__ = [
    # 从 baostock_config 导出
    'DATABASE_PATH', 'DATABASE_DIR', 'USER_DB_PATH', 'PROJECT_ROOT',
    'HISTORY_YEARS', 'WORKERS_NUM', 'REQUEST_INTERVAL',
    'DEFAULT_MARKETS', 'MARKET_LIMITS', 'MARKET_PREFIXES',
    'SUPPORTED_MARKETS',
    
    # 从 strategy_config.py 导出
    'INCLUDE_ST', 'ML_FACTOR_MIN_CONFIDENCE', 'SELECTOR_MARKETS',
    
    # 从 factor_config.py 导出
    'ModelConfig', 'TrainingConfig', 'FactorConfig', 'OptimizationConfig',
    'FactorModelConfig'
]
