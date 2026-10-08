"""
配置模块入口
"""
from .baostock_config import *
from .strategy_config import *
from .factor_config import (
    ModelConfig, TrainingConfig, FactorConfig, OptimizationConfig,
)

__all__ = [
    # 从 baostock_config 导出
    'DATABASE_PATH', 'DATABASE_DIR', 'USER_DB_PATH', 'PROJECT_ROOT',
    'HISTORY_YEARS', 'WORKERS_NUM', 'REQUEST_INTERVAL', 'API_DAILY_QUOTA',
    'DEFAULT_MARKETS', 'MARKET_LIMITS', 'MARKET_PREFIXES',
    'SUPPORTED_MARKETS',
    # baostock 超时相关（2026-09-23 覆盖率事故后新增，注意与 baostock_config.__all__ 同步）
    'TASK_TIMEOUT_SECONDS', 'TASK_HARD_TIMEOUT_SECONDS', 'TASK_MAX_STALLS',
    'MAX_POOL_RESTARTS', 'BAOSTOCK_SOCKET_TIMEOUT',
    'COVERAGE_ACTIVE_WINDOW_DAYS', 'DATABASE_PATH', 'DATABASE_DIR',
    
    # 从 strategy_config.py 导出
    'INCLUDE_ST', 'ML_FACTOR_MIN_CONFIDENCE', 'SELECTOR_MARKETS',
    
    # 从 factor_config.py 导出
    'ModelConfig', 'TrainingConfig', 'FactorConfig', 'OptimizationConfig',
]
