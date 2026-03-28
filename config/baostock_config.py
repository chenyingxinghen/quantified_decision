"""
统一数据源配置文件
"""
import os

# ==================== 项目路径配置 ====================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== 数据库配置 ====================

# 数据库目录
DATABASE_DIR = os.path.join(PROJECT_ROOT, "database")

# 主数据库路径
DATABASE_PATH = os.path.join(DATABASE_DIR, "stock_daily.db")

# 元数据数据库路径
META_DB_PATH = os.path.join(DATABASE_DIR, "stock_meta.db")

# 财务数据数据库路径
FINANCE_DB_PATH = os.path.join(DATABASE_DIR, "stock_finance.db")

# 用户数据数据库路径
USER_DB_PATH = os.path.join(DATABASE_DIR, "user_data.db")

# 系统数据目录
SYSTEM_DATA_DIR = os.path.join(DATABASE_DIR, "system_data")


# ==================== 数据更新配置 ====================

# 历史数据年限
HISTORY_YEARS = 17

# 财务数据年限
FINANCE_YEARS = HISTORY_YEARS

# 并发进程数
WORKERS_NUM = 20

# 请求间隔（秒）
REQUEST_INTERVAL = 0.5


# 增量更新配置
INCREMENTAL_UPDATE = True  # 默认使用增量更新
CHECK_LAST_N_DAYS = 5      # 检查最近N天的数据完整性
AUTO_FILL_GAPS = False      # 自动填补历史数据缺口

# 会话最大复用次数 (建议 50-100 之后注销重登，防止连接缓慢)
SESSION_MAX_STOCKS = 10000

# 单只股票任务超时阈值（秒）。超过此时间未响应则跳过，防止进度卡死
TASK_TIMEOUT_SECONDS = 120


# ==================== 市场配置 ====================

# 支持的市场
SUPPORTED_MARKETS = {
    'sh_main': {
        'name': '上海主板',
        'prefixes': ['60'],
        'code': 'sh'
    },
    'sh_star': {
        'name': '上海科创板',
        'prefixes': ['68'],
        'code': 'sh'
    },
    'sz_main': {
        'name': '深圳主板',
        'prefixes': ['00'],
        'code': 'sz'
    },
    'sz_gem': {
        'name': '深圳创业板',
        'prefixes': ['30'],
        'code': 'sz'
    },
    'bj': {
        'name': '北京证券交易所',
        'prefixes': ['43', '83', '87', '92'],
        'code': 'bj'
    }
}

# 默认市场
DEFAULT_MARKETS = ['sh_main', 'sz_main']

# 市场涨跌幅限制阈值
MARKET_LIMITS = {
    'st': 0.045,        # ST 股票 (5%)
    'gem_star': 0.195,  # 创业板/科创板 (20%)
    'bj': 0.295,        # 北交所 (30%)
    'main': 0.095       # 主板 (10%)
}

# 股票代码前缀映射
MARKET_PREFIXES = {
    'sh': '60',
    'sz_main': '00',
    'sz_gem': '30',
    'star': '68',
    'bj': ('8', '4', '9')
}


# ==================== 复权配置 ====================

# 复权方式
# 1: 后复权, 2: 前复权, 3: 不复权
ADJUST_FLAG = '3'  


# ==================== 财务数据配置 ====================

# 财务数据表列表 (默认全部开启)
FINANCE_TABLES = [
    'profit_ability',
    'growth_ability',
    'balance_ability',
    'dupont',
]





# ==================== 导出配置 ====================

__all__ = [
    'PROJECT_ROOT',
    'DATABASE_DIR',
    'DATABASE_PATH',
    'META_DB_PATH',
    'FINANCE_DB_PATH',
    'USER_DB_PATH',
    'SYSTEM_DATA_DIR',
    'HISTORY_YEARS',
    'FINANCE_YEARS',
    'WORKERS_NUM',
    'REQUEST_INTERVAL',
    'INCREMENTAL_UPDATE',
    'CHECK_LAST_N_DAYS',
    'AUTO_FILL_GAPS',
    'SESSION_MAX_STOCKS',
    'SUPPORTED_MARKETS',
    'DEFAULT_MARKETS',
    'MARKET_LIMITS',
    'MARKET_PREFIXES',
    'ADJUST_FLAG',
    'FINANCE_TABLES',
    'TASK_TIMEOUT_SECONDS',
]
