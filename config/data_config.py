"""
统一数据源配置文件
"""
import os

# ==================== 项目路径配置 ====================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== 数据库配置 ====================

# 数据库目录
# 云端（Gemini 等平台）适配：当设置 GEMINI_DATA_IN1 时，所有数据库路径自动指向
# 平台的只读数据挂载目录；本地运行不设置该变量时回退到项目内 database/。
DATABASE_DIR = os.getenv("GEMINI_DATA_IN1", os.path.join(PROJECT_ROOT, "database"))

# 云端适配（raw-only 策略）：以下"产出类"库可用环境变量覆盖到可写的 $GEMINI_DATA_OUT，
# 仅把"源"库（jydb_raw.db）留在只读挂载点。本地不加这些变量时回退默认相对路径。
#   - 训练主库（行情库）由 build_intermediate_from_raw.py 从 raw 重建 -> 指到 $GEMINI_DATA_OUT
#   - 元数据库（market_sentiment）由 build 顺带产出 -> 指到 $GEMINI_DATA_OUT
#   - 财务库为旧 Baostock 遗留，训练不使用，仅回测/选股用到
DATABASE_PATH   = os.getenv("GEMINI_DATABASE_PATH", os.path.join(DATABASE_DIR, "stock_daily.db"))
META_DB_PATH    = os.getenv("GEMINI_META_DB_PATH", os.path.join(DATABASE_DIR, "stock_meta.db"))
FINANCE_DB_PATH = os.getenv("GEMINI_FINANCE_DB_PATH", os.path.join(DATABASE_DIR, "stock_finance.db"))

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
WORKERS_NUM = 1

# 请求间隔（秒）
REQUEST_INTERVAL = 0.01


# 增量更新配置
INCREMENTAL_UPDATE = True  # 默认使用增量更新
CHECK_LAST_N_DAYS = 5      # 检查最近N天的数据完整性
AUTO_FILL_GAPS = False      # 自动填补历史数据缺口

# 会话最大复用次数 
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
    'st': 0.05,        # 主板ST股票 (5%)；创业板/科创板ST仍为20%，北交所ST仍为30%
    'gem_star': 0.198,  # 创业板/科创板 (20%)
    'bj': 0.295,        # 北交所 (30%)
    'main': 0.098       # 主板 (10%)
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
