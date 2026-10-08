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
WORKERS_NUM = 1

# 请求间隔（秒）
REQUEST_INTERVAL = 0.01

# 每日 Baostock API 请求配额上限（硬上限为 5 万，此处留冗余给 login / 交易日查询等未单独计数的调用）
API_DAILY_QUOTA = 45000


# 增量更新配置
INCREMENTAL_UPDATE = True  # 默认使用增量更新
CHECK_LAST_N_DAYS = 5      # 检查最近N天的数据完整性
AUTO_FILL_GAPS = False      # 自动填补历史数据缺口

# 会话最大复用次数 
SESSION_MAX_STOCKS = 10000

# 单只股票任务超时阈值（秒）。超过此时间未响应则跳过，防止进度卡死
TASK_TIMEOUT_SECONDS = 120

# 单只股票任务的「硬超时」（秒）：某只股票运行超过该时长仍不返回，判定它已卡死。
# 2026-09-23 事故：旧实现在超时分支里把**全部**剩余 future 一次性 cancel，
# 结果 5212 只里只更新了列表前 1173 只（1037 只成功），剩余 4376 只从未被尝试。
# 现在只放弃卡死的这一只（并终止占用它的工作进程），其余继续跑。
#
# 计时口径注意：进程池 worker 的启动（Windows spawn + import pandas/baostock）
# 算在它接到的第一只股票头上。实测 2026-09-24 回填时首只 600000 就因此被判卡死
# （数据本来是好的，白白放弃一只）。故取 300s，给启动留足余量。
TASK_HARD_TIMEOUT_SECONDS = 300

# 连续多少轮「无任何任务完成、也无任务在运行」后判定系统性故障（登录失效/网络断）
# 并放弃整批。单只卡死不算——那只会被单独放弃。
TASK_MAX_STALLS = 5

# 同一批次最多重建进程池的次数（卡死后需要新池续跑，防止无限重建）
MAX_POOL_RESTARTS = 3

# socket 默认超时（秒）。baostock 的 login()/recv() 没有超时参数，服务端不响应时会
# **永久阻塞** —— 2026-09-11 的定时任务因此挂起 12 天，而 max_instances=1 让
# 09-14~09-22 共 7 个交易日的定时任务全部被 skip。给 socket 设默认超时把「永久挂起」
# 变成「抛异常 → 重试/跳过」。
BAOSTOCK_SOCKET_TIMEOUT = 30

# 覆盖校验的“活跃窗口”天数。某只股票最近一根 bar 距今超过该天数即视为
# 长期停牌/退市（dead），不再计入每日覆盖校验的 total/covered/missing。
# 实测：健康日的全市场股票要么当日/昨日有 bar，要么已停牌 ≥31 天，两者零交叠，
# 30 天窗口能干净剔除死股、又不掩盖真实的“整片拉取失败”（失败股只落后 1 个交易日）。
COVERAGE_ACTIVE_WINDOW_DAYS = 30


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
    'API_DAILY_QUOTA',
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
    'TASK_HARD_TIMEOUT_SECONDS',
    'TASK_MAX_STALLS',
    'MAX_POOL_RESTARTS',
    'BAOSTOCK_SOCKET_TIMEOUT',
    'COVERAGE_ACTIVE_WINDOW_DAYS',
]
