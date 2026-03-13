# 配置文件
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据库配置
DATABASE_DIR = "G:/ai_proj/quantified_decision/database"
DATABASE_PATH = os.path.join(DATABASE_DIR, "stock_daily.db")
USER_DB_PATH = os.path.join(DATABASE_DIR, "user_data.db")
SYSTEM_DATA_DIR = os.path.join(DATABASE_DIR, "system_data")
YEARS = 15

# 市场配置
DEFAULT_MARKETS = []  # 初始化所有
# 'sh' - 上海证券交易所 (60xxxx)
# 'sz_main' - 深圳主板 (00xxxx)
# 'sz_gem' - 创业板 (30xxxx)
# 'bj' - 北京证券交易所 (8xxxxx, 4xxxxx)

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
    'star': '688',
    'bj': ('8', '4')
}

# 数据更新配置
INCREMENTAL_UPDATE = True  # 默认使用增量更新
WORKERS_NUM = 5
QUEST_INTERVAL = 0.5 # 接口请求间隔
