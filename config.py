# 配置文件
import os

# 数据库配置
DATABASE_PATH = "stock_data.db"
YEARS=10

# 市场配置
DEFAULT_MARKETS = ['sh', 'sz_main']  # 默认只初始化上证和深圳主板
# 'sh' - 上海证券交易所 (60xxxx)
# 'sz_main' - 深圳主板 (00xxxx)
# 'sz_gem' - 创业板 (30xxxx)
# 'bj' - 北京证券交易所 (8xxxxx, 4xxxxx)

# 数据更新配置
USE_PROXY = True  # 是否使用
PROXY_URL = "https://dps.kdlapi.com/api/getdps/?secret_id=o7pz3us9m7b2j7uktfek&signature=08umtmh5irm6geunoul9gt6i7i8lv07b&num={num}&format=json&sep=1&f_auth=1&generateType=1"
UPDATE_INTERVAL = 300  # 5分钟更新一次
# 临时断点
TEMP_ORDER=0
#是否补充历史数据
BACKDATE=False

MARKET_OPEN_TIME = "09:30"
MARKET_CLOSE_TIME = "15:00"
INCREMENTAL_UPDATE = True  # 默认使用增量更新
WORKERS_NUM = 4
QUEST_INTERVAL = 0.5 # 接口请求间隔
RETRY_DELAYS = [1, 10, 60, 600, 1800, 3600]  # 重试延迟时间（秒）10分钟,30分钟, 60分钟

# 技术指标参数
TECHNICAL_PARAMS = {
    'ma_short': 5,      # 短期均线
    'ma_long': 20,      # 长期均线
    'rsi_period': 14,   # RSI周期
    'macd_fast': 12,    # MACD快线
    'macd_slow': 26,    # MACD慢线
    'macd_signal': 9,   # MACD信号线
    'bb_period': 20,    # 布林带周期
    'bb_std': 2,        # 布林带标准差
}

# 选股条件
SELECTION_CRITERIA = {
    'min_turnover_rate': 1,         # 最小换手率（1%）
    'min_market_cap': 50,           # 最小市值（80亿）
    'max_pe': 80,                   # 最大市盈率
    'min_price': 1,                 # 最小股价
    'max_price': 10,               # 最大股价
}