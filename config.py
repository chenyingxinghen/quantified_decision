# 配置文件
import os

# 数据库配置
DATABASE_PATH = "stock_data.db"

# 市场配置
DEFAULT_MARKETS = ['sh', 'sz_main']  # 默认只初始化上证和深圳主板
# 'sh' - 上海证券交易所 (60xxxx)
# 'sz_main' - 深圳主板 (00xxxx)
# 'sz_gem' - 创业板 (30xxxx)
# 'bj' - 北京证券交易所 (8xxxxx, 4xxxxx)

# 数据更新配置
UPDATE_INTERVAL = 300  # 5分钟更新一次
MARKET_OPEN_TIME = "09:30"
MARKET_CLOSE_TIME = "15:00"
INCREMENTAL_UPDATE = True  # 默认使用增量更新

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
    'min_turnover_rate': 2,         # 最小换手率（3%）
    'min_market_cap': 80,           # 最小市值（80亿）
    'max_pe': 50,                   # 最大市盈率
    'min_price': 1,                 # 最小股价
    'max_price': 100,               # 最大股价
}