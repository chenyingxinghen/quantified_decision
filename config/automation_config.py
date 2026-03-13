"""
自动化交易配置文件

包含 easytrader 运行所需的设置，如开户券商、登录信息文件路径等。
"""

import os

# ==============================================================================
# 1. 基础设置
# ==============================================================================

from config.strategy_config import TIME_STOP_DAYS, TIME_STOP_MIN_LOSS_PCT

# 同花顺 GUI 自动化：'ths'
TRADER_TYPE = 'ths'

# 配置文件路径: 该文件包含登陆账号、密码、客户端可执行程序路径等信息
# 具体格式参照 easytrader 的文档: https://github.com/shidenggui/easytrader
CONFIG_JSON_PATH = os.path.join(os.getcwd(), 'config', 'trader.json')

# 是否启用模拟模式 (Dry Run): 仅生成日志和虚拟交易记录，不发单到券商
DRY_RUN = False

# 是否允许在多窗口模式下执行
MULTI_WINDOW = True


# ==============================================================================
# 2. 交易时间窗
# ==============================================================================

# 开盘买入时间窗点 (格式: HH:MM:SS)
# 策略: 开盘集合竞价后立即挂涨停价买入，确保以开盘价成交（对齐回测的 next_day_open 成交逻辑）
BUY_WINDOW_START = "09:20:00"
BUY_WINDOW_END   = "09:26:00"

# 尾盘卖出时间窗点
# 策略: 在尾盘集合竞价前挂跌停价卖出，确保以当日收盘价附近成交（对齐回测以当日 close 成交的尾盘逻辑）
SELL_WINDOW_START = "14:50:00"
SELL_WINDOW_END   = "14:57:00"


# ==============================================================================
# 3. 仓位策略 (同步回测逻辑)
# ==============================================================================

# 最大持仓数量 (与 strategy_config.MAX_POSITIONS 保持一致)
MAX_POSITIONS_AUTO = 1

# 单只股票最大买入比例 (占可用资金的百分比)
SINGLE_BUY_RATIO = 1.0 / MAX_POSITIONS_AUTO

# 买入金额保留余地 (元)，避免资金不足或滑点
CASH_BUFFER = 100


# ==============================================================================
# 4. 模型 & 信号相关
# ==============================================================================

# 自动化交易使用的模型路径
AUTO_MODEL_PATH = 'models/mark/automation'  # 或者具体的 pkl 文件路径

# 信号生成时使用的最低置信度阈值（百分制，0.0 表示不过滤）
AUTO_MIN_CONFIDENCE = 0.0

# 每次选股最多产生多少信号（top_n），与 MAX_POSITIONS_AUTO 相同时最精准
AUTO_TOP_N = MAX_POSITIONS_AUTO

# 信号历史记录保存路径
POSITIONS_TRACKING_PATH = 'data/automation/positions.json'


# ==============================================================================
# 5. 自动化专属选股筛选条件
#    独立于 strategy_config.py，专门用于实盘自动交易，可单独调整。
#    设置为 None 则不启用该筛选条件（等价于关闭）。
# ==============================================================================

# 是否启用自动化选股的基础条件筛选（市值/PE/股价/ST）
AUTO_APPLY_FILTER = True

# 最小流通市值（亿元），过滤微盘股。None = 不限制
AUTO_MIN_MARKET_CAP = None       # 至少 20 亿市值

# 最大市盈率（倍），过滤估值过高的股票。None = 不限制
AUTO_MAX_PE = None              # PE 不高于 150 倍

# 股价区间（元），过滤极低价或高价股。None = 不限制
AUTO_MIN_PRICE = 1.0             # 最低 1 元
AUTO_MAX_PRICE = 20.0           # 最高 20 元

# 是否包含 ST / *ST 股票。实盘建议设为 False 规避退市风险
AUTO_INCLUDE_ST = True

# 时间止损专属参数（与回测 TIME_STOP_DAYS / TIME_STOP_MIN_LOSS_PCT 对齐）
# 若持有 >= AUTO_TIME_STOP_DAYS 个交易日，且浮亏 >= AUTO_TIME_STOP_MIN_LOSS_PCT，则尾盘清仓
AUTO_TIME_STOP_DAYS = TIME_STOP_DAYS          # 时间止损天数（交易日）
AUTO_TIME_STOP_MIN_LOSS_PCT = TIME_STOP_MIN_LOSS_PCT  # 时间止损最小亏损比例
