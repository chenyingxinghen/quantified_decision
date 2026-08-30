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
BUY_WINDOW_START = "09:15:00"
BUY_WINDOW_END   = "09:30:00"

# 尾盘卖出时间窗点
# 策略: 在尾盘集合竞价前挂跌停价卖出，确保以当日收盘价附近成交（对齐回测以当日 close 成交的尾盘逻辑）
SELL_WINDOW_START = "14:50:00"
SELL_WINDOW_END   = "14:57:00"


# ==============================================================================
# 3. 仓位策略 (同步回测逻辑)
# ==============================================================================

# 最大持仓数量。
MAX_POSITIONS_AUTO = 5

# 单只股票买入比例。仅作向后兼容保留；实际预算由 execution_controller 按
# 「总资产 / MAX_POSITIONS_AUTO」计算（与回测 total_value/max_positions 同口径），
# 不再使用「可用现金 × 比例」——后者在有持仓时会让每笔仓位系统性偏小。
SINGLE_BUY_RATIO = 1.0 / MAX_POSITIONS_AUTO

# 买入金额保留余地 (元)，避免资金不足或滑点
CASH_BUFFER = 10


# ==============================================================================
# 4. 模型 & 信号相关
# ==============================================================================

# 自动化交易使用的模型与其训练期归一化统计量。
#
# ⚠ 自动化模块有自己的模型加载设置，直接指向生产载体 models/mark/
# （生产推理载体一律放在 models/mark/ 下，见 strategy_config.py「模型载体」一节）。
# 不再从 strategy_config 转发 —— 历史上转发导致实盘与回测跑在不同权重上。
# 要换实盘模型，改这里的 AUTO_MODEL_PATH 即可。
AUTO_MODEL_PATH = 'models/mark/T147_yscale2_s42/nam_gate_factor_model.pkl'
AUTO_NORM_STATS_PATH = ''
# 可选多种子等权集成载体（当前留空 = 不启用）
AUTO_ENSEMBLE_MODEL_PATHS: list = []

# 信号生成时使用的最低置信度阈值（百分制，0.0 表示不过滤）
AUTO_MIN_CONFIDENCE = 0.0

# 每次选股最多产生多少信号（top_n），与 MAX_POSITIONS_AUTO 相同时最精准
AUTO_TOP_N = MAX_POSITIONS_AUTO

# 信号历史记录保存路径


# ==============================================================================
# 5. 自动化专属选股筛选条件
#    独立于 strategy_config.py，专门用于实盘自动交易，可单独调整。
#    设置为 None 则回退使用 strategy_config.py 中的 sc 配置（自动化优先、sc 兜底）。
# ==============================================================================

# 是否启用自动化选股的基础条件筛选（市值/PE/股价/ST）
#
# **默认关闭**，与回测口径一致（sc.ENABLE_FUNDAMENTAL_FILTER = False）。
# 注意语义已经变了：筛选现在发生在模型打分**之后**（见 ml_factor_strategy），
# 所以开启它不再污染横截面 rank，只是把不合格的候选从买入名单里剔除。
# 即便如此仍建议保持关闭 —— 回测基线是在无过滤的全市场上跑出来的，
# 任何一条过滤（尤其是下面的价格上限）都会注入一个未经检验的风格暴露。
AUTO_APPLY_FILTER = False

# 最小流通市值（亿元），过滤微盘股。None = 使用 sc.MIN_MARKET_CAP 兜底
AUTO_MIN_MARKET_CAP = None       # 至少 20 亿市值

# 最大市盈率（倍），过滤估值过高的股票。None = 使用 sc.MAX_PE 兜底
AUTO_MAX_PE = None              # PE 不高于 150 倍

# 最大资产负债率（%）。None = 使用 sc.MAX_ZCFZL 兜底
AUTO_MAX_ZCFZL = None

# 股价区间（元）。None = 使用 sc.MIN_PRICE/MAX_PRICE 兜底
# ⚠ AUTO_MAX_PRICE 曾是 20.0，配合 AUTO_APPLY_FILTER=True 把选股域砍到「主板 + 20 元
# 以下」。那是一个强规模/风格暴露，且从未被回测检验过（回测跑的是无过滤全市场）。
# 现已置 None；要重新启用必须先按同样的过滤条件重跑基线。
AUTO_MIN_PRICE = None
AUTO_MAX_PRICE = None

# 是否包含 ST / *ST 股票。None = 使用 sc.INCLUDE_ST 兜底。
# 注意：ST 的排除**不依赖**上面的 AUTO_APPLY_FILTER —— 策略层有一条独立的
# PIT 风险过滤（ML_FACTOR_RISK_EXCLUDE_ST），它同样在打分之后执行，始终生效。
AUTO_INCLUDE_ST = False

# 市场类型列表。None = 使用 sc.SELECTOR_MARKETS 兜底
# ⚠ 曾硬编码为 ['sh_main','sz_main']，把创业板/科创板整个排除在外，与回测的
# 全市场池不一致。置 None 以跟随全局配置。
SELECTOR_MARKETS = None


