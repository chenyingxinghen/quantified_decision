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

# 最大持仓数量 (与 strategy_config.MAX_POSITIONS 保持一致)
MAX_POSITIONS_AUTO = 1

# 单只股票最大买入比例 (占可用资金的百分比)
SINGLE_BUY_RATIO = 1.0 / MAX_POSITIONS_AUTO

# 买入金额保留余地 (元)，避免资金不足或滑点
CASH_BUFFER = 100


# ==============================================================================
# 4. 模型 & 信号相关
# ==============================================================================

# 自动化交易使用的模型与其训练期归一化统计量。
#
# 载体：全量池纯加性 NAM（T105 终审：混合轴关闭、树头部选股已否证），
# 面板：T115 的 224 列（219 基础 + 5 列 index_rel，2026-08-18 按族层面证据晋级）。
#
# 为什么是 s42 而不是 holdout IC 最高的 s37（0.1246 vs 0.1186）：**不在验证集上
# 挑种子**。四种子同配方同数据，只有随机初始化不同，用验证 IC 选一个会把
# 选型偏差（台账实测稳定占 12~13%）当成真实优势带进实盘。s42 是台账全程的
# 首选种子号，与它作为基准的历史一致。
#
# norm_stats 必须与权重**同批产出**（同一存档目录）：归一化统计量和权重对不上
# 会让连续列以错误量纲进模型，且不报错。策略层默认路径就是模型同目录，
# 这里显式写出来是为了让「换模型忘了换 norm_stats」变成不可能。
AUTO_MODEL_PATH = 'models/nam_gate/T115_idxrel_s42/nam_gate_factor_model.pkl'
AUTO_NORM_STATS_PATH = 'models/nam_gate/T115_idxrel_s42/norm_stats.pkl'

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
AUTO_APPLY_FILTER = True

# 最小流通市值（亿元），过滤微盘股。None = 使用 sc.MIN_MARKET_CAP 兜底
AUTO_MIN_MARKET_CAP = None       # 至少 20 亿市值

# 最大市盈率（倍），过滤估值过高的股票。None = 使用 sc.MAX_PE 兜底
AUTO_MAX_PE = None              # PE 不高于 150 倍

# 最大资产负债率（%）。None = 使用 sc.MAX_ZCFZL 兜底
AUTO_MAX_ZCFZL = None

# 股价区间（元），过滤极低价或高价股。None = 使用 sc.MIN_PRICE/MAX_PRICE 兜底
AUTO_MIN_PRICE = None            # None = 使用 sc.MIN_PRICE 兜底，保持与回测/前端一致
AUTO_MAX_PRICE = 20.0           # 最高 20 元

# 是否包含 ST / *ST 股票。None = 使用 sc.INCLUDE_ST 兜底。实盘建议设为 False 规避退市风险
AUTO_INCLUDE_ST = False

# 市场类型列表。None = 使用 sc.SELECTOR_MARKETS 兜底
SELECTOR_MARKETS=['sh_main','sz_main']


