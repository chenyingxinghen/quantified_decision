"""
策略配置文件 - 回测和交易策略相关参数

注意：
1. 此文件包含回测引擎和交易策略的参数
2. 因子计算参数已移至 factor_config.py
3. 参数命名遵循 <模块>_<功能>_<参数名> 的规范
"""
from config.factor_config import *

# ==============================================================================
# 策略配置
# ==============================================================================



# ML因子策略参数
ML_FACTOR_MIN_CONFIDENCE = 0     # 提高置信度阈值以过滤噪音
ML_FACTOR_MODEL_PATH = 'models/latest/xgboost_factor_model.pkl'  # 默认模型路径
ML_FACTOR_RISK_MIN_PRICE = 1.0   # 独立于基本面筛选，排除极端低价退市风险
ML_FACTOR_RISK_EXCLUDE_ST = True # 独立于基本面筛选，默认排除 ST / *ST





# 交易费率（A 股实际口径）
#   券商佣金 + 规费/过户费 : 单边约 0.03%（万 2.5~万 3）
#   印花税                : 仅卖出 0.05%（2023-08 由 0.1% 下调）
#   滑点                  : 单边 0.05%（次日成交的保守估计）
# 历史值为 0.005（单边 0.5%，往返 1%），比实际高约一个数量级。
# 在平均持仓 6.6 天、两年 84 次全仓换手的配置下它单独吃掉 84pp 收益，
# 足以把真实为正的选股 alpha 压成负数（见 TRAINING_ITERATIONS.md T048）。
COMMISSION_RATE = 0.0003          # 单边佣金（含规费）
STAMP_DUTY_RATE = 0.0005          # 印花税，仅卖出
SLIPPAGE_RATE = 0.0005            # 单边滑点

BUY_COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE                     # 0.0008
SELL_COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE + STAMP_DUTY_RATE  # 0.0013

# ==============================================================================
# 回测系统参数
# ==============================================================================

# 基础参数
INITIAL_CAPITAL = 1.0          # 初始资金
MAX_POSITIONS = 5

# ATR相关参数（用于止损止盈计算）
ATR_PERIOD = 14                     # ATR计算周期
ATR_STOP_MULTIPLIER = 2   if TrainingConfig.SHORT_PREDICTION else 3           # ATR止损倍数 (放宽，减少噪音震出)
ATR_TARGET_MULTIPLIER = 6 if TrainingConfig.SHORT_PREDICTION else 9           # ATR目标倍数：降低至2.5x，与7天内最高价分布对齐

# 时间止损参数
TIME_STOP_DAYS = TrainingConfig.FUTURE_DAYS                 # 与FUTURE_DAYS对齐：持满预测周期再评估
TIME_STOP_MIN_LOSS_PCT = 0.15 if TrainingConfig.SHORT_PREDICTION else 0.3     # 时间止损，30%的高要求，确保超时直接卖出或锁定利润

# 卖出控制参数
ENABLE_STOP_LOSS_EXIT = True        # 是否启用止损卖出
ENABLE_TAKE_PROFIT_EXIT = True       # 是否启用止盈卖出
ENABLE_SUPPORT_BREAK_EXIT = False    # 是否启用跌破支撑卖出
ENABLE_TIME_STOP_EXIT = True         # 是否启用时间止损卖出


# ==============================================================================
# 趋势线分析参数
# ==============================================================================

TREND_LINE_LONG_PERIOD = 50         # 长期趋势线回看周期（天）
TREND_LINE_SHORT_PERIOD = 10        # 短期趋势线回看周期（天）
TREND_BROKEN_THRESHOLD = 0.05       # 趋势线跌破阈值（5%）

# 摆动点识别参数
SWING_LONG_WINDOW = 5               # 长期数据摆动点识别窗口
SWING_SHORT_WINDOW = 2              # 短期数据摆动点识别窗口
MIN_SWING_POINTS = 2                # 最少摆动点数量
TOUCH_TOLERANCE = 0.02              # 触点容差（2%）
MIN_TOUCHES = 2                     # 趋势线最少触点数
# ==============================================================================
# 前端中基本面选股的默认过滤参数 默认不启用
# 回测场景：使用以下 sc 配置
# ==============================================================================
ENABLE_FUNDAMENTAL_FILTER = False
MIN_MARKET_CAP = 0
MAX_PE = None
MAX_ZCFZL = None
MIN_PRICE = 1
MAX_PRICE = 20
INCLUDE_ST = False
SELECTOR_MARKETS = ['sh_main', 'sz_main']

# 说明：
# 1. 后端场景：前端必须传入参数，不传或为 None 则不限制该条件
# 2. 回测场景：使用上述 sc 配置
# 3. 自动化场景：使用 automation_config 优先，未设置则使用 sc 兜底
