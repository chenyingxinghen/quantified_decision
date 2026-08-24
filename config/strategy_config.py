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
ML_FACTOR_RISK_MIN_PRICE = 1.0   # 独立于基本面筛选，排除极端低价退市风险
ML_FACTOR_RISK_EXCLUDE_ST = True # 独立于基本面筛选，默认排除 ST / *ST

# 置信度口径：固定参考分布校准（使置信度跨日可比，每日 top 不再恒为 100）
# True  → 置信度 = 原始模型输出在「固定历史窗口参考分布」中的百分位 (0~100)
# 置信度 z-score 固定尺度（不是逐日重排）：confidence = clip(50 + K·(raw−μ)/σ, 0, 100)。
# 固定 K 把横截面 spread 拉开显示（top 不再钉 100，top-20 可区分），且排序不变。
# 调大 K → 顶部/底部更两极分化；调小 → 更向 50 收敛。可按实盘手感微调。
CONFIDENCE_Z = 10.0

# ==============================================================================
# 模型载体（单一事实来源）
# ==============================================================================
# 回测、实盘自动化、Web 后端三条链路都从这里取模型路径，不再各自维护一份。
# automation_config 里的 AUTO_MODEL_PATH / AUTO_NORM_STATS_PATH /
# AUTO_ENSEMBLE_MODEL_PATHS 只是本节的别名转发，换模型只改这里。
#
# 模型载体分工（2026-08-22 明确）：
#   · 生产实盘载体放在 models/mark/ 下（models/nam_gate、models/tree 是训练产物
#     归档区，不直接进生产），由自动化模块 automation_config.AUTO_MODEL_PATH 直接加载，
ML_FACTOR_MODEL_PATH = 'models/nam_gate/T133_yscale4_s42'

# 归一化统计量。必须与权重**同批产出**（同一存档目录）：统计量和权重对不上会让
# 连续列以错误量纲进模型，且不报错。
# ML_FACTOR_NORM_STATS_PATH = 'models/latest/norm_stats.pkl'

# 可选：多种子**等权集成**载体（不含 ML_FACTOR_MODEL_PATH 自身）。
# ⚠ 当前留空 = 不启用。T131 两窗判定不一致，未过「必须赢最幸运单种子」的预注册门：
#   T115 窗赢 +0.00327 / T122 窗输 −0.00173 且跌日否决门不过。
#   它是「降方差」选项而非「更强」选项，要不要用是取舍，不是 IC 结论。
# 启用时必须同批产出（同窗口/同超参/同面板，只差种子），策略层硬校验特征顺序一致。
ML_FACTOR_ENSEMBLE_MODEL_PATHS: list = ['models/mark/T115_idxrel_s42/nam_gate_factor_model.pkl','models/mark/T115_idxrel_s11/nam_gate_factor_model.pkl']





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

# BUY_COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE                     # 0.0008
# SELL_COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE + STAMP_DUTY_RATE  # 0.0013

BUY_COST_RATE = 0.005                    
SELL_COST_RATE = 0.005

# ==============================================================================
# 回测系统参数
# ==============================================================================

# 基础参数
INITIAL_CAPITAL = 1.0          # 初始资金
MAX_POSITIONS = 1

# ATR相关参数（用于止损止盈计算）
ATR_PERIOD = 14                     # ATR计算周期
ATR_STOP_MULTIPLIER = 1   if TrainingConfig.SHORT_PREDICTION else 3           # ATR止损倍数 (放宽，减少噪音震出)
ATR_TARGET_MULTIPLIER = 3 if TrainingConfig.SHORT_PREDICTION else 9           # ATR目标倍数：降低至2.5x，与7天内最高价分布对齐

# 时间止损参数
TIME_STOP_DAYS = TrainingConfig.FUTURE_DAYS                 # 与FUTURE_DAYS对齐：持满预测周期再评估
TIME_STOP_MIN_LOSS_PCT = 0.15 if TrainingConfig.SHORT_PREDICTION else 0.3     # 时间止损，30%的高要求，确保超时直接卖出或锁定利润

# 卖出控制参数
ENABLE_STOP_LOSS_EXIT = True        # 是否启用止损卖出
ENABLE_TAKE_PROFIT_EXIT = True       # 是否启用止盈卖出
ENABLE_SUPPORT_BREAK_EXIT = False    # 是否启用跌破支撑卖出
ENABLE_TIME_STOP_EXIT = True         # 是否启用时间止损卖出

# 退市了结折价。
# 回测原先按停牌前最后一个可见价平价了结退市股，等价于假设退市能原价卖出。
# 现实中退市整理期普遍腰斩以上，转入老三板后流动性接近于零。0.5 是保守中值，
# 影响面很小（退市股本就极少被选中），但方向必须是对的 —— 宁可低估收益。
DELIST_EXIT_HAIRCUT = 0.5


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
