"""
策略参数配置文件
集中管理所有买卖规则、策略参数
便于调整和优化
"""

# ==============================================================================
# 全局卖出控制参数
# ==============================================================================
ENABLE_TAKE_PROFIT_EXIT = False  # 是否启用止盈卖出
ENABLE_BEARISH_SIGNAL_EXIT = False  # 是否启用看空信号卖出
ENABLE_SUPPORT_BREAK_EXIT = True  # 是否启用跌破支撑卖出
ENABLE_TIME_STOP_EXIT = True  # 是否启用时间止损卖出

OPEN_WAIT_THRESHOLD=0.03            #等待回调的高开阈值（3%）
# ==================== 时间止损参数 ====================
TIME_STOP_DAYS = 5  # 持仓未盈利天数阈值（默认7天）
TIME_STOP_MIN_LOSS_PCT = -0.01  # 时间止损最小亏损比例（-2%，避免微小亏损就卖出）


# 趋势线分析参数
TREND_LINE_LONG_PERIOD = 60        # 长期趋势线回看周期（天）
TREND_LINE_SHORT_PERIOD = 18        # 短期趋势线回看周期（天）
TREND_SEGMENTS=4                    # 趋势线段数
TREND_BROKEN_THRESHOLD=0.05         # 趋势线跌破卖出容忍 5%



# ==============================================================================
# SMC流动性猎取策略参数
# ==============================================================================

# ==================== 数据周期参数 ====================
LOOKBACK_PERIOD = 60  # 回看周期（天）

# ==================== 流动性猎取参数 ====================
LIQUIDITY_SWEEP_THRESHOLD = 0.02    # 流动性扫荡阈值（2%）
SPRING_RECOVERY_BARS = 3            # Spring恢复K线数量

# ==================== 订单块参数 ====================
ORDER_BLOCK_STRENGTH = 1.5       # 订单块成交量强度倍数
ORDER_BLOCK_BREAKTHROUGH = 0.02  # 突破幅度（2%）
ORDER_BLOCK_DISTANCE = 0.03      # 订单块回踩距离（3%）

# ==================== FVG参数 ====================
FVG_MIN_GAP_RATIO = 0.001    # FVG最小缺口比例（0.1%）
FVG_MAX_FILLED = 0.5         # FVG最大回填比例（50%）
FVG_DISTANCE = 0.03          # FVG距离阈值（5%）

# ==================== SMC均线参数 ====================
TREND_MA_LONG_PERIOD = 20   # 长期均线周期
TREND_MA_MID_PERIOD = 10    # 中期均线周期
TREND_MA_SHORT_PERIOD = 5   # 短期均线周期

# ==================== SMC趋势判断参数 ====================
TREND_MA_LONG_SLOPE_LOOKBACK = 2     # 长期均线斜率回看天数
TREND_MA_MID_SLOPE_LOOKBACK = 2      # 中期均线斜率回看天数
TREND_MA_SHORT_SLOPE_LOOKBACK = 2    # 短期均线斜率回看天数

# 趋势强度评分阈值
TREND_STRENGTH_STRONG = 60      # 强趋势阈值
TREND_STRENGTH_MODERATE = 45    # 中等趋势阈值
TREND_STRENGTH_WEAK = 30        # 弱趋势阈值

# ==================== SMC成交量参数 ====================
VOLUME_SURGE_RATIO = 1.2        # 成交量放大倍数
VOLUME_SHRINK_RATIO = 0.8       # 成交量萎缩倍数

# ==================== SMC ATR参数 ====================
ATR_PERIOD = 14              # ATR计算周期
ATR_STOP_MULTIPLIER = 1      # ATR止损倍数
ATR_TARGET_MULTIPLIER = 4    # ATR目标倍数

# ==================== SMC入场时机参数 ====================
ENTRY_MAX_DAILY_RETURN = 0.05    # 最大当日涨幅（5%）
ENTRY_MAX_CONSECUTIVE_RISES = 3  # 最大连续上涨天数
ENTRY_MAX_VOLUME_RATIO = 2.0     # 最大成交量倍数
ENTRY_MIN_DISTANCE_FROM_HIGH= 0.1
ENTRY_MAX_DISTANCE_FROM_SUPPORT = 0.05

# ==================== SMC止损和目标参数 ====================
MIN_RISK_REWARD_RATIO = 2    # 最小风险收益比
TARGET_PROFIT_PCT = 0.1      # 目标收益百分比（10%）
STOP_LOSS_MIN_PCT = 0.02     # 最小止损百分比（2%）
STOP_LOSS_MAX_PCT = 0.05     # 最大止损百分比（5%）

# 强趋势止损参数
STRONG_TREND_THRESHOLD = 60   # 强趋势阈值
STRONG_TREND_STOP_MIN = 0.92  # 强趋势最小止损（8%）
STRONG_TREND_STOP_MAX = 0.96  # 强趋势最大止损（4%）
WEAK_TREND_STOP_MIN = 0.96    # 弱趋势最小止损（4%）
WEAK_TREND_STOP_MAX = 0.98    # 弱趋势最大止损（2%）

# ==================== SMC置信度阈值 ====================
MIN_CONFIDENCE = 50         # 最低置信度要求
CONFIDENCE_STRONG_BUY = 70  # 强买入信号置信度
CONFIDENCE_BUY = 50         # 买入信号置信度
MIN_ENTRY_QUALITY = 70      # 最低入场质量要求
MIN_TREND_STRENGTH = 60     # 最低趋势强度要求

# ==================== SMC看空信号参数 ====================
BEARISH_MA_BREAK_THRESHOLD = 0.96  # 跌破均线阈值（4%）
BEARISH_MIN_CONFIDENCE = 85        # 看空信号最低置信度
BEARISH_VOLUME_SURGE = 1.5         # 看空成交量放大倍数

# 动态看空阈值参数（根据趋势强度调整）
BEARISH_DYNAMIC_THRESHOLD_ENABLED = False  # 启用动态阈值
BEARISH_STRONG_TREND_THRESHOLD = 85       # 强趋势看空阈值（更保守）
BEARISH_MODERATE_TREND_THRESHOLD = 75     # 中等趋势看空阈值
BEARISH_WEAK_TREND_THRESHOLD = 65         # 弱趋势看空阈值（更灵敏）

# 看空信号累积参数
BEARISH_SIGNAL_MEMORY_DAYS = 3       # 看空信号记忆天数（累积窗口）
BEARISH_MEMORY_DECAY = 0.7           # 历史信号衰减系数（每天衰减30%）

# 顶部预警参数
TOP_WARNING_MA = 5                       # 顶部预警均线周期
TOP_WARNING_PRICE_DEVIATION = 0.20       # 价格偏离均线的阈值（10%）
TOP_WARNING_RSI_THRESHOLD = 80           # RSI顶部预警阈值
TOP_WARNING_VOLUME_DECLINE = 0.6         # 成交量衰减阈值（相对前期）

# 均线斜率变化率参数（顶部反转检测）
MA_SLOPE_LOOKBACK_SHORT = 5      # 短期斜率计算周期（天）
MA_SLOPE_LOOKBACK_LONG = 5       # 长期斜率计算周期（天）
MA_SLOPE_CHANGE_EXTREME = 2.0    # 极端变化率阈值（200%）
MA_SLOPE_CHANGE_HIGH = 1.0       # 高变化率阈值（100%）
MA_SLOPE_DECELERATION = 0.5      # 减速阈值（上升动能衰竭）
MA_SLOPE_ACCELERATION = 1.5      # 加速阈值（下降加速）

# ==================== SMC RSI参数 ====================
RSI_PERIOD = 6  # RSI计算周期
RSI_OVERSOLD = 14  # RSI超卖阈值
RSI_OVERBOUGHT = 30  # RSI超买阈值
RSI_GOOD_RANGE_MIN = 40  # RSI良好区间下限
RSI_GOOD_RANGE_MAX = 65  # RSI良好区间上限
OVERBOUGHT = 85  # RSI极度超买阈值





# ==============================================================================
# 威科夫Spring策略参数
# ==============================================================================
WYCKOFF_MIN_DATA_DAYS = 60 # 最少需要的数据天数

# ==================== 威科夫均线参数 ====================
WYCKOFF_MA_SHORT_PERIOD = 10   # 短期均线周期
WYCKOFF_MA_MID_PERIOD = 30    # 中期均线周期
WYCKOFF_MA_LONG_PERIOD = 90   # 长期均线周期

# ==================== 威科夫均线趋势判断参数 ====================
WYCKOFF_MA_SLOPE_LOOKBACK = 3       # 均线斜率计算回看天数
WYCKOFF_MA_PRICE_TOLERANCE = 0.01    # 价格接近均线的容差（1%）
WYCKOFF_MA_SPREAD_MAX = 0.1         # 均线间距最大值（10%）
WYCKOFF_MA_MIN_STRENGTH = 60         # 均线向上最低强度要求

# ==================== 威科夫横盘区域参数 ====================
WYCKOFF_CONSOLIDATION_DAYS = 10         # 横盘区域检测天数
WYCKOFF_CONSOLIDATION_RANGE = 0.10      # 横盘区域价格波动范围（15%）

# ==================== 威科夫成交量参数 ====================
WYCKOFF_VOLUME_SHRINK_RATIO = 0.8     # 成交量萎缩比例
WYCKOFF_VOLUME_SURGE_RATIO = 1.5      # Spring形态成交量放大比例

# ==================== 威科夫Spring形态检测参数 ====================
WYCKOFF_SPRING_LOOKBACK_DAYS = 10        # Spring形态检测回看天数
WYCKOFF_SPRING_SUPPORT_LOOKBACK = 30     # 支撑位计算回看天数
WYCKOFF_SPRING_BREAK_THRESHOLD = 0.95    # 跌破支撑阈值（1%）

# ==================== 威科夫RSI参数 ====================
WYCKOFF_RSI_OVERSOLD_MIN = 10       # RSI超卖下限
WYCKOFF_RSI_OVERSOLD_MAX = 40       # RSI超卖上限

# ==================== 威科夫MACD背离检测参数 ====================
WYCKOFF_MACD_DIVERGENCE_LOOKBACK = 10   # MACD背离检测回看天数
WYCKOFF_MACD_DIVERGENCE_SHORT = 5      # MACD背离短期对比天数

# ==================== 威科夫置信度阈值 ====================
WYCKOFF_CONFIDENCE_BUY = 50         # 买入信号置信度阈值
WYCKOFF_CONFIDENCE_WATCH = 100      # 观察信号置信度阈值

# ==================== 威科夫底部区域判断参数 ====================
WYCKOFF_BOTTOM_SUPPORT_LOOKBACK = 20  # 底部支撑位回看天数
WYCKOFF_BOTTOM_SUPPORT_BUFFER = 0.05  # 支撑位缓冲（5%）

# ==================== 威科夫止损和目标参数 ====================
WYCKOFF_STOP_LOSS_BUFFER = 0.01  # 止损缓冲（1%）
WYCKOFF_TARGET_PROFIT = 0.3  # 目标收益（12%）

# ==================== 威科夫置信度权重 ====================
WYCKOFF_CONFIDENCE_CONSOLIDATION = 20  # 横盘区域确认权重
WYCKOFF_CONFIDENCE_VOLUME_SHRINK = 20  # 成交量萎缩权重
WYCKOFF_CONFIDENCE_SPRING = 25  # Spring形态权重
WYCKOFF_CONFIDENCE_RSI = 15  # RSI回升权重
WYCKOFF_CONFIDENCE_MACD = 20  # MACD背离权重

# ==================== 参数说明 ====================
"""
参数调整建议：

【全局卖出控制】
- ENABLE_TAKE_PROFIT_EXIT: 控制是否启用止盈卖出（默认True）
- ENABLE_BEARISH_SIGNAL_EXIT: 控制是否启用看空信号卖出（默认True）
- ENABLE_SUPPORT_BREAK_EXIT: 控制是否启用跌破支撑卖出（默认True）
- ENABLE_TIME_STOP_EXIT: 控制是否启用时间止损卖出（默认True）

【时间止损调整】
- TIME_STOP_DAYS: 持仓未盈利天数阈值（默认7天）
  - 更严格：减少天数（如改为5天）
  - 更宽松：增加天数（如改为10天）
- TIME_STOP_MIN_LOSS_PCT: 时间止损最小亏损比例（默认-2%）
  - 避免微小亏损就卖出，只有亏损超过此比例才触发时间止损

【看空信号灵敏度调整】
- 提高灵敏度：降低 BEARISH_STRONG_TREND_THRESHOLD（如改为80）
- 降低灵敏度：提高 BEARISH_STRONG_TREND_THRESHOLD（如改为90）

【信号累积窗口调整】
- 延长记忆：增加 BEARISH_SIGNAL_MEMORY_DAYS（如改为5）
- 缩短记忆：减少 BEARISH_SIGNAL_MEMORY_DAYS（如改为2）

【顶部预警调整】
- 更早预警：降低 TOP_WARNING_PRICE_DEVIATION（如改为0.08）
- 更晚预警：提高 TOP_WARNING_PRICE_DEVIATION（如改为0.12）

【均线斜率变化率调整（顶部反转检测）】
- 更灵敏检测：降低 MA_SLOPE_CHANGE_EXTREME（如改为1.5，即150%）
- 更保守检测：提高 MA_SLOPE_CHANGE_EXTREME（如改为2.5，即250%）
- 调整检测周期：修改 MA_SLOPE_LOOKBACK_SHORT 和 MA_SLOPE_LOOKBACK_LONG

说明：
- 斜率变化率 = (近期斜率 - 前期斜率) / |前期斜率|
- 极端变化率（200%+）：均线从快速上升突然转为下降，强烈顶部反转信号（+35分）
- 高变化率（100%+）：均线从上升转为下降，顶部反转信号（+25分）
- 加速下降：已在下降趋势中加速（+20分）
- 动能衰竭：上升速度减半，预警信号（+15分）

【止损调整】
- 更严格止损：降低 STRONG_TREND_STOP_MIN（如改为0.90，即10%止损）
- 更宽松止损：提高 STRONG_TREND_STOP_MIN（如改为0.94，即6%止损）

【趋势判断调整】
- 更严格趋势：提高 TREND_STRENGTH_STRONG（如改为75）
- 更宽松趋势：降低 TREND_STRENGTH_STRONG（如改为65）

【移除的冗余参数】
- WYCKOFF_MIN_DATA_DAYS: 数据天数检查已在代码中硬编码
- TREND_PRICE_VS_LONG_TOLERANCE: 功能与趋势强度判断重复
- ENTRY_MAX_DISTANCE_FROM_SUPPORT: 与订单块距离参数功能重复
- TARGET_RISK_MULTIPLIER: 与TARGET_PROFIT_PCT功能重复
"""
