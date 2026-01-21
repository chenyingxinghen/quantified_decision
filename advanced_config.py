# 高级策略配置文件
from datetime import time

# 威科夫理论参数
WYCKOFF_PARAMS = {
    'accumulation_range_threshold': 0.15,  # 积累区间波动阈值15%
    'spring_breakout_threshold': 0.90,     # Spring形态跌破阈值
    'volume_surge_multiplier': 1.3,        # 成交量放大倍数
    'rsi_recovery_range': (30, 50),        # RSI恢复区间
    'stop_loss_buffer': 0.02,              # 止损缓冲2%
    'profit_target_ratio': 1.15            # 盈利目标15%
}

# 黄金交叉动量参数
GOLDEN_CROSS_PARAMS = {
    'ema_short_period': 10,                # 短期EMA周期
    'sma_long_period': 50,                 # 长期SMA周期
    'volume_confirmation_ratio': 1.3,      # 成交量确认倍数
    'rsi_overbought_threshold': 70,        # RSI超买阈值
    'stop_loss_buffer': 0.03,              # 止损缓冲3%
    'profit_target_ratio': 1.20,           # 盈利目标20%
    'trend_strength_threshold': 70         # 趋势强度阈值
}

# 均值回归参数
MEAN_REVERSION_PARAMS = {
    'rsi_oversold_threshold': 30,          # RSI超卖阈值
    'kdj_oversold_threshold': 20,          # KDJ超卖阈值
    'bb_lower_buffer': 1.02,               # 布林带下轨缓冲
    'ma_deviation_threshold': -0.10,       # 均线偏离阈值-10%
    'volume_shrink_ratio': 0.7,            # 成交量萎缩比例
    'stop_loss_buffer': 0.05,              # 止损缓冲5%
    'profit_target_ratio': 1.10            # 盈利目标10%
}

# 突破动量参数
BREAKOUT_PARAMS = {
    'resistance_breakout_buffer': 1.005,   # 阻力突破缓冲0.5%
    'volume_surge_multiplier': 2.0,        # 成交量放大倍数
    'rsi_strength_range': (50, 80),        # RSI强势区间
    'risk_reward_ratio': 2.0,              # 风险收益比1:2
    'stop_loss_buffer': 0.05,              # 止损缓冲5%
    'lookback_period': 30                  # 回看周期
}

# 多周期分析参数
MULTI_TIMEFRAME_PARAMS = {
    'long_term_weight': 0.3,               # 长期趋势权重
    'medium_term_weight': 0.5,             # 中期趋势权重
    'short_term_weight': 0.2,              # 短期趋势权重
    'confluence_threshold': 75,            # 共振阈值
    'key_level_tolerance': 0.02,           # 关键位置容差2%
    'stop_loss_buffer': 0.04,              # 止损缓冲4%
    'profit_target_ratio': 1.18            # 盈利目标18%
}

# K线形态可靠性评分
CANDLESTICK_RELIABILITY = {
    'three_black_crows': 78,
    'abandoned_baby': 78,
    'hammer': 75,
    'three_outside_up': 70,
    'shooting_star': 70,
    'evening_star': 68,
    'morning_star': 68,
    'bullish_engulfing': 78,
    'bearish_engulfing': 78,
    'bullish_harami': 54,
    'doji': 60
}

# 市场结构识别参数
MARKET_STRUCTURE_PARAMS = {
    'swing_point_lookback': 5,             # 摆动点回看周期
    'min_bars_for_analysis': 300,           # 最小分析K线数量
    'trend_strength_periods': [5, 10, 20], # 趋势强度计算周期
    'structure_shift_threshold': 0.02,      # 结构转换阈值
    'consolidation_range_threshold': 0.05   # 盘整区间阈值
}

# ATR动态止损参数
ATR_STOP_LOSS_PARAMS = {
    'atr_period': 14,                      # ATR计算周期
    'atr_multiplier_conservative': 1.5,    # 保守止损倍数
    'atr_multiplier_aggressive': 2.0,      # 激进止损倍数
    'atr_multiplier_breakout': 2.5,        # 突破策略止损倍数
    'min_stop_loss_percent': 0.02,         # 最小止损百分比2%
    'max_stop_loss_percent': 0.08          # 最大止损百分比8%
}

# 仓位管理参数
POSITION_MANAGEMENT = {
    'max_single_position_risk': 0.02,      # 单笔最大风险2%
    'max_total_risk': 0.06,                # 总风险6%
    'max_correlation_positions': 3,        # 最大相关性持仓数
    'position_sizing_method': 'fixed_risk', # 仓位计算方法
    'leverage_factor': 1.0                 # 杠杆因子
}

# 市场环境过滤器
MARKET_FILTER_PARAMS = {
    'market_volatility_threshold': 0.25,   # 市场波动率阈值
    'index_trend_filter': True,            # 指数趋势过滤
    'sector_rotation_filter': True,        # 板块轮动过滤
    'volume_profile_filter': True,         # 成交量分布过滤
    'news_sentiment_weight': 0.1           # 新闻情绪权重
}

# 回测参数
BACKTEST_PARAMS = {
    'initial_capital': 100000,             # 初始资金
    'commission_rate': 0.0003,             # 手续费率
    'slippage_rate': 0.001,                # 滑点率
    'benchmark_symbol': '000001.SH',       # 基准指数
    'rebalance_frequency': 'weekly',       # 再平衡频率
    'max_drawdown_threshold': 0.15         # 最大回撤阈值
}

# 交易时间配置
TRADING_SESSIONS = {
    'morning_session': {
        'start': time(9, 30),
        'end': time(11, 30)
    },
    'afternoon_session': {
        'start': time(13, 0),
        'end': time(15, 0)
    },
    'pre_market_analysis': time(9, 0),     # 盘前分析时间
    'post_market_analysis': time(15, 30)   # 盘后分析时间
}

# 数据质量控制
DATA_QUALITY_PARAMS = {
    'min_trading_days': 60,                # 最少交易天数
    'max_missing_data_ratio': 0.05,        # 最大数据缺失比例
    'price_change_outlier_threshold': 0.15, # 价格异常变动阈值
    'volume_outlier_threshold': 5.0,       # 成交量异常倍数
    'data_freshness_hours': 2              # 数据新鲜度要求(小时)
}

# 风险控制参数
RISK_CONTROL_PARAMS = {
    'max_consecutive_losses': 3,           # 最大连续亏损次数
    'daily_loss_limit': 0.03,              # 日亏损限制3%
    'weekly_loss_limit': 0.08,             # 周亏损限制8%
    'monthly_loss_limit': 0.15,            # 月亏损限制15%
    'correlation_limit': 0.7,              # 相关性限制
    'sector_concentration_limit': 0.3       # 行业集中度限制30%
}

# 策略权重配置
STRATEGY_WEIGHTS = {
    'wyckoff_accumulation': 0.25,          # 威科夫积累策略权重
    'golden_cross_momentum': 0.25,         # 黄金交叉动量策略权重
    'mean_reversion_oversold': 0.20,       # 均值回归策略权重
    'breakout_momentum': 0.20,             # 突破动量策略权重
    'multi_timeframe_confluence': 0.10     # 多周期共振策略权重
}

# 信号过滤参数
SIGNAL_FILTER_PARAMS = {
    'min_confidence_threshold': 60,        # 最小置信度阈值
    'signal_confirmation_bars': 2,         # 信号确认K线数
    'false_signal_penalty': 0.1,           # 虚假信号惩罚
    'signal_decay_rate': 0.05,             # 信号衰减率
    'max_signals_per_day': 5               # 每日最大信号数
}

# 性能监控参数
PERFORMANCE_PARAMS = {
    'benchmark_symbols': ['000001.SH', '399001.SZ', '399006.SZ'],  # 基准指数
    'performance_metrics': [
        'total_return', 'annual_return', 'sharpe_ratio', 
        'max_drawdown', 'win_rate', 'profit_factor'
    ],
    'rolling_window_days': [30, 90, 252],  # 滚动窗口天数
    'report_frequency': 'weekly'           # 报告频率
}