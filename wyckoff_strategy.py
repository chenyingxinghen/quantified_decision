# 威科夫Spring策略模块
# 识别底部区域的Spring形态，捕捉反转机会

import pandas as pd
import numpy as np
from datetime import datetime
from data_fetcher import DataFetcher
from technical_indicators import TechnicalIndicators
from price_action_analyzer import PriceActionAnalyzer
from candlestick_patterns import CandlestickPatterns

# 从配置文件导入所有参数
from strategy_config import *

# ======================================================


class WyckoffStrategy:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.tech_indicators = TechnicalIndicators()
        self.price_action = PriceActionAnalyzer()
        self.candlestick_detector = CandlestickPatterns()  # 新增：K线形态检测器
        from trend_line_analyzer import TrendLineAnalyzer
        self.trend_analyzer = TrendLineAnalyzer(
            long_period=TREND_LINE_LONG_PERIOD,
            short_period=TREND_LINE_SHORT_PERIOD
        )
    
    def wyckoff_accumulation_strategy(self, stock_code):
        """
        威科夫积累策略 - 识别Spring形态
        寻找底部区域的最后一次洗盘机会
        
        新增筛选条件：
        - 均线必须向上（MA20 > MA50 > MA60）
        - 价格在均线上方或接近均线

        
        返回格式与SMC策略一致，用于回测引擎
        """
        # 获取日线和周线数据
        daily_data = self.data_fetcher.get_stock_data(stock_code,1000)
        long_data = daily_data.tail(min(self.trend_analyzer.long_period, len(daily_data)))
        long_uptrend=self.trend_analyzer._find_trendline_by_segment_lows(long_data, self.trend_analyzer.long_segments)


        if self.trend_analyzer._check_broken_support(daily_data, long_uptrend):
            return None
        if len(daily_data) < WYCKOFF_MIN_DATA_DAYS:
            return None
        
        # 均线向上筛选（必须条件）
        ma_check = self._check_moving_average_uptrend(daily_data)
        if not ma_check['is_uptrend']:
            return None  # 均线不向上，直接过滤
        
        # 价格行为分析
        pa_analysis = self.price_action.get_comprehensive_analysis(daily_data)
        if not pa_analysis:
            return None
        
        # 技术指标
        indicators = self.tech_indicators.calculate_all_indicators(daily_data)
        if not indicators:
            return None
        
        # 威科夫积累条件检查
        conditions = self._check_wyckoff_accumulation(daily_data, pa_analysis, indicators)
        
        current_price = daily_data['close'].iloc[-1]
        
        # 计算风险收益比 - 使用价格变化率，支持负价格
        current_price_abs = abs(current_price)
        if current_price_abs > 0:
            risk_rate = abs(current_price - conditions['stop_loss']) / current_price_abs
            reward_rate = abs(conditions['target'] - current_price) / current_price_abs
            risk_reward_ratio = reward_rate / risk_rate if risk_rate > 0 else 0
        else:
            risk_reward_ratio = 0
        
        # 返回格式与SMC策略一致
        return {
            'stock_code': stock_code,
            'signal': conditions['signal'],
            'confidence': conditions['confidence'],
            'current_price': current_price,
            'entry_price': current_price,
            'stop_loss': conditions['stop_loss'],
            'target': conditions['target'],
            'risk_reward_ratio': risk_reward_ratio,
            'signals': conditions['details'] + [f"均线向上 (强度: {ma_check['strength']:.1f}%)"],
            'conditions_met': conditions['details'] + [f"均线向上 (强度: {ma_check['strength']:.1f}%)"],
            'analysis': {
                'price_action': pa_analysis,
                'indicators': indicators,
                'wyckoff_conditions': conditions,
                'ma_uptrend': ma_check
            },
            # 标记为底部策略，用于后续过滤看空信号
            'strategy_type': 'bottom_reversal',
            'ignore_bearish_signals': True  # 忽略顶部看空信号
        }
    
    def _check_wyckoff_accumulation(self, data, pa_analysis, indicators):
        """检查威科夫积累阶段条件"""
        current_price = data['close'].iloc[-1]
        recent_low = data['low'].tail(WYCKOFF_CONSOLIDATION_DAYS).min()
        recent_high = data['high'].tail(WYCKOFF_CONSOLIDATION_DAYS).max()
        
        conditions_met = []
        confidence = 0
        
        # 1. 价格在底部区域横盘
        if recent_low != 0:
            price_range = (recent_high - recent_low) / recent_low
        else:
            price_range = 0
        if price_range < WYCKOFF_CONSOLIDATION_RANGE:
            conditions_met.append("底部横盘区域确认")
            confidence += WYCKOFF_CONFIDENCE_CONSOLIDATION
        
        # 2. 成交量萎缩
        recent_volume = data['volume'].tail(10).mean()
        avg_volume = data['volume'].mean()
        if recent_volume < avg_volume * WYCKOFF_VOLUME_SHRINK_RATIO:
            conditions_met.append("成交量萎缩")
            confidence += WYCKOFF_CONFIDENCE_VOLUME_SHRINK
        
        # 3. Spring形态 - 快速跌破支撑后回升
        if self._detect_spring_pattern(data):
            conditions_met.append("Spring形态确认")
            confidence += WYCKOFF_CONFIDENCE_SPRING
        
        # 4. RSI超卖后回升
        if len(indicators['rsi']) > 0:
            current_rsi = indicators['rsi'][-1]
            if WYCKOFF_RSI_OVERSOLD_MIN < current_rsi < WYCKOFF_RSI_OVERSOLD_MAX:
                conditions_met.append("RSI从超卖回升")
                confidence += WYCKOFF_CONFIDENCE_RSI
        
        # 5. MACD底背离
        if self._detect_macd_bullish_divergence(data, indicators):
            conditions_met.append("MACD底背离")
            confidence += WYCKOFF_CONFIDENCE_MACD
        
        # 6. 判断是否在底部区域（用于忽略顶部看空信号）
        is_in_bottom = self._is_in_bottom_area(data)
        if is_in_bottom:
            conditions_met.append("处于底部区域，忽略顶部看空信号")
        
        # 计算止损和目标价（支持负价格）
        if current_price > 0:
            # 正价格：止损更低，目标更高
            stop_loss = recent_low * (1 - WYCKOFF_STOP_LOSS_BUFFER)
            target = current_price * (1 + WYCKOFF_TARGET_PROFIT)
        else:
            # 负价格：止损更负（数值更小），目标更接近0（数值更大）
            stop_loss = recent_low * (1 + WYCKOFF_STOP_LOSS_BUFFER)
            target = current_price * (1 - WYCKOFF_TARGET_PROFIT)
        
        # 生成信号
        signal = 'buy' if confidence >= WYCKOFF_CONFIDENCE_BUY else 'watch' if confidence >= WYCKOFF_CONFIDENCE_WATCH else 'no_signal'
        return {
            'signal': signal,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'target': target,
            'details': conditions_met,
            'is_in_bottom': is_in_bottom
        }

    def _is_in_bottom_area(self, data):
        """
        判断当前价格是否在底部区域

        底部区域定义：
        1. 价格没有跌破前期低点支撑（60天内的最低点）
        2. 价格在支撑位上方或接近支撑位（5%缓冲）

        返回：
            bool: True表示在底部区域，可以忽略顶部看空信号
        """
        if len(data) < WYCKOFF_BOTTOM_SUPPORT_LOOKBACK:
            return False

        current_price = data['close'].iloc[-1]

        # 计算前期支撑位（60天内的最低点）
        support_level = data['low'].tail(WYCKOFF_BOTTOM_SUPPORT_LOOKBACK).min()

        # 判断是否在支撑位上方（带5%缓冲）
        support_with_buffer = support_level * (1 - WYCKOFF_BOTTOM_SUPPORT_BUFFER)

        # 如果价格在支撑位上方，说明在底部区域
        is_above_support = current_price >= support_with_buffer

        return is_above_support
    
    def _check_moving_average_uptrend(self, data):
        """
        检查均线是否向上
        
        条件：
        1. MA5 > MA20 > MA60（多头排列）
        2. 均线斜率向上（最近N天均线上升）
        3. 价格在MA5上方或接近MA5
        
        返回：
            dict: {
                'is_uptrend': bool,
                'strength': float (0-100),
                'details': list
            }
        """
        if len(data) < WYCKOFF_MA_LONG_PERIOD:
            return {'is_uptrend': False, 'strength': 0, 'details': []}
        
        # 计算均线
        ma_short = data['close'].rolling(window=WYCKOFF_MA_SHORT_PERIOD).mean()
        ma_mid = data['close'].rolling(window=WYCKOFF_MA_MID_PERIOD).mean()
        ma_long = data['close'].rolling(window=WYCKOFF_MA_LONG_PERIOD).mean()
        
        current_price = data['close'].iloc[-1]
        current_ma_short = ma_short.iloc[-1]
        current_ma_mid = ma_mid.iloc[-1]
        current_ma_long = ma_long.iloc[-1]
        
        details = []
        strength = 0
        
        # 条件1: 多头排列
        if current_ma_short > current_ma_mid > current_ma_long:
            details.append(f"均线多头排列(MA{WYCKOFF_MA_SHORT_PERIOD}>MA{WYCKOFF_MA_MID_PERIOD}>MA{WYCKOFF_MA_LONG_PERIOD})")
            strength += 40
        
        # 条件2: 均线斜率向上
        ma_short_slope = (ma_short.iloc[-1] - ma_short.iloc[-WYCKOFF_MA_SLOPE_LOOKBACK]) / ma_short.iloc[-WYCKOFF_MA_SLOPE_LOOKBACK] if len(ma_short) >= WYCKOFF_MA_SLOPE_LOOKBACK else 0
        ma_mid_slope = (ma_mid.iloc[-1] - ma_mid.iloc[-WYCKOFF_MA_SLOPE_LOOKBACK]) / ma_mid.iloc[-WYCKOFF_MA_SLOPE_LOOKBACK] if len(ma_mid) >= WYCKOFF_MA_SLOPE_LOOKBACK else 0
        
        if ma_short_slope > 0 and ma_mid_slope > 0:
            details.append("均线斜率向上")
            strength += 30
        elif ma_short_slope > 0:
            details.append(f"MA{WYCKOFF_MA_SHORT_PERIOD}向上")
            strength += 15
        
        # 条件3: 价格位置
        price_to_ma_pct = (current_price - current_ma_short) / current_ma_short
        
        if price_to_ma_pct >= 0:
            details.append(f"价格在MA{WYCKOFF_MA_SHORT_PERIOD}上方 (+{price_to_ma_pct*100:.1f}%)")
            strength += 20
        elif price_to_ma_pct >= -WYCKOFF_MA_PRICE_TOLERANCE:
            details.append(f"价格接近MA{WYCKOFF_MA_SHORT_PERIOD} ({price_to_ma_pct*100:.1f}%)")
            strength += 10
        
        # 条件4: 均线间距合理
        ma_spread = (current_ma_short - current_ma_long) / current_ma_long
        if 0 < ma_spread < WYCKOFF_MA_SPREAD_MAX:
            details.append("均线间距合理")
            strength += 10
        
        # 判断是否为上升趋势
        is_uptrend = strength >= WYCKOFF_MA_MIN_STRENGTH
        
        return {
            'is_uptrend': is_uptrend,
            'strength': strength,
            'details': details
        }
        
    
    def _detect_spring_pattern(self, data):
        """检测Spring形态"""
        if len(data) < WYCKOFF_SPRING_LOOKBACK_DAYS:
            return False
        
        recent_data = data.tail(WYCKOFF_SPRING_LOOKBACK_DAYS)
        support_level = data.tail(WYCKOFF_SPRING_SUPPORT_LOOKBACK)['low'].min()
        
        # 寻找跌破支撑后快速回升的模式
        for i in range(len(recent_data) - 2):
            if (recent_data.iloc[i]['low'] < support_level * WYCKOFF_SPRING_BREAK_THRESHOLD and  # 跌破支撑
                recent_data.iloc[i+1]['close'] > recent_data.iloc[i]['close'] and  # 次日回升
                recent_data.iloc[i]['volume'] > recent_data['volume'].mean() * WYCKOFF_VOLUME_SURGE_RATIO):  # 放量
                return True
        
        return False
    
    def _detect_macd_bullish_divergence(self, data, indicators):
        """检测MACD底背离"""
        if len(indicators['macd']) < WYCKOFF_MACD_DIVERGENCE_LOOKBACK:
            return False
        
        # 寻找价格创新低但MACD不创新低的情况
        recent_prices = data['close'].tail(WYCKOFF_MACD_DIVERGENCE_LOOKBACK)
        recent_macd = indicators['macd'][-WYCKOFF_MACD_DIVERGENCE_LOOKBACK:]
        
        # 简化的背离检测
        if len(recent_macd) >= WYCKOFF_MACD_DIVERGENCE_SHORT:
            return recent_macd[-1] > recent_macd[-WYCKOFF_MACD_DIVERGENCE_SHORT] and recent_prices.iloc[-1] <= recent_prices.iloc[-WYCKOFF_MACD_DIVERGENCE_SHORT]
        
        return False
    

    

    

    

    
