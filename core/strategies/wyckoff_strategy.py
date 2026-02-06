# 威科夫Spring策略模块
# 识别底部区域的Spring形态，捕捉反转机会

import pandas as pd
import numpy as np
from datetime import datetime
from core.data import DataFetcher
from core.indicators import TechnicalIndicators
from core.analysis import PriceActionAnalyzer, CandlestickPatterns

# 从配置文件导入所有参数
from config.strategy_config import *

# ======================================================


class WyckoffStrategy:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.tech_indicators = TechnicalIndicators()
        self.price_action = PriceActionAnalyzer()
        self.candlestick_detector = CandlestickPatterns()  # 新增：K线形态检测器
        from  core.analysis.trend_line_analyzer import TrendLineAnalyzer
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
        short_data= daily_data.tail(min(self.trend_analyzer.short_period, len(daily_data)))
        short_uptrend=self.trend_analyzer._find_trendline_by_segment_lows(short_data, self.trend_analyzer.short_segments)

        if self.trend_analyzer._check_broken_support(daily_data, long_uptrend) or self.trend_analyzer._check_broken_support(daily_data, short_uptrend):
            return None
        if len(daily_data) < WYCKOFF_MIN_DATA_DAYS:
            return None



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
            'signals': conditions['details'],
            'conditions_met': conditions['details'],
            'analysis': {
                'price_action': pa_analysis,
                'indicators': indicators,
                'wyckoff_conditions': conditions,
            },
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


        # 6. 均线向上评分
        ma_analysis = self._check_moving_average_uptrend(data)
        ma_scores=ma_analysis.get('strength')
        if ma_analysis.get('is_uptrend'):
            confidence += 0.2 * ma_scores
        else:
            return {
                'signal': 'no_signal',
                'confidence': 0,
                'stop_loss': 0,
                'target': 0,
                'details': [],
            }

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
        signal = 'buy' if confidence >= WYCKOFF_CONFIDENCE_BUY and self.trend_analyzer else 'watch' if confidence >= WYCKOFF_CONFIDENCE_WATCH else 'no_signal'
        return {
            'signal': signal,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'target': target,
            'details': conditions_met,
        }


    
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
            strength += 30
        
        # 条件2: 均线斜率向上
        ma_short_slope = (ma_short.iloc[-1] - ma_short.iloc[-WYCKOFF_MA_SLOPE_LOOKBACK]) / ma_short.iloc[-WYCKOFF_MA_SLOPE_LOOKBACK] if len(ma_short) >= WYCKOFF_MA_SLOPE_LOOKBACK else 0
        ma_mid_slope = (ma_mid.iloc[-1] - ma_mid.iloc[-WYCKOFF_MA_SLOPE_LOOKBACK]) / ma_mid.iloc[-WYCKOFF_MA_SLOPE_LOOKBACK] if len(ma_mid) >= WYCKOFF_MA_SLOPE_LOOKBACK else 0
        
        if ma_short_slope > 0 and ma_mid_slope > 0:
            details.append("均线斜率向上")
            strength += 30
        elif ma_short_slope > 0:
            details.append(f"MA{WYCKOFF_MA_SHORT_PERIOD}向上")
            strength += 20
        
        # 条件3: 价格位置
        price_to_ma_pct = (current_price - current_ma_short) / current_ma_short
        
        if price_to_ma_pct >= 0:
            details.append(f"价格在MA{WYCKOFF_MA_SHORT_PERIOD}上方 (+{price_to_ma_pct*100:.1f}%)")
            strength += 20
        elif price_to_ma_pct >= -WYCKOFF_MA_PRICE_TOLERANCE:
            details.append(f"价格接近MA{WYCKOFF_MA_SHORT_PERIOD} ({price_to_ma_pct*100:.1f}%)")
            strength += 15

        
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
        
        # 详细背离检测逻辑
        if len(recent_macd) >= WYCKOFF_MACD_DIVERGENCE_SHORT:
            # 获取近期价格和MACD的局部低点
            price_lows = []
            macd_lows = []
            
            # 在指定周期内寻找局部低点
            for i in range(1, len(recent_prices) - 1):
                # 价格局部低点
                if (recent_prices.iloc[i] < recent_prices.iloc[i-1] and 
                    recent_prices.iloc[i] < recent_prices.iloc[i+1]):
                    price_lows.append((i, recent_prices.iloc[i]))
                
                # MACD局部低点
                if (recent_macd[i] < recent_macd[i-1] and 
                    recent_macd[i] < recent_macd[i+1]):
                    macd_lows.append((i, recent_macd[i]))
            
            # 检查是否存在背离：价格创新低但MACD没有创新低
            if len(price_lows) >= 2 and len(macd_lows) >= 1:
                # 最近的价格低点
                latest_price_low = price_lows[-1]
                previous_price_low = price_lows[-2] if len(price_lows) >= 2 else None
                
                # 最近的MACD低点
                latest_macd_low = macd_lows[-1]
                
                # 判断背离条件
                if (previous_price_low and 
                    latest_price_low[1] < previous_price_low[1] and  # 价格创新低
                    latest_macd_low[1] >= macd_lows[-2][1] if len(macd_lows) >= 2 else True):  # MACD未创新低
                    return True
        
        return False


    

    

    

    
