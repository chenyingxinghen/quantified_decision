# 威科夫Spring策略模块
# 识别底部区域的Spring形态，捕捉反转机会

import pandas as pd
import numpy as np
from datetime import datetime
from data_fetcher import DataFetcher
from technical_indicators import TechnicalIndicators
from price_action_analyzer import PriceActionAnalyzer


# ==================== 策略参数配置 ====================
# 可根据回测结果调整这些参数

# 数据周期参数
MIN_DATA_DAYS = 60  # 最少需要的数据天数

# 均线参数
MA_SHORT_PERIOD = 5   # 短期均线周期
MA_MID_PERIOD = 20    # 中期均线周期
MA_LONG_PERIOD = 60   # 长期均线周期

# 均线趋势判断参数
MA_SLOPE_LOOKBACK = 6  # 均线斜率计算回看天数
MA_PRICE_TOLERANCE = 0.02  # 价格接近均线的容差（2%）
MA_SPREAD_MAX = 0.15  # 均线间距最大值（15%）
MA_MIN_STRENGTH = 50  # 均线向上最低强度要求

# 横盘区域参数
CONSOLIDATION_DAYS = 20  # 横盘区域检测天数
CONSOLIDATION_RANGE = 0.15  # 横盘区域价格波动范围（15%）

# 成交量参数
VOLUME_SHRINK_RATIO = 0.8  # 成交量萎缩比例
VOLUME_SURGE_RATIO = 1.5  # Spring形态成交量放大比例

# Spring形态检测参数
SPRING_LOOKBACK_DAYS = 10  # Spring形态检测回看天数
SPRING_SUPPORT_LOOKBACK = 30  # 支撑位计算回看天数
SPRING_BREAK_THRESHOLD = 0.99  # 跌破支撑阈值（1%）

# RSI参数
RSI_OVERSOLD_MIN = 30  # RSI超卖下限
RSI_OVERSOLD_MAX = 50  # RSI超卖上限

# MACD背离检测参数
MACD_DIVERGENCE_LOOKBACK = 20  # MACD背离检测回看天数
MACD_DIVERGENCE_SHORT = 10  # MACD背离短期对比天数

# 置信度阈值
CONFIDENCE_BUY = 60  # 买入信号置信度阈值
CONFIDENCE_WATCH = 40  # 观察信号置信度阈值

# 止损和目标参数
STOP_LOSS_BUFFER = 0.015  # 止损缓冲（1.5%）
TARGET_PROFIT = 0.12  # 目标收益（12%）

# 置信度权重
CONFIDENCE_CONSOLIDATION = 20  # 横盘区域确认权重
CONFIDENCE_VOLUME_SHRINK = 15  # 成交量萎缩权重
CONFIDENCE_SPRING = 25  # Spring形态权重
CONFIDENCE_RSI = 15  # RSI回升权重
CONFIDENCE_MACD = 25  # MACD背离权重

# ======================================================


class WyckoffStrategy:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.tech_indicators = TechnicalIndicators()
        self.price_action = PriceActionAnalyzer()
    
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
        daily_data = self.data_fetcher.get_stock_data(stock_code)
        
        if len(daily_data) < MIN_DATA_DAYS:
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
        
        # 计算风险收益比
        risk = current_price - conditions['stop_loss']
        reward = conditions['target'] - current_price
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
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
            }
        }
    
    def _check_wyckoff_accumulation(self, data, pa_analysis, indicators):
        """检查威科夫积累阶段条件"""
        current_price = data['close'].iloc[-1]
        recent_low = data['low'].tail(CONSOLIDATION_DAYS).min()
        recent_high = data['high'].tail(CONSOLIDATION_DAYS).max()
        
        conditions_met = []
        confidence = 0
        
        # 1. 价格在底部区域横盘
        price_range = (recent_high - recent_low) / recent_low
        if price_range < CONSOLIDATION_RANGE:
            conditions_met.append("底部横盘区域确认")
            confidence += CONFIDENCE_CONSOLIDATION
        
        # 2. 成交量萎缩
        recent_volume = data['volume'].tail(10).mean()
        avg_volume = data['volume'].mean()
        if recent_volume < avg_volume * VOLUME_SHRINK_RATIO:
            conditions_met.append("成交量萎缩")
            confidence += CONFIDENCE_VOLUME_SHRINK
        
        # 3. Spring形态 - 快速跌破支撑后回升
        if self._detect_spring_pattern(data):
            conditions_met.append("Spring形态确认")
            confidence += CONFIDENCE_SPRING
        
        # 4. RSI超卖后回升
        if len(indicators['rsi']) > 0:
            current_rsi = indicators['rsi'][-1]
            if RSI_OVERSOLD_MIN < current_rsi < RSI_OVERSOLD_MAX:
                conditions_met.append("RSI从超卖回升")
                confidence += CONFIDENCE_RSI
        
        # 5. MACD底背离
        if self._detect_macd_bullish_divergence(data, indicators):
            conditions_met.append("MACD底背离")
            confidence += CONFIDENCE_MACD
        
        # 生成信号
        signal = 'buy' if confidence >= CONFIDENCE_BUY else 'watch' if confidence >= CONFIDENCE_WATCH else 'no_signal'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'stop_loss': recent_low * (1 - STOP_LOSS_BUFFER),
            'target': current_price * (1 + TARGET_PROFIT),
            'details': conditions_met
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
        if len(data) < MA_LONG_PERIOD:
            return {'is_uptrend': False, 'strength': 0, 'details': []}
        
        # 计算均线
        ma_short = data['close'].rolling(window=MA_SHORT_PERIOD).mean()
        ma_mid = data['close'].rolling(window=MA_MID_PERIOD).mean()
        ma_long = data['close'].rolling(window=MA_LONG_PERIOD).mean()
        
        current_price = data['close'].iloc[-1]
        current_ma_short = ma_short.iloc[-1]
        current_ma_mid = ma_mid.iloc[-1]
        current_ma_long = ma_long.iloc[-1]
        
        details = []
        strength = 0
        
        # 条件1: 多头排列
        if current_ma_short > current_ma_mid > current_ma_long:
            details.append(f"均线多头排列(MA{MA_SHORT_PERIOD}>MA{MA_MID_PERIOD}>MA{MA_LONG_PERIOD})")
            strength += 40
        
        # 条件2: 均线斜率向上
        ma_short_slope = (ma_short.iloc[-1] - ma_short.iloc[-MA_SLOPE_LOOKBACK]) / ma_short.iloc[-MA_SLOPE_LOOKBACK] if len(ma_short) >= MA_SLOPE_LOOKBACK else 0
        ma_mid_slope = (ma_mid.iloc[-1] - ma_mid.iloc[-MA_SLOPE_LOOKBACK]) / ma_mid.iloc[-MA_SLOPE_LOOKBACK] if len(ma_mid) >= MA_SLOPE_LOOKBACK else 0
        
        if ma_short_slope > 0 and ma_mid_slope > 0:
            details.append("均线斜率向上")
            strength += 30
        elif ma_short_slope > 0:
            details.append(f"MA{MA_SHORT_PERIOD}向上")
            strength += 15
        
        # 条件3: 价格位置
        price_to_ma_pct = (current_price - current_ma_short) / current_ma_short
        
        if price_to_ma_pct >= 0:
            details.append(f"价格在MA{MA_SHORT_PERIOD}上方 (+{price_to_ma_pct*100:.1f}%)")
            strength += 20
        elif price_to_ma_pct >= -MA_PRICE_TOLERANCE:
            details.append(f"价格接近MA{MA_SHORT_PERIOD} ({price_to_ma_pct*100:.1f}%)")
            strength += 10
        
        # 条件4: 均线间距合理
        ma_spread = (current_ma_short - current_ma_long) / current_ma_long
        if 0 < ma_spread < MA_SPREAD_MAX:
            details.append("均线间距合理")
            strength += 10
        
        # 判断是否为上升趋势
        is_uptrend = strength >= MA_MIN_STRENGTH
        
        return {
            'is_uptrend': is_uptrend,
            'strength': strength,
            'details': details
        }
        
    
    def _detect_spring_pattern(self, data):
        """检测Spring形态"""
        if len(data) < SPRING_LOOKBACK_DAYS:
            return False
        
        recent_data = data.tail(SPRING_LOOKBACK_DAYS)
        support_level = data.tail(SPRING_SUPPORT_LOOKBACK)['low'].min()
        
        # 寻找跌破支撑后快速回升的模式
        for i in range(len(recent_data) - 2):
            if (recent_data.iloc[i]['low'] < support_level * SPRING_BREAK_THRESHOLD and  # 跌破支撑
                recent_data.iloc[i+1]['close'] > recent_data.iloc[i]['close'] and  # 次日回升
                recent_data.iloc[i]['volume'] > recent_data['volume'].mean() * VOLUME_SURGE_RATIO):  # 放量
                return True
        
        return False
    
    def _detect_macd_bullish_divergence(self, data, indicators):
        """检测MACD底背离"""
        if len(indicators['macd']) < MACD_DIVERGENCE_LOOKBACK:
            return False
        
        # 寻找价格创新低但MACD不创新低的情况
        recent_prices = data['close'].tail(MACD_DIVERGENCE_LOOKBACK)
        recent_macd = indicators['macd'][-MACD_DIVERGENCE_LOOKBACK:]
        
        # 简化的背离检测
        if len(recent_macd) >= MACD_DIVERGENCE_SHORT:
            return recent_macd[-1] > recent_macd[-MACD_DIVERGENCE_SHORT] and recent_prices.iloc[-1] <= recent_prices.iloc[-MACD_DIVERGENCE_SHORT]
        
        return False
    

    

    

    

    
