# 价格行为分析模块 - 基于K线结构的趋势识别
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PriceActionAnalyzer:
    def __init__(self):
        self.min_bars_for_analysis = 50
    
    def identify_market_structure(self, data):
        """
        识别市场结构：HH-HL-LH-LL 四元素结构法则
        返回趋势类型和关键点位
        """
        if len(data) < self.min_bars_for_analysis:
            return None
        
        # 计算高低点
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        # 寻找局部高点和低点
        swing_highs = self._find_swing_points(highs, 'high')
        swing_lows = self._find_swing_points(lows, 'low')
        
        # 分析市场结构
        structure = self._analyze_structure(swing_highs, swing_lows, data)
        
        return structure
    
    def _find_swing_points(self, prices, point_type='high', lookback=5):
        """寻找摆动高点和低点"""
        swing_points = []
        
        for i in range(lookback, len(prices) - lookback):
            if point_type == 'high':
                # 寻找局部高点
                if all(prices[i] >= prices[j] for j in range(i-lookback, i+lookback+1) if j != i):
                    swing_points.append({'index': i, 'price': prices[i], 'type': 'high'})
            else:
                # 寻找局部低点
                if all(prices[i] <= prices[j] for j in range(i-lookback, i+lookback+1) if j != i):
                    swing_points.append({'index': i, 'price': prices[i], 'type': 'low'})
        
        return swing_points
    
    def _analyze_structure(self, swing_highs, swing_lows, data):
        """分析市场结构模式"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'trend': 'insufficient_data', 'strength': 0}
        
        # 获取最近的高低点
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
        
        # 分析趋势模式
        trend_analysis = self._determine_trend_pattern(recent_highs, recent_lows)
        
        # 计算趋势强度
        strength = self._calculate_trend_strength(data, recent_highs, recent_lows)
        
        return {
            'trend': trend_analysis['direction'],
            'strength': strength,
            'pattern': trend_analysis['pattern'],
            'key_levels': {
                'resistance': recent_highs[-1]['price'] if recent_highs else None,
                'support': recent_lows[-1]['price'] if recent_lows else None
            },
            'structure_shift': trend_analysis.get('structure_shift', False)
        }
    
    def _determine_trend_pattern(self, highs, lows):
        """确定趋势模式"""
        if len(highs) < 2 or len(lows) < 2:
            return {'direction': 'neutral', 'pattern': 'insufficient_data'}
        
        # 检查高点序列
        hh_pattern = len(highs) >= 2 and highs[-1]['price'] > highs[-2]['price']
        lh_pattern = len(highs) >= 2 and highs[-1]['price'] < highs[-2]['price']
        
        # 检查低点序列
        hl_pattern = len(lows) >= 2 and lows[-1]['price'] > lows[-2]['price']
        ll_pattern = len(lows) >= 2 and lows[-1]['price'] < lows[-2]['price']
        
        # 判断趋势
        if hh_pattern and hl_pattern:
            return {
                'direction': 'uptrend',
                'pattern': 'HH_HL',
                'structure_shift': False
            }
        elif lh_pattern and ll_pattern:
            return {
                'direction': 'downtrend', 
                'pattern': 'LH_LL',
                'structure_shift': False
            }
        elif (hh_pattern and ll_pattern) or (lh_pattern and hl_pattern):
            return {
                'direction': 'neutral',
                'pattern': 'mixed_signals',
                'structure_shift': True
            }
        else:
            return {
                'direction': 'consolidation',
                'pattern': 'sideways',
                'structure_shift': False
            }
    
    def _calculate_trend_strength(self, data, highs, lows):
        """计算趋势强度 (0-100)"""
        if len(data) < 20:
            return 0
        
        # 基于价格动量
        price_momentum = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20] * 100
        
        # 基于成交量确认
        recent_volume = data['volume'].tail(10).mean()
        avg_volume = data['volume'].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # 基于波动率
        atr = self._calculate_atr(data, 14)
        volatility_factor = min(atr[-1] / data['close'].iloc[-1] * 100, 10) if len(atr) > 0 else 5
        
        # 综合强度计算
        strength = abs(price_momentum) * volume_ratio * (volatility_factor / 5)
        return min(max(strength, 0), 100)
    
    def identify_candlestick_patterns(self, data):
        """识别关键蜡烛图形态"""
        if len(data) < 5:
            return []
        
        patterns = []
        
        # 获取最近几根K线
        recent_data = data.tail(5)
        
        for pattern_func in [
            self._detect_hammer,
            self._detect_shooting_star,
            self._detect_engulfing,
            self._detect_doji,
            self._detect_three_crows,
            self._detect_morning_evening_star
        ]:
            pattern = pattern_func(recent_data)
            if pattern:
                patterns.append(pattern)
        
        return patterns
    
    def _detect_hammer(self, data):
        """检测锤头线形态"""
        if len(data) < 1:
            return None
        
        last = data.iloc[-1]
        
        # 计算实体和影线
        body = abs(last['close'] - last['open'])
        lower_shadow = min(last['open'], last['close']) - last['low']
        upper_shadow = last['high'] - max(last['open'], last['close'])
        
        # 锤头线条件：下影线至少是实体的2倍，上影线很小
        if (lower_shadow >= 2 * body and 
            upper_shadow <= 0.1 * body and 
            body > 0):
            
            return {
                'pattern': 'hammer',
                'type': 'bullish_reversal',
                'reliability': 75,
                'position': len(data) - 1,
                'description': '锤头线 - 底部反转信号'
            }
        return None
    
    def _detect_shooting_star(self, data):
        """检测射击之星形态"""
        if len(data) < 1:
            return None
        
        last = data.iloc[-1]
        
        body = abs(last['close'] - last['open'])
        lower_shadow = min(last['open'], last['close']) - last['low']
        upper_shadow = last['high'] - max(last['open'], last['close'])
        
        # 射击之星条件：上影线至少是实体的2倍，下影线很小
        if (upper_shadow >= 2 * body and 
            lower_shadow <= 0.1 * body and 
            body > 0):
            
            return {
                'pattern': 'shooting_star',
                'type': 'bearish_reversal',
                'reliability': 70,
                'position': len(data) - 1,
                'description': '射击之星 - 顶部反转信号'
            }
        return None
    
    def _detect_engulfing(self, data):
        """检测吞没形态"""
        if len(data) < 2:
            return None
        
        prev = data.iloc[-2]
        curr = data.iloc[-1]
        
        prev_body = abs(prev['close'] - prev['open'])
        curr_body = abs(curr['close'] - curr['open'])
        
        # 看涨吞没
        if (prev['close'] < prev['open'] and  # 前一根是阴线
            curr['close'] > curr['open'] and  # 当前是阳线
            curr['open'] < prev['close'] and  # 当前开盘低于前一收盘
            curr['close'] > prev['open'] and  # 当前收盘高于前一开盘
            curr_body > prev_body):           # 当前实体更大
            
            return {
                'pattern': 'bullish_engulfing',
                'type': 'bullish_reversal',
                'reliability': 78,
                'position': len(data) - 1,
                'description': '看涨吞没 - 强烈反转信号'
            }
        
        # 看跌吞没
        elif (prev['close'] > prev['open'] and  # 前一根是阳线
              curr['close'] < curr['open'] and  # 当前是阴线
              curr['open'] > prev['close'] and  # 当前开盘高于前一收盘
              curr['close'] < prev['open'] and  # 当前收盘低于前一开盘
              curr_body > prev_body):           # 当前实体更大
            
            return {
                'pattern': 'bearish_engulfing',
                'type': 'bearish_reversal',
                'reliability': 78,
                'position': len(data) - 1,
                'description': '看跌吞没 - 强烈反转信号'
            }
        
        return None
    
    def _detect_doji(self, data):
        """检测十字星形态"""
        if len(data) < 1:
            return None
        
        last = data.iloc[-1]
        
        body = abs(last['close'] - last['open'])
        total_range = last['high'] - last['low']
        
        # 十字星条件：实体很小（小于总区间的5%）
        if body <= 0.05 * total_range and total_range > 0:
            return {
                'pattern': 'doji',
                'type': 'indecision',
                'reliability': 60,
                'position': len(data) - 1,
                'description': '十字星 - 市场犹豫信号'
            }
        return None
    
    def _detect_three_crows(self, data):
        """检测三只乌鸦形态"""
        if len(data) < 3:
            return None
        
        last_three = data.tail(3)
        
        # 检查是否都是阴线且逐渐走低
        all_bearish = all(row['close'] < row['open'] for _, row in last_three.iterrows())
        descending = (last_three.iloc[0]['close'] > last_three.iloc[1]['close'] > 
                     last_three.iloc[2]['close'])
        
        if all_bearish and descending:
            return {
                'pattern': 'three_black_crows',
                'type': 'bearish_reversal',
                'reliability': 78,
                'position': len(data) - 1,
                'description': '三只乌鸦 - 强烈看跌信号'
            }
        return None
    
    def _detect_morning_evening_star(self, data):
        """检测晨星/暮星形态"""
        if len(data) < 3:
            return None
        
        last_three = data.tail(3)
        first, second, third = last_three.iloc[0], last_three.iloc[1], last_three.iloc[2]
        
        # 晨星形态
        if (first['close'] < first['open'] and  # 第一根阴线
            abs(second['close'] - second['open']) < 0.3 * abs(first['close'] - first['open']) and  # 第二根小实体
            third['close'] > third['open'] and  # 第三根阳线
            third['close'] > (first['open'] + first['close']) / 2):  # 第三根深入第一根
            
            return {
                'pattern': 'morning_star',
                'type': 'bullish_reversal',
                'reliability': 68,
                'position': len(data) - 1,
                'description': '晨星 - 底部反转信号'
            }
        
        # 暮星形态
        elif (first['close'] > first['open'] and  # 第一根阳线
              abs(second['close'] - second['open']) < 0.3 * abs(first['close'] - first['open']) and  # 第二根小实体
              third['close'] < third['open'] and  # 第三根阴线
              third['close'] < (first['open'] + first['close']) / 2):  # 第三根深入第一根
            
            return {
                'pattern': 'evening_star',
                'type': 'bearish_reversal',
                'reliability': 68,
                'position': len(data) - 1,
                'description': '暮星 - 顶部反转信号'
            }
        
        return None
    
    def _calculate_atr(self, data, period=14):
        """计算平均真实波幅"""
        if len(data) < period + 1:
            return []
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 计算真实波幅
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算ATR
        atr = tr.rolling(window=period).mean()
        
        return atr.dropna().values
    
    def detect_breakout_patterns(self, data):
        """检测突破形态"""
        if len(data) < 20:
            return None
        
        # 计算支撑阻力位
        recent_data = data.tail(20)
        resistance = recent_data['high'].max()
        support = recent_data['low'].min()
        
        current_price = data['close'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].tail(10).mean()
        
        # 检测突破
        breakout_info = {
            'type': None,
            'strength': 0,
            'volume_confirmation': False
        }
        
        # 向上突破
        if current_price > resistance * 1.001:  # 0.1%的缓冲
            breakout_info['type'] = 'upward_breakout'
            breakout_info['strength'] = (current_price - resistance) / resistance * 100
            breakout_info['volume_confirmation'] = current_volume > avg_volume * 1.5
        
        # 向下突破
        elif current_price < support * 0.999:  # 0.1%的缓冲
            breakout_info['type'] = 'downward_breakout'
            breakout_info['strength'] = (support - current_price) / support * 100
            breakout_info['volume_confirmation'] = current_volume > avg_volume * 1.5
        
        return breakout_info if breakout_info['type'] else None
    
    def analyze_six_bar_drift(self, data):
        """六根K线漂移程度分析"""
        if len(data) < 6:
            return None
        
        last_six = data.tail(6)
        
        # 分析趋势一致性
        closes = last_six['close'].values
        trend_score = 0
        
        # 计算连续性得分
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                trend_score += 1
            elif closes[i] < closes[i-1]:
                trend_score -= 1
        
        # 计算波动率
        volatility = last_six['close'].std() / last_six['close'].mean()
        
        # 判断趋势质量
        if trend_score >= 4:
            trend_quality = 'strong_uptrend'
        elif trend_score <= -4:
            trend_quality = 'strong_downtrend'
        elif abs(trend_score) <= 1:
            trend_quality = 'consolidation'
        else:
            trend_quality = 'weak_trend'
        
        return {
            'trend_score': trend_score,
            'trend_quality': trend_quality,
            'volatility': volatility,
            'consistency': abs(trend_score) / 5  # 0-1的一致性评分
        }
    
    def get_comprehensive_analysis(self, data):
        """综合价格行为分析"""
        if len(data) < self.min_bars_for_analysis:
            return None
        
        analysis = {
            'market_structure': self.identify_market_structure(data),
            'candlestick_patterns': self.identify_candlestick_patterns(data),
            'breakout_analysis': self.detect_breakout_patterns(data),
            'six_bar_analysis': self.analyze_six_bar_drift(data),
            'timestamp': datetime.now()
        }
        
        # 生成综合信号
        analysis['composite_signal'] = self._generate_composite_signal(analysis)
        
        return analysis
    
    def _generate_composite_signal(self, analysis):
        """生成综合交易信号"""
        signal_strength = 0
        signal_direction = 'neutral'
        confidence = 0
        
        # 市场结构权重
        if analysis['market_structure']:
            structure = analysis['market_structure']
            if structure['trend'] == 'uptrend':
                signal_strength += structure['strength'] * 0.3
                signal_direction = 'bullish'
            elif structure['trend'] == 'downtrend':
                signal_strength -= structure['strength'] * 0.3
                signal_direction = 'bearish'
        
        # K线形态权重
        pattern_score = 0
        for pattern in analysis['candlestick_patterns']:
            if pattern['type'] == 'bullish_reversal':
                pattern_score += pattern['reliability'] * 0.2
            elif pattern['type'] == 'bearish_reversal':
                pattern_score -= pattern['reliability'] * 0.2
        
        signal_strength += pattern_score
        
        # 突破分析权重
        if analysis['breakout_analysis']:
            breakout = analysis['breakout_analysis']
            if breakout['type'] == 'upward_breakout':
                signal_strength += breakout['strength'] * (2 if breakout['volume_confirmation'] else 1)
            elif breakout['type'] == 'downward_breakout':
                signal_strength -= breakout['strength'] * (2 if breakout['volume_confirmation'] else 1)
        
        # 六根K线分析权重
        if analysis['six_bar_analysis']:
            six_bar = analysis['six_bar_analysis']
            if six_bar['trend_quality'] == 'strong_uptrend':
                signal_strength += 20 * six_bar['consistency']
            elif six_bar['trend_quality'] == 'strong_downtrend':
                signal_strength -= 20 * six_bar['consistency']
        
        # 确定最终方向和置信度
        if signal_strength > 10:
            signal_direction = 'bullish'
            confidence = min(signal_strength / 50, 1.0)
        elif signal_strength < -10:
            signal_direction = 'bearish'
            confidence = min(abs(signal_strength) / 50, 1.0)
        else:
            signal_direction = 'neutral'
            confidence = 0.5
        
        return {
            'direction': signal_direction,
            'strength': abs(signal_strength),
            'confidence': confidence,
            'raw_score': signal_strength
        }