# 价格行为分析模块 - 基于K线结构的趋势识别
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config.strategy_config import BEARISH_STRONG_TREND_THRESHOLD, BEARISH_MODERATE_TREND_THRESHOLD, \
    BEARISH_WEAK_TREND_THRESHOLD, TREND_MA_MID_PERIOD, TREND_MA_SHORT_PERIOD, TREND_MA_LONG_PERIOD, \
    STRONG_TREND_THRESHOLD, ENTRY_MAX_DAILY_RETURN, ENTRY_MAX_DISTANCE_FROM_SUPPORT, ENTRY_MAX_CONSECUTIVE_RISES, \
    ENTRY_MAX_VOLUME_RATIO, ENTRY_MIN_DISTANCE_FROM_HIGH


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
        close_20_days_ago = data['close'].iloc[-20]
        if close_20_days_ago != 0:
            price_momentum = (data['close'].iloc[-1] - close_20_days_ago) / close_20_days_ago * 100
        else:
            price_momentum = 0
        
        # 基于成交量确认
        recent_volume = data['volume'].tail(10).mean()
        avg_volume = data['volume'].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # 基于波动率
        atr = self._calculate_atr(data, 14)
        current_close = data['close'].iloc[-1]
        if len(atr) > 0 and current_close != 0:
            volatility_factor = min(atr[-1] / current_close * 100, 10)
        else:
            volatility_factor = 5
        
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
    
    def _check_entry_timing(self, data, current_price,
                          max_daily_return=ENTRY_MAX_DAILY_RETURN,
                          min_distance_from_high=ENTRY_MIN_DISTANCE_FROM_HIGH,
                          max_consecutive_rises=ENTRY_MAX_CONSECUTIVE_RISES,
                          max_volume_ratio=ENTRY_MAX_VOLUME_RATIO,
                          max_distance_from_support=ENTRY_MAX_DISTANCE_FROM_SUPPORT):
        """
        通用的入场时机检查 - 避免追高
        
        条件：
        1. 当日涨幅 < max_daily_return (默认3%)
        2. 距离近期高点 >= min_distance_from_high (默认3%)
        3. 没有连续max_consecutive_rises天大涨 (默认3天)
        4. 成交量不是异常放大 (< max_volume_ratio倍，默认3倍)
        5. 距离支撑位 < max_distance_from_support (默认8%)
        
        参数:
            data: 价格数据
            current_price: 当前价格
            max_daily_return: 最大当日涨幅
            min_distance_from_high: 距离高点最小距离
            max_consecutive_rises: 最大连续上涨天数
            max_volume_ratio: 最大成交量倍数
            max_distance_from_support: 距离支撑最大距离
        
        返回: {'is_good': bool, 'daily_return_pct': float, ...}
        """
        if len(data) < 10:
            return {'is_good': False}
        
        recent_data = data.tail(10)
        current_bar = data.iloc[-1]
        prev_close = data.iloc[-2]['close']
        
        # 条件1：当日涨幅
        if prev_close != 0:
            daily_return = (current_bar['close'] - prev_close) / prev_close
        else:
            daily_return = 0
        condition1 = daily_return < max_daily_return
        
        # 条件2：距离高点
        recent_high = recent_data['high'].max()
        if recent_high != 0:
            distance_from_high = (recent_high - current_price) / recent_high
        else:
            distance_from_high = 0
        condition2 = distance_from_high >= min_distance_from_high
        
        # 条件3：连续大涨检查
        consecutive_rises = 0
        for i in range(len(recent_data) - 3, len(recent_data)):
            if i > 0:
                prev_close_val = recent_data.iloc[i-1]['close']
                if prev_close_val != 0:
                    ret = (recent_data.iloc[i]['close'] - prev_close_val) / prev_close_val
                else:
                    ret = 0
                if ret > 0.02:
                    consecutive_rises += 1
                else:
                    break
        condition3 = consecutive_rises < max_consecutive_rises
        
        # 条件4：成交量
        current_volume = data['volume'].iloc[-1]
        avg_volume = recent_data['volume'].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        condition4 = volume_ratio < max_volume_ratio
        
        # 条件5：距离支撑
        recent_low = recent_data['low'].min()
        if recent_low != 0:
            distance_from_support = (current_price - recent_low) / recent_low
        else:
            distance_from_support = 0
        condition5 = distance_from_support < max_distance_from_support
        
        is_good = condition1 and condition2 and condition3 and condition4 and condition5
        
        return {
            'is_good': is_good,
            'daily_return_pct': daily_return * 100,
            'distance_from_high_pct': distance_from_high * 100,
            'consecutive_rises': consecutive_rises,
            'volume_ratio': volume_ratio,
            'distance_from_support_pct': distance_from_support * 100
        }
    
    def check_multi_period_trend(self, data, 
                                 long_period=TREND_MA_LONG_PERIOD,
                                 mid_period=TREND_MA_MID_PERIOD,
                                 short_period=TREND_MA_SHORT_PERIOD):
        """
        多周期趋势验证 - 避免将反弹误判为上升趋势
        
        核心功能：
        1. 三周期均线系统：长期、中期、短期
        2. 大趋势验证：长期均线必须向上
        3. 多周期一致性：短中长期趋势方向一致
        4. 反弹识别：区分趋势转换和短期反弹
        5. 趋势强度评分：综合评估趋势质量
        
        参数:
            data: 价格数据
            long_period: 长期均线周期 (默认90)
            mid_period: 中期均线周期 (默认30)
            short_period: 短期均线周期 (默认5)
        
        返回: {'is_uptrend': bool, 'trend_strength': float, ...}
        """
        import talib
        
        if len(data) < long_period:
            return {'is_uptrend': False, 'trend_strength': 0}
        
        # 计算三条均线
        ma_long = talib.SMA(data['close'].values, timeperiod=long_period)
        ma_mid = talib.SMA(data['close'].values, timeperiod=mid_period)
        ma_short = talib.SMA(data['close'].values, timeperiod=short_period)
        current_price = data['close'].iloc[-1]
        
        # ========== 第一步：大趋势验证（长期均线） ==========
        
        # 长期均线斜率
        long_slope_lookback = max(10, int(long_period / 5))
        ma_long_slope = (ma_long[-1] - ma_long[-long_slope_lookback]) / ma_long[-long_slope_lookback] if ma_long[-long_slope_lookback] > 0 else 0
        long_trend_up = ma_long_slope >= -0.01
        
        # 价格相对长期均线位置
        if ma_long[-1] != 0:
            price_vs_long = (current_price - ma_long[-1]) / ma_long[-1]
        else:
            price_vs_long = 0
        above_long_ma = price_vs_long > -0.05
        
        # ========== 第二步：多周期一致性检查 ==========
        
        # 中期均线斜率
        mid_slope_lookback = max(5, int(mid_period / 3))
        ma_mid_slope = (ma_mid[-1] - ma_mid[-mid_slope_lookback]) / ma_mid[-mid_slope_lookback] if ma_mid[-mid_slope_lookback] > 0 else 0
        mid_trend_up = ma_mid_slope > -0.02
        
        # 短期均线斜率
        short_slope_lookback = max(2, int(short_period / 2))
        ma_short_slope = (ma_short[-1] - ma_short[-short_slope_lookback]) / ma_short[-short_slope_lookback] if ma_short[-short_slope_lookback] > 0 else 0
        short_trend_up = ma_short_slope > -0.05
        
        # 均线排列（多头排列）
        ma_alignment = ma_short[-1] > ma_mid[-1] * 0.98 and ma_mid[-1] > ma_long[-1] * 0.98
        
        # ========== 第三步：反弹识别机制 ==========
        
        is_bounce_in_downtrend = False
        if not long_trend_up:
            recent_low = data['low'].tail(10).min()
            if recent_low != 0:
                bounce_strength = (current_price - recent_low) / recent_low
            else:
                bounce_strength = 0
            if bounce_strength > 0.05:
                is_bounce_in_downtrend = True
        
        # ========== 第四步：HH-HL结构验证 ==========
        
        recent_data = data.tail(30)
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent_data) - 2):
            if (recent_data.iloc[i]['high'] > recent_data.iloc[i-1]['high'] and
                recent_data.iloc[i]['high'] > recent_data.iloc[i+1]['high']):
                swing_highs.append(recent_data.iloc[i]['high'])
            
            if (recent_data.iloc[i]['low'] < recent_data.iloc[i-1]['low'] and
                recent_data.iloc[i]['low'] < recent_data.iloc[i+1]['low']):
                swing_lows.append(recent_data.iloc[i]['low'])
        
        has_hh_hl = False
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            has_hh_hl = (swing_highs[-1] > swing_highs[-2] and 
                        swing_lows[-1] > swing_lows[-2])
        
        # ========== 第五步：趋势强度评分（0-100） ==========
        
        trend_strength = 0
        
        # 1. 长期趋势方向（30分）
        if long_trend_up:
            if ma_long_slope > 0.05:
                trend_strength += 20
            elif ma_long_slope > 0.02:
                trend_strength += 15
            elif ma_long_slope > 0:
                trend_strength += 10
        
        # 2. 多周期一致性（35分）
        consistency_score = sum([long_trend_up, mid_trend_up, short_trend_up])
        if consistency_score == 3:
            trend_strength += 35
        elif consistency_score == 2:
            trend_strength += 25
        elif consistency_score == 1:
            trend_strength += 15
        
        # 3. 均线排列（20分）
        if ma_alignment:
            trend_strength += 20
        elif ma_short[-1] > ma_mid[-1] or ma_mid[-1] > ma_long[-1]:
            trend_strength += 15
        
        # 4. HH-HL结构（20分）
        if has_hh_hl:
            trend_strength += 20
        
        # 5. 价格位置（10分）
        if price_vs_long > 0.05:
            trend_strength += 5
        elif price_vs_long > 0:
            trend_strength += 10
        elif price_vs_long > -0.03:
            trend_strength += 3
        
        # ========== 第六步：综合判断 ==========
        
        is_uptrend = (
            long_trend_up and
            consistency_score >= 2 and
            trend_strength >= 20
        )
        
        return {
            'is_uptrend': is_uptrend,
            'trend_strength': trend_strength,
            'ma_long': ma_long[-1],
            'ma_mid': ma_mid[-1],
            'ma_short': ma_short[-1],
            'ma_long_slope': ma_long_slope,
            'ma_mid_slope': ma_mid_slope,
            'ma_short_slope': ma_short_slope,
            'long_trend_up': long_trend_up,
            'mid_trend_up': mid_trend_up,
            'short_trend_up': short_trend_up,
            'consistency_score': consistency_score,
            'ma_alignment': ma_alignment,
            'has_hh_hl': has_hh_hl,
            'is_bounce': is_bounce_in_downtrend,
            'price_vs_long_pct': price_vs_long * 100,
            'conditions_met': consistency_score + (1 if ma_alignment else 0) + (1 if has_hh_hl else 0),
            'strength': 'strong' if trend_strength >= 70 else ('moderate' if trend_strength >= 50 else 'weak')
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
    
    def detect_bearish_signals(self, data, candlestick_detector=None, trend_strength=None, 
                              historical_signals=None, is_bottom_strategy=False,
                              dynamic_threshold_enabled=True,
                              min_confidence=60):
        """
        检测看空信号 - 通用方法
        
        核心功能：
        1. 动态阈值：根据趋势强度调整看空信号阈值
        2. 信号累积：3天内的看空信号可累积（每天衰减30%）
        3. 顶部预警：增加顶部特征检测
        4. 底部策略支持：对于底部反转策略，忽略顶部看空信号
        
        看空信号包括：
        1. 跌破关键支撑位（MA50/MA30）
        2. 看空K线形态（射击之星、乌云盖顶、三只乌鸦等）
        3. 技术指标看空（MACD死叉、RSI超买后回落）
        4. 成交量异常放大的阴线
        5. 顶部预警信号
        
        参数:
            data: 价格数据
            candlestick_detector: K线形态检测器实例
            trend_strength: 当前趋势强度（0-100）
            historical_signals: 历史看空信号列表
            is_bottom_strategy: 是否为底部策略
            dynamic_threshold_enabled: 是否启用动态阈值
            min_confidence: 最小置信度阈值
        
        返回: {'detected': bool, 'confidence': float, 'reasons': list, ...}
        """
        import talib
        
        try:
            if len(data) < 30:
                return {'detected': False, 'confidence': 0, 'reasons': [], 
                       'threshold': min_confidence, 'top_warning': False}
            
            reasons = []
            confidence = 0
            top_warning = False
            ignored_signals = []
            
            current_bar = data.iloc[-1]
            current_price = current_bar['close']
            current_date = current_bar.name if hasattr(current_bar, 'name') else None
            
            # ========== 动态阈值 ==========
            
            if dynamic_threshold_enabled and trend_strength is not None:
                if trend_strength >= STRONG_TREND_THRESHOLD:  # 强趋势
                    dynamic_threshold = BEARISH_STRONG_TREND_THRESHOLD
                    reasons.append(f'强趋势逃顶模式(阈值{dynamic_threshold})')
                elif trend_strength >= 50:  # 中等趋势
                    dynamic_threshold = BEARISH_MODERATE_TREND_THRESHOLD
                else:  # 弱趋势
                    dynamic_threshold = BEARISH_WEAK_TREND_THRESHOLD
            else:
                dynamic_threshold = min_confidence
            
            # ========== 累积历史信号 ==========
            
            accumulated_confidence = 0
            if historical_signals and len(historical_signals) > 0:
                import pandas as pd
                for hist_signal in historical_signals[-3:]:  # 最近3天
                    if current_date and 'date' in hist_signal:
                        try:
                            days_ago = (pd.to_datetime(current_date) - pd.to_datetime(hist_signal['date'])).days
                            if 0 < days_ago <= 3:
                                decay_factor = 0.7 ** days_ago
                                accumulated_confidence += hist_signal.get('confidence', 0) * decay_factor
                        except:
                            pass
                
                if accumulated_confidence > 0:
                    reasons.append(f'累积历史信号+{accumulated_confidence:.1f}分')
                    confidence += accumulated_confidence
            
            # ========== 顶部预警检测 ==========
            
            if not is_bottom_strategy:
                ma_top = talib.SMA(data['close'].values, timeperiod=50)
                
                # 价格偏离均线过远
                if ma_top[-1] != 0:
                    price_deviation = (current_price - ma_top[-1]) / ma_top[-1]
                else:
                    price_deviation = 0
                if price_deviation > 0.10:
                    top_warning = True
                    reasons.append(f'价格偏离MA50过远({price_deviation*100:.1f}%)')
                    confidence += 10
                
                # RSI极度超买
                rsi = talib.RSI(data['close'].values, timeperiod=14)
                if len(rsi) >= 2 and rsi[-1] > 75:
                    top_warning = True
                    reasons.append(f'RSI极度超买({rsi[-1]:.1f})')
                    confidence += 25
                
                # 成交量背离
                if len(data) >= 10:
                    recent_volume = data['volume'].tail(5).mean()
                    prev_volume = data['volume'].tail(15).head(10).mean()
                    close_5_days_ago = data['close'].iloc[-5]
                    if close_5_days_ago != 0:
                        recent_price_change = (data['close'].iloc[-1] - close_5_days_ago) / close_5_days_ago
                    else:
                        recent_price_change = 0
                    
                    if recent_price_change > 0 and recent_volume <= prev_volume * 0.8:
                        top_warning = True
                        reasons.append('价涨量缩背离')
                        confidence += 10
            else:
                ignored_signals.append('顶部预警信号已忽略（底部策略）')
            
            # ========== 均线斜率变化率 ==========
            
            if not is_bottom_strategy:
                if len(data) >= 60:
                    ma_top = talib.SMA(data['close'].values, timeperiod=50)
                    recent_slope = (ma_top[-1] - ma_top[-5]) / ma_top[-5] if ma_top[-5] > 0 else 0
                    prev_slope = (ma_top[-5] - ma_top[-10]) / ma_top[-10] if ma_top[-10] > 0 else 0
                    
                    if prev_slope != 0:
                        slope_change_rate = (recent_slope - prev_slope) / abs(prev_slope)
                    else:
                        slope_change_rate = 0
                    
                    if prev_slope > 0.01 and recent_slope < 0:
                        if abs(slope_change_rate) > 2.0:
                            reasons.append(f'{ma_top}斜率急转直下(变化率{slope_change_rate*100:.0f}%)')
                            confidence += 30
                            top_warning = True
                        elif abs(slope_change_rate) > 1.0:
                            reasons.append(f'{ma_top}斜率转负(变化率{slope_change_rate*100:.0f}%)')
                            confidence += 20
                            top_warning = True
            else:
                ignored_signals.append('均线斜率变化率已忽略（底部策略）')
            
            # ========== 跌破关键均线 ==========
            
            ma50 = talib.SMA(data['close'].values, timeperiod=50) if len(data) >= 50 else None
            ma30 = talib.SMA(data['close'].values, timeperiod=30) if len(data) >= 30 else None
            
            if ma50 is not None and current_price < ma50[-1] * 0.98:
                reasons.append('跌破MA50')
                confidence += 30
            
            if ma30 is not None and current_price < ma30[-1] * 0.98:
                reasons.append('跌破MA30')
                confidence += 20
            
            # ========== K线形态 ==========
            
            if candlestick_detector:
                bearish_patterns = candlestick_detector.detect_all_bearish_patterns(data)
                
                for pattern in bearish_patterns:
                    if is_bottom_strategy and pattern.get('description') == '三只乌鸦':
                        ignored_signals.append('三只乌鸦形态已忽略（底部策略）')
                        continue
                    
                    reasons.append(pattern['description'] + '形态')
                    confidence += pattern['score']
            
            # ========== 技术指标 ==========
            
            # MACD死叉
            macd, macd_signal, macd_hist = talib.MACD(data['close'].values)
            if len(macd) >= 2:
                if macd[-1] < macd_signal[-1] and macd[-2] >= macd_signal[-2]:
                    reasons.append('MACD死叉')
                    confidence += 20
                elif macd[-1] < macd_signal[-1] and macd_hist[-1] < 0:
                    reasons.append('MACD看空')
                    confidence += 15
            
            # RSI超买后回落
            if len(data) >= 14:
                rsi = talib.RSI(data['close'].values, timeperiod=14)
                if len(rsi) >= 2:
                    if rsi[-2] > 70 and rsi[-1] < 65:
                        reasons.append('RSI超买回落')
                        confidence += 15
            
            # ========== 成交量异常 ==========
            
            avg_volume = data['volume'].tail(20).mean()
            if (current_bar['close'] < current_bar['open'] and
                current_bar['volume'] > avg_volume * 2.0):
                reasons.append('放量下跌')
                confidence += 20
            
            # ========== 综合判断 ==========
            
            detected = confidence >= dynamic_threshold
            
            result = {
                'detected': detected,
                'confidence': min(confidence, 100),
                'reasons': reasons,
                'threshold': dynamic_threshold,
                'top_warning': top_warning,
                'accumulated_score': accumulated_confidence
            }
            
            if is_bottom_strategy and ignored_signals:
                result['ignored_signals'] = ignored_signals
            
            return result
            
        except Exception as e:
            print(f"检测看空信号时出错: {e}")
            return {'detected': False, 'confidence': 0, 'reasons': [], 
                   'threshold': min_confidence, 'top_warning': False}
    
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