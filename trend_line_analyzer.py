# 趋势线分析器 - 用于识别支撑/阻力趋势线并判断入场时机
# 解决"买在高点"问题的核心模块

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


class TrendLineAnalyzer:
    """
    趋势线分析器（优化版）
    
    核心功能：
    1. 识别上升趋势线（连接低点）- 使用更长历史数据（最长6个月）
    2. 识别下降趋势线（连接高点）
    3. 判断当前价格相对趋势线的位置
    4. 检测是否跌破趋势线（看空信号）
    5. 评估趋势强度，为强趋势股票提供更宽松的止损
    """
    
    def __init__(self, lookback_days: int = 120):
        """
        初始化趋势线分析器
        
        参数:
            lookback_days: 回溯天数，默认120天（约6个月）
        """
        self.lookback_days = lookback_days
        self.min_touches = 2  # 至少2个触点才算有效趋势线
        self.touch_tolerance = 0.02  # 2%的容差范围
    
    def analyze(self, data: pd.DataFrame) -> Dict:
        """
        完整的趋势线分析（使用更长历史数据）
        
        返回:
        {
            'uptrend_line': {...},  # 上升趋势线
            'downtrend_line': {...},  # 下降趋势线
            'current_position': str,  # 当前价格位置
            'entry_quality': float,  # 入场质量评分 (0-100)
            'broken_support': bool,  # 是否跌破支撑
            'trend_strength': float,  # 趋势强度 (0-100)
            'suggested_stop_buffer': float,  # 建议止损缓冲（百分比）
        }
        """
        if len(data) < 20:
            return self._empty_result()
        
        # 使用更长的历史数据进行趋势线分析
        analysis_data = data.tail(min(self.lookback_days, len(data)))
        current_price = data['close'].iloc[-1]
        
        # 识别上升趋势线（连接低点）
        uptrend_line = self._find_uptrend_line(analysis_data)
        
        # 识别下降趋势线（连接高点）
        downtrend_line = self._find_downtrend_line(analysis_data)
        
        # 评估趋势强度
        trend_strength = self._calculate_trend_strength(analysis_data, uptrend_line)
        
        # 判断当前价格位置
        current_position = self._analyze_current_position(
            current_price, uptrend_line, downtrend_line, analysis_data
        )
        
        # 计算入场质量评分
        entry_quality = self._calculate_entry_quality(
            current_price, uptrend_line, downtrend_line, analysis_data
        )
        
        # 检测是否跌破支撑
        broken_support = self._check_broken_support(
            data, uptrend_line
        )
        
        # 根据趋势强度计算建议止损缓冲
        suggested_stop_buffer = self._calculate_stop_buffer(trend_strength, uptrend_line)
        
        return {
            'uptrend_line': uptrend_line,
            'downtrend_line': downtrend_line,
            'current_position': current_position,
            'entry_quality': entry_quality,
            'broken_support': broken_support,
            'trend_strength': trend_strength,
            'suggested_stop_buffer': suggested_stop_buffer,
            'analysis': {
                'distance_from_support_pct': self._distance_from_line(
                    current_price, uptrend_line, len(analysis_data) - 1
                ) if uptrend_line['valid'] else None,
                'distance_from_resistance_pct': self._distance_from_line(
                    current_price, downtrend_line, len(analysis_data) - 1
                ) if downtrend_line['valid'] else None,
                'lookback_days': len(analysis_data),
            }
        }
    
    def _find_uptrend_line(self, data: pd.DataFrame) -> Dict:
        """
        寻找上升趋势线（连接低点）
        
        方法：
        1. 识别所有摆动低点
        2. 尝试不同的低点组合
        3. 选择最佳拟合线（最多触点、最小偏差）
        """
        swing_lows = self._find_swing_lows(data)
        
        if len(swing_lows) < 2:
            return {'valid': False}
        
        best_line = None
        best_score = 0
        
        # 尝试不同的起点
        for i in range(len(swing_lows) - 1):
            for j in range(i + 1, len(swing_lows)):
                idx1, price1 = swing_lows[i]
                idx2, price2 = swing_lows[j]
                
                # 必须是上升的
                if price2 <= price1:
                    continue
                
                # 计算斜率和截距
                slope = (price2 - price1) / (idx2 - idx1)
                intercept = price1 - slope * idx1
                
                # 检查这条线的质量
                touches = 0
                total_deviation = 0
                
                for idx, price in swing_lows:
                    expected_price = slope * idx + intercept
                    # 避免除以零
                    if abs(expected_price) < 1e-10:
                        deviation = float('inf')
                    else:
                        deviation = abs(price - expected_price) / abs(expected_price)
                    
                    if deviation < self.touch_tolerance:
                        touches += 1
                    
                    total_deviation += deviation
                
                # 评分：触点数量 * 100 - 平均偏差 * 10
                score = touches * 100 - (total_deviation / len(swing_lows)) * 10
                
                if touches >= self.min_touches and score > best_score:
                    best_score = score
                    best_line = {
                        'valid': True,
                        'slope': slope,
                        'intercept': intercept,
                        'touches': touches,
                        'start_idx': idx1,
                        'start_price': price1,
                        'end_idx': idx2,
                        'end_price': price2,
                        'angle_degrees': np.degrees(np.arctan(slope)),
                    }
        
        return best_line if best_line else {'valid': False}
    
    def _find_downtrend_line(self, data: pd.DataFrame) -> Dict:
        """
        寻找下降趋势线（连接高点）
        """
        swing_highs = self._find_swing_highs(data)
        
        if len(swing_highs) < 2:
            return {'valid': False}
        
        best_line = None
        best_score = 0
        
        for i in range(len(swing_highs) - 1):
            for j in range(i + 1, len(swing_highs)):
                idx1, price1 = swing_highs[i]
                idx2, price2 = swing_highs[j]
                
                # 必须是下降的
                if price2 >= price1:
                    continue
                
                slope = (price2 - price1) / (idx2 - idx1)
                intercept = price1 - slope * idx1
                
                touches = 0
                total_deviation = 0
                
                for idx, price in swing_highs:
                    expected_price = slope * idx + intercept
                    # 避免除以零
                    if abs(expected_price) < 1e-10:
                        deviation = float('inf')
                    else:
                        deviation = abs(price - expected_price) / abs(expected_price)
                    
                    if deviation < self.touch_tolerance:
                        touches += 1
                    
                    total_deviation += deviation
                
                score = touches * 100 - (total_deviation / len(swing_highs)) * 10
                
                if touches >= self.min_touches and score > best_score:
                    best_score = score
                    best_line = {
                        'valid': True,
                        'slope': slope,
                        'intercept': intercept,
                        'touches': touches,
                        'start_idx': idx1,
                        'start_price': price1,
                        'end_idx': idx2,
                        'end_price': price2,
                        'angle_degrees': np.degrees(np.arctan(slope)),
                    }
        
        return best_line if best_line else {'valid': False}
    
    def _find_swing_lows(self, data: pd.DataFrame, window: int = 5) -> List[Tuple[int, float]]:
        """识别摆动低点"""
        swing_lows = []
        
        for i in range(window, len(data) - window):
            current_low = data['low'].iloc[i]
            
            # 检查是否是局部最低点
            is_swing_low = True
            for j in range(i - window, i + window + 1):
                if j != i and data['low'].iloc[j] < current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append((i, current_low))
        
        return swing_lows
    
    def _find_swing_highs(self, data: pd.DataFrame, window: int = 5) -> List[Tuple[int, float]]:
        """识别摆动高点"""
        swing_highs = []
        
        for i in range(window, len(data) - window):
            current_high = data['high'].iloc[i]
            
            is_swing_high = True
            for j in range(i - window, i + window + 1):
                if j != i and data['high'].iloc[j] > current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append((i, current_high))
        
        return swing_highs
    
    def _analyze_current_position(self, current_price: float, 
                                  uptrend_line: Dict, 
                                  downtrend_line: Dict,
                                  data: pd.DataFrame) -> str:
        """
        分析当前价格位置
        
        返回:
        - 'near_support': 接近支撑线（好的买入位置）
        - 'mid_channel': 通道中部（中性）
        - 'near_resistance': 接近阻力线（不好的买入位置）
        - 'above_channel': 超出通道上方（追高）
        - 'below_support': 跌破支撑（危险）
        """
        current_idx = len(data) - 1
        
        if uptrend_line['valid']:
            support_price = uptrend_line['slope'] * current_idx + uptrend_line['intercept']
            distance_from_support = (current_price - support_price) / support_price
            
            # 接近支撑线（±3%）
            if -0.03 <= distance_from_support <= 0.03:
                return 'near_support'
            
            # 跌破支撑
            if distance_from_support < -0.03:
                return 'below_support'
            
            # 如果有下降趋势线（通道）
            if downtrend_line['valid']:
                resistance_price = downtrend_line['slope'] * current_idx + downtrend_line['intercept']
                channel_height = resistance_price - support_price
                
                # 避免除以零
                if channel_height <= 0:
                    return 'near_support' if distance_from_support <= 0.03 else 'mid_channel'
                
                position_in_channel = (current_price - support_price) / channel_height
                
                if position_in_channel < 0.3:
                    return 'near_support'
                elif position_in_channel < 0.7:
                    return 'mid_channel'
                elif position_in_channel < 1.1:
                    return 'near_resistance'
                else:
                    return 'above_channel'
            
            # 只有上升趋势线
            if distance_from_support > 0.08:
                return 'near_resistance'
            else:
                return 'mid_channel'
        
        return 'unknown'
    
    def _calculate_entry_quality(self, current_price: float,
                                 uptrend_line: Dict,
                                 downtrend_line: Dict,
                                 data: pd.DataFrame) -> float:
        """
        计算入场质量评分 (0-100)
        
        评分标准：
        - 接近支撑线：80-100分
        - 通道下半部：60-80分
        - 通道中部：40-60分
        - 通道上半部：20-40分
        - 接近阻力/追高：0-20分
        """
        if not uptrend_line['valid']:
            return 50  # 无趋势线，中性评分
        
        current_idx = len(data) - 1
        support_price = uptrend_line['slope'] * current_idx + uptrend_line['intercept']
        distance_from_support = (current_price - support_price) / support_price
        
        # 跌破支撑
        if distance_from_support < -0.03:
            return 0
        
        # 接近支撑（最佳入场）
        if distance_from_support <= 0.03:
            quality = 100 - abs(distance_from_support) * 1000
            return max(80, min(100, quality))
        
        # 如果有通道
        if downtrend_line['valid']:
            resistance_price = downtrend_line['slope'] * current_idx + downtrend_line['intercept']
            channel_height = resistance_price - support_price
            
            # 避免除以零
            if channel_height <= 0:
                return 80 if distance_from_support <= 0.03 else 50
            
            # 避免除以零
            if abs(channel_height) < 1e-10:
                position_in_channel = float('inf')
            else:
                position_in_channel = (current_price - support_price) / channel_height
            
            if position_in_channel < 0.3:
                return 80 - position_in_channel * 100
            elif position_in_channel < 0.5:
                return 60 - (position_in_channel - 0.3) * 100
            elif position_in_channel < 0.7:
                return 40 - (position_in_channel - 0.5) * 100
            elif position_in_channel < 1.0:
                return 20 - (position_in_channel - 0.7) * 67
            else:
                return 0
        
        # 只有上升趋势线
        if distance_from_support < 0.05:
            return 70
        elif distance_from_support < 0.08:
            return 50
        elif distance_from_support < 0.12:
            return 30
        else:
            return 10
    
    def _check_broken_support(self, data: pd.DataFrame, uptrend_line: Dict) -> bool:
        """
        检测是否跌破支撑趋势线
        
        条件：
        1. 收盘价跌破趋势线至少2%
        2. 最近3根K线都在趋势线下方
        """
        if not uptrend_line['valid']:
            return False
        
        if len(data) < 3:
            return False
        
        # 检查最近3根K线
        broken_count = 0
        for i in range(len(data) - 3, len(data)):
            expected_price = uptrend_line['slope'] * i + uptrend_line['intercept']
            actual_close = data['close'].iloc[i]
            
            if actual_close < expected_price * 0.98:  # 跌破2%
                broken_count += 1
        
        return broken_count >= 2
    
    def _distance_from_line(self, price: float, line: Dict, idx: int) -> float:
        """计算价格距离趋势线的百分比"""
        if not line['valid']:
            return None
        
        expected_price = line['slope'] * idx + line['intercept']
        return (price - expected_price) / expected_price * 100
    
    def _calculate_trend_strength(self, data: pd.DataFrame, uptrend_line: Dict) -> float:
        """
        计算趋势强度 (0-100)
        
        评估标准：
        1. 趋势线斜率（越陡越强）
        2. 触点数量（越多越可靠）
        3. 趋势持续时间（越长越稳定）
        4. 价格沿趋势线的一致性
        5. 成交量配合度
        """
        if not uptrend_line['valid']:
            return 0
        
        strength = 0
        
        # 1. 斜率评分（最高30分）
        # 理想斜率：每天上涨0.1%-0.3%
        daily_slope_pct = (uptrend_line['slope'] / uptrend_line['start_price']) * 100
        
        if 0.05 <= daily_slope_pct <= 0.5:  # 理想范围
            strength += 30
        elif 0.02 <= daily_slope_pct < 0.05:  # 较缓
            strength += 20
        elif 0.5 < daily_slope_pct <= 1.0:  # 较陡
            strength += 25
        elif daily_slope_pct > 0:  # 至少是上升的
            strength += 10
        
        # 2. 触点数量（最高25分）
        touches = uptrend_line['touches']
        if touches >= 5:
            strength += 25
        elif touches >= 4:
            strength += 20
        elif touches >= 3:
            strength += 15
        elif touches >= 2:
            strength += 10
        
        # 3. 趋势持续时间（最高20分）
        duration = uptrend_line['end_idx'] - uptrend_line['start_idx']
        if duration >= 90:  # 3个月以上
            strength += 20
        elif duration >= 60:  # 2个月以上
            strength += 15
        elif duration >= 30:  # 1个月以上
            strength += 10
        elif duration >= 15:  # 半个月以上
            strength += 5
        
        # 4. 价格一致性（最高15分）
        # 检查最近价格是否都在趋势线上方
        recent_data = data.tail(20)
        above_count = 0
        
        for i in range(len(recent_data)):
            idx = len(data) - len(recent_data) + i
            expected_price = uptrend_line['slope'] * idx + uptrend_line['intercept']
            actual_price = recent_data.iloc[i]['close']
            
            if actual_price >= expected_price * 0.98:  # 允许2%偏差
                above_count += 1
        
        consistency = above_count / len(recent_data)
        strength += consistency * 15
        
        # 5. 成交量配合（最高10分）
        # 上涨时成交量放大，下跌时成交量萎缩
        recent_data = data.tail(20)
        volume_score = 0
        
        for i in range(1, len(recent_data)):
            price_change = recent_data.iloc[i]['close'] - recent_data.iloc[i-1]['close']
            volume_change = recent_data.iloc[i]['volume'] - recent_data.iloc[i-1]['volume']
            
            # 价格上涨且成交量放大
            if price_change > 0 and volume_change > 0:
                volume_score += 1
            # 价格下跌且成交量萎缩
            elif price_change < 0 and volume_change < 0:
                volume_score += 0.5
        
        strength += (volume_score / len(recent_data)) * 10
        
        return min(strength, 100)
    
    def _calculate_stop_buffer(self, trend_strength: float, uptrend_line: Dict) -> float:
        """
        根据趋势强度计算建议止损缓冲（百分比）
        
        趋势越强，给予越宽松的止损空间
        
        返回: 止损缓冲百分比（如0.05表示5%）
        """
        if not uptrend_line['valid']:
            return 0.03  # 默认3%
        
        # 基础止损：3%
        base_buffer = 0.03
        
        # 根据趋势强度调整
        if trend_strength >= 80:
            # 强趋势：给予5-6%的止损空间
            return 0.05 + (trend_strength - 80) / 100 * 0.01
        elif trend_strength >= 60:
            # 中等趋势：给予4-5%的止损空间
            return 0.04 + (trend_strength - 60) / 100 * 0.01
        elif trend_strength >= 40:
            # 弱趋势：给予3-4%的止损空间
            return 0.03 + (trend_strength - 40) / 100 * 0.01
        else:
            # 很弱的趋势：保持3%
            return base_buffer
    
    def _empty_result(self) -> Dict:
        """返回空结果"""
        return {
            'uptrend_line': {'valid': False},
            'downtrend_line': {'valid': False},
            'current_position': 'unknown',
            'entry_quality': 50,
            'broken_support': False,
            'trend_strength': 0,
            'suggested_stop_buffer': 0.03,
            'analysis': {}
        }
