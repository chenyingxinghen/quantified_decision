# 趋势线分析器 - 用于识别支撑/阻力趋势线并判断入场时机
# 解决"买在高点"问题的核心模块

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from config.strategy_config import TREND_LINE_LONG_PERIOD, TREND_LINE_SHORT_PERIOD, TREND_SEGMENTS, \
    TREND_BROKEN_THRESHOLD


class TrendLineAnalyzer:
    """
    趋势线分析器（优化版）
    
    核心功能：
    1. 识别上升趋势线（连接低点）- 使用更长历史数据
    2. 识别下降趋势线（连接高点）
    3. 判断当前价格相对趋势线的位置
    4. 检测是否跌破趋势线（看空信号）
    5. 评估趋势强度，为强趋势股票提供更宽松的止损
    
    关键改进：
    - 使用最低价而非收盘价连接低点
    - 使用最高价而非收盘价连接高点
    - 支持长期和短期两个时间窗口
    - 趋势线斜率可正可负（连接低点不一定是上升趋势）
    """
    
    def __init__(self, long_period: int = TREND_LINE_LONG_PERIOD, short_period: int = TREND_LINE_SHORT_PERIOD):
        """
        初始化趋势线分析器
        
        参数:
            long_period: 长期回溯天数，默认120天
            short_period: 短期回溯天数，默认10天
        """
        self.long_period = long_period
        self.short_period = short_period
        self.min_touches = 2  # 至少2个触点才算有效趋势线
        self.touch_tolerance = 0.01  # 2%的容差范围
        
        # 分段策略：将时间段分成N个部分，每个部分找一个最低点
        # 这样摆动点数量固定，趋势线更稳定
        # 长期：120天分成5段 = 每段24天，找5个最低点
        # 短期：30天分成5段 = 每段6天，找5个最低点
        self.long_segments = TREND_SEGMENTS  # 长期分成5段
        self.short_segments = TREND_SEGMENTS  # 短期分成5段
    
    def analyze(self, data: pd.DataFrame) -> Dict:
        """
        完整的趋势线分析（长期+短期双窗口）
        
        返回:
        {
            'uptrend_line': {...},  # 长期趋势线（连接低点）
            'downtrend_line': {...},  # 长期趋势线（连接高点）
            'short_uptrend_line': {...},  # 短期趋势线（连接低点）
            'short_downtrend_line': {...},  # 短期趋势线（连接高点）
            'current_position': str,  # 当前价格位置
            'entry_quality': float,  # 入场质量评分 (0-100)
            'broken_support': bool,  # 是否跌破支撑
            'trend_strength': float,  # 趋势强度 (0-100)
            'suggested_stop_buffer': float,  # 建议止损缓冲（百分比）
            'trend_reversal': bool,  # 是否趋势反转（长期下降+短期上升+均线向上）
        }
        """
        if len(data) < 20:
            return self._empty_result()
        
        current_price = data['close'].iloc[-1]
        
        # 长期趋势线分析（使用分段策略）
        long_data = data.tail(min(self.long_period, len(data)))
        long_uptrend = self._find_trendline_by_segment_lows(long_data, self.long_segments)
        long_downtrend = self._find_trendline_by_segment_highs(long_data, self.long_segments)
        
        # 短期趋势线分析（使用分段策略）
        short_data = data.tail(min(self.short_period, len(data)))
        short_uptrend = self._find_trendline_by_segment_lows(short_data, self.short_segments)
        short_downtrend = self._find_trendline_by_segment_highs(short_data, self.short_segments)
        
        # 检测趋势反转：长期下降+短期上升+均线向上发散
        trend_reversal = self._detect_trend_reversal(
            data, long_uptrend, short_uptrend, long_downtrend, short_downtrend
        )
        
        # 评估趋势强度（基于长期趋势线）
        trend_strength = self._calculate_trend_strength(long_data, long_uptrend, short_uptrend, trend_reversal)
        
        # 判断当前价格位置（优先使用短期趋势线）
        current_position = self._analyze_current_position(
            current_price, short_uptrend, short_downtrend, short_data
        )
        
        # 计算入场质量评分（综合长短期）
        entry_quality = self._calculate_entry_quality(
            current_price, short_uptrend, short_data, trend_reversal
        )
        
        # 检测是否跌破支撑（使用短期趋势线）
        broken_support = self._check_broken_support(data, short_uptrend)
        broken_support = broken_support or self._check_broken_support(data, long_uptrend)
        
        # 根据趋势强度计算建议止损缓冲
        suggested_stop_buffer = self._calculate_stop_buffer(trend_strength, short_uptrend)
        
        return {
            'uptrend_line': long_uptrend,
            'downtrend_line': long_downtrend,
            'short_uptrend_line': short_uptrend,
            'short_downtrend_line': short_downtrend,
            'current_position': current_position,
            'entry_quality': entry_quality,
            'broken_support': broken_support,
            'trend_strength': trend_strength,
            'suggested_stop_buffer': suggested_stop_buffer,
            'trend_reversal': trend_reversal,
            'analysis': {
                'distance_from_support_pct': self._distance_from_line(
                    current_price, short_uptrend, len(short_data) - 1
                ) if short_uptrend['valid'] else None,
                'distance_from_resistance_pct': self._distance_from_line(
                    current_price, short_downtrend, len(short_data) - 1
                ) if short_downtrend['valid'] else None,
                'long_period_days': len(long_data),
                'short_period_days': len(short_data),
                'long_segments': self.long_segments,
                'short_segments': self.short_segments,
            }
        }
    
    def _find_trendline_by_segment_lows(self, data: pd.DataFrame, num_segments: int = 5) -> Dict:
        """
        通过分段找最低点来构建趋势线（新方法）
        
        核心逻辑：
        1. 将时间段分成N个部分
        2. 每个部分找一个最低点
        3. 连接这N个最低点形成趋势线
        
        优点：
        - 摆动点数量固定，趋势线稳定
        - 避免过度拟合
        - 计算快速
        
        参数:
            data: 价格数据
            num_segments: 分成多少段（默认5段）
        """
        if len(data) < num_segments:
            return {'valid': False}
        
        # 分段找最低点
        segment_lows = []
        segment_size = len(data) / num_segments
        
        for i in range(num_segments):
            start_idx = int(i * segment_size)
            end_idx = int((i + 1) * segment_size)
            
            # 最后一段包含剩余的所有数据
            if i == num_segments - 1:
                end_idx = len(data)
            
            # 在这个段内找最低点
            segment_data = data.iloc[start_idx:end_idx]
            if len(segment_data) > 0:
                min_idx = segment_data['low'].idxmin()
                min_price = segment_data.loc[min_idx, 'low']
                
                # 转换为相对于整个data的索引
                actual_idx = min_idx
                segment_lows.append((actual_idx, min_price))
        
        if len(segment_lows) < 2:
            return {'valid': False}
        
        # 用这些点拟合趋势线
        return self._fit_trendline(segment_lows, data, 'low')
    
    def _find_trendline_by_segment_highs(self, data: pd.DataFrame, num_segments: int = 5) -> Dict:
        """
        通过分段找最高点来构建趋势线（新方法）
        
        参数:
            data: 价格数据
            num_segments: 分成多少段（默认5段）
        """
        if len(data) < num_segments:
            return {'valid': False}
        
        # 分段找最高点
        segment_highs = []
        segment_size = len(data) / num_segments
        
        for i in range(num_segments):
            start_idx = int(i * segment_size)
            end_idx = int((i + 1) * segment_size)
            
            if i == num_segments - 1:
                end_idx = len(data)
            
            segment_data = data.iloc[start_idx:end_idx]
            if len(segment_data) > 0:
                max_idx = segment_data['high'].idxmax()
                max_price = segment_data.loc[max_idx, 'high']
                
                actual_idx = max_idx
                segment_highs.append((actual_idx, max_price))
        
        if len(segment_highs) < 2:
            return {'valid': False}
        
        return self._fit_trendline(segment_highs, data, 'high')
    
    def _fit_trendline(self, points: List[Tuple[int, float]], data: pd.DataFrame, point_type: str) -> Dict:
        """
        用给定的点拟合趋势线
        
        参数:
            points: [(index, price), ...] 列表
            data: 原始数据（用于计算偏差）
            point_type: 点的类型（用于调试）
        """
        if len(points) < 2:
            return {'valid': False}
        if points[-1][1]==data.iloc[-1][point_type]:
            points.pop()
        # 线性回归
        indices = np.array([p[0] for p in points])
        prices = np.array([p[1] for p in points])
        
        # 计算斜率和截距
        slope = (prices[-1] - prices[0]) / (indices[-1] - indices[0])
        intercept = prices[0] - slope * indices[0]
        # 计算所有点的偏差
        touches = 0
        total_deviation = 0
        
        for idx, price in points:
            expected_price = slope * idx + intercept
            if abs(expected_price) < 1e-10:
                deviation = float('inf')
            else:
                deviation = abs(price - expected_price) / abs(expected_price)
            
            if deviation < self.touch_tolerance:
                touches += 1
            
            total_deviation += deviation

        # 检查有效性
        if touches < self.min_touches:
            return {'valid': False}
        
        return {
            'valid': True,
            'slope': slope,
            'intercept': intercept,
            'touches': touches,
            'start_idx': indices[0],
            'start_price': prices[0],
            'end_idx': indices[-1],
            'end_price': prices[-1],
            'angle_degrees': np.degrees(np.arctan(slope)),
            'is_rising': slope > 0,
            'point_type': point_type,
            'num_points': len(points),
        }
    

    
    def _detect_trend_reversal(self, data, long_uptrend, short_uptrend, long_downtrend, short_downtrend):
        """
        检测趋势反转：长期下降+短期上升+均线向上发散
        
        条件：
        1. 连接低点的长期趋势线斜率为负（下降趋势）
        2. 连接低点的短期趋势线斜率为正（上升趋势）
        3. 均线向上发散（MA5 > MA20 > MA60）
        
        返回: bool
        """
        if not long_uptrend['valid'] or not short_uptrend['valid']:
            return False
        
        # 条件1: 长期趋势线下降
        long_declining = long_uptrend['slope'] < 0
        
        # 条件2: 短期趋势线上升
        short_rising = short_uptrend['slope'] > 0
        
        # 条件3: 均线向上发散
        if len(data) < 60:
            return False
        
        ma5 = data['close'].rolling(window=5).mean().iloc[-1]
        ma20 = data['close'].rolling(window=20).mean().iloc[-1]
        ma60 = data['close'].rolling(window=60).mean().iloc[-1]
        
        ma_diverging_up = ma5 > ma20 > ma60
        
        return long_declining and short_rising and ma_diverging_up
    
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
        current_idx = data.index[-1]  # 使用DataFrame的实际索引
        
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
                                 short_uptrend: Dict,
                                 short_data: pd.DataFrame,
                                 trend_reversal: bool,
                                 additional_indicators: Dict = None) -> float:
        """
        计算入场质量评分 (0-100)
        
        评分标准：
        - 接近短期支撑线：80-100分
        - 通道下半部：60-80分
        - 通道中部：40-60分
        - 通道上半部：20-40分
        - 接近阻力/追高：0-20分
        - 趋势反转加分：+10分
        - 技术指标综合评分：最多+20分
        """
        if not short_uptrend['valid']:
            base_quality = 50  # 无趋势线，中性评分
        else:
            current_idx = short_data.index[-1]  # 使用DataFrame的实际索引
            support_price = short_uptrend['slope'] * current_idx + short_uptrend['intercept']
            distance_from_support = (current_price - support_price) / support_price

            # 接近支撑（最佳入场）
            if distance_from_support <= 0.03:
                base_quality = 100 - abs(distance_from_support) * 1000
                base_quality = max(80, min(100, base_quality))
            # 距离支撑较远
            elif distance_from_support < 0.05:
                base_quality = 70
            elif distance_from_support < 0.08:
                base_quality = 50
            elif distance_from_support < 0.12:
                base_quality = 30
            else:
                base_quality = 10
        
        # 趋势反转加分
        trend_reversal_bonus = 0
        if trend_reversal:
            trend_reversal_bonus = 10
        
        # 技术指标综合评分
        tech_score = self._calculate_technical_score(
            current_price, short_data, additional_indicators
        )
        
        # 综合评分 = 基础评分 + 技术指标评分 + 趋势反转加分
        quality = base_quality + tech_score + trend_reversal_bonus
        
        # 确保评分在0-100范围内
        return max(0, min(100, quality))

    def _calculate_technical_score(self, current_price: float, data: pd.DataFrame, 
                                   additional_indicators: Dict = None) -> float:
        """
        计算技术指标综合评分
        
        参数:
        - current_price: 当前价格
        - data: 价格数据
        - additional_indicators: 额外的技术指标数据
        
        返回:
        - 技术指标评分 (-20 到 +20)
        """
        if len(data) < 30 or additional_indicators is None:
            return 0  # 数据不足，返回0分
        
        score = 0.0
        latest_idx = -1  # 最新数据索引
        
        # RSI 评分
        if 'rsi' in additional_indicators and not np.isnan(additional_indicators['rsi'][latest_idx]):
            rsi_value = additional_indicators['rsi'][latest_idx]
            if 30 <= rsi_value <= 50:  # RSI 在30-50之间，偏底部但不过度超卖
                score += 8
            elif 50 < rsi_value <= 70:  # RSI 在50-70之间，健康上涨
                score += 5
            elif 20 <= rsi_value < 30:  # 轻度超卖，可能反弹
                score += 3
            elif rsi_value > 70:  # 过于强势，风险增加
                score -= 2
            elif rsi_value < 20:  # 过度超卖，风险增加
                score -= 3

        # MACD 评分
        if ('macd' in additional_indicators and 'macd_signal' in additional_indicators and 
            not (np.isnan(additional_indicators['macd'][latest_idx]) or 
                 np.isnan(additional_indicators['macd_signal'][latest_idx]))):
            macd_val = additional_indicators['macd'][latest_idx]
            signal_val = additional_indicators['macd_signal'][latest_idx]
            hist_val = additional_indicators['macd_hist'][latest_idx]
            
            # MACD金叉或接近金叉
            if macd_val > signal_val and hist_val > 0:  # 金叉
                score += 6
            elif macd_val > signal_val:  # MACD在零轴上方
                score += 4
            elif hist_val > 0 and abs(macd_val - signal_val) < 0.01:  # 接近金叉
                score += 3
            elif macd_val < signal_val and hist_val < 0:  # 死叉下方
                score -= 4

        # 均线排列评分
        if ('ma5' in additional_indicators and 'ma10' in additional_indicators and 
            'ma20' in additional_indicators):
            ma5 = additional_indicators['ma5'][latest_idx]
            ma10 = additional_indicators['ma10'][latest_idx]
            ma20 = additional_indicators['ma20'][latest_idx]
            
            if not (np.isnan(ma5) or np.isnan(ma10) or np.isnan(ma20)):
                # 多头排列
                if ma5 >= ma10 >= ma20:
                    score += 8
                elif ma5 >= ma20 and ma10 >= ma20:  # 接近多头排列
                    score += 5
                elif ma5 < ma10 < ma20:  # 空头排列
                    score -= 6

        # 布林带评分
        if ('bb_upper' in additional_indicators and 'bb_middle' in additional_indicators and 
            'bb_lower' in additional_indicators):
            bb_upper = additional_indicators['bb_upper'][latest_idx]
            bb_middle = additional_indicators['bb_middle'][latest_idx]
            bb_lower = additional_indicators['bb_lower'][latest_idx]
            
            if not (np.isnan(bb_upper) or np.isnan(bb_middle) or np.isnan(bb_lower)):
                # 价格在布林带中轨和下轨之间，偏向底部
                if bb_lower <= current_price <= bb_middle:
                    score += 5
                elif bb_middle < current_price <= bb_middle + (bb_upper - bb_middle) * 0.3:  # 略高于中轨
                    score += 3
                elif current_price > bb_upper:  # 价格过高
                    score -= 5

        # KDJ 评分
        if ('kdj_k' in additional_indicators and 'kdj_d' in additional_indicators):
            k_val = additional_indicators['kdj_k'][latest_idx]
            d_val = additional_indicators['kdj_d'][latest_idx]
            
            if not (np.isnan(k_val) or np.isnan(d_val)):
                if 20 <= k_val <= 50 and k_val > d_val:  # K线在20-50之间且上穿D线
                    score += 6
                elif 20 <= k_val <= 50:  # K线在20-50之间
                    score += 4
                elif k_val > 80 or d_val > 80:  # 超买区
                    score -= 5
                elif k_val < 20 and d_val < 20:  # 超卖区但未见底背离
                    score -= 2

        # 成交量评分
        if ('vol_ratio' in additional_indicators):
            vol_ratio = additional_indicators['vol_ratio'][latest_idx]
            
            if not np.isnan(vol_ratio):
                if 0.8 <= vol_ratio <= 1.5:  # 成交量适中，符合趋势
                    score += 3
                elif 1.5 < vol_ratio <= 2.5:  # 成交量放大，配合上涨
                    score += 2
                elif vol_ratio > 2.5:  # 成交量过度放大，需警惕
                    score -= 2
                elif vol_ratio < 0.5:  # 成交量萎缩，趋势可能减弱
                    score -= 3

        # 限制技术指标评分为-20到+20
        return max(-20, min(20, score))
    
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
            actual_idx = data.index[i]  # 使用DataFrame的实际索引
            expected_price = uptrend_line['slope'] * actual_idx + uptrend_line['intercept']
            actual_close = data['close'].iloc[i]
            if abs(expected_price) > 1e-10:  # 避免除零
                deviation_percent = (actual_close - expected_price) / abs(expected_price)
            else:
                deviation_percent = 0
            if deviation_percent < -TREND_BROKEN_THRESHOLD:
                broken_count += 1
            # print(f"{actual_idx}: {actual_close:.2f} vs {expected_price:.2f} ({deviation_percent:.2%})")
        return broken_count >= 1
    
    def _distance_from_line(self, price: float, line: Dict, idx: int) -> float:
        """计算价格距离趋势线的百分比"""
        if not line['valid']:
            return None
        
        expected_price = line['slope'] * idx + line['intercept']
        return (price - expected_price) / expected_price * 100
    
    def _calculate_trend_strength(self, data: pd.DataFrame, long_uptrend: Dict, short_uptrend: Dict, trend_reversal: bool) -> float:
        """
        计算趋势强度 (0-100)
        
        评估标准：
        1. 长期趋势线斜率（越陡越强）
        2. 短期趋势线斜率
        3. 触点数量（越多越可靠）
        4. 趋势持续时间（越长越稳定）
        5. 价格沿趋势线的一致性
        6. 成交量配合度
        7. 趋势反转加分
        """
        if not long_uptrend['valid']:
            return 0
        
        strength = 0
        
        # 1. 长期趋势线斜率评分（最高25分）
        if long_uptrend['is_rising']:
            daily_slope_pct = (long_uptrend['slope'] / long_uptrend['start_price']) * 100
            
            if 0.05 <= daily_slope_pct <= 0.5:  # 理想范围
                strength += 25
            elif 0.02 <= daily_slope_pct < 0.05:  # 较缓
                strength += 20
            elif 0.5 < daily_slope_pct <= 1.0:  # 较陡
                strength += 22
            elif daily_slope_pct > 0:  # 至少是上升的
                strength += 15
        else:
            # 下降趋势给较低分
            strength += 5
        
        # 2. 短期趋势线斜率评分（最高15分）
        if short_uptrend['valid'] and short_uptrend['is_rising']:
            short_slope_pct = (short_uptrend['slope'] / short_uptrend['start_price']) * 100
            if short_slope_pct > 0.1:
                strength += 15
            elif short_slope_pct > 0:
                strength += 10
        
        # 3. 触点数量（最高20分）
        touches = long_uptrend['touches']
        if touches >= self.long_segments:
            strength += 15
        elif touches <= self.long_segments-1:
            strength += 10
        elif touches <= self.long_segments-2:
            strength += 6
        elif touches <= 0:
            strength += 0
        
        # 4. 趋势持续时间（最高15分）
        duration = long_uptrend['end_idx'] - long_uptrend['start_idx']
        if duration >= 90:  # 3个月以上
            strength += 10
        elif duration >= 60:  # 2个月以上
            strength += 7
        elif duration >= 30:  # 1个月以上
            strength += 4
        elif duration >= 15:  # 半个月以上
            strength += 7
        
        # 5. 价格一致性（最高10分）
        recent_data = data.tail(20)
        above_count = 0
        
        for i in range(len(recent_data)):
            actual_idx = recent_data.index[i]  # 使用DataFrame的实际索引
            expected_price = long_uptrend['slope'] * actual_idx + long_uptrend['intercept']
            actual_price = recent_data.iloc[i]['close']
            
            if actual_price >= expected_price * 0.98:  # 允许2%偏差
                above_count += 1
        
        consistency = above_count / len(recent_data)
        strength += consistency * 10
        
        # 6. 成交量配合（最高10分）
        recent_data = data.tail(20)
        volume_score = 0
        
        for i in range(1, len(recent_data)):
            price_change = recent_data.iloc[i]['close'] - recent_data.iloc[i-1]['close']
            volume_change = recent_data.iloc[i]['volume'] - recent_data.iloc[i-1]['volume']
            
            if price_change > 0 and volume_change > 0:
                volume_score += 1
            elif price_change < 0 and volume_change < 0:
                volume_score += 0.5
        
        strength += (volume_score / len(recent_data)) * 10
        
        # 7. 趋势反转加分（最高5分）
        if trend_reversal:
            strength += 13
        
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
            'short_uptrend_line': {'valid': False},
            'short_downtrend_line': {'valid': False},
            'current_position': 'unknown',
            'entry_quality': 50,
            'broken_support': False,
            'trend_strength': 0,
            'suggested_stop_buffer': 0.03,
            'trend_reversal': False,
            'analysis': {}
        }
