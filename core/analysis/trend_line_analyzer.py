# 趋势线分析器 - 用于识别支撑/阻力趋势线并判断入场时机
# 解决"买在高点"问题的核心模块
#
# 核心设计原则：
# 1. 斜率/截距基于日期（转为天数偏移量），而非DataFrame索引
# 2. 使用两点法枚举所有点对，选触点最多且不穿越K线的趋势线
# 3. 长期趋势线：触点最多优先，近期低点连接的线更重要
# 4. 短期趋势线：优先分段法，远端时间点更重要（辅助逃顶卖出）

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from config.strategy_config import TREND_LINE_LONG_PERIOD, TREND_LINE_SHORT_PERIOD, \
    TREND_BROKEN_THRESHOLD, SWING_LONG_WINDOW, SWING_SHORT_WINDOW, MIN_SWING_POINTS, \
    TOUCH_TOLERANCE, MIN_TOUCHES


def _date_to_day_number(date_index: pd.Index) -> np.ndarray:
    """
    将日期索引转换为天数偏移量（相对于首个日期的天数）
    
    这是斜率/截距计算的基础，确保斜率单位是 "价格/天"
    """
    if isinstance(date_index, pd.DatetimeIndex):
        base = date_index[0]
        return (date_index - base).days.astype(float)
    
    # 如果是纯数值索引（RangeIndex / Int64Index），直接兜底
    if isinstance(date_index, pd.RangeIndex) or pd.api.types.is_integer_dtype(date_index):
        return np.arange(len(date_index), dtype=float)
    
    # 尝试将字符串等转为datetime
    try:
        dt_index = pd.to_datetime(date_index)
        base = dt_index[0]
        return (dt_index - base).days.astype(float)
    except Exception:
        # 最终兜底：使用0到N-1的序列
        return np.arange(len(date_index), dtype=float)


def _single_date_to_day_number(date_val, base_date) -> float:
    """将单个日期转换为相对于base_date的天数偏移量"""
    try:
        if not isinstance(date_val, pd.Timestamp):
            date_val = pd.Timestamp(date_val)
        if not isinstance(base_date, pd.Timestamp):
            base_date = pd.Timestamp(base_date)
        return float((date_val - base_date).days)
    except Exception:
        return 0.0


class TrendLineAnalyzer:
    """
    趋势线分析器（两点法 + 日期数值化版本）
    
    核心功能：
    1. 识别摆动低点和摆动高点（局部极值）
    2. 通过两点法枚举候选趋势线，选择不穿越K线且触点最多的
    3. 长期趋势线：触点最多优先，近期低点连接的线更重要
    4. 短期趋势线：优先分段法，远端时间点更重要（辅助逃顶）
    5. 检测是否跌破趋势线（看空信号）
    
    关键特性：
    - 斜率单位为 "价格/天"，截距为基准日期处的价格
    - 两点法确保趋势线不穿越K线（支撑线在所有K线下方/阻力线在所有K线上方）
    - 长期和短期分别使用不同策略
    """
    
    def __init__(self, long_period: int = TREND_LINE_LONG_PERIOD, short_period: int = TREND_LINE_SHORT_PERIOD):
        """
        初始化趋势线分析器
        
        参数:
            long_period: 长期回溯天数，默认从配置文件读取
            short_period: 短期回溯天数，默认从配置文件读取
        """
        self.long_period = long_period
        self.short_period = short_period
        self.min_touches = MIN_TOUCHES
        self.touch_tolerance = TOUCH_TOLERANCE
        
        # 摆动点识别参数
        self.long_swing_window = SWING_LONG_WINDOW
        self.short_swing_window = SWING_SHORT_WINDOW
        self.min_swing_points = MIN_SWING_POINTS
    
    @staticmethod
    def _ensure_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
        """
        确保 DataFrame 使用 DatetimeIndex
        
        如果 DataFrame 有 'date' 列但索引不是 DatetimeIndex，
        将 'date' 列设为索引并转为 DatetimeIndex。
        这样斜率/截距才能正确基于日期计算。
        """
        if isinstance(data.index, pd.DatetimeIndex):
            return data
        
        if 'date' in data.columns:
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
            return data
        
        # 无法转换，原样返回（兜底到 _date_to_day_number 的 RangeIndex 处理）
        return data
    
    def analyze(self, data: pd.DataFrame) -> Dict:
        """
        趋势线分析（针对短期上升/下降及长期上升/下降）
        
        返回:
        {
            'uptrend_line': {...},        # 长期上升趋势线
            'downtrend_line': {...},      # 长期下降趋势线
            'short_uptrend_line': {...},  # 短期上升趋势线
            'short_downtrend_line': {...},# 短期下降趋势线
            'broken_support': bool,       # 是否跌破短期支撑
        }
        """
        if len(data) < 20:
            return self._empty_result()
        
        # 确保使用 DatetimeIndex（处理SQL加载的数据有'date'列但整数索引的情况）
        data = self._ensure_datetime_index(data)
        
        # 长期趋势线分析
        long_data = data.tail(min(self.long_period, len(data)))
        long_uptrend = self._find_long_trendline(long_data, 'low')
        long_downtrend = self._find_long_trendline(long_data, 'high')
        
        # 短期趋势线分析
        short_data = data.tail(min(self.short_period, len(data)))
        short_uptrend = self._find_short_trendline(short_data, 'low')
        short_downtrend = self._find_short_trendline(short_data, 'high')
        
        # 检测是否跌破支撑
        broken_support = self._check_broken_support(data, short_uptrend)
        
        return {
            'uptrend_line': long_uptrend,
            'downtrend_line': long_downtrend,
            'short_uptrend_line': short_uptrend,
            'short_downtrend_line': short_downtrend,
            'broken_support': broken_support
        }
    
    # ========================================================================
    # 摆动点识别
    # ========================================================================
    
    def _find_swing_lows(self, data: pd.DataFrame, window: int = None) -> List[Tuple]:
        """
        识别摆动低点（局部最低点）
        
        返回:
            [(date, price, day_number), ...] 摆动低点列表
            其中 day_number 是相对于 data 首行的天数偏移
        """
        if window is None:
            window = self.long_swing_window
        
        day_numbers = _date_to_day_number(data.index)
        base_date = data.index[0]
        swing_lows = []
        
        for i in range(window, len(data) - window):
            current_low = data['low'].iloc[i]
            
            is_swing_low = True
            
            for j in range(i - window, i):
                if data['low'].iloc[j] <= current_low:
                    is_swing_low = False
                    break
            
            if not is_swing_low:
                continue
            
            for j in range(i + 1, i + window + 1):
                if data['low'].iloc[j] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append((data.index[i], current_low, day_numbers[i]))
        
        return swing_lows
    
    def _find_swing_highs(self, data: pd.DataFrame, window: int = None) -> List[Tuple]:
        """
        识别摆动高点（局部最高点）
        
        返回:
            [(date, price, day_number), ...] 摆动高点列表
        """
        if window is None:
            window = self.long_swing_window
        
        day_numbers = _date_to_day_number(data.index)
        swing_highs = []
        
        for i in range(window, len(data) - window):
            current_high = data['high'].iloc[i]
            
            is_swing_high = True
            
            for j in range(i - window, i):
                if data['high'].iloc[j] >= current_high:
                    is_swing_high = False
                    break
            
            if not is_swing_high:
                continue
            
            for j in range(i + 1, i + window + 1):
                if data['high'].iloc[j] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append((data.index[i], current_high, day_numbers[i]))
        
        return swing_highs
    
    # ========================================================================
    # 分段法获取关键点
    # ========================================================================
    
    def _find_segment_points(self, data: pd.DataFrame, point_type: str, 
                             num_segments: int = 4) -> List[Tuple]:
        """
        通过分段找极值点
        
        参数:
            data: 价格数据
            point_type: 'low' 或 'high'
            num_segments: 分段数量
            
        返回:
            [(date, price, day_number), ...] 分段极值点列表
        """
        if len(data) < num_segments:
            return []
        
        day_numbers = _date_to_day_number(data.index)
        segment_points = []
        segment_size = len(data) / num_segments
        
        col = 'low' if point_type == 'low' else 'high'
        
        for i in range(num_segments):
            start_pos = int(i * segment_size)
            end_pos = int((i + 1) * segment_size)
            if i == num_segments - 1:
                end_pos = len(data)
            
            segment_data = data.iloc[start_pos:end_pos]
            if len(segment_data) == 0:
                continue
            
            if point_type == 'low':
                ext_idx = segment_data[col].idxmin()
            else:
                ext_idx = segment_data[col].idxmax()
            
            ext_price = segment_data.loc[ext_idx, col]
            
            # 找到该日期在data中的位置以获取对应的day_number
            pos_in_data = data.index.get_loc(ext_idx)
            d_num = day_numbers[pos_in_data]
            
            segment_points.append((ext_idx, ext_price, d_num))
        
        return segment_points
    
    # ========================================================================
    # 长期趋势线（触点最多优先，近期低点更重要）
    # ========================================================================
    
    def _find_long_trendline(self, data: pd.DataFrame, point_type: str) -> Dict:
        """
        寻找长期趋势线
        
        策略：
        1. 先寻找摆动点
        2. 若摆动点不足(< min_swing_points)，则使用分段法获取关键点
        3. 两点法枚举所有候选趋势线
        4. 评分标准：触点数 * 权重 + 近期优先加分
        """
        if point_type == 'low':
            swing_points = self._find_swing_lows(data, window=self.long_swing_window)
        else:
            swing_points = self._find_swing_highs(data, window=self.long_swing_window)
        
        # 如果摆动点不足，补充分段法的点
        if len(swing_points) < self.min_swing_points:
            segment_points = self._find_segment_points(data, point_type, num_segments=4)
            # 合并并去重（按日期）
            existing_dates = {p[0] for p in swing_points}
            for sp in segment_points:
                if sp[0] not in existing_dates:
                    swing_points.append(sp)
                    existing_dates.add(sp[0])
        
        if len(swing_points) < 2:
            return {'valid': False}
        
        # 按day_number排序
        swing_points.sort(key=lambda x: x[2])
        
        # 使用两点法枚举最优趋势线
        return self._find_best_trendline_two_point(
            swing_points, data, point_type, prefer_recent=True, trend_type='long'
        )
    
    # ========================================================================
    # 短期趋势线（优先分段法，远端时间更重要 → 辅助逃顶）
    # ========================================================================
    
    def _find_short_trendline(self, data: pd.DataFrame, point_type: str) -> Dict:
        """
        寻找短期趋势线
        
        策略：
        1. 优先使用分段法（因为短窗口内摆动点往往不足）
        2. 如果分段法找到 >= 2 个点，直接使用
        3. 否则尝试摆动点（降低窗口）
        4. 评分标准：远端点（离现在更远）更重要
        """
        # 优先分段法
        segment_points = self._find_segment_points(data, point_type, num_segments=4)
        
        candidate_points = segment_points
        
        # 如果分段点不够，补充摆动点
        if len(candidate_points) < 2:
            if point_type == 'low':
                swing_points = self._find_swing_lows(data, window=self.short_swing_window)
            else:
                swing_points = self._find_swing_highs(data, window=self.short_swing_window)
            
            existing_dates = {p[0] for p in candidate_points}
            for sp in swing_points:
                if sp[0] not in existing_dates:
                    candidate_points.append(sp)
                    existing_dates.add(sp[0])
        
        if len(candidate_points) < 2:
            return {'valid': False}
        
        # 按day_number排序
        candidate_points.sort(key=lambda x: x[2])
        
        # 短期趋势线：远端时间更重要（prefer_recent=False）
        return self._find_best_trendline_two_point(
            candidate_points, data, point_type, prefer_recent=False ,trend_type='short'
        )
    
    # ========================================================================
    # 两点法核心：枚举所有点对，选最优趋势线
    # ========================================================================
    
    def _find_best_trendline_two_point(self, points: List[Tuple], 
                                        data: pd.DataFrame, 
                                        point_type: str,
                                        prefer_recent: bool = True,
                                        trend_type: str = 'short') -> Dict:
        """
        两点法枚举所有候选趋势线，选触点最多且不穿越K线的最优线
        
        参数:
            points: [(date, price, day_number), ...] 关键点列表（已按day_number排序）
            data: 原始OHLC数据
            point_type: 'low'（支撑线） 或 'high'（阻力线）
            prefer_recent: True=近期低点连接的线更优（长期用），
                          False=远端时间更重要（短期用）
        
        返回:
            趋势线信息字典
        """
        if len(points) < 2:
            return {'valid': False}
        
        # 预计算data中每个bar的day_number
        day_numbers = _date_to_day_number(data.index)
        base_date = data.index[0]
        
        if point_type == 'low':
            bar_prices = data['low'].values
        else:
            bar_prices = data['high'].values
        
        best_line = None
        best_score = -float('inf')
        
        n = len(points)
        slopes=[]
        intercepts=[]
        # 枚举所有两点组合
        for i in range(n):
            for j in range(i + 1, n):
                date_i, price_i, day_i = points[i]
                date_j, price_j, day_j = points[j]
                
                # 两点不能重合（天数一样）
                if abs(day_j - day_i) < 1e-6:
                    continue
                
                # 计算斜率和截距（基于天数偏移量）
                slope = (price_j - price_i) / (day_j - day_i)
                intercept = price_i - slope * day_i
                slopes.append(slope)
                intercepts.append(intercept)
                
                # 检查是否穿越K线
                if not self._check_no_crossing(slope, intercept, day_numbers, 
                                                bar_prices, point_type):
                    continue
                
                # 计算触点数（在points中，趋势线值与point价格的偏差在容差内的数量）
                touches = self._count_touches_two_point(
                    points, slope, intercept
                )
                
                if touches < self.min_touches:
                    continue
                
                # 计算评分
                score = self._score_trendline(
                    slope, intercept, points[i], points[j],
                    touches, data, day_numbers, point_type, prefer_recent
                )
                if score > best_score:
                    best_score = score
                    best_line = self._build_line_dict(
                        slope, intercept, points[i], points[j],
                        touches, score, point_type, base_date)
        
        if best_line is None:
            return {'valid': False}
        
        return best_line
    
    def _check_no_crossing(self, slope: float, intercept: float,
                           day_numbers: np.ndarray, bar_prices: np.ndarray,
                           point_type: str) -> bool:
        """
        检查趋势线是否穿越K线
        
        对于支撑线(low)：趋势线值应该 <= 所有K线的low（趋势线在K线下方）
        对于阻力线(high)：趋势线值应该 >= 所有K线的high（趋势线在K线上方）
        
        允许小幅穿越（容差范围内），避免因微小偏差导致没有有效趋势线
        """
        trendline_values = slope * day_numbers + intercept
        
        if point_type == 'low':
            # 支撑线：不应高于K线low太多
            violations = trendline_values - bar_prices
            # 允许容差内的轻微穿越
            max_violation = np.max(violations)
            if max_violation <= 0:
                return True
            # 检查穿越幅度是否在容差范围内
            avg_price = np.mean(bar_prices)
            if avg_price > 0 and (max_violation / avg_price) <= self.touch_tolerance:
                return True
            return False
        else:
            # 阻力线：不应低于K线high太多
            violations = bar_prices - trendline_values
            max_violation = np.max(violations)
            if max_violation <= 0:
                return True
            avg_price = np.mean(bar_prices)
            if avg_price > 0 and (max_violation / avg_price) <= self.touch_tolerance:
                return True
            return False
    
    def _count_touches_two_point(self, points: List[Tuple], 
                                  slope: float, intercept: float) -> int:
        """
        统计有多少关键点在趋势线容差范围内（触点数）
        
        参数:
            points: [(date, price, day_number), ...]
            slope: 趋势线斜率
            intercept: 趋势线截距
            
        返回:
            触点数量
        """
        touches = 0
        for _, price, day_num in points:
            expected_price = slope * day_num + intercept
            if abs(expected_price) < 1e-10:
                continue
            deviation = abs(price - expected_price) / abs(expected_price)
            if deviation <= self.touch_tolerance:
                touches += 1
        return touches
    
    def _score_trendline(self, slope: float, intercept: float,
                         point_start: Tuple, point_end: Tuple,
                         touches: int, data: pd.DataFrame,
                         day_numbers: np.ndarray,
                         point_type: str, prefer_recent: bool) -> float:
        """
        计算趋势线的综合评分
        
        评分维度:
        1. 触点数（核心指标）
        2. 跨度（趋势线覆盖的时间范围）
        3. 近期/远期偏好加分
        4. 斜率合理性
        """
        score = 0.0
        
        # 1. 触点数得分 (最重要，权重最高)
        score += touches * 100
        
        # 2. 跨度得分：趋势线跨越的时间越长越好
        span_days = point_end[2] - point_start[2]
        total_days = day_numbers[-1] - day_numbers[0] if len(day_numbers) > 1 else 1
        span_ratio = span_days / total_days if total_days > 0 else 0
        score += span_ratio * 50
        
        # 3. 近期/远期偏好
        last_day = day_numbers[-1]
        
        if prefer_recent:
            # 长期趋势线：由更近日期中的低/高点连接成的趋势线更重要
            # 结束点越近越好
            end_recency = 1.0 - (last_day - point_end[2]) / total_days if total_days > 0 else 0
            score += end_recency * 80
            
            # 如果两个锚点都在后半部分，额外加分
            mid_day = (day_numbers[0] + day_numbers[-1]) / 2 
            if point_start[2] >= mid_day:
                score += 30
        else:
            # 短期趋势线：远端时间（离当前更远的起点）更重要，有助于逃顶
            # 起始点越早越好（覆盖更长的趋势）
            start_earliness = (point_start[2] - day_numbers[0]) / total_days if total_days > 0 else 0
            # start_earliness 越小（越靠近Day 0），越好
            score += (1.0 - start_earliness) * 80
            
            # 跨度长的短期趋势线更可靠
            score += span_ratio * 30
        
        # 4. 斜率合理性
        start_price = point_start[1]
        if start_price > 0 and span_days > 0:
            daily_change_pct = abs(slope / start_price) * 100
            
            if 0.05 <= daily_change_pct <= 0.5:
                score += 20
            elif 0.02 <= daily_change_pct < 0.05:
                score += 15
            elif 0.5 < daily_change_pct <= 1.0:
                score += 12
            elif daily_change_pct > 0:
                score += 5
        
        return score
    
    def _build_line_dict(self, slope: float, intercept: float,
                         point_start: Tuple, point_end: Tuple,
                         touches: int, score: float, 
                         point_type: str, base_date) -> Dict:
        """
        构建趋势线信息字典
        
        注意：
        - slope: 价格/天 的斜率
        - intercept: 基准日期（base_date）处的截距
        - base_date: data.index[0]，用于将任意日期转为day_number后计算趋势线值
        """
        return {
            'valid': True,
            'slope': slope,                    # 价格/天
            'intercept': intercept,            # 基准日期处的截距
            'base_date': base_date,            # 基准日期
            'touches': touches,
            'start_date': point_start[0],      # 起始锚点日期
            'start_price': point_start[1],     # 起始锚点价格
            'start_day': point_start[2],       # 起始锚点天数偏移
            'end_date': point_end[0],          # 结束锚点日期
            'end_price': point_end[1],         # 结束锚点价格
            'end_day': point_end[2],           # 结束锚点天数偏移
            'is_rising': slope > 0,
            'point_type': point_type,
            'score': score,
            'angle_degrees': np.degrees(np.arctan(slope)),
        }

    def _get_trendline_value(self, line: Dict, date_val) -> float:
        """
        计算给定日期在趋势线上的价格值
        
        参数:
            line: 趋势线字典（需包含 slope, intercept, base_date）
            date_val: 要计算的日期
            
        返回:
            趋势线在该日期处的价格
        """
        day_num = _single_date_to_day_number(date_val, line['base_date'])
        return line['slope'] * day_num + line['intercept']

    # ========================================================================
    # 跌破支撑检测
    # ========================================================================
    
    def _check_broken_support(self, data: pd.DataFrame, uptrend_line: Dict) -> bool:
        """
        检测是否跌破支撑趋势线
        
        条件：
        1. 收盘价跌破趋势线超过阈值
        2. 最近3根K线中至少有1根跌破
        """
        if not uptrend_line.get('valid', False):
            return False
        
        if len(data) < 3:
            return False
        
        broken_count = 0
        base_date = uptrend_line['base_date']
        slope = uptrend_line['slope']
        intercept = uptrend_line['intercept']
        
        for i in range(len(data) - 3, len(data)):
            date_val = data.index[i]
            day_num = _single_date_to_day_number(date_val, base_date)
            expected_price = slope * day_num + intercept
            actual_close = data['close'].iloc[i]
            
            if abs(expected_price) > 1e-10:
                deviation_percent = (actual_close - expected_price) / abs(expected_price)
            else:
                deviation_percent = 0
            
            if deviation_percent < -TREND_BROKEN_THRESHOLD:
                broken_count += 1
                
        return broken_count >= 1


    def _empty_result(self) -> Dict:
        """返回空结果"""
        return {
            'uptrend_line': {'valid': False},
            'downtrend_line': {'valid': False},
            'short_uptrend_line': {'valid': False},
            'short_downtrend_line': {'valid': False},
            'broken_support': False
        }
