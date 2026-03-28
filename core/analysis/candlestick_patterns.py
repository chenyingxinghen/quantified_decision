"""
K线形态检测模块 - 通用K线形态识别
供所有策略共享使用

包含常见的看涨和看空K线形态：
- 单根K线形态：锤子线、射击之星、十字星等
- 双根K线形态：吞没形态、乌云盖顶、刺透形态等
- 三根K线形态：晨星、暮星、三只乌鸦、三个白兵等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class CandlestickPatterns:
    """K线形态检测器"""
    
    def __init__(self):
        """初始化K线形态检测器"""
        pass
    
    # ==================== 统一检测接口 (基于向量化逻辑) ====================

    def _get_pattern_metadata(self) -> Dict[str, Dict]:
        """获取所有形态的属性映射"""
        return {
            'white_candle': {'description': '阳线', 'score': 2, 'type': 'bullish', 'reliability': 40},
            'black_candle': {'description': '阴线', 'score': 2, 'type': 'bearish', 'reliability': 40},
            'doji': {'description': '十字星', 'score': 10, 'type': 'indecision', 'reliability': 60},
            'hammer': {'description': '锤子线', 'score': 20, 'type': 'bullish_reversal', 'reliability': 75},
            'hanging_man': {'description': '上吊线', 'score': 20, 'type': 'bearish_reversal', 'reliability': 70},
            'shooting_star': {'description': '射击之星', 'score': 35, 'type': 'bearish_reversal', 'reliability': 70},
            'inverted_hammer': {'description': '倒锤线', 'score': 15, 'type': 'bullish_reversal', 'reliability': 65},
            'marubozu': {'description': '光头光脚线', 'score': 15, 'type': 'trend', 'reliability': 65},
            'spinning_top': {'description': '纺锤线', 'score': 8, 'type': 'indecision', 'reliability': 50},
            'bullish_engulfing': {'description': '看涨吞没', 'score': 25, 'type': 'bullish_reversal', 'reliability': 78},
            'bearish_engulfing': {'description': '看跌吞没', 'score': 25, 'type': 'bearish_reversal', 'reliability': 78},
            'piercing_line': {'description': '刺穿线', 'score': 25, 'type': 'bullish_reversal', 'reliability': 75},
            'dark_cloud_cover': {'description': '乌云盖顶', 'score': 25, 'type': 'bearish_reversal', 'reliability': 75},
            'morning_star': {'description': '晨星', 'score': 25, 'type': 'bullish_reversal', 'reliability': 68},
            'evening_star': {'description': '暮星', 'score': 25, 'type': 'bearish_reversal', 'reliability': 68},
            'three_white_soldiers': {'description': '三个白兵', 'score': 30, 'type': 'bullish_reversal', 'reliability': 78},
            'three_black_crows': {'description': '三只乌鸦', 'score': 30, 'type': 'bearish_reversal', 'reliability': 78},
            'harami': {'description': '孕线', 'score': 15, 'type': 'reversal', 'reliability': 60}
        }

    def identify_three_white_soldiers(self, data: pd.DataFrame, context: pd.DataFrame = None) -> np.ndarray:
        """识别三个白兵 (看涨反转)"""
        is_white = data['close'] > data['open']
        c0 = is_white.shift(2)
        c1 = is_white.shift(1)
        c2 = is_white
        ascending = (data['close'].shift(1) > data['close'].shift(2)) & (data['close'] > data['close'].shift(1))
        
        signal = c0 & c1 & c2 & ascending
        
        if context is not None:
            # 强化：处于下降趋势末端或低位
            signal = signal & (context['is_downtrend'] | (context['price_pos'] < 0.3))
            
        return signal.fillna(False).astype(float).values

    def identify_three_black_crows(self, data: pd.DataFrame, context: pd.DataFrame = None) -> np.ndarray:
        """识别三只乌鸦 (看跌反转)"""
        is_black = data['open'] > data['close']
        c0 = is_black.shift(2)
        c1 = is_black.shift(1)
        c2 = is_black
        descending = (data['close'].shift(1) < data['close'].shift(2)) & (data['close'] < data['close'].shift(1))
        
        signal = c0 & c1 & c2 & descending
        
        if context is not None:
            # 强化：处于上升趋势末端或高位
            signal = signal & (context['is_uptrend'] | (context['price_pos'] > 0.7))
            
        return signal.fillna(False).astype(float).values

    def detect_all_bullish_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """检测当前（最后一行）的所有看涨形态"""
        return self._detect_patterns_at_index(data, -1, 'bullish')

    def detect_all_bearish_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """检测当前（最后一行）的所有看跌形态"""
        return self._detect_patterns_at_index(data, -1, 'bearish')

    def detect_all_patterns(self, data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """检测当前（最后一行）的所有形态"""
        return {
            'bullish': self.detect_all_bullish_patterns(data),
            'bearish': self.detect_all_bearish_patterns(data),
            'neutral': self._detect_patterns_at_index(data, -1, 'indecision')
        }

    def _detect_patterns_at_index(self, data: pd.DataFrame, idx: int, filter_type: str) -> List[Dict]:
        """在指定索引处检测特定类型的形态"""
        if len(data) < 3: return []
        
        context = self.calculate_context(data)
        meta_all = self._get_pattern_metadata()
        results = []
        
        # 定义需要检查的方法名列表
        pattern_ids = [
            'hammer', 'inverted_hammer', 'bullish_engulfing', 'piercing_line', 
            'morning_star', 'three_white_soldiers', 'harami', # Bullish
            'shooting_star', 'hanging_man', 'bearish_engulfing', 'dark_cloud_cover', 
            'evening_star', 'three_black_crows', # Bearish
            'doji', 'spinning_top' # Neutral/Indecision
        ]
        
        for pid in pattern_ids:
            meta = meta_all.get(pid, {})
            # 根据类型过滤
            p_type = meta.get('type', '')
            if filter_type == 'bullish' and 'bullish' not in p_type: continue
            if filter_type == 'bearish' and 'bearish' not in p_type: continue
            if filter_type == 'indecision' and p_type != 'indecision': continue

            method = getattr(self, f"identify_{pid}", None)
            if not method: continue
            
            # 部分方法需要 context
            if pid in ['hammer', 'hanging_man', 'shooting_star', 'inverted_hammer', 
                      'bullish_engulfing', 'bearish_engulfing', 'morning_star', 'evening_star']:
                signal_arr = method(data, context)
            else:
                signal_arr = method(data)
            
            if signal_arr[idx] > 0:
                res = meta.copy()
                res['detected'] = True
                results.append(res)
        
        return results

    def scan_patterns_history(self, data: pd.DataFrame, scan_len: int = 90) -> Dict[str, List[Dict]]:
        """
        全量扫描历史形态信号 (高效向量化实现)
        
        Args:
            data: 包含OHLC数据和date列的DataFrame
            scan_len: 扫描最近的天数
        """
        if len(data) < 3: return {'bullish': [], 'bearish': []}
        
        context = self.calculate_context(data)
        meta_all = self._get_pattern_metadata()
        bullish_history = []
        bearish_history = []
        
        # 需要扫描的核心形态
        pattern_ids = [
            'hammer', 'inverted_hammer', 'bullish_engulfing', 'piercing_line', 
            'morning_star', 'three_white_soldiers', 'harami',
            'shooting_star', 'hanging_man', 'bearish_engulfing', 'dark_cloud_cover', 
            'evening_star', 'three_black_crows'
        ]
        
        # 预先计算所有信号矩阵（向量化，只需算一次）
        signals = {}
        for pid in pattern_ids:
            method = getattr(self, f"identify_{pid}", None)
            if not method: continue
            if pid in ['hammer', 'hanging_man', 'shooting_star', 'inverted_hammer', 
                      'bullish_engulfing', 'bearish_engulfing', 'morning_star', 'evening_star']:
                signals[pid] = method(data, context)
            else:
                signals[pid] = method(data)
        
        # 扫描最后 scan_len 条记录
        start_idx = max(0, len(data) - scan_len)
        # 获取日期序列
        dates = data['date'].values
        
        for i in range(start_idx, len(data)):
            date_val = dates[i]
            date_str = date_val.strftime('%Y-%m-%d') if hasattr(date_val, 'strftime') else str(date_val).split('T')[0]
            
            for pid, sig_arr in signals.items():
                if sig_arr[i] > 0:
                    item = meta_all[pid].copy()
                    item['date'] = date_str
                    item['detected'] = True
                    if 'bullish' in item['type']:
                        bullish_history.append(item)
                    elif 'bearish' in item['type']:
                        bearish_history.append(item)
        
        return {
            'bullish': bullish_history,
            'bearish': bearish_history
        }
    
    # ==================== 向量化识别逻辑 (精细识别) ====================
    
    def calculate_context(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        计算价格位置和趋势强度 (向量化)
        
        返回:
            DataFrame 包含 price_pos(0-1), is_uptrend, is_downtrend, is_sideways, range_high, range_low
        """
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 1. 价格位置 (0-1: 0=低位, 1=高位)
        low_min = low.rolling(window).min()
        high_max = high.rolling(window).max()
        # 避免除以零
        denominator = (high_max - low_min).replace(0, 1e-6)
        price_pos = (close - low_min) / denominator
        
        # 2. 趋势方向 (使用更稳健的 MA 组合或线性回归，这里使用多周期 MA 确认)
        ma_short = close.rolling(int(window/2)).mean()
        ma_long = close.rolling(window).mean()
        
        is_uptrend = (close > ma_short) & (ma_short > ma_long)
        is_downtrend = (close < ma_short) & (ma_short < ma_long)
        
        # 3. 波动幅度 (判断是否横盘/窄幅震荡)
        # 使用 ATR 或价格极差的百分比。这里定义：window日内波幅小于 7% 且 价格在 MA 附近波动
        diff_pct = (high_max - low_min) / low_min.replace(0, 1)
        # 辅助判断：价格是否在 ma_long 的上下 2% 范围内
        near_ma = (close / ma_long - 1).abs() < 0.02
        is_sideways = (diff_pct < 0.07) | ((diff_pct < 0.12) & near_ma)
        
        return pd.DataFrame({
            'price_pos': price_pos,
            'is_uptrend': is_uptrend,
            'is_downtrend': is_downtrend,
            'is_sideways': is_sideways,
            'range_high': high_max,
            'range_low': low_min
        }, index=data.index)

    def identify_white_candle(self, data: pd.DataFrame) -> np.ndarray:
        """识别阳线"""
        return (data['close'] > data['open']).astype(float)
    
    def identify_black_candle(self, data: pd.DataFrame) -> np.ndarray:
        """识别阴线"""
        return (data['open'] > data['close']).astype(float)
    
    def identify_doji(self, data: pd.DataFrame, threshold: float = 0.003) -> np.ndarray:
        """识别十字星"""
        body_size = np.abs(data['close'] - data['open'])
        price_scaled_threshold = data['close'] * threshold
        return (body_size < price_scaled_threshold).astype(float)
    
    def identify_hammer(self, data: pd.DataFrame, context: pd.DataFrame, 
                        lower_ratio: float = 2.0, upper_ratio: float = 0.5) -> np.ndarray:
        """
        识别锤子线 (看涨反转)
        符合市场常识：下影线至少是实体的2倍，上影线极短，且处于低位（超跌或趋势末端）
        """
        body = np.abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        # 最小实体保护，避免极小实体导致比例失效
        safe_body = np.where(body < data['close'] * 0.001, data['close'] * 0.001, body)
        
        is_hammer = (
            (lower_shadow > safe_body * lower_ratio) &
            (upper_shadow < safe_body * upper_ratio) &
            (context['price_pos'] < 0.3) & (~context['is_sideways'])
        )
        return is_hammer.astype(float)
    
    def identify_hanging_man(self, data: pd.DataFrame, context: pd.DataFrame, 
                            lower_ratio: float = 2.0, upper_ratio: float = 0.5) -> np.ndarray:
        """
        识别上吊线 (看跌反转)
        符合市场常识：虽然形状像锤子，但出现在高位，预示买盘衰竭
        """
        body = np.abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        safe_body = np.where(body < data['close'] * 0.001, data['close'] * 0.001, body)
        
        is_hanging_man = (
            (lower_shadow > safe_body * lower_ratio) &
            (upper_shadow < safe_body * upper_ratio) &
            (context['price_pos'] > 0.75) & (context['is_uptrend'])
        )
        return is_hanging_man.astype(float)
    
    def identify_shooting_star(self, data: pd.DataFrame, context: pd.DataFrame, 
                               upper_ratio: float = 2.0, lower_ratio: float = 0.5) -> np.ndarray:
        """
        识别射击之星 (看跌反转)
        符合市场常识：长上影线（实体2倍以上），小实体，处于上涨后的高位
        """
        body = np.abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        safe_body = np.where(body < data['close'] * 0.001, data['close'] * 0.001, body)
        
        is_shooting_star = (
            (upper_shadow > safe_body * upper_ratio) &
            (lower_shadow < safe_body * lower_ratio) &
            (context['price_pos'] > 0.75) & (context['is_uptrend'])
        )
        return is_shooting_star.astype(float)
    
    def identify_inverted_hammer(self, data: pd.DataFrame, context: pd.DataFrame, 
                                 upper_ratio: float = 2.0, lower_ratio: float = 0.5) -> np.ndarray:
        """
        识别倒锤子线 (看涨反转)
        符合市场常识：长上影线（实体2倍以上），且处于低位，预示买盘尝试反攻
        """
        body = np.abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        safe_body = np.where(body < data['close'] * 0.001, data['close'] * 0.001, body)
        
        is_inverted_hammer = (
            (upper_shadow > safe_body * upper_ratio) &
            (lower_shadow < safe_body * lower_ratio) &
            (context['price_pos'] < 0.25) & (context['is_downtrend'])
        )
        return is_inverted_hammer.astype(float)
    
    def identify_marubozu(self, data: pd.DataFrame, context: pd.DataFrame = None, threshold_ratio: float = 0.002) -> np.ndarray:
        """
        识别光头光脚线 (趋势持续或横盘突破)
        符合市场常识：无影线或影线极短。
        - 逻辑：如果是横盘期间突破，指导意义极大。
        """
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        threshold = data['close'] * threshold_ratio
        
        body = np.abs(data['close'] - data['open'])
        is_long_body = body > (data['close'] * 0.015) # 实体至少1.5%
        
        is_pure = (lower_shadow < threshold) & (upper_shadow < threshold) & is_long_body
        
        # 增加过滤：横盘突破或趋势中继
        if context is not None:
            # 横盘突破
            is_breakout = context['is_sideways'] & (
                (data['close'] > context['range_high'].shift(1)) | 
                (data['close'] < context['range_low'].shift(1))
            )
            # 趋势中继
            is_continuation = (~context['is_sideways'])
            return (is_pure & (is_breakout | is_continuation)).astype(float)
            
        return is_pure.astype(float)
    
    def identify_spinning_top(self, data: pd.DataFrame) -> np.ndarray:
        """识别纺锤线"""
        body = np.abs(data['close'] - data['open'])
        candle_range = data['high'] - data['low']
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        candle_range = np.where(candle_range == 0, 1, candle_range)
        
        is_spinning_top = (
            (body / candle_range < 0.1) &
            (np.abs(lower_shadow - upper_shadow) / candle_range < 0.3)
        )
        return is_spinning_top.astype(float)
    
    def identify_bullish_engulfing(self, data: pd.DataFrame, context: pd.DataFrame) -> np.ndarray:
        """识别看涨吞没"""
        prev_open = data['open'].shift(1)
        prev_close = data['close'].shift(1)
        curr_open = data['open']
        curr_close = data['close']
        
        prev_is_black = prev_open > prev_close
        curr_is_white = curr_close > curr_open
        
        engulfed = (
            prev_is_black & curr_is_white &
            (curr_close > prev_open * 1.002) & # 显著吞没
            (curr_open < prev_close) &
            (
                ((context['price_pos'] < 0.4) & (~context['is_sideways'])) | # 反转
                (context['is_sideways'] & (curr_close > context['range_high'].shift(1))) # 横盘突破
            )
        )
        return engulfed.fillna(False).astype(float).values
    
    def identify_bearish_engulfing(self, data: pd.DataFrame, context: pd.DataFrame) -> np.ndarray:
        """识别看跌吞没"""
        prev_open = data['open'].shift(1)
        prev_close = data['close'].shift(1)
        curr_open = data['open']
        curr_close = data['close']
        
        prev_is_white = prev_close > prev_open
        curr_is_black = curr_open > curr_close
        
        engulfed = (
            prev_is_white & curr_is_black &
            (curr_close < prev_open * 0.998) & # 显著吞没
            (curr_open > prev_close) &
            (
                ((context['price_pos'] > 0.6) & (~context['is_sideways'])) | # 反转
                (context['is_sideways'] & (curr_close < context['range_low'].shift(1))) # 横盘突破
            )
        )
        return engulfed.fillna(False).astype(float).values
    
    def identify_piercing_line(self, data: pd.DataFrame) -> np.ndarray:
        """识别刺穿线"""
        prev_open = data['open'].shift(1)
        prev_close = data['close'].shift(1)
        curr_open = data['open']
        curr_close = data['close']
        
        prev_is_black = prev_open > prev_close
        curr_is_white = curr_close > curr_open
        midpoint = (prev_open + prev_close) / 2
        
        piercing = (
            prev_is_black & curr_is_white &
            (curr_close > midpoint) &
            (curr_close < prev_open)
        )
        return piercing.fillna(False).astype(float).values
    
    def identify_dark_cloud_cover(self, data: pd.DataFrame) -> np.ndarray:
        """识别乌云盖顶"""
        prev_open = data['open'].shift(1)
        prev_close = data['close'].shift(1)
        curr_open = data['open']
        curr_close = data['close']
        
        prev_is_white = prev_close > prev_open
        curr_is_black = curr_open > curr_close
        midpoint = (prev_open + prev_close) / 2
        
        dark_cloud = (
            prev_is_white & curr_is_black &
            (curr_close < midpoint) &
            (curr_close > prev_open)
        )
        return dark_cloud.fillna(False).astype(float).values
    
    def identify_morning_star(self, data: pd.DataFrame, context: pd.DataFrame) -> np.ndarray:
        """识别晨星"""
        open0 = data['open'].shift(2)
        close0 = data['close'].shift(2)
        open1 = data['open'].shift(1)
        close1 = data['close'].shift(1)
        open2 = data['open']
        close2 = data['close']
        
        first_is_black = open0 > close0
        second_body = np.abs(close1 - open1)
        first_body = np.abs(close0 - open0)
        second_is_small = second_body < first_body * 0.5
        third_is_white = close2 > open2
        midpoint = (open0 + close0) / 2
        
        morning_star = (
            first_is_black & second_is_small & third_is_white & 
            (close2 > midpoint) & (context['price_pos'] < 0.3)
        )
        return morning_star.fillna(False).astype(float).values
    
    def identify_evening_star(self, data: pd.DataFrame, context: pd.DataFrame) -> np.ndarray:
        """识别暮星"""
        open0 = data['open'].shift(2)
        close0 = data['close'].shift(2)
        open1 = data['open'].shift(1)
        close1 = data['close'].shift(1)
        open2 = data['open']
        close2 = data['close']
        
        first_is_white = close0 > open0
        second_body = np.abs(close1 - open1)
        first_body = np.abs(close0 - open0)
        second_is_small = second_body < first_body * 0.5
        third_is_black = open2 > close2
        midpoint = (open0 + close0) / 2
        
        evening_star = (
            first_is_white & second_is_small & third_is_black & 
            (close2 < midpoint) & (context['price_pos'] > 0.7)
        )
        return evening_star.fillna(False).astype(float).values
    
    def identify_harami(self, data: pd.DataFrame) -> np.ndarray:
        """识别孕线"""
        open0 = data['open'].shift(1)
        close0 = data['close'].shift(1)
        open1 = data['open']
        close1 = data['close']
        
        body0 = np.abs(close0 - open0)
        body1 = np.abs(close1 - open1)
        high0 = np.maximum(open0, close0)
        low0 = np.minimum(open0, close0)
        high1 = np.maximum(open1, close1)
        low1 = np.minimum(open1, close1)
        
        harami = (body0 > body1 * 2) & (high1 < high0) & (low1 > low0)
        return harami.fillna(False).astype(float).values
    
    # --- 强度与确认度计算 ---
    
    def get_candle_body_ratio(self, data: pd.DataFrame) -> np.ndarray:
        """计算实体比率"""
        body = np.abs(data['close'] - data['open'])
        candle_range = data['high'] - data['low']
        candle_range = np.where(candle_range == 0, 1, candle_range)
        return body / candle_range
    
    def get_upper_shadow_ratio(self, data: pd.DataFrame) -> np.ndarray:
        """计算上影线比率"""
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        candle_range = data['high'] - data['low']
        candle_range = np.where(candle_range == 0, 1, candle_range)
        return upper_shadow / candle_range
    
    def get_lower_shadow_ratio(self, data: pd.DataFrame) -> np.ndarray:
        """计算下影线比率"""
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        candle_range = data['high'] - data['low']
        candle_range = np.where(candle_range == 0, 1, candle_range)
        return lower_shadow / candle_range
    
    def get_pattern_strength(self, data: pd.DataFrame, window: int = 5) -> np.ndarray:
        """计算形态强度 (rolling consistent direction)"""
        is_white = (data['close'] > data['open']).astype(int)
        is_black = (data['open'] > data['close']).astype(int)
        white_count = is_white.rolling(window=window).sum()
        black_count = is_black.rolling(window=window).sum()
        strength = np.maximum(white_count, black_count) / window
        return strength.fillna(0).values
    
    def get_pattern_confirmation(self, data: pd.DataFrame, window: int = 3) -> np.ndarray:
        """计算形态确认度 (future direction alignment)"""
        direction = np.where(data['close'] > data['open'], 1, -1)
        direction_ser = pd.Series(direction, index=data.index)
        confirm_count = pd.Series(0.0, index=data.index)
        for i in range(1, window):
            next_direction = direction_ser.shift(-i)
            confirm_count += (next_direction == direction_ser).astype(int)
        return (confirm_count / (window - 1)).fillna(0).values

    def get_total_bearish_score(self, data: pd.DataFrame) -> int:
        """获取看跌形态的总分"""
        bearish_patterns = self.detect_all_bearish_patterns(data)
        return sum(pattern['score'] for pattern in bearish_patterns)
    
    def get_total_bullish_score(self, data: pd.DataFrame) -> int:
        """获取看涨形态的总分"""
        bullish_patterns = self.detect_all_bullish_patterns(data)
        return sum(pattern['score'] for pattern in bullish_patterns)
