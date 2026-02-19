"""
K线形态量化因子模块

基于K线形态的量化因子计算，包括：
- 单根K线形态（白线、黑线、十字星、锤子线等）
- 多根K线形态（吞没、刺穿线、乌云盖顶、晨星、暮星等）
- K线形态强度指标

参考资源：
- Candlestick Trading Bible by Honma Munehisa
- 量化交易K线形态识别研究
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from config.factor_config import FactorConfig


class CandlestickPatternFactors:
    """K线形态因子计算类"""
    
    def __init__(self, config: Optional[FactorConfig] = None):
        """初始化K线形态因子计算器"""
        self.pattern_names = []
        self.config = config if config is not None else FactorConfig
    
    # ==================== 单根K线形态 ====================
    
    def calculate_white_candle(self, data: pd.DataFrame) -> np.ndarray:
        """
        白线/阳线：收盘价 > 开盘价
        表示上升动力，看涨信号
        """
        return (data['close'] > data['open']).astype(float)
    
    def calculate_black_candle(self, data: pd.DataFrame) -> np.ndarray:
        """
        黑线/阴线：开盘价 > 收盘价
        表示下降动力，看跌信号
        """
        return (data['open'] > data['close']).astype(float)
    
    def calculate_doji(self, data: pd.DataFrame, threshold: float = 0.001) -> np.ndarray:
        """
        十字星：开盘价 ≈ 收盘价（差异 < 阈值）
        表示市场犹豫不决，可能反转
        
        Args:
            threshold: 开收价差异阈值（可选，如未提供则使用配置中的小实体阈值）
        """
        if threshold == 0.001:  # 默认值
            threshold = self.config.BODY_SIZE_THRESHOLD_SMALL
        
        body_size = np.abs(data['close'] - data['open'])
        price_scaled_threshold = data['close'] * threshold
        
        return (body_size < price_scaled_threshold).astype(float)
    
    def calculate_hammer(self, data: pd.DataFrame) -> np.ndarray:
        """
        锤子线：下影线长，上影线短，实体小
        看涨反转信号，通常出现在下跌后
        
        条件：
        - 下影线 > 实体高度 * 2
        条件：
        - 下影线 > 实体高度 * 2
        - 上影线 < 实体高度 * 1
        - 实体在上半部分
        """
        body = np.abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        # 避免除以零
        body = np.where(body == 0, 0.001, body)
        
        # 使用配置中的阈值（如果有定义），否则使用默认倍数
        lower_shadow_ratio = getattr(self.config, 'HAMMER_LOWER_SHADOW_RATIO', 2.0)
        upper_shadow_ratio = getattr(self.config, 'HAMMER_UPPER_SHADOW_RATIO', 1.0)
        
        is_hammer = (
            (lower_shadow > body * lower_shadow_ratio) &
            (upper_shadow < body * upper_shadow_ratio) &
            (data['close'] > data['open'])  # 收盘价 > 开盘价
        )
        
        return is_hammer.astype(float)
    
    def calculate_hanging_man(self, data: pd.DataFrame) -> np.ndarray:
        """
        上吊线：下影线长，上影线短，实体小
        看跌反转信号，通常出现在上升后
        
        条件：
        - 下影线 > 实体高度 * 2
        条件：
        - 下影线 > 实体高度 * 2
        - 上影线 < 实体高度 * 1
        - 实体在上半部分
        - 收盘价 < 开盘价（黑线）
        """
        body = np.abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        body = np.where(body == 0, 0.001, body)
        
        lower_shadow_ratio = getattr(self.config, 'HAMMER_LOWER_SHADOW_RATIO', 2.0)
        upper_shadow_ratio = getattr(self.config, 'HAMMER_UPPER_SHADOW_RATIO', 1.0)
        
        is_hanging_man = (
            (lower_shadow > body * lower_shadow_ratio) &
            (upper_shadow < body * upper_shadow_ratio) &
            (data['close'] < data['open'])  # 收盘价 < 开盘价
        )
        
        return is_hanging_man.astype(float)
    
    def calculate_shooting_star(self, data: pd.DataFrame) -> np.ndarray:
        """
        射击之星：上影线长，下影线短，实体小
        看跌反转信号，通常出现在上升后
        
        条件：
        - 上影线 > 实体高度 * 2
        条件：
        - 上影线 > 实体高度 * 2
        - 下影线 < 实体高度 * 1
        - 实体在下半部分
        """
        body = np.abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        body = np.where(body == 0, 0.001, body)
        
        lower_shadow_ratio = getattr(self.config, 'HAMMER_LOWER_SHADOW_RATIO', 2.0)
        upper_shadow_ratio = getattr(self.config, 'HAMMER_UPPER_SHADOW_RATIO', 1.0)
        
        is_shooting_star = (
            (upper_shadow > body * lower_shadow_ratio) &
            (lower_shadow < body * upper_shadow_ratio) &
            (data['close'] < data['open'])
        )
        
        return is_shooting_star.astype(float)
    
    def calculate_inverted_hammer(self, data: pd.DataFrame) -> np.ndarray:
        """
        倒锤线：上影线长，下影线短，实体小
        看涨反转信号，通常出现在下跌后
        
        条件：
        - 上影线 > 实体高度 * 2
        条件：
        - 上影线 > 实体高度 * 2
        - 下影线 < 实体高度 * 1
        - 实体在下半部分
        - 收盘价 > 开盘价（白线）
        """
        body = np.abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        body = np.where(body == 0, 0.001, body)
        
        lower_shadow_ratio = getattr(self.config, 'HAMMER_LOWER_SHADOW_RATIO', 2.0)
        upper_shadow_ratio = getattr(self.config, 'HAMMER_UPPER_SHADOW_RATIO', 1.0)
        
        is_inverted_hammer = (
            (upper_shadow > body * lower_shadow_ratio) &
            (lower_shadow < body * upper_shadow_ratio) &
            (data['close'] > data['open'])
        )
        
        return is_inverted_hammer.astype(float)
    
    def calculate_marubozu(self, data: pd.DataFrame) -> np.ndarray:
        """
        光头光脚线：无上下影线，实体完整
        表示强势趋势
        
        条件：
        - 上影线 ≈ 0
        - 下影线 ≈ 0
        """
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        # 使用收盘价比例作为阈值
        threshold = data['close'] * self.config.BODY_SIZE_THRESHOLD_SMALL
        is_marubozu = (lower_shadow < threshold) & (upper_shadow < threshold)
        
        return is_marubozu.astype(float)
    
    def calculate_spinning_top(self, data: pd.DataFrame) -> np.ndarray:
        """
        纺锤线：实体很小，上下影线相近
        表示市场犹豫，可能反转
        """
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
    
    # ==================== 多根K线形态 ====================
    
    def calculate_bullish_engulfing(self, data: pd.DataFrame) -> np.ndarray:
        """
        看涨吞没：前一根黑线被后一根白线完全吞没
        看涨反转信号
        """
        prev_open = data['open'].shift(1)
        prev_close = data['close'].shift(1)
        curr_open = data['open']
        curr_close = data['close']
        
        prev_is_black = prev_open > prev_close
        curr_is_white = curr_close > curr_open
        
        engulfed = (
            prev_is_black & curr_is_white &
            (curr_close > prev_open) &
            (curr_open < prev_close)
        )
        
        return engulfed.fillna(False).astype(float).values
    
    def calculate_bearish_engulfing(self, data: pd.DataFrame) -> np.ndarray:
        """
        看跌吞没：前一根白线被后一根黑线完全吞没
        看跌反转信号
        """
        prev_open = data['open'].shift(1)
        prev_close = data['close'].shift(1)
        curr_open = data['open']
        curr_close = data['close']
        
        prev_is_white = prev_close > prev_open
        curr_is_black = curr_open > curr_close
        
        engulfed = (
            prev_is_white & curr_is_black &
            (curr_close < prev_open) &
            (curr_open > prev_close)
        )
        
        return engulfed.fillna(False).astype(float).values
    
    def calculate_piercing_line(self, data: pd.DataFrame) -> np.ndarray:
        """
        刺穿线：前一根黑线，后一根白线穿过前一根的中点
        看涨反转信号
        """
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
    
    def calculate_dark_cloud_cover(self, data: pd.DataFrame) -> np.ndarray:
        """
        乌云盖顶：前一根白线，后一根黑线穿过前一根的中点
        看跌反转信号
        """
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
    
    def calculate_morning_star(self, data: pd.DataFrame) -> np.ndarray:
        """
        晨星：三根K线形态
        """
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
        third_above_midpoint = close2 > midpoint
        
        morning_star = first_is_black & second_is_small & third_is_white & third_above_midpoint
        return morning_star.fillna(False).astype(float).values
    
    def calculate_evening_star(self, data: pd.DataFrame) -> np.ndarray:
        """
        暮星：三根K线形态
        """
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
        third_below_midpoint = close2 < midpoint
        
        evening_star = first_is_white & second_is_small & third_is_black & third_below_midpoint
        return evening_star.fillna(False).astype(float).values
    
    def calculate_harami(self, data: pd.DataFrame) -> np.ndarray:
        """
        孕线：前一根大实体，后一根小实体完全在前一根内部
        """
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
    
    # ==================== K线形态强度指标 ====================
    
    def calculate_candle_body_ratio(self, data: pd.DataFrame) -> np.ndarray:
        """
        K线实体比率：实体 / 整个K线范围
        值越大表示趋势越强
        """
        body = np.abs(data['close'] - data['open'])
        candle_range = data['high'] - data['low']
        
        candle_range = np.where(candle_range == 0, 1, candle_range)
        
        return body / candle_range
    
    def calculate_upper_shadow_ratio(self, data: pd.DataFrame) -> np.ndarray:
        """
        上影线比率：上影线 / 整个K线范围
        值越大表示上方压力越大
        """
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        candle_range = data['high'] - data['low']
        
        candle_range = np.where(candle_range == 0, 1, candle_range)
        
        return upper_shadow / candle_range
    
    def calculate_lower_shadow_ratio(self, data: pd.DataFrame) -> np.ndarray:
        """
        下影线比率：下影线 / 整个K线范围
        值越大表示下方支撑越大
        """
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        candle_range = data['high'] - data['low']
        
        candle_range = np.where(candle_range == 0, 1, candle_range)
        
        return lower_shadow / candle_range
    
    def calculate_pattern_strength(self, data: pd.DataFrame, window: int = 5) -> np.ndarray:
        """
        形态强度指标：计算最近N根K线的形态一致性
        """
        is_white = (data['close'] > data['open']).astype(int)
        is_black = (data['open'] > data['close']).astype(int)
        
        white_count = is_white.rolling(window=window).sum()
        black_count = is_black.rolling(window=window).sum()
        
        strength = np.maximum(white_count, black_count) / window
        return strength.fillna(0).values
    
    def calculate_pattern_confirmation(self, data: pd.DataFrame, window: int = 3) -> np.ndarray:
        """
        形态确认度：检查形态是否被后续K线确认
        """
        direction = np.where(data['close'] > data['open'], 1, -1)
        direction_ser = pd.Series(direction, index=data.index)
        
        # 计算后续 N-1 根线的方向一致性
        confirm_count = pd.Series(0, index=data.index)
        for i in range(1, window):
            # 将后续方向向前移动，与当前位置对比
            next_direction = direction_ser.shift(-i)
            confirm_count += (next_direction == direction_ser).astype(int)
            
        return (confirm_count / (window - 1)).fillna(0).values
    
    def calculate_all_candlestick_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有K线形态因子
        
        Args:
            data: 包含OHLC数据的DataFrame
        
        Returns:
            包含所有K线形态因子的DataFrame
        """
        factors = pd.DataFrame(index=data.index)
        
        # 单根K线形态
        factors['white_candle'] = self.calculate_white_candle(data)
        factors['black_candle'] = self.calculate_black_candle(data)
        factors['doji'] = self.calculate_doji(data)
        factors['hammer'] = self.calculate_hammer(data)
        factors['hanging_man'] = self.calculate_hanging_man(data)
        factors['shooting_star'] = self.calculate_shooting_star(data)
        factors['inverted_hammer'] = self.calculate_inverted_hammer(data)
        factors['marubozu'] = self.calculate_marubozu(data)
        factors['spinning_top'] = self.calculate_spinning_top(data)
        
        # 多根K线形态
        factors['bullish_engulfing'] = self.calculate_bullish_engulfing(data)
        factors['bearish_engulfing'] = self.calculate_bearish_engulfing(data)
        factors['piercing_line'] = self.calculate_piercing_line(data)
        factors['dark_cloud_cover'] = self.calculate_dark_cloud_cover(data)
        factors['morning_star'] = self.calculate_morning_star(data)
        factors['evening_star'] = self.calculate_evening_star(data)
        factors['harami'] = self.calculate_harami(data)
        
        # K线形态强度指标
        factors['candle_body_ratio'] = self.calculate_candle_body_ratio(data)
        factors['upper_shadow_ratio'] = self.calculate_upper_shadow_ratio(data)
        factors['lower_shadow_ratio'] = self.calculate_lower_shadow_ratio(data)
        factors['pattern_strength'] = self.calculate_pattern_strength(data)
        factors['pattern_confirmation'] = self.calculate_pattern_confirmation(data)
        
        return factors
    
    def get_pattern_names(self) -> List[str]:
        """获取所有K线形态因子名称"""
        return [
            # 单根K线形态
            'white_candle', 'black_candle', 'doji', 'hammer', 'hanging_man',
            'shooting_star', 'inverted_hammer', 'marubozu', 'spinning_top',
            # 多根K线形态
            'bullish_engulfing', 'bearish_engulfing', 'piercing_line',
            'dark_cloud_cover', 'morning_star', 'evening_star', 'harami',
            # 强度指标
            'candle_body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
            'pattern_strength', 'pattern_confirmation'
        ]
