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
        
        条件：
        - 前一根：黑线（close < open）
        - 后一根：白线（close > open）
        - 后一根的close > 前一根的open
        - 后一根的open < 前一根的close
        """
        result = np.zeros(len(data))
        
        for i in range(1, len(data)):
            prev_is_black = data['open'].iloc[i-1] > data['close'].iloc[i-1]
            curr_is_white = data['close'].iloc[i] > data['open'].iloc[i]
            
            engulfed = (
                prev_is_black and curr_is_white and
                data['close'].iloc[i] > data['open'].iloc[i-1] and
                data['open'].iloc[i] < data['close'].iloc[i-1]
            )
            
            result[i] = float(engulfed)
        
        return result
    
    def calculate_bearish_engulfing(self, data: pd.DataFrame) -> np.ndarray:
        """
        看跌吞没：前一根白线被后一根黑线完全吞没
        看跌反转信号
        """
        result = np.zeros(len(data))
        
        for i in range(1, len(data)):
            prev_is_white = data['close'].iloc[i-1] > data['open'].iloc[i-1]
            curr_is_black = data['open'].iloc[i] > data['close'].iloc[i]
            
            engulfed = (
                prev_is_white and curr_is_black and
                data['close'].iloc[i] < data['open'].iloc[i-1] and
                data['open'].iloc[i] > data['close'].iloc[i-1]
            )
            
            result[i] = float(engulfed)
        
        return result
    
    def calculate_piercing_line(self, data: pd.DataFrame) -> np.ndarray:
        """
        刺穿线：前一根黑线，后一根白线穿过前一根的中点
        看涨反转信号
        
        条件：
        - 前一根：黑线
        - 后一根：白线
        - 后一根close > 前一根中点
        - 后一根close < 前一根open
        """
        result = np.zeros(len(data))
        
        for i in range(1, len(data)):
            prev_is_black = data['open'].iloc[i-1] > data['close'].iloc[i-1]
            curr_is_white = data['close'].iloc[i] > data['open'].iloc[i]
            
            midpoint = (data['open'].iloc[i-1] + data['close'].iloc[i-1]) / 2
            
            piercing = (
                prev_is_black and curr_is_white and
                data['close'].iloc[i] > midpoint and
                data['close'].iloc[i] < data['open'].iloc[i-1]
            )
            
            result[i] = float(piercing)
        
        return result
    
    def calculate_dark_cloud_cover(self, data: pd.DataFrame) -> np.ndarray:
        """
        乌云盖顶：前一根白线，后一根黑线穿过前一根的中点
        看跌反转信号
        """
        result = np.zeros(len(data))
        
        for i in range(1, len(data)):
            prev_is_white = data['close'].iloc[i-1] > data['open'].iloc[i-1]
            curr_is_black = data['open'].iloc[i] > data['close'].iloc[i]
            
            midpoint = (data['open'].iloc[i-1] + data['close'].iloc[i-1]) / 2
            
            dark_cloud = (
                prev_is_white and curr_is_black and
                data['close'].iloc[i] < midpoint and
                data['close'].iloc[i] > data['open'].iloc[i-1]
            )
            
            result[i] = float(dark_cloud)
        
        return result
    
    def calculate_morning_star(self, data: pd.DataFrame) -> np.ndarray:
        """
        晨星：三根K线形态
        - 第一根：黑线（下跌）
        - 第二根：小实体（犹豫）
        - 第三根：白线（上升）
        看涨反转信号
        """
        result = np.zeros(len(data))
        
        for i in range(2, len(data)):
            # 第一根黑线
            first_is_black = data['open'].iloc[i-2] > data['close'].iloc[i-2]
            
            # 第二根小实体
            second_body = np.abs(data['close'].iloc[i-1] - data['open'].iloc[i-1])
            first_body = np.abs(data['close'].iloc[i-2] - data['open'].iloc[i-2])
            second_is_small = second_body < first_body * 0.5
            
            # 第三根白线
            third_is_white = data['close'].iloc[i] > data['open'].iloc[i]
            
            # 第三根close > 第一根中点
            midpoint = (data['open'].iloc[i-2] + data['close'].iloc[i-2]) / 2
            third_above_midpoint = data['close'].iloc[i] > midpoint
            
            morning_star = (
                first_is_black and second_is_small and 
                third_is_white and third_above_midpoint
            )
            
            result[i] = float(morning_star)
        
        return result
    
    def calculate_evening_star(self, data: pd.DataFrame) -> np.ndarray:
        """
        暮星：三根K线形态
        - 第一根：白线（上升）
        - 第二根：小实体（犹豫）
        - 第三根：黑线（下跌）
        看跌反转信号
        """
        result = np.zeros(len(data))
        
        for i in range(2, len(data)):
            # 第一根白线
            first_is_white = data['close'].iloc[i-2] > data['open'].iloc[i-2]
            
            # 第二根小实体
            second_body = np.abs(data['close'].iloc[i-1] - data['open'].iloc[i-1])
            first_body = np.abs(data['close'].iloc[i-2] - data['open'].iloc[i-2])
            second_is_small = second_body < first_body * 0.5
            
            # 第三根黑线
            third_is_black = data['open'].iloc[i] > data['close'].iloc[i]
            
            # 第三根close < 第一根中点
            midpoint = (data['open'].iloc[i-2] + data['close'].iloc[i-2]) / 2
            third_below_midpoint = data['close'].iloc[i] < midpoint
            
            evening_star = (
                first_is_white and second_is_small and 
                third_is_black and third_below_midpoint
            )
            
            result[i] = float(evening_star)
        
        return result
    
    def calculate_harami(self, data: pd.DataFrame) -> np.ndarray:
        """
        孕线：前一根大实体，后一根小实体完全在前一根内部
        反转信号
        """
        result = np.zeros(len(data))
        
        for i in range(1, len(data)):
            prev_body = np.abs(data['close'].iloc[i-1] - data['open'].iloc[i-1])
            curr_body = np.abs(data['close'].iloc[i] - data['open'].iloc[i])
            
            prev_high = np.maximum(data['open'].iloc[i-1], data['close'].iloc[i-1])
            prev_low = np.minimum(data['open'].iloc[i-1], data['close'].iloc[i-1])
            
            curr_high = np.maximum(data['open'].iloc[i], data['close'].iloc[i])
            curr_low = np.minimum(data['open'].iloc[i], data['close'].iloc[i])
            
            harami = (
                prev_body > curr_body * 2 and
                curr_high < prev_high and
                curr_low > prev_low
            )
            
            result[i] = float(harami)
        
        return result
    
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
        值越大表示形态越强
        """
        result = np.zeros(len(data))
        
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            
            # 计算同向K线数量
            white_candles = (window_data['close'] > window_data['open']).sum()
            black_candles = (window_data['open'] > window_data['close']).sum()
            
            # 形态强度 = max(白线数, 黑线数) / 总数
            strength = max(white_candles, black_candles) / window
            
            result[i] = strength
        
        return result
    
    def calculate_pattern_confirmation(self, data: pd.DataFrame, window: int = 3) -> np.ndarray:
        """
        形态确认度：检查形态是否被后续K线确认
        """
        result = np.zeros(len(data))
        
        for i in range(window, len(data)):
            # 当前K线方向
            curr_direction = 1 if data['close'].iloc[i] > data['open'].iloc[i] else -1
            
            # 后续K线确认
            confirmation_count = 0
            for j in range(1, window):
                if i + j < len(data):
                    next_direction = 1 if data['close'].iloc[i+j] > data['open'].iloc[i+j] else -1
                    if next_direction == curr_direction:
                        confirmation_count += 1
            
            result[i] = confirmation_count / (window - 1)
        
        return result
    
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
