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
from core.analysis.candlestick_patterns import CandlestickPatterns


class CandlestickPatternFactors:
    """K线形态因子计算类"""
    
    def __init__(self, config: Optional[FactorConfig] = None):
        """初始化K线形态因子计算器"""
        self.config = config if config is not None else FactorConfig
        self.analyzer = CandlestickPatterns()
        self.pattern_names = self.get_pattern_names()
    
    # ==================== 单根K线形态 ====================
    
    def calculate_white_candle(self, data: pd.DataFrame) -> np.ndarray:
        """白线/阳线"""
        return self.analyzer.identify_white_candle(data)
    
    def calculate_black_candle(self, data: pd.DataFrame) -> np.ndarray:
        """黑线/阴线"""
        return self.analyzer.identify_black_candle(data)
    
    def calculate_doji(self, data: pd.DataFrame, threshold: float = 0.001) -> np.ndarray:
        """十字星"""
        if threshold == 0.001:
            threshold = self.config.BODY_SIZE_THRESHOLD_SMALL
        return self.analyzer.identify_doji(data, threshold=threshold)
    
    def calculate_hammer(self, data: pd.DataFrame, context: pd.DataFrame = None) -> np.ndarray:
        """锤子线"""
        if context is None:
            context = self.analyzer.calculate_context(data)
        return self.analyzer.identify_hammer(
            data, context, 
            lower_ratio=getattr(self.config, 'HAMMER_LOWER_SHADOW_RATIO', 2.0),
            upper_ratio=getattr(self.config, 'HAMMER_UPPER_SHADOW_RATIO', 1.0)
        )
    
    def calculate_hanging_man(self, data: pd.DataFrame, context: pd.DataFrame = None) -> np.ndarray:
        """上吊线"""
        if context is None:
            context = self.analyzer.calculate_context(data)
        return self.analyzer.identify_hanging_man(
            data, context,
            lower_ratio=getattr(self.config, 'HAMMER_LOWER_SHADOW_RATIO', 2.0),
            upper_ratio=getattr(self.config, 'HAMMER_UPPER_SHADOW_RATIO', 1.0)
        )
    
    def calculate_shooting_star(self, data: pd.DataFrame, context: pd.DataFrame = None) -> np.ndarray:
        """射击之星"""
        if context is None:
            context = self.analyzer.calculate_context(data)
        return self.analyzer.identify_shooting_star(
            data, context,
            upper_ratio=getattr(self.config, 'HAMMER_LOWER_SHADOW_RATIO', 2.0),
            lower_ratio=getattr(self.config, 'HAMMER_UPPER_SHADOW_RATIO', 1.0)
        )
    
    def calculate_inverted_hammer(self, data: pd.DataFrame, context: pd.DataFrame = None) -> np.ndarray:
        """倒锤线"""
        if context is None:
            context = self.analyzer.calculate_context(data)
        return self.analyzer.identify_inverted_hammer(
            data, context,
            upper_ratio=getattr(self.config, 'HAMMER_LOWER_SHADOW_RATIO', 2.0),
            lower_ratio=getattr(self.config, 'HAMMER_UPPER_SHADOW_RATIO', 1.0)
        )
    
    def calculate_marubozu(self, data: pd.DataFrame) -> np.ndarray:
        """光头光脚线"""
        return self.analyzer.identify_marubozu(data, threshold_ratio=self.config.BODY_SIZE_THRESHOLD_SMALL)
    
    def calculate_spinning_top(self, data: pd.DataFrame) -> np.ndarray:
        """纺锤线"""
        return self.analyzer.identify_spinning_top(data)
    
    # ==================== 多根K线形态 ====================
    
    def calculate_bullish_engulfing(self, data: pd.DataFrame, context: pd.DataFrame = None) -> np.ndarray:
        """看涨吞没"""
        if context is None:
            context = self.analyzer.calculate_context(data)
        return self.analyzer.identify_bullish_engulfing(data, context)
    
    def calculate_bearish_engulfing(self, data: pd.DataFrame, context: pd.DataFrame = None) -> np.ndarray:
        """看跌吞没"""
        if context is None:
            context = self.analyzer.calculate_context(data)
        return self.analyzer.identify_bearish_engulfing(data, context)
    
    def calculate_piercing_line(self, data: pd.DataFrame) -> np.ndarray:
        """刺穿线"""
        return self.analyzer.identify_piercing_line(data)
    
    def calculate_dark_cloud_cover(self, data: pd.DataFrame) -> np.ndarray:
        """乌云盖顶"""
        return self.analyzer.identify_dark_cloud_cover(data)
    
    def calculate_morning_star(self, data: pd.DataFrame, context: pd.DataFrame = None) -> np.ndarray:
        """晨星"""
        if context is None:
            context = self.analyzer.calculate_context(data)
        return self.analyzer.identify_morning_star(data, context)
    
    def calculate_evening_star(self, data: pd.DataFrame, context: pd.DataFrame = None) -> np.ndarray:
        """暮星"""
        if context is None:
            context = self.analyzer.calculate_context(data)
        return self.analyzer.identify_evening_star(data, context)
    
    def calculate_harami(self, data: pd.DataFrame) -> np.ndarray:
        """孕线"""
        return self.analyzer.identify_harami(data)
    
    # ==================== K线形态强度指标 ====================
    
    def calculate_candle_body_ratio(self, data: pd.DataFrame) -> np.ndarray:
        """K线实体比率"""
        return self.analyzer.get_candle_body_ratio(data)
    
    def calculate_upper_shadow_ratio(self, data: pd.DataFrame) -> np.ndarray:
        """上影线比率"""
        return self.analyzer.get_upper_shadow_ratio(data)
    
    def calculate_lower_shadow_ratio(self, data: pd.DataFrame) -> np.ndarray:
        """下影线比率"""
        return self.analyzer.get_lower_shadow_ratio(data)
    
    def calculate_pattern_strength(self, data: pd.DataFrame, window: int = 5) -> np.ndarray:
        """形态强度指标"""
        return self.analyzer.get_pattern_strength(data, window=window)
    
    def calculate_pattern_confirmation(self, data: pd.DataFrame, window: int = 3) -> np.ndarray:
        """形态确认度"""
        return self.analyzer.get_pattern_confirmation(data, window=window)
    
    def calculate_all_candlestick_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有K线形态因子
        
        Args:
            data: 包含OHLC数据的DataFrame
        
        Returns:
            包含所有K线形态因子的DataFrame
        """
        factors = pd.DataFrame(index=data.index)
        
        # 统一计算背景上下文
        context = self.analyzer.calculate_context(data)
        
        # 单根K线形态
        factors['white_candle'] = self.calculate_white_candle(data)
        factors['black_candle'] = self.calculate_black_candle(data)
        factors['doji'] = self.calculate_doji(data)
        factors['hammer'] = self.calculate_hammer(data, context=context)
        factors['hanging_man'] = self.calculate_hanging_man(data, context=context)
        factors['shooting_star'] = self.calculate_shooting_star(data, context=context)
        factors['inverted_hammer'] = self.calculate_inverted_hammer(data, context=context)
        factors['marubozu'] = self.calculate_marubozu(data)
        factors['spinning_top'] = self.calculate_spinning_top(data)
        
        # 多根K线形态
        factors['bullish_engulfing'] = self.calculate_bullish_engulfing(data, context=context)
        factors['bearish_engulfing'] = self.calculate_bearish_engulfing(data, context=context)
        factors['piercing_line'] = self.calculate_piercing_line(data)
        factors['dark_cloud_cover'] = self.calculate_dark_cloud_cover(data)
        factors['morning_star'] = self.calculate_morning_star(data, context=context)
        factors['evening_star'] = self.calculate_evening_star(data, context=context)
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
