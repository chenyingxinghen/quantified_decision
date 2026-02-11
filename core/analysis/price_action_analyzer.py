"""
价格行为分析器 - 提供原子化的价格行为分析方法

核心功能：
1. 摆动点识别（高点/低点）
2. 市场结构分析（HH-HL-LH-LL）
3. 趋势强度计算
4. ATR计算

所有方法都是独立的、可组合的原子操作
不包含复杂的综合分析，避免与其他模块重复

注意：
- K线形态识别请使用 candlestick_patterns.py
- 趋势线分析请使用 trend_line_analyzer.py
- 技术指标计算请使用 technical_indicators.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from config import FactorConfig


class PriceActionAnalyzer:
    """
    价格行为分析器 - 提供原子化的价格行为分析方法
    
    核心功能：
    1. 摆动点识别（高点/低点）
    2. 市场结构分析（HH-HL-LH-LL）
    3. 趋势强度计算
    4. ATR计算
    
    所有方法都是独立的、可组合的原子操作
    """
    def __init__(self):
        self.min_bars_for_analysis = 50
    
    def identify_market_structure(self, data):
        """
        识别市场结构：HH-HL-LH-LL 四元素结构法则
        
        参数:
            data: 价格数据DataFrame
            
        返回:
            {
                'trend': str,  # 趋势类型
                'strength': float,  # 趋势强度 (0-100)
                'pattern': str,  # 结构模式
                'key_levels': dict,  # 关键点位
                'structure_shift': bool  # 是否结构转换
            }
        """
        if len(data) < self.min_bars_for_analysis:
            return None
        
        # 计算高低点
        highs = data['high'].values
        lows = data['low'].values
        
        # 寻找局部高点和低点
        swing_highs = self.find_swing_points(highs, 'high')
        swing_lows = self.find_swing_points(lows, 'low')
        
        # 分析市场结构
        structure = self._analyze_structure(swing_highs, swing_lows, data)
        
        return structure
    
    def find_swing_points(self, prices, point_type='high', lookback=5):
        """
        寻找摆动高点和低点（原子操作）
        
        参数:
            prices: 价格数组
            point_type: 'high' 或 'low'
            lookback: 回看周期
            
        返回:
            摆动点列表 [{'index': int, 'price': float, 'type': str}]
        """
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
        """
        分析市场结构模式（内部方法）
        
        参数:
            swing_highs: 高点列表
            swing_lows: 低点列表
            data: 价格数据
            
        返回:
            市场结构分析结果
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'trend': 'insufficient_data', 'strength': 0}
        
        # 获取最近的高低点
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
        
        # 分析趋势模式
        trend_analysis = self._determine_trend_pattern(recent_highs, recent_lows)
        
        # 计算趋势强度
        strength = self.calculate_trend_strength(data, recent_highs, recent_lows)
        
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
        """
        确定趋势模式（原子操作）
        
        参数:
            highs: 高点列表
            lows: 低点列表
            
        返回:
            {'direction': str, 'pattern': str, 'structure_shift': bool}
        """
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
    
    def calculate_trend_strength(self, data, highs=None, lows=None):
        """
        计算趋势强度（原子操作）
        
        参数:
            data: 价格数据DataFrame
            highs: 高点列表（可选）
            lows: 低点列表（可选）
            
        返回:
            趋势强度 (0-100)
        """
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
        atr = self.calculate_atr(data, FactorConfig.ATR_PERIOD)
        current_close = data['close'].iloc[-1]
        if len(atr) > 0 and current_close != 0:
            volatility_factor = min(atr[-1] / current_close * 100, 10)
        else:
            volatility_factor = 5
        
        # 综合强度计算
        strength = abs(price_momentum) * volume_ratio * (volatility_factor / 5)
        return min(max(strength, 0), 100)
    
    def calculate_atr(self, data, period=FactorConfig.ATR_PERIOD):
        """
        计算平均真实波幅（原子操作）
        
        参数:
            data: 价格数据DataFrame
            period: ATR周期
            
        返回:
            ATR数组
        """
        if len(data) < period + 1:
            return np.array([])
        
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
