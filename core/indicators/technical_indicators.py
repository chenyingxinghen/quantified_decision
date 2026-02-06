# 技术指标计算模块
import pandas as pd
import numpy as np
import talib
from config import TECHNICAL_PARAMS
from config.strategy_config import TREND_MA_LONG_PERIOD, TREND_MA_MID_PERIOD, TREND_MA_SHORT_PERIOD


class TechnicalIndicators:
    def __init__(self):
        self.params = TECHNICAL_PARAMS
    
    def calculate_ma(self, data, period=None):
        """计算移动平均线"""
        if period is None:
            period = self.params['ma_short']
        return talib.SMA(data['close'].values, timeperiod=period)
    
    def calculate_rsi(self, data, period=None):
        """计算RSI指标"""
        if period is None:
            period = self.params['rsi_period']
        return talib.RSI(data['close'].values, timeperiod=period)
    
    def calculate_macd(self, data):
        """计算MACD指标"""
        macd, signal, hist = talib.MACD(
            data['close'].values,
            fastperiod=self.params['macd_fast'],
            slowperiod=self.params['macd_slow'],
            signalperiod=self.params['macd_signal']
        )
        return macd, signal, hist
    
    def calculate_bollinger_bands(self, data):
        """计算布林带"""
        upper, middle, lower = talib.BBANDS(
            data['close'].values,
            timeperiod=self.params['bb_period'],
            nbdevup=self.params['bb_std'],
            nbdevdn=self.params['bb_std']
        )
        return upper, middle, lower
    
    def calculate_kdj(self, data, n=9, m1=3, m2=3):
        """计算KDJ指标"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # 计算RSV
        rsv = (close - pd.Series(low).rolling(n).min()) / \
              (pd.Series(high).rolling(n).max() - pd.Series(low).rolling(n).min()) * 100
        
        # 计算K值
        k = rsv.ewm(alpha=1/m1).mean()
        # 计算D值
        d = k.ewm(alpha=1/m2).mean()
        # 计算J值
        j = 3 * k - 2 * d
        
        return k.values, d.values, j.values
    
    def calculate_volume_indicators(self, data):
        """计算成交量指标"""
        # 成交量移动平均
        vol_ma5 = talib.SMA(data['volume'].values, timeperiod=5)
        vol_ma10 = talib.SMA(data['volume'].values, timeperiod=10)
        
        # 量比
        volume_ratio = data['volume'] / vol_ma5
        
        return vol_ma5, vol_ma10, volume_ratio.values
    
    def calculate_all_indicators(self, data):
        """计算所有技术指标"""
        if len(data) < TREND_MA_LONG_PERIOD:  # 数据不足
            return None
        
        indicators = {}
        
        try:
            # 移动平均线
            indicators['ma5'] = self.calculate_ma(data, TREND_MA_SHORT_PERIOD)
            indicators['ma10'] = self.calculate_ma(data, TREND_MA_MID_PERIOD)
            indicators['ma20'] = self.calculate_ma(data, TREND_MA_LONG_PERIOD)
            
            # RSI
            indicators['rsi'] = self.calculate_rsi(data)
            
            # MACD
            macd, signal, hist = self.calculate_macd(data)
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_hist'] = hist
            
            # 布林带
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            
            # KDJ
            k, d, j = self.calculate_kdj(data)
            indicators['kdj_k'] = k
            indicators['kdj_d'] = d
            indicators['kdj_j'] = j
            
            # 成交量指标
            vol_ma5, vol_ma10, vol_ratio = self.calculate_volume_indicators(data)
            indicators['vol_ma5'] = vol_ma5
            indicators['vol_ma10'] = vol_ma10
            indicators['vol_ratio'] = vol_ratio
            
            return indicators
            
        except Exception as e:
            print(f"计算技术指标失败: {e}")
            return None
