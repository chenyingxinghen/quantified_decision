"""
高级因子模块
包含时间序列特征、相对强度特征、周期特征等
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import sqlite3
from config.config import DATABASE_PATH


class TimeSeriesFactors:
    """时间序列特征计算器"""
    
    @staticmethod
    def calculate_price_series_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格序列特征 (Rolling)
        
        参数:
            data: 包含OHLCV的DataFrame
        
        返回:
            价格序列特征 DataFrame
        """
        features = pd.DataFrame(index=data.index)
        
        if len(data) < 20:
            return features
        
        close = data['close']
        high = data['high']
        low = data['low']
        open_price = data['open']
        
        # 1. 高低价差比
        hl_range = (high - low) / close
        features['hl_range_mean'] = hl_range.rolling(20).mean()
        features['hl_range_std'] = hl_range.rolling(20).std()
        
        # 2. 开收价差比
        oc_ratio = (close - open_price) / open_price
        features['oc_ratio_mean'] = oc_ratio.rolling(20).mean()
        features['oc_ratio_std'] = oc_ratio.rolling(20).std()
        
        # 3. 价格波动率
        returns = close.pct_change()
        features['price_volatility_20'] = returns.rolling(20).std()
        features['price_volatility_60'] = returns.rolling(60).std()
        
        # 4. 偏度和峰度
        features['price_skewness'] = returns.rolling(20).skew()
        features['price_kurtosis'] = returns.rolling(20).kurt()
        
        # 5. 最高价和最低价相对于收盘价的位置 (Rolling Range)
        roll_high = high.rolling(20).max()
        roll_low = low.rolling(20).min()
        roll_range = roll_high - roll_low
        
        features['high_position'] = (high - roll_low) / roll_range
        features['low_position'] = (close - roll_low) / roll_range
        
        # 补充处理
        features = features.replace([np.inf, -np.inf], 0).fillna(0)
        
        return features
    
    @staticmethod
    def calculate_volume_series_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量序列特征 (Rolling)
        
        参数:
            data: 包含OHLCV的DataFrame
        
        返回:
            成交量序列特征 DataFrame
        """
        features = pd.DataFrame(index=data.index)
        
        if len(data) < 20:
            return features
        
        volume = data['volume']
        amount = data['amount']
        close = data['close']
        
        # 1. 成交量变化率 (V/MA20)
        vol_ma20 = volume.rolling(20).mean()
        features['volume_change_rate'] = volume / vol_ma20
        
        # 2. 成交量波动率
        vol_returns = volume.pct_change()
        features['volume_volatility'] = vol_returns.rolling(20).std()
        
        # 3. 价量相关性
        price_returns = close.pct_change()
        features['price_volume_corr'] = price_returns.rolling(20).corr(vol_returns)
        
        # 4. 单位成交金额变化率 (当前均价 vs 历史均价的相对变化)
        # 修复：原始 amount/volume = 均价（元/股），横截面排名后等价于"股价排名"，
        # 这是价格水平偏差而非预测信号。改为计算均价的相对变化率，
        # 捕捉"成交均价是否在抬升/下降"这一有意义的量价信号。
        avg_price_20 = amount.rolling(20).mean() / vol_ma20.replace(0, np.nan)
        avg_price_60 = amount.rolling(60).mean() / volume.rolling(60).mean().replace(0, np.nan)
        features['amount_per_volume'] = (avg_price_20 / avg_price_60.replace(0, np.nan) - 1).fillna(0)
        
        # 5. 成交金额变化率
        amt_ma20 = amount.rolling(20).mean()
        features['amount_change_rate'] = amount / amt_ma20
        
        # 补充处理
        features = features.replace([np.inf, -np.inf], 0).fillna(0)
        
        return features
    
    @staticmethod
    def calculate_momentum_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量特征 (Rolling)
        
        参数:
            data: 包含OHLCV的DataFrame
        
        返回:
            动量特征 DataFrame
        """
        features = pd.DataFrame(index=data.index)
        
        if len(data) < 60:
            return features
        
        close = data['close']
        
        # 1. 不同周期的收益率
        features['return_5d'] = close.pct_change(5)
        features['return_10d'] = close.pct_change(10)
        features['return_20d'] = close.pct_change(20)
        features['return_60d'] = close.pct_change(60)
        
        # 2. 动量（收益率的累计）
        # 这里直接用 pct_change 已经表达了动量，额外补充 rolling sum
        returns = close.pct_change()
        features['momentum_5d'] = returns.rolling(5).sum()
        features['momentum_10d'] = returns.rolling(10).sum()
        features['momentum_20d'] = returns.rolling(20).sum()
        
        # 3. 加速度（动量的二阶导数）
        features['acceleration_5d'] = features['return_5d'] - features['return_5d'].shift(5)
        features['acceleration_10d'] = features['return_10d'] - features['return_10d'].shift(10)
        
        # 4. 状态持续期 (Duration Factors - 极其关键，解决时间盲区)
        # 连续上涨天数
        is_up = (close > close.shift(1)).astype(int)
        # 利用 groupby 巧妙计算连续为 1 的次数
        consecutive_up = is_up * (is_up.groupby((is_up != is_up.shift()).cumsum()).cumcount() + 1)
        features['consecutive_up_days'] = consecutive_up

        # 站在 20 日均线上方的天数
        ma20 = close.rolling(20).mean()
        above_ma20 = (close > ma20).astype(int)
        consecutive_above_ma20 = above_ma20 * (above_ma20.groupby((above_ma20 != above_ma20.shift()).cumsum()).cumcount() + 1)
        features['days_above_ma20'] = consecutive_above_ma20

        # 5. 相对位置 (Micro-structure)
        # 过去60天价格分位数，刻画当前处于高位还是低位
        min_60d = close.rolling(60).min()
        max_60d = close.rolling(60).max()
        features['price_percentile_60d'] = (close - min_60d) / (max_60d - min_60d + 1e-8)
        
        # 过去5天的日内最大回撤（最高价到收盘价跌幅）均值，刻画高位震荡
        intraday_drawdown = (data['high'] - close) / data['high']
        features['intraday_drawdown_avg_5d'] = intraday_drawdown.rolling(5).mean()
        
        # 补充处理
        features = features.replace([np.inf, -np.inf], 0).fillna(0)
        
        return features


class RiskFactors:
    """风险特征计算器"""
    
    @staticmethod
    def calculate_risk_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        计算风险特征 (Rolling)
        
        参数:
            data: 包含OHLCV的DataFrame
        
        返回:
            风险特征 DataFrame
        """
        features = pd.DataFrame(index=data.index)
        
        if len(data) < 20:
            return features
        
        close = data['close']
        returns = close.pct_change()
        
        # 1. 下行风险 (Rolling Standard Deviation of Negative Returns)
        negative_returns = returns.copy()
        negative_returns[negative_returns > 0] = np.nan
        features['downside_risk'] = negative_returns.rolling(20).std()
        
        # 2. 回撤
        cumulative_returns = (1 + returns.fillna(0)).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        features['drawdown'] = drawdown
        features['max_drawdown_20'] = drawdown.rolling(20).min()
        
        # 3. 风险调整收益
        roll_mean = returns.rolling(20).mean()
        roll_std = returns.rolling(20).std()
        features['sharpe_ratio'] = (roll_mean / roll_std) * np.sqrt(252)
        
        # 4. 收益率偏度与峰度
        features['return_skewness'] = returns.rolling(60).skew()
        features['return_kurtosis'] = returns.rolling(60).kurt()
        
        # 补充处理
        features = features.replace([np.inf, -np.inf], 0).fillna(0)
        
        return features

