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
        
        # 4. 单位成交金额 (均值)
        features['amount_per_volume'] = amount.rolling(20).mean() / vol_ma20
        
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
        
        # 3. 加速度（动量的差值）
        features['acceleration'] = features['momentum_10d'] - features['momentum_20d'].shift(10)
        
        # 补充处理
        features = features.replace([np.inf, -np.inf], 0).fillna(0)
        
        return features


class RelativeStrengthFactors:
    """相对强度特征计算器"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        """初始化"""
        self.db_path = db_path
    
    def get_industry_average(self, industry: str) -> Dict[str, float]:
        """
        获取行业平均指标 (当前禁用，防止未来数据泄露)
        """
        # 注意: 数据库中仅有实时快照，在历史回测中使用会导致严重泄露
        return {}
    
    def get_sector_average(self, sector: str) -> Dict[str, float]:
        """
        获取板块平均指标 (当前禁用，防止未来数据泄露)
        """
        # 注意: 数据库中仅有实时快照，在历史回测中使用会导致严重泄露
        return {}
    
    def calculate_relative_strength_factors(self, code: str, info: pd.Series) -> Dict[str, float]:
        """
        计算相对强度因子
        
        参数:
            code: 股票代码
            info: 股票基本面信息
        
        返回:
            相对强度因子字典
        """
        features = {}
        
        if info is None:
            return features
        
        industry = info.get('industry')
        sector = info.get('sector')
        
        # 安全转换数值字段
        try:
            pe_ratio = pd.to_numeric(info.get('pe_ratio'), errors='coerce')
            pb_ratio = pd.to_numeric(info.get('pb_ratio'), errors='coerce')
            roe = pd.to_numeric(info.get('return_on_equity'), errors='coerce')
            roa = pd.to_numeric(info.get('return_on_assets'), errors='coerce')
            revenue_growth = pd.to_numeric(info.get('revenue_growth'), errors='coerce')
            
            # 确保这些值不是 NaN 或 Inf
            pe_ratio = pe_ratio if np.isfinite(pe_ratio) else None
            pb_ratio = pb_ratio if np.isfinite(pb_ratio) else None
            roe = roe if np.isfinite(roe) else None
            roa = roa if np.isfinite(roa) else None
            revenue_growth = revenue_growth if np.isfinite(revenue_growth) else None
        except:
            pe_ratio = pb_ratio = roe = roa = revenue_growth = None
        
        # 获取行业平均值
        if industry:
            industry_avg = self.get_industry_average(industry)
            
            # 相对估值
            if industry_avg.get('pe') and industry_avg['pe'] > 0 and pe_ratio and pe_ratio > 0:
                rel_pe = pe_ratio / industry_avg['pe']
                features['relative_pe_to_industry'] = 0.0 if pd.isna(rel_pe) else rel_pe
            else:
                features['relative_pe_to_industry'] = 0.0
            
            if industry_avg.get('pb') and industry_avg['pb'] > 0 and pb_ratio and pb_ratio > 0:
                rel_pb = pb_ratio / industry_avg['pb']
                features['relative_pb_to_industry'] = 0.0 if pd.isna(rel_pb) else rel_pb
            else:
                features['relative_pb_to_industry'] = 0.0
            
            # 相对盈利能力
            if industry_avg.get('roe') and industry_avg['roe'] > 0 and roe and roe > 0:
                rel_roe = roe / industry_avg['roe']
                features['relative_roe_to_industry'] = 0.0 if pd.isna(rel_roe) else rel_roe
            else:
                features['relative_roe_to_industry'] = 0.0
            
            if industry_avg.get('roa') and industry_avg['roa'] > 0 and roa and roa > 0:
                rel_roa = roa / industry_avg['roa']
                features['relative_roa_to_industry'] = 0.0 if pd.isna(rel_roa) else rel_roa
            else:
                features['relative_roa_to_industry'] = 0.0
            
            # 相对成长性
            if industry_avg.get('growth') and industry_avg['growth'] > 0 and revenue_growth and revenue_growth > 0:
                rel_growth = revenue_growth / industry_avg['growth']
                features['relative_growth_to_industry'] = 0.0 if pd.isna(rel_growth) else rel_growth
            else:
                features['relative_growth_to_industry'] = 0.0
        
        # 获取板块平均值
        if sector:
            sector_avg = self.get_sector_average(sector)
            
            # 相对估值
            if sector_avg.get('pe') and sector_avg['pe'] > 0 and pe_ratio and pe_ratio > 0:
                rel_pe = pe_ratio / sector_avg['pe']
                features['relative_pe_to_sector'] = 0.0 if pd.isna(rel_pe) else rel_pe
            else:
                features['relative_pe_to_sector'] = 0.0
            
            if sector_avg.get('pb') and sector_avg['pb'] > 0 and pb_ratio and pb_ratio > 0:
                rel_pb = pb_ratio / sector_avg['pb']
                features['relative_pb_to_sector'] = 0.0 if pd.isna(rel_pb) else rel_pb
            else:
                features['relative_pb_to_sector'] = 0.0
            
            # 相对盈利能力
            if sector_avg.get('roe') and sector_avg['roe'] > 0 and roe and roe > 0:
                rel_roe = roe / sector_avg['roe']
                features['relative_roe_to_sector'] = 0.0 if pd.isna(rel_roe) else rel_roe
            else:
                features['relative_roe_to_sector'] = 0.0
        
        return features


class CyclicalSeasonalFactors:
    """周期和季节特征计算器"""
    
    @staticmethod
    def calculate_temporal_features(date: pd.Timestamp) -> Dict[str, int]:
        """
        计算时间特征
        
        参数:
            date: 日期
        
        返回:
            时间特征字典
        """
        features = {}
        
        # 星期几（0=周一，4=周五）
        features['day_of_week'] = date.dayofweek
        
        # 月份
        features['month'] = date.month
        
        # 季度
        features['quarter'] = date.quarter
        
        # 是否月初（1-5天）
        features['is_month_start'] = 1 if date.day <= 5 else 0
        
        # 是否月末（25-31天）
        features['is_month_end'] = 1 if date.day >= 25 else 0
        
        # 是否季度末
        features['is_quarter_end'] = 1 if date.month % 3 == 0 and date.day >= 20 else 0
        
        # 是否年末
        features['is_year_end'] = 1 if date.month == 12 and date.day >= 20 else 0
        
        # 距离年末的天数
        year_end = pd.Timestamp(year=date.year, month=12, day=31)
        features['days_to_year_end'] = (year_end - date).days
        
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


# 使用示例
if __name__ == '__main__':
    # 测试时间序列特征
    print("=" * 60)
    print("时间序列特征测试")
    print("=" * 60)
    
    # 创建示例数据
    dates = pd.date_range('2024-01-01', periods=100)
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.rand(100) * 100 + 50,
        'high': np.random.rand(100) * 100 + 55,
        'low': np.random.rand(100) * 100 + 45,
        'close': np.random.rand(100) * 100 + 50,
        'volume': np.random.rand(100) * 1e6,
        'amount': np.random.rand(100) * 1e8,
    })
    
    ts_features = TimeSeriesFactors.calculate_price_series_features(data)
    print("价格序列特征:")
    for key, value in ts_features.items():
        print(f"  {key}: {value:.4f}")
    
    vol_features = TimeSeriesFactors.calculate_volume_series_features(data)
    print("\n成交量序列特征:")
    for key, value in vol_features.items():
        print(f"  {key}: {value:.4f}")
    
    # 测试周期特征
    print("\n" + "=" * 60)
    print("周期特征测试")
    print("=" * 60)
    
    temporal_features = CyclicalSeasonalFactors.calculate_temporal_features(pd.Timestamp('2024-02-08'))
    print("时间特征:")
    for key, value in temporal_features.items():
        print(f"  {key}: {value}")
    
    # 测试风险特征
    print("\n" + "=" * 60)
    print("风险特征测试")
    print("=" * 60)
    
    risk_features = RiskFactors.calculate_risk_features(data)
    print("风险特征:")
    for key, value in risk_features.items():
        print(f"  {key}: {value:.4f}")

