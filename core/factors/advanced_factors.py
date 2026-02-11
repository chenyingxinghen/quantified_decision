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
    def calculate_price_series_features(data: pd.DataFrame) -> Dict[str, float]:
        """
        计算价格序列特征
        
        参数:
            data: 包含OHLCV的DataFrame
        
        返回:
            价格序列特征字典
        """
        features = {}
        
        if len(data) < 20:
            return features
        
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 高低价差比（最近20天）
        hl_range = np.divide(high - low, close, where=close!=0, out=np.zeros_like(close))
        hl_range_mean = hl_range.tail(20).mean()
        hl_range_std = hl_range.tail(20).std()
        features['hl_range_mean'] = 0.0 if pd.isna(hl_range_mean) else hl_range_mean
        features['hl_range_std'] = 0.0 if pd.isna(hl_range_std) else hl_range_std
        
        # 开收价差比
        oc_ratio = np.divide(close - data['open'], data['open'], where=data['open']!=0, out=np.zeros_like(data['open']))
        oc_ratio_mean = oc_ratio.tail(20).mean()
        oc_ratio_std = oc_ratio.tail(20).std()
        
        # 价格波动率（标准差）
        returns = close.pct_change()
        
        # 安全地计算统计量（排除非有限值以避免RuntimeWarning）
        hl_range_clean = hl_range[np.isfinite(hl_range)].tail(20)
        oc_ratio_clean = oc_ratio[np.isfinite(oc_ratio)].tail(20)
        returns_clean = returns[np.isfinite(returns)]
        
        features['hl_range_mean'] = hl_range_clean.mean() if len(hl_range_clean) > 0 else 0.0
        features['hl_range_std'] = hl_range_clean.std() if len(hl_range_clean) > 1 else 0.0
        
        features['oc_ratio_mean'] = oc_ratio_clean.mean() if len(oc_ratio_clean) > 0 else 0.0
        features['oc_ratio_std'] = oc_ratio_clean.std() if len(oc_ratio_clean) > 1 else 0.0
        
        features['price_volatility_20'] = returns_clean.tail(20).std() if len(returns_clean.tail(20)) > 1 else 0.0
        features['price_volatility_60'] = returns_clean.tail(60).std() if len(returns_clean.tail(60)) > 1 else 0.0
        
        # 偏度和峰度
        features['price_skewness'] = returns_clean.tail(20).skew() if len(returns_clean.tail(20)) > 2 else 0.0
        features['price_kurtosis'] = returns_clean.tail(20).kurt() if len(returns_clean.tail(20)) > 3 else 0.0
        
        # 最高价和最低价相对于收盘价的位置
        high_tail_max = high.tail(20).max()
        low_tail_min = low.tail(20).min()
        range_val = high_tail_max - low_tail_min
        
        if range_val > 0:
            high_pos = (high.iloc[-1] - low_tail_min) / range_val
            low_pos = (close.iloc[-1] - low_tail_min) / range_val
        else:
            high_pos = 0.5
            low_pos = 0.5
        
        features['high_position'] = 0.5 if pd.isna(high_pos) else high_pos
        features['low_position'] = 0.5 if pd.isna(low_pos) else low_pos
        
        return features
    
    @staticmethod
    def calculate_volume_series_features(data: pd.DataFrame) -> Dict[str, float]:
        """
        计算成交量序列特征
        
        参数:
            data: 包含OHLCV的DataFrame
        
        返回:
            成交量序列特征字典
        """
        features = {}
        
        if len(data) < 20:
            return features
        
        volume = data['volume']
        amount = data['amount']
        close = data['close']
        
        # 成交量变化率
        vol_ma20 = volume.tail(20).mean()
        if vol_ma20 > 0:
            vol_change = volume.iloc[-1] / vol_ma20
            features['volume_change_rate'] = 0.0 if pd.isna(vol_change) else vol_change
        else:
            features['volume_change_rate'] = 0.0
        
        # 成交量波动率
        vol_returns = volume.pct_change()
        vol_volatility = vol_returns.tail(20).std()
        features['volume_volatility'] = 0.0 if pd.isna(vol_volatility) else vol_volatility
        
        # 成交量趋势（斜率）
        try:
            vol_trend = np.polyfit(range(20), volume.tail(20).values, 1)[0]
            vol_trend_normalized = vol_trend / vol_ma20 if vol_ma20 > 0 else 0
            features['volume_trend'] = 0.0 if pd.isna(vol_trend_normalized) else vol_trend_normalized
        except:
            features['volume_trend'] = 0.0
        
        # 价量相关性
        price_returns = close.pct_change()
        if len(price_returns) >= 20:
            corr = price_returns.tail(20).corr(vol_returns.tail(20))
            features['price_volume_corr'] = 0.0 if pd.isna(corr) else corr
        else:
            features['price_volume_corr'] = 0.0
        
        # 单位成交金额
        if vol_ma20 > 0:
            amt_per_vol = amount.tail(20).mean() / vol_ma20
            features['amount_per_volume'] = 0.0 if pd.isna(amt_per_vol) else amt_per_vol
        else:
            features['amount_per_volume'] = 0.0
        
        # 成交金额变化率
        amt_ma20 = amount.tail(20).mean()
        if amt_ma20 > 0:
            amt_change = amount.iloc[-1] / amt_ma20
            features['amount_change_rate'] = 0.0 if pd.isna(amt_change) else amt_change
        else:
            features['amount_change_rate'] = 0.0
        
        return features
    
    @staticmethod
    def calculate_momentum_features(data: pd.DataFrame) -> Dict[str, float]:
        """
        计算动量特征
        
        参数:
            data: 包含OHLCV的DataFrame
        
        返回:
            动量特征字典
        """
        features = {}
        
        if len(data) < 60:
            return features
        
        close = data['close']
        returns = close.pct_change()
        
        # 不同周期的收益率
        ret_5d = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 and close.iloc[-5] > 0 else 0
        ret_10d = (close.iloc[-1] / close.iloc[-10] - 1) if len(close) >= 10 and close.iloc[-10] > 0 else 0
        ret_20d = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 and close.iloc[-20] > 0 else 0
        ret_60d = (close.iloc[-1] / close.iloc[-60] - 1) if len(close) >= 60 and close.iloc[-60] > 0 else 0
        
        features['return_5d'] = 0.0 if pd.isna(ret_5d) else ret_5d
        features['return_10d'] = 0.0 if pd.isna(ret_10d) else ret_10d
        features['return_20d'] = 0.0 if pd.isna(ret_20d) else ret_20d
        features['return_60d'] = 0.0 if pd.isna(ret_60d) else ret_60d
        
        # 动量（收益率的变化）
        mom_5d = returns.tail(5).sum()
        mom_10d = returns.tail(10).sum()
        mom_20d = returns.tail(20).sum()
        
        features['momentum_5d'] = 0.0 if pd.isna(mom_5d) else mom_5d
        features['momentum_10d'] = 0.0 if pd.isna(mom_10d) else mom_10d
        features['momentum_20d'] = 0.0 if pd.isna(mom_20d) else mom_20d
        
        # 加速度（动量的变化）
        if len(returns) >= 20:
            recent_momentum = returns.tail(10).sum()
            past_momentum = returns.tail(20).head(10).sum()
            acceleration = recent_momentum - past_momentum
            features['acceleration'] = 0.0 if pd.isna(acceleration) else acceleration
        else:
            features['acceleration'] = 0.0
        
        return features


class RelativeStrengthFactors:
    """相对强度特征计算器"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        """初始化"""
        self.db_path = db_path
    
    def get_industry_average(self, industry: str) -> Dict[str, float]:
        """
        获取行业平均指标
        
        参数:
            industry: 行业名称
        
        返回:
            行业平均指标字典
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    AVG(pe_ratio) as avg_pe,
                    AVG(pb_ratio) as avg_pb,
                    AVG(return_on_equity) as avg_roe,
                    AVG(return_on_assets) as avg_roa,
                    AVG(profit_margins) as avg_margin,
                    AVG(revenue_growth) as avg_growth,
                    AVG(market_cap) as avg_market_cap
                FROM stock_info_extended
                WHERE industry = ?
            """
            
            df = pd.read_sql_query(query, conn, params=(industry,))
        
        if len(df) > 0:
            return {
                'pe': df['avg_pe'].iloc[0],
                'pb': df['avg_pb'].iloc[0],
                'roe': df['avg_roe'].iloc[0],
                'roa': df['avg_roa'].iloc[0],
                'margin': df['avg_margin'].iloc[0],
                'growth': df['avg_growth'].iloc[0],
                'market_cap': df['avg_market_cap'].iloc[0],
            }
        
        return {}
    
    def get_sector_average(self, sector: str) -> Dict[str, float]:
        """
        获取板块平均指标
        
        参数:
            sector: 板块名称
        
        返回:
            板块平均指标字典
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    AVG(pe_ratio) as avg_pe,
                    AVG(pb_ratio) as avg_pb,
                    AVG(return_on_equity) as avg_roe,
                    AVG(market_cap) as avg_market_cap
                FROM stock_info_extended
                WHERE sector = ?
            """
            
            df = pd.read_sql_query(query, conn, params=(sector,))
        
        if len(df) > 0:
            return {
                'pe': df['avg_pe'].iloc[0],
                'pb': df['avg_pb'].iloc[0],
                'roe': df['avg_roe'].iloc[0],
                'market_cap': df['avg_market_cap'].iloc[0],
            }
        
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
    def calculate_risk_features(data: pd.DataFrame) -> Dict[str, float]:
        """
        计算风险特征
        
        参数:
            data: 包含OHLCV的DataFrame
        
        返回:
            风险特征字典
        """
        features = {}
        
        if len(data) < 20:
            return features
        
        close = data['close']
        returns = close.pct_change()
        
        # 下行风险（只考虑负收益）
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside = negative_returns.std()
            features['downside_risk'] = 0.0 if pd.isna(downside) else downside
        else:
            features['downside_risk'] = 0.0
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_dd = drawdown.min()
        features['max_drawdown'] = 0.0 if pd.isna(max_dd) else max_dd
        
        # 动量特征 (风险调整后)
        returns_clean = returns[np.isfinite(returns)]
        if len(returns_clean) > 1:
            std = returns_clean.std()
            if std > 0:
                features['sharpe_ratio'] = returns_clean.mean() / std * np.sqrt(252)
            else:
                features['sharpe_ratio'] = 0.0
            
            # 索提诺比率 (只考虑下行偏差)
            downside_returns = returns_clean[returns_clean < 0]
            if len(downside_returns) > 1:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    features['sortino_ratio'] = returns_clean.mean() / downside_std * np.sqrt(252)
                else:
                    features['sortino_ratio'] = 0.0
            else:
                features['sortino_ratio'] = 0.0
        else:
            features['sharpe_ratio'] = 0.0
            features['sortino_ratio'] = 0.0
        
        # 收益率的偏度（风险不对称性）
        skew = returns.tail(60).skew()
        features['return_skewness'] = 0.0 if pd.isna(skew) else skew
        
        # 收益率的峰度（尾部风险）
        kurt = returns.tail(60).kurt()
        features['return_kurtosis'] = 0.0 if pd.isna(kurt) else kurt
        
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

