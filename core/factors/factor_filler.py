"""
因子填充模块
对计算失败的因子进行填充，确保因子数始终保持最大值
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)


class FactorFiller:
    """因子填充器 - 对缺失的因子进行填充"""
    
    # 所有可能的因子名称（参考模型训练时的特征）
    ALL_POSSIBLE_FACTORS = {
        # 技术指标因子 (49个)
        'rsi_6', 'rsi_12', 'rsi_24', 'stochrsi_k', 'stochrsi_d',
        'macd', 'macd_signal', 'macd_hist', 'atr_14', 'natr_14',
        'adx_14', 'di_plus_14', 'di_minus_14', 'cci_20', 'roc_12',
        'williams_r_14', 'obv', 'obv_ma', 'ad_line', 'cmf_20',
        'mfi_14', 'kama_10', 'kama_30', 'kama_efficiency', 'dpo_20',
        'keltner_upper', 'keltner_lower', 'keltner_width', 'bb_upper', 'bb_lower',
        'bb_width', 'bb_position', 'ma_slope_20', 'ma_slope_50', 'ma_slope_200',
        'price_slope_20', 'price_slope_50', 'price_slope_200', 'volume_slope_20',
        'volume_slope_50', 'volume_slope_200', 'volatility_20', 'volatility_60',
        'high_low_ratio', 'close_open_ratio', 'volume_ma_ratio', 'price_ma_ratio',
        'trend_strength', 'trend_direction', 'momentum_20', 'momentum_50',
        
        # K线形态因子 (21个)
        'white_candle', 'black_candle', 'doji', 'hammer', 'hanging_man',
        'shooting_star', 'inverted_hammer', 'marubozu', 'spinning_top',
        'bullish_engulfing', 'bearish_engulfing', 'piercing_line', 'dark_cloud_cover',
        'morning_star', 'evening_star', 'harami', 'candle_body_ratio',
        'upper_shadow_ratio', 'lower_shadow_ratio', 'pattern_strength', 'pattern_confirmation',
        
        # 时间序列特征 (24个)
        'hl_range_mean', 'hl_range_std', 'price_volatility_20', 'price_volatility_60',
        'price_skewness', 'price_kurtosis', 'volume_change_rate', 'volume_volatility',
        'return_5d', 'return_10d', 'momentum_5d', 'downside_risk',
        'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
        'information_ratio', 'treynor_ratio', 'jensen_alpha', 'beta',
        'correlation_market', 'autocorrelation_lag1', 'autocorrelation_lag5', 'hurst_exponent',
        
        # 风险特征 (6个)
        'value_at_risk_95', 'conditional_var_95', 'expected_shortfall',
        'tail_ratio', 'extreme_value_index', 'stress_test_loss',
        
        # 基本面因子 (多个)
        'pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio', 'peg_ratio',
        'dividend_yield', 'earnings_yield', 'fcf_yield', 'book_value_yield',
        'roe', 'roa', 'roic', 'profit_margin', 'gross_margin',
        'operating_margin', 'ebitda_margin', 'net_margin', 'current_ratio',
        'quick_ratio', 'debt_to_equity', 'debt_to_assets', 'equity_multiplier',
        'interest_coverage', 'debt_service_coverage', 'revenue_growth',
        'earnings_growth', 'earnings_quarterly_growth', 'fcf_growth',
        'dividend_growth', 'five_year_avg_dividend_yield', 'forward_pe',
        'forward_earnings_growth', 'peg_ratio_forward', 'enterprise_to_ebitda',
        'enterprise_to_revenue', 'enterprise_to_fcf', 'price_to_tangible_book',
        'price_to_working_capital', 'price_to_sales_growth', 'price_to_fcf_growth',
        'earnings_quality', 'accruals_ratio', 'cash_conversion_ratio',
        'operating_cash_flow', 'free_cash_flow', 'capital_expenditure',
        'cash_flow_from_operations', 'cash_flow_from_investing', 'cash_flow_from_financing',
        'operating_cash_flow_to_net_income', 'fcf_to_net_income', 'fcf_to_market_cap',
        'ocf_to_debt', 'ocf_to_market_cap', 'cash_to_market_cap',
        'debt_to_market_cap', 'equity_to_market_cap', 'tangible_book_value',
        'intangible_assets_ratio', 'goodwill_ratio', 'working_capital',
        'working_capital_ratio', 'cash_ratio', 'inventory_turnover',
        'receivables_turnover', 'payables_turnover', 'asset_turnover',
        'fixed_asset_turnover', 'return_on_equity', 'return_on_assets',
        'return_on_invested_capital', 'dupont_roe', 'dupont_roa',
        'relative_pe_to_industry', 'relative_pe_to_sector', 'relative_pb_to_industry',
        'relative_pb_to_sector', 'relative_roe_to_industry', 'relative_roe_to_sector',
        'relative_roa_to_industry', 'relative_growth_to_industry', 'roe_to_leverage',
        'roe_to_pb', 'roe_growth', 'institution_quality', 'dividend_quality',
        'mcap_to_revenue_growth', 'pe_earnings_growth', 'log_market_cap',
        'log_book_value', 'log_enterprise_value', 'float_ratio',
        'turnover_ratio', 'liquidity_ratio',
    }
    
    def __init__(self, fill_value: float = 0.0, fill_method: str = 'zero'):
        """
        初始化因子填充器
        
        参数:
            fill_value: 填充值 (默认0)
            fill_method: 填充方法 ('zero', 'mean', 'median', 'forward_fill')
        """
        self.fill_value = fill_value
        self.fill_method = fill_method
        self.factor_stats = {}  # 存储因子统计信息
    
    def get_missing_factors(self, existing_factors: Set[str]) -> Set[str]:
        """
        获取缺失的因子
        
        参数:
            existing_factors: 现有因子集合
        
        返回:
            缺失因子集合
        """
        return self.ALL_POSSIBLE_FACTORS - existing_factors
    
    def fill_missing_factors(self, factors_df: pd.DataFrame, 
                            target_factors: Optional[List[str]] = None,
                            keep_all_generated: bool = True) -> pd.DataFrame:
        """
        填充缺失的因子
        
        参数:
            factors_df: 因子DataFrame
            target_factors: 目标因子列表 (如果为None，则使用所有可能的因子)
            keep_all_generated: 是否保留所有生成的特征 (默认True，保留特征工程生成的所有特征)
        
        返回:
            填充后的因子DataFrame
        """
        # 如果keep_all_generated为True，直接返回原DataFrame（保留所有生成的特征）
        if keep_all_generated:
            # 只填充NaN和无穷大值，不限制因子数量
            return factors_df
        
        # 原有逻辑：限制在预定义的因子集合内
        if target_factors is None:
            target_factors = sorted(list(self.ALL_POSSIBLE_FACTORS))
        
        existing_factors = set(factors_df.columns)
        missing_factors = [f for f in target_factors if f not in existing_factors]
        
        if not missing_factors:
            # 所有因子都存在，只需要重新排序
            return factors_df[target_factors]
        
        # 高效地创建缺失因子的字典
        missing_data = {}
        
        for factor in missing_factors:
            if self.fill_method == 'zero':
                missing_data[factor] = self.fill_value
            elif self.fill_method == 'mean':
                # 使用现有数据的均值
                mean_val = self._calculate_mean_from_similar_factors(factor, factors_df)
                missing_data[factor] = mean_val
            elif self.fill_method == 'median':
                # 使用现有数据的中位数
                median_val = self._calculate_median_from_similar_factors(factor, factors_df)
                missing_data[factor] = median_val
            else:
                missing_data[factor] = self.fill_value
        
        # 一次性创建缺失因子的DataFrame
        missing_df = pd.DataFrame(missing_data, index=factors_df.index)
        
        # 合并现有因子和缺失因子
        result_df = pd.concat([factors_df, missing_df], axis=1)
        
        # 按目标顺序重新排列
        result_df = result_df[target_factors]
        
        # 记录填充信息
        self._log_filling_info(factors_df.shape[1], len(missing_factors), len(target_factors))
        
        return result_df
    
    def fill_nan_values(self, factors_df: pd.DataFrame, 
                       fill_method: str = 'zero') -> pd.DataFrame:
        """
        填充NaN值
        
        参数:
            factors_df: 因子DataFrame
            fill_method: 填充方法 ('zero', 'mean', 'median', 'forward_fill', 'backward_fill')
        
        返回:
            填充后的因子DataFrame
        """
        result_df = factors_df.copy()
        
        for col in result_df.columns:
            # 确保列是数值类型
            if not np.issubdtype(result_df[col].dtype, np.number):
                continue
                
            nan_count = result_df[col].isna().sum()
            
            if nan_count == 0:
                continue
            
            if fill_method == 'zero':
                result_df[col] = result_df[col].fillna(0)
            elif fill_method == 'mean':
                mean_val = result_df[col].mean()
                if not np.isnan(mean_val):
                    result_df[col] = result_df[col].fillna(mean_val)
                else:
                    result_df[col] = result_df[col].fillna(0)
            elif fill_method == 'median':
                median_val = result_df[col].median()
                if not np.isnan(median_val):
                    result_df[col] = result_df[col].fillna(median_val)
                else:
                    result_df[col] = result_df[col].fillna(0)
            elif fill_method == 'forward_fill':
                result_df[col] = result_df[col].ffill()
                result_df[col] = result_df[col].fillna(0)
            elif fill_method == 'backward_fill':
                result_df[col] = result_df[col].bfill()
                result_df[col] = result_df[col].fillna(0)
            
            if nan_count > 0:
                logger.debug(f"填充 {col}: {nan_count} 个NaN值")
        
        return result_df
    
    def fill_inf_values(self, factors_df: pd.DataFrame, 
                       fill_value: float = 0.0) -> pd.DataFrame:
        """
        填充无穷大值
        
        参数:
            factors_df: 因子DataFrame
            fill_value: 填充值
        
        返回:
            填充后的因子DataFrame
        """
        result_df = factors_df.copy()
        
        for col in result_df.columns:
            # 使用 np.isfinite 处理所有非有限值 (inf, -inf, nan)
            # 这里我们专门处理 inf
            inf_mask = np.isinf(result_df[col])
            inf_count = inf_mask.sum()
            
            if inf_count > 0:
                result_df.loc[inf_mask, col] = fill_value
                logger.debug(f"填充 {col}: {inf_count} 个无穷大值")
        
        return result_df
    
    def fill_outliers(self, factors_df: pd.DataFrame, 
                     method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        填充异常值
        
        参数:
            factors_df: 因子DataFrame
            method: 异常值检测方法 ('iqr', 'zscore')
            threshold: 阈值 (IQR方法: 1.5-3.0, Z-score方法: 2.0-3.0)
        
        返回:
            填充后的因子DataFrame
        """
        result_df = factors_df.copy()
        
        for col in result_df.columns:
            if method == 'iqr':
                Q1 = result_df[col].quantile(0.25)
                Q3 = result_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (result_df[col] < lower_bound) | (result_df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    # 使用中位数填充异常值
                    median_val = result_df[col].median()
                    result_df.loc[outlier_mask, col] = median_val
                    logger.debug(f"填充 {col}: {outlier_count} 个异常值 (IQR方法)")
            
            elif method == 'zscore':
                mean_val = result_df[col].mean()
                std_val = result_df[col].std()
                
                if std_val > 0:
                    z_scores = np.abs((result_df[col] - mean_val) / std_val)
                    outlier_mask = z_scores > threshold
                    outlier_count = outlier_mask.sum()
                    
                    if outlier_count > 0:
                        # 使用均值填充异常值
                        result_df.loc[outlier_mask, col] = mean_val
                        logger.debug(f"填充 {col}: {outlier_count} 个异常值 (Z-score方法)")
        
        return result_df
    
    def _calculate_mean_from_similar_factors(self, factor: str, 
                                            factors_df: pd.DataFrame) -> float:
        """
        从相似因子计算均值
        
        参数:
            factor: 因子名称
            factors_df: 因子DataFrame
        
        返回:
            均值
        """
        # 简单实现：使用所有现有因子的均值
        if len(factors_df) == 0:
            return self.fill_value
        
        # 计算所有数值列的均值
        numeric_cols = factors_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return self.fill_value
        
        return factors_df[numeric_cols].values.mean()
    
    def _calculate_median_from_similar_factors(self, factor: str, 
                                              factors_df: pd.DataFrame) -> float:
        """
        从相似因子计算中位数
        
        参数:
            factor: 因子名称
            factors_df: 因子DataFrame
        
        返回:
            中位数
        """
        # 简单实现：使用所有现有因子的中位数
        if len(factors_df) == 0:
            return self.fill_value
        
        # 计算所有数值列的中位数
        numeric_cols = factors_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return self.fill_value
        
        return factors_df[numeric_cols].values.flatten().mean()
    
    def _log_filling_info(self, original_count: int, missing_count: int, 
                         target_count: int) -> None:
        """
        记录填充信息
        
        参数:
            original_count: 原始因子数
            missing_count: 缺失因子数
            target_count: 目标因子数
        """
        logger.info(f"因子填充: {original_count} → {target_count} "
                   f"(填充 {missing_count} 个缺失因子)")
    
    @staticmethod
    def get_all_possible_factors() -> Set[str]:
        """获取所有可能的因子"""
        return FactorFiller.ALL_POSSIBLE_FACTORS.copy()
    
    @staticmethod
    def get_factor_count() -> int:
        """获取因子总数"""
        return len(FactorFiller.ALL_POSSIBLE_FACTORS)


def fill_factors_with_defaults(factors_df: pd.DataFrame, 
                               fill_value: float = 0.0,
                               fill_method: str = 'zero') -> pd.DataFrame:
    """
    快速填充因子的便利函数
    
    参数:
        factors_df: 因子DataFrame
        fill_value: 填充值
        fill_method: 填充方法
    
    返回:
        填充后的因子DataFrame
    """
    filler = FactorFiller(fill_value=fill_value, fill_method=fill_method)
    
    # 填充缺失因子
    result_df = filler.fill_missing_factors(factors_df)
    
    # 填充NaN值
    result_df = filler.fill_nan_values(result_df, fill_method='zero')
    
    # 填充无穷大值
    result_df = filler.fill_inf_values(result_df, fill_value=fill_value)
    
    return result_df
