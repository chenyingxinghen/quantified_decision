"""
综合因子计算器
统一管理所有因子子模块，确保训练和预测时的因子计算逻辑完全一致
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import io
from typing import Dict, List, Optional, Tuple

from core.factors.quantitative_factors import QuantitativeFactors
from core.factors.candlestick_pattern_factors import CandlestickPatternFactors
from core.factors.fundamental_factors import FundamentalFactors
from core.factors.ml_factor_model import MLFactorModel
from core.factors.feature_engineering import FeatureEngineer
from core.factors.advanced_factors import TimeSeriesFactors, RiskFactors
from core.factors.factor_filler import FactorFiller
from config import DATABASE_PATH, FactorConfig

class ComprehensiveFactorCalculator:
    """综合因子计算器，整合所有因子计算模块"""
    
    def __init__(self, db_path: str = DATABASE_PATH, config: Optional[FactorConfig] = None):
        """初始化各子模块"""
        self.db_path = db_path
        self.factor_config = config if config is not None else FactorConfig
        self.factor_calculator = QuantitativeFactors(config=self.factor_config)
        self.candlestick_calculator = CandlestickPatternFactors()
        self.fundamental_calculator = FundamentalFactors(db_path)
        self.feature_engineer = FeatureEngineer()
        self.filler = FactorFiller(fill_value=0.0, fill_method='zero')
        
    def calculate_all_factors(self, code: str, data: pd.DataFrame, 
                             apply_feature_engineering: bool = True,
                             target_features: Optional[List[str]] = None,
                             verbose: bool = False,
                             include_fundamentals: bool = True) -> pd.DataFrame:
        """
        计算所有类型的因子并整合
        
        参数:
            code: 股票代码
            data: 股票数据 (OHLCV)
            apply_feature_engineering: 是否应用特征工程 (比率、乘积等变换)
            target_features: 目标特征列表 (如果提供，将确保输出包含这些特征)
            verbose: 是否打印详细日志
            include_fundamentals: 是否包含基本面因子
            
        返回:
            整合后的特征 DataFrame
        """
        # 1. 计算基础因子 (技术指标 + K线形态 + 基本面 + 高级特征)
        base_factors = self._calculate_base_factors(code, data, include_fundamentals=include_fundamentals)
        
        if base_factors is None or len(base_factors) == 0:
            return pd.DataFrame()
            
        # 2. 应用特征工程
        if apply_feature_engineering:
            all_factors = self._apply_feature_engineering(base_factors, verbose)
        else:
            all_factors = base_factors
            
        # 3. 填充缺失因子并对齐目标特征
        # 即使计算失败或由于数据不足无法计算，也要确保列存在，且没有 NaN/Inf
        all_factors = self.filler.fill_missing_factors(all_factors, target_factors=target_features, keep_all_generated=True)
        all_factors = self.filler.fill_nan_values(all_factors, fill_method='zero')
        all_factors = self.filler.fill_inf_values(all_factors, fill_value=0.0)
        
        # 4. 统一数据清理 (向量化操作)
        # 将整个 DataFrame 转换为数值类型以获得更好的性能
        all_factors = all_factors.apply(pd.to_numeric, errors='coerce').fillna(0)
        all_factors = all_factors.replace([np.inf, -np.inf], 0)
            
        # 5. 如果指定了目标特征且不仅是用来填充，则按目标特征排序/筛选
        if target_features:
            missing = [f for f in target_features if f not in all_factors.columns]
            if missing:
                if verbose:
                    print(f"  警告: 仍有 {len(missing)} 个特征无法生成，已进行批量填充")
                
                # 批量创建缺失列的 DataFrame 以避免碎片化
                missing_df = pd.DataFrame(0.0, index=all_factors.index, columns=missing)
                all_factors = pd.concat([all_factors, missing_df], axis=1)
                
            # 只保留目标特征列表中的特征，并保持顺序一致
            all_factors = all_factors[target_features]
            
        return all_factors

    def _calculate_base_factors(self, code: str, data: pd.DataFrame, include_fundamentals: bool = True) -> pd.DataFrame:
        """计算各类基础因子分支（抑制中间计算中的 RuntimeWarning）"""
        # 抑制 pandas/numpy 在中间计算（如 std, corr, subtract）时产生的 RuntimeWarning
        # 这些 NaN/inf 值已在后续步骤中被正确清理和替换
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow.*')
            
            # A. 技术指标 (MA, RSI, MACD 等)
            tech_factors = self.factor_calculator.calculate_all_factors(data)
            
            # B. K线形态
            candlestick_factors = self.candlestick_calculator.calculate_all_candlestick_patterns(data)
            
            # C. 基本面因子 (PE, ROE等)
            fundamental_factors = pd.DataFrame(index=data.index)
            if include_fundamentals:
                # 严重警告: 这里的 fundamental_calculator 目前仅返回实时快照数据。
                # 在历史数据上广播这些快照会导致严重的未来信息泄露（Look-ahead Bias）。
                # 只有在进行实盘/盘后实时预测，且 data 只有最新一行时，才允许使用。
                is_realtime_prediction = len(data) <= 5 # 启发式判断：如果是实时预测，行数通常很少
                
                if is_realtime_prediction:
                    fundamental_dict = self.fundamental_calculator.calculate_all_fundamental_factors(code)
                    if fundamental_dict:
                        fundamental_factors = pd.DataFrame([fundamental_dict] * len(data), index=data.index)
                        fundamental_factors = fundamental_factors.apply(pd.to_numeric, errors='coerce')
                else:
                    # 在历史回测或训练模式下，除非有 PIT 数据库，否则强行跳过基本面因子
                    # 这能防止模型通过“预知”未来的 PE 或市值来作弊
                    pass
                
            # D. 高级特征 (时间序列、风险) - 现在返回的是 Rolling DataFrames
            try:
                ts_price = TimeSeriesFactors.calculate_price_series_features(data)
                ts_vol = TimeSeriesFactors.calculate_volume_series_features(data)
                ts_mom = TimeSeriesFactors.calculate_momentum_features(data)
                risk = RiskFactors.calculate_risk_features(data)
                
                # 直接连接这些 DataFrames
                adv_factors = pd.concat([ts_price, ts_vol, ts_mom, risk], axis=1)
            except Exception as e:
                print(f"  警告: 计算高级因子失败: {e}")
                adv_factors = pd.DataFrame(index=data.index)
                
            # E. 交易状态因子 (涨停/停牌)
            status_factors = pd.DataFrame(index=data.index)
            # 获取 ST 标签：如果在 data 中存在则直接使用，否则通过 fundamental_calculator 查询数据库
            is_st = False
            if 'is_st' in data.columns:
                is_st = data['is_st'].iloc[0] == 1
            else:
                info = self.fundamental_calculator.get_stock_info(code)
                if info is not None:
                    is_st = info.get('is_st') == 1
            
            # ST 股涨停阈值取 4.5% (对应 5% 限制)，普通股取 9.3% (对应 10% 限制)
            limit_threshold = 0.045 if is_st else 0.093
            
            status_factors['is_limit_up'] = ((data['close'] == data['high']) & (data['close'].pct_change() > limit_threshold)).astype(int)
            # 停牌检测：成交量为 0
            status_factors['is_suspended'] = (data['volume'] == 0).astype(int)

            # 合并所有
            factors_list = [tech_factors, candlestick_factors, status_factors]
            if not fundamental_factors.empty:
                factors_list.append(fundamental_factors)
            if not adv_factors.empty:
                factors_list.append(adv_factors)
                
            # 确保索引对齐
            for i in range(len(factors_list)):
                factors_list[i].index = data.index
                
            factors = pd.concat(factors_list, axis=1)
            return factors

    def _apply_feature_engineering(self, base_factors: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """应用特征工程变换"""
        try:
            # 临时重置或配置 feature_engineer
            config = {
                'ratio': True, 'product': True, 'difference': True,
                'log': True, 'sqrt': True, 'rank': True,
                'interaction': True, 'categorical': True,
            }
            
            if not verbose:
                # 抑制冗余输出
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    engineered = self.feature_engineer.apply_all_transformations(base_factors, config=config)
                finally:
                    sys.stdout = old_stdout
            else:
                engineered = self.feature_engineer.apply_all_transformations(base_factors, config=config)
                
            return engineered
        except Exception as e:
            if verbose:
                print(f"  警告: 特征工程失败: {e}")
            return base_factors

    def get_feature_names(self, code: str, sample_data: pd.DataFrame) -> List[str]:
        """获取计算器能产生的所有特征名称列表"""
        factors = self.calculate_all_factors(code, sample_data, apply_feature_engineering=True)
        return factors.columns.tolist()
