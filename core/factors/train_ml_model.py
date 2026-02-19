"""
机器学习因子模型训练脚本

功能：
1. 加载历史数据
2. 计算量化因子
3. 准备训练数据
4. 训练多个模型
5. 模型评估和对比
6. 保存最佳模型
"""

import sys
import os
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 抑制因子计算过程中 pandas/numpy 在 NaN/inf 数据上执行 std/corr/subtract 时的 RuntimeWarning
# 这些中间警告是无害的，因为所有 NaN/inf 值已在后续数据清理步骤中被正确处理
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')

import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from joblib import Parallel, delayed

from core.factors.quantitative_factors import QuantitativeFactors
from core.factors.candlestick_pattern_factors import CandlestickPatternFactors
from core.factors.fundamental_factors import FundamentalFactors
from core.factors.ml_factor_model import MLFactorModel
from core.factors.comprehensive_factor_calculator import ComprehensiveFactorCalculator
from core.factors.advanced_factors import TimeSeriesFactors, RiskFactors
from core.factors.factor_filler import FactorFiller, fill_factors_with_defaults
from config import DATABASE_PATH, TrainingConfig, FactorConfig, YEARS

class MLModelTrainer:
    """机器学习模型训练器"""
    
    def __init__(self, db_path: str = DATABASE_PATH, punish_unbuyable: bool = False):
        """
        初始化训练器
        
        参数:
            db_path: 数据库路径
            punish_unbuyable: 是否使用样本权重（方案2）
        """
        self.db_path = db_path
        # 任务类型固定为 hybrid，模型内部会根据类型自动分配任务
        self.task = 'hybrid'
        print(f"模型训练任务类型已固定为: {self.task} (LGBM: ranking, XGB: regression)")
            
        self.punish_unbuyable = punish_unbuyable
        self.factor_calculator = ComprehensiveFactorCalculator(db_path)
        self.models = {}
        self.factors_cache_dir = TrainingConfig.CACHE_DIR
        os.makedirs(self.factors_cache_dir, exist_ok=True)

    @property
    def tech_calculator(self):
        """技术指标计算器"""
        return self.factor_calculator.factor_calculator

    @property
    def candlestick_calculator(self):
        """K线形态计算器"""
        return self.factor_calculator.candlestick_calculator

    @property
    def fundamental_calculator(self):
        """基本面计算器"""
        return self.factor_calculator.fundamental_calculator

    @property
    def feature_engineer(self):
        """特征工程器"""
        return self.factor_calculator.feature_engineer
    
    def load_training_data(self, stock_codes: List[str], 
                          start_date: str, end_date: str,
                          batch_size: int = 500) -> Dict[str, pd.DataFrame]:
        """
        加载训练数据（批量加载优化）
        
        参数:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            batch_size: 每批加载的股票数量（默认500）
        
        返回:
            股票数据字典
        """
        print(f"正在加载 {len(stock_codes)} 只股票的数据...")
        
        stocks_data = {}
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 提高查询效率
        
        # 分批加载，避免 IN 子句过长
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i+batch_size]
            placeholders = ','.join(['?' for _ in batch_codes])
            
            query = f'''
                SELECT code, date, open, high, low, close, volume, amount
                FROM daily_data
                WHERE code IN ({placeholders}) AND date >= ? AND date <= ?
                ORDER BY code, date ASC
            '''
            
            params = batch_codes + [start_date, end_date]
            df = pd.read_sql_query(query, conn, params=params)
            
            # 按股票分组
            for code in df['code'].unique():
                stock_df = df[df['code'] == code].copy()
                stock_df = stock_df.sort_values('date').reset_index(drop=True)
                
                if len(stock_df) >= 100:  # 至少100个交易日
                    stocks_data[code] = stock_df
            
            # 进度提示
            loaded_count = min(i + batch_size, len(stock_codes))
            print(f"  已加载: {loaded_count}/{len(stock_codes)} 只股票")
        
        conn.close()
        
        print(f"成功加载 {len(stocks_data)} 只股票的数据")
        return stocks_data
    
    def calculate_and_save_factors(self, code: str, data: pd.DataFrame, 
                                  apply_feature_engineering: bool = True,
                                  target_features: Optional[List[str]] = None,
                                  verbose: bool = False,
                                  include_fundamentals: bool = True) -> pd.DataFrame:
        """
        计算并保存单只股票的因子（使用统一分发的综合计算器）
        
        参数:
            code: 股票代码
            data: 股票数据
            apply_feature_engineering: 是否应用特征工程
            target_features: 目标特征列表
            verbose: 是否输出详细日志
            include_fundamentals: 是否包含基本面因子
        
        返回:
            合并后的因子DataFrame
        """
        cache_file = os.path.join(self.factors_cache_dir, f'{code}_factors.parquet')
        
        # 检查缓存
        if os.path.exists(cache_file):
            try:
                cached_factors = pd.read_parquet(cache_file)
                # 如果提供了target_features，检查缓存是否包含所有特征
                if target_features:
                    missing = [f for f in target_features if f not in cached_factors.columns]
                    if not missing and len(cached_factors) == len(data):
                        return cached_factors[target_features]
                elif len(cached_factors) == len(data):
                    return cached_factors
            except:
                pass
        
        # 使用统一的综合计算器
        all_factors = self.factor_calculator.calculate_all_factors(
            code=code, 
            data=data, 
            apply_feature_engineering=apply_feature_engineering,
            target_features=target_features,
            verbose=verbose,
            include_fundamentals=include_fundamentals
        )
        
        if all_factors.empty:
            return None
            
        # 在保存前确保包含日期列，以便回测时能正确对齐
        if 'date' in data.columns:
            all_factors = all_factors.copy()
            # 确保日期列不是 index，而是普通列
            all_factors['date'] = data['date'].values
            
        # 保存到缓存
        try:
            all_factors.to_parquet(cache_file, index=False)
            if verbose:
                print(f"  ✓ {code} 因子已缓存 ({len(all_factors.columns)} 个因子)")
        except Exception as e:
            if verbose:
                print(f"  ✗ 保存因子缓存失败 ({code}): {e}")
        
        return all_factors
    
    
    def _load_or_compute_factors(self, code: str, data: pd.DataFrame, 
                                apply_feature_engineering: bool = True,
                                target_features: Optional[List[str]] = None,
                                verbose: bool = False,
                                include_fundamentals: bool = True) -> pd.DataFrame:
        """
        加载缓存的因子或计算新因子（包括特征工程）
        
        参数:
            code: 股票代码
            data: 股票数据
            apply_feature_engineering: 是否应用特征工程
            target_features: 目标特征列表
            verbose: 是否输出详细日志
            include_fundamentals: 是否包含基本面因子
        
        返回:
            因子DataFrame或None
        """
        return self.calculate_and_save_factors(code, data, apply_feature_engineering, target_features, verbose, include_fundamentals)

    def _process_single_stock(self, code: str, data: pd.DataFrame, 
                             forward_days: int,
                             apply_feature_engineering: bool = False,
                             target_features: Optional[List[str]] = None,
                             verbose: bool = False,
                             train_start_date: str = None,
                             train_end_date: str = None,
                             include_fundamentals: bool = True) -> tuple:
        """
        处理单只股票的因子计算和标签生成
        """
        try:
            # 1. 加载或计算因子
            factors = self._load_or_compute_factors(code, data, apply_feature_engineering, target_features, verbose, include_fundamentals)
            
            if factors is not None and len(factors) > forward_days:

                # 获取价格序列
                close = data['close']
                high = data['high']
                low = data['low']
                
                
                # A. 收益率计算 (未来 n 日收盘涨幅)
                f_close = close.shift(-forward_days)
                f_returns = (f_close / close - 1)
                
                # 获取未来 n 日内的最大涨幅 (Max Run-up)
                f_high_max = high.shift(-forward_days).rolling(window=forward_days, min_periods=1).max()
                f_max_returns = (f_high_max / close - 1)
                
                # 获取未来 n 日内的最大跌幅 (Max Drawdown/Pain)
                f_low_min = low.shift(-forward_days).rolling(window=forward_days, min_periods=1).min()
                f_min_returns = (f_low_min / close - 1)

                # 1. 路径质量分 (Path-aware Score)
                # 最终收益权重 1.0, 过程最大高点分红 0.2, 过程最大低点惩罚 1.5
                # 显著惩罚回撤大、先跌后涨的标的，引导模型选择“走势稳健”的头部标的
                y = f_returns + 0.5 * f_max_returns + 1.0 * f_min_returns
                
                # 用于计算 IC 的参考收益率 (使用最终涨幅)
                ref_returns = f_returns.values
                target_returns = f_returns.values

                # 3. 对齐数据
                # 必须过滤掉因子 NaN、目标收益率 NaN 以及标签 y 中的 NaN (ATR 可能产生前置 NaN)
                y_series = pd.Series(y, index=data.index)
                target_series = pd.Series(target_returns, index=data.index)
                
                valid_idx = ~(factors.isna().any(axis=1) | 
                              pd.Series(ref_returns, index=data.index).isna() | 
                              y_series.isna() | 
                              target_series.isna())
                
                # 4. 时间窗口切片
                if train_start_date or train_end_date:
                    date_series = data['date']
                    if train_start_date:
                        valid_idx = valid_idx & (date_series >= train_start_date)
                    if train_end_date:
                        valid_idx = valid_idx & (date_series <= train_end_date)
                
                if valid_idx.sum() > 0:
                    X_df = factors[valid_idx].copy()
                    dates = data['date'][valid_idx].values
                    
                    # 最终使用的标签和收益率
                    # 修复：确保 final_y 是 Series 以支持 prepare_dataset 中的 pd.concat
                    y_val = y[valid_idx] if isinstance(y, pd.Series) else y[valid_idx]
                    final_y = pd.Series(y_val) if not isinstance(y_val, pd.Series) else y_val
                    
                    final_returns = target_returns[valid_idx] if isinstance(target_returns, pd.Series) else target_returns[valid_idx]
                    
                    # 封板/停牌检测
                    is_limit_up = (data['close'] == data['high']) & (data['close'].pct_change() > 0.093)
                    is_suspended = data['volume'] == 0
                    unbuyable_mask = (is_limit_up | is_suspended)[valid_idx].values
                    
                    if 'date' in X_df.columns:
                        X_df = X_df.drop(columns=['date'])
                        
                    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    if len(X_df) > 0:
                        return X_df, final_y, final_returns, dates, unbuyable_mask
            
            return None, None, None, None, None
        
        except Exception as e:
            import traceback
            print(f"  警告: 处理股票 {code} 失败: {e}")
            print(traceback.format_exc())
            return None, None, None, None, None

    def _validate_and_filter_stocks(self, stocks_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """
        验证并过滤特征不完整的股票
        
        参数:
            stocks_data: 股票数据字典
        
        返回:
            (过滤后的股票数据, 验证统计信息)
        """
        print("\n验证缓存特征完整性...")
        
        # 加载模型特征
        model_features = set(self.model.feature_names) if hasattr(self, 'model') and self.model else None
        
        if not model_features:
            # 如果没有模型，从第一个缓存文件推断
            cache_files = [f for f in os.listdir(self.factors_cache_dir) if f.endswith('.parquet')]
            if cache_files:
                try:
                    sample_factors = pd.read_parquet(os.path.join(self.factors_cache_dir, cache_files[0]))
                    # 排除非数值列（如 date）进行特征匹配
                    numeric_cols = sample_factors.select_dtypes(include=[np.number]).columns
                    model_features = set(numeric_cols)
                except:
                    model_features = None
        
        if not model_features:
            print("  警告: 无法获取模型特征，尝试实时计算")
            return stocks_data, {'filtered': 0, 'kept': len(stocks_data)}
        
        filtered_stocks = {}
        filtered_count = 0
        recomputed_count = 0
        
        for code, data in stocks_data.items():
            # 检查缓存
            cache_file = os.path.join(self.factors_cache_dir, f'{code}_factors.parquet')
            
            if os.path.exists(cache_file):
                try:
                    # 使用 pyarrow 引擎快速读取元数据或部分数据进行检查
                    # 为了检查 NaNs，我们仍需要读取数值列
                    factors = pd.read_parquet(cache_file)
                    
                    # 仅对数值列进行检查
                    numeric_factors = factors.select_dtypes(include=[np.number])
                    cache_features = set(numeric_factors.columns)
                    
                    # 检查特征完整性 (子集匹配即可，允许缓存特征多于模型特征)
                    is_feature_match = (model_features.issubset(cache_features))
                    
                    # 只有当特征不匹配时，才标记为重新计算
                    # 不再因为 NaN 或 Inf 删文件，因为技术指标产生前置 NaN 是正常的
                    # 且后续计算流程中会有 fillna(0) 处理
                    if is_feature_match:
                        filtered_stocks[code] = data
                    else:
                        # 只有特征不匹配才计入重新计算
                        filtered_stocks[code] = data
                        recomputed_count += 1
                        
                except Exception as e:
                    # 缓存读取失败，说明文件可能损坏，保留并重新计算
                    filtered_stocks[code] = data
                    recomputed_count += 1
                    try: os.remove(cache_file) # 仅在读取失败（损坏）时尝试删除
                    except: pass
            else:
                # 缓存不存在，保留该股票（会实时计算）
                filtered_stocks[code] = data
        
        kept_count = len(filtered_stocks)
        print(f"  验证完成: {kept_count} 只股票待处理")
        if recomputed_count > 0:
            print(f"  其中 {recomputed_count} 只股票的缓存无效或不匹配，将重新计算")
        
        return filtered_stocks, {'filtered': filtered_count, 'recomputed': recomputed_count, 'kept': kept_count}
    
    def prepare_dataset(self, stocks_data: Dict[str, pd.DataFrame],
                       forward_days: int = None,
                       n_jobs: int = 15, 
                       enable_feature_engineering: bool = False,
                       cache_engineered_features: bool = True,
                       filter_incomplete_cache: bool = True,
                       train_start_date: str = None,
                       train_end_date: str = None,
                       include_fundamentals: bool = True) -> tuple:
        """
        准备训练数据集
        
        参数:
            stocks_data: 股票数据字典（应包含全量日期以生成缓存）
            forward_days: 预测未来N天
            n_jobs: 并行任务数
            enable_feature_engineering: 是否启用特征工程
            cache_engineered_features: 是否缓存特征工程结果
            filter_incomplete_cache: 是否过滤不完整缓存
            train_start_date: 训练样本开始日期
            train_end_date: 训练样本结束日期
            include_fundamentals: 是否包含基本面因子（注意：当前数据库仅含实时基本面，在历史训练中使用会导致数据泄露）
        
        返回:
            (X, y, returns, factor_names, dates)
        """
        # if include_fundamentals:
        #     print("\n" + "!"*80)
        #     print("警告: 正在使用基本面因子进行训练。")
        #     print("注意: 当前数据库 (stock_info_extended) 仅包含股票的实时基本面数据。")
        #     print("在历史数据上训练时，使用实时基本面会导致严重的数据泄露（Look-ahead Bias）。")
        #     print("建议: 如果是纯粹为了验证模型逻辑，可以继续；如果是为了回测实战，请考虑禁用基本面因子。")
        #     print("!"*80 + "\n")

        # print("\n正在计算量化因子（技术指标 + K线形态 + 基本面）...")
        
        # 使用配置中的默认值
        if forward_days is None:
            forward_days = TrainingConfig.FUTURE_DAYS
        
        # 过滤特征不完整的股票
        if filter_incomplete_cache:
            stocks_data, filter_stats = self._validate_and_filter_stocks(stocks_data)
        
        # 检查缓存情况
        cache_info = self.get_cache_info()
        print(f"  缓存状态: {cache_info['cached_stocks']}/{len(stocks_data)} 只股票已缓存")
        
        if cache_engineered_features:
            print(f"  缓存策略: 保存特征工程后的完整因子")
            
            # 特征发现：识别完整的特征集
            print("  正在识别完整特征集...")
            all_possible_features = set()
            # 选取具有代表性的股票（包含不同板块和充足历史数据）
            discovery_codes = list(stocks_data.keys())[:30]  # 增加搜索范围到30只
            for code in discovery_codes:
                try:
                    # 获取该股票最长的一段连续数据
                    discovery_data = stocks_data[code]
                    if len(discovery_data) < 200: continue # 跳过数据太少的
                    
                    # 预先探测所有可能的列
                    f = self.factor_calculator.calculate_all_factors(
                        code, discovery_data, apply_feature_engineering=True, include_fundamentals=include_fundamentals
                    )
                    if not f.empty:
                        all_possible_features.update(f.columns)
                except Exception as e:
                    continue
            
            target_features = sorted([f for f in all_possible_features if f != 'date'])
            print(f"  识别到总计 {len(target_features)} 个特征（含特征工程生成项）")
        else:
            print(f"  缓存策略: 仅保存基础因子，训练时应用特征工程（方案A）")
            target_features = None
        
        # 使用 joblib 并行处理
        print(f"  使用 {n_jobs if n_jobs > 0 else '所有'} CPU核心进行并行计算")
        
        # 创建一个包装函数，用于第一个股票输出日志
        stock_list = list(stocks_data.items())
        
        def process_with_logging(idx, code, data):
            """处理单只股票，第一个输出日志"""
            verbose = (idx == 0) and cache_engineered_features
            return self._process_single_stock(code, data, forward_days, 
                                             cache_engineered_features, target_features, verbose,
                                             train_start_date, train_end_date, include_fundamentals)
        
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_with_logging)(i, code, data)
            for i, (code, data) in enumerate(stock_list)
        )
        
        # 过滤有效结果
        all_factors = []
        all_labels = []
        all_returns = []
        all_dates = []
        all_unbuyable = []
        
        for result in results:
            if result[0] is not None:
                # 兼容旧版本或失败情况（如果是4个返回值）
                if len(result) == 5:
                    X, y, r, d, u = result
                else:
                    X, y, r, d = result[:4]
                    u = np.zeros(len(y), dtype=bool) if y is not None else None
                
                if X is not None:
                    all_factors.append(X)
                    all_labels.append(y)
                    all_returns.append(r)
                    all_dates.append(d)
                    all_unbuyable.append(u)
        
        # 合并所有数据
        X_combined = pd.concat(all_factors, axis=0, ignore_index=True)
        y_combined = pd.concat(all_labels, axis=0, ignore_index=True)
        returns_combined = np.concatenate(all_returns)
        dates_combined = np.concatenate(all_dates)
        unbuyable_combined = np.concatenate(all_unbuyable)
        
        # 关键修复：按时间排序（解决全局数据泄露问题）
        # 确保训练集完全在验证集之前，而不是按照股票列表的顺序
        sort_idx = np.argsort(dates_combined)
        X_combined = X_combined.iloc[sort_idx].reset_index(drop=True)
        y_combined = y_combined.iloc[sort_idx].reset_index(drop=True)
        returns_combined = returns_combined[sort_idx]
        dates_combined = dates_combined[sort_idx]
        unbuyable_combined = unbuyable_combined[sort_idx]
        
        # 应用：按日横向归一化 (Cross-sectional Normalization)
        # 彻底消除市场整体涨跌（Beta）对因子和标签的影响，强制模型学习选股（Alpha）
        unique_dates = np.unique(dates_combined)
        print(f"\n应用：进行每日横向处理 (共 {len(unique_dates)} 个交易日)...")
        
        # 1. 因子归一化 (Z-Score)
        print("  - 因子横向 Z-Score 归一化 (向量化优化)...")
        # 暂时将日期加入以利用 groupby 性能
        X_combined['temp_date'] = dates_combined
        # 转换为 float64 确保计算精度，并计算分组统计量
        X_combined_float = X_combined.drop(columns=['temp_date']).astype(np.float64)
        grouped = X_combined_float.groupby(dates_combined)
        
        # 批量执行 Z-Score：减去组内均值并除以组内标准差
        # 使用 transform('mean') 和 transform('std') 是经过优化的内置路径
        X_combined = (X_combined_float - grouped.transform('mean')) / grouped.transform('std').replace(0, 1.0)
        X_combined = X_combined.fillna(0)
        
        # 2. 标签百分位归一化 (针对 XGBoost 回归任务)
        print("  - 标签横向百分位排名归一化 (向量化优化)...")
        # 使用同样的 groupby 逻辑处理标签
        y_df = pd.DataFrame({'y': y_combined.values, 'date': dates_combined})
        y_norm = y_df.groupby('date')['y'].transform(
            lambda x: (x.rank(method='average', pct=True) - (0.5 / len(x))) if len(x) > 0 else 0.5
        )
        y_combined = pd.Series(y_norm.values)
        
        # 统计因子分类详情 (Factor Audit Report)
        all_cols = X_combined.columns.tolist()
        
        tech_cols = [c for c in all_cols if any(x in c.lower() for x in self.tech_calculator.get_factor_names())]
        candle_cols = [c for c in all_cols if any(x in c.lower() for x in self.candlestick_calculator.get_pattern_names())]
        engineered_cols = [c for c in all_cols if c in self.feature_engineer.get_generated_features()]
        
        # 排除已识别的分类，剩下的通常是基础基本面因子或高级特征
        remaining_cols = set(all_cols) - set(tech_cols) - set(candle_cols) - set(engineered_cols)
        
        # 识别高级/风险特征
        adv_keywords = ['volatility', 'return_', 'momentum_', 'sharpe', 'drawdown', 'skewness', 'kurtosis', 'acceleration', 'position']
        adv_cols = [c for c in remaining_cols if any(k in c.lower() for k in adv_keywords)]
        
        # 剩下的就是纯基本面或状态因子
        fund_cols = list(remaining_cols - set(adv_cols))

        print("\n" + "="*50)
        print("数据集审计报告 (Factor Audit Report)")
        print("="*50)
        print(f"总样本数: {len(X_combined):<10} | 正样本率: {y_combined.mean():.2%}")
        print(f"总特征数: {len(all_cols):<10} | 无法买入: {np.mean(unbuyable_combined):.2%}")
        print("-" * 50)
        print(f"1. 技术指标 (Technical):    {len(tech_cols):>4} 个")
        print(f"2. K线形态 (Candlestick):   {len(candle_cols):>4} 个")
        print(f"3. 基础基本面 (Fundamental): {len(fund_cols):>4} 个")
        print(f"4. 高级时序 (Advanced):     {len(adv_cols):>4} 个")
        print(f"5. 特征工程 (Engineered):   {len(engineered_cols):>4} 个")
        print("="*50 + "\n")
        
        return X_combined.values.astype(np.float64), y_combined.values, returns_combined, all_cols, dates_combined, unbuyable_combined
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    returns: np.ndarray,
                    factor_names: List[str],
                    dates: np.ndarray,
                    unbuyable_mask: np.ndarray = None,
                    model_types: List[str] = ['xgboost', 'lightgbm']) -> Dict:
        """
        训练多个模型
        
        参数:
            X: 特征矩阵
            y: 标签向量
            returns: 原始收益率（用于计算权重或排序评价）
            factor_names: 特征名称列表
            dates: 样本日期（用于排序组划分）
            unbuyable_mask: 无法买入的样本掩码（涨停或停牌）
            model_types: 要训练的模型类型列表
        
        返回:
            训练结果字典
        """
        # 数据验证和清理
        print("\n数据验证...")
        
        # 确保X是numpy数组
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # 确保数据类型正确
        X = X.astype(np.float64)
        y = y.astype(np.float32)  # 支持软标签，不能强制转 int32
        
        # 最后一次NaN/inf检查和替换
        if np.isnan(X).any():
            print(f"  警告: 发现 {np.isnan(X).sum()} 个NaN值，进行替换")
            X = np.nan_to_num(X, nan=0.0)
        
        if np.isinf(X).any():
            print(f"  警告: 发现 {np.isinf(X).sum()} 个无穷大值，进行替换")
            X[np.isinf(X)] = 0.0
        
        print(f"  数据验证完成: {X.shape[0]} 行, {X.shape[1]} 列")
        if self.task == 'classification':
            print(f"  正样本比例: {y.mean():.2%}")
        
        # 准备样本权重
        sample_weight = None
        if self.punish_unbuyable:
            print(f"\n样本权重: weight = abs(returns)")
            sample_weight = np.abs(returns)
            
            # 惩罚无法买入的样本（如涨停或停牌）
            if unbuyable_mask is not None:
                print(f"  惩罚无法买入的样本: 权重降低为 10%")
                sample_weight = np.where(unbuyable_mask, sample_weight * 0.1, sample_weight)
            
            # 缩放权重，避免数值过大
            sample_weight = sample_weight / sample_weight.mean()
        
        results = {}
        
        for model_type in model_types:
            print(f"\n{'='*80}")
            print(f"训练 {model_type.upper()} 模型")
            print(f"{'='*80}")
            
            try:
                # 显式指定任务类型：LGBM 使用 ranking，XGBoost 使用 regression
                # 模型内部 __init__ 也会自动处理，这里保持一致
                task = 'ranking' if model_type == 'lightgbm' else 'regression'
                model = MLFactorModel(model_type=model_type, task=task)
                
                # 统一计算分组信息和 split_idx (所有任务通用，用于按组评估)
                extra_params = {}
                
                # 关键修复：确保 split_idx 对齐到日期边界 (所有任务一致)
                raw_split_idx = int(len(dates) * 0.8)
                split_date = dates[raw_split_idx]
                split_idx = np.searchsorted(dates, split_date, side='left')
                
                train_dates = dates[:split_idx]
                val_dates = dates[split_idx:]
                
                _, train_group = np.unique(train_dates, return_counts=True)
                _, val_group = np.unique(val_dates, return_counts=True)
                
                extra_params['dates'] = dates  # 传递日期用于按组评估
                extra_params['split_idx'] = split_idx  # 传递对齐后的 split 点
                
                if task == 'ranking':
                    extra_params['group'] = train_group
                    extra_params['eval_group'] = val_group
                    print(f"  排序任务: 分割点对齐 - 原始={raw_split_idx}, 对齐后={split_idx} (日期边界: {split_date})")
                    print(f"  排序任务: 分组数量 - 训练集 {len(train_group)}, 验证集 {len(val_group)}")
                else:
                    print(f"  {task}任务评估准备: 分割点对齐于日期 {split_date}")
                
                # 训练模型
                train_result = model.train(X, y, validation_split=0.2, 
                                          use_time_series_split=True,
                                          feature_names=factor_names,
                                          sample_weight=sample_weight,
                                          returns=returns,
                                          **extra_params)
                
                # 保存模型
                self.models[model_type] = model
                results[model_type] = train_result
                
            except Exception as e:
                import traceback
                print(f"训练 {model_type} 失败: {e}")
                print(traceback.format_exc())
                continue
        
        return results
    
    def compare_models(self, results: Dict):
        """对比模型性能"""
        if not results:
            print("\n警告: 没有模型训练成功，无法进行性能对比。")
            return None
            
        print(f"\n{'='*80}")
        print("模型性能对比 (核心指标: Rank IC & Top-N 精度)")
        print(f"{'='*80}")
        
        comparison = []
        for model_type, result in results.items():
            val_metrics = result['val_metrics']
            
            # 基础指标
            row = {
                '模型': model_type,
                '任务': '排序' if model_type == 'lightgbm' else '回归(软标签)',
                'Rank IC': f"{val_metrics.get('rank_ic', 0.0):.4f}",
                'Top-1精度': f"{val_metrics.get('top1_precision', 0.0):.2%}",
                'Top-5精度': f"{val_metrics.get('top5_precision', 0.0):.2%}",
            }
            
            # 补充任务特有指标
            if model_type == 'lightgbm':
                row['NDCG@5'] = f"{val_metrics.get('ndcg@5', 0.0):.4f}"
                row['辅助指标'] = f"IC_Std: {val_metrics.get('rank_ic_std', 0.0):.4f}"
            else:
                row['NDCG@5'] = "N/A"
                row['辅助指标'] = f"AUC: {val_metrics.get('auc', 0.0):.4f}"
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        # 按 Rank IC 排序
        df = df.sort_values('Rank IC', ascending=False)
        print(df.to_string(index=False))
        
        # 选股策略下，最佳模型应基于 Rank IC 或 Top-1 精度
        best_model = max(results.items(), key=lambda x: x[1]['val_metrics'].get('rank_ic', -1.0))
        print(f"\n最佳选股模型: {best_model[0].upper()} (Rank IC: {best_model[1]['val_metrics'].get('rank_ic', 0.0):.4f})")
        
        return best_model[0]
    
    def save_models(self, save_dir: str = 'models', years: int = 5, stocks: int = 5000):
        """
        保存所有训练好的模型，并根据任务、天数、阈值等元数据自动归档
        
        参数:
            save_dir: 基础保存目录
            years: 训练数据年数
            stocks: 训练股票数量
            
        返回:
            归档目录路径
        """
        if not self.models:
            print("警告: 没有已训练的模型可以保存。")
            return None
            
        # 1. 任务类型 (分类/回归/排序)
        task_str = self.task
        
        # 2. 预测天数
        forward_days = getattr(TrainingConfig, 'FUTURE_DAYS', 3)
        

        
        # 4. 标签类型 (软标签/硬标签)
        label_type = 'soft' if getattr(TrainingConfig, 'LABEL_SOFTENING', False) else 'hard'
        
        # 5. 权重状态
        weight_status = 'punish' if self.punish_unbuyable else 'unpunish'
        
        # 6. 数据体量
        data_volume = f"{years}y_{stocks}s"
        
        # 7. 当前时间戳 (增加唯一性)
        timestamp = datetime.now().strftime('%m%d_%H%M')
        
        # 例如: train_classification_3d_5pct_punish_soft_5y_500s_0213_2130
        archive_name = f"train_{task_str}_{forward_days}d_{weight_status}_{label_type}_{data_volume}_{timestamp}"
        
        archive_dir = os.path.join(save_dir, archive_name)
        os.makedirs(archive_dir, exist_ok=True)
        
        for model_type, model in self.models.items():
            filepath = os.path.join(archive_dir, f'{model_type}_factor_model.pkl')
            model.save_model(filepath)
            
        # 8. 同时更新一个 "latest" 目录，方便自动调用
        latest_dir = os.path.join(save_dir, 'latest')
        import shutil
        if os.path.exists(latest_dir):
            try: shutil.rmtree(latest_dir)
            except: pass
        try:
            shutil.copytree(archive_dir, latest_dir)
            print(f"  ✓ 已同步至最新目录: {latest_dir}")
        except Exception as e:
            print(f"  ！同步最新目录失败: {e}")
            
        return archive_dir
    
    def save_factor_summary(self, factor_names: List[str], save_dir: str = 'models'):
        """保存因子汇总信息"""
        os.makedirs(save_dir, exist_ok=True)
        
        tech_factor_count = len(self.tech_calculator.get_factor_names())
        candlestick_factor_count = len(self.candlestick_calculator.get_pattern_names())
        base_factor_count = tech_factor_count + candlestick_factor_count
        fundamental_factor_count = max(0, len(factor_names) - base_factor_count - len(self.feature_engineer.get_generated_features()))
        engineered_factor_count = len(self.feature_engineer.get_generated_features())
        
        summary = {
            'total_factors': len(factor_names),
            'technical_factors': tech_factor_count,
            'candlestick_factors': candlestick_factor_count,
            'fundamental_factors': fundamental_factor_count,
            'engineered_factors': engineered_factor_count,
            'factor_names': factor_names,
            'technical_factor_names': self.tech_calculator.get_factor_names(),
            'candlestick_factor_names': self.candlestick_calculator.get_pattern_names(),
            'engineered_factor_names': self.feature_engineer.get_generated_features(),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        import json
        filepath = os.path.join(save_dir, 'factor_summary.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n因子汇总已保存: {filepath}")
    
    def clear_factors_cache(self):
        """清理因子缓存"""
        import shutil
        if os.path.exists(self.factors_cache_dir):
            shutil.rmtree(self.factors_cache_dir)
            os.makedirs(self.factors_cache_dir, exist_ok=True)
            print(f"已清理因子缓存: {self.factors_cache_dir}")
    
    def get_cache_info(self) -> Dict:
        """获取缓存信息"""
        if not os.path.exists(self.factors_cache_dir):
            return {'cached_stocks': 0, 'cache_size_mb': 0}
        
        cached_files = [f for f in os.listdir(self.factors_cache_dir) if f.endswith('.parquet')]
        total_size = sum(
            os.path.getsize(os.path.join(self.factors_cache_dir, f)) 
            for f in cached_files
        )
        
        return {
            'cached_stocks': len(cached_files),
            'cache_size_mb': total_size / (1024 * 1024)
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='机器学习因子模型训练')

    parser.add_argument('--cache-engineered', action='store_true',
                       help='在缓存时就应用特征工程（ 默认）')
    parser.add_argument('--no-cache-engineered', dest='cache_engineered', action='store_false',
                       help='仅缓存基础因子，训练时应用特征工程（方案A）')
    parser.set_defaults(cache_engineered=True)
    
    args = parser.parse_args()
    
    print("="*80)
    print("机器学习因子模型训练（整合技术指标 + K线形态因子）")
    print(f"涨停板惩罚: {'开启' if TrainingConfig.PUNISH_UNBUYABLE else '关闭'} | 缓存工程: {'开启' if args.cache_engineered else '关闭'}")
    print("="*80)
    
    # 1. 初始化训练器
    trainer = MLModelTrainer(punish_unbuyable=TrainingConfig.PUNISH_UNBUYABLE)
    
    # 显示缓存信息
    cache_info = trainer.get_cache_info()
    print(f"\n因子缓存信息:")
    print(f"  缓存目录: {trainer.factors_cache_dir}")
    print(f"  已缓存股票: {cache_info['cached_stocks']}")
    print(f"  缓存大小: {cache_info['cache_size_mb']:.2f} MB")
    
    # 2. 获取股票列表（示例：从数据库获取所有股票）
    conn = sqlite3.connect(DATABASE_PATH)
    stock_codes_df = pd.read_sql_query(
        f"SELECT DISTINCT code FROM daily_data ORDER BY RANDOM() LIMIT {TrainingConfig.STOCK_NUM}", 
        conn
    )
    conn.close()
    stock_codes = stock_codes_df['code'].tolist()
    
    # 3. 设置训练时间范围（用于过滤训练样本，但不限制数据加载）
    train_start_date = (datetime.now() - timedelta(365*YEARS)).strftime('%Y-%m-%d')
    train_end_date = ((datetime.now() - timedelta(365*YEARS)) + timedelta(365*TrainingConfig.YEARS_FOR_TRAINING)).strftime('%Y-%m-%d')

    # 3.5 设置数据加载/缓存的时间范围（加载全量以支持回测）
    all_data_start = "2016-01-01" 
    all_data_end = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n时间范围配置:")
    print(f"  缓存/加载时段: {all_data_start} 至 {all_data_end}")
    print(f"  训练/模型时段: {train_start_date} 至 {train_end_date}")
    
    # 4. 加载全量数据（为了计算完整周期的因子）
    stocks_data = trainer.load_training_data(stock_codes, all_data_start, all_data_end)
    

    
    # 5. 准备数据集
    X, y, returns, factor_names, dates, unbuyable = trainer.prepare_dataset(
        stocks_data,
        cache_engineered_features=args.cache_engineered,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS
    )
    
    # 6. 训练模型
    model_types = ['xgboost', 'lightgbm']
    results = trainer.train_models(X, y, returns, factor_names, dates, unbuyable, model_types)
    
    # 7. 对比模型
    best_model_type = trainer.compare_models(results)
    
    if best_model_type is None:
        print("\n[错误] 模型训练全部失败，请检查数据或参数设置。")
        return
    
    # 8. 保存模型
    print("\n保存模型...")
    archive_dir = trainer.save_models(
        save_dir=TrainingConfig.SAVE_DIR, 
        years=TrainingConfig.YEARS_FOR_TRAINING, 
        stocks=len(stock_codes)
    )
    
    # 9. 保存因子汇总
    trainer.save_factor_summary(factor_names, save_dir=archive_dir)
    
    
    # 显示最终缓存信息
    cache_info = trainer.get_cache_info()
    
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    print(f"模型保存于: {archive_dir}/")
    print(f"因子缓存: {trainer.factors_cache_dir}/ ({cache_info['cached_stocks']} 只股票, {cache_info['cache_size_mb']:.2f} MB)")
    print(f"因子汇总: {archive_dir}/factor_summary.json")
    
    if args.cache_engineered:
        print(f"\n✓ 缓存包含完整特征")
        print(f"  回测时可以直接使用缓存，无需特征工程")
        print(f"  建议: 回测时设置 enable_feature_engineering=False")
    else:
        print(f"\n✓ 缓存仅包含基础因子（方案A）")
        print(f"  回测时需要启用特征工程")
        print(f"  建议: 回测时设置 enable_feature_engineering=True")
    
    print(f"\n提示: 使用 trainer.clear_factors_cache() 可清理缓存")


if __name__ == '__main__':
    main()
