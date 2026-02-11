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
    
    def __init__(self, db_path: str = DATABASE_PATH, task: str = 'classification', use_sample_weight: bool = False):
        """
        初始化训练器
        
        参数:
            db_path: 数据库路径
            task: 任务类型 ('classification', 'regression', 'ranking')
            use_sample_weight: 是否使用样本权重（方案2）
        """
        self.db_path = db_path
        self.task = task
        self.use_sample_weight = use_sample_weight
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
                                  verbose: bool = False) -> pd.DataFrame:
        """
        计算并保存单只股票的因子（使用统一分发的综合计算器）
        
        参数:
            code: 股票代码
            data: 股票数据
            apply_feature_engineering: 是否应用特征工程
            target_features: 目标特征列表
            verbose: 是否输出详细日志
        
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
            verbose=verbose
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
                                verbose: bool = False) -> pd.DataFrame:
        """
        加载缓存的因子或计算新因子（包括特征工程）
        
        参数:
            code: 股票代码
            data: 股票数据
            apply_feature_engineering: 是否应用特征工程
            target_features: 目标特征列表
            verbose: 是否输出详细日志
        
        返回:
            因子DataFrame或None
        """
        return self.calculate_and_save_factors(code, data, apply_feature_engineering, target_features, verbose)

    def _process_single_stock(self, code: str, data: pd.DataFrame, 
                             forward_days: int, threshold: float,
                             apply_feature_engineering: bool = False,
                             target_features: Optional[List[str]] = None,
                             verbose: bool = False,
                             train_start_date: str = None,
                             train_end_date: str = None) -> tuple:
        """
        处理单只股票的因子计算和标签生成
        
        参数:
            code: 股票代码
            data: 股票数据
            forward_days: 预测未来N天
            threshold: 分类阈值
            apply_feature_engineering: 是否在缓存时应用特征工程
            target_features: 目标特征列表
            verbose: 是否输出详细日志
            train_start_date: 训练集开始日期
            train_end_date: 训练集结束日期
        
        返回:
            (X, y, returns, date_series) 或 (None, None, None, None) 如果处理失败
        """
        try:
            # 1. 加载或计算因子（这里会处理全量数据并保存缓存）
            factors = self._load_or_compute_factors(code, data, apply_feature_engineering, target_features, verbose)
            
            if factors is not None and len(factors) > forward_days:
                # 2. 计算未来收益率（全量）
                future_returns = data['close'].pct_change(forward_days).shift(-forward_days)
                
                # 3. 对齐数据
                valid_idx = ~(factors.isna().any(axis=1) | future_returns.isna())
                
                # 4. 关键修改：根据训练时间窗口进行切片
                if train_start_date or train_end_date:
                    date_series = data['date']
                    if train_start_date:
                        valid_idx = valid_idx & (date_series >= train_start_date)
                    if train_end_date:
                        valid_idx = valid_idx & (date_series <= train_end_date)
                
                if valid_idx.sum() > 0:
                    X_df = factors[valid_idx].copy()
                    
                    # 记录日期以便后续按日期分组（排序任务需要）
                    dates = data['date'][valid_idx].values
                    returns = future_returns[valid_idx].values
                    
                    # 移除日期列，不参与训练
                    if 'date' in X_df.columns:
                        X_df = X_df.drop(columns=['date'])
                        
                    if self.task == 'classification':
                        y = (future_returns[valid_idx] > threshold).astype(int)
                    else:
                        # 回归或排序任务：直接使用未来收益率
                        y = future_returns[valid_idx]
                    
                    # 数据验证：确保没有无穷大值
                    X_df = X_df.replace([np.inf, -np.inf], np.nan)
                    X_df = X_df.fillna(0)
                    
                    # 检查是否有有效数据
                    if len(X_df) > 0:
                        return X_df, y, returns, dates
            
            return None, None, None, None
        
        except Exception as e:
            print(f"  警告: 处理股票 {code} 失败: {e}")
            return None, None

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
                    factors = pd.read_parquet(cache_file)
                    # 仅对数值列进行检查
                    numeric_factors = factors.select_dtypes(include=[np.number])
                    cache_features = set(numeric_factors.columns)
                    
                    # 检查特征完整性 (子集匹配即可，允许缓存特征多于模型特征)
                    is_feature_match = (model_features.issubset(cache_features))
                    has_nan = numeric_factors.isna().any().any()
                    has_inf = np.isinf(numeric_factors.values).any()
                    
                    if is_feature_match and not has_nan and not has_inf:
                        filtered_stocks[code] = data
                    else:
                        # 缓存无效或不匹配，保留股票代码以便实时重新计算
                        filtered_stocks[code] = data
                        recomputed_count += 1
                        # 如果缓存明显有问题（如有 NaN/Inf），可选删除它
                        if has_nan or has_inf:
                            try: os.remove(cache_file)
                            except: pass
                except Exception as e:
                    # 缓存读取失败，也保留并重新计算
                    filtered_stocks[code] = data
                    recomputed_count += 1
            else:
                # 缓存不存在，保留该股票（会实时计算）
                filtered_stocks[code] = data
        
        kept_count = len(filtered_stocks)
        print(f"  验证完成: {kept_count} 只股票待处理")
        if recomputed_count > 0:
            print(f"  其中 {recomputed_count} 只股票的缓存无效或不匹配，将重新计算")
        
        return filtered_stocks, {'filtered': filtered_count, 'recomputed': recomputed_count, 'kept': kept_count}
    
    def prepare_dataset(self, stocks_data: Dict[str, pd.DataFrame],
                       forward_days: int = None, threshold: float = None,
                       n_jobs: int = 15, 
                       enable_feature_engineering: bool = False,
                       cache_engineered_features: bool = True,
                       filter_incomplete_cache: bool = True,
                       train_start_date: str = None,
                       train_end_date: str = None) -> tuple:
        """
        准备训练数据集
        
        参数:
            stocks_data: 股票数据字典（应包含全量日期以生成缓存）
            forward_days: 预测未来N天
            threshold: 分类阈值
            n_jobs: 并行任务数
            enable_feature_engineering: 是否启用特征工程
            cache_engineered_features: 是否缓存特征工程结果
            filter_incomplete_cache: 是否过滤不完整缓存
            train_start_date: 训练样本开始日期
            train_end_date: 训练样本结束日期
        
        返回:
            (X, y, returns, factor_names, dates)
        """
        print("\n正在计算量化因子（技术指标 + K线形态 + 基本面）...")
        
        # 使用配置中的默认值
        if forward_days is None:
            forward_days = TrainingConfig.FUTURE_DAYS
        if threshold is None:
            threshold = TrainingConfig.RETURN_THRESHOLD
        
        # 过滤特征不完整的股票
        if filter_incomplete_cache:
            stocks_data, filter_stats = self._validate_and_filter_stocks(stocks_data)
        
        # 检查缓存情况
        cache_info = self.get_cache_info()
        print(f"  缓存状态: {cache_info['cached_stocks']}/{len(stocks_data)} 只股票已缓存")
        
        if cache_engineered_features:
            print(f"  缓存策略: 保存特征工程后的完整因子（方案B）")
            
            # 特征发现：识别完整的特征集
            print("  正在识别完整特征集...")
            all_possible_features = set()
            # 随机选取几只股票进行特征发现，确保覆盖不同行业和板块
            discovery_codes = list(stocks_data.keys())[:50]  # 增加搜索范围到50只
            for code in discovery_codes:
                try:
                    # 预先探测所有可能的列
                    f = self.factor_calculator.calculate_all_factors(
                        code, stocks_data[code], apply_feature_engineering=True
                    )
                    all_possible_features.update(f.columns)
                except:
                    continue
            
            target_features = sorted(list(all_possible_features))
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
            return self._process_single_stock(code, data, forward_days, threshold, 
                                             cache_engineered_features, target_features, verbose,
                                             train_start_date, train_end_date)
        
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_with_logging)(i, code, data)
            for i, (code, data) in enumerate(stock_list)
        )
        
        # 过滤有效结果
        all_factors = []
        all_labels = []
        all_returns = []
        all_dates = []
        for X, y, r, d in results:
            if X is not None and y is not None:
                all_factors.append(X)
                all_labels.append(y)
                all_returns.append(r)
                all_dates.append(d)
        
        # 合并所有数据
        X_combined = pd.concat(all_factors, axis=0, ignore_index=True)
        y_combined = pd.concat(all_labels, axis=0, ignore_index=True)
        returns_combined = np.concatenate(all_returns)
        dates_combined = np.concatenate(all_dates)
        
        # 统计因子数量（从实际计算结果中获取）
        # 如果还没有计算过因子，先计算一个样本来获取因子名称
        if len(self.tech_calculator.get_factor_names()) == 0 and len(stocks_data) > 0:
            sample_code = list(stocks_data.keys())[0]
            sample_data = list(stocks_data.values())[0]
            _ = self.factor_calculator.calculate_all_factors(sample_code, sample_data)
        
        tech_factor_count = len(self.tech_calculator.get_factor_names())
        candlestick_factor_count = len(self.candlestick_calculator.get_pattern_names())
        
        # 估计基本面因子数量（包括相对强度因子）
        fundamental_factor_count = 0
        try:
            sample_factors = self.fundamental_calculator.calculate_all_fundamental_factors(list(stocks_data.keys())[0])
            fundamental_factor_count = len(sample_factors)
        except:
            fundamental_factor_count = 50  # 默认估计值
        
        # 时间序列和风险特征数量
        ts_risk_factor_count = 0
        try:
            sample_data = list(stocks_data.values())[0]
            ts_price = TimeSeriesFactors.calculate_price_series_features(sample_data)
            ts_volume = TimeSeriesFactors.calculate_volume_series_features(sample_data)
            ts_momentum = TimeSeriesFactors.calculate_momentum_features(sample_data)
            risk = RiskFactors.calculate_risk_features(sample_data)
            ts_risk_factor_count = len(ts_price) + len(ts_volume) + len(ts_momentum) + len(risk)
        except:
            ts_risk_factor_count = 30  # 默认估计值
        
        print(f"\n数据集准备完成:")
        print(f"  总样本数: {len(X_combined)}")
        print(f"  基础特征数量: {X_combined.shape[1]}")
        print(f"    - 技术指标因子: {tech_factor_count}")
        print(f"    - K线形态因子: {candlestick_factor_count}")
        print(f"    - 基本面因子（含相对强度）: {fundamental_factor_count}")
        print(f"    - 时间序列和风险特征: {ts_risk_factor_count}")
        print(f"  正样本比例: {y_combined.mean():.2%}")
        
        # 应用特征工程（如果缓存时没有应用）
        if enable_feature_engineering and not cache_engineered_features:
            print("\n" + "="*80)
            print("应用特征工程...")
            print("="*80)
            
            self.feature_engineer.reset()
            X_combined = self.feature_engineer.apply_all_transformations(
                X_combined,
                config={
                    'ratio': True,      # 比率特征
                    'product': True,    # 乘积特征
                    'difference': True, # 差值特征
                    'log': True,        # 对数特征
                    'sqrt': True,       # 平方根特征
                    'rank': True,       # 排名特征
                    'interaction': True, # 交互特征
                    'categorical': True, # 分类特征编码
                }
            )
            
            engineered_features = len(self.feature_engineer.get_generated_features())
            print(f"\n特征工程后总特征数: {X_combined.shape[1]}")
            print(f"  新增特征: {engineered_features}")
        elif cache_engineered_features:
            print(f"\n特征工程已在缓存时应用，跳过")
            print(f"  当前特征数: {X_combined.shape[1]}")
        
        # 处理分类特征（如果存在）
        categorical_cols = [col for col in X_combined.columns if col in ['sector', 'industry']]
        if categorical_cols:
            print(f"\n处理分类特征: {categorical_cols}")
            # 移除原始分类列，保留编码后的特征
            X_combined = X_combined.drop(columns=categorical_cols, errors='ignore')
        
        # 数据清理和验证
        print("\n数据清理和验证...")
        
        # 1. 转换所有列为数值类型
        for col in X_combined.columns:
            try:
                X_combined[col] = pd.to_numeric(X_combined[col], errors='coerce')
            except:
                X_combined[col] = 0
        
        # 2. 替换无穷大值为NaN（便于后续处理）
        X_combined = X_combined.replace([np.inf, -np.inf], np.nan)
        
        # 3. 统计NaN和inf值
        nan_counts = X_combined.isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        if len(cols_with_nan) > 0:
            print(f"  发现 {len(cols_with_nan)} 列包含NaN值")
            for col in cols_with_nan.index:
                print(f"    - {col}: {cols_with_nan[col]} 个NaN值")
        
        # 4. 填充NaN值（使用中位数或0）
        for col in X_combined.columns:
            if X_combined[col].isna().any():
                # 尝试使用中位数，如果中位数为NaN或0，则使用0
                median_val = X_combined[col].median()
                if pd.isna(median_val):
                    X_combined[col] = X_combined[col].fillna(0)
                else:
                    # 使用中位数填充，但如果中位数为0，也用0
                    X_combined[col] = X_combined[col].fillna(median_val if median_val != 0 else 0)
        
        # 5. 最终检查：确保没有NaN或inf
        X_combined = X_combined.fillna(0)
        X_combined = X_combined.replace([np.inf, -np.inf], 0)
        
        # 6. 验证数据有效性
        invalid_rows = X_combined.isna().any(axis=1) | np.isinf(X_combined.values).any(axis=1)
        if invalid_rows.any():
            print(f"  警告: 移除 {invalid_rows.sum()} 行无效数据")
            X_combined = X_combined[~invalid_rows]
            y_combined = y_combined[~invalid_rows]
        
        # 7. 确保所有值都是有限的（转换为numpy数组后处理）
        X_array = X_combined.values.astype(np.float64)
        print("深度清理特征矩阵中的非有限值...")
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 8. 最后一次验证：检查是否还有NaN或inf
        if np.isnan(X_array).any():
            print(f"  警告: 发现 {np.isnan(X_array).sum()} 个NaN值，进行替换")
            X_array = np.nan_to_num(X_array, nan=0.0)
        
        if np.isinf(X_array).any():
            print(f"  警告: 发现 {np.isinf(X_array).sum()} 个无穷大值，进行替换")
            X_array[np.isinf(X_array)] = 0.0
        
        print(f"  数据清理完成: {X_array.shape[0]} 行, {X_array.shape[1]} 列")
        print(f"  数据类型检查: 所有列均为数值类型")
        print(f"  有效数据检查: 无NaN或无穷大值")
        
        return X_array, y_combined.values, returns_combined, X_combined.columns.tolist(), dates_combined
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    returns: np.ndarray,
                    factor_names: List[str],
                    dates: np.ndarray,
                    model_types: List[str] = ['xgboost', 'lightgbm']) -> Dict:
        """
        训练多个模型
        
        参数:
            X: 特征矩阵
            y: 标签向量
            returns: 原始收益率（用于计算权重或排序评价）
            factor_names: 特征名称列表
            dates: 样本日期（用于排序组划分）
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
        if self.task == 'classification':
            y = y.astype(np.int32)
        elif self.task == 'ranking':
            # 排序任务：将连续收益率离散化为整数等级 (如 0-9)
            # 某些库（如 LightGBM）要求 Ranking 标签必须为整数
            print("  将连续收益率转换为 10 个等级的整数标签用于排序训练")
            y_series = pd.Series(y)
            y = pd.qcut(y_series.rank(method='first'), 10, labels=False).values.astype(np.int32)
        else:
            y = y.astype(np.float64)
        
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
        
        # 准备样本权重（方案2）
        sample_weight = None
        if self.use_sample_weight:
            print(f"\n应用样本权重（方案2）: weight = abs(returns)")
            sample_weight = np.abs(returns)
        
        # 准备分组信息（方案4: 排序任务）
        groups = None
        if self.task == 'ranking':
            print(f"\n准备排序分组信息（方案4）: 按日期分组")
            # 排序任务需要按日期排序并计算每组大小
            sort_idx = np.argsort(dates)
            X = X[sort_idx]
            y = y[sort_idx]
            dates = dates[sort_idx]
            
            # 计算每组（每个日期）的样本数
            unique_dates, counts = np.unique(dates, return_counts=True)
            groups = counts
            print(f"  总组数 (交易日): {len(groups)}, 平均每组股票数: {groups.mean():.1f}")
        
        results = {}
        
        for model_type in model_types:
            print(f"\n{'='*80}")
            print(f"训练 {model_type.upper()} 模型")
            print(f"{'='*80}")
            
            try:
                model = MLFactorModel(model_type=model_type, task=self.task)
                
                # 训练模型
                train_result = model.train(X, y, validation_split=0.2, 
                                          use_time_series_split=True,
                                          feature_names=factor_names,
                                          sample_weight=sample_weight,
                                          groups=groups)
                
                # 保存模型
                self.models[model_type] = model
                results[model_type] = train_result
                
            except Exception as e:
                print(f"训练 {model_type} 失败: {e}")
                continue
        
        return results
    
    def compare_models(self, results: Dict):
        """对比模型性能"""
        print(f"\n{'='*80}")
        print("模型性能对比")
        print(f"{'='*80}")
        
        comparison = []
        for model_type, result in results.items():
            val_metrics = result['val_metrics']
            
            if self.task == 'classification':
                comparison.append({
                    '模型': model_type,
                    '准确率': f"{val_metrics['accuracy']:.4f}",
                    '精确率': f"{val_metrics['precision']:.4f}",
                    '召回率': f"{val_metrics['recall']:.4f}",
                    'F1分数': f"{val_metrics['f1']:.4f}",
                    'AUC': f"{val_metrics['auc']:.4f}"
                })
            elif self.task == 'ranking':
                comparison.append({
                    '模型': model_type,
                    '相关性': f"{val_metrics['correlation']:.4f}"
                })
            else:
                comparison.append({
                    '模型': model_type,
                    'MSE': f"{val_metrics['mse']:.6f}",
                    'MAE': f"{val_metrics['mae']:.6f}",
                    'RMSE': f"{val_metrics['rmse']:.6f}",
                    'R²': f"{val_metrics['r2']:.4f}"
                })
        
        df = pd.DataFrame(comparison)
        print(df.to_string(index=False))
        
        # 找出最佳模型
        if self.task == 'classification':
            best_model = max(results.items(), key=lambda x: x[1]['val_metrics']['auc'])
            print(f"\n最佳模型: {best_model[0].upper()} (AUC: {best_model[1]['val_metrics']['auc']:.4f})")
        elif self.task == 'ranking':
            best_model = max(results.items(), key=lambda x: x[1]['val_metrics']['correlation'])
            print(f"\n最佳模型: {best_model[0].upper()} (相关性: {best_model[1]['val_metrics']['correlation']:.4f})")
        else:
            best_model = max(results.items(), key=lambda x: x[1]['val_metrics']['r2'])
            print(f"\n最佳模型: {best_model[0].upper()} (R²: {best_model[1]['val_metrics']['r2']:.4f})")
        
        return best_model[0]
    
    def save_models(self, save_dir: str = 'models'):
        """保存所有训练好的模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        for model_type, model in self.models.items():
            filepath = os.path.join(save_dir, f'{model_type}_factor_model.pkl')
            model.save_model(filepath)
            print(f"  已保存: {filepath}")
    
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

    parser.add_argument('--task', type=str, choices=['classification', 'regression', 'ranking'],
                       default='classification', help='训练任务类型 (默认: classification)')
    parser.add_argument('--use-weight', action='store_true',
                       help='是否启用收益率绝对值加权 (方案2)')
    parser.add_argument('--cache-engineered', action='store_true',
                       help='在缓存时就应用特征工程（方案B, 默认）')
    parser.add_argument('--no-cache-engineered', dest='cache_engineered', action='store_false',
                       help='仅缓存基础因子，训练时应用特征工程（方案A）')
    parser.set_defaults(cache_engineered=True, use_weight=TrainingConfig.USE_WEIGHT, task=TrainingConfig.TASK_TYPE)
    
    args = parser.parse_args()
    
    print("="*80)
    print("机器学习因子模型训练（整合技术指标 + K线形态因子）")
    print(f"模式: {args.task.upper()} | 加权: {'开启' if args.use_weight else '关闭'} | 缓存工程: {'开启' if args.cache_engineered else '关闭'}")
    print("="*80)
    
    # 1. 初始化训练器
    trainer = MLModelTrainer(task=args.task, use_sample_weight=args.use_weight)
    
    # 显示缓存信息
    cache_info = trainer.get_cache_info()
    print(f"\n因子缓存信息:")
    print(f"  缓存目录: {trainer.factors_cache_dir}")
    print(f"  已缓存股票: {cache_info['cached_stocks']}")
    print(f"  缓存大小: {cache_info['cache_size_mb']:.2f} MB")
    
    # 2. 获取股票列表（示例：从数据库获取所有股票）
    conn = sqlite3.connect(DATABASE_PATH)
    stock_codes_df = pd.read_sql_query(
        f"SELECT DISTINCT code FROM daily_data LIMIT {TrainingConfig.STOCK_NUM}", 
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
    X, y, returns, factor_names, dates = trainer.prepare_dataset(
        stocks_data,
        cache_engineered_features=args.cache_engineered,
        train_start_date=train_start_date,
        train_end_date=train_end_date
    )
    
    # 6. 训练模型
    model_types = ['xgboost', 'lightgbm']
    results = trainer.train_models(X, y, returns, factor_names, dates, model_types)
    
    # 7. 对比模型
    best_model_type = trainer.compare_models(results)
    
    # 8. 保存模型
    print("\n保存模型...")
    trainer.save_models(save_dir=TrainingConfig.SAVE_DIR)
    
    # 9. 保存因子汇总
    trainer.save_factor_summary(factor_names, save_dir=TrainingConfig.SAVE_DIR)
    
    
    # 显示最终缓存信息
    cache_info = trainer.get_cache_info()
    
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    print(f"模型文件: {TrainingConfig.SAVE_DIR}/")
    print(f"因子缓存: {trainer.factors_cache_dir}/ ({cache_info['cached_stocks']} 只股票, {cache_info['cache_size_mb']:.2f} MB)")
    print(f"因子汇总: {TrainingConfig.SAVE_DIR}/factor_summary.json")
    
    if args.cache_engineered:
        print(f"\n✓ 缓存包含完整特征（方案B）")
        print(f"  回测时可以直接使用缓存，无需特征工程")
        print(f"  建议: 回测时设置 enable_feature_engineering=False")
    else:
        print(f"\n✓ 缓存仅包含基础因子（方案A）")
        print(f"  回测时需要启用特征工程")
        print(f"  建议: 回测时设置 enable_feature_engineering=True")
    
    print(f"\n提示: 使用 trainer.clear_factors_cache() 可清理缓存")


if __name__ == '__main__':
    main()
