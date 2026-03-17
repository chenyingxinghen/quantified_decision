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
from core.data.market_sentiment_calculator import MarketSentimentCalculator
from config import DATABASE_PATH, TrainingConfig, FactorConfig, MARKET_LIMITS, MARKET_PREFIXES
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
        # 挂载元数据库以支持 is_st 查询
        db_dir = os.path.dirname(self.db_path)
        meta_db = os.path.join(db_dir, 'stock_meta.db')
        if os.path.exists(meta_db):
            conn.execute(f"ATTACH DATABASE '{meta_db}' AS meta")
        
        conn.row_factory = sqlite3.Row
        
        # 分批加载，避免 IN 子句过长
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i+batch_size]
            placeholders = ','.join(['?' for _ in batch_codes])
            
            query = f'''
                SELECT d.code, d.date, d.open, d.high, d.low, d.close, d.volume, d.amount, d.turnover_rate,
                       IFNULL(s.is_st, 0) as is_st
                FROM daily_data d
                LEFT JOIN meta.stock_info_extended s ON d.code = s.code
                WHERE d.code IN ({placeholders}) AND d.date >= ? AND d.date <= ?
                ORDER BY d.code, d.date ASC
            '''
            
            params = list(batch_codes) + [str(start_date), str(end_date)]
            # print(f"Executing query with {len(params)} params for {len(batch_codes)} stocks")
            df = pd.read_sql_query(query, conn, params=tuple(params))
            
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
        
        支持增量更新：如果缓存中已有历史因子，只计算新增日期的因子并追加，
        而不是重新计算所有历史数据。
        
        参数:
            code: 股票代码
            data: 股票数据（应包含全量日期）
            apply_feature_engineering: 是否应用特征工程
            target_features: 目标特征列表
            verbose: 是否输出详细日志
            include_fundamentals: 是否包含基本面因子
        
        返回:
            合并后的因子DataFrame
        """
        cache_file = os.path.join(self.factors_cache_dir, f'{code}_factors.parquet')
        
        # 确保 data 中的 date 列为字符串（方便比较）
        if 'date' in data.columns and not pd.api.types.is_string_dtype(data['date']):
            data = data.copy()
            data['date'] = data['date'].astype(str)
        
        # ── 1. 尝试从缓存加载 ──────────────────────────────────────────────
        cached_factors = None
        if os.path.exists(cache_file):
            try:
                cached_factors = pd.read_parquet(cache_file)
                if 'date' in cached_factors.columns and not pd.api.types.is_string_dtype(cached_factors['date']):
                    cached_factors['date'] = cached_factors['date'].astype(str)
            except Exception:
                cached_factors = None
        
        # ── 2. 判断是否需要更新 ────────────────────────────────────────────
        need_full_recompute = False
        new_data_rows = None

        if cached_factors is not None and 'date' in cached_factors.columns and 'date' in data.columns:
            # 特征不匹配时必须全量重算
            if target_features:
                missing_feats = [f for f in target_features if f not in cached_factors.columns]
                if missing_feats:
                    if verbose:
                        print(f"  {code}: 缓存缺少 {len(missing_feats)} 个特征，触发全量重算")
                    need_full_recompute = True
            
            if not need_full_recompute:
                cache_last_date = cached_factors['date'].max()
                data_last_date  = data['date'].max()
                cache_first_date = cached_factors['date'].min()
                data_first_date = data['date'].min()
                
                if cache_first_date > data_first_date:
                    if verbose:
                        print(f"  {code}: 缓存起始日期 {cache_first_date} 晚于数据起始日期 {data_first_date}，触发全量重算")
                    need_full_recompute = True
                
                elif cache_last_date >= data_last_date:
                    # 缓存已是最新，直接命中
                    if target_features:
                        available = [f for f in target_features if f in cached_factors.columns]
                        if 'date' in cached_factors.columns and 'date' not in available:
                            available.append('date')
                        return cached_factors[available] if available else cached_factors
                    return cached_factors
                else:
                    # 有新数据：记录需要增量计算的新行
                    new_data_rows = data[data['date'] > cache_last_date].copy()
                    if verbose:
                        print(f"  {code}: 增量更新 {len(new_data_rows)} 行 "
                              f"({cache_last_date} -> {data_last_date})")
        elif cached_factors is not None and len(cached_factors) == len(data):
            # 无 date 列但行数一致，视为已是最新
            if target_features:
                available = [f for f in target_features if f in cached_factors.columns]
                if 'date' in cached_factors.columns and 'date' not in available:
                    available.append('date')
                return cached_factors[available] if available else cached_factors
            return cached_factors
        else:
            need_full_recompute = True

        # ── 3. 计算因子 ─────────────────────────────────────────────────────
        # 无论增量还是全量，都必须传入完整 data（技术指标需要历史 lookback 窗口）
        all_factors = self.factor_calculator.calculate_all_factors(
            code=code,
            data=data,
            apply_feature_engineering=apply_feature_engineering,
            target_features=target_features,
            verbose=verbose,
            include_fundamentals=include_fundamentals
        )

        if all_factors is None or all_factors.empty:
            return None

        # ── 4. 附加日期列 ──────────────────────────────────────────────────
        if 'date' in data.columns:
            all_factors = all_factors.copy()
            all_factors['date'] = data['date'].values

        # ── 5. 拼接缓存（增量模式）────────────────────────────────────────
        if not need_full_recompute and cached_factors is not None and new_data_rows is not None:
            new_date_set = set(new_data_rows['date'].astype(str).tolist())
            if 'date' in all_factors.columns:
                new_factor_rows = all_factors[all_factors['date'].astype(str).isin(new_date_set)].copy()
            else:
                new_factor_rows = all_factors.tail(len(new_data_rows)).copy()

            if new_factor_rows.empty:
                # 增量行全为 NaN，无法追加，仍返回旧缓存
                if target_features:
                    available = [f for f in target_features if f in cached_factors.columns]
                    if 'date' in cached_factors.columns and 'date' not in available:
                        available.append('date')
                    return cached_factors[available] if available else cached_factors
                return cached_factors

            # 列对齐：新行缺少的列补 NaN，多余列丢弃
            for col in cached_factors.columns:
                if col not in new_factor_rows.columns:
                    new_factor_rows[col] = np.nan
            new_factor_rows = new_factor_rows[cached_factors.columns]

            all_factors = pd.concat([cached_factors, new_factor_rows], ignore_index=True)

        # ── 6. 保存到缓存 ──────────────────────────────────────────────────
        try:
            all_factors.to_parquet(cache_file, index=False)
            if verbose:
                mode = '增量' if (not need_full_recompute and cached_factors is not None) else '全量'
                print(f"  {code} 因子{mode}缓存 ({len(all_factors)} 行, {len(all_factors.columns)} 列)")
        except Exception as e:
            if verbose:
                print(f"  保存因子缓存失败 ({code}): {e}")

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
            
            if factors is not None:
                if 'date' in factors.columns and 'date' in data.columns:
                    factors = pd.merge(data[['date']], factors, on='date', how='left')
                elif len(factors) != len(data):
                    factors = factors.iloc[-len(data):].reset_index(drop=True)

            if factors is not None and len(factors) > forward_days:

                # 获取价格序列
                close = data['close']
                high = data['high']
                low = data['low']
                
                
                # A. 收益率计算 (未来 n 日收盘涨幅)
                f_close = close.shift(-forward_days)
                f_returns = (f_close / close - 1)
                
                # 获取未来 n 日内的最大涨幅 (Max Run-up)，基于 T+1 到 T+n
                f_high_max = high.rolling(window=forward_days).max().shift(-forward_days)
                f_max_returns = (f_high_max / close - 1)
                
                # 获取未来 n 日内的最大跌幅 (Max Drawdown/Pain)，基于 T+1 到 T+n
                f_low_min = low.rolling(window=forward_days).min().shift(-forward_days)
                f_min_returns = (f_low_min / close - 1)

                # 1. 路径质量分 (Path-aware Score)
                # 显著惩罚回撤大、先跌后涨的标的，引导模型选择“走势稳健”的头部标的
                # 修复：使用符号位保留的幂运算，避免负收益率产生 NaN
                y = np.sign(f_returns) * (np.abs(f_returns) ** 1.5) + 0.5 * f_max_returns + 1.0 * f_min_returns
                
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
                    y_val = y[valid_idx] if isinstance(y, pd.Series) else y[valid_idx]
                    final_y = pd.Series(y_val) if not isinstance(y_val, pd.Series) else y_val
                    
                    final_returns = target_returns[valid_idx] if isinstance(target_returns, pd.Series) else target_returns[valid_idx]
                    
                    # 涨跌停判定增强：兼容主板(10%)、创业板/科创板(20%)、北交所(30%)
                    is_st = data['is_st'].iloc[0] == 1 if 'is_st' in data.columns else False
                    if is_st:
                        limit_threshold = MARKET_LIMITS['st']
                    elif code.startswith(MARKET_PREFIXES['sz_gem']) or code.startswith(MARKET_PREFIXES['star']):
                        limit_threshold = MARKET_LIMITS['gem_star']
                    elif code.startswith(MARKET_PREFIXES['bj']):
                        limit_threshold = MARKET_LIMITS['bj']
                    else:
                        limit_threshold = MARKET_LIMITS['main']
                        
                    is_limit_up = (data['close'] == data['high']) & (data['close'].pct_change() > limit_threshold)
                    is_suspended = data['volume'] == 0
                    unbuyable_mask = (is_limit_up | is_suspended)[valid_idx].values
                    
                    # 5. 显式剔除元数据列 (确保 is_st, date 不进入模型)
                    drop_cols = ['date', 'is_st', 'code', 'ORG_TYPE']
                    if not getattr(TrainingConfig, 'USE_AMOUNT_TURNOVER', False):
                        drop_cols.extend(['amount', 'turnover_rate'])
                    X_df = X_df.drop(columns=[c for c in drop_cols if c in X_df.columns], errors='ignore')
                    
                    if len(X_df) > 0:
                        limit_groups = np.full(len(X_df), limit_threshold, dtype=np.float32)
                        return X_df, final_y, final_returns, dates, unbuyable_mask, limit_groups
            
            return None, None, None, None, None, None
        
        except Exception as e:
            import traceback
            print(f"  警告: 处理股票 {code} 失败: {e}")
            print(traceback.format_exc())
            return None, None, None, None, None, None

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
                       n_jobs: int = 12, 
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
            cache_engineered_features: 是否缓存特征工程结果
            filter_incomplete_cache: 是否过滤不完整缓存
            train_start_date: 训练样本开始日期
            train_end_date: 训练样本结束日期
            include_fundamentals: 是否包含基本面因子
        
        返回:
            (X, y, returns, factor_names, dates, unbuyable, limit_groups)
        """

        # print("\n正在计算量化因子（技术指标 + K线形态 + 基本面）...")
        
        # 0. 自动更新市场情绪数据 (全局性指标，只需计算一次)
        print("\n正在检查并更新全市场情绪指标...")
        sentiment_calc = MarketSentimentCalculator(self.db_path)
        sentiment_calc.check_and_update()
        
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
        all_limit_groups = []
        
        for result in results:
            if result[0] is not None:
                # 兼容旧版本或失败情况
                if len(result) == 6:
                    X, y, r, d, u, l = result
                elif len(result) == 5:
                    X, y, r, d, u = result
                    l = np.full(len(X), 0.1, dtype=np.float32) # 默认 10% 限制
                else:
                    raise ValueError("因子计算失败，返回值不正确")                
                if X is not None:
                    all_factors.append(X)
                    all_labels.append(y)
                    all_returns.append(r)
                    all_dates.append(d)
                    all_unbuyable.append(u)
                    all_limit_groups.append(l)
        
        # 优化方案：分发直填模式 (Scatter Fill)
        # 核心改进：先只合并日期，计算排序索引映射，最后直接分配有序数组并填入。
        # 避免了 X_arr = X_arr[sort_idx] 导致的内存瞬间翻倍（旧方案中会同时存在两个巨大的特征矩阵副本）。
        
        total_rows = sum(len(f) for f in all_factors)
        num_features = all_factors[0].shape[1]
        col_names = all_factors[0].columns.tolist()
        
        print(f"  合并数据: 总行数 {total_rows}, 特征数 {num_features}, 预计占用内存: {total_rows * num_features * 4 / 1024 / 1024:.2f} MB (float32)")
        
        # 1. 预计算有序索引 (极省内存，因为只操作日期列)
        print("  - 预计算全局时间排序索引...")
        temp_dates = np.empty(total_rows, dtype=object)
        cursor = 0
        for d in all_dates:
            n = len(d)
            temp_dates[cursor:cursor+n] = d
            cursor += n
        
        sort_idx = np.argsort(temp_dates)
        # 构建逆映射：原始数据的 cursor+i 行，应该放在有序数组的哪个位置
        inverse_sort_idx = np.empty(total_rows, dtype=np.int32)
        inverse_sort_idx[sort_idx] = np.arange(total_rows, dtype=np.int32)
        del temp_dates # 立即释放
        
        # 2. 预分配最终的有序内存空间
        X_arr = np.empty((total_rows, num_features), dtype=np.float32)
        y_arr = np.empty(total_rows, dtype=np.float32)
        returns_arr = np.empty(total_rows, dtype=np.float32)
        dates_arr = np.empty(total_rows, dtype=object)
        unbuyable_arr = np.empty(total_rows, dtype=bool)
        limit_groups_arr = np.empty(total_rows, dtype=np.float32)
        
        # 3. 循环填充并即时释放块内存
        print("  - 正在直接按时间顺序填充数据 (Scatter Fill)...")
        cursor = 0
        for i in range(len(all_factors)):
            n = len(all_factors[i])
            target_pos = inverse_sort_idx[cursor : cursor + n]
            
            # 使用 target_pos 离散填充，保证 X_arr 从创建起就是有序的
            X_arr[target_pos] = all_factors[i].values.astype(np.float32)
            y_arr[target_pos] = all_labels[i].values.astype(np.float32)
            returns_arr[target_pos] = all_returns[i].astype(np.float32)
            dates_arr[target_pos] = all_dates[i]
            unbuyable_arr[target_pos] = all_unbuyable[i]
            limit_groups_arr[target_pos] = all_limit_groups[i]
            
            # 手动释放已处理的列表元素，最大程度压低峰值内存
            all_factors[i] = None
            all_labels[i] = None
            all_returns[i] = None
            all_dates[i] = None
            all_unbuyable[i] = None
            all_limit_groups[i] = None
            cursor += n
            
        # 彻底清理中间变量
        del all_factors, all_labels, all_returns, all_dates, all_unbuyable, all_limit_groups, inverse_sort_idx, sort_idx
        import gc
        gc.collect()
        
        # 应用：按日横向归一化 (Cross-sectional Normalization)
        # 获取日期分组索引（排序后日期是连续的）
        _, date_group_start, date_group_counts = np.unique(
            dates_arr, return_index=True, return_counts=True
        )
        
        print(f"\n应用：进行每日横向处理 (共 {len(date_group_start)} 个交易日)...")
        
        # [已移除] 因子横向 Z-Score 归一化
        # 移除原因：用户要求移除归一化步骤，以支持单只股票的绝对值预测
        # print("  - 跳过因子横向 Z-Score 归一化 (使用因子原始值)...")
        
        # 最后的无效值填充 (in-place)
        np.nan_to_num(X_arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 构建轻量级包装，仅用于审计报告分析，不进行大内存拷贝
        all_cols = col_names
        
        # 2. 标签值映射 (Squashing)
        # 即使移除了每日排名归一化，我们仍需确保标签在 [0, 1] 区间内，以满足 reg:logistic 损失函数的要求。
        # 采用固定的 Sigmoid 映射：P = 1 / (1 + exp(-20 * y))。
        # 这样处理保持了单兵打分的独立性（不耦合其他股票），且让 0% 涨幅对应 0.5，5% 涨幅对应约 0.73，符合“信心分”直觉。
        print("  - 正在通过固定 Sigmoid 将路径质量分映射至 [0, 1] 置信度空间...")
        y_arr = 1.0 / (1.0 + np.exp(-20.0 * y_arr))

        # 3. 惩罚不可买入样本 (涨停/停牌)
        # 将无法买入的标的标签强制设为极低值 (0.05)，迫使模型学习避开这些标的。
        if unbuyable_arr is not None:
            penalty_count = np.sum(unbuyable_arr)
            if penalty_count > 0:
                print(f"  - 施加不可买入惩罚: 将 {penalty_count} 个涨停/停牌标的的标签强制设为 0.05")
                y_arr[unbuyable_arr] = 0.05

        
        # 统计因子分类详情 (Factor Audit Report)
        remaining_all = set(all_cols)
        
        # 1. 状态因子
        status_cols = [c for c in all_cols if c in ['is_limit_up', 'is_suspended', 'market_type']]
        remaining_all -= set(status_cols)
        
        # 2. 市场情绪因子 (全市场维度)
        # 匹配 market_sentiment 表中的字段名
        sentiment_keywords = ['up_ratio', 'down_ratio', 'mean_return', 'adv_vol', 'breadth_', 'total_volume', 'sentiment_', 'mkt_']
        sentiment_cols = [c for c in all_cols if c in remaining_all and any(k in c.lower() for k in sentiment_keywords)]
        remaining_all -= set(sentiment_cols)
        
        # 3. 特征工程 (衍生的复合因子)
        engineered_cols = [c for c in all_cols if c in remaining_all and c in self.feature_engineer.get_generated_features()]
        remaining_all -= set(engineered_cols)
        
        # 4. K线形态
        candle_names = [x.lower() for x in self.candlestick_calculator.get_pattern_names()]
        candle_cols = [c for c in all_cols if c in remaining_all and any(x == c.lower() or f"cdl_{x}" in c.lower() for x in candle_names)]
        remaining_all -= set(candle_cols)
        
        # 5. 技术指标 (基础量价指标)
        tech_names = [x.lower() for x in self.tech_calculator.get_factor_names()]
        tech_cols = [c for c in all_cols if c in remaining_all and any(x in c.lower() for x in tech_names)]
        remaining_all -= set(tech_cols)
        
        # 6. 基础基本面
        fund_keywords = ['roe', 'xsjll', 'zzcjll', 'rev_yoy', 'np_yoy', 'pe', 'pb', 'eps', 'bps', 'ocf', 'zcfzl', 'qycs', 'bank_', 'org_type', 'peg', 'sue', 'eav']
        fund_cols = [c for c in all_cols if c in remaining_all and any(k in c.lower() for k in fund_keywords)]
        remaining_all -= set(fund_cols)
        
        # 7. 高级时序/风险特征
        adv_keywords = ['volatility', 'return_', 'momentum_', 'sharpe', 'drawdown', 'skewness', 'kurtosis', 'acceleration', 'position']
        adv_cols = [c for c in all_cols if c in remaining_all and any(k in c.lower() for k in adv_keywords)]
        remaining_all -= set(adv_cols)
        
        # 8. 其它
        other_cols = list(remaining_all)

        print("\n" + "="*50)
        print("数据集审计报告 (Factor Audit Report)")
        print("="*50)
        print(f"1. 技术指标 (Technical):    {len(tech_cols):>4} 个")
        print(f"2. K线形态 (Candlestick):   {len(candle_cols):>4} 个")
        print(f"3. 基础基本面 (Fundamental): {len(fund_cols):>4} 个")
        print(f"4. 市场情绪 (Sentiment):   {len(sentiment_cols):>4} 个")
        print(f"5. 高级时序 (Advanced):     {len(adv_cols):>4} 个")
        print(f"6. 特征工程 (Engineered):   {len(engineered_cols):>4} 个")
        print(f"7. 其它状态 (Others):       {len(status_cols) + len(other_cols):>4} 个")
        print("="*50 + "\n")
        
        # 统一输出 float32 以节省模型训练阶段的内存，XGB/LGB 内部也会转成 32 位
        return X_arr, y_arr, returns_arr, all_cols, dates_arr, unbuyable_arr, limit_groups_arr
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    returns: np.ndarray,
                    factor_names: List[str],
                    dates: np.ndarray,
                    unbuyable_mask: np.ndarray = None,
                    limit_groups: np.ndarray = None,
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
        
        # 确保数据类型正确 (copy=False 避免不必要的内存复制)
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)  # 支持软标签，不能强制转 int32
        
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
            # 优化：使用相对涨跌幅 (returns / limit_threshold) 作为权重
            # 这样一来，10% 市场涨 8% 的样本与 20% 市场涨 16% 的样本具有同等重要性
            if limit_groups is not None:
                print(f"\n样本权重: weight = abs(returns / limit_groups)")
                sample_weight = np.abs(returns / np.clip(limit_groups, 0.04, 0.3))
            else:
                print(f"\n样本权重: weight = abs(returns)")
                sample_weight = np.abs(returns)
            
            # 由于在预处理中已经将 unbuyable_mask 对应的标签强设为 0.1，
            # 我们希望模型充分学习这次负向惩罚，因此这里**不再**降低它们的权重。
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
                raw_split_idx = int(len(dates) * TrainingConfig.TRAIN_TEST_SPLIT)
                split_date = dates[raw_split_idx]
                split_idx = np.searchsorted(dates, split_date, side='left')
                
                train_dates = dates[:split_idx]
                val_dates = dates[split_idx:]
                
                _, train_group = np.unique(train_dates, return_counts=True)
                _, val_group = np.unique(val_dates, return_counts=True)
                
                extra_params['dates'] = dates  # 传递日期用于按组评估
                extra_params['split_idx'] = split_idx  # 传递对齐后的 split 点
                extra_params['group'] = train_group
                extra_params['eval_group'] = val_group
                            
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
                row['辅助指标'] = f"IC_Std: {val_metrics.get('rank_ic_std', 0.0):.4f}"
            else:
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
    train_start_date = (datetime.now() - timedelta(365*TrainingConfig.YEARS)).strftime('%Y-%m-%d')
    train_end_date = ((datetime.now() - timedelta(365*TrainingConfig.YEARS)) + timedelta(365*TrainingConfig.YEARS_FOR_TRAINING)).strftime('%Y-%m-%d')

    # 3.5 设置数据加载/缓存的时间范围（加载全量以支持回测）
    all_data_start = "2016-01-01" 
    all_data_end = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n时间范围配置:")
    print(f"  缓存/加载时段: {all_data_start} 至 {all_data_end}")
    print(f"  训练/模型时段: {train_start_date} 至 {train_end_date}")
    
    # 4. 加载全量数据（为了计算完整周期的因子）
    stocks_data = trainer.load_training_data(stock_codes, all_data_start, all_data_end)
    

    
    # 5. 准备数据集
    X, y, returns, factor_names, dates, unbuyable, limit_groups = trainer.prepare_dataset(
        stocks_data,
        cache_engineered_features=args.cache_engineered,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS
    )
    
    # 6. 训练模型
    model_types = ['xgboost', 'lightgbm']
    results = trainer.train_models(X, y, returns, factor_names, dates, unbuyable, limit_groups, model_types)
    
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
    else:
        print(f"\n✓ 缓存仅包含基础因子（方案A）")
        print(f"  回测时需要启用特征工程")
    
    print(f"\n提示: 使用 trainer.clear_factors_cache() 可清理缓存")


if __name__ == '__main__':
    main()
