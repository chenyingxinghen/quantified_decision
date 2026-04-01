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
from scipy.stats import spearmanr, rankdata
from time import time
from tqdm import tqdm

from core.factors.quantitative_factors import QuantitativeFactors
from core.factors.candlestick_pattern_factors import CandlestickPatternFactors
from core.factors.fundamental_factors import FundamentalFactors
from core.factors.ml_factor_model import MLFactorModel
from core.factors.comprehensive_factor_calculator import ComprehensiveFactorCalculator
from core.factors.advanced_factors import TimeSeriesFactors, RiskFactors
from core.factors.factor_filler import FactorFiller, fill_factors_with_defaults
from core.data.market_sentiment_calculator import MarketSentimentCalculator
from config import DATABASE_PATH, TrainingConfig, FactorConfig, MARKET_LIMITS, MARKET_PREFIXES,strategy_config
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
                          batch_size: int = 300) -> Dict[str, pd.DataFrame]:
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
        pbar = tqdm(total=len(stock_codes), desc="加载进度")
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i+batch_size]
            placeholders = ','.join(['?' for _ in batch_codes])
            
            # 重要改进：增加对 adjust_factor 的关联查询，实现动态复权，消除数据库增量更新导致的跳变
            query = f'''
                SELECT k.code, k.date, k.open, k.high, k.low, k.close, k.preclose, k.volume, k.amount, k.turnover_rate,
                       k.is_st, k.peTTM, k.pbMRQ, a.fore_adjust_factor
                FROM daily_data k
                LEFT JOIN adjust_factor a ON k.code = a.code AND k.date = a.date
                WHERE k.code IN ({placeholders}) AND k.date >= ? AND k.date <= ?
                ORDER BY k.code, k.date ASC
            '''
            
            params = list(batch_codes) + [str(start_date), str(end_date)]
            df = pd.read_sql_query(query, conn, params=tuple(params))
            
            if df.empty:
                continue

            # 按股票分组并执行动态复权
            for code in df['code'].unique():
                stock_df = df[df['code'] == code].copy()
                stock_df = stock_df.sort_values('date').reset_index(drop=True)
                
                # 动态复权处理
                if 'fore_adjust_factor' in stock_df.columns:
                    # 获取该段数据最后的复权因子作为基准 (最新日期的 qfq)
                    valid_adj = stock_df['fore_adjust_factor'].dropna()
                    if not valid_adj.empty:
                        base_factor = float(valid_adj.iloc[-1])
                        # 仅在基准非零时处理
                        if base_factor != 0:
                            ratio = stock_df['fore_adjust_factor'].ffill().fillna(1.0) / base_factor
                            for col in ['open', 'high', 'low', 'close', 'preclose']:
                                if col in stock_df.columns:
                                    stock_df[col] = stock_df[col] * ratio
                
                if len(stock_df) < 100:
                    continue
                
                stocks_data[code] = stock_df
            
            # 进度提示
            pbar.update(len(batch_codes))
        
        pbar.close()
        conn.close()
        
        print(f"成功加载并复权 {len(stocks_data)} 只股票的数据")
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
                print(f"  {code}: 缓存文件损坏，触发全量重算")
                cached_factors = None
        
        # ── 2. 判断是否需要更新 ────────────────────────────────────────────
        need_full_recompute = False
        new_data_rows = None

        if cached_factors is not None and 'date' in cached_factors.columns and 'date' in data.columns:
            # 修复问题四：无论是否传入 target_features，都检查缓存列数是否与当前计算结果一致
            # 当 target_features=None 时，用缓存中的数值列数量与一次探测计算结果对比
            if target_features:
                missing_feats = [f for f in target_features if f not in cached_factors.columns]
                if missing_feats:
                    if verbose:
                        print(f"  {code}: 缓存缺少 {len(missing_feats)} 个特征，触发全量重算")
                    need_full_recompute = True
            else:
                # target_features=None 时，通过缓存列数做轻量版本检测
                # 若缓存数值列数为 0（空缓存），触发全量重算
                cached_numeric_cols = cached_factors.select_dtypes(include=[np.number]).shape[1]
                if cached_numeric_cols == 0:
                    if verbose:
                        print(f"  {code}: 缓存无数值列，触发全量重算")
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
        # 优化点：如果是增量更新且不需要全量重算，仅传递最近的历史数据窗口即可
        # 技术指标通常需要一定的历史回望（Cold Start），500 行足以满足绝大多数指标（如 250 日线）
        if not need_full_recompute and cached_factors is not None and new_data_rows is not None:
        # 增量模式：选取最后 N 行数据进行计算，N 取最大因子回望窗口 + 新增行数的较大值
            max_lookback = max(
                getattr(FactorConfig, 'BB_PERIOD', 100),
                getattr(FactorConfig, 'MA_RATIO_PERIOD', 120),
                getattr(FactorConfig, 'ROC_PERIOD', 60),
                250,  # 年线保底
            ) + 50  # 额外缓冲
            calc_window = max(max_lookback, len(new_data_rows) + max_lookback)
            calculation_data = data.tail(calc_window).copy()
            if verbose:
                print(f"  {code}: 采用增量计算模式 (窗口={len(calculation_data)} 行)")
        else:
            # 全量模式
            calculation_data = data
            if verbose and need_full_recompute:
                print(f"  {code}: 采用全量重算模式")

        all_factors = self.factor_calculator.calculate_all_factors(
            code=code,
            data=calculation_data,
            apply_feature_engineering=apply_feature_engineering,
            target_features=target_features,
            verbose=verbose,
            include_fundamentals=include_fundamentals
        )

        if all_factors is None or all_factors.empty:
            return None

        # ── 4. 附加日期列 ──────────────────────────────────────────────────
        # 确保日期对齐：使用用于计算的 data 部分的日期
        if 'date' in calculation_data.columns:
            all_factors = all_factors.copy()
            if len(all_factors) == len(calculation_data):
                all_factors['date'] = calculation_data['date'].values
            else:
                all_factors['date'] = calculation_data['date'].reindex(all_factors.index).values

        # ── 5. 拼接缓存（增量模式）────────────────────────────────────────
        if not need_full_recompute and cached_factors is not None and new_data_rows is not None:
            new_date_set = set(new_data_rows['date'].astype(str).tolist())
            if 'date' in all_factors.columns:
                new_factor_rows = all_factors[all_factors['date'].astype(str).isin(new_date_set)].copy()
            else:
                new_factor_rows = all_factors.tail(len(new_data_rows)).copy()

            if new_factor_rows.empty:
                # 增量日期过滤后无匹配行（通常是步骤4日期对齐失败导致）
                # 记录警告并回退到全量重算，避免新数据被静默丢弃
                print(f"  {code}: 警告 - 增量过滤后无新行（日期对齐可能失败），回退到全量重算")
                need_full_recompute = True
                # 重新计算已在步骤3完成，all_factors 已是全量结果，直接跳到步骤6保存
            else:
                # 列对齐：新行缺少的列补 NaN，多余列丢弃
                missing_cols = [col for col in cached_factors.columns if col not in new_factor_rows.columns]
                if missing_cols:
                    # 批量添加缺失列以避免 DataFrame 碎片化 (Fix PerformanceWarning)
                    nan_df = pd.DataFrame(np.nan, index=new_factor_rows.index, columns=missing_cols)
                    new_factor_rows = pd.concat([new_factor_rows, nan_df], axis=1)
                
                new_factor_rows = new_factor_rows[cached_factors.columns]

                # 修复问题8：增量合并时去重，避免日期重叠导致的重复行
                all_factors = pd.concat([cached_factors, new_factor_rows], ignore_index=True)
                if 'date' in all_factors.columns:
                    all_factors = all_factors.drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)

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

    def batch_update_factor_cache(self, stocks_data: Dict[str, pd.DataFrame], 
                                 include_fundamentals: bool = True,
                                 target_features: Optional[List[str]] = None,
                                 n_jobs: int = 15,
                                 verbose: bool = False):
        """
        并行批量更新因子的持久化缓存到最新行情日期。
        
        参数:
            stocks_data: {code: DataFrame} (应包含到最新日期的行情)
            include_fundamentals: 是否包含基本面
            n_jobs: 并打数
            verbose: 是否输出详细信息
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
        # 1. 快速去重与预扫描
        all_codes = list(stocks_data.keys())
        total_initial = len(all_codes)
        
        # 2. 增量跳过：快速检查哪些缓存已经是最新
        to_update = {}
        skipped = 0
        
        print(f"\n[因子缓存同步] 正在扫描磁盘缓存状态...")
        for code in all_codes:
            data = stocks_data[code]
            if data.empty:
                continue
                
            cache_file = os.path.join(self.factors_cache_dir, f'{code}_factors.parquet')
            if os.path.exists(cache_file):
                try:
                    # 仅读取最后一行日期进行对比
                    # 注意：为了最高效率，可以使用 pyarrow 直接读 metadata，这里使用简单逻辑
                    import pyarrow.parquet as pq
                    last_row = pq.read_table(cache_file, columns=['date']).to_pandas().tail(1)
                    
                    if not last_row.empty:
                        cache_last_date = str(last_row['date'].iloc[0])
                        data_last_date = str(data['date'].max())
                        
                        if cache_last_date >= data_last_date:
                            # 如果缓存已经覆盖了数据的最新日期，跳过
                            skipped += 1
                            continue
                except Exception:
                    # 任何异常（如列不存在）都触发重算
                    pass
            
            to_update[code] = data

        if skipped > 0:
            print(f"  已跳过 {skipped} 只已同步的股票缓存")
            
        if not to_update:
            print(f"✓ 缓存已是最新，无需更新。")
            return

        print(f"  正在并行更新 {len(to_update)} 只股票的缓存 (workers={n_jobs})...")
        
        start_time = time()
        success = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    self.calculate_and_save_factors, 
                    code, data, 
                    target_features=target_features,
                    include_fundamentals=include_fundamentals,
                    verbose=verbose
                ): code 
                for code, data in to_update.items()
            }
            
            with tqdm(total=len(futures), desc="更新因子缓存") as pbar:
                for future in as_completed(futures):
                    code = futures[future]
                    try:
                        future.result()
                        success += 1
                    except Exception as e:
                        tqdm.write(f"  ✗ {code} 缓存更新失败: {e}")
                        failed += 1
                    pbar.set_postfix({"成功": success, "失败": failed})
                    pbar.update(1)
                    
        elapsed = time() - start_time
        print(f"✓ 缓存同步完成: 成功 {success}, 失败 {failed} | 已跳过 {skipped} | 耗时 {elapsed:.1f}s")
    

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

    def _compute_path_quality_score(self, f_returns, f_low_min, f_high_idx, f_low_idx, atr_raw, next_open, rel_atr):
        """
        参数:
            f_returns: 
            f_low_min: 
            f_high_idx: 
            f_low_idx: 
            atr_raw: 
            next_open: 
            rel_atr: 
        """
        # 避免除以零
        eps = 1e-4
        
        # ---------------------------------------------------------
        # 1. 核心收益项 (转换为 ATR 倍数)
        # ---------------------------------------------------------
        core_term = f_returns / (rel_atr + eps)*getattr(TrainingConfig, 'LABEL_TARGET_SCALE', 1.5)
            
        # ---------------------------------------------------------
        # 2. 损失厌恶项 (非线性穿透惩罚)
        # ---------------------------------------------------------
        lambda_val = getattr(TrainingConfig, 'LABEL_LAMBDA', 1.0)
        downside_gap = (next_open - f_low_min) / (atr_raw + eps) # 下跌倍数
        
        # 【优化】非线性惩罚：小于 1 ATR 线性计算，大于 1 ATR 呈2次方级放大
        # 这样模型会极度厌恶“破位”的股票
        loss_aversion = np.where(downside_gap <= 1.0,
                                -lambda_val * np.maximum(0, downside_gap),
                                -lambda_val * (1.0 + (downside_gap - 1.0) ** 2))
        
        # ---------------------------------------------------------
        # 3. 路径保护惩罚 (V型反转过滤)
        # ---------------------------------------------------------
        path_punish_coef = getattr(TrainingConfig, 'LABEL_PATH_PUNISH', 0.5)
        
        # 判定条件：低点早于高点 (V型) 且 下跌穿透了 1 倍 ATR
        is_v_shape = (f_low_idx < f_high_idx) & (downside_gap > 1.0)
        
        # 【优化】对于先大跌再拉回的票，我们将其核心收益进行“打折”，而不是单纯叠加负分
        # 如果 core_term 是正的，按下跌深度削减其得分；如果是负的，保持原样（因为 loss_aversion 已经惩罚过了）
        path_penalty = np.where(is_v_shape & (core_term > 0),
                                -core_term * path_punish_coef * np.clip(downside_gap, 1.0, 2.0), 
                                0)
                                
        # 4. 资金效率奖励 (如果高点发生得很早，且核心收益为正，给予微弱加分)
        # 修复：仅对正收益生效，避免对"高点早但最终亏损"的票产生额外惩罚
        time_bonus = np.where((f_high_idx < 2) & (core_term > 0), TrainingConfig.LABEL_TIME_BONUS * core_term, 0)

        final_score = core_term + loss_aversion + path_penalty+time_bonus
        
        return final_score


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
                    raise ValueError(f"Length mismatch: factors({len(factors)}) != data({len(data)}) for {code}")

            if factors is not None and len(factors) > forward_days:

                # 获取价格序列
                close = data['close']
                high = data['high']
                low = data['low']
                
                # A. 收益率计算 (避免偷价：假设 T+1 日开盘买入)
                next_open = data['open'].shift(-1)
                f_close = close.shift(-forward_days)
                f_returns = (f_close / next_open - 1)
                
                # 获取未来 n 日内的最大涨幅 (Max Run-up)，基于 T+1 到 T+n，相对于买入价
                f_high_max = high.rolling(window=forward_days).max().shift(-forward_days)
                f_max_returns = (f_high_max / next_open - 1)
                
                # 获取未来 n 日内的最大跌幅 (Max Drawdown/Pain)，基于 T+1 到 T+n，相对于买入价
                f_low_min = low.rolling(window=forward_days).min().shift(-forward_days)
                f_min_returns = (f_low_min / next_open - 1)
                
                # 获取极值位置 (用于路径保护惩罚)
                f_high_idx = high.rolling(window=forward_days).apply(np.argmax, raw=True).shift(-forward_days)
                f_low_idx = low.rolling(window=forward_days).apply(np.argmin, raw=True).shift(-forward_days)

                # 计算当前波动率 (ATR) 作为分母，衡量收益的“性价比”
                # 使用相对 ATR (ATR / close)
                atr_raw = talib.ATR(high.values, low.values, close.values, timeperiod=strategy_config.ATR_PERIOD)
                atr_rel = atr_raw / close.values
                    

                y = self._compute_path_quality_score(f_returns.values, f_low_min.values, 
                                                   f_high_idx.values, f_low_idx.values, 
                                                   atr_raw, next_open.values, atr_rel)
                
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
                    # 修复：使用当前行的 is_st 状态而非首行，因为 ST 状态会随时间变化
                    is_st_series = data['is_st'][valid_idx] == 1 if 'is_st' in data.columns else pd.Series(False, index=data.index[valid_idx])
                    
                    # 为每一行计算对应的涨停阈值
                    limit_thresholds = np.full(len(data), MARKET_LIMITS['main'], dtype=np.float32)
                    
                    if 'is_st' in data.columns:
                        limit_thresholds[data['is_st'] == 1] = MARKET_LIMITS['st']
                    
                    if code.startswith(MARKET_PREFIXES['sz_gem']) or code.startswith(MARKET_PREFIXES['star']):
                        limit_thresholds[:] = MARKET_LIMITS['gem_star']
                    elif code.startswith(MARKET_PREFIXES['bj']):
                        limit_thresholds[:] = MARKET_LIMITS['bj']
                    
                    # 使用逐行的阈值判断涨停 (引入 epsilon 容差防止浮点数和四舍五入判定失效)
                    pct_change = data['close'].pct_change()
                    epsilon = 0.002
                    is_limit_up = (data['close'] == data['high']) & (pct_change >= pd.Series(limit_thresholds, index=data.index) - epsilon)
                    is_suspended = data['volume'] == 0
                    unbuyable_mask = (is_limit_up | is_suspended)[valid_idx].values
                    
                    # 提取有效行的涨停阈值用于后续分组
                    limit_groups = limit_thresholds[valid_idx]
                    
                    # 5. 显式剔除元数据列 (确保 is_st, date 不进入模型)
                    drop_cols = ['date', 'is_st', 'code', 'fore_adjust_factor', 'back_adjust_factor']
                    if not getattr(TrainingConfig, 'USE_AMOUNT_TURNOVER', False):
                        drop_cols.extend(['amount', 'turnover_rate'])
                    X_df = X_df.drop(columns=[c for c in drop_cols if c in X_df.columns], errors='ignore')
                    
                    if len(X_df) > 0:
                        return X_df, final_y, final_returns, dates, unbuyable_mask, limit_groups
            
            return None, None, None, None, None, None
        
        except Exception as e:
            import traceback
            print(f"  警告: 处理股票 {code} 失败: {e}")
            print(traceback.format_exc())
            return None, None, None, None, None, None

    def _validate_and_filter_stocks(self, stocks_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """
        验证缓存特征完整性，过滤掉缓存损坏的股票（触发重算），统计需要重算的数量。
        注意：特征不匹配的股票仍会保留（后续会触发重算），只有无法恢复的损坏缓存才被标记。
        
        参数:
            stocks_data: 股票数据字典
        
        返回:
            (处理后的股票数据, 验证统计信息)
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
            print("  警告: 无法获取模型特征，跳过验证")
            return stocks_data, {'filtered': 0, 'kept': len(stocks_data)}
        
        filtered_stocks = {}
        filtered_count = 0
        recomputed_count = 0
        
        for code, data in tqdm(stocks_data.items(), desc="验证缓存完整性"):
            cache_file = os.path.join(self.factors_cache_dir, f'{code}_factors.parquet')
            
            if os.path.exists(cache_file):
                try:
                    factors = pd.read_parquet(cache_file)
                    
                    numeric_factors = factors.select_dtypes(include=[np.number])
                    cache_features = set(numeric_factors.columns)
                    
                    # 特征不匹配：保留股票，标记为需要重算（calculate_and_save_factors 会处理）
                    if not model_features.issubset(cache_features):
                        recomputed_count += 1
                    
                    filtered_stocks[code] = data
                        
                except Exception as e:
                    # 缓存读取失败（文件损坏），删除损坏文件并保留股票触发重算
                    recomputed_count += 1
                    filtered_stocks[code] = data
                    try:
                        os.remove(cache_file)
                    except:
                        pass
            else:
                # 缓存不存在，保留该股票（会实时计算）
                filtered_stocks[code] = data
        
        kept_count = len(filtered_stocks)
        print(f"  验证完成: {kept_count} 只股票待处理")
        if recomputed_count > 0:
            print(f"  其中 {recomputed_count} 只股票的缓存无效或不匹配，将重新计算")
        
        return filtered_stocks, {'filtered': filtered_count, 'recomputed': recomputed_count, 'kept': kept_count}
    
    def discover_target_features(self, stocks_data: Dict[str, pd.DataFrame],
                                  include_fundamentals: bool = True,
                                  n_discovery: int = 30) -> List[str]:
        """
        对前 n_discovery 只股票做一次因子计算，取列名并集，返回完整特征列表。
        供 batch_update_factor_cache 和 prepare_dataset 共享，确保两者使用相同的
        target_features，避免 Step 0 写缓存后 Step 2 因列不匹配而触发全量重算。
        """
        print(f"  正在识别完整特征集（采样 {n_discovery} 只股票）...")
        all_possible_features: set = set()
        discovery_codes = list(stocks_data.keys())[:n_discovery]
        for code in discovery_codes:
            try:
                discovery_data = stocks_data[code]
                if len(discovery_data) < 200:
                    continue
                f = self.factor_calculator.calculate_all_factors(
                    code, discovery_data,
                    apply_feature_engineering=True,
                    include_fundamentals=include_fundamentals
                )
                if f is not None and not f.empty:
                    all_possible_features.update(f.columns)
            except Exception:
                continue
        target_features = sorted([f for f in all_possible_features if f != 'date'])
        print(f"  识别到总计 {len(target_features)} 个特征（含特征工程生成项）")
        return target_features

    def prepare_dataset(self, stocks_data: Dict[str, pd.DataFrame],
                       forward_days: int = None,
                       n_jobs: int = 15,
                       cache_engineered_features: bool = True,
                       filter_incomplete_cache: bool = False,
                       train_start_date: str = None,
                       train_end_date: str = None,
                       include_fundamentals: bool = True,
                       target_features: Optional[List[str]] = None) -> tuple:
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
            target_features: 外部传入的完整特征列表；若提供则跳过内部特征发现，
                             与 batch_update_factor_cache 共享同一列表以保证缓存命中
        
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
            
            if target_features is not None:
                # 外部已传入（由 Step 0 发现并共享），直接复用，跳过重复计算
                print(f"  复用外部传入的特征集: {len(target_features)} 个特征")
            else:
                # 特征发现：识别完整的特征集
                target_features = self.discover_target_features(
                    stocks_data, include_fundamentals=include_fundamentals
                )
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
        
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(process_with_logging)(i, code, data)
            for i, (code, data) in enumerate(stock_list)
        )
        
        # 释放输入数据以腾出空间
        del stock_list
        import gc; gc.collect()

        # 1. 第一遍扫描：统计实际有效行数和特征数
        print("  - 统计有效样本量...")
        actual_rows = 0
        num_features = 0
        col_names = None
        valid_results_indices = []
        
        for i, res in enumerate(results):
            if res is not None and res[0] is not None:
                actual_rows += len(res[0])
                if col_names is None:
                    col_names = res[0].columns.tolist()
                    num_features = len(col_names)
                valid_results_indices.append(i)
        
        if actual_rows == 0:
            raise ValueError("没有生成的有效样本")
            
        print(f"  - 实际有效行数: {actual_rows}, 特征数: {num_features}")
        
        # 2. 预分配最终的内存空间
        X_arr = np.empty((actual_rows, num_features), dtype=np.float32)
        y_raw_arr = np.empty(actual_rows, dtype=np.float32) # 原始路径质量分
        returns_arr = np.empty(actual_rows, dtype=np.float32)
        dates_arr = np.empty(actual_rows, dtype=object)
        unbuyable_arr = np.empty(actual_rows, dtype=bool)
        limit_groups_arr = np.empty(actual_rows, dtype=np.float32)
        
        # 3. 填充数据并立即从 results 中剔除已处理对象
        print("  - 正在合并数据 (Incremental Fill)...")
        cursor = 0
        for idx in valid_results_indices:
            res = results[idx]
            n = len(res[0])
            
            # 分解结果
            if len(res) == 6:
                X, y, r, d, u, l = res
            else:
                X, y, r, d, u = res
                l = np.full(n, 0.1, dtype=np.float32)

            X_arr[cursor:cursor+n] = X.values.astype(np.float32, copy=False)
            y_raw_arr[cursor:cursor+n] = y.values if isinstance(y, pd.Series) else y
            returns_arr[cursor:cursor+n] = r
            dates_arr[cursor:cursor+n] = d
            unbuyable_arr[cursor:cursor+n] = u
            limit_groups_arr[cursor:cursor+n] = l
            
            # 手动销毁 results 中的引用，释放 DataFrame 内存
            results[idx] = None
            cursor += n
            
        # 清理结果列表和索引
        del results, valid_results_indices
        gc.collect()

        # 4. 全局时间排序
        print("  - 全局时间排序...")
        sort_idx = np.argsort(dates_arr)
        
        # 排序所有数组
        dates_arr = dates_arr[sort_idx]
        y_raw_arr = y_raw_arr[sort_idx]
        returns_arr = returns_arr[sort_idx]
        unbuyable_arr = unbuyable_arr[sort_idx]
        limit_groups_arr = limit_groups_arr[sort_idx]
        
        # X_arr 排序
        X_sorted = X_arr[sort_idx] 
        del X_arr, sort_idx
        X_arr = X_sorted
        del X_sorted
        gc.collect()
        all_cols = col_names
        # 3. 不可买入样本处理 (涨停/停牌) - 提前到归一化之前，确保排序不含涨停股
        if unbuyable_arr is not None:
            penalty_count = np.sum(unbuyable_arr)
            if penalty_count > 0:
                handling = getattr(TrainingConfig, 'UNBUYABLE_HANDLING', 'remove')
                if handling == 'remove':
                    print(f"  - 剔除不可买入样本 (排序前): 正在剔除 {penalty_count} 个涨停/停牌样本")
                    keep_mask = ~unbuyable_arr
                    X_arr = X_arr[keep_mask]
                    y_raw_arr = y_raw_arr[keep_mask]
                    returns_arr = returns_arr[keep_mask]
                    dates_arr = dates_arr[keep_mask]
                    limit_groups_arr = limit_groups_arr[keep_mask]
                    # 重新获取日期分组信息
                    _, date_group_start, date_group_counts = np.unique(dates_arr, return_index=True, return_counts=True)
                    unbuyable_arr = unbuyable_arr[keep_mask]
                else:
                    print(f"  - 施加不可买入惩罚: 将 {penalty_count} 个涨停/停牌标的的标签强制设为 0.05")
                    y_raw_arr[unbuyable_arr] = 0.05
                    # 重新获取日期分组信息 (虽然行数没变)
                    _, date_group_start, date_group_counts = np.unique(dates_arr, return_index=True, return_counts=True)
        else:
            _, date_group_start, date_group_counts = np.unique(dates_arr, return_index=True, return_counts=True)

        # 2. 标签值映射 (Board-Neutral Normalization)
        # 注意：此处生成的 y_norm_arr 仅供 XGB 等回归模型默认使用
        y_norm_arr = y_raw_arr.copy()
        print("  - 正在进行每日板块中性化排名归一化标签...")
        for start, count in zip(date_group_start, date_group_counts):
            end = start + count
            if count > 1:
                day_limits = limit_groups_arr[start:end]
                day_y = y_norm_arr[start:end].copy()
                unique_limits = np.unique(day_limits)
                for limit_val in unique_limits:
                    board_mask = day_limits == limit_val
                    board_count = np.sum(board_mask)
                    if board_count > 0:
                        day_y[board_mask] = rankdata(day_y[board_mask], method='average') / (board_count + 1)
                y_norm_arr[start:end] = day_y
            else:
                y_norm_arr[start:end] = 0.5


        
        # 统计因子分类详情 (Factor Audit Report)
        # 使用各模块的精确列名进行匹配，而非关键词启发式匹配
        remaining_all = set(all_cols)
        
        # 1. 状态因子 (comprehensive_factor_calculator 中硬编码的3个)
        _status_known = {'is_limit_up', 'is_suspended', 'market_type'}
        status_cols = [c for c in all_cols if c in _status_known]
        remaining_all -= set(status_cols)
        
        # 2. 市场情绪因子 — 精确匹配 market_sentiment 表的列名 (不含 date)
        _sentiment_known = {
            'up_ratio', 'strong_up_ratio', 'down_ratio',
            'limit_up_ratio', 'limit_down_ratio', 'mean_return',
            'total_volume', 'adv_vol_ratio', 'breadth_ma20'
        }
        sentiment_cols = [c for c in all_cols if c in remaining_all and c in _sentiment_known]
        remaining_all -= set(sentiment_cols)
        
        # 3. 特征工程 (衍生的复合因子) — 从 FeatureEngineer 实例获取精确列表
        _engineered_set = set(self.feature_engineer.get_generated_features())
        engineered_cols = [c for c in all_cols if c in remaining_all and c in _engineered_set]
        remaining_all -= set(engineered_cols)
        
        # 4. K线形态 — 精确匹配 CandlestickPatternFactors.get_pattern_names()
        _candle_set = set(self.candlestick_calculator.get_pattern_names())
        candle_cols = [c for c in all_cols if c in remaining_all and c in _candle_set]
        remaining_all -= set(candle_cols)
        
        # 5. 技术指标 — 精确匹配 QuantitativeFactors.get_factor_names()
        _tech_set = set(self.tech_calculator.get_factor_names())
        tech_cols = [c for c in all_cols if c in remaining_all and c in _tech_set]
        remaining_all -= set(tech_cols)
        
        # 6. 基本面因子 — 精确匹配 FundamentalFactors.NUMERIC_COLS + 已知衍生列
        _fund_known = set(FundamentalFactors.NUMERIC_COLS) | {
            'dynamic_pe', 'dynamic_pb', 'inv_pe', 'inv_pb', 'market_cap',
            'roe_x_np_growth', 'roe_to_pb',
            'peg', 'sue', 'eav'
        }
        fund_cols = [c for c in all_cols if c in remaining_all and c in _fund_known]
        remaining_all -= set(fund_cols)
        
        # 7. 高级时序/风险特征 — 精确匹配 advanced_factors.py 中的列名
        _adv_known = {
            # TimeSeriesFactors.calculate_price_series_features
            'hl_range_mean', 'hl_range_std', 'oc_ratio_mean', 'oc_ratio_std',
            'price_volatility_20', 'price_volatility_60', 'price_skewness', 'price_kurtosis',
            'high_position', 'low_position',
            # TimeSeriesFactors.calculate_volume_series_features
            'volume_change_rate', 'volume_volatility', 'price_volume_corr',
            'amount_per_volume', 'amount_change_rate',
            # TimeSeriesFactors.calculate_momentum_features
            'return_5d', 'return_10d', 'return_20d', 'return_60d',
            'momentum_5d', 'momentum_10d', 'momentum_20d', 'acceleration',
            # RiskFactors.calculate_risk_features
            'downside_risk', 'drawdown', 'max_drawdown_20', 'sharpe_ratio',
            'return_skewness', 'return_kurtosis',
        }
        adv_cols = [c for c in all_cols if c in remaining_all and c in _adv_known]
        remaining_all -= set(adv_cols)
        
        # 8. 其它 (未被以上任何类别匹配到的列)
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
        if other_cols:
            print(f"   未分类列: {other_cols[:20]}{'...' if len(other_cols) > 20 else ''}")
        print("="*50 + "\n")
        
        # 统一输出 float32 以节省模型训练阶段的内存，XGB/LGB 内部也会转成 32 位
        return X_arr, y_norm_arr, returns_arr, all_cols, dates_arr, unbuyable_arr, limit_groups_arr, y_raw_arr
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    returns: np.ndarray,
                    factor_names: List[str],
                    dates: np.ndarray,
                    unbuyable_mask: np.ndarray = None,
                    limit_groups: np.ndarray = None,
                    model_types: List[str] = TrainingConfig.MODEL_TYPES,
                    path_scores: np.ndarray = None) -> Dict:
        """
        训练多个模型
        """
        # 数据验证和清理
        print("\n数据验证...")
        
        # 确保数据类型正确 (copy=False 避免不必要的内存复制)
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        
        # 最后一次NaN/inf检查和替换
        if np.isnan(X).any():
            np.nan_to_num(X, copy=False, nan=0.0)
        
        print(f"  数据验证完成: {X.shape[0]} 行, {X.shape[1]} 列")
        
        # 修复问题2: 在 train/val split 之后，分别进行横截面归一化
        print("\n修复问题2: 准备在 split 后分别进行横截面归一化...")
        
        # 先进行时间序列划分
        raw_split_idx = int(len(dates) * TrainingConfig.TRAIN_TEST_SPLIT)
        split_date = dates[raw_split_idx]
        split_idx = np.searchsorted(dates, split_date, side='left')
        
        print(f"  划分点: {split_date}, 索引: {split_idx}")
        print(f"  训练集: {split_idx} 样本, 验证集: {len(dates) - split_idx} 样本")
        
        # 1. 对训练样本进行原位横截面归一化
        print("\n  对训练样本进行横截面归一化...")
        self._apply_cross_sectional_normalization_inplace(
            X[:split_idx], dates[:split_idx], factor_names
        )
        
        # 2. 对验证样本进行原位横截面归一化
        print("  对验证样本进行横截面归一化...")
        self._apply_cross_sectional_normalization_inplace(
            X[split_idx:], dates[split_idx:], factor_names
        )
        
        # Default sample_weight (for XGBM and others)
        default_sample_weight = None
        if self.punish_unbuyable:
            if limit_groups is not None:
                default_sample_weight = np.abs(returns / np.clip(limit_groups, 0.04, 0.3))
            else:
                default_sample_weight = np.abs(returns)
            default_sample_weight = default_sample_weight / (default_sample_weight.mean() + 1e-6)
        
        results = {}
        for model_type in model_types:
            print(f"\n训练 {model_type.upper()} 模型")
            try:
                task = 'ranking' if model_type == 'lightgbm' else 'regression'
                model = MLFactorModel(model_type=model_type, task=task)
                
                # LGBM 特殊逻辑：使用原始收益作为排序标准，路径得分变换为权重
                current_y = y
                current_weight = default_sample_weight
                
                if model_type == 'lightgbm':
                    # [LGBM 标签/权重分离]
                    # 训练目标：用原始收益率(returns)作为排序标准，让模型直接优化收益排名
                    # 样本权重：用路径质量分(path_scores)变换为权重，让高质量路径的样本梯度更大
                    # 评估基准：_evaluate 中 reference=y=returns，与训练目标一致
                    print("  [LGBM 优化] 使用原始收益率(returns)作为标签，路径质量分变换为权重")
                    current_y = returns # 使用原始收益作为排序标准
                    
                    if path_scores is not None:
                        # 1. 取绝对值并处理异常值
                        processed_scores = np.abs(np.nan_to_num(path_scores, nan=0.0))
                        
                        # 2. 稳健的中位数平移
                        median_val = np.nanmedian(processed_scores)
                        shifted_scores = processed_scores - median_val
                        
                        # 3. 限制极端权重 (建议缩减 clip 范围)
                        # exp(2) 约 7.4 倍权重, exp(-2) 约 0.13 倍权重，这个跨度对模型比较友好
                        log_weight = np.clip(shifted_scores * 0.5, -1, 1) 
                        current_weight = np.exp(log_weight)
                        
                        # 4. 归一化：保持总梯度规模不变
                        # 这一步非常重要，防止因为加了权重导致整体 Learning Rate 失效
                        current_weight = current_weight / (current_weight.mean() + 1e-8)
                
                # 统一计算分组信息（所有任务通用，用于按组评估）
                # 注意：对于 LightGBM Ranking 任务，group 信息是必须的
                _, train_group = np.unique(dates[:split_idx], return_counts=True)
                _, val_group = np.unique(dates[split_idx:], return_counts=True)
                
                # 训练模型（直接传入 X 的视图，减少内存拷贝）
                train_result = model.train(X, current_y, validation_split=0.2, 
                                          use_time_series_split=True,
                                          feature_names=factor_names,
                                          sample_weight=current_weight,
                                          returns=returns,
                                          split_idx=split_idx,
                                          dates=dates,
                                          group=train_group,
                                          eval_group=val_group)
                
                self.models[model_type] = model
                results[model_type] = train_result
            except Exception as e:
                import traceback
                print(f"训练 {model_type} 失败: {e}")
                print(traceback.format_exc())
                continue
        
        return results

    def _apply_cross_sectional_normalization_inplace(self, X: np.ndarray, dates: np.ndarray, 
                                                   factor_names: List[str]):
        """
        原位对特征矩阵进行横截面归一化（按日期分组），降低内存占用。
        """
        # 精确匹配情绪因子集合，避免关键词匹配误伤同名技术因子（如 mean_return_20d）
        _sentiment_exact = {
            'up_ratio', 'strong_up_ratio', 'down_ratio',
            'limit_up_ratio', 'limit_down_ratio', 'mean_return',
            'total_volume', 'adv_vol_ratio', 'breadth_ma20',
            'market_type',
        }
        rank_cols_mask = np.array([col not in _sentiment_exact for col in factor_names])
        rank_cols_idx = np.where(rank_cols_mask)[0]
        
        if len(rank_cols_idx) > 0:
            unique_dates, group_start, group_counts = np.unique(
                dates, return_index=True, return_counts=True
            )
            
            for start, count in zip(group_start, group_counts):
                if count <= 1:
                    X[start:start+count, rank_cols_idx] = 0.5
                    continue
                
                # 仅提取当日数据进行排名
                day_data = X[start:start+count, rank_cols_idx]
                # 使用 scipy.stats.rankdata 进行向量化排名
                day_ranks = rankdata(day_data, method='average', axis=0) / (count + 1)
                X[start:start+count, rank_cols_idx] = day_ranks.astype(np.float32)
            
            import gc; gc.collect()
    
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
    
    def _apply_cross_sectional_normalization(self, X: np.ndarray, dates: np.ndarray, 
                                            factor_names: List[str]) -> np.ndarray:
        """
        对特征矩阵进行横截面归一化（按日期分组）
        
        参数:
            X: 特征矩阵
            dates: 日期数组
            factor_names: 特征名称列表
        
        返回:
            归一化后的特征矩阵
        """
        X_normalized = X.copy()
        
        # 获取日期分组
        _, date_group_start, date_group_counts = np.unique(
            dates, return_index=True, return_counts=True
        )
        
        # 识别需要进行横截面排名的因子索引
        # 排除市场情绪因子，因为它们在同一天对所有股票相同
        # 使用精确集合匹配，避免关键词匹配误伤同名技术因子（如 mean_return_20d）
        _sentiment_exact = {
            'up_ratio', 'strong_up_ratio', 'down_ratio',
            'limit_up_ratio', 'limit_down_ratio', 'mean_return',
            'total_volume', 'adv_vol_ratio', 'breadth_ma20',
            'market_type',
        }
        rank_cols_mask = np.array([col not in _sentiment_exact for col in factor_names])
        rank_cols_idx = np.where(rank_cols_mask)[0]
        
        if len(rank_cols_idx) > 0:
            print(f"    - 正在对 {len(rank_cols_idx)} 个特征进行向量化横截面归一化...")
            # 优化：使用 pandas 分组排序，极大提升大规模数据的处理速度（由数分钟缩短至数秒）
            # 注意：此处必须保持 input X 的原始索引顺序，pandas groupby.rank 默认满足
            rank_df = pd.DataFrame(X_normalized[:, rank_cols_idx], columns=[factor_names[i] for i in rank_cols_idx])
            rank_df['date'] = dates
            
            # 使用更加稳健的归一化映射 (rank / (n + 1))
            # groupby().rank() 返回与原 DataFrame 同形状且索引对齐的排名
            ranked_values = rank_df.groupby('date', sort=False).rank(method='average')
            
            # 映射到 (0, 1) 空间
            # transform('count') 会将每组的数量扩展到所有行，实现矢量化除法
            counts = rank_df.groupby('date', sort=False)['date'].transform('count')
            X_normalized[:, rank_cols_idx] = (ranked_values.values / (counts.values[:, np.newaxis] + 1)).astype(np.float32)
            
            del rank_df, ranked_values, counts
            import gc; gc.collect()
        
        return X_normalized
    
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
        """保存因子汇总信息（使用精确匹配，与审计报告逻辑一致）"""
        os.makedirs(save_dir, exist_ok=True)
        
        all_set = set(factor_names)
        
        # 精确匹配各类别
        _tech_set = set(self.tech_calculator.get_factor_names())
        _candle_set = set(self.candlestick_calculator.get_pattern_names())
        _engineered_set = set(self.feature_engineer.get_generated_features())
        _fund_known = set(FundamentalFactors.NUMERIC_COLS) | {
            'dynamic_pe', 'dynamic_pb', 'inv_pe', 'inv_pb', 'market_cap',
            'roe_x_np_growth', 'roe_to_pb', 'peg', 'sue', 'eav'
        }
        _sentiment_known = {
            'up_ratio', 'strong_up_ratio', 'down_ratio',
            'limit_up_ratio', 'limit_down_ratio', 'mean_return',
            'total_volume', 'adv_vol_ratio', 'breadth_ma20'
        }
        _adv_known = {
            'hl_range_mean', 'hl_range_std', 'oc_ratio_mean', 'oc_ratio_std',
            'price_volatility_20', 'price_volatility_60', 'price_skewness', 'price_kurtosis',
            'high_position', 'low_position',
            'volume_change_rate', 'volume_volatility', 'price_volume_corr',
            'amount_per_volume', 'amount_change_rate',
            'return_5d', 'return_10d', 'return_20d', 'return_60d',
            'momentum_5d', 'momentum_10d', 'momentum_20d', 'acceleration',
            'downside_risk', 'drawdown', 'max_drawdown_20', 'sharpe_ratio',
            'return_skewness', 'return_kurtosis',
        }
        _status_known = {'is_limit_up', 'is_suspended', 'market_type'}
        
        tech_names = sorted(all_set & _tech_set)
        candle_names = sorted(all_set & _candle_set)
        fund_names = sorted(all_set & _fund_known)
        sentiment_names = sorted(all_set & _sentiment_known)
        adv_names = sorted(all_set & _adv_known)
        engineered_names = sorted(all_set & _engineered_set)
        status_names = sorted(all_set & _status_known)
        classified = _tech_set | _candle_set | _fund_known | _sentiment_known | _adv_known | _engineered_set | _status_known
        other_names = sorted(all_set - classified)
        
        summary = {
            'total_factors': len(factor_names),
            'technical_factors': len(tech_names),
            'candlestick_factors': len(candle_names),
            'fundamental_factors': len(fund_names),
            'sentiment_factors': len(sentiment_names),
            'advanced_factors': len(adv_names),
            'engineered_factors': len(engineered_names),
            'status_factors': len(status_names),
            'other_factors': len(other_names),
            'factor_names': factor_names,
            'technical_factor_names': tech_names,
            'candlestick_factor_names': candle_names,
            'fundamental_factor_names': fund_names,
            'sentiment_factor_names': sentiment_names,
            'advanced_factor_names': adv_names,
            'engineered_factor_names': engineered_names,
            'status_factor_names': status_names,
            'other_factor_names': other_names,
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
    print(f"不可买入处理: {TrainingConfig.UNBUYABLE_HANDLING} | 缓存工程: {'开启' if args.cache_engineered else '关闭'}")
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
    dataset = trainer.prepare_dataset(
        stocks_data,
        cache_engineered_features=args.cache_engineered,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS
    )
    
    # 解析数据集
    if len(dataset) == 8:
        X, y, returns, factor_names, dates, unbuyable, limit_groups, path_scores = dataset
    else:
        X, y, returns, factor_names, dates, unbuyable, limit_groups = dataset
        path_scores = None
    
    # 6. 训练模型
    model_types = TrainingConfig.MODEL_TYPES
    results = trainer.train_models(X, y, returns, factor_names, dates, unbuyable, limit_groups, model_types, path_scores=path_scores)
    
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
