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
from numpy.lib.stride_tricks import sliding_window_view
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
        # 挂载元数据库以支持 is_st 查询 & 退市日期
        db_dir = os.path.dirname(self.db_path)
        meta_db = os.path.join(db_dir, 'stock_meta.db')
        if os.path.exists(meta_db):
            conn.execute(f"ATTACH DATABASE '{meta_db}' AS meta")
        
        conn.row_factory = sqlite3.Row
        
        # 预加载退市日期映射 {code: outDate_str or None}
        delist_map: Dict[str, Optional[str]] = {}
        try:
            placeholders_all = ','.join(['?' for _ in stock_codes])
            delist_df = pd.read_sql_query(
                f"SELECT code, outDate FROM meta.stock_basic WHERE code IN ({placeholders_all})",
                conn, params=stock_codes
            )
            for _, row in delist_df.iterrows():
                out = row['outDate']
                delist_map[row['code']] = out if (out and str(out).strip() not in ('', 'None', 'nan')) else None
        except Exception as e:
            print(f"  ⚠ 读取退市日期失败，退市特征将不可用: {e}")
        
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
                
                # 注入退市临近特征
                # days_to_delist: 距退市日的自然日数，已退市为 0，无退市计划为 -1（哨兵值）
                # 模型可从历史已退市股票的负样本中学习到"临近退市 → 规避"的规律
                out_date_str = delist_map.get(code)
                if out_date_str:
                    try:
                        out_dt = pd.Timestamp(out_date_str)
                        dates_ts = pd.to_datetime(stock_df['date'])
                        days_diff = (out_dt - dates_ts).dt.days.clip(lower=0).astype(np.float32)
                        stock_df['days_to_delist'] = days_diff
                    except Exception:
                        stock_df['days_to_delist'] = np.float32(-1)
                else:
                    stock_df['days_to_delist'] = np.float32(-1)
                
                stocks_data[code] = stock_df
            
            # 进度提示
            pbar.update(len(batch_codes))
        
        pbar.close()
        conn.close()
        
        delist_count = sum(1 for v in delist_map.values() if v is not None)
        print(f"成功加载并复权 {len(stocks_data)} 只股票的数据 (其中 {delist_count} 只含退市日期)")
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
                    import pyarrow.parquet as pq
                    pf = pq.read_table(cache_file, columns=['date'])
                    last_row = pf.to_pandas().tail(1)
                    
                    if not last_row.empty:
                        cache_last_date = str(last_row['date'].iloc[0])
                        data_last_date = str(data['date'].max())
                        
                        if cache_last_date >= data_last_date:
                            # 日期已是最新，还需检查列是否与 target_features 一致
                            if target_features is not None:
                                cached_cols = set(pq.read_schema(cache_file).names)
                                missing = [f for f in target_features if f not in cached_cols]
                                if missing:
                                    # 列不匹配，需要重算
                                    to_update[code] = data
                                    continue
                            skipped += 1
                            continue
                except Exception:
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
        is_atty = sys.stdout.isatty()
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
            
            with tqdm(total=len(futures), desc="更新因子缓存", disable=not is_atty) as pbar:
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
        路径质量评分函数，综合评估持仓期间的收益、回撤风险和路径质量。

        设计目标：
          - 标签截面均值接近 0，正负各半，便于模型学习相对排序
          - 低波动稳涨 > 高波动暴涨（波动率归一化）
          - 大幅回撤受非线性惩罚（小回撤线性，破位后二次方放大）
          - V型反转（先大跌后拉回）额外打折
          - 高点早出现给予小幅奖励

        参数:
            f_returns:  未来 N 日收益率（相对 T+1 开盘）
            f_low_min:  未来 N 日最低价
            f_high_idx: 未来 N 日最高点所在日索引
            f_low_idx:  未来 N 日最低点所在日索引
            atr_raw:    当前 ATR（绝对值，单位：元）
            next_open:  T+1 开盘价（买入价）
            rel_atr:    相对 ATR = atr_raw / close
        """
        eps = 1e-4

        # ---------------------------------------------------------
        # 1. 核心收益项（波动率归一化，类夏普思路）
        #    除以 rel_atr**0.5 使低波动稳涨得分更高
        # ---------------------------------------------------------
        core_term = f_returns / (rel_atr ** 0.5) * getattr(TrainingConfig, 'LABEL_TARGET_SCALE', 2.0)

        # ---------------------------------------------------------
        # 2. 损失厌恶项（非线性穿透惩罚）
        #    downside_gap: 买入价到最低价的距离，以 ATR 为单位（仅计正向回撤）
        #    <= 1 ATR：线性惩罚（正常波动范围内）
        #    >  1 ATR：超出部分二次方放大（破位惩罚）
        # ---------------------------------------------------------
        lambda_val   = getattr(TrainingConfig, 'LABEL_LAMBDA', 0.35)
        downside_gap = np.maximum(next_open - f_low_min, 0) / (atr_raw + eps)

        linear_part    = np.minimum(downside_gap, 1.0)
        nonlinear_part = np.maximum(downside_gap - 1.0, 0.0) ** 2
        loss_aversion  = -lambda_val * (linear_part + nonlinear_part)

        # ---------------------------------------------------------
        # 3. 路径保护惩罚（V型反转过滤）
        #    条件：低点早于高点 且 下跌超过 1 ATR
        #    对正收益按超出 1 ATR 的深度打折，避免先大跌再拉回的票得高分
        # ---------------------------------------------------------
        path_punish_coef = getattr(TrainingConfig, 'LABEL_PATH_PUNISH', 0.4)
        is_v_shape   = (f_low_idx < f_high_idx) & (downside_gap > 1.0)
        path_penalty = np.where(
            is_v_shape & (core_term > 0),
            -path_punish_coef * np.clip(downside_gap - 1.0, 0, 1.5),
            0
        )

        # ---------------------------------------------------------
        # 4. 资金效率奖励（高点早出现）
        #    条件：高点在前 3 日内 且 核心收益 > 0.5（避免对微涨票过度奖励）
        #    奖励幅度较小，仅作为同等收益下的微弱加分
        # ---------------------------------------------------------
        time_bonus = np.where(
            (f_high_idx < 3) & (core_term > 0.5),
            getattr(TrainingConfig, 'LABEL_TIME_BONUS', 0.15) * core_term,
            0
        )

        final_score = core_term + loss_aversion + path_penalty + time_bonus

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
                # 向量化实现：用 stride_tricks 替代 rolling().apply(np.argmax/argmin)
                # 避免逐行 Python 回调，速度提升 10-50x
                #
                # 原逻辑：rolling(w).apply(argmax, raw=True).shift(-w)
                #   → 索引 i 对应窗口 arr[i-w+1:i+1] 的 argmax，shift(-w) 后
                #     索引 i 拿到的是 arr[i+1:i+w+1]（T+1 起的未来 w 行）的 argmax
                # 等价实现：对 arr[1:] 做 sliding_window_view，窗口大小 w
                #   → view[i] = arr[i+1 : i+w+1]，argmax 结果直接对应索引 i
                _n = len(high)
                _w = forward_days
                _high_arr = high.values.astype(np.float64)
                _low_arr  = low.values.astype(np.float64)

                # 从 arr[1:] 开始，view[i] = arr[i+1 : i+w+1]，长度 = _n - _w
                _high_wins = sliding_window_view(_high_arr[1:], _w)
                _low_wins  = sliding_window_view(_low_arr[1:],  _w)

                _high_idx_raw = np.argmax(_high_wins, axis=1).astype(np.float64)  # 长度 _n - _w
                _low_idx_raw  = np.argmin(_low_wins,  axis=1).astype(np.float64)

                # 末尾 _w 行没有完整未来窗口，填 NaN
                _pad = np.full(_w, np.nan)
                _high_idx_aligned = np.concatenate([_high_idx_raw, _pad])[:_n]
                _low_idx_aligned  = np.concatenate([_low_idx_raw,  _pad])[:_n]

                f_high_idx = pd.Series(_high_idx_aligned, index=high.index)
                f_low_idx  = pd.Series(_low_idx_aligned,  index=low.index)

                # 计算当前波动率 (ATR) 作为分母，衡量收益的“性价比”
                # 使用相对 ATR (ATR / close)
                atr_raw = talib.ATR(high.values, low.values, close.values, timeperiod=strategy_config.ATR_PERIOD)
                atr_rel = atr_raw / close.values
                    

                y = self._compute_path_quality_score(f_returns.values, f_low_min.values, 
                                                   f_high_idx.values, f_low_idx.values, 
                                                   atr_raw, next_open.values, atr_rel)
                
                # 退市临近惩罚：对 30 个自然日内将退市的样本，将标签压至极低值
                # 模型不会在推理时看到 days_to_delist，但会从其他特征（量价、ST状态等）
                # 中学到"临近退市股票的共性模式 → 规避"
                if 'days_to_delist' in data.columns:
                    dtd = data['days_to_delist'].values  # -1 表示无退市计划
                    delist_penalty_mask = (dtd >= 0) & (dtd <= getattr(TrainingConfig, 'DELIST_PENALTY_DAYS', 30))
                    if delist_penalty_mask.any():
                        y = y.copy() if isinstance(y, np.ndarray) else np.array(y, dtype=np.float32)
                        # 压到接近 0 的极低分，远低于正常股票的均值（约 1.0）
                        y[delist_penalty_mask] = getattr(TrainingConfig, 'DELIST_PENALTY_SCORE', 0.01)

                # ST 标签惩罚：直接压低 ST 样本的标签，让模型从标签层面学到"ST = 低质量"
                # 仅靠 sample_weight 降权不够，因为 ST 股票炒作时收益排名靠前，
                # 降权后绝对权重仍高，模型仍会学到"ST 特征 → 高分"的错误映射
                st_label_score = getattr(TrainingConfig, 'ST_LABEL_SCORE', None)
                if st_label_score is not None and 'is_st' in data.columns:
                    st_mask = data['is_st'].values == 1
                    if st_mask.any():
                        y = y.copy() if isinstance(y, np.ndarray) else np.array(y, dtype=np.float32)
                        # 只压低标签，不完全清零，保留少量信号避免模型对 ST 特征过拟合到极端值
                        y[st_mask] = np.minimum(y[st_mask], st_label_score)
                
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
                    
                    if code.startswith(MARKET_PREFIXES['sz_gem']) or code.startswith(MARKET_PREFIXES['star']):
                        limit_thresholds[:] = MARKET_LIMITS['gem_star']
                    elif code.startswith(MARKET_PREFIXES['bj']):
                        limit_thresholds[:] = MARKET_LIMITS['bj']
                    
                    # 只有主板ST才是5%；创业板/科创板ST仍是20%，北交所ST仍是30%
                    if 'is_st' in data.columns:
                        is_main_board = ~(
                            code.startswith(MARKET_PREFIXES['sz_gem']) or
                            code.startswith(MARKET_PREFIXES['star']) or
                            code.startswith(MARKET_PREFIXES['bj'])
                        )
                        if is_main_board:
                            limit_thresholds[data['is_st'] == 1] = MARKET_LIMITS['st']
                    
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
                    is_st_mask = is_st_series.values
                    
                    if len(X_df) > 0:
                        return X_df, final_y, final_returns, dates, unbuyable_mask, limit_groups, is_st_mask
            
            return None, None, None, None, None, None, None
        
        except Exception as e:
            import traceback
            print(f"  警告: 处理股票 {code} 失败: {e}")
            print(traceback.format_exc())
            return None, None, None, None, None, None, None

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
        is_st_arr = np.empty(actual_rows, dtype=bool)
        
        # 3. 填充数据并立即从 results 中剔除已处理对象
        print("  - 正在合并数据 (Incremental Fill)...")
        cursor = 0
        for idx in valid_results_indices:
            res = results[idx]
            n = len(res[0])
            
            # 分解结果
            if len(res) == 7:
                X, y, r, d, u, l, is_st = res
            elif len(res) == 6:
                X, y, r, d, u, l = res
                is_st = np.zeros(n, dtype=bool)
            else:
                X, y, r, d, u = res
                l = np.full(n, 0.1, dtype=np.float32)
                is_st = np.zeros(n, dtype=bool)

            X_arr[cursor:cursor+n] = X.values.astype(np.float32, copy=False)
            y_raw_arr[cursor:cursor+n] = y.values if isinstance(y, pd.Series) else y
            returns_arr[cursor:cursor+n] = r
            dates_arr[cursor:cursor+n] = d
            unbuyable_arr[cursor:cursor+n] = u
            limit_groups_arr[cursor:cursor+n] = l
            is_st_arr[cursor:cursor+n] = is_st
            
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
        is_st_arr = is_st_arr[sort_idx]
        
        # X_arr 排序
        X_sorted = X_arr[sort_idx] 
        del X_arr, sort_idx
        X_arr = X_sorted
        del X_sorted
        gc.collect()
        all_cols = col_names
        # 3. 不可买入样本处理 (涨停/停牌) - 提前到归一化之前，确保排序不含涨停股
        penalty_count = np.sum(unbuyable_arr)
        if penalty_count > 0:
            handling = getattr(TrainingConfig, 'UNBUYABLE_HANDLING', 'remove')
            if handling == 'remove':
                print(f"  - 剔除不可买入样本 (排序前): 正在剔除 {penalty_count} 个涨停/停牌样本")
                keep_mask = ~unbuyable_arr
                X_arr        = X_arr[keep_mask]
                y_raw_arr    = y_raw_arr[keep_mask]
                returns_arr  = returns_arr[keep_mask]
                dates_arr    = dates_arr[keep_mask]
                limit_groups_arr = limit_groups_arr[keep_mask]
                is_st_arr    = is_st_arr[keep_mask]
            else:
                print(f"  - 施加不可买入惩罚: 将 {penalty_count} 个涨停/停牌标的的标签强制设为 0.05")
                y_raw_arr[unbuyable_arr] = 0.05

        # 日期分组信息（仅计算一次，供后续归一化和 group 划分共用）
        _, date_group_start, date_group_counts = np.unique(dates_arr, return_index=True, return_counts=True)

        # 2. 标签值映射 (Cross-Section Rank Normalization)
        # 每日全市场横截面排名归一化，跨板块统一尺度，输出 (0, 1)
        # 注意：此处生成的 y_norm_arr 仅供 XGB 等回归模型默认使用
        y_norm_arr = np.empty_like(y_raw_arr)
        print("  - 正在进行每日全市场横截面排名归一化标签...")
        for start, count in zip(date_group_start, date_group_counts):
            end = start + count
            if count > 1:
                y_norm_arr[start:end] = rankdata(y_raw_arr[start:end], method='average') / (count + 1)
            else:
                y_norm_arr[start:end] = 0.5


        
        # 统计因子分类详情 (Factor Audit Report)
        # 使用各模块的精确列名进行匹配，而非关键词启发式匹配
        remaining_all = set(all_cols)
        
        # 1. 状态因子 (comprehensive_factor_calculator 中硬编码的3个 + 退市特征)
        _status_known = {'is_limit_up', 'is_suspended', 'market_type', 'days_to_delist'}
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
        return X_arr, y_norm_arr, returns_arr, all_cols, dates_arr, unbuyable_arr, limit_groups_arr, y_raw_arr, is_st_arr
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    returns: np.ndarray,
                    factor_names: List[str],
                    dates: np.ndarray,
                    unbuyable_mask: np.ndarray = None,
                    limit_groups: np.ndarray = None,
                    model_types: List[str] = TrainingConfig.MODEL_TYPES,
                    path_scores: np.ndarray = None,
                    is_st_arr: np.ndarray = None) -> Dict:
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
        
        # 先进行时间序列划分
        raw_split_idx = int(len(dates) * TrainingConfig.TRAIN_TEST_SPLIT)
        split_date = dates[raw_split_idx]
        split_idx = np.searchsorted(dates, split_date, side='left')
        
        print(f"  划分点: {split_date}, 索引: {split_idx}")
        print(f"  训练集: {split_idx} 样本, 验证集: {len(dates) - split_idx} 样本")
        
        # 训练集：正常横截面排名归一化，同时收集每日统计量
        print("\n  对训练样本进行横截面归一化，并收集统计量...")
        train_daily_stats = self._apply_cross_sectional_normalization_inplace(
            X[:split_idx], dates[:split_idx], factor_names, collect_stats=True
        )
        # 验证集用训练集统计量做参考（ref_stats 传入但归一化方式与训练集保持一致）
        # 避免验证集用自己的横截面排名（look-ahead bias：推理时不知道当天全市场分布）
        print("  对验证样本进行归一化（与训练集保持相同的横截面排名方式）...")
        self._apply_cross_sectional_normalization_inplace(
            X[split_idx:], dates[split_idx:], factor_names,
            ref_stats=train_daily_stats, ref_window=60
        )

        # 分组信息（dates 已全局排序，unique 结果直接用于 group 参数）
        _, train_group = np.unique(dates[:split_idx], return_counts=True)
        _, val_group   = np.unique(dates[split_idx:], return_counts=True)

        # 修复标签泄漏：y 是全量数据上的横截面排名归一化，验证集参与了全局排名
        # 需要在 split 之后，对训练集和验证集分别重新做每日横截面排名归一化
        _label_source = path_scores if path_scores is not None else y
        y_clean = np.empty(len(dates), dtype=np.float32)
        for _start, _end in [(0, split_idx), (split_idx, len(dates))]:
            _dates_sub = dates[_start:_end]
            _scores_sub = _label_source[_start:_end]
            _, _d_starts, _d_counts = np.unique(_dates_sub, return_index=True, return_counts=True)
            for _ds, _dc in zip(_d_starts, _d_counts):
                _de = _ds + _dc
                if _dc > 1:
                    y_clean[_start + _ds:_start + _de] = rankdata(_scores_sub[_ds:_de], method='average') / (_dc + 1)
                else:
                    y_clean[_start + _ds:_start + _de] = 0.5
        y = y_clean
        print(f"  [标签] 已在 train/val 分割后分别重新做横截面排名归一化，消除标签泄漏")

        # === 排名权重方案 (全市场中性化排名) ===
        # 1. 中性化：消除板块差异
        returns_normalized = returns / (limit_groups + 1e-8)
        
        # 2. 在 train/val 分割后分别计算排名权重，避免验证集权重包含训练集收益分布信息
        rank_scores = np.full(len(returns), np.nan, dtype=np.float32)
        for _w_start, _w_end in [(0, split_idx), (split_idx, len(returns))]:
            _ret_sub = returns_normalized[_w_start:_w_end]
            _valid = ~np.isnan(_ret_sub)
            if _valid.any():
                _ranks = rankdata(_ret_sub[_valid], method='average')
                _n = np.sum(_valid)
                _scores = np.full(len(_ret_sub), np.nan, dtype=np.float32)
                _scores[_valid] = (_ranks - 1.0) / (_n - 1.0) if _n > 1 else 0.5
                rank_scores[_w_start:_w_end] = _scores
        
        returns_weight = rank_scores
        
        # 叠加 ST 降权（如果已计算）
        ST_WEIGHT_FACTOR = getattr(TrainingConfig, 'ST_WEIGHT_FACTOR', None)
        if is_st_arr is not None and ST_WEIGHT_FACTOR is not None and ST_WEIGHT_FACTOR < 1.0:
            st_weight = np.ones(len(is_st_arr), dtype=np.float32)
            st_weight[is_st_arr] = st_weight[is_st_arr] * ST_WEIGHT_FACTOR
            returns_weight = returns_weight * st_weight

        # 统一归一化到均值 1.0，无论是否有 ST 降权，保证权重尺度稳定
        _valid_w = ~np.isnan(returns_weight)
        if _valid_w.any():
            returns_weight = returns_weight / (returns_weight[_valid_w].mean() + 1e-8)

        # 开关：是否使用自定义样本权重
        use_sample_weight = getattr(TrainingConfig, 'USE_SAMPLE_WEIGHT', True)
        if not use_sample_weight:
            returns_weight = None
            print("  [样本权重] 已禁用自定义样本权重，使用均匀权重")
        else:
            print("  [样本权重] 已启用自定义样本权重（收益率排名 + ST 降权）")

        results = {}
        for model_type in model_types:
            print(f"\n训练 {model_type.upper()} 模型")
            try:
                task = 'ranking' if model_type == 'lightgbm' else 'regression'
                model = MLFactorModel(model_type=model_type, task=task)
                
                # [正交设计]
                # 标签：路径质量分（稳定性）
                #   - LGBM ranking：使用原始路径质量分(path_scores)排序，保留分布形态
                #   - XGB regression：使用每日板块中性化归一化后的路径质量分(y)
                # 权重：基于原始收益率，强调高收益样本
                swap_label_weight = getattr(TrainingConfig, 'SWAP_LABEL_WEIGHT', False)
                if swap_label_weight and returns_weight is not None:
                    # 交换：标签←收益排名权重，权重←路径质量分（平移归一化为非负）
                    _rw = returns_weight.astype(np.float32) if isinstance(returns_weight, np.ndarray) else np.array(returns_weight, dtype=np.float32)
                    # LGBM lambdarank 要求标签为非负整数（relevance grade）
                    # 将 [0,1] 的排名分数量化为 0~9 的整数 grade
                    if model_type == 'lightgbm':
                        current_y = np.floor(_rw * 9).clip(0, 9).astype(np.int32)
                    else:
                        current_y = _rw
                    path_as_weight = np.array(y, dtype=np.float32)
                    path_as_weight = path_as_weight - path_as_weight.min()  # 平移至非负
                    path_as_weight = path_as_weight / (path_as_weight.mean() + 1e-8)
                    current_weight = path_as_weight
                    print(f"  [{model_type.upper()}] [SWAP] 标签=收益排名权重{'(整数grade)' if model_type == 'lightgbm' else ''}, 权重=路径质量分(归一化)")
                else:
                    current_weight = returns_weight
                    if model_type == 'lightgbm':
                        # LGBM ranking：用原始路径质量分（非均匀分布）做分档依据
                        # 避免对已均匀化的 y_clean 再做百分位分档导致 label_gain 失效
                        lgbm_label_source = path_scores if path_scores is not None else y
                        current_y = lgbm_label_source
                        print("  [LGBM] 标签=原始路径质量分(ranking分档), 权重=收益归一化权重")
                    else:
                        # XGB regression：标签用每日板块中性化归一化后的路径质量分
                        current_y = y
                        print("  [XGB] 标签=归一化路径质量分(regression), 权重=收益归一化权重")
                
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
                                                   factor_names: List[str],
                                                   collect_stats: bool = False,
                                                   ref_stats: dict = None,
                                                   ref_window: int = 60):
        """
        原位对特征矩阵进行横截面归一化（按日期分组），降低内存占用。

        参数:
            collect_stats: 若为 True，收集每日的均值/标准差并返回，供验证集使用
            ref_stats: 若提供，则用该统计量（训练集末尾 ref_window 天的均值/std）
                       对当前数据做 z-score 归一化，而非用自身横截面排名
                       这消除了验证集的 look-ahead bias（推理时不知道当天全市场分布）
            ref_window: 从 ref_stats 中取最后多少个交易日的统计量做平均
        """
        # 跳过横截面排名归一化的特征集合
        # 原则：以下类型的特征不具备"今日全市场排名"的语义，归一化会破坏其信息：
        #   1. 全市场情绪因子 —— 所有股票当天值相同，排名无意义
        #   2. 类别/枚举特征 —— market_type, industry_encoded 等整数编码
        #   3. 0/1 二值标志位 —— is_limit_up, is_suspended, K线形态信号, One-Hot 行业列
        #   4. 绝对天数 —— days_to_delist（排名会把"还有500天退市"和"还有1天退市"混在一起）
        _skip_normalization = {
            # 全市场情绪因子（所有股票当天值相同）
            'up_ratio', 'strong_up_ratio', 'down_ratio',
            'limit_up_ratio', 'limit_down_ratio', 'mean_return',
            'total_volume', 'adv_vol_ratio', 'breadth_ma20',
            # 类别/枚举特征
            'market_type', 'industry_encoded',
            # 绝对天数
            'days_to_delist',
            # 0/1 交易状态标志位
            'is_limit_up', 'is_suspended',
            # K线形态 0/1 信号（19个形态 + 强度指标不是0/1但也是有界的，保留归一化）
            'white_candle', 'black_candle', 'doji', 'hammer', 'hanging_man',
            'shooting_star', 'inverted_hammer', 'marubozu', 'spinning_top',
            'bullish_engulfing', 'bearish_engulfing', 'piercing_line',
            'dark_cloud_cover', 'morning_star', 'evening_star', 'harami',
            'three_white_soldiers', 'three_black_crows',
        }
        # One-Hot 行业列（前缀匹配）
        def _should_skip(col: str) -> bool:
            if col in _skip_normalization:
                return True
            # industry_农、林、牧、渔业 等 One-Hot 列
            if col.startswith('industry_') and not col.endswith('_encoded'):
                return True
            return False

        rank_cols_mask = np.array([not _should_skip(col) for col in factor_names])
        rank_cols_idx = np.where(rank_cols_mask)[0]
        
        if len(rank_cols_idx) == 0:
            return {} if collect_stats else None

        unique_dates, group_start, group_counts = np.unique(
            dates, return_index=True, return_counts=True
        )

        # 模式A：验证集同样使用横截面排名归一化，保持与训练集一致的特征分布
        # 注意：之前用 z-score → sigmoid 会导致验证集特征分布与训练集不一致，
        # 是验证集损失无法下降的根本原因。ref_stats 参数保留但不再用于改变归一化方式。
        if ref_stats is not None:
            for start, count in zip(group_start, group_counts):
                if count <= 1:
                    X[start:start+count, rank_cols_idx] = 0.5
                    continue
                day_data = X[start:start+count, rank_cols_idx]
                day_ranks = rankdata(day_data, method='average', axis=0) / (count + 1)
                X[start:start+count, rank_cols_idx] = day_ranks.astype(np.float32)

            import gc; gc.collect()
            return None

        # 模式B：正常横截面排名归一化（训练集使用）
        daily_stats = {} if collect_stats else None

        for start, count in zip(group_start, group_counts):
            if count <= 1:
                X[start:start+count, rank_cols_idx] = 0.5
                if collect_stats:
                    daily_stats[unique_dates[np.searchsorted(group_start, start)]] = {
                        'mean': X[start, rank_cols_idx].copy(),
                        'std':  np.ones(len(rank_cols_idx), dtype=np.float32),
                    }
                continue

            day_data = X[start:start+count, rank_cols_idx]

            if collect_stats:
                # 收集归一化前的原始统计量，供验证集 z-score 使用
                d_mean = np.nanmean(day_data, axis=0).astype(np.float32)
                d_std  = np.nanstd(day_data,  axis=0).astype(np.float32)
                date_key = unique_dates[np.searchsorted(group_start, start)]
                daily_stats[date_key] = {'mean': d_mean, 'std': d_std}

            day_ranks = rankdata(day_data, method='average', axis=0) / (count + 1)
            X[start:start+count, rank_cols_idx] = day_ranks.astype(np.float32)

        import gc; gc.collect()
        return daily_stats if collect_stats else None
    
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
        _status_known = {'is_limit_up', 'is_suspended', 'market_type', 'days_to_delist'}
        
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
