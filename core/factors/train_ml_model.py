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
import gc

from core.factors.quantitative_factors import QuantitativeFactors
from core.factors.candlestick_pattern_factors import CandlestickPatternFactors
from core.factors.fundamental_factors import FundamentalFactors
from core.factors.ml_factor_model import MLFactorModel
from core.factors.comprehensive_factor_calculator import ComprehensiveFactorCalculator
from core.factors.advanced_factors import TimeSeriesFactors, RiskFactors
from core.factors.factor_filler import FactorFiller, fill_factors_with_defaults
from core.data.market_sentiment_calculator import MarketSentimentCalculator
from config import DATABASE_PATH, TrainingConfig, FactorConfig, MARKET_LIMITS, MARKET_PREFIXES,ModelConfig


# ============================================================================
# 顶层 worker 函数（进程池要求 picklable，必须定义在模块顶层）
# ============================================================================

def _cache_worker(args):
    """
    进程池 worker：在独立子进程中计算并保存单只股票的因子缓存。
    
    使用顶层函数而非实例方法，是因为 ProcessPoolExecutor 要求参数可 pickle，
    而含有 SQLite 连接的对象无法跨进程序列化。
    每个子进程独立创建计算器实例，避免共享状态。
    """
    code, data, db_path, factors_cache_dir, target_features, include_fundamentals = args
    try:
        import io, contextlib
        # 子进程内直接构建轻量计算器，避免 MLModelTrainer.__init__ 的打印输出
        from core.factors.comprehensive_factor_calculator import ComprehensiveFactorCalculator
        
        # 复用 calculate_and_save_factors 的逻辑，但不依赖完整的 MLModelTrainer
        # 直接构造一个最小化的 trainer-like 对象
        trainer = MLModelTrainer.__new__(MLModelTrainer)
        trainer.db_path = db_path
        trainer.task = 'hybrid'
        trainer.punish_unbuyable = False
        trainer.factors_cache_dir = factors_cache_dir
        trainer.models = {}
        trainer.factor_calculator = ComprehensiveFactorCalculator(db_path)
        
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.calculate_and_save_factors(
                code, data,
                target_features=target_features,
                include_fundamentals=include_fundamentals,
                verbose=False
            )
        return code, True, None
    except Exception as e:
        return code, False, str(e)


def _scan_cache_file(args):
    """并行扫描单个缓存文件状态，返回 (code, needs_update: bool)"""
    code, data_last_date, cache_file, target_features = args
    try:
        import pyarrow.parquet as pq
        if not os.path.exists(cache_file):
            return code, True
        pf = pq.read_table(cache_file, columns=['date'])
        last_row = pf.to_pandas().tail(1)
        if last_row.empty:
            return code, True
        cache_last_date = str(last_row['date'].iloc[0])
        if cache_last_date < data_last_date:
            return code, True
        # 日期已是最新，检查列是否匹配
        if target_features is not None:
            cached_cols = set(pq.read_schema(cache_file).names)
            missing = [f for f in target_features if f not in cached_cols]
            if missing:
                return code, True
        return code, False
    except Exception:
        return code, True


class MLModelTrainer:
    """机器学习模型训练器"""
    
    def __init__(self, db_path: str = DATABASE_PATH, punish_unbuyable: bool = False):
        """
        初始化训练器

        参数:
            db_path: 数据库路径
            punish_unbuyable: 保留参数，用于归档目录命名（实际处理逻辑由 UNBUYABLE_HANDLING 控制）
        """
        self.db_path = db_path
        self.task = 'hybrid'
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
                            # bfill 先向后填充（处理开头缺失），再 ffill 向前填充（处理中间缺失）
                            # 避免纯 ffill 在序列开头缺失时回退到 fillna(1.0) 导致价格跳变
                            ratio = stock_df['fore_adjust_factor'].bfill().ffill().fillna(1.0) / base_factor
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
            n_jobs: 并行进程数（使用 ProcessPoolExecutor 实现真正的多核并行）
            verbose: 是否输出详细信息
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        
        # 1. 快速去重与预扫描
        all_codes = list(stocks_data.keys())
        
        # 2. 增量跳过：并行扫描磁盘缓存状态（I/O 密集，用线程池加速）
        from concurrent.futures import ThreadPoolExecutor
        to_update = {}
        skipped = 0
        
        print(f"\n[因子缓存同步] 正在并行扫描磁盘缓存状态 ({len(all_codes)} 只)...")
        scan_args = []
        for code in all_codes:
            data = stocks_data[code]
            if data.empty:
                continue
            cache_file = os.path.join(self.factors_cache_dir, f'{code}_factors.parquet')
            data_last_date = str(data['date'].max())
            scan_args.append((code, data_last_date, cache_file, target_features))
        
        # 用线程池并行扫描（I/O 密集型，线程足够）
        scan_workers = min(32, len(scan_args))
        with ThreadPoolExecutor(max_workers=scan_workers) as scan_executor:
            for code, needs_update in scan_executor.map(_scan_cache_file, scan_args):
                if needs_update:
                    to_update[code] = stocks_data[code]
                else:
                    skipped += 1

        if skipped > 0:
            print(f"  已跳过 {skipped} 只已同步的股票缓存")
            
        if not to_update:
            print(f"[OK] 缓存已是最新，无需更新。")
            return

        # 3. 使用 ProcessPoolExecutor 实现真正的多核并行（绕过 GIL）
        # 因子计算是 CPU 密集型（talib、pandas rolling、numpy），多进程可充分利用多核
        effective_jobs = min(n_jobs, multiprocessing.cpu_count(), len(to_update))
        print(f"  正在多进程并行更新 {len(to_update)} 只股票的缓存 (进程数={effective_jobs})...")
        
        worker_args = [
            (code, data, self.db_path, self.factors_cache_dir, target_features, include_fundamentals)
            for code, data in to_update.items()
        ]
        
        start_time = time()
        success = 0
        failed = 0
        is_atty = sys.stdout.isatty()
        
        with ProcessPoolExecutor(max_workers=effective_jobs) as executor:
            futures = {executor.submit(_cache_worker, arg): arg[0] for arg in worker_args}
            
            with tqdm(total=len(futures), desc="更新因子缓存", disable=not is_atty) as pbar:
                for future in as_completed(futures):
                    try:
                        code, ok, err = future.result()
                        if ok:
                            success += 1
                        else:
                            tqdm.write(f"  ✗ {code} 缓存更新失败: {err}")
                            failed += 1
                    except Exception as e:
                        code = futures[future]
                        tqdm.write(f"  ✗ {code} 进程异常: {e}")
                        failed += 1
                    pbar.set_postfix({"成功": success, "失败": failed})
                    pbar.update(1)
                    
        elapsed = time() - start_time
        print(f"✓ 缓存同步完成: 成功 {success}, 失败 {failed} | 已跳过 {skipped} | 耗时 {elapsed:.1f}s")
    



    def _compute_path_quality_score(self, f_returns_norm, f_low_min_norm, f_high_idx, f_low_idx, atr_raw, next_open, rel_atr, f_high_max_norm, intraday_intensity, volume_ratio, relative_intensity, limits=0.1,):

        upside  = np.where(f_high_max_norm > 0, f_high_max_norm, 0)
        downside = np.where(f_low_min_norm  < 0, f_low_min_norm,  0)

        # 从配置中读取权重
        w_upside   = getattr(TrainingConfig, 'UPSIDE_WEIGHT',        2.0)
        w_downside = getattr(TrainingConfig, 'DOWNSIDE_WEIGHT',       1.0)
        w_final    = getattr(TrainingConfig, 'FINAL_RETURN_WEIGHT',   3.0)

        # ── 1. 基础得分：落袋收益 + 上行空间 - 下行伤害 ──────────────────
        base_score = (f_returns_norm * w_final) + (upside * w_upside) + (downside * w_downside)

        # ── 2. 波动率调整：ATR 越大爆发力越强 ────────────────────────────
        vol_booster = 1.0 + (rel_atr * 10.0)

        # ── 3. 入场日动能乘数 (intraday_intensity × relative_intensity) ──
        #   intraday_intensity = 当日振幅 / ATR，反映绝对爆发力
        #   relative_intensity = 当日振幅/ATR 相对近5日均值，反映相对活跃度
        #   两者乘积 > 1 表示当日动能高于近期均值，给予正向加权
        #   clip 到 [0.5, 2.0]，避免极端值主导得分


        # ── 6. 合并所有调制因子 ───────────────────────────────────────────
        if TrainingConfig.SHORT_PREDICTION:
            momentum_mult = np.clip(
                getattr(TrainingConfig, 'MOMENTUM_MULT_BASE', 0.5) +
                getattr(TrainingConfig, 'MOMENTUM_MULT_SCALE', 0.5) *
                np.clip(intraday_intensity * relative_intensity, 0.0, 4.0),
                0.5, 2.0
            )

            # ── 4. 资金参与度乘数 (volume_ratio) ─────────────────────────────
            #   volume_ratio = 当日量 / 20日均量
            #   无量行情（< 0.5）折扣；放量行情（> 2.0）适度奖励
            #   clip 到 [0.5, 1.5]，防止单日天量过度放大
            volume_mult = np.clip(
                getattr(TrainingConfig, 'VOLUME_MULT_BASE', 0.5) +
                getattr(TrainingConfig, 'VOLUME_MULT_SCALE', 0.5) *
                np.clip(volume_ratio, 0.0, 3.0) / 3.0,
                0.5, 1.5
            )

            # ── 5. 路径形态奖惩 (f_high_idx vs f_low_idx) ────────────────────
            #   f_high_idx：持仓期内最高点出现在第几天（0-based）
            #   f_low_idx ：持仓期内最低点出现在第几天（0-based）
            #   先涨后跌（high_idx < low_idx）：路径友好，给予奖励
            #   先跌后涨（low_idx < high_idx）：路径不友好，给予惩罚
            #   两者相等或含 NaN 时保持中性（1.0）
            high_idx = np.asarray(f_high_idx, dtype=np.float64)
            low_idx  = np.asarray(f_low_idx,  dtype=np.float64)
            path_bonus = getattr(TrainingConfig, 'PATH_BONUS',   0.15)  # 先涨后跌奖励幅度
            path_penalty = getattr(TrainingConfig, 'PATH_PENALTY', 0.10) # 先跌后涨惩罚幅度
            path_mult = np.where(
                np.isnan(high_idx) | np.isnan(low_idx),
                1.0,
                np.where(high_idx < low_idx, 1.0 + path_bonus,   # 先涨后跌：路径优质
                np.where(low_idx  < high_idx, 1.0 - path_penalty, # 先跌后涨：路径劣质
                1.0))                                              # 同天：中性
            )
            final_score = base_score * vol_booster * momentum_mult * volume_mult * path_mult
        else:
            final_score = base_score * vol_booster
        return final_score

    def _extract_stock_components(self, code: str, data: pd.DataFrame, 
                                 forward_days: int,
                                 apply_feature_engineering: bool = False,
                                 target_features: Optional[List[str]] = None,
                                 verbose: bool = False,
                                 train_start_date: str = None,
                                 train_end_date: str = None,
                                 include_fundamentals: bool = True) -> Optional[dict]:
        """
        提取单只股票的特征和用于构造标签的原始组件。
        """
        try:
            # 1. 因子计算 (逻辑保持不变)
            if train_end_date is not None and 'date' in data.columns:
                max_lookback = 300 # 保守回看
                data_for_factors = data[data['date'] <= train_end_date].copy()
                if len(data_for_factors) < 100:
                    data_for_factors = data
            else:
                data_for_factors = data

            factors = self.calculate_and_save_factors(code, data_for_factors, apply_feature_engineering, target_features, verbose, include_fundamentals)
            
            if factors is not None:
                if 'date' in factors.columns and 'date' in data.columns:
                    factors = pd.merge(data[['date']], factors, on='date', how='left')
                    factors = factors.reset_index(drop=True)
                
                # 过滤有效行 (因子不为 NaN)
                valid_factor_idx = ~factors.isna().any(axis=1)
                
                # 2. 构造标签所需的原始组件 (向量化计算)
                close = data['close'].values
                high = data['high'].values
                low = data['low'].values
                open_val = data['open'].values
                
                next_open = data['open'].shift(-1).values
                f_close = data['close'].shift(-forward_days).values
                
                # 滑动窗口计算期内极值
                if len(data) > forward_days:
                    _high_wins = sliding_window_view(high[1:], forward_days)
                    _low_wins  = sliding_window_view(low[1:],  forward_days)
                    
                    f_high_max = np.concatenate([np.max(_high_wins, axis=1), np.full(forward_days, np.nan)])[:len(data)]
                    f_low_min  = np.concatenate([np.min(_low_wins, axis=1), np.full(forward_days, np.nan)])[:len(data)]
                    f_high_idx = np.concatenate([np.argmax(_high_wins, axis=1), np.full(forward_days, np.nan)])[:len(data)]
                    f_low_idx  = np.concatenate([np.argmin(_low_wins, axis=1), np.full(forward_days, np.nan)])[:len(data)]
                else:
                    f_high_max = f_low_min = f_high_idx = f_low_idx = np.full(len(data), np.nan)

                # 计算 ATR
                atr_raw = talib.ATR(high, low, close, timeperiod=FactorConfig.ATR_PERIOD)
                atr_rel = atr_raw / (close + 1e-6)

                # 涨跌停阈值计算
                limit_thresholds = np.full(len(data), MARKET_LIMITS['main'], dtype=np.float32)
                if code.startswith(MARKET_PREFIXES['sz_gem']) or code.startswith(MARKET_PREFIXES['star']):
                    limit_thresholds[:] = MARKET_LIMITS['gem_star']
                elif code.startswith(MARKET_PREFIXES['bj']):
                    limit_thresholds[:] = MARKET_LIMITS['bj']
                
                if 'is_st' in data.columns:
                    is_main_board = ~(code.startswith(MARKET_PREFIXES['sz_gem']) or code.startswith(MARKET_PREFIXES['star']) or code.startswith(MARKET_PREFIXES['bj']))
                    if is_main_board:
                        limit_thresholds[data['is_st'] == 1] = MARKET_LIMITS['st']

                # 不可买入判定 (T+1日一字涨停或停牌，即预测生成后的执行日能否买入)
                epsilon = 0.002
                t_plus_1_preclose = data['close'].values # T日收盘价即为T+1日昨收
                t_plus_1_open = data['open'].shift(-1).values
                t_plus_1_high = data['high'].shift(-1).values
                t_plus_1_low = data['low'].shift(-1).values
                t_plus_1_vol = data['volume'].shift(-1).values
                
                is_one_word_up = (t_plus_1_open == t_plus_1_high) & (t_plus_1_open == t_plus_1_low) & \
                                 (t_plus_1_open >= t_plus_1_preclose * (1 + limit_thresholds) - epsilon)
                is_suspended = t_plus_1_vol == 0
                unbuyable = is_one_word_up | is_suspended

                # 收集组件
                comp_df = pd.DataFrame({
                    'date': data['date'],
                    'code': code,
                    'next_open': next_open,
                    'f_close': f_close,
                    'f_high_max': f_high_max,
                    'f_low_min': f_low_min,
                    'f_high_idx': f_high_idx,
                    'f_low_idx': f_low_idx,
                    'atr_raw': atr_raw,
                    'atr_rel': atr_rel,
                    'intraday_intensity': (data['high'].values - data['low'].values) / (atr_raw + 1e-6),
                    'volume_ratio': data['volume'].values / (data['volume'].rolling(20).mean().values + 1e-6),
                    'relative_intensity': ((data['high'].values - data['low'].values) / (atr_raw + 1e-6)) / \
                                          (pd.Series((data['high'].values - data['low'].values) / (atr_raw + 1e-6)).rolling(5).mean().values + 1e-6),
                    'limit_thresholds': limit_thresholds,
                    'unbuyable': unbuyable,
                    'is_st': data['is_st'] if 'is_st' in data.columns else 0,
                    'days_to_delist': data['days_to_delist'] if 'days_to_delist' in data.columns else -1
                })

                # 过滤掉标签组件中含 NaN 的行
                # 重要修复：必须包含 atr_raw 和 atr_rel，否则会产生 NaN 标签导致 LightGBM 分档异常（全入第0档）
                label_valid_idx = ~comp_df[['next_open', 'f_close', 'f_high_max', 'f_low_min', 'atr_raw', 'atr_rel']].isna().any(axis=1)
                final_valid_idx = valid_factor_idx & label_valid_idx
                
                # 时间窗口过滤
                if train_start_date:
                    final_valid_idx = final_valid_idx & (data['date'] >= train_start_date)
                if train_end_date:
                    final_valid_idx = final_valid_idx & (data['date'] <= train_end_date)

                if final_valid_idx.sum() > 0:
                    # 准备 X
                    # is_suspended: 状态位，在训练样本中几乎全为 0（停牌股已被 unbuyable 过滤），方差为 0 无区分度
                    drop_cols = ['date', 'is_st', 'is_suspended', 'code', 'fore_adjust_factor', 'back_adjust_factor', 'days_to_delist', 'amount', 'turnover_rate']
                    
                    
                    X_df = factors[final_valid_idx].drop(columns=[c for c in drop_cols if c in factors.columns], errors='ignore')
                    
                    return {
                        'X': X_df,
                        'components': comp_df[final_valid_idx]
                    }
            
            return None
        except Exception as e:
            if verbose:
                import traceback
                print(f"  Extraction error for {code}: {e}\n{traceback.format_exc()}")
            return None

    def _calculate_vectorized_labels(self, components: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行批量化、向量化的标签计算和截面排名。
        """
        eps = 1e-8
        
        # 1. 收益率标准化
        next_open = components['next_open'].values
        limits = components['limit_thresholds'].values
        
        f_returns_raw = (components['f_close'].values / (next_open + eps)) - 1
        f_high_max_raw = (components['f_high_max'].values / (next_open + eps)) - 1
        f_low_min_raw = (components['f_low_min'].values / (next_open + eps)) - 1
        
        # 标准化 (除以涨跌停阈值)
        f_returns_norm = f_returns_raw / limits
        f_high_max_norm = f_high_max_raw / limits
        f_low_min_norm = f_low_min_raw / limits
        
        # 2. 计算路径质量原始分
        # 注意：使用关键字参数传递 limits，避免与 intraday_intensity 的位置混淆
        raw_scores = self._compute_path_quality_score(
            f_returns_norm, f_low_min_norm, 
            components['f_high_idx'].values, components['f_low_idx'].values,
            components['atr_raw'].values, next_open, components['atr_rel'].values,
            f_high_max_norm,
            components['intraday_intensity'].values,
            components['volume_ratio'].values,
            components['relative_intensity'].values,
            limits=limits,
        )
        
        # 修复：确保 raw_scores 无 NaN/Inf，防止后续 rankdata 产生异常
        if np.isnan(raw_scores).any():
            raw_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. 风险样本惩罚 (ST & 退市)
        # 退市惩罚
        dtd = components['days_to_delist'].values
        delist_mask = (dtd >= 0) & (dtd <= getattr(TrainingConfig, 'DELIST_PENALTY_DAYS', 30))
        raw_scores[delist_mask] = getattr(TrainingConfig, 'DELIST_PENALTY_SCORE', -100)
        
        # ST 惩罚
        st_label_score = getattr(TrainingConfig, 'ST_LABEL_SCORE', None)
        if st_label_score is not None:
            st_mask = components['is_st'].values == 1
            raw_scores[st_mask] = np.minimum(raw_scores[st_mask], st_label_score)
            
        # 4. 截面排名 (Cross-sectional Ranking)
        # 优化：使用 numpy 矢量化计算替代 pandas groupby.rank，性能更高
        dates = components['date'].values
        sort_idx = np.argsort(dates)
        sorted_dates = dates[sort_idx]
        sorted_scores = raw_scores[sort_idx]
        
        unique_dates, group_start, group_counts = np.unique(sorted_dates, return_index=True, return_counts=True)
        y_final_sorted = np.zeros_like(sorted_scores, dtype=np.float32)
        
        for start, count in zip(group_start, group_counts):
            if count > 1:
                # 组内百分位排名
                y_final_sorted[start:start+count] = rankdata(sorted_scores[start:start+count], method='average') / (count + 1)
            else:
                y_final_sorted[start:start+count] = 0.5
        
        # 还原到原始顺序
        y_final = y_final_sorted[np.argsort(sort_idx)]
        
        # 重要性权重计算 (使用标准化最高收益率作为权重参考)
        # 使用 f_high_max_norm（已除以涨跌停阈值）而非原始收益率，
        # 消除板块间的权重偏差（创业板20%限制 vs 主板10%限制）
        w_sig = f_high_max_norm.astype(np.float32)
        
        return y_final, raw_scores, f_returns_raw, w_sig
                    


    def _validate_and_filter_stocks(self, stocks_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """
        验证缓存特征完整性，过滤掉缓存损坏的股票（触发重算），统计需要重算的数量。
        特征不匹配的股票仍会保留（后续会触发重算），只有无法恢复的损坏缓存才被标记删除。
        """
        print("\n验证缓存特征完整性...")

        # 从第一个可读的缓存文件推断期望的特征集
        model_features = None
        cache_files = [f for f in os.listdir(self.factors_cache_dir) if f.endswith('.parquet')]
        if cache_files:
            try:
                sample_factors = pd.read_parquet(os.path.join(self.factors_cache_dir, cache_files[0]))
                numeric_cols = sample_factors.select_dtypes(include=[np.number]).columns
                model_features = set(numeric_cols)
            except Exception:
                model_features = None

        if not model_features:
            print("  警告: 无法获取模型特征，跳过验证")
            return stocks_data, {'recomputed': 0, 'kept': len(stocks_data)}

        filtered_stocks = {}
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

        return filtered_stocks, {'recomputed': recomputed_count, 'kept': kept_count}
    
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
        all_codes = list(stocks_data.keys())
        import random
        random.seed(42)
        discovery_codes = random.sample(all_codes, min(len(all_codes), n_discovery))
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
            (X, y, returns, factor_names, dates, unbuyable, limit_groups, y_raw, is_st, w_sig)
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
            """提取单只股票组件，第一个输出日志"""
            verbose = (idx == 0) and cache_engineered_features
            return self._extract_stock_components(code, data, forward_days, 
                                                 cache_engineered_features, target_features, verbose,
                                                 train_start_date, train_end_date, include_fundamentals)
        
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(process_with_logging)(i, code, data)
            for i, (code, data) in enumerate(stock_list)
        )
        
        # 释放输入数据
        del stock_list
        gc.collect()

        # 1. 第一遍扫描：收集所有有效的 X 和 组件
        print("  - 合并特征与标签组件...")
        all_X = []
        all_comps = []
        for res in results:
            if res is not None:
                all_X.append(res['X'])
                all_comps.append(res['components'])
        
        if not all_X:
            raise ValueError("没有生成的有效样本")
            
        # 2. 批量合并 (Vectorized)
        X_df = pd.concat(all_X, ignore_index=True)
        comps_df = pd.concat(all_comps, ignore_index=True)
        
        # 释放中间列表
        del all_X, all_comps, results
        gc.collect()
        
        print(f"  - 原始样本量: {len(X_df)}, 特征数: {X_df.shape[1]}")
        
        # 3. 向量化计算标签
        print("  - 向量化生成标签 (标准化 + 截面排名)...")
        y_ranked, raw_scores, returns_raw, w_sig = self._calculate_vectorized_labels(comps_df)
        
        # 提取其他必要数组
        dates_arr = comps_df['date'].values
        unbuyable_arr = comps_df['unbuyable'].values
        limit_groups_arr = comps_df['limit_thresholds'].values
        is_st_arr = (comps_df['is_st'] == 1).values
        
        factor_names = X_df.columns.tolist()

        # 全局时间排序与转换为 numpy
        print("  - 全局时间排序与转换为 numpy...")
        sort_idx = np.argsort(dates_arr)
        
        dates_arr = dates_arr[sort_idx]
        y_final_arr = y_ranked[sort_idx]
        raw_scores_arr = raw_scores[sort_idx]  # 同步排序原始分
        returns_arr = returns_raw[sort_idx]
        unbuyable_arr = unbuyable_arr[sort_idx]
        limit_groups_arr = limit_groups_arr[sort_idx]
        is_st_arr = is_st_arr[sort_idx]
        w_sig_arr = w_sig[sort_idx]
        X_arr = X_df.values[sort_idx].astype(np.float32)
        
        # 释放 DataFrame
        del X_df, comps_df, sort_idx
        gc.collect()

        # 5. 不可买入样本处理（涨停/停牌）——在归一化之前剔除，确保排序不含涨停股
        penalty_count = np.sum(unbuyable_arr)
        if penalty_count > 0:
            handling = getattr(TrainingConfig, 'UNBUYABLE_HANDLING', 'remove')
            if handling == 'remove':
                print(f"  - 剔除不可买入样本: 正在剔除 {penalty_count} 个涨停/停牌样本")
                keep_mask = ~unbuyable_arr
                X_arr            = X_arr[keep_mask]
                y_final_arr      = y_final_arr[keep_mask]
                returns_arr      = returns_arr[keep_mask]
                dates_arr        = dates_arr[keep_mask]
                limit_groups_arr = limit_groups_arr[keep_mask]
                is_st_arr        = is_st_arr[keep_mask]
                w_sig_arr        = w_sig_arr[keep_mask]
            else:
                print(f"  - 施加不可买入惩罚: 将 {penalty_count} 个涨停/停牌标的的权重强制设为 0.2")
                w_sig_arr[unbuyable_arr] = 0.2

        # 日期分组信息（仅计算一次，供后续归一化和 group 划分共用）
        _, date_group_start, date_group_counts = np.unique(dates_arr, return_index=True, return_counts=True)

        # 6. 标签归一化已移至 train_models，在 train/val 分割后分别执行，避免标签泄漏。

        # 7. 因子分类审计报告（精确匹配各模块列名，非关键词启发式匹配）
        all_cols = factor_names
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
            'momentum_5d', 'momentum_10d', 'momentum_20d',
            'acceleration_5d', 'acceleration_10d',  # fix: was 'acceleration' (nonexistent)
            'consecutive_up_days', 'days_above_ma20',
            'price_percentile_60d', 'intraday_drawdown_avg_5d',
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
        return X_arr, y_final_arr, returns_arr, all_cols, dates_arr, unbuyable_arr, limit_groups_arr, raw_scores_arr, is_st_arr, w_sig_arr

    def _apply_cross_sectional_normalization(self, X_df: pd.DataFrame, dates: np.ndarray) -> pd.DataFrame:
        """
        对特征进行横截面归一化 (Cross-sectional Ranking)
        使用 numpy 矢量化计算替代 pandas groupby.rank，在大数据集上提速约 20-50 倍。
        """
        # 1. 提取特征名
        factor_names = X_df.columns.tolist()
        
        # 2. 转换为 numpy 数组进行计算 (float32 节省内存且满足精度)
        X_values = X_df.values.astype(np.float32)
        
        # 3. 性能优化关键：对日期进行排序并按块处理
        # 相比 groupby.rank()，手动切片 + scipy.stats.rankdata(axis=0) 可以极大减少 Python 循环开销
        sort_idx = np.argsort(dates)
        sorted_dates = dates[sort_idx]
        sorted_X = X_values[sort_idx]
        
        # 4. 调用已有的高效 inplace 版本执行归一化
        self._apply_cross_sectional_normalization_inplace(sorted_X, sorted_dates, factor_names)
        
        # 5. 还原原始顺序并写回 DataFrame
        rev_idx = np.argsort(sort_idx)
        # 注意：此处使用 iloc 覆盖值，保持 DataFrame 索引不变
        X_df.iloc[:, :] = sorted_X[rev_idx]
        
        return X_df
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    returns: np.ndarray,
                    factor_names: List[str],
                    dates: np.ndarray,
                    unbuyable_mask: np.ndarray = None,
                    limit_groups: np.ndarray = None,
                    model_types: List[str] = TrainingConfig.MODEL_TYPES,
                    path_scores: np.ndarray = None,
                    is_st_arr: np.ndarray = None,
                    w_sig_arr: np.ndarray = None) -> Dict:
        """
        训练多个模型
        """
        # 数据验证和清理
        print("\n数据验证...")
        
        # 确保数据类型正确 (copy=False 避免不必要的内存复制)
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        
        # 最后一次NaN/inf检查和替换
        # 对于排名后的特征(0-1)，0.5是中性值。确保 X 为 float32。
        # 注意：inf 检查必须独立于 NaN 检查，否则当 nan_count==0 但存在 inf 时
        # nan_to_num 不会被调用，导致 float32 溢出产生天文数字。
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0 or inf_count > 0:
            if nan_count > 0:
                print(f"  警告: 发现 {nan_count} 个 NaN 值，已替换为 0.5")
            if inf_count > 0:
                print(f"  警告: 发现 {inf_count} 个 Inf 值，已替换为边界值 (posinf→1.0, neginf→0.0)")
            X = np.nan_to_num(X, copy=False, nan=0.5, posinf=1.0, neginf=0.0)
        
        # 特征质量诊断：检查是否所有特征都是常数（无区分度）
        feature_stds = np.std(X, axis=0)
        zero_std_count = np.sum(feature_stds < 1e-6)
        if zero_std_count > 0:
            print(f"  警告: 发现 {zero_std_count} 个零方差特征（无区分度），建议检查特征工程")
            if zero_std_count > len(factor_names) * 0.5:
                print(f"  严重警告: 超过 50% 的特征无区分度，模型可能无法学习！")
        
        print(f"  数据验证完成: {X.shape[0]} 行, {X.shape[1]} 列")
        # 注意：此处统计为归一化前的原始特征值，仅用于诊断异常值是否已被清理
        print(f"  特征统计(归一化前): mean={X.mean():.4f}, std={X.std():.4f}, min={X.min():.4f}, max={X.max():.4f}")
        
        # 先进行时间序列划分 (增加 Embargo 阻隔期，防止数据泄漏)
        forward_days = getattr(TrainingConfig, 'FUTURE_DAYS', 7)
        raw_split_idx = int(len(dates) * TrainingConfig.TRAIN_TEST_SPLIT)
        split_date = dates[raw_split_idx]
        
        # 训练集：[0, split_idx)
        split_idx = np.searchsorted(dates, split_date, side='left')
        
        # 验证集：[val_start_idx, len(dates))
        # 阻隔期逻辑：由于样本标签包含未来 forward_days 的信息，
        # 验证集的特征必须在训练集标签所涉及的最晚日期之后。
        unique_dates = np.unique(dates)
        split_date_idx = np.searchsorted(unique_dates, split_date)
        # 验证集开始日期推迟 forward_days 个交易日
        val_start_date = unique_dates[min(split_date_idx + forward_days, len(unique_dates)-1)]
        val_start_idx = np.searchsorted(dates, val_start_date, side='left')
        
        print(f"  划分点 (Embargo): {split_date}, 阻隔期后起: {val_start_date}")
        print(f"  训练集: {split_idx} 样本, 验证集: {len(dates) - val_start_idx} 样本")
        print(f"  已剔除阻隔期重叠样本: {val_start_idx - split_idx} 个")
        
        # 重新定义切片范围
        X_train, X_val = X[:split_idx], X[val_start_idx:]
        y_train_full, y_val_full = y[:split_idx], y[val_start_idx:]
        dates_train, dates_val = dates[:split_idx], dates[val_start_idx:]
        returns_train, returns_val = returns[:split_idx], returns[val_start_idx:]
        is_st_train, is_st_val = is_st_arr[:split_idx], is_st_arr[val_start_idx:]
        w_sig_train, w_sig_val = w_sig_arr[:split_idx], w_sig_arr[val_start_idx:]
        
        # 训练集：正常横截面排名归一化
        print("\n  对训练样本进行横截面归一化...")
        skip_col_stats = self._apply_cross_sectional_normalization_inplace(
            X[:split_idx], dates[:split_idx], factor_names
        )
        # 持久化到实例，供 save_models 写入磁盘，推理时复用
        self.norm_stats = {
            'skip_col_stats': skip_col_stats,
            'factor_names': factor_names,
        }
        
        # 修复 Bug 1：使用 val_start_idx 而非 split_idx，排除阻隔期样本对验证集归一化的污染。
        # 阻隔期内的样本（split_idx ~ val_start_idx）标签含未来信息，不应参与任何处理。
        # 同时传入训练集的 skip_col_stats，确保情绪因子等全局列使用相同的缩放参数。
        print("  对验证样本进行归一化...")
        self._apply_cross_sectional_normalization_inplace(
            X[val_start_idx:], dates[val_start_idx:], factor_names,
            skip_col_stats=skip_col_stats
        )
        
        # 归一化后统计：正常情况下 mean≈0.5, std≈0.29, min≈0, max≈1
        print(f"  特征统计(归一化后): mean={X_train.mean():.4f}, std={X_train.std():.4f}, min={X_train.min():.4f}, max={X_train.max():.4f}")

        # 分组信息
        _, train_group = np.unique(dates_train, return_counts=True)
        _, val_group   = np.unique(dates_val, return_counts=True)

        # 修复标签泄漏：需要在 split 之后，对训练集和验证集分别重新做每日横截面排名归一化
        _label_source = path_scores if path_scores is not None else y
        # 强制清理：确保用于排名的标签源无 NaN
        if np.isnan(_label_source).any():
            _label_source = np.nan_to_num(_label_source, nan=0.5)
        y_train = np.empty(len(dates_train), dtype=np.float32)
        y_val   = np.empty(len(dates_val),   dtype=np.float32)
        
        # 离散档位标签：XGBoost 和 LightGBM 共用同一档位数（N_BINS）。
        # N_BINS 由 ModelConfig.get_n_bins() 统一获取（以 LIGHTGBM_PARAMS.label_gain 长度为准）。
        # 设计原则：档位数 × truncation_level ≈ 股票池大小，确保 lambdarank 有效 pair 密度。
        _n_bins = ModelConfig.get_n_bins()
        _mid_bin = _n_bins // 2  # 单样本日期的默认档位（中间档）
        y_train_discrete = np.empty(len(dates_train), dtype=np.int32)
        y_val_discrete = np.empty(len(dates_val), dtype=np.int32)
        
        for _dates_sub, _scores_sub, _y_sub, _y_discrete in [
            (dates_train, _label_source[:split_idx], y_train, y_train_discrete), 
            (dates_val, _label_source[val_start_idx:], y_val, y_val_discrete)
        ]:
            _, _d_starts, _d_counts = np.unique(_dates_sub, return_index=True, return_counts=True)
            for _ds, _dc in zip(_d_starts, _d_counts):
                _de = _ds + _dc
                if _dc > 1:
                    # 获取当前截面的原始分数
                    scores = _scores_sub[_ds:_de]
                    
                    # 1. 连续标签：排名归一化到 0-1 (所有样本参与排名)
                    ranks = rankdata(scores, method='average') / (_dc + 1)
                    _y_sub[_ds:_de] = ranks.astype(np.float32)       
                                 
                    # 2. 离散标签：头部优化的非均匀分档（解决 Top-1 精度为 0% 的核心痛点）
                    # 设计原则：
                    #   - 底部样本（前 50%）合并为 1 档，LambdaRank 不需要区分"差"和"更差"
                    #   - 中部样本（50%~80%）分 3 档，提供足够的负向对比信号
                    #   - 头部样本（80%~100%）分 6 档，给模型清晰的头部区分信号
                    # 诊断发现旧方案 [0.0,0.3,0.5,...] 导致档位 0 占 30%、档位 9 缺失，
                    # 调整为更均衡的分布，确保每档样本量差距不超过 10x
                    try:
                        if _n_bins == 10:
                            # 底部 35% → 档位 0（合并，控制负样本占比）
                            # 35%~50% → 档位 1
                            # 50%~63% → 档位 2
                            # 63%~74% → 档位 3
                            # 74%~83% → 档位 4
                            # 83%~90% → 档位 5
                            # 90%~95% → 档位 6
                            # 95%~98% → 档位 7
                            # 98%~99% → 档位 8
                            # 99%~100% → 档位 9
                            q_skewed = [0.0, 0.35, 0.50, 0.63, 0.74, 0.83, 0.90, 0.95, 0.98, 0.99, 1.0]
                        else:
                            # 动态生成：底部 35% 合并，其余均匀分配
                            # 需要 _n_bins+1 个边界点才能产生 _n_bins 个 bin
                            q_skewed = np.concatenate([
                                [0.0, 0.35],
                                np.linspace(0.35, 1.0, _n_bins - 1)
                            ])
                        bins = pd.qcut(scores, q=q_skewed, labels=False, duplicates='drop')
                        _y_discrete[_ds:_de] = bins.astype(np.int32)
                    except ValueError:
                        # 样本数太少或分数重复过多，回退到基于排名的简单分档
                        bins = np.clip((ranks * _n_bins).astype(np.int32), 0, _n_bins-1)
                        _y_discrete[_ds:_de] = bins
                else:
                    _y_discrete[_ds:_de] = _mid_bin
                    _y_sub[_ds:_de] = 0.5
        
        # 诊断输出：检查标签分布
        print(f"\n[标签分布诊断]")
        print(f"  训练集离散标签分布:")
        unique_train, counts_train = np.unique(y_train_discrete, return_counts=True)
        for bin_id, count in zip(unique_train, counts_train):
            print(f"    档位 {bin_id}: {count:>7} 样本 ({count/len(y_train_discrete)*100:>5.2f}%)")
        print(f"  验证集离散标签分布:")
        unique_val, counts_val = np.unique(y_val_discrete, return_counts=True)
        for bin_id, count in zip(unique_val, counts_val):
            print(f"    档位 {bin_id}: {count:>7} 样本 ({count/len(y_val_discrete)*100:>5.2f}%)")
        print(f"  原始分数统计 (训练集): min={_label_source[:split_idx].min():.4f}, max={_label_source[:split_idx].max():.4f}, std={_label_source[:split_idx].std():.4f}")
        # print(f"  原始分数统计 (验证集): min={_label_source[val_start_idx:].min():.4f}, max={_label_source[val_start_idx:].max():.4f}, std={_label_source[val_start_idx:].std():.4f}")

        
        print(f"  [标签] 已在 train/val 分割后分别重新做横截面排名归一化，消除标签泄漏")
        

        # 1. 初始化基础权重
        sample_weight_train = np.ones(len(y_train), dtype=np.float32)

        # 2. 叠加 ST 降权
        st_weight_factor = getattr(TrainingConfig, 'ST_WEIGHT_FACTOR', None)
        if is_st_train is not None and st_weight_factor is not None and st_weight_factor < 1.0:
            sample_weight_train *= np.where(is_st_train, st_weight_factor, 1.0).astype(np.float32)

        # 3. 叠加正交优化权重 (改为每日排名分档逻辑)
        if w_sig_arr is not None and getattr(TrainingConfig, 'USE_SAMPLE_WEIGHT', False):
            # 获取训练集和验证集的权重数据
            w_sig_train = w_sig_arr[:split_idx]
            w_sig_val = w_sig_arr[val_start_idx:]
            
            # 初始化处理后的权重数组
            w_sig_processed_train = np.empty_like(w_sig_train)
            w_sig_processed_val = np.empty_like(w_sig_val)
            
            # 使用相同的每日循环进行权重排名和离散化
            # 逻辑：权重 = 组内未来收益率的分位点 (0-1) -> 映射为离散等级 (如 1-5)
            # 这样保证了：
            # 1. 免疫极端大涨/大跌个股的过拟合 (只看排名)
            # 2. 与 lambdarank 的 top-K 优化目标天然匹配 (Top 样本权重更高)
            for _dates_sub, _w_raw, _w_processed in [
                (dates_train, w_sig_train, w_sig_processed_train),
                (dates_val, w_sig_val, w_sig_processed_val)
            ]:
                _, _d_starts, _d_counts = np.unique(_dates_sub, return_index=True, return_counts=True)
                for _ds, _dc in zip(_d_starts, _d_counts):
                    _de = _ds + _dc
                    if _dc > 1:
                        # 计算组内分位排名 (0 到 1)
                        # 注意：此处 w_raw 为原始收益率，值越大排名越靠前
                        w_ranks = rankdata(_w_raw[_ds:_de], method='average') / (_dc + 1)
                        
                        # 映射公式: 指数型头部权重 (Exponential Head Weighting)
                        # 让排名越靠前的样本权重呈指数级暴增，而非之前的线性增长
                        w_exp = getattr(TrainingConfig, 'WEIGHT_EXPONENT', 3.0)
                        
                        # 方案一：指数法 (e的几次幂)。如果 w_power=3，头部权重约等于尾部的 20 倍 (e^3 vs e^0)
                        _w_processed[_ds:_de] = np.exp(w_ranks * w_exp)
                    else:
                        # 单样本日期：使用中性权重，无法计算组内排名
                        _w_processed[_ds:_de] = 1.0
            
            # 更新训练集权重 (仅训练集需要权重)
            sample_weight_train *= w_sig_processed_train

        sample_weight_train = np.nan_to_num(sample_weight_train, nan=1.0, posinf=1.0, neginf=1.0)
        sample_weight_train /= (sample_weight_train.mean() + 1e-8)
        
        # 5. 特征选择：减少冗余和高度相关的特征 (New)
        selection_cache_file = os.path.join(TrainingConfig.SAVE_DIR, "selected_features.json")
        os.makedirs(TrainingConfig.SAVE_DIR, exist_ok=True)
        
        # 记录原始特征列表，用于同步过滤 X_val
        original_factor_names = list(factor_names)
        
        # 尝试从缓存读取特征选择结果
        loaded_from_cache = False
        if os.path.exists(selection_cache_file):
            try:
                import json
                with open(selection_cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    # 只有当原始特征集完全一致时，才复用缓存
                    if set(cached_data.get('original_features', [])) == set(original_factor_names):
                        factor_names = cached_data['selected_features']
                        print(f"\n[特征优化] 从缓存加载特征选择结果 (保留 {len(factor_names)} 个核心特征)")
                        loaded_from_cache = True
            except Exception as e:
                print(f"  读取特征选择缓存失败: {e}")

        if not loaded_from_cache:
            print(f"\n[特征优化] 正在进行特征选择 (原始特征数: {len(factor_names)})...")
            # 使用相关性过滤
            X_train, factor_names = self._select_features(
                X_train, factor_names
            )
            # 保存特征选择结果到缓存
            try:
                import json
                with open(selection_cache_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'original_features': original_factor_names,
                        'selected_features': factor_names,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"  保存特征选择结果失败: {e}")
        
        # 同步更新数据矩阵 (无论是否来自缓存，都要确保矩阵列数对齐)
        if len(factor_names) < len(original_factor_names):
            keep_indices = [original_factor_names.index(f) for f in factor_names]
            if loaded_from_cache:
                X_train = X_train[:, keep_indices] if not isinstance(X_train, pd.DataFrame) else X_train[factor_names]
            
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val[factor_names]
            else:
                X_val = X_val[:, keep_indices]
            
        if not loaded_from_cache:
            print(f"  特征优化完成: 保留 {len(factor_names)} 个核心特征")



        results = {}
        for model_type in model_types:
            print(f"\n训练 {model_type.upper()} 模型 (ranking)")
            try:
                model = MLFactorModel(model_type=model_type, task='ranking')

                # ── Early Stopping 验证集策略 ──────────────────────────────────
                # 直接使用外部纯净验证集（阻隔期后）做 early stopping，
                # 避免内部切分验证集时间分布与外部验证集不一致导致的过早停止。
                # 外部验证集时间更靠后，与真实推理场景一致，early stopping 更可靠。
                _, es_val_group = np.unique(dates_val, return_counts=True)

                train_result = model.train(
                    X_train, y_train_discrete,
                    validation_split=0.2,
                    use_time_series_split=True,
                    feature_names=factor_names,
                    sample_weight=sample_weight_train,
                    returns=returns_train,
                    split_idx=len(X_train),
                    X_val_external=X_val,
                    y_val_external=y_val_discrete,   # 长度已是 len(dates_val)，无需切片
                    dates_val_external=dates_val,
                    dates=dates_train,
                    group=train_group,
                    eval_group=es_val_group,
                )
                
                # 在外部纯净验证集上做最终评估（阻隔期后）
                val_eval = model._evaluate(X_val, y_val, "验证集(阻隔期后)", returns=returns_val, dates=dates_val)
                train_result['val_metrics'] = val_eval
                
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
                                                   skip_col_stats: Optional[dict] = None):
        """
        原位对特征矩阵进行横截面归一化（按日期分组），降低内存占用。
        
        参数:
            X: 特征矩阵（原位修改）
            dates: 日期数组
            factor_names: 特征名列表
            skip_col_stats: 跳过截面排名的列的预计算缩放统计量 {'median', 'iqr', 'valid_iqr'}。
                           若为 None，则从当前 X 切片自行计算（会导致 train/val 缩放不一致）。
                           应传入从训练集计算的统计量，以保证 train/val 分布一致。
        
        返回:
            skip_col_stats: 本次计算的缩放统计量（仅当 skip_col_stats=None 时有意义，
                           调用方应将其保存并传给验证集的调用）
        """
        # 跳过横截面排名归一化的特征集合
        # 原则：以下类型的特征不具备"今日全市场排名"的语义，归一化会破坏其信息：
        _skip_normalization = {
            # 全市场情绪因子（所有股票当天值相同）
            'up_ratio', 'strong_up_ratio', 'down_ratio', 'limit_up_ratio', 
            'limit_down_ratio', 'mean_return', 'total_volume', 'adv_vol_ratio', 
            'breadth_ma20', 'market_type',
            # 交易状态标志位
            'is_limit_up', 'is_suspended',
            # K线形态 0/1 信号
            'white_candle', 'black_candle', 'doji', 'hammer', 'hanging_man',
            'shooting_star', 'inverted_hammer', 'marubozu', 'spinning_top',
            'bullish_engulfing', 'bearish_engulfing', 'piercing_line',
            'dark_cloud_cover', 'morning_star', 'evening_star', 'harami',
            'three_white_soldiers', 'three_black_crows',
        }
        # One-Hot 行业列（前缀匹配）
        def _should_skip(col: str) -> bool:
            if col in _skip_normalization: return True
            # 行业、个股状态、退市天数等不参与横截面排名
            if col.startswith(('industry_', 'sector_', 'is_', 'days_to_')):
                if col.endswith('_encoded'): return False
                return True
            # 关键改进：全市场维度因子（大盘指标、情绪指标）不参与横截面排名
            # 这些指标在同一天对所有股票是相同的，排名后会全部变成 0.5，导致模型丢失大盘环境信息
            if col.startswith(('mkt_', 'market_', 'index_', 'sentiment_', 'vix_')):
                return True
            return False

        rank_cols_mask = np.array([not _should_skip(col) for col in factor_names])
        rank_cols_idx = np.where(rank_cols_mask)[0]
        skip_cols_idx = np.where(~rank_cols_mask)[0]

        if len(rank_cols_idx) == 0:
            return None

        # ── 跳过截面排名的列：按语义分三类处理 ──────────────────────────────
        # skip_cols 内部性质不同，统一用 robust+sigmoid 会破坏部分特征的语义：
        #   A. 0/1 二值特征（状态位、K线形态）：直接保留原值，不做任何缩放
        #      理由：median 通常为 0，IQR≈0，valid_iqr=False → 原代码会把所有值置 0，
        #            信号完全消失；且 0/1 本身已在 [0,1] 范围内，无需缩放。
        #   B. 全市场宏观/情绪因子（所有股票当天值相同）：robust+sigmoid
        #      理由：量级差异极大（total_volume 千亿级），需要缩放；
        #            使用训练集统计量保证 train/val 分布一致。
        #   C. 行业编码、退市天数等其他跳过列：同 B，robust+sigmoid
        _binary_skip_names = {
            # 交易状态标志位
            'is_limit_up', 'is_suspended', 'is_st',
            # K线形态 0/1 信号
            'white_candle', 'black_candle', 'doji', 'hammer', 'hanging_man',
            'shooting_star', 'inverted_hammer', 'marubozu', 'spinning_top',
            'bullish_engulfing', 'bearish_engulfing', 'piercing_line',
            'dark_cloud_cover', 'morning_star', 'evening_star', 'harami',
            'three_white_soldiers', 'three_black_crows',
        }
        def _is_binary_skip(col: str) -> bool:
            if col in _binary_skip_names:
                return True
            # is_* 前缀（非 _encoded 结尾）均视为二值
            if col.startswith('is_') and not col.endswith('_encoded'):
                return True
            return False

        if len(skip_cols_idx) > 0:
            skip_names = [factor_names[i] for i in skip_cols_idx]
            binary_mask  = np.array([_is_binary_skip(n) for n in skip_names])  # A 类
            robust_mask  = ~binary_mask                                          # B/C 类

            robust_local_idx = np.where(robust_mask)[0]   # 在 skip_cols_idx 内的相对下标
            robust_global_idx = skip_cols_idx[robust_local_idx]  # 在 X 中的绝对列下标

            # ── A 类：0/1 二值特征，直接保留，不做任何变换 ──────────────
            # 无需操作，原始值已是 0 或 1，语义完整。

            # ── B/C 类：连续型跳过列，robust scaler + sigmoid → [0,1] ──
            if robust_local_idx.size > 0:
                skip_data = X[:, robust_global_idx].astype(np.float64)
                if skip_col_stats is None:
                    # 从当前切片计算（训练集调用时）
                    p25       = np.nanpercentile(skip_data, 25, axis=0)
                    p75       = np.nanpercentile(skip_data, 75, axis=0)
                    median    = np.nanpercentile(skip_data, 50, axis=0)
                    iqr       = p75 - p25
                    valid_iqr = iqr > 1e-8
                    skip_col_stats = {
                        'median': median, 'iqr': iqr, 'valid_iqr': valid_iqr,
                        'robust_global_idx': robust_global_idx,  # 保存列索引供推理复用
                    }
                else:
                    median    = skip_col_stats['median']
                    iqr       = skip_col_stats['iqr']
                    valid_iqr = skip_col_stats['valid_iqr']
                scaled = np.where(
                    valid_iqr,
                    (skip_data - median) / np.where(valid_iqr, iqr, 1.0),
                    0.0
                )
                # sigmoid 压缩到 [0, 1]，平滑处理极端值
                scaled_01 = 1.0 / (1.0 + np.exp(-scaled.clip(-10, 10)))
                X[:, robust_global_idx] = scaled_01.astype(np.float32)
            elif skip_col_stats is None:
                # 全部是二值列，无需 robust 统计量，但仍需返回非 None 以区分"已计算"
                skip_col_stats = {
                    'median': np.array([]), 'iqr': np.array([]),
                    'valid_iqr': np.array([], dtype=bool),
                    'robust_global_idx': np.array([], dtype=int),
                }

        unique_dates, group_start, group_counts = np.unique(
            dates, return_index=True, return_counts=True
        )

        for start, count in zip(group_start, group_counts):
            if count <= 1:
                X[start:start+count, rank_cols_idx] = 0.5
                continue

            day_data = X[start:start+count, rank_cols_idx]
            day_ranks = rankdata(day_data, method='average', axis=0) / (count + 1)
            X[start:start+count, rank_cols_idx] = day_ranks.astype(np.float32)

        gc.collect()
        return skip_col_stats

    def _select_features(self, X: np.ndarray, feature_names: List[str], 
                       corr_threshold: float = 0.85,
                       method: str = 'correlation') -> Tuple[np.ndarray, List[str]]:
        """
        特征选择：过滤高度相关的特征
        """
        if method == 'none' or len(feature_names) <= 50:
            return X, feature_names
            
        df_temp = pd.DataFrame(X, columns=feature_names)
        
        # 1. 简单相关性过滤
        corr_matrix = df_temp.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        
        if to_drop:
            print(f"  - 发现 {len(to_drop)} 个高度相关特征 (corr > {corr_threshold})，已剔除")
            df_temp = df_temp.drop(columns=to_drop)
            new_feature_names = df_temp.columns.tolist()
            return df_temp.values, new_feature_names
            
        return X, feature_names
    
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
                '任务': '排序',
                'Rank IC': f"{val_metrics.get('rank_ic', 0.0):.4f}",
                'Top-1精度': f"{val_metrics.get('top1_precision', 0.0):.2%}",
                'Top-5精度': f"{val_metrics.get('top5_precision', 0.0):.2%}",
            }
            
            # 补充任务特有指标
            row['辅助指标'] = f"IC_Std: {val_metrics.get('rank_ic_std', 0.0):.4f} | AUC: {val_metrics.get('auc', 0.0):.4f}"
            
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

        forward_days = getattr(TrainingConfig, 'FUTURE_DAYS', 7)
        data_volume  = f"{years}y_{stocks}s"
        timestamp    = datetime.now().strftime('%m%d_%H%M')

        # 归档目录名示例：train_hybrid_7d_6y_6000s_0429_1530
        archive_name = f"train_{self.task}_{forward_days}d_{data_volume}_{timestamp}"
        archive_dir  = os.path.join(save_dir, archive_name)
        os.makedirs(archive_dir, exist_ok=True)
        
        for model_type, model in self.models.items():
            filepath = os.path.join(archive_dir, f'{model_type}_factor_model.pkl')
            model.save_model(filepath)

        # 保存归一化统计量（推理时复用，保证 train/inference 分布一致）
        norm_stats = getattr(self, 'norm_stats', None)
        if norm_stats is not None:
            import pickle as _pickle
            norm_path = os.path.join(archive_dir, 'norm_stats.pkl')
            with open(norm_path, 'wb') as f:
                _pickle.dump(norm_stats, f)
            print(f"  [OK] 归一化统计量已保存: norm_stats.pkl")
            
        # 8. 同时更新一个 "latest" 目录，方便自动调用
        latest_dir = os.path.join(save_dir, 'latest')
        import shutil
        if os.path.exists(latest_dir):
            try: shutil.rmtree(latest_dir)
            except: pass
        try:
            shutil.copytree(archive_dir, latest_dir)
            print(f"  [OK] 已同步至最新目录: {latest_dir}")
        except Exception as e:
            print(f"  [Error] 同步最新目录失败: {e}")
            
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
            'momentum_5d', 'momentum_10d', 'momentum_20d',
            'acceleration_5d', 'acceleration_10d',  # fix: was 'acceleration' (nonexistent)
            'consecutive_up_days', 'days_above_ma20',
            'price_percentile_60d', 'intraday_drawdown_avg_5d',
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
