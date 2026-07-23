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

# 启用 numexpr 加速 pandas 数值计算（3-5倍提速）
import pandas as pd
try:
    import numexpr
    pd.set_option('compute.use_numexpr', True)
    pd.set_option('compute.use_bottleneck', True)
except ImportError:
    pass


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

FACTOR_CACHE_PREPROCESSING_VERSION = "2"


def _read_factor_cache_version(cache_file: str) -> Optional[str]:
    try:
        import pyarrow.parquet as pq
        metadata = pq.read_metadata(cache_file).metadata or {}
        value = metadata.get(b'factor_preprocessing_version')
        return value.decode('utf-8') if value else None
    except Exception:
        return None


def _write_factor_cache(frame: pd.DataFrame, cache_file: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b'factor_preprocessing_version'] = (
        FACTOR_CACHE_PREPROCESSING_VERSION.encode('utf-8')
    )
    pq.write_table(table.replace_schema_metadata(metadata), cache_file)

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
        trainer.task = TrainingConfig.TASK
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
        if _read_factor_cache_version(cache_file) != FACTOR_CACHE_PREPROCESSING_VERSION:
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


def _fast_rankdata_1d(a):
    """
    使用 numpy argsort 实现的超快速 rankdata (等价于 scipy.stats.rankdata(..., method='average'))。
    在没有重复值（Ties）时快数倍，在有重复值时也极其高效。
    """
    n = len(a)
    if n <= 1:
        return np.array([1.0], dtype=np.float32)
    
    sorter = np.argsort(a)
    
    # 检查是否有重复值
    a_sorted = a[sorter]
    obs = np.empty(n, dtype=bool)
    obs[0] = True
    obs[1:] = a_sorted[1:] != a_sorted[:-1]
    
    if np.all(obs):
        # 无重复值：直接使用 argsort 的逆作为排名
        inv = np.empty(n, dtype=np.float32)
        inv[sorter] = np.arange(1.0, n + 1.0, dtype=np.float32)
        return inv
        
    # 有重复值：计算平均排名
    obs_idx = np.where(obs)[0]
    repeats = np.diff(np.append(obs_idx, n))
    
    avg_ranks = obs_idx + (repeats + 1) / 2.0
    sorted_ranks = np.repeat(avg_ranks, repeats)
    
    ranks = np.empty(n, dtype=np.float32)
    ranks[sorter] = sorted_ranks
    return ranks


def _rank_finite_to_unit_interval(a, missing_value: float = 0.5):
    """仅对有限值排名，缺失值在排名完成后置为中性值。

    不能在横截面排名前把 NaN 替换成 0.5：原始因子的量纲并不一定在
    [0, 1]，提前填充会让“缺失”获得一个虚假的可排序数值，并泄露数据覆盖。
    """
    values = np.asarray(a)
    result = np.full(len(values), missing_value, dtype=np.float32)
    valid = np.isfinite(values)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return result
    if valid_count == 1:
        result[valid] = 0.5
        return result
    result[valid] = _fast_rankdata_1d(values[valid]) / (valid_count + 1)
    return result


def _normalize_chunk_worker(X, rank_cols_idx, chunk_start_row, group_starts, group_counts):
    """
    线程池中的子工作任务：原地归一化特征子矩阵中的一组日期。
    """
    F = len(rank_cols_idx)
    
    for start, count in zip(group_starts, group_counts):
        row_start = chunk_start_row + start
        row_end = row_start + count
        if count <= 1:
            X[row_start:row_end, rank_cols_idx] = 0.5
            continue
            
        for j in range(F):
            col = rank_cols_idx[j]
            col_data = X[row_start:row_end, col]
            X[row_start:row_end, col] = _rank_finite_to_unit_interval(col_data)
            
    return None


def _train_objective_worker(payload):
    """多进程 worker：在独立进程中训练单个目标模型。

    通过限制每个进程内的 LightGBM/XGBoost 线程数，避免
    "多进程 × n_jobs=-1" 的 CPU oversubscription。
    大特征矩阵以只读 mmap 文件共享，避免进程间重复序列化/拷贝。
    """
    import os as _os
    import numpy as _np
    from config.factor_config import ModelConfig

    db_path = payload['db_path']
    y = _np.asarray(payload['y'], dtype=_np.float32)
    dates = _np.asarray(payload['dates'])
    names = payload['names']
    model_type = payload['model_type']
    n_threads = max(1, int(payload.get('n_threads', 1)))

    # 限制本进程内各数值库的线程数
    for _ev in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        _os.environ[_ev] = str(n_threads)
    # 直接覆盖 ModelConfig 的 n_jobs（仅影响本 worker 进程）
    try:
        ModelConfig.LIGHTGBM_PARAMS = dict(ModelConfig.LIGHTGBM_PARAMS)
        ModelConfig.LIGHTGBM_PARAMS['n_jobs'] = n_threads
    except Exception:
        pass
    try:
        ModelConfig.XGBOOST_PARAMS = dict(ModelConfig.XGBOOST_PARAMS)
        ModelConfig.XGBOOST_PARAMS['n_jobs'] = n_threads
    except Exception:
        pass

    X = _np.load(payload['x_path'], mmap_mode='r')
    trainer = MLModelTrainer(db_path)
    res = trainer.train_models(
        _np.asarray(X), y, y, names, dates,
        model_types=[model_type], task='ranking', apply_feature_selection=False,
        skip_normalization=True,
    )
    model = trainer.models.get(model_type)
    if model is None:
        raise RuntimeError("目标训练未产出模型")
    return (model, res.get(model_type))


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
        self.task = TrainingConfig.TASK
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
        has_meta_stock_basic = False
        if os.path.exists(meta_db):
            conn.execute(f"ATTACH DATABASE '{meta_db}' AS meta")
            has_meta_stock_basic = conn.execute(
                "SELECT 1 FROM meta.sqlite_master WHERE type='table' AND name='stock_basic'"
            ).fetchone() is not None
        
        conn.row_factory = sqlite3.Row
        
        # 预加载退市日期映射 {code: outDate_str or None}
        delist_map: Dict[str, Optional[str]] = {}
        try:
            if not has_meta_stock_basic:
                print("  [INFO] 未配置 stock_basic 元数据，退市日期特征使用默认值")
            else:
                placeholders_all = ','.join(['?' for _ in stock_codes])
                delist_df = pd.read_sql_query(
                    f"SELECT code, outDate FROM meta.stock_basic WHERE code IN ({placeholders_all})",
                    conn, params=stock_codes
                )
                for _, row in delist_df.iterrows():
                    out = row['outDate']
                    delist_map[row['code']] = out if (out and str(out).strip() not in ('', 'None', 'nan')) else None
        except Exception as e:
            print(f"  [WARN] 读取退市日期失败，退市特征将不可用: {e}")
        
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

            # 按股票分组并执行动态复权。groupby 避免为每只股票重复构造整批布尔掩码。
            for code, stock_df in df.groupby('code', sort=False):
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

                float_cols = stock_df.select_dtypes(include=['float64']).columns
                if len(float_cols) > 0:
                    stock_df[float_cols] = stock_df[float_cols].astype(np.float32, copy=False)
                int_cols = [c for c in stock_df.select_dtypes(include=['int64']).columns if c not in ('code', 'date')]
                for col in int_cols:
                    stock_df[col] = pd.to_numeric(stock_df[col], downcast='integer')
                
                stocks_data[code] = stock_df

            del df
            gc.collect()
            
            # 进度提示
            pbar.update(len(batch_codes))
        
        pbar.close()
        conn.close()
        
        delist_count = sum(1 for v in delist_map.values() if v is not None)
        print(f"成功加载并复权 {len(stocks_data)} 只股票的数据 (其中 {delist_count} 只含退市日期)")
        return stocks_data

    def load_label_data(self, stock_codes: List[str],
                        start_date: str, end_date: str,
                        batch_size: int = 500) -> Dict[str, pd.DataFrame]:
        """
        仅加载生成训练标签所需的行情列。

        因子从 parquet 缓存读取时使用该路径，避免为已缓存股票加载基本面、估值等训练特征源数据。
        """
        print(f"正在加载 {len(stock_codes)} 只股票的标签行情数据...")

        stocks_data = {}
        conn = sqlite3.connect(self.db_path)
        db_dir = os.path.dirname(self.db_path)
        meta_db = os.path.join(db_dir, 'stock_meta.db')
        has_meta_stock_basic = False
        if os.path.exists(meta_db):
            conn.execute(f"ATTACH DATABASE '{meta_db}' AS meta")
            has_meta_stock_basic = conn.execute(
                "SELECT 1 FROM meta.sqlite_master WHERE type='table' AND name='stock_basic'"
            ).fetchone() is not None

        delist_map: Dict[str, Optional[str]] = {}
        try:
            if not has_meta_stock_basic:
                print("  [INFO] 未配置 stock_basic 元数据，退市日期特征使用默认值")
            else:
                placeholders_all = ','.join(['?' for _ in stock_codes])
                delist_df = pd.read_sql_query(
                    f"SELECT code, outDate FROM meta.stock_basic WHERE code IN ({placeholders_all})",
                    conn, params=stock_codes
                )
                for _, row in delist_df.iterrows():
                    out = row['outDate']
                    delist_map[row['code']] = out if (out and str(out).strip() not in ('', 'None', 'nan')) else None
        except Exception as e:
            print(f"  [WARN] 读取退市日期失败，退市特征将不可用: {e}")

        pbar = tqdm(total=len(stock_codes), desc="标签行情加载")
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i+batch_size]
            placeholders = ','.join(['?' for _ in batch_codes])
            query = f'''
                SELECT k.code, k.date, k.open, k.high, k.low, k.close, k.volume, k.amount,
                       k.is_st, a.fore_adjust_factor
                FROM daily_data k
                LEFT JOIN adjust_factor a ON k.code = a.code AND k.date = a.date
                WHERE k.code IN ({placeholders}) AND k.date >= ? AND k.date <= ?
                ORDER BY k.code, k.date ASC
            '''
            params = list(batch_codes) + [str(start_date), str(end_date)]
            df = pd.read_sql_query(query, conn, params=tuple(params))
            if df.empty:
                pbar.update(len(batch_codes))
                continue

            for code, stock_df in df.groupby('code', sort=False):
                stock_df = stock_df.sort_values('date').reset_index(drop=True)

                if 'fore_adjust_factor' in stock_df.columns:
                    valid_adj = stock_df['fore_adjust_factor'].dropna()
                    if not valid_adj.empty:
                        base_factor = float(valid_adj.iloc[-1])
                        if base_factor != 0:
                            ratio = stock_df['fore_adjust_factor'].bfill().ffill().fillna(1.0) / base_factor
                            for col in ['open', 'high', 'low', 'close']:
                                stock_df[col] = stock_df[col] * ratio
                    stock_df = stock_df.drop(columns=['fore_adjust_factor'])

                if len(stock_df) < 100:
                    continue

                out_date_str = delist_map.get(code)
                if out_date_str:
                    try:
                        out_dt = pd.Timestamp(out_date_str)
                        dates_ts = pd.to_datetime(stock_df['date'])
                        stock_df['days_to_delist'] = (out_dt - dates_ts).dt.days.clip(lower=0).astype(np.float32)
                    except Exception:
                        stock_df['days_to_delist'] = np.float32(-1)
                else:
                    stock_df['days_to_delist'] = np.float32(-1)

                float_cols = stock_df.select_dtypes(include=['float64']).columns
                if len(float_cols) > 0:
                    stock_df[float_cols] = stock_df[float_cols].astype(np.float32, copy=False)
                int_cols = [c for c in stock_df.select_dtypes(include=['int64']).columns if c not in ('code', 'date')]
                for col in int_cols:
                    stock_df[col] = pd.to_numeric(stock_df[col], downcast='integer')

                stocks_data[code] = stock_df

            del df
            gc.collect()
            pbar.update(len(batch_codes))

        pbar.close()
        conn.close()
        print(f"成功加载标签行情 {len(stocks_data)} 只股票")
        return stocks_data

    def build_multiobjective_labels(
        self,
        stocks_data: Dict[str, pd.DataFrame],
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """构造多目标标签宽表，不改变现有单目标训练 API。"""
        from core.factors.multi_objective_labels import (
            MultiObjectiveLabelBuilder,
            cross_sectional_rank_targets,
        )
        builder = MultiObjectiveLabelBuilder(
            return_horizons=getattr(
                TrainingConfig, 'MULTI_OBJECTIVE_RETURN_HORIZONS', (5, 20, 60)
            ),
            risk_horizon=getattr(TrainingConfig, 'MULTI_OBJECTIVE_RISK_HORIZON', 20),
            orthogonal_legs=getattr(TrainingConfig, 'MULTI_OBJECTIVE_ORTHOGONAL_LEGS', True),
            use_matching_risk_for_sharpe=getattr(
                TrainingConfig, 'MULTI_OBJECTIVE_MATCHING_RISK_SHARPE', True
            ),
        )
        orthogonalize = getattr(TrainingConfig, 'MULTI_OBJECTIVE_ORTHOGONALIZE', False)
        labels = builder.build_universe(
            stocks_data, start_date, end_date, orthogonalize=orthogonalize
        )
        if labels.empty:
            return labels
        target_cols = list(getattr(TrainingConfig, 'MULTI_OBJECTIVE_WEIGHTS', {}).keys())
        target_cols = [col for col in target_cols if col in labels.columns]
        if orthogonalize:
            # 正交化后使用 orth_* 列作为训练目标（需与权重键对应）。
            orth_cols = [f"orth_{c}" for c in target_cols]
            if all(c in labels.columns for c in orth_cols):
                target_cols = orth_cols
        return cross_sectional_rank_targets(
            labels,
            target_cols,
            risk_cols={'y_downvol_20d', 'y_illiq_20d'},
        )
    
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
        os.makedirs(self.factors_cache_dir, exist_ok=True)
        cache_file = os.path.join(self.factors_cache_dir, f'{code}_factors.parquet')
        
        # 确保 data 中的 date 列为字符串（方便比较）
        if 'date' in data.columns and not pd.api.types.is_string_dtype(data['date']):
            data = data.copy()
            data['date'] = data['date'].astype(str)
        
        # ── 1. 尝试从缓存加载 ──────────────────────────────────────────────
        cached_factors = None
        if os.path.exists(cache_file):
            try:
                cache_version = _read_factor_cache_version(cache_file)
                if cache_version != FACTOR_CACHE_PREPROCESSING_VERSION:
                    if verbose:
                        print(
                            f"  {code}: 缓存预处理版本 {cache_version or 'legacy'} "
                            f"!= {FACTOR_CACHE_PREPROCESSING_VERSION}，触发全量重算"
                        )
                else:
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
            _write_factor_cache(all_factors, cache_file)
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
                                 n_jobs: int = None,  # 默认使用配置值
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
        # 默认使用配置中的 N_JOBS_FACTOR_CALC，-1 表示使用所有核心
        if n_jobs is None:
            n_jobs = getattr(TrainingConfig, 'N_JOBS_FACTOR_CALC', -1)
        effective_jobs = min(n_jobs if n_jobs > 0 else multiprocessing.cpu_count(), 
                             multiprocessing.cpu_count(), len(to_update))
        print(f"  正在多进程并行更新 {len(to_update)} 只股票的缓存 (进程数={effective_jobs})...")
        
        worker_args = [
            (code, data, self.db_path, self.factors_cache_dir, target_features, include_fundamentals)
            for code, data in to_update.items()
        ]
        
        start_time = time()
        success = 0
        failed = 0
        is_atty = sys.stdout.isatty()
        
        executor = ProcessPoolExecutor(max_workers=effective_jobs)
        try:
            futures = {executor.submit(_cache_worker, arg): arg[0] for arg in worker_args}
            
            with tqdm(total=len(futures), desc="更新因子缓存", disable=not is_atty) as pbar:
                for future in as_completed(futures):
                    try:
                        code, ok, err = future.result()
                        if ok:
                            success += 1
                        else:
                            tqdm.write(f"  [FAIL] {code} 缓存更新失败: {err}")
                            failed += 1
                    except Exception as e:
                        code = futures[future]
                        tqdm.write(f"  [FAIL] {code} 进程异常: {e}")
                        failed += 1
                    pbar.set_postfix({"成功": success, "失败": failed})
                    pbar.update(1)
        finally:
            # Windows 上 atexit 会通过管理线程 t.join() 等待所有子进程退出，
            # 即使 shutdown(wait=False) 也无法绕过。
            # 唯一可靠的方式是直接 terminate() 所有子进程，再 shutdown。
            try:
                for pid, proc in executor._processes.items():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
                    
        elapsed = time() - start_time
        print(f"[OK] 缓存同步完成: 成功 {success}, 失败 {failed} | 已跳过 {skipped} | 耗时 {elapsed:.1f}s")
    



    def _compute_path_quality_score(self, f_returns_norm, f_low_min_norm, f_high_idx, f_low_idx, f_high_max_norm, limits=0.1):

        upside  = np.where(f_high_max_norm > 0, f_high_max_norm, 0)
        downside = np.where(f_low_min_norm  < 0, f_low_min_norm,  0)

        w_upside   = getattr(TrainingConfig, 'UPSIDE_WEIGHT',        2.0)
        w_downside = getattr(TrainingConfig, 'DOWNSIDE_WEIGHT',       1.0)
        w_final    = getattr(TrainingConfig, 'FINAL_RETURN_WEIGHT',   3.0)

        base_score = (f_returns_norm * w_final) + (upside * w_upside) + (downside * w_downside)

        high_idx = np.asarray(f_high_idx, dtype=np.float64)
        low_idx  = np.asarray(f_low_idx,  dtype=np.float64)
        path_bonus = getattr(TrainingConfig, 'PATH_BONUS',   0.15)
        path_penalty = getattr(TrainingConfig, 'PATH_PENALTY', 0.10)
        path_mult = np.where(
            np.isnan(high_idx) | np.isnan(low_idx),
            1.0,
            np.where(high_idx < low_idx, 1.0 + path_bonus,
            np.where(low_idx  < high_idx, 1.0 - path_penalty,
            1.0))
        )
        final_score = base_score * path_mult
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
                atr_raw = talib.ATR(
                    high.astype(np.float64, copy=False),
                    low.astype(np.float64, copy=False),
                    close.astype(np.float64, copy=False),
                    timeperiod=FactorConfig.ATR_PERIOD
                )
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

    def _extract_stock_components_from_cache(self, code: str, data: pd.DataFrame,
                                             forward_days: int,
                                             target_features: Optional[List[str]] = None,
                                             train_start_date: str = None,
                                             train_end_date: str = None,
                                             verbose: bool = False) -> Optional[dict]:
        """从因子 parquet 缓存读取 X，仅用行情数据生成标签组件。"""
        try:
            cache_file = os.path.join(self.factors_cache_dir, f'{code}_factors.parquet')
            if not os.path.exists(cache_file):
                return None

            read_cols = None
            if target_features:
                try:
                    import pyarrow.parquet as pq
                    schema_names = set(pq.read_schema(cache_file).names)
                    read_cols = ['date'] + [c for c in target_features if c in schema_names]
                except Exception:
                    read_cols = None

            factors = pd.read_parquet(cache_file, columns=read_cols)
            if factors is None or factors.empty or 'date' not in factors.columns:
                return None
            if not pd.api.types.is_string_dtype(factors['date']):
                factors['date'] = factors['date'].astype(str)

            data_dates = data[['date']].copy()
            if not pd.api.types.is_string_dtype(data_dates['date']):
                data_dates['date'] = data_dates['date'].astype(str)
            factors = pd.merge(data_dates, factors, on='date', how='left').reset_index(drop=True)

            factor_value_cols = [c for c in factors.columns if c != 'date']
            if target_features:
                missing_features = [c for c in target_features if c not in factors.columns]
                if missing_features:
                    # 保留“整只股票没有该字段”的真实缺失状态，供训练段覆盖率
                    # 筛选使用；中性值只能在横截面归一化完成之后填充。
                    missing_df = pd.DataFrame(
                        np.nan, index=factors.index, columns=missing_features,
                        dtype=np.float32,
                    )
                    factors = pd.concat([factors, missing_df], axis=1)
                factor_value_cols = [c for c in target_features if c in factors.columns]
            if not factor_value_cols:
                return None
            factors[factor_value_cols] = factors[factor_value_cols].apply(pd.to_numeric, errors='coerce')
            valid_factor_idx = ~factors[factor_value_cols].isna().all(axis=1)

            close = data['close'].values
            high = data['high'].values
            low = data['low'].values

            next_open = data['open'].shift(-1).values
            f_close = data['close'].shift(-forward_days).values

            if len(data) > forward_days:
                _high_wins = sliding_window_view(high[1:], forward_days)
                _low_wins  = sliding_window_view(low[1:],  forward_days)
                f_high_max = np.concatenate([np.max(_high_wins, axis=1), np.full(forward_days, np.nan)])[:len(data)]
                f_low_min  = np.concatenate([np.min(_low_wins, axis=1), np.full(forward_days, np.nan)])[:len(data)]
                f_high_idx = np.concatenate([np.argmax(_high_wins, axis=1), np.full(forward_days, np.nan)])[:len(data)]
                f_low_idx  = np.concatenate([np.argmin(_low_wins, axis=1), np.full(forward_days, np.nan)])[:len(data)]
            else:
                f_high_max = f_low_min = f_high_idx = f_low_idx = np.full(len(data), np.nan)

            atr_raw = talib.ATR(
                high.astype(np.float64, copy=False),
                low.astype(np.float64, copy=False),
                close.astype(np.float64, copy=False),
                timeperiod=FactorConfig.ATR_PERIOD
            )
            atr_rel = atr_raw / (close + 1e-6)

            limit_thresholds = np.full(len(data), MARKET_LIMITS['main'], dtype=np.float32)
            if code.startswith(MARKET_PREFIXES['sz_gem']) or code.startswith(MARKET_PREFIXES['star']):
                limit_thresholds[:] = MARKET_LIMITS['gem_star']
            elif code.startswith(MARKET_PREFIXES['bj']):
                limit_thresholds[:] = MARKET_LIMITS['bj']

            if 'is_st' in data.columns:
                is_main_board = ~(code.startswith(MARKET_PREFIXES['sz_gem']) or code.startswith(MARKET_PREFIXES['star']) or code.startswith(MARKET_PREFIXES['bj']))
                if is_main_board:
                    limit_thresholds[data['is_st'] == 1] = MARKET_LIMITS['st']

            epsilon = 0.002
            t_plus_1_preclose = data['close'].values
            t_plus_1_open = data['open'].shift(-1).values
            t_plus_1_high = data['high'].shift(-1).values
            t_plus_1_low = data['low'].shift(-1).values
            t_plus_1_vol = data['volume'].shift(-1).values

            is_one_word_up = (t_plus_1_open == t_plus_1_high) & (t_plus_1_open == t_plus_1_low) & \
                             (t_plus_1_open >= t_plus_1_preclose * (1 + limit_thresholds) - epsilon)
            is_suspended = t_plus_1_vol == 0
            unbuyable = is_one_word_up | is_suspended

            price_range_rel_atr = (data['high'].values - data['low'].values) / (atr_raw + 1e-6)
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
                'intraday_intensity': price_range_rel_atr,
                'volume_ratio': data['volume'].values / (data['volume'].rolling(20).mean().values + 1e-6),
                'relative_intensity': price_range_rel_atr / (pd.Series(price_range_rel_atr).rolling(5).mean().values + 1e-6),
                'limit_thresholds': limit_thresholds,
                'unbuyable': unbuyable,
                'is_st': data['is_st'] if 'is_st' in data.columns else 0,
                'days_to_delist': data['days_to_delist'] if 'days_to_delist' in data.columns else -1
            })

            label_valid_idx = ~comp_df[['next_open', 'f_close', 'f_high_max', 'f_low_min', 'atr_raw', 'atr_rel']].isna().any(axis=1)
            final_valid_idx = valid_factor_idx & label_valid_idx
            if train_start_date:
                final_valid_idx = final_valid_idx & (data['date'] >= train_start_date)
            if train_end_date:
                final_valid_idx = final_valid_idx & (data['date'] <= train_end_date)

            if final_valid_idx.sum() <= 0:
                return None

            drop_cols = ['date', 'is_st', 'is_suspended', 'code', 'fore_adjust_factor', 'back_adjust_factor', 'days_to_delist', 'amount', 'turnover_rate']
            if target_features:
                X_df = factors.loc[final_valid_idx, factor_value_cols]
            else:
                X_df = factors[final_valid_idx].drop(columns=[c for c in drop_cols if c in factors.columns], errors='ignore')
            return {
                'X': X_df,
                'components': comp_df[final_valid_idx]
            }
        except Exception as e:
            if verbose:
                import traceback
                print(f"  Cached extraction error for {code}: {e}\n{traceback.format_exc()}")
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
        
        raw_scores = self._compute_path_quality_score(
            f_returns_norm, f_low_min_norm,
            components['f_high_idx'].values, components['f_low_idx'].values,
            f_high_max_norm,
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
                # 组内百分位排名，使用超快速 numpy argsort 实现
                y_final_sorted[start:start+count] = _fast_rankdata_1d(sorted_scores[start:start+count]) / (count + 1)
            else:
                y_final_sorted[start:start+count] = 0.5
        
        # 还原到原始顺序
        y_final = y_final_sorted[np.argsort(sort_idx)]
        
        # 重要性权重计算 (使用标准化最高收益率作为权重参考)
        # 使用 f_high_max_norm（已除以涨跌停阈值）而非原始收益率，
        # 消除板块间的权重偏差（创业板20%限制 vs 主板10%限制）
        w_sig = f_returns_norm.astype(np.float32)
        
        return y_final, raw_scores, f_returns_raw, w_sig

    @staticmethod
    def _apply_unbuyable_penalty(
        raw_scores: np.ndarray,
        returns_raw: np.ndarray,
        dates: np.ndarray,
        unbuyable_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Force next-day untradeable samples to the bottom of each date group.

        The model predicts after the close and enters at the next open. A sample
        that is suspended or locked at a one-price limit-up on that next day is
        not a realizable opportunity. ``punish`` therefore changes the target
        itself; it must not depend on optional sample weighting.
        """
        penalized_scores = np.asarray(raw_scores, dtype=np.float64).copy()
        penalized_returns = np.asarray(returns_raw, dtype=np.float32).copy()
        dates = np.asarray(dates)
        mask = np.asarray(unbuyable_mask, dtype=bool)
        ranked = np.empty(len(penalized_scores), dtype=np.float32)
        configured_floor = float(
            getattr(TrainingConfig, 'UNBUYABLE_PENALTY_SCORE', -100.0)
        )

        _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
        for start, count in zip(starts, counts):
            end = start + count
            group_mask = mask[start:end]
            group_scores = penalized_scores[start:end]
            group_returns = penalized_returns[start:end]

            if group_mask.any():
                buyable = ~group_mask
                if buyable.any():
                    score_floor = min(
                        configured_floor,
                        float(np.nanmin(group_scores[buyable])) - 1.0,
                    )
                    min_return = float(np.nanmin(group_returns[buyable]))
                    return_margin = max(1e-4, abs(min_return) * 0.01)
                    return_floor = min_return - return_margin
                else:
                    score_floor = configured_floor
                    return_floor = -1.0
                group_scores[group_mask] = score_floor
                group_returns[group_mask] = return_floor

            if count > 1:
                ranked[start:end] = _fast_rankdata_1d(group_scores) / (count + 1)
            else:
                ranked[start:end] = 0.5

        return ranked, penalized_scores, penalized_returns
                    


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
                       target_features: Optional[List[str]] = None,
                       use_factor_cache_only: bool = False,
                       return_codes: bool = False) -> tuple:
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
            use_factor_cache_only: 仅从 parquet 因子缓存读取特征，stocks_data 只用于生成标签
        
        返回:
            (X, y, returns, factor_names, dates, unbuyable, limit_groups, y_raw, is_st, w_sig)
        """

        # print("\n正在计算量化因子（技术指标 + K线形态 + 基本面）...")
        
        # 0. 自动更新市场情绪数据。直接使用因子缓存时，情绪因子已在 parquet 中，无需重复检查。
        if not use_factor_cache_only:
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
        
        if use_factor_cache_only:
            print(f"  缓存策略: 直接使用因子 parquet 构建训练集")
        elif cache_engineered_features:
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
            if use_factor_cache_only:
                return self._extract_stock_components_from_cache(
                    code, data, forward_days, target_features,
                    train_start_date, train_end_date, verbose
                )
            return self._extract_stock_components(
                code, data, forward_days, cache_engineered_features,
                target_features, verbose, train_start_date, train_end_date,
                include_fundamentals
            )
        
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
        del results
        gc.collect()
        
        if not all_X:
            raise ValueError("没有生成的有效样本")
            
        # 2. 批量合并标签组件，并按日期进行一次性的全局时间排序。
        # 特征矩阵直接散写到最终 float32 数组，避免同时持有未排序和已排序的巨大 X DataFrame。
        comps_df = pd.concat(all_comps, ignore_index=True)
        del all_comps
        gc.collect()

        sort_idx = np.argsort(comps_df['date'].values, kind='mergesort')
        
        comps_df = comps_df.iloc[sort_idx].reset_index(drop=True)
        inverse_sort = np.empty_like(sort_idx)
        inverse_sort[sort_idx] = np.arange(len(sort_idx), dtype=sort_idx.dtype)
        
        # 事件类别、指数成份等稀疏外部特征并不会在每只股票上都出现，
        # 因此缓存列集合允许不同。按首次出现顺序构造全股票特征并集，再将
        # 各股票重索引到统一 schema；不存在的列保留 NaN 供覆盖率筛选处理。
        factor_names = []
        seen_factor_names = set()
        for x_part in all_X:
            for name in x_part.columns:
                if name not in seen_factor_names:
                    factor_names.append(name)
                    seen_factor_names.add(name)
        X_arr = np.empty((len(sort_idx), len(factor_names)), dtype=np.float32)
        offset = 0
        for part_idx, x_part in enumerate(all_X):
            n_rows = len(x_part)
            dest_idx = inverse_sort[offset:offset + n_rows]
            aligned_part = x_part.reindex(columns=factor_names)
            X_arr[dest_idx, :] = aligned_part.to_numpy(dtype=np.float32, copy=False)
            offset += n_rows
            all_X[part_idx] = None
        
        # 释放中间列表
        del all_X, sort_idx, inverse_sort
        gc.collect()
        
        print(f"  - 原始样本量: {len(X_arr)}, 特征数: {X_arr.shape[1]}")
        
        # 特征极端值截断：对每列按 1%/99% 分位数 ± 5×IQR 截断。
        # JYDB 等外部特征使用原始元单位（如 EnterpriseValue 达 5e13），
        # 内部 market_cap 等单位使用 亿，量纲不统一。截断不影响截面排名
        # （极端值在排名中本就在首尾），但避免聚合统计被污染。
        for col_idx in range(X_arr.shape[1]):
            col_data = X_arr[:, col_idx]
            finite = col_data[np.isfinite(col_data)]
            if len(finite) > 10:
                lo, hi = np.percentile(finite, [1, 99])
                rng = hi - lo
                if rng > 0:
                    lower = lo - 5 * rng
                    upper = hi + 5 * rng
                    X_arr[:, col_idx] = np.clip(col_data, lower, upper)
        
        # 3. 向量化计算标签 (此时输入已是有序，截面排序速度最快，返回的标签和分数天然有序)
        print("  - 向量化生成标签 (标准化 + 截面排名)...")
        y_ranked, raw_scores, returns_raw, w_sig = self._calculate_vectorized_labels(comps_df)
        
        # 全局时间已是有序状态，直接提取和转换 numpy，完全避免了多次复制和多重排序！
        dates_arr = comps_df['date'].values
        codes_arr = comps_df['code'].astype(str).values
        unbuyable_arr = comps_df['unbuyable'].values
        limit_groups_arr = comps_df['limit_thresholds'].values
        is_st_arr = (comps_df['is_st'] == 1).values
        
        y_final_arr = y_ranked
        raw_scores_arr = raw_scores
        returns_arr = returns_raw
        w_sig_arr = w_sig

        del comps_df
        gc.collect()

        # 5. 不可买入样本处理（次日一字涨停/停牌）
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
                codes_arr        = codes_arr[keep_mask]
                limit_groups_arr = limit_groups_arr[keep_mask]
                is_st_arr        = is_st_arr[keep_mask]
                w_sig_arr        = w_sig_arr[keep_mask]
                raw_scores_arr   = raw_scores_arr[keep_mask]
                unbuyable_arr    = unbuyable_arr[keep_mask]
            elif handling == 'punish':
                print(f"  - 惩罚不可买入样本: {penalty_count} 个样本强制进入当日最低标签档")
                y_final_arr, raw_scores_arr, returns_arr = self._apply_unbuyable_penalty(
                    raw_scores_arr,
                    returns_arr,
                    dates_arr,
                    unbuyable_arr,
                )
            else:
                raise ValueError(
                    f"不支持的 UNBUYABLE_HANDLING={handling!r}，仅支持 'remove' 或 'punish'"
                )


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
        
        def _is_engineered_name(col: str) -> bool:
            """缓存特征只保留列名时，用稳定命名模式识别工程特征。"""
            return (
                col.startswith(('rank_', 'log_')) or
                any(token in col for token in ('_div_', '_mul_', '_sub_', '_x_'))
            )

        # 3. 特征工程 (衍生的复合因子)
        # 缓存训练路径不会重建 FeatureEngineer 实例的 generated_features，
        # 因此除实例列表外，还按列名模式识别交互、比率、rank/log 变换。
        _engineered_set = set(self.feature_engineer.get_generated_features())
        engineered_cols = [
            c for c in all_cols
            if c in remaining_all and (c in _engineered_set or _is_engineered_name(c))
        ]
        remaining_all -= set(engineered_cols)
        
        # 4. K线形态 — 精确匹配 CandlestickPatternFactors.get_pattern_names()
        _candle_set = set(self.candlestick_calculator.get_pattern_names())
        candle_cols = [c for c in all_cols if c in remaining_all and c in _candle_set]
        remaining_all -= set(candle_cols)
        
        # 5. 基本面因子 — 精确匹配 FundamentalFactors.NUMERIC_COLS + 已知衍生列
        _fund_known = set(FundamentalFactors.NUMERIC_COLS) | {
            'dynamic_pe', 'dynamic_pb', 'inv_pe', 'inv_pb', 'market_cap',
            'roe_x_np_growth', 'roe_to_pb',
            'peg', 'sue', 'eav'
        }
        fund_cols = [c for c in all_cols if c in remaining_all and c in _fund_known]
        remaining_all -= set(fund_cols)

        # 6. 聚源外部 PIT 特征
        external_cols = [c for c in all_cols if c in remaining_all and c.startswith('jy_')]
        remaining_all -= set(external_cols)
        
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

        # 7. 技术指标 — 兼容缓存中的周期后缀命名，如 rsi_12、atr_14、amount_ma_10
        _tech_set = set(self.tech_calculator.get_factor_names())
        _tech_prefixes = (
            'rsi_', 'roc_', 'mtm_', 'cmo_', 'stochrsi_', 'rvi_',
            'macd', 'adx_', 'dmi_', 'aroon_', 'trix_', 'ma_ratio_',
            'ma_slope_', 'price_slope_', 'atr_', 'natr_', 'bb_',
            'cci_', 'price_variance_', 'obv', 'ad', 'adosc', 'mfi_',
            'vroc_', 'vrsi_', 'vmacd', 'volume_ma_', 'volume_std_',
            'amount_ma_', 'amount_std_', 'willr_', 'bias_', 'vr_',
            'psy_', 'ar_', 'br_', 'cr_', 'kdj_', 'vol_ma_', 'vol_std_',
            'plus_di', 'minus_di', 'ulcer_', 'price_var_', 'sqrt_atr_', 'sqrt_natr_',
        )
        tech_cols = [
            c for c in all_cols
            if c in remaining_all and (c in _tech_set or c.startswith(_tech_prefixes))
        ]
        remaining_all -= set(tech_cols)
        
        # 9. 其它 (未被以上任何类别匹配到的列)
        other_cols = list(remaining_all)

        print("\n" + "="*50)
        print("数据集审计报告 (Factor Audit Report)")
        print("="*50)
        print(f"1. 技术指标 (Technical):    {len(tech_cols):>4} 个")
        print(f"2. K线形态 (Candlestick):   {len(candle_cols):>4} 个")
        print(f"3. 基础基本面 (Fundamental): {len(fund_cols):>4} 个")
        print(f"4. 市场情绪 (Sentiment):   {len(sentiment_cols):>4} 个")
        print(f"5. 聚源 PIT (JYDB):         {len(external_cols):>4} 个")
        print(f"6. 高级时序 (Advanced):     {len(adv_cols):>4} 个")
        print(f"7. 特征工程 (Engineered):   {len(engineered_cols):>4} 个")
        print(f"8. 其它状态 (Others):       {len(status_cols) + len(other_cols):>4} 个")
        if other_cols:
            print(f"   未分类列: {other_cols[:20]}{'...' if len(other_cols) > 20 else ''}")
        print("="*50 + "\n")
        
        # 统一输出 float32 以节省模型训练阶段的内存，XGB/LGB 内部也会转成 32 位
        output = (
            X_arr, y_final_arr, returns_arr, all_cols, dates_arr,
            unbuyable_arr, limit_groups_arr, raw_scores_arr, is_st_arr, w_sig_arr
        )
        return output + (codes_arr,) if return_codes else output

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
                w_sig_arr: np.ndarray = None,
                task: str = None,
                apply_feature_selection: bool = True,
                skip_normalization: bool = False) -> Dict:
        """
        训练多个模型
        """
        if task is None:
            task = getattr(TrainingConfig, 'TASK', 'ranking')

        # 数据验证和清理
        print("\n数据验证...")
        
        # 确保数据类型正确 (copy=False 避免不必要的内存复制)
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)

        # 这两个输入在公共 API 中是可选的。多目标训练不依赖 ST/信号权重时，
        # 使用中性的默认值，避免后续时间切分直接对 None 进行切片。
        if is_st_arr is None:
            is_st_arr = np.zeros(len(y), dtype=bool)
        else:
            is_st_arr = np.asarray(is_st_arr, dtype=bool)
        if w_sig_arr is None:
            w_sig_arr = np.ones(len(y), dtype=np.float32)
        else:
            w_sig_arr = np.asarray(w_sig_arr, dtype=np.float32)

        if len(is_st_arr) != len(y):
            raise ValueError(
                f"is_st_arr 长度 {len(is_st_arr)} 与标签长度 {len(y)} 不一致"
            )
        if len(w_sig_arr) != len(y):
            raise ValueError(
                f"w_sig_arr 长度 {len(w_sig_arr)} 与标签长度 {len(y)} 不一致"
            )
        
        # 记录原始缺失状态，但不要在覆盖率筛选和横截面排名之前填充。
        # Inf 同样视为无效观测；归一化结束后才统一填为中性值 0.5。
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0 or inf_count > 0:
            if nan_count > 0:
                print(f"  提示: 发现 {nan_count} 个缺失值，将先用于覆盖率筛选")
            if inf_count > 0:
                print(f"  警告: 发现 {inf_count} 个 Inf 值，按缺失值处理")
                X[np.isinf(X)] = np.nan
        
        # 特征质量诊断：检查是否所有特征都是常数（无区分度）
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            feature_stds = np.nanstd(X, axis=0)
        zero_std_count = np.sum(np.isfinite(feature_stds) & (feature_stds < 1e-6))
        if zero_std_count > 0:
            print(f"  警告: 发现 {zero_std_count} 个零方差特征（无区分度），建议检查特征工程")
            if zero_std_count > len(factor_names) * 0.5:
                print(f"  严重警告: 超过 50% 的特征无区分度，模型可能无法学习！")
        
        print(f"  数据验证完成: {X.shape[0]} 行, {X.shape[1]} 列")
        # 注意：此处统计为归一化前的原始特征值，仅用于诊断异常值是否已被清理
        finite_values = X[np.isfinite(X)]
        if finite_values.size:
            print(
                f"  特征统计(归一化前): mean={finite_values.mean():.4f}, "
                f"std={finite_values.std():.4f}, min={finite_values.min():.4f}, "
                f"max={finite_values.max():.4f}"
            )
        
        # 时间序列划分 (Embargo 阻隔期) + 横截面归一化。
        # skip_normalization=True 时仅返回划分点、不做任何写操作
        # （并行训练：父进程已统一归一化并落盘为只读 mmap，worker 直接复用）。
        split_idx, val_start_idx = self._normalize_train_val(
            X, dates, factor_names, skip_normalization=skip_normalization
        )

        # 重新定义切片范围
        X_train, X_val = X[:split_idx], X[val_start_idx:]
        y_train_full, y_val_full = y[:split_idx], y[val_start_idx:]
        dates_train, dates_val = dates[:split_idx], dates[val_start_idx:]
        returns_train, returns_val = returns[:split_idx], returns[val_start_idx:]
        is_st_train, is_st_val = is_st_arr[:split_idx], is_st_arr[val_start_idx:]
        w_sig_train, w_sig_val = w_sig_arr[:split_idx], w_sig_arr[val_start_idx:]

        # 覆盖率必须在任何填充/归一化之前、且只用训练段计算。
        feature_coverage_train = np.isfinite(X_train).mean(axis=0)

        # 归一化后统计：正常情况下 mean≈0.5, std≈0.29, min≈0, max≈1
        if not skip_normalization:
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
        _label_weighted = getattr(TrainingConfig, 'LABEL_WEIGHTED_FOR_XGB', False)
        _label_exponent = getattr(TrainingConfig, 'LABEL_WEIGHT_EXPONENT', 1.0)
        
        # LightGBM ranking 固定使用原始离散档位；XGBoost ranking 直接复用连续标签变换。
        # N_BINS 由 ModelConfig.get_n_bins() 统一获取（以 LIGHTGBM_PARAMS.label_gain 长度为准）。
        # 设计原则：档位数 × truncation_level ≈ 股票池大小，确保 lambdarank 有效 pair 密度。
        _n_bins = ModelConfig.get_n_bins()
        _mid_bin = _n_bins // 2  # 单样本日期的默认档位（中间档）
        y_train_discrete = np.empty(len(dates_train), dtype=np.int32)
        y_val_discrete = np.empty(len(dates_val), dtype=np.int32)
        
        for _dates_sub, _scores_sub, _y_sub, _y_discrete_lgb in [
            (dates_train, _label_source[:split_idx], y_train, y_train_discrete), 
            (dates_val, _label_source[val_start_idx:], y_val, y_val_discrete)
        ]:
            _, _d_starts, _d_counts = np.unique(_dates_sub, return_index=True, return_counts=True)
            for _ds, _dc in zip(_d_starts, _d_counts):
                _de = _ds + _dc
                if _dc > 1:
                    # 获取当前截面的原始分数
                    scores = _scores_sub[_ds:_de]
                    # clipped = sub_scores= np.clip(scores, -100, 50)
                    # min_val = clipped.min()
                    # max_val = clipped.max()
                    # sub_scores = (clipped - min_val) / (max_val - min_val)
                    ranks = _fast_rankdata_1d(scores) / (_dc + 1)
                    if _label_weighted:
                        _y_sub[_ds:_de] = np.power(ranks.astype(np.float32), _label_exponent)
                    else:
                        _y_sub[_ds:_de] = ranks
                    try:
                        # 需要 _n_bins+1 个边界点才能产生 _n_bins 个 bin
                        bin_0_watershed = 1/_n_bins
                        q_skewed = np.concatenate([
                            [0.0, bin_0_watershed],
                            np.linspace(bin_0_watershed, 1.0, _n_bins)
                        ])
                        bins = pd.qcut(scores, q=q_skewed, labels=False, duplicates='drop')
                        # 常量截面或有效分位点不足时 qcut 会返回全 NaN；直接转 int32
                        # 会变成 -2147483648，随后被 LightGBM 判为非法负标签。
                        if pd.isna(bins).all():
                            _y_discrete_lgb[_ds:_de] = _mid_bin
                        else:
                            _y_discrete_lgb[_ds:_de] = (
                                pd.Series(bins).fillna(_mid_bin).to_numpy(dtype=np.int32)
                            )

                    except ValueError:
                        raise ValueError(f"  {_dates_sub[_ds:_de]} 样本标签分布不均匀，请检查数据质量。")
                else:
                    _y_discrete_lgb[_ds:_de] = _mid_bin
                    _y_sub[_ds:_de] = 0.5
        
        # # 诊断输出：检查标签分布
        # print(f"\n[标签分布诊断]")
        # print(f"  训练集离散标签分布 (LightGBM 原始档位):")
        # unique_train, counts_train = np.unique(y_train_discrete, return_counts=True)
        # for bin_id, count in zip(unique_train, counts_train):
        #     print(f"    档位 {bin_id}: {count:>7} 样本 ({count/len(y_train_discrete)*100:>5.2f}%)")
        # if _label_weighted:
        #     print(f"  XGBoost/回归连续标签变换: exponent={_label_exponent}")
        # print(f"  原始分数统计 (训练集): min={_label_source[:split_idx].min():.4f}, max={_label_source[:split_idx].max():.4f}, std={_label_source[:split_idx].std():.4f}")
        

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
            for _dates_sub, _w_raw, _w_processed in [
                (dates_train, w_sig_train, w_sig_processed_train),
                (dates_val, w_sig_val, w_sig_processed_val)
            ]:
                _, _d_starts, _d_counts = np.unique(_dates_sub, return_index=True, return_counts=True)
                for _ds, _dc in zip(_d_starts, _d_counts):
                    _de = _ds + _dc
                    if _dc > 1:
                        w_norm = np.abs(_w_raw[_ds:_de]) / np.abs(_w_raw[_ds:_de]).mean()
                        w_exp = getattr(TrainingConfig, 'WEIGHT_EXPONENT', 2.0)
                        _w_processed[_ds:_de] = np.power(w_norm , w_exp).clip(0.0, 5.0)
                    else:
                        # 单样本日期：使用中性权重，无法计算组内排名
                        _w_processed[_ds:_de] = 1.0
            
            # 叠加不可买入样本的惩罚（如果在配置中设置了 punish）
            if unbuyable_mask is not None and getattr(TrainingConfig, 'UNBUYABLE_HANDLING', 'remove') == 'punish':
                unbuyable_train = unbuyable_mask[:split_idx]
                w_sig_processed_train = np.where(unbuyable_train, 0.2, w_sig_processed_train)
                
            # 更新训练集权重 (仅训练集需要权重)
            sample_weight_train *= w_sig_processed_train

        sample_weight_train = np.nan_to_num(sample_weight_train, nan=1.0, posinf=1.0, neginf=1.0)
        sample_weight_train /= (sample_weight_train.mean() + 1e-8)
        
        # 5. 特征选择：减少冗余和高度相关的特征 (New)
        from config.factor_config import OptimizationConfig
        selection_cache_file = os.path.join(TrainingConfig.SAVE_DIR, "selected_features.json")
        os.makedirs(TrainingConfig.SAVE_DIR, exist_ok=True)
        
        # 记录原始特征列表，用于同步过滤 X_val
        original_factor_names = list(factor_names)

        if apply_feature_selection:        
            # 尝试从缓存读取特征选择结果
            loaded_from_cache = False
            if os.path.exists(selection_cache_file):
                try:
                    import json
                    with open(selection_cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        # 只有当原始特征集完全一致时，才复用缓存
                        current_max_features = getattr(
                            OptimizationConfig, 'N_FEATURES_TO_SELECT', 200
                        )
                        if (
                            set(cached_data.get('original_features', [])) == set(original_factor_names)
                            and cached_data.get('max_features') == current_max_features
                            and cached_data.get('preprocessing_version') == 2
                        ):
                            factor_names = cached_data['selected_features']
                            print(f"\n[特征优化] 从缓存加载特征选择结果 (保留 {len(factor_names)} 个核心特征)")
                            loaded_from_cache = True
                except Exception as e:
                    print(f"  读取特征选择缓存失败: {e}")

            if not loaded_from_cache:
                print(f"\n[特征优化] 正在进行特征选择 (原始特征数: {len(factor_names)})...")
                # 使用相关性过滤
                X_train, factor_names = self._select_features(
                    X_train, factor_names, y=y_train, corr_threshold=0.95,
                    feature_coverage=feature_coverage_train,
                )
                # 保存特征选择结果到缓存
                try:
                    import json
                    with open(selection_cache_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'original_features': original_factor_names,
                            'selected_features': factor_names,
                            'max_features': getattr(
                                OptimizationConfig, 'N_FEATURES_TO_SELECT', 200
                            ),
                            'preprocessing_version': 2,
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
        else:
            feature_names=original_factor_names



        results = {}
        for model_type in model_types:
            print(f"\n训练 {model_type.upper()} 模型 ({task})")
            try:
                model = MLFactorModel(model_type=model_type, task=task)

                # ── Early Stopping 验证集策略 ──────────────────────────────────
                # 直接使用外部纯净验证集（阻隔期后）做 early stopping，
                # 避免内部切分验证集时间分布与外部验证集不一致导致的过早停止。
                # 外部验证集时间更靠后，与真实推理场景一致，early stopping 更可靠。
                _, es_val_group = np.unique(dates_val, return_counts=True)
                if model.task == 'ranking':
                    if model_type == 'xgboost':
                        y_train_rank = y_train
                        y_val_rank = y_val
                    else:
                        y_train_rank = y_train_discrete
                        y_val_rank = y_val_discrete
                    train_result = model.train(
                        X_train, y_train_rank,
                        validation_split=0.2,
                        use_time_series_split=True,
                        feature_names=factor_names,
                        sample_weight=sample_weight_train,
                        returns=returns_train,
                        split_idx=len(X_train),
                        X_val_external=X_val,
                        y_val_external=y_val_rank,
                        dates_val_external=dates_val,
                        dates=dates_train,
                        group=train_group,
                        eval_group=es_val_group,
                    )
                else:
                    train_result = model.train(
                        X_train, y_train,
                        validation_split=0.2,
                        use_time_series_split=True,
                        feature_names=factor_names,
                        sample_weight=sample_weight_train,
                        returns=returns_train,
                        split_idx=len(X_train),
                        X_val_external=X_val,
                        y_val_external=y_val,
                        dates_val_external=dates_val,
                        dates=dates_train,
                    )
                # 训练集评估（采样，用于与验证集对比学习效果）
                train_eval = model._evaluate(X_train, y_train, "训练集", returns=returns_train, dates=dates_train,sample_ratio=1)
                train_result['train_metrics'] = train_eval

                # 在验证集（阻隔期后）上做最终评估
                val_eval = model._evaluate(X_val, y_val, "验证集", returns=returns_val, dates=dates_val,sample_ratio=1)
                train_result['val_metrics'] = val_eval
                
                self.models[model_type] = model
                results[model_type] = train_result
            except Exception as e:
                import traceback
                print(f"训练 {model_type} 失败: {e}")
                print(traceback.format_exc())
                continue
        
        return results

    def _normalize_train_val(self, X: np.ndarray, dates: np.ndarray,
                             factor_names: List[str],
                             skip_normalization: bool = False):
        """时间序列划分 (Embargo 阻隔期) + 横截面归一化，返回 (split_idx, val_start_idx)。

        归一化只依赖 X 与 dates（与标签 y 无关），因此可由父进程统一做一次，
        供并行训练的多个 worker 复用同一份已归一化矩阵，避免每个 worker 重复写大矩阵。

        skip_normalization=True：仅计算并返回划分点，不做任何写操作
        （用于并行训练 worker 在只读 mmap 上运行）。
        """
        forward_days = getattr(TrainingConfig, 'FUTURE_DAYS', 7)
        raw_split_idx = int(len(dates) * TrainingConfig.TRAIN_TEST_SPLIT)
        split_date = dates[raw_split_idx]

        # 训练集：[0, split_idx)
        split_idx = np.searchsorted(dates, split_date, side='left')

        # 验证集：[val_start_idx, len(dates))
        # 阻隔期逻辑：标签含未来 forward_days 信息，验证集特征必须在训练集标签最晚日期之后。
        unique_dates = np.unique(dates)
        split_date_idx = np.searchsorted(unique_dates, split_date)
        val_start_date = unique_dates[min(split_date_idx + forward_days, len(unique_dates) - 1)]
        val_start_idx = np.searchsorted(dates, val_start_date, side='left')

        print(f"  划分点 (Embargo): {split_date}, 阻隔期后起: {val_start_date}")
        print(f"  训练集: {split_idx} 样本, 验证集: {len(dates) - val_start_idx} 样本")
        print(f"  已剔除阻隔期重叠样本: {val_start_idx - split_idx} 个")

        if not skip_normalization:
            print("\n  对训练样本进行横截面归一化...")
            skip_col_stats = self._apply_cross_sectional_normalization_inplace(
                X[:split_idx], dates[:split_idx], factor_names
            )
            # 持久化到实例，供 save_models 写入磁盘，推理时复用
            self.norm_stats = {
                'skip_col_stats': skip_col_stats,
                'factor_names': factor_names,
            }
            print("  对验证样本进行归一化...")
            self._apply_cross_sectional_normalization_inplace(
                X[val_start_idx:], dates[val_start_idx:], factor_names,
                skip_col_stats=skip_col_stats
            )
        return split_idx, val_start_idx

    def train_multiobjective_models(
        self,
        X: np.ndarray,
        targets: pd.DataFrame,
        factor_names: List[str],
        dates: np.ndarray,
        objective_weights: Dict[str, float] = None,
        model_type: str = 'lightgbm',
        objective_workers: int = 1,
    ):
        """训练多个方向统一的目标模型并返回组合模型及训练结果。

        ``targets`` 应包含 ``rank_y_*`` 列，所有列均遵循越大越好的语义。
        特征仅在训练数据上进行一次多目标联合筛选，随后每个目标复用相同列。
        """
        from core.factors.feature_selector import CrossSectionalFeatureSelector
        from core.factors.ml_factor_model import MultiObjectiveFactorModel
        from config.factor_config import OptimizationConfig

        weights = dict(objective_weights or getattr(TrainingConfig, 'MULTI_OBJECTIVE_WEIGHTS', {}))
        objective_cols = []
        normalized_weights = {}
        for raw_name, weight in weights.items():
            rank_name = raw_name if raw_name.startswith('rank_') else f'rank_{raw_name}'
            if rank_name in targets.columns and weight > 0:
                objective_cols.append(rank_name)
                normalized_weights[rank_name] = float(weight)
        if not objective_cols:
            raise ValueError("targets 中没有与 MULTI_OBJECTIVE_WEIGHTS 匹配的 rank 目标")

        target_matrix = targets[objective_cols].to_numpy(dtype=np.float32)
        complete = np.isfinite(target_matrix).all(axis=1)
        if complete.sum() < 100:
            raise ValueError(f"多目标完整样本不足: {int(complete.sum())}")
        X_fit = np.asarray(X, dtype=np.float32)[complete]
        y_fit = target_matrix[complete]
        dates_fit = np.asarray(dates)[complete]

        # 与 train_models 使用相同的时间切分口径。目标可学习性判断和特征选择
        # 都只能看到训练段，验证段标签不得参与选列。
        raw_selection_split = int(len(dates_fit) * TrainingConfig.TRAIN_TEST_SPLIT)
        if raw_selection_split <= 0 or raw_selection_split >= len(dates_fit):
            raise ValueError("多目标训练时间切分后没有足够的训练/验证样本")
        selection_split_date = dates_fit[raw_selection_split]
        selection_train_mask = dates_fit < selection_split_date
        if selection_train_mask.sum() < 100:
            raise ValueError(f"多目标特征筛选训练样本不足: {int(selection_train_mask.sum())}")

        # 某些目标在特定训练股票池中可能没有正负样本，例如全体股票未来窗口
        # 都可交易。此时不存在可学习的横截面关系，应明确跳过并对剩余权重
        # 重新归一化，而不是训练伪模型或让离散化产生非法标签。
        # 判断的是“同一交易日内”是否存在排序差异，不能只看全样本标准差；
        # 当每日标签全相同但当日股票数不同，pct-rank 的日均值会轻微变化，
        # 全局标准差会误判为可学习信号。
        variable_mask = np.zeros(y_fit.shape[1], dtype=bool)
        selection_dates = dates_fit[selection_train_mask]
        selection_targets = y_fit[selection_train_mask]
        for date in np.unique(selection_dates):
            day_values = selection_targets[selection_dates == date]
            if len(day_values) > 1:
                variable_mask |= np.nanmax(day_values, axis=0) - np.nanmin(day_values, axis=0) > 1e-8
        if not variable_mask.all():
            skipped = [name for name, keep in zip(objective_cols, variable_mask) if not keep]
            print(f"  [多目标训练] 跳过无横截面变化目标: {skipped}")
            objective_cols = [name for name, keep in zip(objective_cols, variable_mask) if keep]
            y_fit = y_fit[:, variable_mask]
            normalized_weights = {
                name: weight for name, weight in normalized_weights.items()
                if name in objective_cols
            }
        if not objective_cols:
            raise ValueError("所有多目标标签在训练样本中均无横截面变化")

        selector = CrossSectionalFeatureSelector(
            max_features=getattr(OptimizationConfig, 'N_FEATURES_TO_SELECT', 200),
            min_coverage=0.20,
            corr_threshold=getattr(OptimizationConfig, 'CORRELATION_THRESHOLD', 0.95),
        )
        ordered_weights = [normalized_weights[name] for name in objective_cols]

        # 筛选必须使用与最终模型一致的训练段横截面预处理；覆盖率则来自
        # 预处理前的原始缺失状态，避免中性值 0.5 被误算为有效观测。
        raw_selection_X = X_fit[selection_train_mask]
        feature_coverage = np.isfinite(raw_selection_X).mean(axis=0)
        selection_X = raw_selection_X.copy()
        self._apply_cross_sectional_normalization_inplace(
            selection_X, dates_fit[selection_train_mask], factor_names,
        )
        selector.fit(
            selection_X, factor_names,
            y_fit[selection_train_mask], target_weights=ordered_weights,
            feature_coverage=feature_coverage,
        )
        selected_names = list(selector.report_.selected_features)
        selected_indices = [factor_names.index(name) for name in selected_names]
        X_selected = X_fit[:, selected_indices]

        trained_models = {}
        objective_results = {}
        previous_models = self.models
        try:
            if objective_workers and objective_workers > 1 and len(objective_cols) > 1:
                # 并行路径：各目标在独立进程中训练，训练代码路径与串行完全一致
                trained_models, objective_results = self._train_objectives_parallel(
                    X_selected, y_fit, dates_fit, selected_names, objective_cols,
                    normalized_weights, model_type, int(objective_workers),
                )
            else:
                # 串行路径（默认）：与历史行为一致
                for idx, objective in enumerate(objective_cols):
                    print(f"\n[多目标训练] {objective} ({idx + 1}/{len(objective_cols)})")
                    objective_y = y_fit[:, idx]
                    result = self.train_models(
                        X_selected.copy(), objective_y.copy(), objective_y.copy(),
                        selected_names, dates_fit,
                        model_types=[model_type], task='ranking',
                        apply_feature_selection=False,
                    )
                    if model_type not in self.models or model_type not in result:
                        raise RuntimeError(f"目标 {objective} 训练失败")
                    objective_result = result[model_type]
                    train_metrics = objective_result.get('train_metrics', {})
                    val_metrics = objective_result.get('val_metrics', {})
                    if (
                        objective.endswith('tradable_20d')
                        and abs(float(train_metrics.get('rank_ic', 0.0))) < 1e-12
                        and abs(float(val_metrics.get('rank_ic', 0.0))) < 1e-12
                        and abs(float(val_metrics.get('auc', 0.5)) - 0.5) < 1e-12
                    ):
                        print(f"  [多目标训练] 跳过无预测区分度目标: {objective}")
                        self.models = {}
                        continue
                    trained_models[objective] = self.models[model_type]
                    objective_results[objective] = objective_result
                    self.models = {}
        finally:
            self.models = previous_models

        wrapper = MultiObjectiveFactorModel(trained_models, {
            name: normalized_weights[name] for name in trained_models
        })
        return wrapper, objective_results, selected_names


    def _train_objectives_parallel(self, X_selected, y_fit, dates_fit, selected_names,
                                   objective_cols, normalized_weights, model_type,
                                   objective_workers):
        """并行训练多个目标。

        设计要点：
        - 大特征矩阵 X_selected 只落盘一次为 .npy，各 worker 以只读 mmap 共享，
          避免把数百 MB 矩阵随每个任务 pickle 一份。
        - 每个 worker 进程内把 LightGBM/XGBoost 线程数限制为 cpu//n_workers，
          既保证多目标真正并发，又避免线程 oversubscription 导致反而变慢。
        - 每个 worker 在独立进程里重建 MLModelTrainer 并调用 train_models，
          训练逻辑与串行路径完全相同，结果可复现。
        """
        import os as _os
        import tempfile
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor

        cpu = _mp.cpu_count()
        n_workers = min(objective_workers, len(objective_cols))
        threads_per = max(1, cpu // n_workers)
        print(f"\n[多目标并行训练] 目标数={len(objective_cols)} 进程数={n_workers} "
              f"每进程LightGBM线程≈{threads_per} (避免 oversubscription)")

        # 大特征矩阵只归一化+落盘一次，worker 只读 mmap 共享（归一化与标签无关，做一次即可）
        # 归一化会就地写回 X_selected（父进程内为可写副本），并设置 self.norm_stats 供落盘。
        self._normalize_train_val(X_selected, dates_fit, selected_names)
        tmpf = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        _np_save = __import__('numpy').save
        _np_save(tmpf.name, np.asarray(X_selected, dtype=np.float32))
        tmpf.close()
        try:
            futures = {}
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                for objective in objective_cols:
                    idx = objective_cols.index(objective)
                    payload = dict(
                        db_path=self.db_path,
                        x_path=tmpf.name,
                        y=np.asarray(y_fit[:, idx], dtype=np.float32),
                        dates=np.asarray(dates_fit),
                        names=selected_names,
                        model_type=model_type,
                        n_threads=threads_per,
                    )
                    futures[objective] = ex.submit(_train_objective_worker, payload)
                raw_results = {obj: fut.result() for obj, fut in futures.items()}
        finally:
            try:
                _os.remove(tmpf.name)
            except OSError:
                pass

        trained_models = {}
        objective_results = {}
        for objective, (model, objective_result) in raw_results.items():
            train_metrics = (objective_result or {}).get('train_metrics', {})
            val_metrics = (objective_result or {}).get('val_metrics', {})
            if (
                objective.endswith('tradable_20d')
                and abs(float(train_metrics.get('rank_ic', 0.0))) < 1e-12
                and abs(float(val_metrics.get('rank_ic', 0.0))) < 1e-12
                and abs(float(val_metrics.get('auc', 0.5)) - 0.5) < 1e-12
            ):
                print(f"  [多目标并行训练] 跳过无预测区分度目标: {objective}")
                continue
            trained_models[objective] = model
            objective_results[objective] = objective_result
        return trained_models, objective_results


    def _apply_cross_sectional_normalization_inplace(self, X: np.ndarray, dates: np.ndarray, 
                                                   factor_names: List[str],
                                                   skip_col_stats: Optional[dict] = None):
        """
        使用并行化处理和内存视图，原位对特征矩阵进行横截面归一化，降低内存占用并大幅提升性能。
        """
        rank_cols_mask = np.array([
            not TrainingConfig.should_skip_rank(col) for col in factor_names
        ])
        rank_cols_idx = np.where(rank_cols_mask)[0]
        skip_cols_idx = np.where(~rank_cols_mask)[0]

        # ── B/C 类：连续型跳过列，robust scaler + sigmoid → [0,1] ──
        # 优化：采用单列内存视图处理，完全避免生成大面积临时 2D 矩阵的复制
        _binary_skip_names = {
            'is_limit_up', 'is_suspended', 'is_st',
            'white_candle', 'black_candle', 'doji', 'hammer', 'hanging_man',
            'shooting_star', 'inverted_hammer', 'marubozu', 'spinning_top',
            'bullish_engulfing', 'bearish_engulfing', 'piercing_line',
            'dark_cloud_cover', 'morning_star', 'evening_star', 'harami',
            'three_white_soldiers', 'three_black_crows',
        }
        def _is_binary_skip(col: str) -> bool:
            if col in _binary_skip_names:
                return True
            if col.startswith('is_') and not col.endswith('_encoded'):
                return True
            return False

        if len(skip_cols_idx) > 0:
            skip_names = [factor_names[i] for i in skip_cols_idx]
            binary_mask  = np.array([_is_binary_skip(n) for n in skip_names])  # A 类
            robust_mask  = ~binary_mask                                          # B/C 类

            robust_local_idx = np.where(robust_mask)[0]   # 在 skip_cols_idx 内的相对下标
            robust_global_idx = skip_cols_idx[robust_local_idx]  # 在 X 中的绝对列下标

            if robust_local_idx.size > 0:
                if skip_col_stats is None:
                    # 单列视图 Robust 统计量计算以降低拷贝
                    median = np.empty(len(robust_global_idx), dtype=np.float32)
                    iqr = np.empty(len(robust_global_idx), dtype=np.float32)
                    valid_iqr = np.empty(len(robust_global_idx), dtype=bool)
                    
                    for idx, col in enumerate(robust_global_idx):
                        col_view = X[:, col]
                        p25 = np.nanpercentile(col_view, 25)
                        p75 = np.nanpercentile(col_view, 75)
                        median[idx] = np.nanpercentile(col_view, 50)
                        iqr[idx] = p75 - p25
                        valid_iqr[idx] = iqr[idx] > 1e-8
                        
                    skip_col_stats = {
                        'median': median, 'iqr': iqr, 'valid_iqr': valid_iqr,
                        'robust_global_idx': robust_global_idx,
                    }
                else:
                    median    = skip_col_stats['median']
                    iqr       = skip_col_stats['iqr']
                    valid_iqr = skip_col_stats['valid_iqr']
                
                # 原地鲁棒标准化 + sigmoid，无多余临时数组分配
                for idx, col in enumerate(robust_global_idx):
                    col_view = X[:, col]
                    if valid_iqr[idx]:
                        scaled = (col_view - median[idx]) / iqr[idx]
                        np.clip(scaled, -10.0, 10.0, out=scaled)
                        scaled_01 = 1.0 / (1.0 + np.exp(-scaled))
                        X[:, col] = scaled_01
                    else:
                        missing = ~np.isfinite(col_view)
                        X[:, col] = 0.0
                        X[missing, col] = 0.5
            elif skip_col_stats is None:
                skip_col_stats = {
                    'median': np.array([]), 'iqr': np.array([]),
                    'valid_iqr': np.array([], dtype=bool),
                    'robust_global_idx': np.array([], dtype=int),
                }

        # ── 横截面归一化优化：使用 Joblib 并行化处理日期分组 ──
        unique_dates, group_start, group_counts = np.unique(
            dates, return_index=True, return_counts=True
        )
        
        # 确定并行任务的 CPU 核心数
        n_jobs = getattr(TrainingConfig, 'N_JOBS_FACTOR_CALC', -1)
        if n_jobs <= 0:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        
        num_dates = len(unique_dates)
        n_jobs = min(n_jobs, num_dates)
        
        if n_jobs > 1 and num_dates > 5:
            # 采用大规模日期切片，将大矩阵分割为核心数个连续子块，避免大量小数组序列化开销
            date_chunks = np.array_split(np.arange(num_dates), n_jobs)
            
            tasks = []
            for chunk in date_chunks:
                if len(chunk) == 0:
                    continue
                first_date_idx = chunk[0]
                last_date_idx = chunk[-1]
                
                chunk_start_row = group_start[first_date_idx]
                chunk_end_row = group_start[last_date_idx] + group_counts[last_date_idx]
                
                rel_starts = group_start[chunk] - chunk_start_row
                counts = group_counts[chunk]
                
                tasks.append((chunk_start_row, chunk_end_row, rel_starts, counts))
                
            # 在每个连续行块内原地赋值，避免 loky 子进程序列化大矩阵切片。
            from joblib import Parallel, delayed
            Parallel(n_jobs=n_jobs,  require="sharedmem")(
                delayed(_normalize_chunk_worker)(
                    X,
                    rank_cols_idx,
                    chunk_start_row,
                    rel_starts,
                    counts,
                )
                for chunk_start_row, chunk_end_row, rel_starts, counts in tasks
            )
        else:
            # 串行备用方案，利用 _fast_rankdata_1d 依然极快
            for start, count in zip(group_start, group_counts):
                if count <= 1:
                    X[start:start+count, rank_cols_idx] = 0.5
                    continue
                
                day_data = X[start:start+count, rank_cols_idx]
                for j in range(len(rank_cols_idx)):
                    col_data = day_data[:, j]
                    X[start:start+count, rank_cols_idx[j]] = _rank_finite_to_unit_interval(col_data)

        # 模型最终只消费有限的 [0, 1] 数值。缺失值在完成覆盖率统计和
        # 有效值排名后才置为中性 0.5，避免参与排序。
        np.nan_to_num(X, copy=False, nan=0.5, posinf=1.0, neginf=0.0)
        gc.collect()
        return skip_col_stats

    def _select_features(self, X: np.ndarray, feature_names: List[str],
                       y: np.ndarray = None,
                       corr_threshold: float = 0.8,
                       method: str = 'target_aware',
                       feature_coverage: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
        """
        特征选择：过滤高度相关的特征
        """
        if method == 'none' or len(feature_names) <= 50:
            return X, feature_names
        if y is None:
            raise ValueError("目标感知特征筛选需要训练标签 y")
        from core.factors.feature_selector import CrossSectionalFeatureSelector
        from config.factor_config import OptimizationConfig

        selector = CrossSectionalFeatureSelector(
            max_features=getattr(OptimizationConfig, 'N_FEATURES_TO_SELECT', 200),
            min_coverage=0.20,
            corr_threshold=corr_threshold,
        )
        X_selected, selected = selector.fit_transform(
            X, feature_names, y, feature_coverage=feature_coverage,
        )
        report = selector.report_
        print(
            f"  - 目标感知筛选: {len(feature_names)} -> {len(selected)}；"
            f"低覆盖 {len(report.dropped_low_coverage)}，常量 {len(report.dropped_constant)}，"
            f"冗余 {len(report.dropped_redundant)}"
        )
        return X_selected, selected
    
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
        timestamp    = datetime.now().strftime('%m%d_%H%M')
        
        # 获取训练配置的重要特征
        model_types = list(self.models.keys())
        model_type_str = '_'.join(sorted(model_types)) if len(model_types) <= 2 else 'multi'
        
        # 构建简洁明了的文件夹名称
        # 格式: {模型类型缩写}_{预测天数}d_{训练年数}y_{股票数}s_{配置标志}_{时间}
        # 模型类型缩写: xg=xgboost, lgb=lightgbm, xl=双模型
        if len(model_types) == 2 and 'xgboost' in model_types and 'lightgbm' in model_types:
            model_abbr = 'xl'
        elif 'xgboost' in model_types:
            model_abbr = 'xg'
        elif 'lightgbm' in model_types:
            model_abbr = 'lgb'
        else:
            model_abbr = model_type_str[:3]
        
        # 配置标志: 使用单个字符表示重要配置
        # G=GPU, W=加权, F=基本面, C=K线形态
        config_flags = []
        if getattr(TrainingConfig, 'USE_GPU', False):
            config_flags.append('G')
        if getattr(TrainingConfig, 'USE_SAMPLE_WEIGHT', False):
            config_flags.append('W')
        if getattr(TrainingConfig, 'INCLUDE_FUNDAMENTALS', True):
            config_flags.append('F')
        if getattr(TrainingConfig, 'INCLUDE_CANDLE_PATTERN', False):
            config_flags.append('C')
        
        config_str = ''.join(config_flags) if config_flags else 'N'
        
        # 任务类型缩写
        task_abbr = self.task[:2] if len(self.task) > 2 else self.task
        
        # 构建文件夹名称 (更简洁的格式)
        archive_name = f"{model_abbr}_{forward_days}d_{years}y_{stocks}s_{config_str}_{task_abbr}_{timestamp}"
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
        
        # 保存训练配置信息
        self._save_training_config(archive_dir, years, stocks, forward_days)
            
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
    
    def _save_training_config(self, save_dir: str, years: int, stocks: int, forward_days: int):
        """
        保存训练配置信息到JSON文件
        
        参数:
            save_dir: 保存目录
            years: 训练数据年数
            stocks: 训练股票数量
            forward_days: 预测天数
        """
        import json
        from config.factor_config import TrainingConfig, ModelConfig
        
        config_info = {
            'training_info': {
                'model_types': list(self.models.keys()),
                'task': self.task,
                'forward_days': forward_days,
                'training_years': years,
                'stock_count': stocks,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'folder_name': os.path.basename(save_dir)
            },
            'training_config': {
                'use_gpu': getattr(TrainingConfig, 'USE_GPU', False),
                'use_sample_weight': getattr(TrainingConfig, 'USE_SAMPLE_WEIGHT', False),
                'include_fundamentals': getattr(TrainingConfig, 'INCLUDE_FUNDAMENTALS', True),
                'include_candle_pattern': getattr(TrainingConfig, 'INCLUDE_CANDLE_PATTERN', False),
                'filter_by_strategy': getattr(TrainingConfig, 'FILTER_BY_STRATEGY', False),
                'memory_efficient': getattr(TrainingConfig, 'MEMORY_EFFICIENT', True),
                'short_prediction': getattr(TrainingConfig, 'SHORT_PREDICTION', False),
                'train_test_split': getattr(TrainingConfig, 'TRAIN_TEST_SPLIT', 0.7),
                'unbuyable_handling': getattr(TrainingConfig, 'UNBUYABLE_HANDLING', 'punish'),
                'weight_exponent': getattr(TrainingConfig, 'WEIGHT_EXPONENT', 1.5)
            },
            'model_config': {
                'n_bins': ModelConfig.get_n_bins(),
                'xgboost_params': {
                    'n_estimators': ModelConfig.XGBOOST_PARAMS.get('n_estimators'),
                    'max_depth': ModelConfig.XGBOOST_PARAMS.get('max_depth'),
                    'learning_rate': ModelConfig.XGBOOST_PARAMS.get('learning_rate'),
                    'objective': ModelConfig.XGBOOST_PARAMS.get('objective')
                },
                'lightgbm_params': {
                    'n_estimators': ModelConfig.LIGHTGBM_PARAMS.get('n_estimators'),
                    'max_depth': ModelConfig.LIGHTGBM_PARAMS.get('max_depth'),
                    'num_leaves': ModelConfig.LIGHTGBM_PARAMS.get('num_leaves'),
                    'learning_rate': ModelConfig.LIGHTGBM_PARAMS.get('learning_rate'),
                    'objective': ModelConfig.LIGHTGBM_PARAMS.get('objective')
                }
            },
            'performance_metrics': {
                'model_count': len(self.models),
                'trained_models': list(self.models.keys())
            }
        }
        
        config_path = os.path.join(save_dir, 'training_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, ensure_ascii=False, indent=2)
        
        print(f"  [OK] 训练配置已保存: training_config.json")
    
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
        external_names = sorted(name for name in all_set if name.startswith('jy_'))
        classified = _tech_set | _candle_set | _fund_known | _sentiment_known | _adv_known | _engineered_set | _status_known | set(external_names)
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
            'external_source_factors': len(external_names),
            'other_factors': len(other_names),
            'factor_names': factor_names,
            'technical_factor_names': tech_names,
            'candlestick_factor_names': candle_names,
            'fundamental_factor_names': fund_names,
            'sentiment_factor_names': sentiment_names,
            'advanced_factor_names': adv_names,
            'engineered_factor_names': engineered_names,
            'status_factor_names': status_names,
            'external_source_factor_names': external_names,
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
