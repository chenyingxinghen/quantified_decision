"""
ML因子策略（回测版本）
将ML因子模型集成到新的回测框架
回测时完全依赖训练阶段生成的因子缓存，不再实时计算特征工程
"""
import os
import sys
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import talib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from core.backtest.strategy import BaseStrategy, StrategySignal
from core.factors.ml_factor_model import MLFactorModel
import config.strategy_config as sc
import config.factor_config as fc
from config import DATABASE_PATH, PROJECT_ROOT, SUPPORTED_MARKETS

class MLFactorBacktestStrategy(BaseStrategy):
    """ML因子回测策略
    
    回测时完全依赖训练阶段生成的因子缓存。
    """
    
    def __init__(self,
                 model_path: str,
                 min_confidence: float = sc.ML_FACTOR_MIN_CONFIDENCE,
                 use_cache: bool = True,
                 cache_dir: str = None,
                 norm_stats_path: str = None,
                 name: str = "ML因子策略"):
        """初始化策略"""
        super().__init__(name)
        self.model_path = (
            model_path if os.path.isabs(model_path)
            else os.path.join(PROJECT_ROOT, model_path)
        )
        self.norm_stats_path = (
            norm_stats_path if not norm_stats_path or os.path.isabs(norm_stats_path)
            else os.path.join(PROJECT_ROOT, norm_stats_path)
        )
        self.min_confidence = min_confidence
        self.use_cache = use_cache
        
        if cache_dir is None:
            cache_dir = fc.TrainingConfig.CACHE_DIR
        self.cache_dir = cache_dir
        
        self.model = None
        self._factors_cache = {}  # 内存缓存，用于存放 parquet 加载全量因子数据
        self._factor_dates_cache = {}
        self._factor_matrix_cache = {}
        self._finance_history = {}
        self._warned_stocks = set()
    
    def initialize(self, **kwargs):
        """初始化策略 - 预定全量缓存以极大提升速度"""
        super().initialize(**kwargs)
        self._active_stock_codes = set(kwargs.get('stock_codes') or [])
        
        # 0. 预加载所有 PIT 筛选所需的基本面指标 (性能优化核心)
        print(f"正在进行 PIT 数据全量预缓存...")
        self._precompute_pit_data()
        
        # 智能加载模型
        from core.factors.ml_factor_model import MLFactorModel, EnsembleFactorModel
        def _load_smart_model(target_path):
            if os.path.isdir(target_path):
                pkls = [
                    os.path.join(target_path, f)
                    for f in os.listdir(target_path)
                    if f.endswith('_factor_model.pkl') or f == 'ensemble_factor_model.pkl'
                ]
                if pkls:
                    latest_pkl = sorted(pkls, key=os.path.getmtime)[-1]
                    return _load_smart_model(latest_pkl)
                return None
            if not os.path.exists(target_path): return None
            try:
                return EnsembleFactorModel.load_model(target_path)
            except:
                try:
                    m = MLFactorModel(); m.load_model(target_path); return m
                except: return None

        self.model = _load_smart_model(self.model_path)
        if self.model is None: raise ValueError(f"无法加载模型: {self.model_path}")

        # 加载归一化统计量（与模型同目录的 norm_stats.pkl）
        import pickle as _pickle
        _model_dir = os.path.dirname(os.path.abspath(self.model_path)) if os.path.isfile(self.model_path) else os.path.abspath(self.model_path)
        _norm_path = (
            os.path.abspath(self.norm_stats_path)
            if self.norm_stats_path
            else os.path.join(_model_dir, 'norm_stats.pkl')
        )
        if os.path.exists(_norm_path):
            try:
                with open(_norm_path, 'rb') as _f:
                    self.norm_stats = _pickle.load(_f)
                print(f"  已加载归一化统计量: {_norm_path}")
            except Exception as _e:
                print(f"  警告: 加载归一化统计量失败: {_e}")
                self.norm_stats = None
        else:
            if self.norm_stats_path:
                raise FileNotFoundError(f"指定的归一化统计量不存在: {_norm_path}")
            print(f"  警告: 模型目录缺少 norm_stats.pkl，跳过全局列归一化: {_model_dir}")
            self.norm_stats = None

        if self.use_cache:
            self._preload_factor_cache()
        
        print(f"策略初始化完成: {self.name} (已启用 PIT 预缓存 ✓)")
    
    def generate_signals(self,
                        current_date: str,
                        market_data: Any,
                        portfolio_state: Dict[str, Any],
                        criteria: Optional[Dict] = None,
                        min_confidence: Optional[float] = None) -> List[StrategySignal]:
        """生成交易信号 (极速版)"""
        signals = []
        existing_positions = portfolio_state.get('positions', {})
        
        # 确定置信度阈值：优先使用参数，其次使用实例属性
        effective_min_confidence = min_confidence if min_confidence is not None else self.min_confidence
        
        # 优先使用 portfolio_state 中的 available_slots (用于实盘选股指定数量)
        # 如果未指定，则根据最大仓位限制计算剩余空位
        if 'available_slots' in portfolio_state:
            available_slots = portfolio_state['available_slots']
        else:
            available_slots = sc.MAX_POSITIONS - len(existing_positions)
            
        if available_slots <= 0: return signals

        # 1. 获取所有股票列表
        all_codes = market_data.keys()

        # 筛选逻辑：
        # 1. 回测场景：criteria 为 None，完全使用 sc 配置
        # 2. 后端场景：criteria 由前端传入，前端必须明确传入所有参数（不传或空值则不限制该条件）
        # 3. 自动化场景：criteria 由自动化传入，自动化配置优先，未设置的参数使用 sc 兜底
        if criteria is not None:
            # 后端/自动化场景：criteria 已传入
            filter_criteria = criteria
            should_apply_filter = criteria.get('apply_filter', False)  # 默认 False，必须显式启用
        else:
            # 回测场景：使用 sc 配置
            filter_criteria = {
                'min_market_cap': sc.MIN_MARKET_CAP, 'max_pe': sc.MAX_PE, 'max_zcfzl': sc.MAX_ZCFZL,
                'min_price': sc.MIN_PRICE, 'max_price': sc.MAX_PRICE, 'include_st': sc.INCLUDE_ST,
                'markets': sc.SELECTOR_MARKETS
            }
            should_apply_filter = sc.ENABLE_FUNDAMENTAL_FILTER

        if should_apply_filter:
            # 2. 预筛选 (利用内存快照，无 SQL)。仅在启用过滤时构建，避免无谓遍历全市场。
            info_map = self._get_optimized_info_map(current_date, market_data)
            predict_codes, _ = self._pre_filter_stocks(all_codes, info_map, 
                                                 apply_filter=True, 
                                                 criteria=filter_criteria)
        else:
            predict_codes = all_codes
        
        if not predict_codes: return signals

        # 3. 批量获取当前因子的最新行
        raw_rows = []
        stock_codes_with_data = []
        feature_names = self._get_model_feature_names()
        if not feature_names:
            return signals
        for code in predict_codes:
            if code in existing_positions:
                continue
            feature_row = self._get_factor_row_array(code, current_date)
            if feature_row is None:
                continue
            raw_rows.append(feature_row)
            stock_codes_with_data.append(code)
        
        if not raw_rows: return signals

        # 4. 批量预测。优先保持 numpy 矩阵，避免每日反复构造 DataFrame。
        X_arr = np.vstack(raw_rows).astype(np.float32, copy=False)

        # 横截面归一化 (与训练时精确匹配逻辑保持一致)
        rank_cols_idx = [i for i, col in enumerate(feature_names) if not fc.TrainingConfig.should_skip_rank(col)]

        if rank_cols_idx and len(X_arr) > 1:
            from scipy.stats import rankdata as _rankdata
            ranked = _rankdata(X_arr[:, rank_cols_idx], method='average', axis=0) / (len(X_arr) + 1)
            X_arr[:, rank_cols_idx] = ranked.astype(np.float32)

        # skip_cols：复用训练集 robust 统计量（仅连续型，二值列保留原值）
        norm_stats = getattr(self, 'norm_stats', None)
        if norm_stats is not None:
            import numpy as _np
            skip_col_stats = norm_stats.get('skip_col_stats')
            if skip_col_stats is not None and skip_col_stats.get('robust_global_idx', _np.array([])).size > 0:
                train_factor_names = norm_stats.get('factor_names', [])
                robust_global_idx  = skip_col_stats['robust_global_idx']
                robust_col_names   = [train_factor_names[i] for i in robust_global_idx
                                      if i < len(train_factor_names)]
                feature_pos = {name: idx for idx, name in enumerate(feature_names)}
                present_robust = [c for c in robust_col_names if c in feature_pos]
                if present_robust:
                    col_to_idx = {train_factor_names[i]: j
                                  for j, i in enumerate(robust_global_idx)
                                  if i < len(train_factor_names)}
                    median    = skip_col_stats['median']
                    iqr       = skip_col_stats['iqr']
                    valid_iqr = skip_col_stats['valid_iqr']
                    for col in present_robust:
                        j = col_to_idx[col]
                        arr_idx = feature_pos[col]
                        if valid_iqr[j]:
                            z = (X_arr[:, arr_idx].astype(float) - median[j]) / iqr[j]
                            X_arr[:, arr_idx] = (1.0 / (1.0 + _np.exp(-_np.clip(z, -10, 10)))).astype(_np.float32)
                        else:
                            # 必须与训练端保持一致；训练端零 IQR 列固定为 0.0。
                            X_arr[:, arr_idx] = 0.0
        X_arr = np.nan_to_num(X_arr, nan=0.5, posinf=1.0, neginf=0.0)
        if getattr(self.model, 'models', None):
            probs = self.model.predict(pd.DataFrame(X_arr, columns=feature_names))
        else:
            probs = self.model.predict(X_arr)
        
        # 5. 生成信号
        candidates = []
        for i, code in enumerate(stock_codes_with_data):
            confidence = float(probs[i] * 100)
            if confidence < effective_min_confidence or code in existing_positions: continue
            
            # 使用 md5 哈希在概率相同时保持排序稳定
            tie_breaker = int(hashlib.md5(code.encode()).hexdigest(), 16) % 1000 / 100000.0
            candidates.append({'code': code, 'score': confidence + tie_breaker, 'prob': probs[i]})
            
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        for cand in candidates[:available_slots]:
            code = cand['code']
            bar = market_data.get_bar(code)
            if bar is None: continue
            
            # 获取 ATR 需要历史数据
            hist_df = market_data[code]
            atr = self._calculate_atr(hist_df)
            signals.append(StrategySignal(
                stock_code=code, signal_type='buy', timestamp=current_date, 
                price=bar['close'], confidence=cand['score'], 
                stop_loss=bar['close'] - sc.ATR_STOP_MULTIPLIER * atr, 
                take_profit=bar['close'] + sc.ATR_TARGET_MULTIPLIER * atr,
                metadata={'strategy': 'ml_factor_integrated', 'prediction': cand['prob']}
            ))
        return signals

    def _precompute_pit_data(self):
        """预加载元数据，消除循环内的 SQL 压力"""
        db_dir = os.path.dirname(DATABASE_PATH)
        conn = sqlite3.connect(DATABASE_PATH)
        for db in ['stock_meta.db', 'stock_finance.db']:
            path = os.path.join(db_dir, db)
            if os.path.exists(path): conn.execute(f"ATTACH DATABASE '{path}' AS {db.split('_')[1].split('.')[0]}")
        try:
            self._meta_map = pd.read_sql_query("SELECT code, code_name AS name FROM meta.stock_basic", conn).set_index('code')['name'].to_dict()
            self._all_finance_df = pd.read_sql_query("""
                SELECT p.code, p.pub_date, p.stat_date, p.epsTTM AS EPSJB, p.totalShare, b.liabilityToAsset AS ZCFZL
                FROM finance.profit_ability p
                LEFT JOIN finance.balance_ability b ON p.code = b.code AND p.stat_date = b.stat_date
                WHERE p.pub_date IS NOT NULL AND p.pub_date != ''
            """, conn)
            num_cols = ['EPSJB', 'totalShare', 'ZCFZL']
            for col in num_cols:
                self._all_finance_df[col] = pd.to_numeric(self._all_finance_df[col], errors='coerce').astype('float32')
            self._all_finance_df = self._all_finance_df.sort_values('pub_date')
            self._finance_history = {}
            for code, group in self._all_finance_df.groupby('code', sort=False):
                dates = group['pub_date'].astype(str).str[:10].to_numpy()
                values = group[num_cols].to_numpy(dtype=np.float32, copy=True)
                self._finance_history[code] = (dates, values)
        finally: conn.close()

    def _get_optimized_info_map(self, target_date: str, market_data: Any) -> Dict[str, Dict]:
        """极速信息映射逻辑，复用预缓存信息"""
        info_map = {}
        
        for code in market_data.keys():
            bar = market_data.get_bar(code)
            if bar is None: continue
            fin_hist = getattr(self, '_finance_history', {}).get(code)
            eps = ts = zcfzl = None
            if fin_hist is not None:
                fin_dates, fin_values = fin_hist
                fin_idx = np.searchsorted(fin_dates, target_date, side='right') - 1
                if fin_idx >= 0:
                    eps = fin_values[fin_idx, 0]
                    ts = fin_values[fin_idx, 1]
                    zcfzl = fin_values[fin_idx, 2]
            # 组装基本面字典供筛选
            raw_close = float(bar.get('raw_close', bar['close']))
            info_map[code] = {
                'name': self._meta_map.get(code, '-'), 'is_st': int(bar.get('is_st', 0)),
                'pe_ratio': (raw_close/eps if eps and eps>0 else None),
                'zcfzl': zcfzl, 'current_price': raw_close,
                'market_cap': (raw_close*ts/1e8 if ts else None)
            }
        return info_map

    def _preload_factor_cache(self):
        """预加载回测所需因子列，并建立按日期二分查找的缓存。"""
        feature_names = self._get_model_feature_names()
        if not feature_names:
            return
        if not os.path.isdir(self.cache_dir):
            return

        active_codes = getattr(self, '_active_stock_codes', set())
        if active_codes:
            files = [
                f'{code}_factors.parquet'
                for code in active_codes
                if os.path.exists(os.path.join(self.cache_dir, f'{code}_factors.parquet'))
            ]
        else:
            files = [f for f in os.listdir(self.cache_dir) if f.endswith('_factors.parquet')]
        if not files:
            return

        scope = "当前回测股票池" if active_codes else "全部缓存"
        print(f"正在预加载因子缓存({scope}): {len(files)} 个 parquet...")

        def _load_one(filename):
            code = filename[:-len('_factors.parquet')]
            path = os.path.join(self.cache_dir, filename)
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(path)
                schema_names = set(pf.schema.names)
                if 'date' not in schema_names:
                    return None
                read_cols = ['date'] + [c for c in feature_names if c in schema_names]
                df = pd.read_parquet(path, columns=read_cols)
                if df.empty or 'date' not in df.columns:
                    return None

                dates = df['date'].astype(str).str[:10].to_numpy()
                order = np.argsort(dates, kind='mergesort')
                if not np.all(order == np.arange(len(order))):
                    df = df.iloc[order].reset_index(drop=True)
                    dates = dates[order]

                matrix = np.full((len(df), len(feature_names)), 0.5, dtype=np.float32)
                for col_idx, col in enumerate(feature_names):
                    if col in df.columns:
                        matrix[:, col_idx] = pd.to_numeric(df[col], errors='coerce').fillna(0.5).to_numpy(dtype=np.float32)
                return code, dates, matrix
            except Exception:
                return None

        from concurrent.futures import ThreadPoolExecutor
        workers = min(8, max(1, len(files)))
        loaded = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for item in executor.map(_load_one, files):
                if item is None:
                    continue
                code, dates, matrix = item
                self._factor_dates_cache[code] = dates
                self._factor_matrix_cache[code] = matrix
                loaded += 1
        print(f"  因子缓存预加载完成: {loaded}/{len(files)}")

    def _get_factor_row_array(self, stock_code: str, current_date: str) -> Optional[np.ndarray]:
        """按股票和日期快速取得 PIT 特征行。"""
        dates = self._factor_dates_cache.get(stock_code)
        matrix = self._factor_matrix_cache.get(stock_code)
        if dates is not None and matrix is not None:
            idx = np.searchsorted(dates, current_date, side='right') - 1
            if idx >= 0:
                return matrix[idx]

        factors = self._get_factors(stock_code, None, current_date)
        if factors is None or factors.empty:
            return None
        row = factors.drop(columns=['date'], errors='ignore').iloc[-1]
        return row.reindex(self._get_model_feature_names()).fillna(0.5).to_numpy(dtype=np.float32)

    def _get_model_feature_names(self) -> List[str]:
        """获取模型需要的特征列，兼容单模型与集成模型。"""
        names = getattr(self.model, 'feature_names', None)
        if names:
            return list(names)
        models = getattr(self.model, 'models', None)
        if models:
            ordered = []
            seen = set()
            for model in models:
                for name in getattr(model, 'feature_names', []) or []:
                    if name not in seen:
                        ordered.append(name)
                        seen.add(name)
            return ordered
        return []

    def _pre_filter_stocks(self, all_codes: List[str], info_map: Dict[str, Dict], apply_filter: bool, criteria: Dict) -> Tuple[List[str], Dict]:
        """
        股票池预筛选逻辑
        
        参数优先级说明：
        - 后端场景：criteria 中的参数由前端传入，None 表示不限制该条件
        - 回测场景：criteria 使用 sc 配置的默认值
        - 自动化场景：criteria 使用 automation_config 优先，未设置则使用 sc 兜底
        """
        if not apply_filter: return all_codes, {}
        passed = []
        
        def _get_val(k, default):
            v = criteria.get(k)
            return v if v is not None else default

        # 预计算允许的市场前缀
        markets_filter = _get_val('markets', [])
        allowed_prefixes = []
        if markets_filter:
            for m in markets_filter:
                p = SUPPORTED_MARKETS.get(m, {}).get('prefixes')
                if p: 
                    allowed_prefixes.extend(p)
                else:
                    # 兼容性处理：支持交易所级别代码 (如 'sh' -> ['60', '68'])
                    for m_info in SUPPORTED_MARKETS.values():
                        if m_info.get('code') == m:
                            allowed_prefixes.extend(m_info.get('prefixes', []))
        allowed_prefixes = tuple(allowed_prefixes) if allowed_prefixes else None
            
        for code in all_codes:
            info = info_map.get(code)
            if not info: continue
            
            # 1. 市场类型筛选
            if allowed_prefixes and not str(code).startswith(allowed_prefixes):
                continue

            # 2. ST股及退市搜索
            # 股票名称来自当前元数据，不具备 PIT 语义，回测中不能用“退”字样
            # 反向污染历史。风险过滤只依赖当日行情中的 is_st。
            if not _get_val('include_st', sc.INCLUDE_ST) and info['is_st'] == 1: continue
            
            # 3. PE 筛选
            pe = info.get('pe_ratio')
            max_pe = _get_val('max_pe', sc.MAX_PE if sc.MAX_PE is not None else float('inf'))
            if pe is not None and (pe <= 0 or pe > max_pe): continue
            
            # 4. 价格筛选
            price = info['current_price']
            if price < _get_val('min_price', sc.MIN_PRICE if sc.MIN_PRICE is not None else 0) or \
               price > _get_val('max_price', sc.MAX_PRICE if sc.MAX_PRICE is not None else float('inf')): continue
            
            # 5. 市值筛选
            mkt_cap = info.get('market_cap')
            if mkt_cap is not None:
                if mkt_cap < _get_val('min_market_cap', 0): continue
            
            # 6. 资产负债率筛选
            zcfzl = info.get('zcfzl')
            if zcfzl is not None:
                if zcfzl > _get_val('max_zcfzl', float('inf')): continue
                
            passed.append(code)
        return passed, {}

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """ATR 计算"""
        if len(data) < period + 1: return 0.0
        try:
            atr_series = talib.ATR(data['high'].values.astype(np.float64), 
                                   data['low'].values.astype(np.float64), 
                                   data['close'].values.astype(np.float64), 
                                   timeperiod=period)
            val = atr_series[-1]
            return float(val) if np.isfinite(val) else 0.0
        except: return 0.0

    def _get_factors(self, stock_code: str, stock_data: pd.DataFrame, current_date: str) -> Optional[pd.DataFrame]:
        """获取 PIT 因子行"""
        cached_factors = self._load_factors_from_cache(stock_code)
        if cached_factors is None or 'date' not in cached_factors.columns: return None
        target_dt = datetime.strptime(current_date, '%Y-%m-%d')
        # 获取不晚于当前日期的最新因子
        factors = cached_factors[pd.to_datetime(cached_factors['date']) <= target_dt]
        if factors.empty: return None
        factors = factors.iloc[[-1]]
        
        feature_names = self._get_model_feature_names()
        if self.model and feature_names:
            # 填充缺失列为0，保持特征列顺序与模型一致
            missing = [f for f in feature_names if f not in factors.columns]
            if missing:
                missing_df = pd.DataFrame(0.5, index=factors.index, columns=missing)
                factors = pd.concat([factors, missing_df], axis=1)
            f_cols = feature_names + (['date'] if 'date' not in feature_names else [])
            factors = factors[[c for c in f_cols if c in factors.columns]]
        return factors

    def _load_factors_from_cache(self, stock_code: str) -> Optional[pd.DataFrame]:
        """从磁盘加载 parquet 文件并维护内存缓存"""
        if stock_code in self._factors_cache: return self._factors_cache[stock_code]
        cache_file = os.path.join(self.cache_dir, f'{stock_code}_factors.parquet')
        if not os.path.exists(cache_file): return None
        try:
            factors = pd.read_parquet(cache_file)
            # 容量上限提升至 10,000，足以容纳全市场股票。只有在内存极度受限且股票池巨大时才会触发清理。
            if len(self._factors_cache) >= 10000:
                # 简单 FIFO 淘汰
                first_key = next(iter(self._factors_cache))
                self._factors_cache.pop(first_key)
            self._factors_cache[stock_code] = factors
            return factors
        except: return None

    def select_for_live(self,
                        db_path: str,
                        top_n: int = 10,
                        lookback_days: int = 500,
                        criteria: Optional[Dict] = None) -> List[Dict]:
        """
        实盘选股入口，完全复用 generate_signals 逻辑，保证与回测一致。

        返回列表，每项包含：
            stock_code, confidence, current_price, stop_loss, take_profit
        """
        import sqlite3
        from datetime import datetime, timedelta

        today = datetime.now().strftime('%Y-%m-%d')

        # --- 构造轻量 LiveMarketData 适配器 ---
        class LiveMarketData:
            """从数据库读取最新行情，提供与回测 MarketSnapshot 相同的接口"""
            def __init__(self, db_path: str, lookback_days: int):
                self._db_path = db_path
                self._lookback_days = lookback_days
                self._cache: Dict[str, pd.DataFrame] = {}
                self._bar_cache: Dict[str, Optional[Dict]] = {}
                self._codes: Optional[List[str]] = None

            def _load(self, code: str) -> Optional[pd.DataFrame]:
                if code in self._cache:
                    return self._cache[code]
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=self._lookback_days)).strftime('%Y-%m-%d')
                try:
                    conn = sqlite3.connect(self._db_path)
                    df = pd.read_sql_query(
                        """SELECT k.date, k.open, k.high, k.low, k.close, k.volume,
                                  k.amount, k.turnover_rate, k.is_st, a.fore_adjust_factor
                           FROM daily_data k
                           LEFT JOIN adjust_factor a ON k.code = a.code AND k.date = a.date
                           WHERE k.code = ? AND k.date >= ? AND k.date <= ?
                           ORDER BY k.date ASC""",
                        conn, params=(code, start_date, end_date)
                    )
                    conn.close()
                    if df.empty or len(df) < 35:
                        return None
                    self._cache[code] = df
                    return df
                except Exception:
                    return None

            def get_bar(self, code: str) -> Optional[Dict]:
                if code in self._bar_cache:
                    return self._bar_cache[code]
                df = self._load(code)
                if df is None:
                    self._bar_cache[code] = None
                    return None
                row = df.iloc[-1]
                bar = {c: row[c] for c in df.columns}
                # is_st 已从数据库读取，无需覆盖
                self._bar_cache[code] = bar
                return bar

            def __getitem__(self, code: str) -> Optional[pd.DataFrame]:
                return self._load(code)

            def keys(self) -> List[str]:
                if self._codes is not None:
                    return self._codes
                try:
                    conn = sqlite3.connect(self._db_path)
                    rows = pd.read_sql_query(
                        "SELECT DISTINCT code FROM daily_data WHERE date >= date('now', '-30 days')",
                        conn
                    )
                    conn.close()
                    self._codes = rows['code'].tolist()
                except Exception:
                    self._codes = []
                return self._codes

        market_data = LiveMarketData(db_path, lookback_days)

        # 复用 generate_signals，portfolio_state 传空持仓、足够的 slots
        portfolio_state = {'positions': {}, 'available_slots': top_n}
        
        # 调用信号生成逻辑，显式传递 criteria
        # 内部会优先使用传入的 criteria 进行筛选，且不使用 sc 中的默认覆盖值
        signals = self.generate_signals(today, market_data, portfolio_state, 
                                        criteria=criteria)

        # 转换为与 select_stocks 兼容的字典格式
        results = []
        # signals 已经是经过排序和 top_n 截取的了 (通过 available_slots)
        for sig in signals:
            results.append({
                'stock_code': sig.stock_code,
                'confidence': sig.confidence,
                'current_price': sig.price,
                'stop_loss': sig.stop_loss,
                'take_profit': sig.take_profit,
            })
        return results

    def cleanup(self):
        """清理缓存"""
        self._factors_cache.clear()
        print(f"策略清理完成: {self.name}")
