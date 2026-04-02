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
from config import DATABASE_PATH, SUPPORTED_MARKETS

class MLFactorBacktestStrategy(BaseStrategy):
    """ML因子回测策略
    
    回测时完全依赖训练阶段生成的因子缓存。
    """
    
    def __init__(self,
                 model_path: str,
                 min_confidence: float = sc.ML_FACTOR_MIN_CONFIDENCE,
                 use_cache: bool = True,
                 cache_dir: str = None,
                 name: str = "ML因子策略"):
        """初始化策略"""
        super().__init__(name)
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.use_cache = use_cache
        
        if cache_dir is None:
            cache_dir = fc.TrainingConfig.CACHE_DIR
        self.cache_dir = cache_dir
        
        self.model = None
        self._factors_cache = {}  # 内存缓存，用于存放 parquet 加载全量因子数据
        self._warned_stocks = set()
    
    def initialize(self, **kwargs):
        """初始化策略 - 预定全量缓存以极大提升速度"""
        super().initialize(**kwargs)
        
        # 0. 预加载所有 PIT 筛选所需的基本面指标 (性能优化核心)
        print(f"正在进行 PIT 数据全量预缓存...")
        self._precompute_pit_data()
        
        # 智能加载模型
        from core.factors.ml_factor_model import MLFactorModel, EnsembleFactorModel
        def _load_smart_model(target_path):
            if os.path.isdir(target_path):
                pkls = [os.path.join(target_path, f) for f in os.listdir(target_path) if f.endswith('.pkl')]
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
        
        print(f"策略初始化完成: {self.name} (已启用 PIT 预缓存 ✓)")
    
    def generate_signals(self,
                        current_date: str,
                        market_data: Any,
                        portfolio_state: Dict[str, Any]) -> List[StrategySignal]:
        """生成交易信号 (极速版)"""
        signals = []
        existing_positions = portfolio_state.get('positions', {})
        available_slots = sc.MAX_POSITIONS - len(existing_positions)
        if available_slots <= 0: return signals

        # 1. 获取所有股票列表
        if not hasattr(self, '_all_db_codes'):
            # 兼容旧版本：_factors.parquet 后缀
            self._all_db_codes = [f[:-16] for f in os.listdir(self.cache_dir) if f.endswith('_factors.parquet')]
        all_codes = self._all_db_codes
        
        # 2. 预筛选 (利用内存快照，无 SQL)
        info_map = self._get_optimized_info_map(current_date, market_data)
        
        # 按照基本面指标筛选 predict_codes (与原始 select_stocks 逻辑严格一致)
        # 优先使用实时传入的 criteria
        filter_criteria = getattr(self, '_custom_criteria', {
            'min_market_cap': sc.MIN_MARKET_CAP, 'max_pe': sc.MAX_PE, 'max_zcfzl': sc.MAX_ZCFZL,
            'min_price': sc.MIN_PRICE, 'max_price': sc.MAX_PRICE, 'include_st': sc.INCLUDE_ST,
            'markets': sc.SELECTOR_MARKETS
        })
        
        predict_codes, _ = self._pre_filter_stocks(all_codes, info_map, 
                                                 apply_filter=sc.ENABLE_FUNDAMENTAL_FILTER, 
                                                 criteria=filter_criteria)
        
        if not predict_codes: return signals

        # 3. 批量获取当前因子的最新行
        raw_rows = []
        stock_codes_with_data = []
        for code in predict_codes:
            # 此处可能触发磁盘读取，如果内存不足 1万只股票，会有少量淘汰
            factors = self._get_factors(code, None, current_date)
            if factors is not None and not factors.empty:
                latest_row = factors.iloc[-1:]
                if 'date' in latest_row.columns:
                    # 严格日期限制，不取未来日期
                    if str(latest_row['date'].iloc[0])[:10] > current_date: continue
                    feature_row = latest_row.drop(columns=['date']).values[0]
                else: continue
                raw_rows.append(feature_row)
                stock_codes_with_data.append(code)
        
        if not raw_rows: return signals

        # 4. 批量预测
        all_X = pd.DataFrame(raw_rows, columns=self.model.feature_names)
        
        # 横截面归一化 (与训练时精确匹配逻辑保持一致)
        _sentiment_exact = {
            'up_ratio', 'strong_up_ratio', 'down_ratio',
            'limit_up_ratio', 'limit_down_ratio', 'mean_return',
            'total_volume', 'adv_vol_ratio', 'breadth_ma20',
            'market_type',
        }
        rank_cols = [col for col in all_X.columns if col not in _sentiment_exact]
        if rank_cols and len(all_X) > 1:
            from scipy.stats import rankdata as _rankdata
            arr = all_X[rank_cols].values
            ranked = _rankdata(arr, method='average', axis=0) / (len(all_X) + 1)
            all_X[rank_cols] = ranked.astype(np.float32)

        probs = self.model.predict(all_X.fillna(0.5))
        
        # 5. 生成信号
        candidates = []
        for i, code in enumerate(stock_codes_with_data):
            confidence = float(probs[i] * 100)
            if confidence < self.min_confidence or code in existing_positions: continue
            
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
        finally: conn.close()

    def _get_optimized_info_map(self, target_date: str, market_data: Any) -> Dict[str, Dict]:
        """极速信息映射逻辑，复用预缓存信息"""
        info_map = {}
        # 筛选截止日期前的最新财务数据
        actual_fin = self._all_finance_df[self._all_finance_df['pub_date'] <= target_date]
        latest_fin = actual_fin.groupby('code').last()
        fin_map = latest_fin.to_dict('index')
        
        for code in market_data.keys():
            bar = market_data.get_bar(code)
            if bar is None: continue
            fin = fin_map.get(code, {})
            eps, ts = fin.get('EPSJB'), fin.get('totalShare')
            # 组装基本面字典供筛选
            info_map[code] = {
                'name': self._meta_map.get(code, '-'), 'is_st': int(bar.get('is_st', 0)),
                'pe_ratio': (bar['close']/eps if eps and eps>0 else None),
                'zcfzl': fin.get('ZCFZL'), 'current_price': bar['close'],
                'market_cap': (bar['close']*ts/1e8 if ts else None)
            }
        return info_map

    def _pre_filter_stocks(self, all_codes: List[str], info_map: Dict[str, Dict], apply_filter: bool, criteria: Dict) -> Tuple[List[str], Dict]:
        """股票池预筛选，保持逻辑完全一致"""
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
                if p: allowed_prefixes.extend(p)
        allowed_prefixes = tuple(allowed_prefixes) if allowed_prefixes else None
            
        for code in all_codes:
            info = info_map.get(code)
            if not info: continue
            
            # 1. 市场类型筛选
            if allowed_prefixes and not str(code).startswith(allowed_prefixes):
                continue

            # 2. ST股及退市搜索
            if not _get_val('include_st', True) and (info['is_st'] == 1 or '退' in info['name']): continue
            
            # 3. PE 筛选
            pe = info.get('pe_ratio')
            max_pe = _get_val('max_pe', float('inf'))
            if pe is not None and (pe <= 0 or pe > max_pe): continue
            
            # 4. 价格筛选
            price = info['current_price']
            if price < _get_val('min_price', 0) or price > _get_val('max_price', float('inf')): continue
            
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
        
        if self.model and self.model.feature_names:
            # 填充缺失列为0，保持特征列顺序与模型一致
            missing = [f for f in self.model.feature_names if f not in factors.columns]
            if missing:
                missing_df = pd.DataFrame(0.0, index=factors.index, columns=missing)
                factors = pd.concat([factors, missing_df], axis=1)
            f_cols = self.model.feature_names + (['date'] if 'date' not in self.model.feature_names else [])
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

        self._custom_criteria = criteria

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
                                  k.amount, k.turnover_rate, a.fore_adjust_factor
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
                bar['is_st'] = 0  # 实盘时 ST 信息由 _pre_filter_stocks 处理
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
        # 临时覆盖 min_confidence 为 0，让所有信号通过，由调用方自行过滤
        orig_min = self.min_confidence
        self.min_confidence = 0.0
        signals = self.generate_signals(today, market_data, portfolio_state)
        self.min_confidence = orig_min

        # 转换为与 select_stocks 兼容的字典格式
        results = []
        for sig in sorted(signals, key=lambda s: s.confidence, reverse=True)[:top_n]:
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
