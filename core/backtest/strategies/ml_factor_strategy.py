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
from typing import Dict, List, Any, Optional, Tuple
from core.backtest.strategy import BaseStrategy, StrategySignal
from core.factors.ml_factor_model import MLFactorModel
import config.strategy_config as sc
import config.factor_config as fc
from config import DATABASE_PATH, SUPPORTED_MARKETS
class MLFactorBacktestStrategy(BaseStrategy):
    """ML因子回测策略
    
    回测时完全依赖训练阶段生成的因子缓存。
    缓存中应包含模型需要的所有特征（含特征工程生成的特征）。
    如果缓存缺失某只股票，该股票将被跳过。
    如果缓存中缺少部分特征列，缺失列将被填充为0。
    """
    
    def __init__(self,
                 model_path: str,
                 min_confidence: float = sc.ML_FACTOR_MIN_CONFIDENCE,
                 use_cache: bool = True,
                 cache_dir: str = None,
                 name: str = "ML因子策略"):
        """
        初始化策略
        
        参数:
            model_path: 模型文件路径
            min_confidence: 最小置信度阈值
            use_cache: 是否使用因子缓存
            cache_dir: 缓存目录路径
            name: 策略名称
        """
        super().__init__(name)
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.use_cache = use_cache
        
        # 设置缓存目录
        if cache_dir is None:
            cache_dir = fc.TrainingConfig.CACHE_DIR
        self.cache_dir = cache_dir
        
        self.model = None
        self._factors_cache = {}  # 内存缓存
        self._warned_stocks = set()  # 已警告过的股票，避免重复日志
    
    def initialize(self, **kwargs):
        """初始化策略"""
        super().initialize(**kwargs)
        
        # 智能加载模型 (支持单模型和集成模型)
        def _load_smart_model(target_path):
            from core.factors.ml_factor_model import MLFactorModel, EnsembleFactorModel
            
            if os.path.isdir(target_path):
                xgb_p = os.path.join(target_path, 'xgboost_factor_model.pkl')
                lgb_p = os.path.join(target_path, 'lightgbm_factor_model.pkl')
                if os.path.exists(xgb_p) and os.path.exists(lgb_p):
                    m1 = MLFactorModel(model_type='xgboost'); m1.load_model(xgb_p)
                    m2 = MLFactorModel(model_type='lightgbm'); m2.load_model(lgb_p)
                    return EnsembleFactorModel(models=[m1, m2], weights=[0.5, 0.5])
                
                # 寻找目录下最新的 pkl
                pkls = [os.path.join(target_path, f) for f in os.listdir(target_path) if f.endswith('.pkl')]
                if pkls:
                    latest_pkl = sorted(pkls, key=os.path.getmtime)[-1]
                    return _load_smart_model(latest_pkl)
                return None
            
            if not os.path.exists(target_path): return None
            
            try:
                # 尝试加载为集成模型
                return EnsembleFactorModel.load_model(target_path)
            except Exception as e:
                # 降级为单模型，但记录原因（如果不是因为结构不匹配导致的加载失败）
                if "pickle" in str(e).lower() or "not a directory" in str(e).lower():
                    print(f"  [DEBUG] 集成模型加载失败，将尝试加载为单模型: {e}")
                
                try:
                    m = MLFactorModel()
                    m.load_model(target_path)
                    return m
                except Exception as e2:
                    print(f"  [ERROR] 单模型加载也失败: {e2}")
                    return None

        self.model = _load_smart_model(self.model_path)
        
        if self.model is None:
            raise ValueError(f"无法从 {self.model_path} 加载模型")
        
        if not getattr(self.model, 'is_trained', False):
            raise ValueError("加载的模型未经过训练")
        
        # 检测缓存状态
        cache_status = "未找到"
        if self.use_cache and os.path.exists(self.cache_dir):
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.parquet')]
            cache_status = f"{len(cache_files)} 只股票"
            
            # 检查第一个缓存文件的特征是否与模型匹配
            if cache_files:
                try:
                    sample = pd.read_parquet(os.path.join(self.cache_dir, cache_files[0]))
                    missing = [f for f in self.model.feature_names if f not in sample.columns]
                    if missing:
                        print(f"  警告: 缓存缺少 {len(missing)} 个模型特征 (如 {missing[:3]})，将用0填充")
                    else:
                        print(f"  缓存特征与模型完全匹配 ✓")
                except Exception as e:
                    print(f"  警告: 无法验证缓存特征: {e}")
        
        print(f"策略初始化完成: {self.name}")
        print(f"  模型: {self.model_path}")
        print(f"  模型特征数: {len(self.model.feature_names)}")
        print(f"  最小置信度: {self.min_confidence}%")
        print(f"  使用缓存: {self.use_cache}")
        if self.use_cache:
            print(f"  缓存目录: {self.cache_dir}")
            print(f"  缓存状态: {cache_status}")
        print(f"  核心逻辑: 已集成 select_stocks 逻辑 (包含横截面归一化)")
    
    def generate_signals(self,
                        current_date: str,
                        market_data: Dict[str, pd.DataFrame],
                        portfolio_state: Dict[str, Any]) -> List[StrategySignal]:
        """
        生成交易信号 (与 select_stocks.py 逻辑完全一致)
        
        流程:
        1. 获取所有可用特征数据的股票列表
        2. 批量加载当前日期的特征行
        3. 执行横截面 Z-Score 归一化 (关键: 防止分布偏移)
        4. 模型批量预测
        5. 应用基础面和置信度过滤
        6. 排序并输出 top_n
        """
        signals = []
        
        # 获取当前持仓和可用头寸
        existing_positions = portfolio_state.get('positions', {})
        current_count = len(existing_positions)
        available_slots = sc.MAX_POSITIONS - current_count
        
        # 即使 available_slots <= 0，我们也可能需要计算信号以供记录或逻辑处理
        # 但为了效率，通常在无头寸时快速退出
        if available_slots <= 0:
            return signals
        # 1. 获取所有股票代码 (从缓存目录获取)
        if not hasattr(self, '_all_db_codes'):
            # 修复问题17：使用更健壮的文件名解析方式
            suffix = '_factors.parquet'
            self._all_db_codes = [f[:-len(suffix)] for f in os.listdir(self.cache_dir) if f.endswith(suffix)]
        all_codes = self._all_db_codes
        
        # 2. 提前获取基本面信息并预筛选 (提升效率 & 对齐 select_stocks.py)
        # 获取该交易日切片的基本面信息
        info_map = self._get_stock_info_map_pit(current_date)
        
        # 执行条件筛选 (PE/市值/价格/ST/市场)
        predict_codes, _ = self._pre_filter_stocks(
            all_codes, info_map, 
            apply_filter=sc.ENABLE_FUNDAMENTAL_FILTER,
            criteria={
                'min_market_cap': sc.MIN_MARKET_CAP,
                'max_pe': sc.MAX_PE,
                'max_zcfzl': sc.MAX_ZCFZL,
                'min_price': sc.MIN_PRICE,
                'max_price': sc.MAX_PRICE,
                'include_st': sc.INCLUDE_ST,
                'markets': sc.SELECTOR_MARKETS
            }
        )
        
        if not predict_codes:
            return signals
        # 3. 批量获取当前日期的因子数据 (仅针对通过筛选的股票)
        raw_rows = []
        stock_codes_with_data = []
        
        for code in predict_codes:
            factors = self._get_factors(code, None, current_date)
            if factors is not None and not factors.empty:
                # 修复问题5: 严格验证日期对齐
                latest_row = factors.iloc[[-1]].copy()
                row_date = None
                if 'date' in latest_row.columns:
                    row_date = latest_row['date'].iloc[0]
                    row_date_str = str(row_date)[:10]
                    
                    # 仅拒绝“未来日期”的因子；若是历史最近日期，允许使用，避免因缓存更新节奏导致股票池被异常压缩
                    if row_date_str > current_date:
                        if code not in self._warned_stocks:
                            self._warned_stocks.add(code)
                        continue
                    
                    latest_row = latest_row.drop(columns=['date'])
                else:
                    # 没有日期列，无法验证对齐，跳过
                    continue
                
                raw_rows.append(latest_row)
                stock_codes_with_data.append(code)
        
        if not raw_rows:
            return signals
        # 4. 执行横截面分位数排名 (关键修复：确保回测特征分布与训练完全对齐)
        all_X = pd.concat(raw_rows, axis=0, ignore_index=True)
        all_X = all_X.astype(np.float64)
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # 修复问题7: 识别需要进行横截面排名的因子索引 (同步 train_ml_model 逻辑)
            # 排除市场情绪因子，因为它们在同一天对所有股票相同
            sentiment_keys = ['up_ratio', 'down_ratio', 'mean_return', 'adv_vol', 'breadth_', 'sentiment_', 'mkt_', 'market_type']
            rank_cols = [col for col in all_X.columns if not any(k in col.lower() for k in sentiment_keys)]
            
            if rank_cols and len(all_X) > 1:
                # 仅对通过筛选的股票池进行当日排名
                all_X[rank_cols] = all_X[rank_cols].rank(pct=True).fillna(0.5)
        
        # 填充剩余 NaN
        all_X = all_X.fillna(0.5)
        
        try:
            probs = self.model.predict(all_X)
        except Exception as e:
            import traceback
            print(f"  [致命错误] 批量预测发生异常: {e}")
            traceback.print_exc()
            return signals
        # 5. 构造候选者并排序
        candidates = []
        for i, code in enumerate(stock_codes_with_data):
            prob = probs[i]
            confidence = float(prob * 100)
            
            # 置信度过滤
            if confidence < self.min_confidence:
                continue
            
            # 引入哈希扰动以确保排序稳定性
            tie_breaker = int(hashlib.md5(str(code).encode()).hexdigest(), 16) % 1000 / 100000.0
            
            candidates.append({
                'code': code,
                'prob': prob,
                'confidence': confidence,
                'sort_score': confidence + tie_breaker
            })
            
        # 按得分排序
        candidates.sort(key=lambda x: x['sort_score'], reverse=True)
        
        # 7. 生成信号 (仅选取前 top_n)
        for cand in candidates:
            code = cand['code']
            
            # 过滤已持有的
            if code in existing_positions:
                continue
                
            # 获取当日行情 bar (单行 Series)
            bar = market_data.get_bar(code)
            if bar is None:
                continue
            
            current_price = bar['close']
            
            # 计算 ATR 止损止盈 - 需要历史 DataFrame 而非单行 Series
            hist_df = market_data[code]  # 通过 __getitem__ 获取历史 DataFrame
            atr = self._calculate_atr(hist_df, period=fc.FactorConfig.ATR_PERIOD)
            stop_loss = current_price - sc.ATR_STOP_MULTIPLIER * atr
            take_profit = current_price + sc.ATR_TARGET_MULTIPLIER * atr
            
            signal = StrategySignal(
                stock_code=code,
                signal_type='buy',
                timestamp=current_date,
                price=current_price,
                confidence=cand['confidence'],
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'strategy': 'ml_factor_integrated',
                    'prediction': cand['prob'],
                    'is_integrated': True
                }
            )
            signals.append(signal)
            
            if len(signals) >= available_slots:
                break
                
        return signals
    def _get_stock_info_map_pit(self, target_date: str) -> Dict[str, Dict]:
        """
        获取点在时间 (Point-In-Time) 的股票基本信息
        """
        def safe_float(val):
            try:
                if val is None or val == '' or str(val).lower() == 'none': return None
                v = float(val)
                return v if np.isfinite(v) else None
            except: return None
        # 确保数据库路径存在
        if not os.path.exists(DATABASE_PATH):
            raise FileNotFoundError(f"主数据库不存在: {DATABASE_PATH}")

        # 利用 DataHandler 或直接连接，这里保持直接连接但优化 SQL 鲁棒性
        db_dir = os.path.dirname(DATABASE_PATH)
        conn = sqlite3.connect(DATABASE_PATH, timeout=30)
        
        meta_db = os.path.join(db_dir, 'stock_meta.db')
        finance_db = os.path.join(db_dir, 'stock_finance.db')
        
        if os.path.exists(meta_db):
            conn.execute(f"ATTACH DATABASE '{meta_db}' AS meta")
        if os.path.exists(finance_db):
            conn.execute(f"ATTACH DATABASE '{finance_db}' AS finance")

        # 1. 基础信息 from meta (直接查询当前标准的 stock_basic 表)
        meta_df = pd.read_sql_query("SELECT code, code_name AS name FROM meta.stock_basic", conn)
        
        # 2. PIT 财务数据 (取 target_date 之前最新的报告)
        # 精确 PIT 逻辑：必须在 target_date 之前通过 pub_date 发布
        finance_df = pd.read_sql_query(f"""
            SELECT p.code, p.epsTTM AS EPSJB, p.totalShare, b.liabilityToAsset AS ZCFZL
            FROM finance.profit_ability p
            LEFT JOIN finance.balance_ability b ON p.code = b.code AND p.stat_date = b.stat_date
            INNER JOIN (
                SELECT code, MAX(pub_date) AS max_date
                FROM finance.profit_ability
                WHERE pub_date IS NOT NULL AND pub_date != '' AND pub_date <= '{target_date}'
                GROUP BY code
            ) latest ON p.code = latest.code AND p.pub_date = latest.max_date
        """, conn)
        
        # 3. PIT 价格 & 动态快照 (is_st, pbMRQ)
        price_df = pd.read_sql_query(f"""
            SELECT code, close, pbMRQ, is_st
            FROM daily_data
            WHERE (code, date) IN (
                SELECT code, MAX(date) FROM daily_data 
                WHERE date <= '{target_date}' 
                GROUP BY code
            )
        """, conn)
        
        conn.close()
        # 构建映射
        fin_map = {str(r.code): r for r in finance_df.itertuples()}
        prc_map = {str(r.code): r for r in price_df.itertuples()}
        
        info_map = {}
        # 以 price_df 为主，因为只有有行情的股票才能回测
        for r in price_df.itertuples():
            code = str(r.code)
            fin = fin_map.get(code)
            # 根据 code 获取对应名称
            meta_rows = meta_df[meta_df['code'] == code]
            name = meta_rows['name'].iloc[0] if not meta_rows.empty else '-'
            
            close = r.close
            eps = getattr(fin, 'EPSJB', None)
            total_share = getattr(fin, 'totalShare', None)
            
            pe = close / eps if close and eps and eps > 0 else None
            # market_cap 以 “亿” 为单位
            mcap = (close * total_share / 1e8) if close and total_share else None
            
            # 状态标记
            is_st = getattr(r, 'is_st', 0)
            
            info_map[code] = {
                'name': name,
                'is_st': int(is_st or 0),
                'pe_ratio': pe,
                'pb_ratio': getattr(r, 'pbMRQ', None), # Baostock 提供的动态 PB
                'zcfzl': getattr(fin, 'ZCFZL', None),
                'current_price': close,
                'market_cap': mcap
            }
        return info_map
    def _pre_filter_stocks(self, all_codes: List[str], info_map: Dict[str, Dict], apply_filter: bool, criteria: Dict) -> Tuple[List[str], Dict]:
        """
        副本自 select_stocks.py: 根据 criteria 过滤股票
        """
        if not apply_filter:
            return all_codes, {}
            
        min_market_cap = criteria.get('min_market_cap', 0) or 0
        max_pe = criteria.get('max_pe') or float('inf')
        min_price = criteria.get('min_price', 0) or 0
        max_price = criteria.get('max_price') or float('inf')
        include_st = criteria.get('include_st', True)
        markets = criteria.get('markets')
        max_zcfzl = criteria.get('max_zcfzl')
        allowed_prefixes = []
        if markets:
            for m in markets:
                p = SUPPORTED_MARKETS.get(m, {}).get('prefixes')
                if p: allowed_prefixes.extend(p)
        allowed_prefixes = tuple(allowed_prefixes)
        passed = []
        for code in all_codes:
            info = info_map.get(code)
            if not info: continue
            
            # 1. 市场
            if allowed_prefixes and not code.startswith(allowed_prefixes): continue
            # 2. ST
            if not include_st and (info.get('is_st') == 1 or '退' in str(info.get('name', ''))): continue
            # 3. PE
            pe = info.get('pe_ratio')
            if pe is not None and (pe <= 0 or pe > max_pe): continue
            # 4. 价格
            price = info.get('current_price')
            if price is not None and (price < min_price or price > max_price): continue
            # 5. 负债率
            if max_zcfzl is not None:
                zcfzl = info.get('zcfzl')
                if zcfzl is not None and zcfzl > max_zcfzl: continue
            
            passed.append(code)
        return passed, {}
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        使用 talib 计算 ATR，确保与训练逻辑一致
        
        参数:
            data: 股票数据 DataFrame
            period: 周期
            
        返回:
            ATR 值
        """
        if len(data) < period + 1:
            return 0.0
            
        try:
            atr_series = talib.ATR(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                timeperiod=period
            )
            val = atr_series[-1]
            return float(val) if np.isfinite(val) else 0.0
        except Exception as e:
            # 记录异常但返回默认值，避免中断整个回测
            if len(data) > period:
                print(f"  警告: 计算 ATR 失败 ({len(data)} 行数据): {e}")
            return 0.0
    
    def _get_factors(self, stock_code: str, stock_data: pd.DataFrame, current_date: str) -> Optional[pd.DataFrame]:
        """
        从缓存获取因子，不做任何实时计算
        
        参数:
            stock_code: 股票代码
            stock_data: 股票数据（仅用于确定行数以截取缓存）
            current_date: 当前日期
        
        返回:
            因子DataFrame，或None（如果缓存不存在）
        """
        # 从缓存加载
        if not self.use_cache:
            # 缓存被禁用，无法获取因子
            return None
        
        cached_factors = self._load_factors_from_cache(stock_code)
        if cached_factors is None:
            # 没有缓存，跳过该股票
            return None
        
        # 1. 尝试使用日期对齐（最准确，推荐）
        if 'date' in cached_factors.columns:
            # 确保日期列为 datetime 类型
            if not pd.api.types.is_datetime64_any_dtype(cached_factors['date']):
                cached_factors['date'] = pd.to_datetime(cached_factors['date'])
            
            target_dt = pd.Timestamp(current_date)
            
            # 优先精确匹配当前日期，避免使用过期因子
            exact_match = cached_factors[cached_factors['date'] == target_dt]
            if not exact_match.empty:
                factors = exact_match.copy()
            else:
                # 精确日期不存在时，退回到取 <= 当前日期的最近一行
                factors = cached_factors[cached_factors['date'] <= target_dt].copy()
                if factors.empty:
                    return None
                factors = factors.iloc[[-1]]  # 只取最近的一行
        
        # 2. 如果缓存中没有日期列，则尝试使用行号对齐（不推荐，极易出错）
        else:
            raise ValueError("缓存中没有日期列，无法对齐")
        
        # 确保所有模型需要的特征列都存在
        if self.model and self.model.feature_names:
            missing_features = [f for f in self.model.feature_names if f not in factors.columns]
            if missing_features:
                # 用0填充缺失特征（与训练时的填充策略一致）
                if stock_code not in self._warned_stocks:
                    self._warned_stocks.add(stock_code)
                    if len(self._warned_stocks) <= 3:  # 只对前3只股票打印警告
                        print(f"  提示: {stock_code} 缓存缺少 {len(missing_features)} 个特征，用0填充")
                
                missing_df = pd.DataFrame(0.0, index=factors.index, columns=missing_features)
                factors = pd.concat([factors, missing_df], axis=1)
            
            # 只保留模型需要的特征，并包含日期列（用于验证对齐）
            feature_cols = self.model.feature_names.copy()
            if 'date' in factors.columns and 'date' not in feature_cols:
                feature_cols.append('date')
            
            available = [f for f in feature_cols if f in factors.columns]
            if len(available) == 0:
                return None
            factors = factors[available]
        
        return factors
    
    def _load_factors_from_cache(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        从缓存加载因子
        
        参数:
            stock_code: 股票代码
        
        返回:
            因子DataFrame或None
        """
        # 检查内存缓存
        if stock_code in self._factors_cache:
            return self._factors_cache[stock_code]
        
        # 从文件加载
        cache_file = os.path.join(self.cache_dir, f'{stock_code}_factors.parquet')
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            factors = pd.read_parquet(cache_file)
            
            # 缓存到内存
            self._factors_cache[stock_code] = factors
            
            return factors
            
        except Exception as e:
            print(f"  错误: 无法加载股票 {stock_code} 的缓存文件 {cache_file}: {e}")
            return None
    
    def cleanup(self):
        """清理资源"""
        self.model = None
        self._factors_cache.clear()
        self._warned_stocks.clear()
        print(f"策略清理完成: {self.name}")
