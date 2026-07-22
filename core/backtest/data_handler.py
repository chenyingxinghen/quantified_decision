"""
数据处理器

负责数据加载、预处理和缓存
"""
import os
import sqlite3
import pandas as pd
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


_PRICE_COLUMNS = ('open', 'high', 'low', 'close', 'preclose')


def _prepare_adjusted_stock_data(
    stock_df: pd.DataFrame,
    prior_fore_factor: float = None,
    prior_back_factor: float = None,
) -> pd.DataFrame:
    """Build one continuous, corporate-action-adjusted price series.

    聚源 adjust_factor 只在公司行为日有值，有效因子需在事件后前向填充、
    且在首个已知因子前向后填充。保留 raw_* 原始价格供历史时点的价格/市值筛选，
    标准 OHLC 列替换为 ``raw_price * fore_adjust_factor``，使买卖价格在分红送转
    后可比。该函数与数据源无关（JYDB 版复用）。
    """
    stock_df = stock_df.sort_values('date').reset_index(drop=True).copy()
    fore = pd.to_numeric(stock_df.get('fore_adjust_factor'), errors='coerce')
    back = pd.to_numeric(stock_df.get('back_adjust_factor'), errors='coerce')
    if prior_fore_factor is not None and len(fore) > 0 and pd.isna(fore.iloc[0]):
        fore.iloc[0] = prior_fore_factor
    if prior_back_factor is not None and len(back) > 0 and pd.isna(back.iloc[0]):
        back.iloc[0] = prior_back_factor
    stock_df['fore_adjust_factor'] = fore.ffill().bfill().fillna(1.0)
    stock_df['back_adjust_factor'] = back.ffill().bfill().fillna(1.0)

    for col in _PRICE_COLUMNS:
        raw = pd.to_numeric(stock_df[col], errors='coerce')
        stock_df[f'raw_{col}'] = raw
        adjusted = raw * stock_df['fore_adjust_factor']
        stock_df[col] = adjusted
        stock_df[f'adj_{col}'] = adjusted

    return stock_df


def _load_stock_batch(args):
    """多进程加载股票数据"""
    db_path, stock_codes, start_date, end_date = args
    
    
    conn = sqlite3.connect(db_path)
    # 优化: 关联其他数据库
    db_dir = os.path.dirname(db_path)
    meta_db = os.path.join(db_dir, 'stock_meta.db')
    if os.path.exists(meta_db):
        conn.execute(f"ATTACH DATABASE '{meta_db}' AS meta")
    
    placeholders = ','.join(['?' for _ in stock_codes])
    query = f'''
        SELECT d.code, d.date, d.open, d.high, d.low, d.close, d.volume, d.amount, d.turnover_rate,
               IFNULL(d.is_st, 0) as is_st
        FROM daily_data d
        WHERE d.code IN ({placeholders}) AND d.date >= ? AND d.date <= ?
        ORDER BY d.code, d.date
    '''
    
    params = stock_codes + [start_date, end_date]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    # 按股票分组 并进行 float32 转换 (节省内存)
    result = {}
    for code, stock_df in df.groupby('code', sort=False):
        stock_df = stock_df.sort_values('date').reset_index(drop=True)

        # 强制转换为 float32 (除了 date 和 code)
        numeric_cols = stock_df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [c for c in numeric_cols if c not in ['code', 'date']]
        stock_df[numeric_cols] = stock_df[numeric_cols].astype('float32', copy=False)

        if len(stock_df) >= 30:
            result[code] = stock_df
    
    return result


class DataHandler:
    """数据处理器"""
    
    def __init__(self, db_path: str):
        """
        初始化数据处理器
        
        参数:
            db_path: 数据库路径
        """
        self.db_path = db_path
        # 修复问题11：不再持有全局连接，改为每次查询时创建新连接（线程安全）
        # SQLite 连接不能跨线程，FastAPI 多线程环境下会引发 ProgrammingError
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._date_index: Dict[str, Dict[str, int]] = {}
        self._daily_bars: Dict[str, set] = {} # 每日活跃股票代码: date -> set(code, ...)
        self._all_trading_dates: List[str] = []
    
    def _get_connection(self):
        """获取新的数据库连接（线程安全）"""
        conn = sqlite3.connect(self.db_path)
        
        # 优化: 关联其他数据库
        db_dir = os.path.dirname(self.db_path)
        meta_db = os.path.join(db_dir, 'stock_meta.db')
        finance_db = os.path.join(db_dir, 'stock_finance.db')
        
        if os.path.exists(meta_db):
            conn.execute(f"ATTACH DATABASE '{meta_db}' AS meta")
        if os.path.exists(finance_db):
            conn.execute(f"ATTACH DATABASE '{finance_db}' AS finance")
        
        return conn
    
    def load_data(self, 
                  start_date: str,
                  end_date: str,
                  stock_codes: List[str] = None,
                  parallel: bool = True,
                  min_days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        加载数据
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表（None则加载全部）
            parallel: 是否并行加载
            min_days: 最少交易日数
        
        返回:
            {stock_code: DataFrame}
        """
        # 获取股票代码列表
        if stock_codes is None:
            stock_codes = self._get_all_stock_codes(start_date, end_date)
        
        print(f"开始加载数据: {len(stock_codes)} 只股票")
        
        if parallel and len(stock_codes) > 100:
            data = self._load_parallel(stock_codes, start_date, end_date, min_days)
        else:
            data = self._load_sequential(stock_codes, start_date, end_date, min_days)
        
        # 缓存并进一步压缩 (downcast)
        self._data_cache = self.downcast_to_float32(data)
        
        # 构建日期索引和每日行情映射
        self._build_indexes()
        
        # 记录所有交易日
        self._all_trading_dates = sorted(self._daily_bars.keys())
        
        print(f"数据加载完成: {len(data)} 只股票")
        return data

    def downcast_to_float32(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """将数据向下转型为 float32 以节省约 50% 的内存"""
        for code, df in data.items():
            numeric_cols = df.select_dtypes(include=['float64']).columns
            df[numeric_cols] = df[numeric_cols].astype('float32', copy=False)
            
            # 对 int64 如果范围允许也进行 downcast
            int_cols = df.select_dtypes(include=['int64']).columns
            for col in int_cols:
                if col in ['code', 'date']: continue
                df[col] = pd.to_numeric(df[col], downcast='integer')
        return data
    
    def _get_all_stock_codes(self, start_date: str, end_date: str) -> List[str]:
        """获取所有股票代码"""
        query = '''
            SELECT DISTINCT code
            FROM daily_data
            WHERE date >= ? AND date <= ?
        '''
        conn = self._get_connection()
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        return df['code'].tolist()
    
    def _load_sequential(self,
                        stock_codes: List[str],
                        start_date: str,
                        end_date: str,
                        min_days: int) -> Dict[str, pd.DataFrame]:
        """串行加载"""
        placeholders = ','.join(['?' for _ in stock_codes])
        query = f'''
            SELECT d.code, d.date, d.open, d.high, d.low, d.close, d.volume, d.amount, d.turnover_rate,
                   IFNULL(d.is_st, 0) as is_st
            FROM daily_data d
            WHERE d.code IN ({placeholders}) AND d.date >= ? AND d.date <= ?
            ORDER BY d.code, d.date
        '''
        
        params = stock_codes + [start_date, end_date]
        conn = self._get_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        result = {}
        for code, stock_df in df.groupby('code', sort=False):
            stock_df = stock_df.sort_values('date').reset_index(drop=True)

            # 立即进行 float32 转换，减少单线程加载时的内存堆积
            target_cols = stock_df.select_dtypes(include=['float64', 'int64']).columns
            target_cols = [c for c in target_cols if c not in ['code', 'date']]
            stock_df[target_cols] = stock_df[target_cols].astype('float32', copy=False)

            if len(stock_df) >= min_days:
                result[code] = stock_df
        
        return result
    
    def _load_parallel(self,
                      stock_codes: List[str],
                      start_date: str,
                      end_date: str,
                      min_days: int,
                      batch_size: int = 100) -> Dict[str, pd.DataFrame]:
        """并行加载"""
        # 分批
        batches = [stock_codes[i:i+batch_size] 
                  for i in range(0, len(stock_codes), batch_size)]
        
        # 准备任务
        tasks = [(self.db_path, batch, start_date, end_date) for batch in batches]
        
        # 并行处理
        result = {}
        workers = min(multiprocessing.cpu_count(), len(batches))
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_load_stock_batch, task): i 
                      for i, task in enumerate(tasks)}
            
            for future in as_completed(futures):
                batch_result = future.result()
                result.update(batch_result)
        
        return result
    
    def _build_indexes(self):
        """
        构建日期索引和每日活跃股票集合。

        _date_index: {code: {date: row_idx}} 用于 O(1) 行定位
        _daily_bars: {date: set(code, ...)} 仅记录当日活跃股票代码，
                     不再存储 row dict，由 get_bar_data 从 DataFrame 按需读取
        _bar_cache:  延迟填充的 {(code, date): dict} 缓存
        """
        self._date_index = {}
        self._daily_bars = {}
        self._bar_cache = {}

        print(f"  - 正在构建索引，涉及 {len(self._data_cache)} 只股票...")

        for code, df in self._data_cache.items():
            date_list = df['date'].tolist()
            self._date_index[code] = {d: i for i, d in enumerate(date_list)}

            for d in date_list:
                if d not in self._daily_bars:
                    self._daily_bars[d] = set()
                self._daily_bars[d].add(code)
    
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日列表
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
        
        返回:
            交易日列表
        """
        query = '''
            SELECT DISTINCT date
            FROM daily_data
            WHERE date >= ? AND date <= ?
            ORDER BY date
        '''
        conn = self._get_connection()
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        return df['date'].tolist()
    
    def get_historical_data(self,
                           stock_code: str,
                           end_date: str,
                           lookback_days: int = None) -> Optional[pd.DataFrame]:
        """
        获取历史数据（截止到指定日期）
        使用 View (不使用 copy) 以提升性能和减少内存开销
        """
        if stock_code not in self._data_cache:
            return None
        
        df = self._data_cache[stock_code]
        
        # 使用索引快速定位
        if stock_code in self._date_index and end_date in self._date_index[stock_code]:
            end_idx = self._date_index[stock_code][end_date]
            
            if lookback_days:
                start_idx = max(0, end_idx - lookback_days + 1)
                # 移除 .copy()，返回 View
                # 注：如果后续逻辑修改了此 DF，可能会影响全局缓存，但在回测框架中通常是只读的
                return df.iloc[start_idx:end_idx+1]
            else:
                return df.iloc[:end_idx+1]
        
        # 回退到日期过滤
        result = df[df['date'] <= end_date]
        
        if lookback_days and len(result) > lookback_days:
            result = result.iloc[-lookback_days:]
        
        return result if not result.empty else None
    
    def get_bar_data(self, stock_code: str, date: str):
        """
        获取单日行情 (返回 dict，所有调用方均使用 bar["key"] 访问)
        延迟计算并缓存，避免启动时一次性生成 1500 万个 dict
        """
        cache_key = (stock_code, date)
        cached = self._bar_cache.get(cache_key)
        if cached is not None:
            return cached

        if stock_code in self._date_index:
            idx = self._date_index[stock_code].get(date)
            if idx is not None:
                bar = self._data_cache[stock_code].iloc[idx].to_dict()
                self._bar_cache[cache_key] = bar
                return bar

        return None
    
    def get_market_snapshot(self, date: str) -> 'LazyMarketSnapshot':
        """
        获取优化的市场快照代理对象
        """
        return LazyMarketSnapshot(self, date)

    def prune_bar_cache(self, keep_dates: Optional[set] = None, max_items: int = None):
        """按日期裁剪延迟 bar 缓存，避免长回测中无限增长。"""
        if not getattr(self, '_bar_cache', None):
            return
        if keep_dates is not None:
            self._bar_cache = {
                key: value
                for key, value in self._bar_cache.items()
                if key[1] in keep_dates
            }
        if max_items is not None and len(self._bar_cache) > max_items:
            overflow = len(self._bar_cache) - max_items
            for key in list(self._bar_cache.keys())[:overflow]:
                self._bar_cache.pop(key, None)

    def close(self):
        """关闭数据库连接（已废弃，保留以兼容旧代码）"""
        # 不再持有全局连接，此方法保留为空以兼容
        pass


class LazyMarketSnapshot:
    """市场快照延迟加载代理，避免回测主循环中产生大量 DataFrame 拷贝"""
    
    def __init__(self, data_handler: DataHandler, date: str):
        self.data_handler = data_handler
        self.date = date
        self._cache = {}
        # 快速定位当日活跃的所有股票
        self.stock_codes = list(data_handler._daily_bars.get(date, set()))

    def get_bar(self, stock_code):
        """获取指定股票当日的单行行情数据 Series (不触发全量拷贝)"""
        return self.data_handler.get_bar_data(stock_code, self.date)

    def __getitem__(self, stock_code):
        if stock_code not in self._cache:
            # 只有在真正请求时才进行切片和拷贝
            data = self.data_handler.get_historical_data(stock_code, self.date)
            self._cache[stock_code] = data
        return self._cache[stock_code]

    def items(self):
        for code in self.stock_codes:
            yield code, self[code]

    def keys(self):
        return self.stock_codes

    def __len__(self):
        return len(self.stock_codes)
        
    def __contains__(self, stock_code):
        return stock_code in self.data_handler._daily_bars.get(self.date, set())
