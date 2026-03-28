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
    
    # 按股票分组
    result = {}
    for code in df['code'].unique():
        stock_df = df[df['code'] == code].copy()
        stock_df = stock_df.sort_values('date').reset_index(drop=True)
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
        self._daily_bars: Dict[str, Dict[str, pd.Series]] = {} # 每日行情快照: date -> {code -> Series}
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
        
        # 缓存数据
        self._data_cache = data
        
        # 构建日期索引和每日行情映射
        self._build_indexes()
        
        # 记录所有交易日
        self._all_trading_dates = sorted(self._daily_bars.keys())
        
        print(f"数据加载完成: {len(data)} 只股票")
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
        for code in df['code'].unique():
            stock_df = df[df['code'] == code].copy()
            stock_df = stock_df.sort_values('date').reset_index(drop=True)
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
        同时构建日期索引和每日行情哈希表（高性能优化版本）
        
        优化点：
        1. 使用 to_dict('index') 代替 iterrows 循环，性能提升几十倍。
        2. _date_index 用于快速定位股票在 DataFrame 中的行索引。
        3. _daily_bars 用于快速获取某日全市场的行情快照。
        """
        self._date_index = {}
        self._daily_bars = {}
        
        # 预先收集所有交易日，一次性初始化字典，避免循环中判断成员
        all_dates = set()
        for df in self._data_cache.values():
            all_dates.update(df['date'].values)
        
        for date in all_dates:
            self._daily_bars[date] = {}
            
        print(f"  - 正在构建索引，涉及 {len(self._data_cache)} 只股票，共约 {len(all_dates)} 个交易日...")
        
        for code, df in self._data_cache.items():
            # 1. 构建日期到索引的快速映射 (T+1 定位用)
            # 使用 zip 比 set_index().to_dict() 更快
            date_list = df['date'].tolist()
            self._date_index[code] = {d: i for i, d in enumerate(date_list)}
            
            # 2. 构建每日行情映射 (快照用)
            # 技巧：将 DataFrame 按日期设为索引，然后转为字典。
            # 这样一行的所有字段（open, high, low, close 等）都会变成 dict，
            #虽然不是 Series，但 LazyMarketSnapshot.get_bar 可以根据需要转 Series。
            df_temp = df.set_index('date')
            # 核心优化：to_dict('index') 会返回 {date: {col: val, ...}}
            # 这种形式非常契合 _daily_bars[date][code] = row_dict
            code_daily_dict = df_temp.to_dict('index')
            
            for date, row_data in code_daily_dict.items():
                # 为了保持向后兼容性（返回 pd.Series），我们在这里只存 dict。
                # 只有在 get_bar_data 被调用时，才按需转换为 Series。
                # 或者在 LazyMarketSnapshot 中进行转换。
                self._daily_bars[date][code] = row_data
    
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
        
        参数:
            stock_code: 股票代码
            end_date: 截止日期
            lookback_days: 回看天数（None则返回全部）
        
        返回:
            DataFrame或None
        """
        if stock_code not in self._data_cache:
            return None
        
        df = self._data_cache[stock_code]
        
        # 使用索引快速定位
        if stock_code in self._date_index and end_date in self._date_index[stock_code]:
            end_idx = self._date_index[stock_code][end_date]
            
            if lookback_days:
                start_idx = max(0, end_idx - lookback_days + 1)
                return df.iloc[start_idx:end_idx+1].copy()
            else:
                return df.iloc[:end_idx+1].copy()
        
        # 回退到日期过滤
        result = df[df['date'] <= end_date].copy()
        
        if lookback_days and len(result) > lookback_days:
            result = result.tail(lookback_days)
        
        return result if not result.empty else None
    
    def get_bar_data(self, stock_code: str, date: str) -> Optional[pd.Series]:
        """
        获取单日行情
        
        参数:
            stock_code: 股票代码
            date: 日期
        
        返回:
            Series或None
        """
        # 1. 优先使用预构建的每日行情映射 (O(1))
        if date in self._daily_bars and stock_code in self._daily_bars[date]:
            bar = self._daily_bars[date][stock_code]
            if isinstance(bar, dict):
                # 兼容性转换：按需将 dict 转为 pd.Series 并写回缓存
                bar = pd.Series(bar)
                bar.name = date
                self._daily_bars[date][stock_code] = bar
            return bar
            
        # 2. 如果缓存未生效（非预加载范围），使用索引定位
        if stock_code in self._data_cache:
            df = self._data_cache[stock_code]
            if stock_code in self._date_index and date in self._date_index[stock_code]:
                idx = self._date_index[stock_code][date]
                return df.iloc[idx]
        
        return None
    
    def get_market_snapshot(self, date: str) -> 'LazyMarketSnapshot':
        """
        获取优化的市场快照代理对象
        """
        return LazyMarketSnapshot(self, date)

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
        self.stock_codes = [code for code, bars in data_handler._daily_bars.get(date, {}).items()]

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
        return stock_code in self.stock_codes
