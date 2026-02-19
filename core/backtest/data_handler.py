"""
数据处理器

负责数据加载、预处理和缓存
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def _load_stock_batch(args):
    """多进程加载股票数据"""
    db_path, stock_codes, start_date, end_date = args
    
    conn = sqlite3.connect(db_path)
    placeholders = ','.join(['?' for _ in stock_codes])
    query = f'''
        SELECT code, date, open, high, low, close, volume, amount, turnover_rate
        FROM daily_data
        WHERE code IN ({placeholders}) AND date >= ? AND date <= ?
        ORDER BY code, date
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
        self.conn = sqlite3.connect(db_path)
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._date_index: Dict[str, Dict[str, int]] = {}
        self._daily_bars: Dict[str, Dict[str, pd.Series]] = {} # 每日行情快照: date -> {code -> Series}
        self._all_trading_dates: List[str] = []
    
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
        df = pd.read_sql_query(query, self.conn, params=(start_date, end_date))
        return df['code'].tolist()
    
    def _load_sequential(self,
                        stock_codes: List[str],
                        start_date: str,
                        end_date: str,
                        min_days: int) -> Dict[str, pd.DataFrame]:
        """串行加载"""
        placeholders = ','.join(['?' for _ in stock_codes])
        query = f'''
            SELECT code, date, open, high, low, close, volume, amount, turnover_rate
            FROM daily_data
            WHERE code IN ({placeholders}) AND date >= ? AND date <= ?
            ORDER BY code, date
        '''
        
        params = stock_codes + [start_date, end_date]
        df = pd.read_sql_query(query, self.conn, params=params)
        
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
        """同时构建日期索引和每日行情哈希表"""
        self._date_index = {}
        self._daily_bars = {}
        
        for code, df in self._data_cache.items():
            date_to_idx = {}
            for idx, row in df.iterrows():
                date = row['date']
                date_to_idx[date] = idx
                
                # 存入每日行情映射
                if date not in self._daily_bars:
                    self._daily_bars[date] = {}
                self._daily_bars[date][code] = row
            
            self._date_index[code] = date_to_idx
    
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
        df = pd.read_sql_query(query, self.conn, params=(start_date, end_date))
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
            return self._daily_bars[date][stock_code]
            
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
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()


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
