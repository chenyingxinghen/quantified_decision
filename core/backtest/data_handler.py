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
        
        # 构建日期索引
        self._build_date_index()
        
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
    
    def _build_date_index(self):
        """构建日期索引以加速查询"""
        self._date_index = {}
        
        for code, df in self._data_cache.items():
            date_to_idx = {}
            for idx, date in enumerate(df['date'].values):
                date_to_idx[date] = idx
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
        if stock_code not in self._data_cache:
            return None
        
        df = self._data_cache[stock_code]
        
        # 使用索引快速定位
        if stock_code in self._date_index and date in self._date_index[stock_code]:
            idx = self._date_index[stock_code][date]
            return df.iloc[idx]
        
        # 回退到日期过滤
        result = df[df['date'] == date]
        return result.iloc[0] if not result.empty else None
    
    def get_market_snapshot(self, date: str) -> Dict[str, pd.DataFrame]:
        """
        获取市场快照（所有股票截止到指定日期的历史数据）
        
        参数:
            date: 日期
        
        返回:
            {stock_code: DataFrame}
        """
        snapshot = {}
        for code in self._data_cache.keys():
            hist_data = self.get_historical_data(code, date)
            if hist_data is not None and len(hist_data) >= 30:
                snapshot[code] = hist_data
        return snapshot
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
