"""
Baostock 数据处理器 - 支持动态前复权

核心特性：
1. 在回测过程中，始终以当前日为基准进行前复权
2. 消除未来函数问题
3. 高效的数据加载和缓存
"""
import os
import sqlite3
import pandas as pd
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def _load_stock_batch_baostock(args):
    """多进程加载股票数据（Baostock版本）"""
    db_path, stock_codes, start_date, end_date = args
    
    conn = sqlite3.connect(db_path)
    db_dir = os.path.dirname(db_path)
    meta_db = os.path.join(db_dir, 'stock_meta.db')
    if os.path.exists(meta_db):
        conn.execute(f"ATTACH DATABASE '{meta_db}' AS meta")
    
    placeholders = ','.join(['?' for _ in stock_codes])
    
    # 加载K线和复权因子
    query = f'''
        SELECT k.code, k.date, k.open, k.high, k.low, k.close, k.preclose,
               k.volume, k.amount, k.turnover_rate, k.tradestatus, k.pctChg,
               k.peTTM, k.pbMRQ, k.psTTM, k.pcfNcfTTM, k.is_st,
               a.fore_adjust_factor, a.back_adjust_factor
        FROM daily_data k
        LEFT JOIN adjust_factor a ON k.code = a.code AND k.date = a.date
        WHERE k.code IN ({placeholders}) AND k.date >= ? AND k.date <= ?
        ORDER BY k.code, k.date
    '''
    
    params = stock_codes + [start_date, end_date]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    # 按股票分组
    result = {}
    for code in df['code'].unique():
        stock_df = df[df['code'] == code].copy()
        stock_df = stock_df.sort_values('date').reset_index(drop=True)
        
        # 填充缺失的复权因子
        stock_df['fore_adjust_factor'] = stock_df['fore_adjust_factor'].ffill().fillna(1.0)
        stock_df['back_adjust_factor'] = stock_df['back_adjust_factor'].ffill().fillna(1.0)
        
        if len(stock_df) >= 30:
            result[code] = stock_df
    
    return result


class BaostockDataHandler:
    """Baostock 数据处理器 - 支持动态前复权"""
    
    def __init__(self, db_path: str):
        """
        初始化数据处理器
        
        参数:
            db_path: 数据库路径
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # 附加其他数据库
        db_dir = os.path.dirname(db_path)
        meta_db = os.path.join(db_dir, 'stock_meta.db')
        finance_db = os.path.join(db_dir, 'stock_finance.db')
        
        if os.path.exists(meta_db):
            self.conn.execute(f"ATTACH DATABASE '{meta_db}' AS meta")
        if os.path.exists(finance_db):
            self.conn.execute(f"ATTACH DATABASE '{finance_db}' AS finance")
        
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._date_index: Dict[str, Dict[str, int]] = {}
        self._daily_bars: Dict[str, Dict[str, pd.Series]] = {}
        self._all_trading_dates: List[str] = []
        
        # 动态复权缓存
        self._adjusted_cache: Dict[str, Dict[str, pd.DataFrame]] = {}  # {code: {date: adjusted_df}}
    
    def load_data(self,
                  start_date: str,
                  end_date: str,
                  stock_codes: List[str] = None,
                  parallel: bool = True,
                  min_days: int = 60) -> Dict[str, pd.DataFrame]:
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
        if stock_codes is None:
            stock_codes = self._get_all_stock_codes(start_date, end_date)
        
        print(f"开始加载数据: {len(stock_codes)} 只股票")
        
        if parallel and len(stock_codes) > 100:
            data = self._load_parallel(stock_codes, start_date, end_date, min_days)
        else:
            data = self._load_sequential(stock_codes, start_date, end_date, min_days)
        
        self._data_cache = data
        self._build_indexes()
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
            SELECT k.code, k.date, k.open, k.high, k.low, k.close, k.preclose,
                   k.volume, k.amount, k.turnover_rate, k.tradestatus, k.pctChg,
                   k.peTTM, k.pbMRQ, k.psTTM, k.pcfNcfTTM, k.is_st,
                   a.fore_adjust_factor, a.back_adjust_factor
        FROM daily_data k
        LEFT JOIN adjust_factor a ON k.code = a.code AND k.date = a.date
        WHERE k.code IN ({placeholders}) AND k.date >= ? AND k.date <= ?
        ORDER BY k.code, k.date
    '''
        
        params = stock_codes + [start_date, end_date]
        df = pd.read_sql_query(query, self.conn, params=params)
        
        result = {}
        for code in df['code'].unique():
            stock_df = df[df['code'] == code].copy()
            stock_df = stock_df.sort_values('date').reset_index(drop=True)
            
            # 填充缺失的复权因子
            stock_df['fore_adjust_factor'] = stock_df['fore_adjust_factor'].ffill().fillna(1.0)
            stock_df['back_adjust_factor'] = stock_df['back_adjust_factor'].ffill().fillna(1.0)
            
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
        batches = [stock_codes[i:i+batch_size] 
                  for i in range(0, len(stock_codes), batch_size)]
        
        tasks = [(self.db_path, batch, start_date, end_date) for batch in batches]
        
        result = {}
        workers = min(multiprocessing.cpu_count(), len(batches))
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_load_stock_batch_baostock, task): i 
                      for i, task in enumerate(tasks)}
            
            for future in as_completed(futures):
                batch_result = future.result()
                result.update(batch_result)
        
        return result
    
    def _build_indexes(self):
        """构建日期索引和每日行情哈希表"""
        self._date_index = {}
        self._daily_bars = {}
        
        for code, df in self._data_cache.items():
            date_to_idx = {}
            for idx, row in df.iterrows():
                date = row['date']
                date_to_idx[date] = idx
                
                if date not in self._daily_bars:
                    self._daily_bars[date] = {}
                self._daily_bars[date][code] = row
            
            self._date_index[code] = date_to_idx
    
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表"""
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
                           lookback_days: int = None,
                           adjust_to_date: bool = True) -> Optional[pd.DataFrame]:
        """
        获取历史数据（支持动态前复权）
        
        参数:
            stock_code: 股票代码
            end_date: 截止日期
            lookback_days: 回看天数
            adjust_to_date: 是否以end_date为基准进行前复权
        
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
                result = df.iloc[start_idx:end_idx+1].copy()
            else:
                result = df.iloc[:end_idx+1].copy()
        else:
            result = df[df['date'] <= end_date].copy()
            if lookback_days and len(result) > lookback_days:
                result = result.tail(lookback_days)
        
        if result.empty:
            return None
        
        # 动态前复权
        if adjust_to_date:
            result = self._apply_dynamic_adjustment(result, end_date)
        
        return result
    
    def _apply_dynamic_adjustment(self, df: pd.DataFrame, base_date: str) -> pd.DataFrame:
        """
        应用动态前复权
        
        参数:
            df: 原始数据
            base_date: 复权基准日
        
        返回:
            复权后的DataFrame
        """
        if df.empty:
            return df
        
        # 获取基准日的复权因子
        base_row = df[df['date'] == base_date]
        if base_row.empty:
            base_row = df.iloc[-1:]
        
        base_factor = base_row['fore_adjust_factor'].iloc[0]
        if pd.isna(base_factor) or base_factor == 0:
            base_factor = 1.0
        
        # 计算复权比例
        df = df.copy()
        df['adj_factor_ratio'] = df['fore_adjust_factor'] / base_factor
        
        # 应用复权计算并覆盖原始价格列以兼容回测引擎
        for col in ['open', 'high', 'low', 'close', 'preclose']:
            df[col] = df[col] * df['adj_factor_ratio']
            df[f'adj_{col}'] = df[col]  # 冗余保存一个 adj_ 前缀以备不时之需
        
        return df
    
    def get_bar_data(self, stock_code: str, date: str, adjusted: bool = True) -> Optional[pd.Series]:
        """
        获取单日行情
        
        参数:
            stock_code: 股票代码
            date: 日期
            adjusted: 是否返回复权数据（以当日为基准）
        
        返回:
            Series或None
        """
        if date in self._daily_bars and stock_code in self._daily_bars[date]:
            bar = self._daily_bars[date][stock_code].copy()
            
            if adjusted:
                # 已经是当日基准，价格无需变动但添加 adj_ 前缀以对齐接口
                for col in ['open', 'high', 'low', 'close', 'preclose']:
                    bar[f'adj_{col}'] = bar[col]
            
            return bar
        
        return None
    
    def get_market_snapshot(self, date: str) -> 'LazyMarketSnapshotBaostock':
        """获取市场快照（支持动态复权）"""
        return LazyMarketSnapshotBaostock(self, date)
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()


class LazyMarketSnapshotBaostock:
    """市场快照延迟加载代理 - Baostock版本（支持动态复权）"""
    
    def __init__(self, data_handler: BaostockDataHandler, date: str):
        self.data_handler = data_handler
        self.date = date
        self._cache = {}
        self.stock_codes = list(data_handler._daily_bars.get(date, {}).keys())
    
    def get_bar(self, stock_code: str, adjusted: bool = True):
        """获取指定股票当日的单行行情数据"""
        return self.data_handler.get_bar_data(stock_code, self.date, adjusted)
    
    def __getitem__(self, stock_code):
        if stock_code not in self._cache:
            # 获取截止到当日的历史数据，并以当日为基准复权
            data = self.data_handler.get_historical_data(
                stock_code, self.date, adjust_to_date=True
            )
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
