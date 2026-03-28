"""
数据获取适配层（保持向后兼容）

本模块作为适配层，内部调用 baostock_main.BaostockDataManager
为保持向后兼容性，保留原有的 DataFetcher 接口
"""

import pandas as pd
import warnings
from typing import List, Optional
from .baostock_main import BaostockDataManager
from config.baostock_config import DATABASE_PATH, HISTORY_YEARS, WORKERS_NUM, DEFAULT_MARKETS


class DataFetcher:
    """
    数据获取适配器类
    
    内部使用 BaostockDataManager，保持向后兼容的接口
    
    警告：此类已废弃，建议直接使用 BaostockDataManager
    """
    
    def __init__(self):
        """初始化数据获取器"""
        warnings.warn(
            "DataFetcher 已废弃，建议直接使用 BaostockDataManager",
            DeprecationWarning,
            stacklevel=2
        )
        self.db_path = DATABASE_PATH
        self._manager = BaostockDataManager()
    
    @property
    def conn(self):
        """获取数据库连接（兼容旧接口）"""
        return self._manager._get_conn()
    
    def close(self):
        """关闭连接"""
        # BaostockDataManager 使用上下文管理器，无需手动关闭
        pass
    
    def init_database(self):
        """初始化数据库（已由 BaostockDataManager 处理）"""
        pass
    
    def update_daily_data(self, symbol: str, incremental: bool = True):
        """
        更新单只股票的日线数据
        
        参数:
            symbol: 股票代码（6位）
            incremental: 是否增量更新
        """
        return self._manager.update_stock_data(symbol, incremental=incremental)
    
    def init_all_stocks_data(self, markets: List[str] = None, 
                            incremental: bool = True, 
                            workers: int = WORKERS_NUM):
        """
        初始化所有股票数据
        
        参数:
            markets: 市场列表，如 ['sh', 'sz']
            incremental: 是否增量更新
            workers: 并发线程数
        """
        if markets is None:
            markets = DEFAULT_MARKETS
        return self._manager.init_all_stocks(incremental=incremental, workers=workers)
    
    def get_stock_list(self, markets=None) -> pd.DataFrame:
        """
        获取股票列表
        
        参数:
            markets: 市场列表
            
        返回:
            DataFrame: 股票列表
        """
        return self._manager.get_stock_list_from_db(markets=markets)
    
    def get_stock_data(self, symbol: str, days: int = 1) -> pd.DataFrame:
        """
        获取股票最近N天的数据
        
        参数:
            symbol: 股票代码
            days: 天数
            
        返回:
            DataFrame: 股票数据
        """
        return self.get_historical_data(symbol, days=days)
    
    def get_historical_data(self, symbol: str, 
                           start_date: str = None, 
                           end_date: str = None,
                           days: int = None,
                           adjust: str = 'qfq') -> pd.DataFrame:
        """
        获取历史数据
        
        参数:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            days: 天数（如果指定，则忽略 start_date）
            adjust: 复权方式 ('qfq'前复权, 'hfq'后复权, None不复权)
            
        返回:
            DataFrame: 历史数据
        """
        # 计算日期范围
        if days is not None:
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y-%m-%d')
        
        if start_date is None:
            from datetime import datetime, timedelta
            start_date = (datetime.now() - timedelta(days=HISTORY_YEARS*365)).strftime('%Y-%m-%d')
        
        if end_date is None:
            from datetime import datetime
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 调用 BaostockDataManager 获取数据
        df = self._manager.get_adjusted_kline(
            code=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust_type=adjust
        )
        
        # 如果指定了天数，截取最后N天
        if days is not None and df is not None and len(df) > 0:
            df = df.tail(days)
        
        return df
    
    def get_adjusted_kline(self, code: str, 
                          start_date: str, 
                          end_date: str,
                          adjust_type: str = 'qfq') -> pd.DataFrame:
        """
        获取复权K线数据
        
        参数:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adjust_type: 复权方式 ('qfq'前复权, 'hfq'后复权, None不复权)
            
        返回:
            DataFrame: 复权后的K线数据
        """
        return self._manager.get_adjusted_kline(
            code=code,
            start_date=start_date,
            end_date=end_date,
            adjust_type=adjust_type
        )


# 向后兼容：保留旧的导入方式
__all__ = ['DataFetcher']
