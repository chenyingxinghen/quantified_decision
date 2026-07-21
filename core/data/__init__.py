"""
数据获取和处理模块

推荐使用 BaostockDataManager，DataFetcher 仅作为兼容层保留
"""
from .data_fetcher import DataFetcher
from .baostock_main import BaostockDataManager
from .jydb_feature_store import JYDBETL, JYDBFeatureStore
from .jydb_market_etl import JYDBMarketETL

__all__ = ['DataFetcher', 'BaostockDataManager', 'JYDBETL', 'JYDBFeatureStore', 'JYDBMarketETL']
