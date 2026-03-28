"""
数据获取和处理模块

推荐使用 BaostockDataManager，DataFetcher 仅作为兼容层保留
"""
from .data_fetcher import DataFetcher
from .baostock_main import BaostockDataManager

__all__ = ['DataFetcher', 'BaostockDataManager']
