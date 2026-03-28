"""
Core Backtest Module - 核心回测模块

提供模块化、可扩展的回测框架
"""

from .engine import BacktestEngine
from .strategy import BaseStrategy, StrategySignal
from .portfolio import Portfolio, Position, Trade
from .baostock_data_handler import BaostockDataHandler as DataHandler
from .performance import PerformanceAnalyzer

__all__ = [
    'BacktestEngine',
    'BaseStrategy',
    'StrategySignal',
    'Portfolio',
    'Position',
    'Trade',
    'DataHandler',
    'PerformanceAnalyzer'
]
