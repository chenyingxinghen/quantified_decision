"""
策略基类和信号定义

提供统一的策略接口，支持策略扩展
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd


@dataclass
class StrategySignal:
    """策略信号"""
    stock_code: str
    signal_type: str  # 'buy', 'sell', 'hold'
    timestamp: str
    price: float
    confidence: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """验证信号有效性"""
        if self.signal_type not in ['buy', 'sell', 'hold']:
            raise ValueError(f"Invalid signal type: {self.signal_type}")
        if not 0 <= self.confidence <= 100:
            raise ValueError(f"Confidence must be between 0 and 100, got {self.confidence}")


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self._initialized = False
    
    def initialize(self, **kwargs):
        """初始化策略（可选）"""
        self._initialized = True
    
    @abstractmethod
    def generate_signals(self, 
                        current_date: str,
                        market_data: Dict[str, pd.DataFrame],
                        portfolio_state: Dict[str, Any]) -> list[StrategySignal]:
        """
        生成交易信号
        
        参数:
            current_date: 当前日期
            market_data: 市场数据 {stock_code: DataFrame}
            portfolio_state: 当前持仓状态
        
        返回:
            信号列表
        """
        pass
    
    def on_trade(self, trade):
        """交易完成回调（可选）"""
        pass
    
    def on_bar(self, date: str, data: Dict[str, pd.DataFrame]):
        """每个交易日回调（可选）"""
        pass
    
    def cleanup(self):
        """清理资源（可选）"""
        pass
