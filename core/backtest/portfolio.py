"""
投资组合管理

处理持仓、交易执行和资金管理
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
import pandas as pd


@dataclass
class Position:
    """持仓"""
    stock_code: str
    entry_date: str
    entry_price: float
    shares: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    
    # 动态跟踪字段
    holding_days: int = 0
    max_profit_rate: float = 0.0
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        """当前市值"""
        return self.shares * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """成本基础"""
        return self.shares * abs(self.entry_price)
    
    @property
    def unrealized_pnl(self) -> float:
        """未实现盈亏"""
        if self.entry_price == 0:
            return 0.0
        price_change_rate = (self.current_price - self.entry_price) / abs(self.entry_price)
        return self.cost_basis * price_change_rate
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """未实现盈亏百分比"""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / abs(self.entry_price)
    
    def update_price(self, price: float):
        """更新当前价格"""
        self.current_price = price
        self.holding_days += 1
        
        # 更新最大盈利率
        current_pnl_pct = self.unrealized_pnl_pct
        if current_pnl_pct > self.max_profit_rate:
            self.max_profit_rate = current_pnl_pct


@dataclass
class Trade:
    """交易记录"""
    stock_code: str
    direction: str  # 'buy' or 'sell'
    buy_date: str
    buy_price: float
    sell_date: str
    sell_price: float
    shares: float
    commission: float
    pnl: float
    pnl_pct: float
    holding_days: int
    exit_reason: str = ""
    metadata: Dict = field(default_factory=dict)


class Portfolio:
    """投资组合"""
    
    def __init__(self, 
                 initial_capital: float,
                 commission_rate: float = 0.001,
                 max_positions: int = 1):
        """
        初始化投资组合
        
        参数:
            initial_capital: 初始资金
            commission_rate: 手续费率
            max_positions: 最大持仓数量
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.max_positions = max_positions
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[tuple] = []
    
    @property
    def total_value(self) -> float:
        """总资产"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def position_count(self) -> int:
        """当前持仓数量"""
        return len(self.positions)
    
    def can_open_position(self) -> bool:
        """是否可以开新仓"""
        return self.position_count < self.max_positions
    
    def has_position(self, stock_code: str = None) -> bool:
        """检查是否有持仓"""
        if stock_code:
            return stock_code in self.positions
        return len(self.positions) > 0
    
    def get_position(self, stock_code: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(stock_code)
    
    def open_position(self,
                     stock_code: str,
                     date: str,
                     price: float,
                     capital_allocation: float = None,
                     stop_loss: float = None,
                     take_profit: float = None,
                     metadata: Dict = None) -> Optional[Position]:
        """
        开仓
        
        参数:
            stock_code: 股票代码
            date: 日期
            price: 价格
            capital_allocation: 分配资金（None则使用全部现金）
            stop_loss: 止损价
            take_profit: 止盈价
            metadata: 元数据
        
        返回:
            Position对象或None（如果开仓失败）
        """
        # 检查是否已有持仓
        if stock_code in self.positions:
            return None
        
        # 检查是否超过最大持仓数
        if not self.can_open_position():
            return None
        
        # 确定使用的资金
        if capital_allocation is None:
            capital_allocation = self.cash
        else:
            capital_allocation = min(capital_allocation, self.cash)
        
        if capital_allocation <= 0:
            return None
        
        # 计算手续费
        commission = capital_allocation * self.commission_rate
        available_capital = capital_allocation - commission
        
        # 计算股数
        abs_price = abs(price)
        if abs_price == 0:
            return None
        shares = available_capital / abs_price
        
        # 创建持仓
        position = Position(
            stock_code=stock_code,
            entry_date=date,
            entry_price=price,
            shares=shares,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata or {},
            current_price=price
        )
        
        # 更新现金
        self.cash -= capital_allocation
        
        # 添加到持仓
        self.positions[stock_code] = position
        
        return position
    
    def close_position(self,
                      stock_code: str,
                      date: str,
                      price: float,
                      reason: str = "") -> Optional[Trade]:
        """
        平仓
        
        参数:
            stock_code: 股票代码
            date: 日期
            price: 价格
            reason: 平仓原因
        
        返回:
            Trade对象或None（如果平仓失败）
        """
        position = self.positions.get(stock_code)
        if not position:
            return None
        
        # 计算价格变化率
        if position.entry_price == 0:
            price_change_rate = 0
        else:
            price_change_rate = (price - position.entry_price) / abs(position.entry_price)
        
        # 计算卖出金额（基于成本和价格变化率）
        sell_amount = position.cost_basis * (1 + price_change_rate)
        
        # 计算手续费
        commission = abs(sell_amount) * self.commission_rate
        
        # 净收入
        net_proceeds = sell_amount - commission
        
        # 计算盈亏
        total_cost = position.cost_basis + position.cost_basis * self.commission_rate
        pnl = net_proceeds - total_cost
        pnl_pct = pnl / total_cost if total_cost != 0 else 0
        
        # 创建交易记录
        trade = Trade(
            stock_code=stock_code,
            direction='long',
            buy_date=position.entry_date,
            buy_price=position.entry_price,
            sell_date=date,
            sell_price=price,
            shares=position.shares,
            commission=commission + total_cost - position.cost_basis,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=position.holding_days,
            exit_reason=reason,
            metadata=position.metadata
        )
        
        # 更新现金
        self.cash += net_proceeds
        
        # 移除持仓
        del self.positions[stock_code]
        
        # 记录交易
        self.trades.append(trade)
        
        return trade
    
    def update_positions(self, date: str, market_data: Dict[str, pd.DataFrame]):
        """
        更新所有持仓的价格
        
        参数:
            date: 当前日期
            market_data: 市场数据
        """
        for stock_code, position in self.positions.items():
            if stock_code in market_data:
                stock_df = market_data[stock_code]
                # 获取当日收盘价
                today_data = stock_df[stock_df['date'] == date]
                if not today_data.empty:
                    current_price = today_data.iloc[0]['close']
                    position.update_price(current_price)
    
    def record_equity(self, date: str):
        """记录资产曲线"""
        self.equity_curve.append((date, self.total_value))
    
    def get_portfolio_state(self) -> Dict:
        """获取投资组合状态"""
        return {
            'cash': self.cash,
            'total_value': self.total_value,
            'position_count': self.position_count,
            'positions': {code: {
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'shares': pos.shares,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                'holding_days': pos.holding_days
            } for code, pos in self.positions.items()}
        }
