"""
示例策略

展示如何创建自定义策略
"""

import pandas as pd
from typing import Dict, List, Any

from core.backtest.strategy import BaseStrategy, StrategySignal


class SimpleMovingAverageStrategy(BaseStrategy):
    """简单移动平均策略
    
    买入信号: 短期均线上穿长期均线
    卖出信号: 短期均线下穿长期均线
    """
    
    def __init__(self, 
                 short_window: int = 5,
                 long_window: int = 20,
                 name: str = "简单均线策略"):
        """
        初始化策略
        
        参数:
            short_window: 短期均线窗口
            long_window: 长期均线窗口
            name: 策略名称
        """
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
    
    def initialize(self, **kwargs):
        """初始化策略"""
        super().initialize(**kwargs)
        print(f"策略初始化: {self.name}")
        print(f"  短期窗口: {self.short_window}")
        print(f"  长期窗口: {self.long_window}")
    
    def generate_signals(self,
                        current_date: str,
                        market_data: Dict[str, pd.DataFrame],
                        portfolio_state: Dict[str, Any]) -> List[StrategySignal]:
        """生成交易信号"""
        signals = []
        
        # 如果已有持仓，不生成新信号
        if portfolio_state.get('position_count', 0) > 0:
            return signals
        
        # 筛选股票
        for stock_code, stock_data in market_data.items():
            # 数据量检查
            if len(stock_data) < self.long_window:
                continue
            
            # 计算均线
            short_ma = stock_data['close'].rolling(window=self.short_window).mean()
            long_ma = stock_data['close'].rolling(window=self.long_window).mean()
            
            # 检查金叉
            if len(short_ma) < 2 or len(long_ma) < 2:
                continue
            
            # 当前短期均线 > 长期均线，且前一天短期均线 <= 长期均线
            if (short_ma.iloc[-1] > long_ma.iloc[-1] and 
                short_ma.iloc[-2] <= long_ma.iloc[-2]):
                
                current_price = stock_data['close'].iloc[-1]
                
                # 简单的止损止盈设置
                stop_loss = current_price * 0.95  # 5%止损
                take_profit = current_price * 1.10  # 10%止盈
                
                signal = StrategySignal(
                    stock_code=stock_code,
                    signal_type='buy',
                    timestamp=current_date,
                    price=current_price,
                    confidence=70.0,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'short_ma': short_ma.iloc[-1],
                        'long_ma': long_ma.iloc[-1]
                    }
                )
                
                signals.append(signal)
        
        # 返回第一个信号（单仓位）
        return signals[:1] if signals else []
    
    def cleanup(self):
        """清理资源"""
        print(f"策略清理: {self.name}")


class MomentumStrategy(BaseStrategy):
    """动量策略
    
    买入信号: 过去N天涨幅超过阈值
    """
    
    def __init__(self,
                 lookback_days: int = 20,
                 momentum_threshold: float = 0.10,
                 name: str = "动量策略"):
        """
        初始化策略
        
        参数:
            lookback_days: 回看天数
            momentum_threshold: 动量阈值（涨幅）
            name: 策略名称
        """
        super().__init__(name)
        self.lookback_days = lookback_days
        self.momentum_threshold = momentum_threshold
    
    def initialize(self, **kwargs):
        """初始化策略"""
        super().initialize(**kwargs)
        print(f"策略初始化: {self.name}")
        print(f"  回看天数: {self.lookback_days}")
        print(f"  动量阈值: {self.momentum_threshold * 100:.1f}%")
    
    def generate_signals(self,
                        current_date: str,
                        market_data: Dict[str, pd.DataFrame],
                        portfolio_state: Dict[str, Any]) -> List[StrategySignal]:
        """生成交易信号"""
        signals = []
        
        # 如果已有持仓，不生成新信号
        if portfolio_state.get('position_count', 0) > 0:
            return signals
        
        candidates = []
        
        # 筛选股票
        for stock_code, stock_data in market_data.items():
            # 数据量检查
            if len(stock_data) < self.lookback_days + 1:
                continue
            
            # 计算动量（过去N天的涨幅）
            current_price = stock_data['close'].iloc[-1]
            past_price = stock_data['close'].iloc[-self.lookback_days-1]
            
            if past_price == 0:
                continue
            
            momentum = (current_price - past_price) / abs(past_price)
            
            # 检查是否超过阈值
            if momentum > self.momentum_threshold:
                # 计算ATR用于止损
                atr = self._calculate_atr(stock_data)
                
                candidates.append({
                    'stock_code': stock_code,
                    'momentum': momentum,
                    'current_price': current_price,
                    'atr': atr
                })
        
        # 选择动量最强的
        if candidates:
            best = max(candidates, key=lambda x: x['momentum'])
            
            signal = StrategySignal(
                stock_code=best['stock_code'],
                signal_type='buy',
                timestamp=current_date,
                price=best['current_price'],
                confidence=min(best['momentum'] * 100, 100),
                stop_loss=best['current_price'] - 2 * best['atr'],
                take_profit=best['current_price'] + 3 * best['atr'],
                metadata={
                    'momentum': best['momentum']
                }
            )
            
            signals.append(signal)
        
        return signals
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """计算ATR"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if len(atr) > 0 else 0
    
    def cleanup(self):
        """清理资源"""
        print(f"策略清理: {self.name}")


class BreakoutStrategy(BaseStrategy):
    """突破策略
    
    买入信号: 价格突破N日最高价
    """
    
    def __init__(self,
                 lookback_days: int = 20,
                 volume_threshold: float = 1.5,
                 name: str = "突破策略"):
        """
        初始化策略
        
        参数:
            lookback_days: 回看天数
            volume_threshold: 成交量放大倍数
            name: 策略名称
        """
        super().__init__(name)
        self.lookback_days = lookback_days
        self.volume_threshold = volume_threshold
    
    def initialize(self, **kwargs):
        """初始化策略"""
        super().initialize(**kwargs)
        print(f"策略初始化: {self.name}")
        print(f"  回看天数: {self.lookback_days}")
        print(f"  成交量阈值: {self.volume_threshold}x")
    
    def generate_signals(self,
                        current_date: str,
                        market_data: Dict[str, pd.DataFrame],
                        portfolio_state: Dict[str, Any]) -> List[StrategySignal]:
        """生成交易信号"""
        signals = []
        
        # 如果已有持仓，不生成新信号
        if portfolio_state.get('position_count', 0) > 0:
            return signals
        
        # 筛选股票
        for stock_code, stock_data in market_data.items():
            # 数据量检查
            if len(stock_data) < self.lookback_days + 1:
                continue
            
            # 获取当前价格和成交量
            current_price = stock_data['close'].iloc[-1]
            current_volume = stock_data['volume'].iloc[-1]
            
            # 计算过去N天的最高价
            past_high = stock_data['high'].iloc[-self.lookback_days-1:-1].max()
            
            # 计算平均成交量
            avg_volume = stock_data['volume'].iloc[-self.lookback_days-1:-1].mean()
            
            # 检查突破条件
            if (current_price > past_high and 
                current_volume > avg_volume * self.volume_threshold):
                
                # 使用过去最高价作为止损
                stop_loss = past_high * 0.98
                take_profit = current_price * 1.15
                
                signal = StrategySignal(
                    stock_code=stock_code,
                    signal_type='buy',
                    timestamp=current_date,
                    price=current_price,
                    confidence=75.0,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'past_high': past_high,
                        'volume_ratio': current_volume / avg_volume
                    }
                )
                
                signals.append(signal)
        
        # 返回第一个信号
        return signals[:1] if signals else []
    
    def cleanup(self):
        """清理资源"""
        print(f"策略清理: {self.name}")
