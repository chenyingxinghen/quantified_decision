"""
回测引擎

核心回测逻辑，协调各个模块
"""

from typing import Dict, List, Optional
import pandas as pd
from .strategy import BaseStrategy, StrategySignal
from .portfolio import Portfolio, Trade
from .data_handler import DataHandler
from .performance import PerformanceAnalyzer
import sqlite3
from config import DATABASE_PATH, TrainingConfig
from config.strategy_config import (
    TIME_STOP_DAYS,
    TIME_STOP_MIN_LOSS_PCT,
    TREND_LINE_LONG_PERIOD,
    ENABLE_STOP_LOSS_EXIT,
    ENABLE_TAKE_PROFIT_EXIT,
    ENABLE_SUPPORT_BREAK_EXIT,
    ENABLE_TIME_STOP_EXIT
)


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self,
                 strategy: BaseStrategy,
                 data_handler: DataHandler,
                 initial_capital: float = 1.0,
                 commission_rate: float = 0.01,
                 max_positions: int = 1):
        """
        初始化回测引擎
        
        参数:
            strategy: 策略实例
            data_handler: 数据处理器
            initial_capital: 初始资金
            commission_rate: 手续费率
            max_positions: 最大持仓数
        """
        self.strategy = strategy
        self.data_handler = data_handler
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.max_positions = max_positions
        
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            max_positions=max_positions
        )
        
        self.performance_analyzer = PerformanceAnalyzer()
        
        # 回测状态
        self._current_date = None
        self._trading_dates = []
    
    def run(self,
            start_date: str,
            end_date: str,
            stock_codes: List[str] = None,
            verbose: bool = True) -> Dict:
        """
        运行回测
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表（None则全部）
            verbose: 是否打印详细信息
        
        返回:
            回测结果字典
        """
        if verbose:
            print("=" * 80)
            print("回测引擎启动")
            print("=" * 80)
            print(f"策略: {self.strategy.name}")
            print(f"时间范围: {start_date} 至 {end_date}")
            print(f"初始资金: {self.initial_capital}")
            print(f"手续费率: {self.commission_rate * 100:.2f}%")
            print(f"最大持仓: {self.max_positions}")
            print("=" * 80)
        
        # 初始化策略
        self.strategy.initialize()
        
        # 加载数据 (如果尚未加载)
        if not self.data_handler._data_cache:
            if verbose:
                print("\n加载数据...")
            if stock_codes is None:
                conn = sqlite3.connect(DATABASE_PATH)
                stock_codes_df = pd.read_sql_query(
                    f"SELECT DISTINCT code FROM daily_data LIMIT {TrainingConfig.STOCK_NUM}", 
                    conn
                )
                conn.close()
                stock_codes = stock_codes_df['code'].tolist()
            self.data_handler.load_data(start_date, end_date, stock_codes)
        else:
            if verbose:
                print(f"\n跳过加载数据 (已加载 {len(self.data_handler._data_cache)} 只股票)")
        
        # 获取交易日
        self._trading_dates = self.data_handler.get_trading_dates(start_date, end_date)
        if verbose:
            print(f"交易日数量: {len(self._trading_dates)}")
        
        # 记录初始资金
        self.portfolio.record_equity(start_date)
        
        # 主循环
        if verbose:
            print("\n开始回测...")
            print("-" * 80)
        
        for i, date in enumerate(self._trading_dates):
            self._current_date = date
            
            # 获取市场快照
            market_data = self.data_handler.get_market_snapshot(date)
            
            # 更新持仓价格
            self.portfolio.update_positions(date, market_data)
            
            # 策略回调
            self.strategy.on_bar(date, market_data)
            
            # 检查平仓信号
            self._check_exit_signals(date, market_data, verbose)
            
            # 检查开仓信号
            self._check_entry_signals(date, market_data, verbose)
            
            # 记录资金曲线
            self.portfolio.record_equity(date)
            
            # 进度显示
            if verbose and (i + 1) % 50 == 0:
                print(f"进度: {i+1}/{len(self._trading_dates)} | "
                      f"交易: {len(self.portfolio.trades)} | "
                      f"资金: {self.portfolio.total_value:.4f}")
        
        if verbose:
            print("-" * 80)
            print("回测完成")
        
        # 清理策略
        self.strategy.cleanup()
        
        # 计算性能指标
        metrics = self.performance_analyzer.calculate_metrics(
            self.portfolio.trades,
            self.initial_capital,
            self.portfolio.total_value,
            self.portfolio.equity_curve
        )
        
        # 打印摘要
        if verbose:
            self.performance_analyzer.print_summary(
                metrics,
                start_date,
                end_date,
                self.strategy.name
            )
        
        return {
            'metrics': metrics,
            'trades': self.portfolio.trades,
            'equity_curve': self.portfolio.equity_curve,
            'portfolio_state': self.portfolio.get_portfolio_state()
        }
    
    def _check_entry_signals(self, date: str, market_data: Dict, verbose: bool):
        """检查开仓信号"""
        # 如果已满仓，跳过
        if not self.portfolio.can_open_position():
            return
        
        # 获取投资组合状态
        portfolio_state = self.portfolio.get_portfolio_state()
        
        # 生成信号
        signals = self.strategy.generate_signals(date, market_data, portfolio_state)
        
        if not signals:
            return

        # 优化 4: 计算每只股票应分配的资金 (等权分配)
        # 使用总资产除以最大持仓数，确保即使当前有现金也能按预定比例买入
        capital_per_position = self.portfolio.total_value / self.max_positions
        
        # 处理买入信号
        for signal in signals:
            if signal.signal_type != 'buy':
                continue
            
            # 检查是否已有持仓
            if self.portfolio.has_position(signal.stock_code):
                continue
            
            # 获取下一交易日的开盘价
            next_date, entry_price = self._get_next_entry_price(
                date, signal.stock_code, signal.price
            )
            
            if next_date is None or entry_price is None:
                continue
            
            # 优化 1: 确保止损/止盈价相对于实际入场价有效
            actual_stop_loss = signal.stop_loss
            actual_take_profit = signal.take_profit
            
            if signal.price > 0:
                if actual_stop_loss is not None:
                    sl_dist = signal.price - actual_stop_loss
                    actual_stop_loss = entry_price - sl_dist
                
                if actual_take_profit is not None:
                    tp_dist = actual_take_profit - signal.price
                    actual_take_profit = entry_price + tp_dist

            # 获取置信度信息并存入 metadata
            metadata = (signal.metadata or {}).copy()
            if 'confidence' not in metadata:
                metadata['confidence'] = signal.confidence

            # 开仓 (传入计算好的分配资金)
            position = self.portfolio.open_position(
                stock_code=signal.stock_code,
                date=next_date,
                price=entry_price,
                capital_allocation=capital_per_position,
                stop_loss=actual_stop_loss,
                take_profit=actual_take_profit,
                metadata=metadata
            )
            
            if position and verbose:
                print(f"[{next_date}] 买入 {signal.stock_code}: {entry_price:.2f} "
                      f"(分配资金: {capital_per_position:.2f}, 置信度: {signal.confidence:.1f}%)")
            
            # 策略回调
            if position:
                self.strategy.on_trade(position)
            
            # 如果没满仓，继续检查下一个信号
            if not self.portfolio.can_open_position():
                break
    
    def _check_exit_signals(self, date: str, market_data: Dict, verbose: bool):
        """检查平仓信号"""
        positions_to_close = []
        
        for stock_code, position in self.portfolio.positions.items():
            # 获取当日行情
            bar = self.data_handler.get_bar_data(stock_code, date)
            if bar is None:
                continue
            
            # 检查退出条件
            should_exit, exit_price, exit_reason = self._check_exit_conditions(
                position, date, bar, market_data
            )
            
            if should_exit:
                positions_to_close.append((stock_code, exit_price, exit_reason))
        
        # 执行平仓
        for stock_code, exit_price, exit_reason in positions_to_close:
            trade = self.portfolio.close_position(stock_code, date, exit_price, exit_reason)
            
            if trade and verbose:
                print(f"[{date}] 卖出 {stock_code}: {exit_price:.2f} "
                      f"({exit_reason}) 收益: {trade.pnl_pct*100:.2f}%")
            
            # 策略回调
            if trade:
                self.strategy.on_trade(trade)
    
    def _check_exit_conditions(self,
                               position,
                               date: str,
                               bar: pd.Series,
                               market_data: Dict) -> tuple:
        """
        检查退出条件
        
        返回:
            (should_exit, exit_price, exit_reason)
        """
        # T+1规则
        if date == position.entry_date:
            return False, None, None
        
        open_price = bar['open']
        high = bar['high']
        low = bar['low']
        close = bar['close']
        
        # 计算价格变化率
        buy_price_abs = abs(position.entry_price)
        if buy_price_abs == 0:
            return False, None, None
        
        # 止损检查
        if position.stop_loss and ENABLE_STOP_LOSS_EXIT:
            if open_price <= position.stop_loss:
                # 优化 3: 如果开盘即跌破止损，以开盘价成交（更真实，且避免跳空低开导致的亏损被低估）
                return True, open_price, 'stop_loss'
            if low <= position.stop_loss:
                # 日内触及止损价
                return True, position.close, 'stop_loss'
        
        # 止盈检查
        if ENABLE_TAKE_PROFIT_EXIT and position.take_profit:
            if open_price >= position.take_profit:
                # 如果开盘即突破止盈，以开盘价成交
                return True, open_price, 'take_profit'
            if high >= position.take_profit:
                # 日内触及止盈价
                return True, position.take_profit, 'take_profit'
        
        # 时间止损（持仓超过阙值且亏损超过阙值）
        if (ENABLE_TIME_STOP_EXIT and 
            position.holding_days >= TIME_STOP_DAYS and 
            position.unrealized_pnl_pct <= TIME_STOP_MIN_LOSS_PCT):
            return True, close, f'time_stoploss'
        
        # 趋势破位检查
        if ENABLE_SUPPORT_BREAK_EXIT and self._check_trend_break(position.stock_code, date, market_data):
            return True, close, 'trend_break'
        
        return False, None, None
    
    def _check_trend_break(self, stock_code: str, date: str, market_data: Dict) -> bool:
        """检查趋势破位"""
        # 获取历史数据
        hist_data = self.data_handler.get_historical_data(stock_code, date, lookback_days=TREND_LINE_LONG_PERIOD)
        if hist_data is None or len(hist_data) < 30:
            return False
        
        try:
            from core.analysis.trend_line_analyzer import TrendLineAnalyzer
            analyzer = TrendLineAnalyzer()
            result = analyzer.analyze(hist_data)
            return result.get('broken_support', False)
        except:
            return False
    
    def _get_next_entry_price(self, 
                              current_date: str,
                              stock_code: str,
                              signal_price: float) -> tuple:
        """
        获取下一交易日的入场价格
        
        返回:
            (next_date, entry_price)
        """
        # 找到下一交易日
        try:
            current_idx = self._trading_dates.index(current_date)
            if current_idx + 1 >= len(self._trading_dates):
                return None, None
            
            next_date = self._trading_dates[current_idx + 1]
        except ValueError:
            return None, None
        
        # 获取下一交易日行情
        bar = self.data_handler.get_bar_data(stock_code, next_date)
        if bar is None:
            return None, None
        
        # 停牌检测 (成交量为0)
        if bar.get('volume', 0) == 0:
            return None, None
            
        # 一字涨停检测 (一字涨停无法买入)
        if bar['open'] == bar['high'] == bar['low'] == bar['close']:
            # 获取 ST 标签：优先从行情 bar 中获取，否则设为非 ST
            is_st = bar.get('is_st', 0) == 1
            # ST 股涨停阈值取 1.045, 普通股取 1.095
            limit_threshold = 1.045 if is_st else 1.095
            
            if bar['open'] > signal_price * limit_threshold:
                return None, None

        return next_date, bar['open']
    
    def get_results(self) -> Dict:
        """获取回测结果"""
        metrics = self.performance_analyzer.calculate_metrics(
            self.portfolio.trades,
            self.initial_capital,
            self.portfolio.total_value,
            self.portfolio.equity_curve
        )
        
        return {
            'metrics': metrics,
            'trades': self.portfolio.trades,
            'equity_curve': self.portfolio.equity_curve,
            'portfolio_state': self.portfolio.get_portfolio_state()
        }
