# 回测引擎 - 股票交易策略回测系统
# 基于需求文档和设计文档实现

import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import sys
import os

# 添加父目录到路径以导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATABASE_PATH

# 确保数据库路径为绝对路径 - 修正路径计算
if not os.path.isabs(DATABASE_PATH):
    # 获取项目根目录（从当前文件向上两级）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATABASE_PATH = os.path.join(project_root, DATABASE_PATH)
from integrated_screener import IntegratedScreener
from smc_liquidity_strategy import SMCLiquidityStrategy


# 全局函数用于多进程（必须在顶层定义）
def _load_stock_chunk(args):
    """
    加载一批股票数据（多进程工作函数）
    
    参数:
        args: (db_path, codes_batch, start_date, end_date)
    """
    db_path, codes_batch, start_date, end_date = args
    
    # 每个进程创建自己的数据库连接
    conn = sqlite3.connect(db_path)
    
    placeholders = ','.join(['?' for _ in codes_batch])
    query = f'''
        SELECT code, date, open, high, low, close, volume, amount, turnover_rate
        FROM daily_data
        WHERE code IN ({placeholders}) AND date >= ? AND date <= ?
        ORDER BY code, date ASC
    '''
    
    params = codes_batch + [start_date, end_date]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    # 按股票代码分组
    batch_data = {}
    for code in df['code'].unique():
        stock_df = df[df['code'] == code].copy()
        stock_df = stock_df.sort_values('date')
        
        if len(stock_df) >= 30:
            batch_data[code] = stock_df
    
    return batch_data


@dataclass
class Signal:
    """交易信号"""
    stock_code: str
    signal_type: str
    confidence: float
    current_price: float
    entry_price: float
    stop_loss: float
    target: float
    date: str


@dataclass
class Position:
    """持仓"""
    stock_code: str
    buy_date: str
    buy_price: float
    shares: float
    stop_loss: float
    target: float
    cost: float
    no_rise_days: int = 0  # 连续未上涨天数
    last_close: float = 0.0  # 上一交易日收盘价


@dataclass
class Trade:
    """交易记录"""
    stock_code: str
    buy_date: str
    buy_price: float
    sell_date: str
    sell_price: float
    shares: float
    buy_commission: float
    sell_commission: float
    profit: float
    return_rate: float
    exit_reason: str


class DataLoader:
    """数据加载器"""
    
    def __init__(self, db_path: str = None):
        # 如果没有提供数据库路径，则使用全局配置
        if db_path is None:
            db_path = DATABASE_PATH
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def load_all_stocks_data(self, start_date: str, end_date: str, use_parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        加载所有股票的历史数据（支持并行加载）
        
        参数:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            use_parallel: 是否使用并行加载，默认True
        
        返回:
            dict: {stock_code: DataFrame} 股票代码到数据的映射
        """
        if use_parallel:
            return self._load_parallel(start_date, end_date)
        else:
            return self._load_sequential(start_date, end_date)
    
    def _load_sequential(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """串行加载（原始方法）"""
        query = '''
            SELECT code, date, open, high, low, close, volume, amount, turnover_rate
            FROM daily_data
            WHERE date >= ? AND date <= ?
            ORDER BY code, date ASC
        '''
        
        df = pd.read_sql_query(query, self.conn, params=(start_date, end_date))
        
        # 按股票代码分组
        stocks_data = {}
        for code in df['code'].unique():
            stock_df = df[df['code'] == code].copy()
            stock_df = stock_df.sort_values('date')
            
            # 验证数据完整性：至少30个交易日
            if len(stock_df) >= 30:
                stocks_data[code] = stock_df
            else:
                print(f"警告: {code} 数据不完整 (只有{len(stock_df)}个交易日)，已跳过")
        
        return stocks_data
    
    def _load_parallel(self, start_date: str, end_date: str, chunk_size: int = 100) -> Dict[str, pd.DataFrame]:
        """
        并行加载（优化版本）
        
        策略：
        1. 先获取所有股票代码
        2. 分批并行查询
        3. 使用多进程处理数据分组
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        
        # 1. 获取所有股票代码
        code_query = '''
            SELECT DISTINCT code 
            FROM daily_data 
            WHERE date >= ? AND date <= ?
        '''
        codes_df = pd.read_sql_query(code_query, self.conn, params=(start_date, end_date))
        all_codes = codes_df['code'].tolist()
        
        print(f"找到 {len(all_codes)} 只股票，开始并行加载...")
        
        # 2. 分批加载
        code_chunks = [all_codes[i:i + chunk_size] for i in range(0, len(all_codes), chunk_size)]
        
        # 3. 准备参数
        tasks = [(self.db_path, chunk, start_date, end_date) for chunk in code_chunks]
        
        # 4. 并行处理
        stocks_data = {}
        workers = min(multiprocessing.cpu_count(), len(code_chunks))
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_load_stock_chunk, task): i 
                      for i, task in enumerate(tasks)}
            
            for future in as_completed(futures):
                batch_result = future.result()
                stocks_data.update(batch_result)
                print(f"已加载 {len(stocks_data)}/{len(all_codes)} 只股票")
        
        return stocks_data
    
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日列表
        
        返回:
            list: 按升序排列的交易日期列表
        """
        query = '''
            SELECT DISTINCT date
            FROM daily_data
            WHERE date >= ? AND date <= ?
            ORDER BY date ASC
        '''
        
        df = pd.read_sql_query(query, self.conn, params=(start_date, end_date))
        return df['date'].tolist()
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()


class StrategyInterface:
    """策略接口"""
    
    def __init__(self):
        self.screener = IntegratedScreener()
        self.smc_v2 = SMCLiquidityStrategy()  # 新版SMC策略
    
    def get_signals(self, current_date: str, historical_data: Dict[str, pd.DataFrame], 
                   strategy_name: str = 'liquidity_grab') -> Optional[Signal]:
        """
        获取指定日期的交易信号
        
        参数:
            current_date: 当前日期
            historical_data: 截止到current_date的历史数据
            strategy_name: 策略名称 ('liquidity_grab' 或 'wyckoff_spring')
        
        返回:
            Signal: 置信度最高的交易信号，如果没有信号则返回None
        """
        # 获取候选股票列表
        stock_codes = list(historical_data.keys())

        if not stock_codes:
            return None
        
        # 使用历史数据进行筛选
        results = []
        
        if strategy_name == 'liquidity_grab':
            # 对每只股票使用历史数据进行SMC分析 - 使用V2策略
            for stock_code in stock_codes:
                if stock_code in historical_data:
                    stock_data = historical_data[stock_code]
                    if len(stock_data) >= 60:  # 确保有足够的数据
                        result = self._analyze_stock_with_data_v2(stock_code, stock_data, strategy_name)
                        
                        # V2策略已经内置了看空信号检查，直接使用结果
                        if result and result['signal'] in ['buy', 'strong_buy'] and result['confidence'] >= 60:
                            results.append(result)
        
        elif strategy_name == 'wyckoff_spring':
            # 威科夫策略的历史数据分析
            for stock_code in stock_codes:
                if stock_code in historical_data:
                    stock_data = historical_data[stock_code]
                    if len(stock_data) >= 60:
                        result = self._analyze_stock_with_data_v2(stock_code, stock_data, strategy_name)
                        
                        # 检查看空信号
                        if result and result['signal'] in ['buy', 'strong_buy'] and result['confidence'] >= 0:
                            bearish_check = self._check_bearish_signals(stock_data)
                            
                            if bearish_check['detected']:
                                continue
                            
                            results.append(result)
        else:
            print(f"未知策略: {strategy_name}")
            return None
        
        # 选择置信度最高的信号
        if results and len(results) > 0:
            best_result = max(results, key=lambda x: x['confidence'])
            
            return Signal(
                stock_code=best_result['stock_code'],
                signal_type=best_result['signal'],
                confidence=best_result['confidence'],
                current_price=best_result['current_price'],
                entry_price=best_result['entry_price'],
                stop_loss=best_result['stop_loss'],
                target=best_result['target'],
                date=current_date
            )
        
        return None
    
    def _check_bearish_signals(self, stock_data: pd.DataFrame):
        """
        检查看空信号
        
        返回: {'detected': bool, 'confidence': float, 'reasons': list}
        """
        try:
            from smc_liquidity_strategy import SMCLiquidityStrategy
            smc_strategy = SMCLiquidityStrategy()
            return smc_strategy.detect_bearish_signals(stock_data)
        except Exception as e:
            print(f"检测看空信号时出错: {e}")
            return {'detected': False, 'confidence': 0, 'reasons': []}
    
    def _analyze_stock_with_data_v2(self, stock_code: str, stock_data: pd.DataFrame, strategy_name: str):
        """
        使用V2策略分析股票（更严格的筛选）
        """
        try:
            if strategy_name == 'liquidity_grab':
                # 直接替换数据获取方法，使用全部历史数据
                def mock_get_data(symbol, days=None):
                    return stock_data.copy()
                
                self.smc_v2.data_fetcher.get_stock_data = mock_get_data
                result = self.smc_v2.screen_stock(stock_code)
                return result
            
            elif strategy_name == 'wyckoff_spring':
                # 威科夫策略分析，使用全部历史数据
                from wyckoff_strategy import WyckoffStrategy
                
                def mock_get_data(symbol, days=None):
                    return stock_data.copy()
                
                # 创建策略实例并替换数据获取方法
                if not hasattr(self, 'wyckoff_strategy'):
                    self.wyckoff_strategy = WyckoffStrategy()
                
                self.wyckoff_strategy.data_fetcher.get_stock_data = mock_get_data
                result = self.wyckoff_strategy.wyckoff_accumulation_strategy(stock_code)
                return result
            
            return None
            
        except Exception as e:
            print(f"分析股票 {stock_code} 时出错: {e}")
            return None
    
    
    def close(self):
        """关闭资源"""
        self.screener.close()


class PositionManager:
    """持仓管理器"""
    
    def __init__(self, commission_rate: float = 0.01):
        self.commission_rate = commission_rate
        self.current_position: Optional[Position] = None
    
    def buy(self, stock_code: str, date: str, price: float, 
            stop_loss: float, target: float, capital: float) -> Position:
        """
        执行买入操作
        
        参数:
            stock_code: 股票代码
            date: 买入日期
            price: 买入价格（开盘价）
            stop_loss: 止损价
            target: 目标价
            capital: 可用资金
        
        返回:
            Position: 持仓对象
        """
        # 计算手续费
        commission = capital * self.commission_rate
        available_capital = capital - commission
        
        # 计算可买股数
        shares = available_capital / price
        
        # 总成本（包含手续费）
        cost = capital
        
        position = Position(
            stock_code=stock_code,
            buy_date=date,
            buy_price=price,
            shares=shares,
            stop_loss=stop_loss,
            target=target,
            cost=cost,
            no_rise_days=0,
            last_close=price  # 初始化为买入价
        )
        
        self.current_position = position
        return position
    
    def check_exit(self, position: Position, date: str, 
                   open_price: float, high: float, low: float, close: float,
                   stock_data: pd.DataFrame = None, 
                   all_stocks_data: Dict[str, pd.DataFrame] = None) -> Tuple[bool, Optional[float], Optional[str]]:
        """
        检查是否触发卖出条件
        
        新规则：
        1. 止损：必须当天有买入信号才能卖出
        2. 趋势下破（跌破趋势线/市场结构反转）：必须当天有买入信号才能卖出
        3. 看空信号：可以直接卖出（不需要买入信号）
        
        参数:
            position: 当前持仓
            date: 当前日期
            open_price: 开盘价
            high: 最高价
            low: 最低价
            close: 收盘价
            stock_data: 股票历史数据（用于检测看空信号和趋势反转）
            all_stocks_data: 所有股票的历史数据（用于检查是否有买入信号）
        
        返回:
            (should_exit, exit_price, exit_reason): 是否卖出、卖出价格、卖出原因
        """
        # 买入当天不能卖出（T+1规则）
        if date == position.buy_date:
            return False, None, None
        
        # 检查是否触及止损
        stop_loss_triggered = False
        stop_loss_price = None
        
        if open_price <= position.stop_loss:
            stop_loss_triggered = True
            stop_loss_price = position.stop_loss
        elif low <= position.stop_loss:
            stop_loss_triggered = True
            stop_loss_price = position.stop_loss
        
        # 情况1: 检测看空信号 - 可以直接卖出（优先级最高）
        if stock_data is not None and len(stock_data) >= 30:
            from smc_liquidity_strategy import SMCLiquidityStrategy
            smc_v2 = SMCLiquidityStrategy()
            
            bearish_signals = smc_v2.detect_bearish_signals(stock_data)
            
            # 看空信号置信度必须大于65才能直接卖出
            if bearish_signals['detected'] and bearish_signals['confidence'] > 65:
                # 看空信号可以直接卖出，不需要检查买入信号
                return True, close, f"bearish_signal_{int(bearish_signals['confidence'])}"
        
        # 情况2: 止损或趋势下破 - 必须有买入信号才能卖出
        trend_broken = False
        structure_reversed = False
        
        if stock_data is not None and len(stock_data) >= 30:
            from trend_line_analyzer import TrendLineAnalyzer
            from smc_liquidity_strategy import SMCLiquidityStrategy
            
            # 趋势线分析
            trend_analyzer = TrendLineAnalyzer()
            trend_analysis = trend_analyzer.analyze(stock_data)
            
            smc_v2 = SMCLiquidityStrategy()
            market_structure = smc_v2._check_market_structure_strict(stock_data)
            
            # 检查是否触发趋势下破条件
            trend_broken = trend_analysis['broken_support']
            structure_reversed = not market_structure['is_uptrend']
        
        # 如果触发了止损或趋势下破，需要检查买入信号
        if stop_loss_triggered or trend_broken or structure_reversed:
            has_buy_signal = self._check_has_buy_signal_today(date, all_stocks_data)
            
            if has_buy_signal:
                # 有买入信号，允许卖出（换仓）
                if stop_loss_triggered:
                    return True, stop_loss_price, 'stop_loss'
                elif trend_broken:
                    return True, close, 'broken_trendline'
                elif structure_reversed:
                    return True, close, 'trend_reversal_downtrend'
            else:
                # 没有买入信号，继续持有
                pass
        
        # 更新上一交易日收盘价
        position.last_close = close
        
        return False, None, None
    
    def _check_has_buy_signal_today(self, date: str, all_stocks_data: Dict[str, pd.DataFrame]) -> bool:
        """
        检查当天是否有任何股票存在买入信号
        
        参数:
            date: 当前日期
            all_stocks_data: 所有股票的历史数据
        
        返回:
            bool: 是否有买入信号
        """
        if all_stocks_data is None:
            return False
        
        from smc_liquidity_strategy import SMCLiquidityStrategy
        smc_v2 = SMCLiquidityStrategy()
        
        # 检查所有股票，看是否有买入信号
        for stock_code, stock_data in all_stocks_data.items():
            if stock_code in stock_data and len(stock_data) >= 60:
                # 获取截止到当前日期的数据
                date_mask = stock_data['date'] <= date
                if date_mask.any():
                    historical_data = stock_data[date_mask]
                    
                    if len(historical_data) >= 60:
                        try:
                            # 临时替换数据获取方法
                            original_get_data = smc_v2.data_fetcher.get_stock_data
                            
                            def mock_get_data(symbol, days=300):
                                return historical_data.copy()
                            
                            smc_v2.data_fetcher.get_stock_data = mock_get_data
                            
                            try:
                                result = smc_v2.screen_stock(stock_code)
                                if result and result['signal'] in ['buy', 'strong_buy']:
                                    return True
                            finally:
                                smc_v2.data_fetcher.get_stock_data = original_get_data
                        except:
                            continue
        
        return False
    
    def sell(self, position: Position, date: str, price: float, reason: str) -> Trade:
        """
        执行卖出操作
        
        返回:
            Trade: 完整的交易记录
        """
        # 计算卖出金额
        sell_amount = position.shares * price
        
        # 计算卖出手续费
        sell_commission = sell_amount * self.commission_rate
        
        # 净收入
        net_proceeds = sell_amount - sell_commission
        
        # 买入手续费
        buy_commission = position.cost * self.commission_rate
        
        # 净利润
        profit = net_proceeds - position.cost
        
        # 收益率
        return_rate = profit / position.cost
        
        trade = Trade(
            stock_code=position.stock_code,
            buy_date=position.buy_date,
            buy_price=position.buy_price,
            sell_date=date,
            sell_price=price,
            shares=position.shares,
            buy_commission=buy_commission,
            sell_commission=sell_commission,
            profit=profit,
            return_rate=return_rate,
            exit_reason=reason
        )
        
        # 清空持仓
        self.current_position = None
        
        return trade
    
    def has_position(self) -> bool:
        """检查是否有持仓"""
        return self.current_position is not None


class BacktestEngine:
    """回测引擎（优化版）"""
    
    def __init__(self, initial_capital: float = 1.0, commission_rate: float = 0.01):
        """
        初始化回测引擎
        
        参数:
            initial_capital: 初始资金，默认1.0
            commission_rate: 手续费率，默认0.01 (1%)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.capital = initial_capital
        
        self.data_loader = DataLoader()
        self.strategy_interface = StrategyInterface()
        self.position_manager = PositionManager(commission_rate)
        
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[str, float]] = []
        
        # 性能优化：缓存策略对象
        self._smc_strategy = None
    
    def _get_smc_strategy(self):
        """获取SMC策略对象（单例模式）"""
        if self._smc_strategy is None:
            from smc_liquidity_strategy import SMCLiquidityStrategy
            self._smc_strategy = SMCLiquidityStrategy()
        return self._smc_strategy
    
    def _preprocess_data(self, all_stocks_data: Dict[str, pd.DataFrame], trading_dates: List[str]):
        """
        预处理数据：为每个交易日预先构建历史数据索引
        
        返回:
            dict: {date: {stock_code: row_index}} 每个日期对应的股票数据索引
        """
        print("正在预处理数据索引...")
        
        date_index = {}
        for date in trading_dates:
            date_index[date] = {}
            for code, stock_df in all_stocks_data.items():
                # 找到该日期在DataFrame中的位置
                mask = stock_df['date'] <= date
                if mask.any():
                    date_index[date][code] = mask.sum()  # 记录截止到该日期的行数
        
        print(f"预处理完成，索引了 {len(trading_dates)} 个交易日")
        return date_index
    
    def run(self, start_date: str, end_date: str, strategy_name: str = 'liquidity_grab'):
        """
        运行回测（优化版）
        
        参数:
            start_date: 回测开始日期 (YYYY-MM-DD)
            end_date: 回测结束日期 (YYYY-MM-DD)
            strategy_name: 策略名称 ('liquidity_grab' 或 'wyckoff_spring')
        """
        print("=" * 80)
        print("股票交易策略回测系统（优化版）")
        print("=" * 80)
        print(f"回测时间范围: {start_date} 至 {end_date}")
        print(f"策略名称: {strategy_name}")
        print(f"初始资金: {self.initial_capital:.2f}")
        print(f"手续费率: {self.commission_rate * 100:.1f}%")
        print("=" * 80)
        
        # 记录初始资金
        self.equity_curve.append((start_date, self.capital))
        
        # 加载所有股票数据
        print("\n正在加载历史数据...")
        all_stocks_data = self.data_loader.load_all_stocks_data(start_date, end_date)
        print(f"成功加载 {len(all_stocks_data)} 只股票的历史数据")
        
        # 获取交易日列表
        trading_dates = self.data_loader.get_trading_dates(start_date, end_date)
        print(f"交易日数量: {len(trading_dates)}")
        
        # 预处理数据索引（优化性能）
        date_index = self._preprocess_data(all_stocks_data, trading_dates)
        
        print("\n开始回测...")
        print("-" * 80)
        
        # 遍历每个交易日
        for i, current_date in enumerate(trading_dates):
            # 检查是否有持仓
            if self.position_manager.has_position():
                # 有持仓，检查是否触发卖出
                position = self.position_manager.current_position
                stock_code = position.stock_code
                
                # 获取当日行情（优化：使用索引直接定位）
                if stock_code in all_stocks_data:
                    stock_data = all_stocks_data[stock_code]
                    
                    # 使用预处理的索引快速获取数据
                    if current_date in date_index and stock_code in date_index[current_date]:
                        row_idx = date_index[current_date][stock_code] - 1
                        
                        if row_idx >= 0 and row_idx < len(stock_data):
                            today_row = stock_data.iloc[row_idx]
                            
                            # 确保是当天的数据
                            if today_row['date'] == current_date:
                                open_price = today_row['open']
                                high = today_row['high']
                                low = today_row['low']
                                close = today_row['close']
                                
                                # 获取截止到当前日期的历史数据（使用切片，更快）
                                historical_stock_data = stock_data.iloc[:row_idx+1]
                                
                                # 检查是否触发卖出
                                should_exit, exit_price, exit_reason = self._check_exit_optimized(
                                    position, current_date, open_price, high, low, close, 
                                    historical_stock_data, all_stocks_data
                                )
                                
                                if should_exit:
                                    # 执行卖出
                                    trade = self.position_manager.sell(position, current_date, exit_price, exit_reason)
                                    self.trades.append(trade)
                                    
                                    # 更新资金
                                    self.capital = position.shares * exit_price - trade.sell_commission
                                    
                                    # 记录资金曲线
                                    self.equity_curve.append((current_date, self.capital))
                                    
                                    # 显示卖出原因的中文说明
                                    reason_text = self._format_exit_reason(exit_reason)
                                    
                                    print(f"[{current_date}] 卖出 {stock_code}: {exit_price:.2f} "
                                          f"({reason_text}) 收益率: {trade.return_rate*100:.2f}% "
                                          f"资金: {self.capital:.4f}")
            
            else:
                # 无持仓，寻找买入信号（优化：使用预处理的索引）
                historical_data = {}
                for code in all_stocks_data.keys():
                    if current_date in date_index and code in date_index[current_date]:
                        row_count = date_index[current_date][code]
                        if row_count >= 30:  # 至少30个交易日
                            historical_data[code] = all_stocks_data[code].iloc[:row_count]
                
                # 获取交易信号
                signal = self.strategy_interface.get_signals(current_date, historical_data, strategy_name)
                
                if signal:
                    # 找到下一个交易日
                    if i + 1 < len(trading_dates):
                        next_date = trading_dates[i + 1]
                        stock_code = signal.stock_code
                        
                        # 获取下一交易日的开盘价和行情
                        if stock_code in all_stocks_data:
                            stock_data = all_stocks_data[stock_code]
                            
                            # 使用索引快速定位
                            if next_date in date_index and stock_code in date_index[next_date]:
                                row_idx = date_index[next_date][stock_code] - 1
                                
                                if row_idx >= 0 and row_idx < len(stock_data):
                                    next_row = stock_data.iloc[row_idx]
                                    
                                    if next_row['date'] == next_date:
                                        open_price = next_row['open']
                                        high_price = next_row['high']
                                        low_price = next_row['low']
                                        close_price = next_row['close']
                                        
                                        # 优化：检查开盘价是否合理（避免追高）
                                        signal_price = signal.entry_price
                                        gap_up_ratio = (open_price - signal_price) / signal_price
                                        
                                        # 如果开盘跳空超过2%，等待回调
                                        if gap_up_ratio > 0.02:
                                            # 检查当日是否回调到合理价位
                                            if low_price <= signal_price * 1.015:
                                                actual_entry = (low_price + min(close_price, signal_price * 1.02)) / 2
                                            else:
                                                # 没有回调到合理价位，放弃本次交易
                                                continue
                                        else:
                                            actual_entry = open_price
                                        
                                        # 执行买入
                                        position = self.position_manager.buy(
                                            stock_code, next_date, actual_entry,
                                            signal.stop_loss, signal.target, self.capital
                                        )
                                        
                                        print(f"[{next_date}] 买入 {stock_code}: {actual_entry:.2f} "
                                              f"止损: {signal.stop_loss:.2f} 目标: {signal.target:.2f} "
                                              f"置信度: {signal.confidence:.1f}%")
            
            # 进度显示（减少频率）
            if (i + 1) % 100 == 0:
                print(f"进度: {i+1}/{len(trading_dates)} 交易日 | "
                      f"完成交易: {len(self.trades)} 笔 | 当前资金: {self.capital:.4f}")
        
        print("-" * 80)
        print("回测完成！")
        
        # 关闭资源
        self.data_loader.close()
        self.strategy_interface.close()
    
    def _check_exit_optimized(self, position: Position, date: str, 
                             open_price: float, high: float, low: float, close: float,
                             stock_data: pd.DataFrame,
                             all_stocks_data: Dict[str, pd.DataFrame] = None) -> Tuple[bool, Optional[float], Optional[str]]:
        """
        优化版的退出检查（复用策略对象）
        
        新规则：
        1. 止损：必须当天有买入信号才能卖出
        2. 趋势下破：必须当天有买入信号才能卖出
        3. 看空信号：可以直接卖出
        """
        # 买入当天不能卖出（T+1规则）
        if date == position.buy_date:
            return False, None, None
        
        # 检查是否触及止损
        stop_loss_triggered = False
        stop_loss_price = None
        
        if open_price <= position.stop_loss:
            stop_loss_triggered = True
            stop_loss_price = position.stop_loss
        elif low <= position.stop_loss:
            stop_loss_triggered = True
            stop_loss_price = position.stop_loss
        
        # 情况1: 检测看空信号 - 可以直接卖出（优先级最高）
        if stock_data is not None and len(stock_data) >= 30:
            from trend_line_analyzer import TrendLineAnalyzer
            
            smc_v2 = self._get_smc_strategy()
            
            bearish_signals = smc_v2.detect_bearish_signals(stock_data)
            
            # 看空信号置信度必须大于65才能直接卖出
            if bearish_signals['detected'] and bearish_signals['confidence'] > 65:
                # 看空信号可以直接卖出
                return True, close, f"bearish_signal_{int(bearish_signals['confidence'])}"
            
            # 情况2: 止损或趋势下破 - 必须有买入信号才能卖出
            trend_analyzer = TrendLineAnalyzer()
            trend_analysis = trend_analyzer.analyze(stock_data)
            
            market_structure = smc_v2._check_market_structure_strict(stock_data)
            
            # 检查是否触发趋势下破条件
            trend_broken = trend_analysis['broken_support']
            structure_reversed = not market_structure['is_uptrend']
            
            # 如果触发了止损或趋势下破，需要检查买入信号
            if stop_loss_triggered or trend_broken or structure_reversed:
                has_buy_signal = self._check_has_buy_signal_today_optimized(
                    date, all_stocks_data, smc_v2
                )
                
                if has_buy_signal:
                    # 有买入信号，允许卖出（换仓）
                    if stop_loss_triggered:
                        return True, stop_loss_price, 'stop_loss'
                    elif trend_broken:
                        return True, close, 'broken_trendline'
                    elif structure_reversed:
                        return True, close, 'trend_reversal_downtrend'
                else:
                    # 没有买入信号，继续持有
                    pass
        
        return False, None, None
    
    def _check_has_buy_signal_today_optimized(self, date: str, 
                                             all_stocks_data: Dict[str, pd.DataFrame],
                                             smc_v2) -> bool:
        """
        优化版：检查当天是否有任何股票存在买入信号
        
        优化点：
        1. 复用传入的smc_v2对象
        2. 只检查前N只股票（避免全量扫描）
        3. 找到一个买入信号就立即返回
        """
        if all_stocks_data is None:
            return False
        
        # 限制检查数量，避免性能问题（最多检查50只股票）
        check_count = 0
        max_check = 50
        
        for stock_code, stock_data in all_stocks_data.items():
            if check_count >= max_check:
                break
            
            check_count += 1
            
            if len(stock_data) >= 60:
                # 获取截止到当前日期的数据
                date_mask = stock_data['date'] <= date
                if date_mask.any():
                    historical_data = stock_data[date_mask]
                    
                    if len(historical_data) >= 60:
                        try:
                            # 临时替换数据获取方法
                            original_get_data = smc_v2.data_fetcher.get_stock_data
                            
                            def mock_get_data(symbol, days=300):
                                return historical_data.copy()
                            
                            smc_v2.data_fetcher.get_stock_data = mock_get_data
                            
                            try:
                                result = smc_v2.screen_stock(stock_code)
                                if result and result['signal'] in ['buy', 'strong_buy']:
                                    return True  # 找到买入信号，立即返回
                            finally:
                                smc_v2.data_fetcher.get_stock_data = original_get_data
                        except:
                            continue
        
        return False
    
    def _format_exit_reason(self, exit_reason: str) -> str:
        """格式化退出原因"""
        reason_text = {
            'stop_loss': '止损',
            'target': '止盈',
            'no_rise_3days': '连续3天未上涨',
            'trend_reversal_downtrend': '趋势反转为下降',
            'broken_trendline': '跌破支撑趋势线',
        }.get(exit_reason, exit_reason)
        
        # 处理看空信号的原因
        if exit_reason.startswith('bearish_signal_'):
            confidence = exit_reason.split('_')[-1]
            reason_text = f'看空信号 (置信度{confidence}%)'
        
        return reason_text
    
    def get_results(self) -> Dict:
        """获取回测结果"""
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'initial_capital': self.initial_capital,
            'final_capital': self.capital
        }
