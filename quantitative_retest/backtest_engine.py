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
from config.strategy_config import MIN_ENTRY_QUALITY, OPEN_WAIT_THRESHOLD
from config import DATABASE_PATH
from core.analysis import PriceActionAnalyzer

from scripts.integrated_screener import IntegratedScreener
from core.strategies.smc_liquidity_strategy import SMCLiquidityStrategy


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
    strategy_type: str = 'normal'  # 策略类型：'normal'（正常）或'bottom_reversal'（底部反转）
    holding_days: int = 0  # 持仓天数
    max_profit_rate: float = 0.0  # 持仓期间最大盈利率


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

    def _check_bearish_signals(self, stock_data: pd.DataFrame, trend_strength=None, historical_signals=None, is_bottom_strategy=False):
        """
        检查看空信号
        
        参数:
            stock_data: 股票数据
            trend_strength: 趋势强度（0-100）
            historical_signals: 历史看空信号列表
            is_bottom_strategy: 是否为底部策略（如Wyckoff），底部策略会忽略顶部看空信号
        
        返回: {'detected': bool, 'confidence': float, 'reasons': list, 'threshold': float, 'top_warning': bool}
        """
        try:
            from core.strategies. smc_liquidity_strategy import SMCLiquidityStrategy
            smc_strategy = SMCLiquidityStrategy()
            return smc_strategy.detect_bearish_signals(stock_data, trend_strength, historical_signals, is_bottom_strategy)
        except Exception as e:
            print(f"检测看空信号时出错: {e}")
            return {'detected': False, 'confidence': 0, 'reasons': [], 'threshold': 60, 'top_warning': False}
    
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
                from core.strategies import WyckoffStrategy
                
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
            stop_loss: float, target: float, capital: float, strategy_type: str = 'normal') -> Position:
        """
        执行买入操作
        
        参数:
            stock_code: 股票代码
            date: 买入日期
            price: 买入价格（开盘价）
            stop_loss: 止损价
            target: 目标价
            capital: 可用资金
            strategy_type: 策略类型，'normal'（正常）或'bottom_reversal'（底部反转）
        
        返回:
            Position: 持仓对象
        """
        # 处理负价格：使用绝对值确保计算正确
        abs_price = abs(price)
        
        # 计算手续费
        commission = capital * self.commission_rate
        available_capital = capital - commission
        
        # 计算可买股数（使用绝对价格）
        shares = available_capital / abs_price if abs_price > 0 else 0
        
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
            last_close=price,  # 初始化为买入价
            strategy_type=strategy_type,  # 添加策略类型
            holding_days=0,  # 初始化持仓天数
            max_profit_rate=0.0  # 初始化最大盈利率
        )
        
        self.current_position = position
        return position

    
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
        
        from  core.strategies.smc_liquidity_strategy import SMCLiquidityStrategy
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
        # 处理负价格：使用价格变化率计算收益
        # 前复权数据可能为负，但价格变化率仍然有效
        
        # 计算价格变化率（相对买入价的涨跌幅）
        if position.buy_price != 0:
            price_change_rate = (price - position.buy_price) / abs(position.buy_price)
        else:
            price_change_rate = 0
        
        # 使用价格变化率计算卖出金额
        # 卖出金额 = 成本 * (1 + 价格变化率)
        sell_amount = position.cost * (1 + price_change_rate)
        
        # 计算卖出手续费
        sell_commission = abs(sell_amount) * self.commission_rate
        
        # 净收入（扣除卖出手续费）
        net_proceeds = sell_amount - sell_commission
        
        # 买入手续费
        buy_commission = position.cost * self.commission_rate
        
        # 净利润 = 净收入 - 成本
        profit = net_proceeds - position.cost
        
        # 收益率（避免除以零）
        if position.cost != 0:
            return_rate = profit / position.cost
        else:
            return_rate = 0
        
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
            from  core.strategies.smc_liquidity_strategy import SMCLiquidityStrategy
            self._smc_strategy = SMCLiquidityStrategy()
        return self._smc_strategy
    
    def _preprocess_data(self, all_stocks_data: Dict[str, pd.DataFrame], trading_dates: List[str]):
        """
        预处理数据：为每个交易日预先构建历史数据索引（优化版）
        
        返回:
            dict: {date: {stock_code: row_index}} 每个日期对应的股票数据索引
        """
        print("正在预处理数据索引（优化版）...")
        
        # 优化1：将交易日转换为集合，加速查找
        trading_dates_set = set(trading_dates)
        
        # 优化2：为每只股票预先建立日期到索引的映射
        stock_date_indices = {}
        for code, stock_df in all_stocks_data.items():
            # 只为交易日范围内的数据建立索引
            stock_df_dates = stock_df['date'].values
            date_to_idx = {}
            cumulative_idx = 0
            
            for i, date in enumerate(stock_df_dates):
                cumulative_idx = i + 1
                # 只记录交易日的索引
                if date in trading_dates_set:
                    date_to_idx[date] = cumulative_idx
            
            # 为每个交易日填充索引（使用最近的历史数据）
            last_idx = 0
            for trade_date in trading_dates:
                if trade_date in date_to_idx:
                    last_idx = date_to_idx[trade_date]
                elif trade_date > stock_df_dates[0]:  # 确保交易日在股票数据范围内
                    # 使用最近的历史索引
                    date_to_idx[trade_date] = last_idx
            
            stock_date_indices[code] = date_to_idx
        
        # 优化3：重组为按日期索引的结构
        date_index = {}
        for date in trading_dates:
            date_index[date] = {}
            for code, date_to_idx in stock_date_indices.items():
                if date in date_to_idx and date_to_idx[date] > 0:
                    date_index[date][code] = date_to_idx[date]
        
        print(f"预处理完成，索引了 {len(trading_dates)} 个交易日，{len(all_stocks_data)} 只股票")
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
            sold_today = False  # 标记当天是否卖出
            
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
                                    
                                    # 更新资金：使用交易的净收入
                                    # 净收入 = 成本 + 利润（已经在sell方法中计算好）
                                    self.capital = position.cost + trade.profit
                                    
                                    # 记录资金曲线
                                    self.equity_curve.append((current_date, self.capital))
                                    
                                    # 显示卖出原因的中文说明
                                    reason_text = self._format_exit_reason(exit_reason)
                                    
                                    print(f"[{current_date}] 卖出 {stock_code}: {exit_price:.2f} "
                                          f"({reason_text}) 收益率: {trade.return_rate*100:.2f}% "
                                          f"资金: {self.capital:.4f}")
                                    
                                    sold_today = True  # 标记已卖出
            
            # 如果无持仓或当天卖出了，寻找买入信号
            if not self.position_manager.has_position():
                # 构建历史数据（优化：使用预处理的索引）
                historical_data = {}
                for code in all_stocks_data.keys():
                    if current_date in date_index and code in date_index[current_date]:
                        row_count = date_index[current_date][code]
                        if row_count >= 30:  # 至少30个交易日
                            historical_data[code] = all_stocks_data[code].iloc[:row_count]
                
                # 获取交易信号
                signal = self.strategy_interface.get_signals(current_date, historical_data, strategy_name)
                
                if signal:
                    # 如果当天卖出，当天就可以买入（使用当天收盘价）
                    
                    if sold_today:
                        if stock_code==signal.stock_code:
                            continue
                        stock_code = signal.stock_code
                        
                        # 获取当天的收盘价作为买入价
                        if stock_code in all_stocks_data:
                            stock_data = all_stocks_data[stock_code]
                            
                            if current_date in date_index and stock_code in date_index[current_date]:
                                row_idx = date_index[current_date][stock_code] - 1
                                
                                if row_idx >= 0 and row_idx < len(stock_data):
                                    today_row = stock_data.iloc[row_idx]
                                    
                                    if today_row['date'] == current_date:
                                        close_price = today_row['close']
                                        
                                        # 判断策略类型
                                        strategy_type = signal.analysis.get('strategy_type', 'normal') if hasattr(signal, 'analysis') and signal.analysis else 'normal'
                                        
                                        # 执行买入（使用收盘价）
                                        position = self.position_manager.buy(
                                            stock_code, current_date, close_price,
                                            signal.stop_loss, signal.target, self.capital,
                                            strategy_type=strategy_type
                                        )
                                        
                                        print(f"[{current_date}] 买入 {stock_code}: {close_price:.2f} "
                                              f"止损: {signal.stop_loss:.2f} 目标: {signal.target:.2f} "
                                              f"置信度: {signal.confidence:.1f}% (当天换仓)")
                    
                    # 否则，找到下一个交易日买入
                    elif i + 1 < len(trading_dates):
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
                                        if gap_up_ratio > OPEN_WAIT_THRESHOLD:
                                            # 检查当日是否回调到合理价位
                                            if low_price <= signal_price * 1.015:
                                                actual_entry = (low_price + min(close_price, signal_price * 1.02)) / 2
                                            else:
                                                # 没有回调到合理价位，放弃本次交易
                                                continue
                                        else:
                                            actual_entry = open_price
                                        
                                        # 判断策略类型
                                        strategy_type = signal.analysis.get('strategy_type', 'normal') if hasattr(signal, 'analysis') and signal.analysis else 'normal'
                                        
                                        # 执行买入
                                        position = self.position_manager.buy(
                                            stock_code, next_date, actual_entry,
                                            signal.stop_loss, signal.target, self.capital,
                                            strategy_type=strategy_type
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
        
        规则：
        1. 止损：直接卖出（始终启用）
        2. 止盈：根据ENABLE_TAKE_PROFIT_EXIT参数决定
        3. 跌破支撑：根据ENABLE_SUPPORT_BREAK_EXIT参数决定
        4. 看空信号：根据ENABLE_BEARISH_SIGNAL_EXIT参数决定
        5. 时间止损：根据ENABLE_TIME_STOP_EXIT参数决定
        """
        # 导入配置参数
        from  config.strategy_config import (
            ENABLE_TAKE_PROFIT_EXIT, 
            ENABLE_BEARISH_SIGNAL_EXIT,
            ENABLE_SUPPORT_BREAK_EXIT,
            ENABLE_TIME_STOP_EXIT,
            TIME_STOP_DAYS,
            TIME_STOP_MIN_LOSS_PCT
        )
        
        # 买入当天不能卖出（T+1规则）
        if date == position.buy_date:
            return False, None, None
        
        # 更新持仓天数
        position.holding_days += 1
        
        # 更新最大盈利率
        current_profit_rate = (close - position.buy_price) / abs(position.buy_price)
        if current_profit_rate > position.max_profit_rate:
            position.max_profit_rate = current_profit_rate
        
        # 检查是否触及止损（始终启用）
        # 使用价格变化率判断，避免负价格导致的比较错误
        stop_loss_triggered = False
        stop_loss_price = None
        
        # 计算价格相对买入价的变化率
        buy_price_abs = abs(position.buy_price)
        if buy_price_abs > 0:
            open_change_rate = (open_price - position.buy_price) / buy_price_abs
            low_change_rate = (low - position.buy_price) / buy_price_abs
            high_change_rate = (high - position.buy_price) / buy_price_abs
            stop_loss_rate = (position.stop_loss - position.buy_price) / buy_price_abs
            target_rate = (position.target - position.buy_price) / buy_price_abs if hasattr(position, 'target') else 0
            
            # 止损触发：价格变化率 <= 止损变化率
            if open_change_rate <= stop_loss_rate:
                stop_loss_triggered = True
                stop_loss_price = position.stop_loss
            elif low_change_rate <= stop_loss_rate:
                stop_loss_triggered = True
                stop_loss_price = position.stop_loss
        
        # 如果触发止损，直接卖出
        if stop_loss_triggered:
            return True, stop_loss_price, 'stop_loss'
        
        # 检查是否触及止盈（根据开关决定）
        if ENABLE_TAKE_PROFIT_EXIT and hasattr(position, 'target'):
            # 止盈触发：价格变化率 >= 目标变化率
            if buy_price_abs > 0 and high_change_rate >= target_rate:
                return True, position.target, 'target'
        
        # 检查时间止损（根据开关决定）
        if ENABLE_TIME_STOP_EXIT:
            # 条件1：持仓天数超过阈值
            # 条件2：从未盈利（最大盈利率 <= 0）
            # 条件3：当前亏损超过最小亏损比例（避免微小亏损就卖出）
            if (position.holding_days >= TIME_STOP_DAYS and 
                position.max_profit_rate <= 0 and 
                current_profit_rate <= TIME_STOP_MIN_LOSS_PCT):
                return True, close, f'time_stop_{position.holding_days}days'
        
        # 检测看空信号和趋势下破
        if stock_data is not None and len(stock_data) >= 30:
            from core.analysis.trend_line_analyzer import TrendLineAnalyzer
            
            smc_v2 = self._get_smc_strategy()
            
            # 获取趋势强度
            market_structure = smc_v2._check_market_structure_strict(stock_data)
            trend_strength = market_structure.get('trend_strength', 0)
            
            # 获取历史看空信号（从持仓信息中）
            historical_signals = getattr(position, 'bearish_history', [])
            
            # 判断是否为底部策略
            is_bottom_strategy = getattr(position, 'strategy_type', 'normal') == 'bottom_reversal'
            
            bearish_signals = smc_v2.detect_bearish_signals(stock_data, trend_strength, historical_signals, is_bottom_strategy)
            
            # 保存当前看空信号到历史记录
            if bearish_signals['confidence'] > 0:
                if not hasattr(position, 'bearish_history'):
                    position.bearish_history = []
                position.bearish_history.append({
                    'date': date,
                    'confidence': bearish_signals['confidence']
                })
            
            # 看空信号卖出（根据开关决定）
            if ENABLE_BEARISH_SIGNAL_EXIT:
                # 动态阈值判断（使用返回的threshold）
                threshold = bearish_signals.get('threshold', 60)
                if bearish_signals['detected'] and bearish_signals['confidence'] >= threshold:
                    # 看空信号直接卖出
                    reasons = ','.join(bearish_signals['reasons'][:3])  # 取前3个原因
                    return True, close, f"bearish_{int(bearish_signals['confidence'])}_{reasons}"
            
            # 趋势线分析
            trend_analyzer = TrendLineAnalyzer()
            trend_analysis = trend_analyzer.analyze(stock_data)
            
            # 检查是否触发趋势下破条件
            trend_broken = trend_analysis['broken_support']
            structure_reversed = not market_structure['is_uptrend']
            _=PriceActionAnalyzer()
            # 跌破支撑卖出（根据开关决定）
            if ENABLE_SUPPORT_BREAK_EXIT:
                if trend_broken and (not (_._check_entry_timing(stock_data, close)['is_good'])):
                    return True, close, 'broken_trendline'
                # elif structure_reversed:
                #     return True, close, 'trend_reversal_downtrend'
        
        return False, None, None
    
    
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
        
        # 处理时间止损的原因
        if exit_reason.startswith('time_stop_'):
            days = exit_reason.split('_')[-1].replace('days', '')
            reason_text = f'时间止损 ({days}天未盈利)'
        
        return reason_text
    
    def get_results(self) -> Dict:
        """获取回测结果"""
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'initial_capital': self.initial_capital,
            'final_capital': self.capital
        }
