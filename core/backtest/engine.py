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
from core.exit_rules import evaluate_exit
from core.factors.price_limits import limit_prices
import sqlite3
import os
from config import DATABASE_PATH, TrainingConfig
from config.strategy_config import (
    TIME_STOP_DAYS,
    TIME_STOP_MIN_LOSS_PCT,
    TREND_LINE_LONG_PERIOD,
    ENABLE_STOP_LOSS_EXIT,
    ENABLE_TAKE_PROFIT_EXIT,
    ENABLE_SUPPORT_BREAK_EXIT,
    ENABLE_TIME_STOP_EXIT,
    DELIST_EXIT_HAIRCUT,
)


class BacktestEngine:
    """回测引擎"""

    def __init__(
        self,
        strategy: BaseStrategy,
        data_handler: DataHandler,
        initial_capital: float = 1.0,
        commission_rate: float = 0.01,
        max_positions: int = 1,
        buy_cost_rate: float = None,
        sell_cost_rate: float = None,
    ):
        """
        初始化回测引擎

        参数:
            strategy: 策略实例
            data_handler: 数据处理器
            initial_capital: 初始资金
            commission_rate: 手续费率（buy/sell 未显式给出时的回退值）
            max_positions: 最大持仓数
            buy_cost_rate: 买入单边总成本率（佣金+规费+滑点）
            sell_cost_rate: 卖出单边总成本率（佣金+规费+滑点+印花税）
        """
        self.strategy = strategy
        self.data_handler = data_handler
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.buy_cost_rate = buy_cost_rate if buy_cost_rate is not None else commission_rate
        self.sell_cost_rate = sell_cost_rate if sell_cost_rate is not None else commission_rate
        self.max_positions = max_positions

        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            max_positions=max_positions,
            buy_cost_rate=self.buy_cost_rate,
            sell_cost_rate=self.sell_cost_rate,
        )

        self.performance_analyzer = PerformanceAnalyzer()

        # 优化点: 预定义分析器，避免在循环中重复实例化
        try:
            from core.analysis.trend_line_analyzer import TrendLineAnalyzer

            self.trend_analyzer = TrendLineAnalyzer()
        except ImportError:
            self.trend_analyzer = None

        # 回测状态
        self._current_date = None
        self._trading_dates = []
        self._trading_date_index = {}  # {date: idx} O(1) 查找
        self._trend_break_cache = {}  # (stock_code, date) -> result
        self._delist_map = self._load_delist_map()  # {stock_code: outDate}
        # 可成交性统计（cleanup 时打印，用于识别"回测收益里有多少来自买不到/卖不掉的票"）
        self._blocked_entries = 0
        self._blocked_exits = 0

    def run(
        self,
        start_date: str,
        end_date: str,
        stock_codes: List[str] = None,
        verbose: bool = True,
    ) -> Dict:
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
            print(
                f"交易成本: 买入 {self.buy_cost_rate * 100:.3f}% / "
                f"卖出 {self.sell_cost_rate * 100:.3f}% "
                f"(往返 {(self.buy_cost_rate + self.sell_cost_rate) * 100:.3f}%)"
            )
            print(f"最大持仓: {self.max_positions}")
            print("=" * 80)

        # 初始化策略。传入回测股票池，便于策略预加载缓存时按股票池裁剪。
        self.strategy.initialize(stock_codes=stock_codes)

        # 加载数据 (如果尚未加载)
        if not self.data_handler._data_cache:
            if verbose:
                print("\n加载数据...")
            if stock_codes is None:
                conn = sqlite3.connect(DATABASE_PATH)
                stock_codes_df = pd.read_sql_query(
                    f"SELECT DISTINCT code FROM daily_data LIMIT {TrainingConfig.STOCK_NUM}",
                    conn,
                )
                conn.close()
                stock_codes = stock_codes_df["code"].tolist()
            self.data_handler.load_data(start_date, end_date, stock_codes)
        else:
            if verbose:
                print(
                    f"\n跳过加载数据 (已加载 {len(self.data_handler._data_cache)} 只股票)"
                )

        # 获取交易日
        self._trading_dates = self.data_handler.get_trading_dates(start_date, end_date)
        self._trading_date_index = {d: i for i, d in enumerate(self._trading_dates)}
        if verbose:
            print(f"交易日数量: {len(self._trading_dates)}")

        # 主循环
        if verbose:
            print("\n开始回测...")
            print("-" * 80)

        for i, date in enumerate(self._trading_dates):
            self._current_date = date
            if hasattr(self.data_handler, "prune_bar_cache"):
                keep_dates = {date}
                if i + 1 < len(self._trading_dates):
                    keep_dates.add(self._trading_dates[i + 1])
                self.data_handler.prune_bar_cache(keep_dates=keep_dates)

            # 清理昨天的趋势分析缓存，节省内存
            self._trend_break_cache = {}

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
                print(
                    f"进度: {i + 1}/{len(self._trading_dates)} | "
                    f"交易: {len(self.portfolio.trades)} | "
                    f"资金: {self.portfolio.total_value:.4f}"
                )

        if verbose:
            print("-" * 80)
            print("回测完成")
            print(
                f"可成交性拦截: 开盘涨停买不到 {self._blocked_entries} 次 / "
                f"停牌或一字跌停卖不掉 {self._blocked_exits} 次"
            )

        # 清理策略
        self.strategy.cleanup()

        # 计算性能指标
        metrics = self.performance_analyzer.calculate_metrics(
            self.portfolio.trades,
            self.initial_capital,
            self.portfolio.total_value,
            self.portfolio.equity_curve,
        )

        # 打印摘要
        if verbose:
            self.performance_analyzer.print_summary(
                metrics, start_date, end_date, self.strategy.name
            )

        return {
            "metrics": metrics,
            "trades": self.portfolio.trades,
            "equity_curve": self.portfolio.equity_curve,
            "portfolio_state": self.portfolio.get_portfolio_state(),
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
            if signal.signal_type != "buy":
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
            if "confidence" not in metadata:
                metadata["confidence"] = signal.confidence

            # 开仓 (传入计算好的分配资金)
            position = self.portfolio.open_position(
                stock_code=signal.stock_code,
                date=next_date,
                price=entry_price,
                capital_allocation=capital_per_position,
                stop_loss=actual_stop_loss,
                take_profit=actual_take_profit,
                metadata=metadata,
            )

            if position and verbose:
                print(
                    f"[{next_date}] 买入 {signal.stock_code}: {entry_price:.2f} "
                    f"(分配资金: {capital_per_position:.2f}, 置信度: {signal.confidence:.1f}%)"
                )

            # 策略回调
            if position:
                self.strategy.on_trade(position)

            # 如果没满仓，继续检查下一个信号
            if not self.portfolio.can_open_position():
                break

    # ------------------------------------------------------------------
    # 可成交性判定
    #
    # 涨跌停价是 `round(前收 × (1 ± L), 2)` 这个**精确值**。旧实现拿
    # `next_open > signal_price * (1 + MARKET_LIMITS[...])` 去比，有三个问题：
    #   1. 静态 MARKET_LIMITS 没有时间维度 —— 创业板 2020-08-24 由 10% 改 20%、
    #      主板 ST 2026-07-06 由 5% 改 10%，查表一律用错（详见 price_limits.py）。
    #      静态值还被人为削成 0.098/0.198 来容忍舍入，代价是 0.2% 的判定盲区。
    #   2. 用 `>` 而非 `>=`：开盘价正好等于涨停价时判为"没涨停"，直接买进一个
    #      根本买不到的一字板。
    #   3. 只在 open==high==low==close 的"一字板"上检查。开盘封涨停、盘中打开的
    #      情形（open==涨停价 但 low<涨停价）照样按 open 成交 —— 集合竞价买不到。
    # 现在统一改成：拿**未复权**的 raw_preclose 算出精确涨跌停价，与 raw_open /
    # raw_close 直接比。bar 由 BaostockDataHandler 提供，raw_* 一定存在。
    # ------------------------------------------------------------------

    @staticmethod
    def _bar_limits(stock_code: str, date: str, bar: dict):
        """返回该 bar 当日的 (涨停价, 跌停价)，未复权口径。取不到时返回 (None, None)。"""
        prev_close = bar.get("raw_preclose")
        if prev_close is None or not (prev_close > 0):
            return None, None
        return limit_prices(
            stock_code, date, float(prev_close), is_st=int(bar.get("is_st", 0) or 0) == 1
        )

    @staticmethod
    def _is_suspended(bar: dict) -> bool:
        """停牌：成交量为 0，或 tradestatus 明确标记非正常交易。"""
        if not bar.get("volume", 0):
            return True
        ts = bar.get("tradestatus")
        return ts is not None and int(ts) != 1

    def _can_exit_at_close(self, stock_code: str, date: str, bar: dict) -> bool:
        """当日尾盘能否卖出。

        买入侧一直检查停牌与一字涨停，卖出侧却什么都不查 —— 一字跌停、停牌都
        照样按 close 成交。这个不对称是单向的乐观偏差：策略总能在最坏的日子里
        全身而退。这里补齐：停牌、或全天封死跌停（最高价都没离开跌停价），
        视为卖不掉，仓位留到下一个交易日再按同样的规则重新判定。
        """
        if self._is_suspended(bar):
            return False
        _, limit_down = self._bar_limits(stock_code, date, bar)
        if limit_down is None:
            return True
        raw_high = bar.get("raw_high")
        if raw_high is None or not (raw_high > 0):
            return True
        # 全天最高价都没有高于跌停价 ⇒ 一字跌停封死，尾盘挂单排不出去
        return float(raw_high) > limit_down + 1e-9

    def _check_exit_signals(self, date: str, market_data: Dict, verbose: bool):
        """检查平仓信号"""
        positions_to_close = []

        for stock_code, position in self.portfolio.positions.items():
            bar = self.data_handler.get_bar_data(stock_code, date)
            if bar is None:
                exit_info = self._check_delist_exit(stock_code, date)
                if exit_info:
                    positions_to_close.append(exit_info)
                continue

            # 检查退出条件
            should_exit, exit_price, exit_reason = self._check_exit_conditions(
                position, date, bar, market_data
            )

            if should_exit:
                # 触发了退出条件，还要能真的卖得掉
                if not self._can_exit_at_close(stock_code, date, bar):
                    self._blocked_exits += 1
                    continue
                positions_to_close.append((stock_code, exit_price, exit_reason))

        # 执行平仓
        for stock_code, exit_price, exit_reason in positions_to_close:
            trade = self.portfolio.close_position(
                stock_code, date, exit_price, exit_reason
            )

            if trade and verbose:
                print(
                    f"[{date}] 卖出 {stock_code}: {exit_price:.2f} "
                    f"({exit_reason}) 收益: {trade.pnl_pct * 100:.2f}%"
                )

            # 策略回调
            if trade:
                self.strategy.on_trade(trade)

    def _check_exit_conditions(
        self, position, date: str, bar: dict, market_data: Dict
    ) -> tuple:
        """
        检查退出条件

        返回:
            (should_exit, exit_price, exit_reason)
        """
        # T+1规则
        if date == position.entry_date:
            return False, None, None

        close = bar["close"]

        # 计算价格变化率
        buy_price_abs = abs(position.entry_price)
        if buy_price_abs == 0:
            return False, None, None

        decision = evaluate_exit(
            current_price=close,
            entry_price=position.entry_price,
            holding_days=position.holding_days,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            enable_stop_loss=ENABLE_STOP_LOSS_EXIT,
            enable_take_profit=ENABLE_TAKE_PROFIT_EXIT,
            enable_time_stop=ENABLE_TIME_STOP_EXIT,
            time_stop_days=TIME_STOP_DAYS,
            time_stop_max_return_pct=TIME_STOP_MIN_LOSS_PCT,
        )
        if decision.should_exit:
            return True, close, decision.reason

        # 趋势破位检查
        if ENABLE_SUPPORT_BREAK_EXIT and self._check_trend_break(
            position.stock_code, date, market_data
        ):
            return True, close, "trend_break"

        return False, None, None

    def _check_trend_break(self, stock_code: str, date: str, market_data: Dict) -> bool:
        """检查趋势破位（优化后的缓存版本）"""
        if self.trend_analyzer is None:
            return False

        # 优化点: 使用日内缓存，如果是同一日重复检查相同标的，直接返回
        cache_key = (stock_code, date)
        if cache_key in self._trend_break_cache:
            return self._trend_break_cache[cache_key]

        # 获取历史数据
        hist_data = self.data_handler.get_historical_data(
            stock_code, date, lookback_days=TREND_LINE_LONG_PERIOD
        )
        if hist_data is None or len(hist_data) < 30:
            self._trend_break_cache[cache_key] = False
            return False

        try:
            result = self.trend_analyzer.analyze(hist_data)
            broken = result.get("broken_support", False)
            self._trend_break_cache[cache_key] = broken
            return broken
        except Exception:
            self._trend_break_cache[cache_key] = False
            return False

    def _load_delist_map(self) -> Dict[str, str]:
        """从元数据库加载退市日期映射 {code: outDate}"""
        meta_db = os.path.join(os.path.dirname(DATABASE_PATH), "stock_meta.db")
        if not os.path.exists(meta_db):
            return {}
        try:
            conn = sqlite3.connect(meta_db)
            df = pd.read_sql_query(
                "SELECT code, outDate FROM stock_basic WHERE outDate IS NOT NULL AND outDate != ''",
                conn,
            )
            conn.close()
            delist_map = dict(zip(df["code"], df["outDate"]))
            if delist_map:
                print(f"  [INFO] 已加载 {len(delist_map)} 只股票退市日期")
            return delist_map
        except Exception as e:
            print(f"  [WARN] 加载退市日期失败: {e}")
            return {}

    def _check_delist_exit(self, stock_code: str, current_date: str) -> Optional[tuple]:
        """检查股票是否已退市，若是则返回 (stock_code, exit_price, 'delist')

        退市不是按最后一个可见价平价了结的。旧实现直接用 position.current_price
        （停牌前最后一根 K 线的收盘价）记账，等于假设退市股能原价卖出 —— 现实中
        退市整理期普遍腰斩以上，进老三板后流动性接近于零。这里按
        `DELIST_EXIT_HAIRCUT` 打折，把这块损失显式记进回测，而不是当它不存在。
        """
        out_date = self._delist_map.get(stock_code)
        if not out_date or current_date < out_date:
            return None
        position = self.portfolio.positions.get(stock_code)
        if not position:
            return None
        last_price = (
            position.current_price
            if position.current_price != 0
            else position.entry_price
        )
        exit_price = last_price * (1.0 - DELIST_EXIT_HAIRCUT)
        return (stock_code, exit_price, "delist")

    def _get_next_entry_price(
        self, current_date: str, stock_code: str, signal_price: float
    ) -> tuple:
        """
        获取下一交易日的入场价格

        返回:
            (next_date, entry_price)
        """
        # 找到下一交易日
        current_idx = self._trading_date_index.get(current_date)
        if current_idx is None or current_idx + 1 >= len(self._trading_dates):
            return None, None

        next_date = self._trading_dates[current_idx + 1]

        # 获取下一交易日行情
        bar = self.data_handler.get_bar_data(stock_code, next_date)
        if bar is None:
            return None, None

        # 停牌检测（成交量为 0 或 tradestatus 非正常）
        if self._is_suspended(bar):
            return None, None

        # 开盘涨停检测：开盘价触及涨停价即视为买不到。
        # 这既覆盖一字板，也覆盖「开盘封板、盘中打开」—— 我们的委托是在集合竞价
        # 阶段发出的，封板价上排队买不到，盘中是否打开与这一笔无关。
        limit_up, _ = self._bar_limits(stock_code, next_date, bar)
        raw_open = bar.get("raw_open")
        if limit_up is not None and raw_open is not None and raw_open > 0:
            if float(raw_open) >= limit_up - 1e-9:
                self._blocked_entries += 1
                return None, None

        return next_date, bar["open"]

    def get_results(self) -> Dict:
        """获取回测结果"""
        metrics = self.performance_analyzer.calculate_metrics(
            self.portfolio.trades,
            self.initial_capital,
            self.portfolio.total_value,
            self.portfolio.equity_curve,
        )

        return {
            "metrics": metrics,
            "trades": self.portfolio.trades,
            "equity_curve": self.portfolio.equity_curve,
            "portfolio_state": self.portfolio.get_portfolio_state(),
        }
