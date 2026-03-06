"""
交易控制中心 (Execution Engine)

负责根据选股信号计算仓位，并与 trader_interface 交互以执行买卖操作。
实现：开盘买入、盘中维持、尾盘卖出的闭环流程。

执行规则（与回测严格对齐）：
  1. 买入：在开盘时间窗（09:20~09:26）挂涨停价委托，确保以开盘价附近成交。
     对应回测：next_day_open 成交
  2. 卖出：仅在尾盘时间窗（14:50~14:57）检查退出条件，触发则挂跌停价委托，
     确保以当日收盘价附近成交。
     对应回测：以 close / stop_loss / take_profit / time_stop 价格成交（均在尾盘）
  3. T+1 规则：当日买入的股票不能当日卖出。
  4. 时间止损：持有 >= AUTO_TIME_STOP_DAYS 个交易日 且 浮亏 >= AUTO_TIME_STOP_MIN_LOSS_PCT
     才触发（对齐回测 TIME_STOP_DAYS + TIME_STOP_MIN_LOSS_PCT 双条件）。
"""

import sys
import os
import time
import json
import logging
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.automation.trader_interface import AutoTrader
from config.automation_config import (
    MAX_POSITIONS_AUTO, SINGLE_BUY_RATIO, CASH_BUFFER,
    BUY_WINDOW_START, BUY_WINDOW_END, SELL_WINDOW_START, SELL_WINDOW_END,
    AUTO_TIME_STOP_DAYS, AUTO_TIME_STOP_MIN_LOSS_PCT,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', "controller.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ExecutionController")


def _calc_limit_up_price(ref_price: float, is_st: bool = False) -> float:
    """
    计算涨停价（用于买入委托，确保排队靠前）。
    规则：普通股 +10%，ST股 +5%，向下取整到分（0.01精度）。
    """
    rate = 0.05 if is_st else 0.10
    raw = ref_price * (1 + rate)
    return round(int(raw * 100) / 100, 2)  # 向下取整到分


def _calc_limit_down_price(ref_price: float, is_st: bool = False) -> float:
    """
    计算跌停价（用于卖出委托，确保尾盘成交）。
    规则：普通股 -10%，ST股 -5%，向上取整到分（0.01精度）。
    """
    rate = 0.05 if is_st else 0.10
    raw = ref_price * (1 - rate)
    return round((int(raw * 100) + 1) / 100, 2)  # 向上取整到分，确保不低于跌停价


def _get_trading_days_count(entry_date_str: str, today_str: str, db_path: str) -> int:
    """
    计算两个日期之间的交易日数量（含 entry_date，不含 today_str）。
    使用数据库 daily_data 表的日期近似（以有记录的日期为交易日）。
    若查询失败，退回自然日计算。
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(DISTINCT date) FROM daily_data
            WHERE date >= ? AND date < ?
            """,
            (entry_date_str, today_str)
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row and row[0] is not None else 0
    except Exception as e:
        logger.warning(f"查询交易日数量失败，退回自然日计算: {e}")
        try:
            entry_dt = datetime.strptime(entry_date_str, "%Y-%m-%d")
            today_dt = datetime.strptime(today_str, "%Y-%m-%d")
            return (today_dt - entry_dt).days
        except Exception:
            return 0


class ExecutionController:
    """
    交易执行控制核心。
    
    买入逻辑（对齐回测）：
      - 在开盘时间窗内挂涨停价限价买入，保证以开盘价成交（回测用 next_day_open）。
    
    卖出逻辑（对齐回测）：
      - 仅在尾盘时间窗内检查退出条件（止损/止盈/时间止损）。
      - 触发条件后挂跌停价限价卖出，保证当日以收盘价附近成交。
      - 时间止损必须同时满足：持有天数 >= 阈值 且 亏损比例 >= 阈值（双条件）。
    """

    def __init__(self, trader: AutoTrader):
        self.trader = trader
        self.signals_cache = []  # 存储待执行的买入信号

        # 本地状态追踪
        self.tracking_file = os.path.join(PROJECT_ROOT, "data", "automation", "tracking.json")
        self.tracking_data = self._load_tracking()

        # 数据库路径（用于交易日计算）
        from config import DATABASE_PATH
        self._db_path = DATABASE_PATH

    def _load_tracking(self):
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "positions" not in data: data["positions"] = {}
                    if "processed_today" not in data: data["processed_today"] = []
                    return data
            except Exception as e:
                logger.error(f"加载 tracking 文件失败: {e}")
        return {"current_day": "", "pending_buys": [], "processed_today": [], "positions": {}}

    def _save_tracking(self):
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            json.dump(self.tracking_data, f, ensure_ascii=False, indent=4)

    def set_buy_signals(self, signals: List[Dict]):
        """设置今日待执行的买入信号"""
        today = datetime.now().strftime("%Y-%m-%d")
        self.signals_cache = signals

        # 如果是新的一天，重置 processed_today
        if self.tracking_data.get("current_day") != today:
            self.tracking_data["current_day"] = today
            self.tracking_data["processed_today"] = []

        self.tracking_data["pending_buys"] = [s['stock_code'] for s in signals]
        self._save_tracking()
        logger.info(f"已加载今日买入信号: {[s['stock_code'] for s in signals]}")

    def execute_buys(self):
        """
        执行买入任务（开盘时间窗内运行）。
        
        委托价格策略：挂涨停价限价委托（对应回测以开盘价成交的逻辑）。
        涨停价 = 前收盘价 * 1.10（普通股）/ 1.05（ST股）。
        由于 GUI 下单时客户端通常已显示当日行情，current_price 即为前一日收盘价（信号生成时的价格），
        用其计算涨停价作为委托价上限，确保在集合竞价或开盘时以开盘实际价格成交。
        """
        if not self.signals_cache:
            logger.info("队列中无待买入股票。")
            return

        balance = self.trader.get_balance()
        available_cash = float(balance.get('可用', balance.get('可用余额', balance.get('可用金额', 0))))
        logger.info(f"当前可用资金: {available_cash:.2f}")

        if available_cash < 1000:
            logger.warning("资金不足 1000 元，暂不执行买入。")
            return

        # 检查当前持仓
        positions = self.trader.get_positions()
        holding_codes = [p.get('证券代码', p.get('stock_code', '')) for p in positions]

        # 过滤已处理或已持有的信号
        targets = []
        for s in self.signals_cache:
            code = s['stock_code']
            if any(code in h or h in code for h in holding_codes):
                logger.info(f"  {code} 已在持仓中，跳过。")
                continue
            if code in self.tracking_data["processed_today"]:
                logger.info(f"  {code} 今日已处理，跳过。")
                continue
            targets.append(s)

        if not targets:
            logger.info("所有信号已处理或已在持仓中。")
            return

        # 按信号数量均分预算（对齐回测等权分配：capital_per_position = total_value / max_positions）
        budget_per_stock = min(
            available_cash / len(targets),
            available_cash * SINGLE_BUY_RATIO
        )
        budget_per_stock = max(0, budget_per_stock - CASH_BUFFER)

        for signal in targets:
            code = signal['stock_code']
            ref_price = signal.get('current_price', 0)
            if ref_price <= 0:
                logger.warning(f"  {code}: 参考价格无效 ({ref_price})，跳过。")
                continue

            # ---------------------------------------------------------------
            # 【核心】买入委托价 = 涨停价（确保开盘成交，对齐回测 next_day_open）
            # ---------------------------------------------------------------
            is_st = signal.get('is_st', False)
            limit_up_price = _calc_limit_up_price(ref_price, is_st=is_st)

            # 以涨停价估算可购买手数（实际成交后以开盘价计算，此处用上限做保守估算）
            volume = int((budget_per_stock / limit_up_price) / 100) * 100

            if volume >= 100:
                logger.info(f"  执行买入委托: {code} | 数量: {volume}股 | "
                            f"委托价(涨停): {limit_up_price:.2f} | 参考价: {ref_price:.2f}")
                res = self.trader.buy(code, amount=volume, price=limit_up_price)

                if res.get('status') == 'success' or res.get('entrust_no'):
                    self.tracking_data["processed_today"].append(code)
                    self.tracking_data["positions"][code] = {
                        "entry_date": datetime.now().strftime("%Y-%m-%d"),
                        # 以参考价（信号生成时的收盘价）记录，对齐回测 next_day_open 的 entry_price 记录逻辑
                        # （实际成交价应为开盘价，但 GUI 难以实时获取，用参考价近似）
                        "entry_price": ref_price,
                        "stop_loss": signal.get('stop_loss'),
                        "take_profit": signal.get('take_profit'),
                        "confidence": signal.get('confidence'),
                        "is_st": is_st,
                    }
                    self._save_tracking()
                    logger.info(f"  {code} 买入委托成功，记录持仓跟踪。")
                else:
                    logger.warning(f"  {code} 买入委托失败: {res}")
            else:
                logger.warning(f"  {code}: 预算 {budget_per_stock:.2f} 不足一手（涨停价 {limit_up_price:.2f}），跳过。")

    def execute_sells(self):
        """
        执行卖出任务（尾盘时间窗内运行）。
        
        退出条件（完全对齐回测 _check_exit_conditions）：
          1. 止损：当前价 <= stop_loss
          2. 止盈：当前价 >= take_profit
          3. 时间止损：持有 >= AUTO_TIME_STOP_DAYS 交易日 且 亏损 >= AUTO_TIME_STOP_MIN_LOSS_PCT
        
        委托价格策略：挂跌停价限价卖出（确保在尾盘集合竞价前成交，对应回测以 close 成交）。
        """
        from config.strategy_config import (
            ENABLE_STOP_LOSS_EXIT, ENABLE_TAKE_PROFIT_EXIT, ENABLE_TIME_STOP_EXIT
        )

        positions = self.trader.get_positions()
        if not positions:
            logger.info("当前无持仓。")
            return

        today_str = datetime.now().strftime("%Y-%m-%d")

        for p in positions:
            code = p.get('证券代码', p.get('stock_code', ''))
            # 取纯6位代码（部分券商带后缀）
            base_code = code[:6] if len(code) >= 6 else code

            # T+1 保护：今日买入不能卖出
            if base_code in self.tracking_data.get("processed_today", []):
                logger.info(f"  {code} 今日买入（T+1 保护），跳过卖出。")
                continue

            # 获取元数据
            meta = self.tracking_data["positions"].get(base_code, {})
            if not meta:
                # 历史遗留无元数据持仓：尾盘清仓（兜底处理）
                logger.info(f"  {code} 无跟踪元数据，执行兜底尾盘清仓。")
                self._do_sell(code, ref_price=None, is_st=False)
                continue

            # 当前价格（券商返回字段因券商而异）
            current_price = float(p.get('当前价', p.get('市价', p.get('现价', 0))) or 0)
            print(f"{'='*60}")
            print(f"当前价格：{current_price}")
            print(f"{'='*60}")
            entry_price = float(meta.get('entry_price') or 0)
            entry_date_str = meta.get('entry_date', '')
            is_st = bool(meta.get('is_st', False))

            if current_price <= 0 or entry_price <= 0:
                logger.warning(f"  {code}: 价格数据无效 (current={current_price}, entry={entry_price})，跳过。")
                continue

            # ---------------------------------------------------------------
            # 持有天数（交易日计数，对齐回测 position.holding_days）
            # ---------------------------------------------------------------
            holding_days = _get_trading_days_count(entry_date_str, today_str, self._db_path)

            # 浮亏/浮盈比例（对齐回测 position.unrealized_pnl_pct）
            unrealized_pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0

            should_exit = False
            reason = ""

            # ---------------------------------------------------------------
            # 1. 止损检查（对齐回测: low <= stop_loss → 以 stop_loss 成交）
            # ---------------------------------------------------------------
            sl = meta.get('stop_loss')
            if ENABLE_STOP_LOSS_EXIT and sl and current_price <= float(sl):
                should_exit = True
                reason = "stop_loss"

            # ---------------------------------------------------------------
            # 2. 止盈检查（对齐回测: high >= take_profit → 以 take_profit 成交）
            # ---------------------------------------------------------------
            tp = meta.get('take_profit')
            if ENABLE_TAKE_PROFIT_EXIT and tp and current_price >= float(tp):
                should_exit = True
                reason = "take_profit"

            # ---------------------------------------------------------------
            # 3. 时间止损（对齐回测双条件：天数 + 亏损门槛）
            #    回测条件: holding_days >= TIME_STOP_DAYS
            #              AND unrealized_pnl_pct <= TIME_STOP_MIN_LOSS_PCT
            # ---------------------------------------------------------------
            if (ENABLE_TIME_STOP_EXIT
                    and holding_days >= AUTO_TIME_STOP_DAYS
                    and unrealized_pnl_pct <= AUTO_TIME_STOP_MIN_LOSS_PCT):
                should_exit = True
                reason = "time_stop"

            # ---------------------------------------------------------------
            # 执行卖出：挂跌停价委托（确保尾盘以收盘价附近成交）
            # ---------------------------------------------------------------
            if should_exit:
                logger.info(
                    f"  卖出信号触发: {code} | 原因: {reason} | "
                    f"当前价: {current_price:.2f} | 入场价: {entry_price:.2f} | "
                    f"持有: {holding_days} 交易日 | 浮盈: {unrealized_pnl_pct*100:.2f}%"
                )
                self._do_sell(code, ref_price=current_price, is_st=is_st)

                # 移除本地跟踪
                if base_code in self.tracking_data["positions"]:
                    del self.tracking_data["positions"][base_code]
                self._save_tracking()
            else:
                logger.info(
                    f"  {code} | 持有 {holding_days}交易日 | 浮盈 {unrealized_pnl_pct*100:.2f}% | 未触发退出，继续持仓。"
                )

    def _do_sell(self, code: str, ref_price: Optional[float], is_st: bool):
        """
        内部卖出执行：挂跌停价限价委托。
        
        使用跌停价而非市价，原因：
        - 尾盘集合竞价前挂跌停价，交易所接受，保证尾盘能优先成交。
        - 对应回测中以当日 close 成交的逻辑（尾盘挂价 = 接受任何价格成交）。
        - 如果 ref_price 无效，退回 sell_all（让券商以市价处理）。
        """
        if ref_price is None or ref_price <= 0:
            logger.warning(f"  {code}: 参考价格无效，退回全仓市价卖出。")
            res = self.trader.sell_all(code)
        else:
            limit_down_price = _calc_limit_down_price(ref_price, is_st=is_st)
            logger.info(f"  执行卖出委托: {code} | 委托价(跌停): {limit_down_price:.2f} | 参考价: {ref_price:.2f}")

            # 获取可卖数量
            positions = self.trader.get_positions()
            amount = 0
            for pos in positions:
                p_code = pos.get('证券代码', pos.get('stock_code', ''))
                if code in p_code or p_code in code:
                    amount = int(pos.get('可用余额', pos.get('可卖数量', pos.get('stock_num', 0))) or 0)
                    break

            if amount > 0:
                res = self.trader.sell(code, amount=amount, price=limit_down_price)
            else:
                logger.warning(f"  {code}: 可用余额为 0，跳过卖出（可能已委托或 T+1 限制）。")
                return

        if res.get('status') == 'success' or res.get('entrust_no'):
            logger.info(f"  {code} 卖出委托成功。")
        else:
            logger.warning(f"  {code} 卖出委托失败或未确认: {res}")

    def is_in_buy_window(self) -> bool:
        now = datetime.now().strftime("%H:%M:%S")
        return BUY_WINDOW_START <= now <= BUY_WINDOW_END

    def is_in_sell_window(self) -> bool:
        now = datetime.now().strftime("%H:%M:%S")
        return SELL_WINDOW_START <= now <= SELL_WINDOW_END
