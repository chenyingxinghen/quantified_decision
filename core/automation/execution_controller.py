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
from config.config import PROJECT_ROOT, DATABASE_PATH, SYSTEM_DATA_DIR

from core.automation.trader_interface import AutoTrader
from config.automation_config import (
    SINGLE_BUY_RATIO, CASH_BUFFER,
    BUY_WINDOW_START, BUY_WINDOW_END, SELL_WINDOW_START, SELL_WINDOW_END,
    AUTO_TIME_STOP_DAYS, AUTO_TIME_STOP_MIN_LOSS_PCT,
)

from enum import Enum

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT,'database','system_data','automation','logs', "controller.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ExecutionController")


class OperationStatus(Enum):
    SUCCESS = "SUCCESS"          # 明确委托成功
    FAILED = "FAILED"            # 明确执行失败
    RETRY = "RETRY"              # 可重试的失败（如OCR错误、超时）
    SKIPPED = "SKIPPED"          # 无需执行（如已持仓、资金不足）
    UNKNOWN = "UNKNOWN"          # 状态不明（如连接成功但未见明确回执）


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
        self.tracking_file = os.path.join(SYSTEM_DATA_DIR, "automation", "tracking.json")
        self.tracking_data = self._load_tracking()

        # 数据库路径（用于交易日计算）
        self._db_path = DATABASE_PATH

    def _load_tracking(self):
        """加载本地持仓追踪记录"""
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
        default_data = {"current_day": "", "pending_buys": [], "processed_today": {}, "positions": {}}
        
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 补充缺失字段
                    for k, v in default_data.items():
                        if k not in data:
                            data[k] = v
                    return data
            except Exception as e:
                logger.error(f"加载 tracking 文件失败: {e}")
        return default_data

    def _save_tracking(self):
        """保存本地记录"""
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
        try:
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.tracking_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"保存 tracking 文件出错: {e}")

    def set_buy_signals(self, signals: List[Dict]):
        """设置今日待执行的买入信号"""
        today = datetime.now().strftime("%Y-%m-%d")
        self.signals_cache = signals

        # 如果是新的一天，重置 processed_today
        if self.tracking_data.get("current_day") != today:
            logger.info(f"新交易日检测: {today}，重置处理记录。")
            self.tracking_data["current_day"] = today
            self.tracking_data["processed_today"] = {}

        self.tracking_data["pending_buys"] = [s['stock_code'] for s in signals]
        self._save_tracking()
        logger.info(f"已加载今日买入信号: {self.tracking_data['pending_buys']}")

    def sync_positions(self):
        """同步本地追踪与实盘持仓，防止状态不一致"""
        logger.info("正在同步实盘持仓状态...")
        real_positions = self.trader.get_positions()
        if real_positions is None:
            logger.warning("同步持仓失败: 无法获取实盘数据。")
            return

        # 交叉验证：如果 real_positions 为空，但本地记录非空，我们需要二次确认是否真的是空仓
        if len(real_positions) == 0 and len(self.tracking_data["positions"]) > 0:
            balance = self.trader.get_balance()
            if not balance:
                logger.warning("发现实盘持仓为空，但无法同步获取资金读数(可能 GUI 交互异常/被验证码阻挡)。为防止误删本地记录，跳过持仓同步。")
                return
                
            # 另外，如果总资产中明确有股票市值且 > 0，但 positions 又是空的，那肯定也是表格读取失败
            market_value = float(balance.get('参考市值', balance.get('股票市值', balance.get('市值', 0))) or 0)
            if market_value > 0:
                logger.warning(f"发现实盘持仓为空，但资金表显示有股票市值 ({market_value})。持仓读取可能被干扰(如验证码弹窗)，跳过同步。")
                return

        real_codes = []
        for p in real_positions:
            code = p.get('证券代码', p.get('stock_code', ''))
            base_code = code[:6] if len(code) >= 6 else code
            real_codes.append(base_code)

        # 1. 检查本地记录的持仓是否在实盘中消失（可能被手动卖出或未被记录的卖出）
        tracking_codes = list(self.tracking_data["positions"].keys())
        for code in tracking_codes:
            if code not in real_codes:
                logger.info(f"  同步记录: {code} 在实盘中已无持仓，移除本地跟踪。")
                del self.tracking_data["positions"][code]

        # 2. 检查实盘有而本地无记录的（可能是外部操作或历史遗留），这种通常由兜底卖出处理
        for code in real_codes:
            if code not in self.tracking_data["positions"]:
                logger.debug(f"  同步检测: {code} 实盘有持仓但本地无元数据。")

        self._save_tracking()
        logger.info("持仓同步完成。")

    def _execute_with_retry(self, action_func, max_retries=3, retry_delay=2) -> Dict:
        """通用的重试执行逻辑，优雅处理 GUI 自动化的不确定性"""
        last_res = {"status": "error", "msg": "execution_not_started"}
        
        for i in range(max_retries):
            try:
                res = action_func()
                # 判定成功：entrust_no 存在，或者明确 status == success
                if res.get('entrust_no') or res.get('status') == 'success':
                    return {"op_status": OperationStatus.SUCCESS, "raw": res}
                
                # 判定重试：含有验证码错误、界面未响应等关键字
                msg = str(res.get('message', res.get('msg', ''))).lower()
                retry_keywords = ["验证码", "超时", "未响应", "识别", "captcha", "timeout", "failed to refresh"]
                if any(k in msg for k in retry_keywords):
                    logger.warning(f"  执行疑似触发界面故障 ({msg})，准备第 {i+2} 次尝试...")
                    time.sleep(retry_delay * (i + 1))
                    continue
                
                # 判定跳过：资金不足、已撤单、无效价格等
                skip_keywords = ["资金不足", "余额不足", "insufficent", "invalid", "交易时间"]
                if any(k in msg for k in skip_keywords):
                    return {"op_status": OperationStatus.SKIPPED, "raw": res}

                last_res = res
            except Exception as e:
                logger.error(f"  执行异常 (第 {i+1} 次尝试): {e}")
                time.sleep(retry_delay)
                last_res = {"status": "error", "msg": str(e)}

        return {"op_status": OperationStatus.FAILED, "raw": last_res}

    def execute_buys(self):
        """执行买入任务（开盘时间窗内运行）"""
        if not self.signals_cache:
            logger.info("队列中无待买入股票。")
            return

        balance = self.trader.get_balance()
        available_cash = float(balance.get('可用', balance.get('可用余额', balance.get('可用金额', 0))))
        logger.info(f"当前可用资金: {available_cash:.2f}")

        if available_cash < 1000:
            logger.warning("可用资金不足 1000 元，取消买入。")
            return

        # 1. 预清空已有的未成交单，确保资金可用 (可选，但在反复重试中很有用)
        logger.info("  正在清空现有未成交订单以准备新委托...")
        self.trader.cancel_all()
        time.sleep(1)

        # 检查当前持仓
        self.sync_positions()
        positions = self.trader.get_positions()
        if positions is None:
            logger.error("  获取持仓失败，为安全起见，取消本次买入。")
            return
        
        holding_codes = [p.get('证券代码', p.get('stock_code', ''))[:6] for p in positions]

        # 1. 预过滤信号
        targets = []
        for s in self.signals_cache:
            code = s['stock_code']
            base_code = code[:6]
            if base_code in holding_codes:
                logger.info(f"  {code} 已在持仓中，跳过。")
                continue
            
            p_status = self.tracking_data["processed_today"].get(base_code)
            if p_status == OperationStatus.SUCCESS.value:
                logger.info(f"  {code} 今日已买入成功，跳过。")
                continue
            elif p_status == OperationStatus.SKIPPED.value:
                logger.info(f"  {code} 今日购买曾被跳过判定为不可执行(如资金不足)，跳过。")
                continue
            
            targets.append(s)

        if not targets:
            logger.info("所有信号已处理或已在持仓中。")
            return

        # 2. 计算预算 (均分可用资金)
        budget_per_stock = min(
            available_cash / len(targets),
            available_cash * SINGLE_BUY_RATIO
        )
        budget_per_stock = max(0, budget_per_stock - CASH_BUFFER)

        # 3. 循环执行买入
        for signal in targets:
            code = signal['stock_code']
            base_code = code[:6]
            ref_price = signal.get('current_price', 0)
            
            if ref_price <= 0:
                logger.warning(f"  {code}: 参考价格无效 ({ref_price})，跳过。")
                continue

            # 委托价策略：挂涨停价
            is_st = signal.get('is_st', False)
            limit_up_price = _calc_limit_up_price(ref_price, is_st=is_st)
            
            # 数量估算
            volume = int((budget_per_stock / limit_up_price) / 100) * 100

            if volume < 100:
                logger.warning(f"  {code}: 预算不足一手 ({limit_up_price:.2f} * 100 > {budget_per_stock:.2f})")
                continue

            logger.info(f"  执行买入委托: {code} | 数量: {volume} | 委托价: {limit_up_price:.2f}")
            
            def do_buy():
                # 每次买入前简单刷新可用资金，防止超支
                bal = self.trader.get_balance()
                current_avail = float(bal.get('可用', bal.get('可用余额', 0)))
                if current_avail < (volume * limit_up_price):
                    return {"status": "error", "msg": "资金不足以执行下一笔下单"}
                return self.trader.buy(code, amount=volume, price=limit_up_price)

            res_report = self._execute_with_retry(do_buy)

            if res_report["op_status"] == OperationStatus.SUCCESS:
                self.tracking_data["processed_today"][base_code] = OperationStatus.SUCCESS.value
                self.tracking_data["positions"][base_code] = {
                    "entry_date": datetime.now().strftime("%Y-%m-%d"),
                    "entry_price": ref_price,
                    "stop_loss": signal.get('stop_loss'),
                    "take_profit": signal.get('take_profit'),
                    "confidence": signal.get('confidence'),
                    "is_st": is_st,
                }
                logger.info(f"  ✅ {code} 买入成功: {res_report['raw']}")
            elif res_report["op_status"] == OperationStatus.SKIPPED:
                reason = res_report["raw"].get('message', res_report["raw"].get('msg', 'Unknown'))
                logger.warning(f"  ⏭️ {code} 买入被主动跳过 ({res_report['op_status'].value}): {reason}")
                self.tracking_data["processed_today"][base_code] = OperationStatus.SKIPPED.value
            else:
                reason = res_report["raw"].get('message', res_report["raw"].get('msg', 'Unknown'))
                logger.error(f"  ❌ {code} 买入最终失败 ({res_report['op_status'].value}): {reason}")
                self.tracking_data["processed_today"][base_code] = OperationStatus.FAILED.value
            
            self._save_tracking()
            time.sleep(1) # 下单间隔

    def execute_sells(self):
        """执行卖出任务（尾盘时间窗内运行）"""
        from config.strategy_config import (
            ENABLE_STOP_LOSS_EXIT, ENABLE_TAKE_PROFIT_EXIT, ENABLE_TIME_STOP_EXIT
        )

        self.sync_positions()
        positions = self.trader.get_positions()
        if positions is None:
            logger.error("  ❌ 未获取到持仓信息，为防止意外清仓，不执行。")
            return
            
        if not positions:
            logger.info("  当前确认为无持仓。")
            return

        today_str = datetime.now().strftime("%Y-%m-%d")

        for p in positions:
            code = p.get('证券代码', p.get('stock_code', ''))
            base_code = code[:6]

            # T+1 保护
            if self.tracking_data["processed_today"].get(base_code) == OperationStatus.SUCCESS.value:
                logger.info(f"  {code} 今日买入 (T+1 保护)，跳过。")
                continue

            # 获取元数据
            meta = self.tracking_data["positions"].get(base_code, {})
            current_price = float(p.get('当前价', p.get('市价', p.get('现价', 0))) or 0)
            is_st = bool(meta.get('is_st', False))

            if not meta:
                logger.warning(f"  {code} 无跟踪元数据，执行兜底卖出。")
                self._do_sell_robust(code, ref_price=current_price, is_st=is_st)
                continue

            entry_price = float(meta.get('entry_price') or 0)
            entry_date_str = meta.get('entry_date', '')

            if current_price <= 0 or entry_price <= 0:
                logger.warning(f"  {code}: 价格无效 (current={current_price}, entry={entry_price})")
                continue

            # 条件检查
            holding_days = _get_trading_days_count(entry_date_str, today_str, self._db_path)
            unrealized_pnl_pct = (current_price - entry_price) / entry_price
            
            should_exit = False
            reason = ""

            sl = meta.get('stop_loss')
            if ENABLE_STOP_LOSS_EXIT and sl and current_price <= float(sl):
                should_exit = True
                reason = "stop_loss"

            tp = meta.get('take_profit')
            if ENABLE_TAKE_PROFIT_EXIT and tp and current_price >= float(tp):
                should_exit = True
                reason = "take_profit"

            if (ENABLE_TIME_STOP_EXIT
                    and holding_days >= AUTO_TIME_STOP_DAYS
                    and unrealized_pnl_pct <= AUTO_TIME_STOP_MIN_LOSS_PCT):
                should_exit = True
                reason = "time_stop"

            if should_exit:
                logger.info(f"  卖出触发: {code} | 原因: {reason} | 收益: {unrealized_pnl_pct*100:.2f}%")
                success = self._do_sell_robust(code, ref_price=current_price, is_st=is_st)
                if success:
                    if base_code in self.tracking_data["positions"]:
                        del self.tracking_data["positions"][base_code]
                    self._save_tracking()
            else:
                logger.info(f"  {code} | 持有 {holding_days}D | 浮盈 {unrealized_pnl_pct*100:.2f}% | 继续持有。")

    def _do_sell_robust(self, code: str, ref_price: Optional[float], is_st: bool) -> bool:
        """健壮的卖出执行逻辑"""
        base_code = code[:6]
        
        def attempt_sell():
            # 重新获取最新持仓以确认数量
            pos_list = self.trader.get_positions()
            amount = 0
            for pos in pos_list:
                p_code = pos.get('证券代码', pos.get('stock_code', ''))
                if base_code in p_code or p_code in base_code:
                    amount = int(pos.get('可用余额', pos.get('可卖数量', 0)) or 0)
                    break
            
            if amount <= 0:
                return {"status": "skipped", "msg": "可用余额为0（可能已下单）"}
            
            if ref_price is None or ref_price <= 0:
                return self.trader.sell_all(code)
            
            limit_down_price = _calc_limit_down_price(ref_price, is_st=is_st)
            return self.trader.sell(code, amount=amount, price=limit_down_price)

        res_report = self._execute_with_retry(attempt_sell)
        
        if res_report["op_status"] == OperationStatus.SUCCESS:
            logger.info(f"  ✅ {code} 卖出成功: {res_report['raw']}")
            return True
        else:
            logger.error(f"  ❌ {code} 卖出失败: {res_report['raw']}")
            return False


    def is_in_buy_window(self) -> bool:
        now = datetime.now().strftime("%H:%M:%S")
        return BUY_WINDOW_START <= now <= BUY_WINDOW_END

    def is_in_sell_window(self) -> bool:
        now = datetime.now().strftime("%H:%M:%S")
        return SELL_WINDOW_START <= now <= SELL_WINDOW_END
