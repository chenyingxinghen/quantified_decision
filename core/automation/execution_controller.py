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
)
from config.strategy_config import TIME_STOP_DAYS, TIME_STOP_MIN_LOSS_PCT

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


import re as _re


def _calc_limit_up_price(ref_price: float, is_st: bool = False, prev_close: float = None) -> float:
    """
    计算涨停价（用于买入委托，确保排队靠前）。
    规则：普通股 +10%，ST股 +5%，向下取整到分（0.01精度）。
    若提供 prev_close，则基于昨收价计算（更准确）；否则基于 ref_price 估算。
    """
    base = prev_close if prev_close and prev_close > 0 else ref_price
    rate = 0.05 if is_st else 0.10
    raw = base * (1 + rate)
    return round(int(raw * 100) / 100, 2)  # 向下取整到分


def _calc_limit_down_price(ref_price: float, is_st: bool = False, prev_close: float = None) -> float:
    """
    计算跌停价（用于卖出委托，确保尾盘成交）。
    规则：普通股 -10%，ST股 -5%，向上取整到分（0.01精度）。
    若提供 prev_close，则基于昨收价计算（更准确）；否则基于 ref_price 估算。
    注意：结果不得低于交易所实际跌停价，否则委托会被拒绝。
    """
    base = prev_close if prev_close and prev_close > 0 else ref_price
    rate = 0.05 if is_st else 0.10
    raw = base * (1 - rate)
    return round((int(raw * 100) + 1) / 100, 2)  # 向上取整到分，确保不低于跌停价


def _parse_price_limit_from_error(error_msg: str):
    """
    从"超过涨跌限制"错误信息中解析允许的价格范围。
    示例: '超过涨跌限制。9.21-7.53。' -> (7.53, 9.21)
    返回 (min_price, max_price) 或 (None, None)
    """
    match = _re.search(r'(\d+\.?\d*)-(\d+\.?\d*)', error_msg)
    if match:
        try:
            a, b = float(match.group(1)), float(match.group(2))
            return (min(a, b), max(a, b))
        except ValueError:
            pass
    return None, None


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
                logger.error(f"加载 tracking 文件失败: {e}，备份损坏文件并重置。")
                # 备份损坏文件，避免静默丢失数据
                backup_path = self.tracking_file + ".bak"
                try:
                    import shutil
                    shutil.copy2(self.tracking_file, backup_path)
                    logger.warning(f"已备份损坏文件至: {backup_path}")
                except Exception as be:
                    logger.error(f"备份失败: {be}")
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

    def restore_signals_from_tracking(self, full_signals: List[Dict]) -> List[Dict]:
        """
        程序重启后，用 pending_buys 过滤 full_signals，恢复未完成的买入任务到 signals_cache。
        full_signals: 外部传入的完整信号列表（含价格等信息）。
        返回恢复的信号列表（同时更新 signals_cache）。
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if self.tracking_data.get("current_day") != today:
            logger.info("pending_buys 属于昨日，不恢复。")
            return []

        pending = set(self.tracking_data.get("pending_buys", []))
        if not pending:
            return []

        restored = [s for s in full_signals if s.get('stock_code', '')[:6] in pending]
        if restored:
            self.signals_cache = restored
            logger.info(f"从 pending_buys 恢复 {len(restored)} 个买入信号: {[s['stock_code'] for s in restored]}")
        return restored

    def sync_positions(self):
        """同步本地追踪与实盘持仓。
        返回：None=获取失败，[]=确认空仓，[...]=持仓列表
        """
        logger.info("同步持仓...")
        real_positions = self.trader.get_positions()
        if real_positions is None:
            if not self.trader.is_connected:
                logger.error("持仓同步失败：桌面不可交互，等待下一调度周期自动恢复。")
            else:
                logger.warning("持仓同步失败：GUI 数据获取异常。")
            return None

        # 空仓时交叉验证，防止 GUI 被干扰导致误判
        if len(real_positions) == 0 and len(self.tracking_data["positions"]) > 0:
            balance = self.trader.get_balance()
            if not balance:
                logger.warning("实盘持仓为空，但资金读取失败（可能 GUI 异常），跳过同步。")
                return None
            market_value = float(balance.get('参考市值', balance.get('股票市值', balance.get('市值', 0))) or 0)
            if market_value > 0:
                logger.warning(f"实盘持仓为空，但资金表显示股票市值={market_value}，疑似 GUI 干扰，跳过同步。")
                return None

        real_codes = []
        for p in real_positions:
            code = p.get('证券代码', p.get('stock_code', ''))
            base_code = code[:6] if len(code) >= 6 else code
            real_codes.append(base_code)

        # 移除实盘已清仓的本地记录
        today_str = datetime.now().strftime("%Y-%m-%d")
        for code in list(self.tracking_data["positions"].keys()):
            if code not in real_codes:
                pos_meta = self.tracking_data["positions"].get(code, {})
                entry_date = pos_meta.get("entry_date", "")
                if entry_date == today_str:
                    # 今日刚买入，委托可能尚未成交或持仓 GUI 未刷新，保留元数据
                    logger.debug(f"  同步: {code} 实盘暂无持仓，但今日刚买入，保留本地元数据。")
                    continue
                logger.info(f"  同步: {code} 实盘已无持仓，移除本地记录。")
                del self.tracking_data["positions"][code]

        for code in real_codes:
            if code not in self.tracking_data["positions"]:
                logger.debug(f"  同步: {code} 实盘有持仓但无本地元数据。")

        self._save_tracking()
        logger.info(f"持仓同步完成，实盘持仓 {len(real_positions)} 只: {[p.get('证券代码', p.get('stock_code',''))[:6] for p in real_positions]}")
        return real_positions

    def _execute_with_retry(self, action_func, max_retries=3, retry_delay=2,
                            price_adjust_callback=None) -> Dict:
        """通用重试执行，处理 GUI 自动化的不确定性。
        price_adjust_callback: (min_price, max_price) -> bool，遇到涨跌限制时调用。
        """
        last_res = {"status": "error", "msg": "execution_not_started"}
        
        for i in range(max_retries):
            try:
                res = action_func()
                if res.get('entrust_no') or res.get('status') == 'success':
                    return {"op_status": OperationStatus.SUCCESS, "raw": res}
                
                msg = str(res.get('message', res.get('msg', '')))
                msg_lower = msg.lower()

                if "no active desktop" in msg_lower or "moving mouse cursor" in msg_lower:
                    logger.error("桌面不可交互，停止重试。")
                    return {"op_status": OperationStatus.FAILED, "raw": res}

                if "超过涨跌限制" in msg or "涨跌限制" in msg:
                    min_p, max_p = _parse_price_limit_from_error(msg)
                    logger.warning(f"委托价超出涨跌限制，允许范围: {min_p}-{max_p}")
                    if price_adjust_callback and min_p is not None:
                        if price_adjust_callback(min_p, max_p):
                            time.sleep(retry_delay)
                            continue
                    return {"op_status": OperationStatus.FAILED, "raw": res,
                            "price_limit": (min_p, max_p)}

                retry_keywords = ["验证码", "超时", "未响应", "识别", "captcha", "timeout", "failed to refresh"]
                if any(k in msg_lower for k in retry_keywords):
                    logger.warning(f"GUI 故障 ({msg})，第 {i+2}/{max_retries} 次重试...")
                    time.sleep(retry_delay * (i + 1))
                    continue
                
                skip_keywords = ["资金不足", "余额不足", "insufficent", "invalid", "交易时间", "可用余额"]
                if any(k in msg_lower for k in skip_keywords) or res.get("status") == "skipped":
                    return {"op_status": OperationStatus.SKIPPED, "raw": res}

                last_res = res
            except Exception as e:
                err_msg = str(e).lower()
                if "no active desktop" in err_msg or "moving mouse cursor" in err_msg:
                    logger.error(f"桌面不可交互，停止重试: {e}")
                    return {"op_status": OperationStatus.FAILED, "raw": {"status": "error", "msg": str(e)}}
                logger.error(f"执行异常 (第 {i+1}/{max_retries} 次): {e}")
                time.sleep(retry_delay)
                last_res = {"status": "error", "msg": str(e)}

        return {"op_status": OperationStatus.FAILED, "raw": last_res}

    def execute_buys(self):
        """执行买入任务（开盘时间窗内运行）"""
        if not self.signals_cache:
            logger.info("无待买入信号，跳过。")
            return

        balance = self.trader.get_balance()
        if not balance:
            if not self.trader.is_connected:
                logger.warning("桌面不可用，等待恢复后继续买入...")
                if not self.trader.wait_for_desktop(timeout_seconds=300):
                    logger.error("桌面恢复超时，取消本次买入。")
                    return
                balance = self.trader.get_balance()
            if not balance:
                logger.warning("资金数据获取失败（GUI 异常或验证码），取消买入。")
                return

        available_cash = float(balance.get('可用', balance.get('可用余额', balance.get('可用金额', 0))))
        logger.info(f"当前可用资金: {available_cash:.2f}")

        if available_cash < 200:
            logger.warning(f"可用资金 ({available_cash:.2f}) 不足 200 元，取消买入。")
            return

        # 检查当前持仓
        positions = self.sync_positions()
        if positions is None:
            logger.error("  获取持仓失败，为安全起见，取消本次买入。")
            return
        
        holding_codes = [p.get('证券代码', p.get('stock_code', ''))[:6] for p in positions]

        # 2. 计算预算 (均分可用资金)
        budget_per_stock = available_cash * SINGLE_BUY_RATIO
        budget_per_stock = max(0, budget_per_stock - CASH_BUFFER)
        
        # 跟踪当前可用资金（本地跟踪，减少对不稳定性 GUI 的依赖）
        running_avail = available_cash

        # 1. 预过滤信号
        targets = []
        for s in self.signals_cache:
            code = s['stock_code']
            base_code = code[:6]
            if base_code in holding_codes:
                logger.info(f"  {code} 已在持仓，跳过。")
                continue
            
            p_status = self.tracking_data["processed_today"].get(base_code)
            if p_status == OperationStatus.SUCCESS.value:
                logger.info(f"  {code} 今日已买入，跳过。")
                continue
            elif p_status == OperationStatus.SKIPPED.value:
                if running_avail > (budget_per_stock * 0.8):
                    logger.info(f"  {code} 之前跳过但当前资金充足，重新尝试。")
                else:
                    logger.info(f"  {code} 今日已跳过（资金不足），跳过。")
                    continue
            
            targets.append(s)

        if not targets:
            logger.info("所有信号已处理完毕。")
            return

        # 3. 循环执行买入
        for signal in targets:
            code = signal['stock_code']
            base_code = code[:6]
            ref_price = signal.get('current_price', 0)
            
            if ref_price <= 0:
                logger.warning(f"  {code}: 参考价格无效 ({ref_price})，跳过。")
                continue

            is_st = signal.get('is_st', False)
            limit_up_price = _calc_limit_up_price(ref_price, is_st=is_st)
            volume = int((budget_per_stock / limit_up_price) / 100) * 100

            if volume < 100:
                logger.warning(f"  {code}: 预算不足一手 ({limit_up_price:.2f} x 100 > {budget_per_stock:.2f})，跳过。")
                continue

            logger.info(f"  买入委托: {code} x{volume} @ {limit_up_price:.2f}")
            
            buy_price_box = [limit_up_price]

            def do_buy():
                nonlocal running_avail
                required = volume * buy_price_box[0]
                
                bal = self.trader.get_balance()
                if bal:
                    current_avail_gui = float(bal.get('可用', bal.get('可用余额', bal.get('可用金额', 0))))
                    if current_avail_gui == 0 and running_avail > 200:
                        logger.warning(f"  {code}: GUI 资金读数为 0，使用本地缓存 {running_avail:.2f}。")
                    else:
                        running_avail = current_avail_gui
                
                if running_avail < required:
                    logger.warning(f"  {code}: 资金不足 (可用={running_avail:.2f}, 需要={required:.2f})")
                    return {"status": "skipped", "msg": "资金不足以执行下一笔下单"}

                return self.trader.buy(code, amount=volume, price=buy_price_box[0])

            def on_buy_price_limit_error(min_price, max_price) -> bool:
                if max_price is None:
                    return False
                adjusted = round(max_price - 0.01, 2)
                if buy_price_box[0] != adjusted:
                    logger.warning(f"  {code} 买入价 {buy_price_box[0]} 超出限制 [{min_price}, {max_price}]，调整为 {adjusted}")
                    buy_price_box[0] = adjusted
                    return True
                return False

            res_report = self._execute_with_retry(do_buy, price_adjust_callback=on_buy_price_limit_error)

            if res_report["op_status"] == OperationStatus.SUCCESS:
                self.tracking_data["processed_today"][base_code] = OperationStatus.SUCCESS.value
                actual_entry = buy_price_box[0]
                raw_sl = signal.get('stop_loss')
                raw_tp = signal.get('take_profit')
                adjusted_sl = (actual_entry - (ref_price - float(raw_sl))) if ref_price > 0 and raw_sl is not None else raw_sl
                adjusted_tp = (actual_entry + (float(raw_tp) - ref_price)) if ref_price > 0 and raw_tp is not None else raw_tp
                self.tracking_data["positions"][base_code] = {
                    "entry_date": datetime.now().strftime("%Y-%m-%d"),
                    "entry_price": actual_entry,
                    "stop_loss": adjusted_sl,
                    "take_profit": adjusted_tp,
                    "confidence": signal.get('confidence'),
                    "is_st": is_st,
                }
                running_avail -= (volume * buy_price_box[0])
                logger.info(f"  [OK] {code} 买入成功，委托号: {res_report['raw'].get('entrust_no', '-')}，剩余资金估算: {running_avail:.2f}")
            elif res_report["op_status"] == OperationStatus.SKIPPED:
                reason = res_report["raw"].get('message', res_report["raw"].get('msg', ''))
                logger.warning(f"  [跳过] {code}: {reason}")
                self.tracking_data["processed_today"][base_code] = OperationStatus.SKIPPED.value
            else:
                reason = res_report["raw"].get('message', res_report["raw"].get('msg', ''))
                logger.error(f"  [失败] {code}: {reason}")
                self.tracking_data["processed_today"][base_code] = OperationStatus.FAILED.value
            
            self._save_tracking()
            time.sleep(1)

    def execute_sells(self):
        """执行卖出任务（尾盘时间窗内运行）"""
        from config.strategy_config import (
            ENABLE_STOP_LOSS_EXIT, ENABLE_TAKE_PROFIT_EXIT, ENABLE_TIME_STOP_EXIT
        )

        positions = self.sync_positions()
        if positions is None:
            if not self.trader.is_connected:
                logger.warning("桌面不可用，等待恢复后继续卖出...")
                if not self.trader.wait_for_desktop(timeout_seconds=300):
                    logger.error("桌面恢复超时，取消本次卖出。")
                    return
                positions = self.sync_positions()
            if positions is None:
                logger.error("持仓获取失败，为防止意外清仓，不执行卖出。")
                return

        if not positions:
            logger.info("当前无持仓。")
            return

        today_str = datetime.now().strftime("%Y-%m-%d")

        for p in positions:
            code = p.get('证券代码', p.get('stock_code', ''))
            if not code:
                logger.warning(f"持仓记录缺少证券代码，跳过: {p}")
                continue
            base_code = code[:6]

            # T+1 和 今日已操作保护
            if self.tracking_data["processed_today"].get(base_code) == OperationStatus.SUCCESS.value:
                logger.info(f"  {code} 今日买入 (T+1 保护)，跳过。")
                continue
            
            sell_status = self.tracking_data["processed_today"].get(f"sell_{base_code}")
            if sell_status in [OperationStatus.SUCCESS.value, OperationStatus.SKIPPED.value]:
                logger.info(f"  {code} 今日已执行卖出操作 (状态: {sell_status})，跳过防止重复下单。")
                continue

            # 获取元数据
            meta = self.tracking_data["positions"].get(base_code, {})
            current_price = float(p.get('当前价', p.get('市价', p.get('现价', 0))) or 0)
            is_st = bool(meta.get('is_st', False))

            if not meta:
                logger.warning(f"  {code} 无跟踪元数据，执行兜底卖出。")
                avail_amount = int(p.get('可用余额', p.get('可卖数量', 0)) or 0)
                success, op_status = self._do_sell_robust(code, ref_price=current_price, is_st=is_st, avail_amount=avail_amount)
                self.tracking_data["processed_today"][f"sell_{base_code}"] = op_status.value
                self._save_tracking()
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
                    and holding_days >= TIME_STOP_DAYS
                    and unrealized_pnl_pct <= TIME_STOP_MIN_LOSS_PCT):
                should_exit = True
                reason = "time_stop"

            if should_exit:
                logger.info(f"  {code} 触发卖出: {reason} | 持有 {holding_days}D | 浮盈 {unrealized_pnl_pct*100:.2f}%")
                avail_amount = int(p.get('可用余额', p.get('可卖数量', 0)) or 0)
                success, op_status = self._do_sell_robust(code, ref_price=current_price, is_st=is_st, avail_amount=avail_amount)
                if op_status in [OperationStatus.SUCCESS, OperationStatus.SKIPPED]:
                    self.tracking_data["processed_today"][f"sell_{base_code}"] = op_status.value
                    if success and base_code in self.tracking_data["positions"]:
                        del self.tracking_data["positions"][base_code]
                    self._save_tracking()
                elif op_status == OperationStatus.FAILED:
                    # 记录失败状态，当日不再重试，避免反复下单
                    self.tracking_data["processed_today"][f"sell_{base_code}"] = OperationStatus.FAILED.value
                    self._save_tracking()
            else:
                logger.info(f"  {code} 持有 {holding_days}D | 浮盈 {unrealized_pnl_pct*100:.2f}% | 继续持有。")

    def _do_sell_robust(self, code: str, ref_price: Optional[float], is_st: bool, avail_amount: int = 0):
        """健壮卖出执行。返回 (success: bool, OperationStatus)
        avail_amount 由调用方传入可避免重复 get_positions()，为 0 时内部懒加载兜底。
        """
        base_code = code[:6]
        sell_price_box = [_calc_limit_down_price(ref_price, is_st=is_st) if ref_price and ref_price > 0 else None]

        def attempt_sell():
            nonlocal avail_amount
            amount = avail_amount
            if amount <= 0:
                pos_list = self.trader.get_positions()
                if pos_list:
                    for pos in pos_list:
                        p_code = pos.get('证券代码', pos.get('stock_code', ''))
                        if p_code and (base_code in p_code or p_code in base_code):
                            amount = int(pos.get('可用余额', pos.get('可卖数量', 0)) or 0)
                            avail_amount = amount
                            break
            
            if amount <= 0:
                return {"status": "skipped", "msg": "可用余额为0（可能已下单）"}
            
            if sell_price_box[0] is None:
                return self.trader.sell_all(code)
            
            logger.info(f"  卖出委托: {code} x{amount} @ {sell_price_box[0]:.3f}")
            return self.trader.sell(code, amount=amount, price=sell_price_box[0])

        def on_price_limit_error(min_price, max_price) -> bool:
            if min_price is None:
                return False
            adjusted = round(min_price + 0.01, 2)
            if sell_price_box[0] is None or adjusted != sell_price_box[0]:
                logger.warning(f"  {code} 卖出价 {sell_price_box[0]} 超出限制 [{min_price}, {max_price}]，调整为 {adjusted}")
                sell_price_box[0] = adjusted
                return True
            return False

        res_report = self._execute_with_retry(
            attempt_sell,
            price_adjust_callback=on_price_limit_error
        )
        
        if res_report["op_status"] == OperationStatus.SUCCESS:
            logger.info(f"  [OK] {code} 卖出成功，委托号: {res_report['raw'].get('entrust_no', '-')}")
            return True, res_report["op_status"]
        elif res_report["op_status"] == OperationStatus.SKIPPED:
            logger.warning(f"  [跳过] {code} 卖出: {res_report['raw'].get('msg', '')}")
            return False, res_report["op_status"]
        else:
            logger.error(f"  [失败] {code} 卖出: {res_report['raw'].get('message', res_report['raw'].get('msg', ''))}")
            return False, res_report["op_status"]


    def update_entry_prices_from_positions(self):
        """买入窗口结束后，从实盘持仓的「成本价」字段修正 entry_price 并重算止盈止损偏移。"""
        positions = self.sync_positions()
        if not positions:
            logger.warning("update_entry_prices: 持仓获取失败或空仓，跳过修正。")
            return

        updated = []
        for p in positions:
            code = p.get('证券代码', p.get('stock_code', ''))
            if not code:
                continue
            base_code = code[:6]

            meta = self.tracking_data["positions"].get(base_code)
            if not meta:
                continue  # 非本系统买入的持仓，不处理

            raw_cost = p.get('成本价')
            if raw_cost is None:
                logger.warning(f"  {base_code}: 持仓数据中无「成本价」字段，跳过修正。")
                continue

            actual_entry = float(raw_cost)
            if actual_entry <= 0:
                logger.warning(f"  {base_code}: 成本价为 {actual_entry}，跳过修正。")
                continue

            old_entry = meta.get('entry_price', 0)
            if abs(actual_entry - old_entry) < 0.001:
                logger.info(f"  {base_code}: 成本价与委托价一致 ({actual_entry})，无需修正。")
                continue

            # 按原始偏移量重算止盈止损
            old_sl = meta.get('stop_loss')
            old_tp = meta.get('take_profit')
            new_sl = round(actual_entry + (float(old_sl) - old_entry), 3) if old_sl is not None and old_entry > 0 else old_sl
            new_tp = round(actual_entry + (float(old_tp) - old_entry), 3) if old_tp is not None and old_entry > 0 else old_tp

            meta['entry_price'] = actual_entry
            meta['stop_loss'] = new_sl
            meta['take_profit'] = new_tp
            updated.append(base_code)
            logger.info(f"  {base_code}: entry_price {old_entry} → {actual_entry} | SL {old_sl} → {new_sl} | TP {old_tp} → {new_tp}")

        if updated:
            self._save_tracking()
            logger.info(f"update_entry_prices: 已修正 {len(updated)} 只持仓: {updated}")
        else:
            logger.info("update_entry_prices: 无需修正。")

    def is_in_buy_window(self) -> bool:
        now = datetime.now().strftime("%H:%M:%S")
        return BUY_WINDOW_START <= now <= BUY_WINDOW_END

    def is_in_sell_window(self) -> bool:
        now = datetime.now().strftime("%H:%M:%S")
        return SELL_WINDOW_START <= now <= SELL_WINDOW_END
