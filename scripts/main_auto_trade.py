"""
主调度脚本 (Main Entry for Automation)

集成了多模块。通常设置在每天定时启动。
1. 前一晚或当日早盘: 计算选股信号 (select_stocks.py)
2. 开盘时间: 执行买入 (controller.execute_buys)  → 挂涨停价确保开盘成交
3. 尾盘时间: 执行卖出 (controller.execute_sells) → 挂跌停价确保收盘成交

信号执行逻辑完全对齐回测:
  - 开盘买入 = 回测中 next_day_open 成交
  - 尾盘止损/止盈/时间止损卖出 = 回测中以 close/stop_loss/take_profit 成交
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from select_stocks import select_stocks
from core.automation.trader_interface import AutoTrader
from core.automation.execution_controller import ExecutionController
from config.automation_config import (
    AUTO_MODEL_PATH, AUTO_MIN_CONFIDENCE, AUTO_TOP_N, MAX_POSITIONS_AUTO,
    BUY_WINDOW_START, BUY_WINDOW_END, SELL_WINDOW_START, SELL_WINDOW_END,
    AUTO_APPLY_FILTER, AUTO_MIN_MARKET_CAP, AUTO_MAX_PE,
    AUTO_MIN_PRICE, AUTO_MAX_PRICE, AUTO_INCLUDE_ST,
    DRY_RUN
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT,'database','system_data','automation','logs', "main.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutoTraderApp")

def get_latest_signals() -> List[Dict]:
    """
    获取今日信号并计算止损止盈。
    
    选股逻辑：完全对齐回测流程：
      - 使用 automation_config 中的独立筛选条件（市值/PE/股价/ST）
      - top_n = AUTO_TOP_N，与 MAX_POSITIONS_AUTO 一致
      - ATR 止损/止盈参数与 strategy_config 对齐
    """
    from config.strategy_config import ATR_STOP_MULTIPLIER, ATR_TARGET_MULTIPLIER
    from config.factor_config import FactorConfig
    import talib
    import numpy as np
    import sqlite3
    from config import DATABASE_PATH
    from scripts.select_stocks import get_stock_data

    logger.info("正在获取今日信号并计算止损止盈...")
    logger.info(f"  选股配置: top_n={AUTO_TOP_N}, min_confidence={AUTO_MIN_CONFIDENCE}, apply_filter={AUTO_APPLY_FILTER}")
    if AUTO_APPLY_FILTER:
        logger.info(f"  筛选条件: 市值>={AUTO_MIN_MARKET_CAP}亿, PE<={AUTO_MAX_PE}, "
                    f"股价[{AUTO_MIN_PRICE}, {AUTO_MAX_PRICE}], ST={AUTO_INCLUDE_ST}")
    try:
        # 1. 执行选股（传入 automation_config 的专属筛选参数）
        results = select_stocks(
            model_path=AUTO_MODEL_PATH,
            min_confidence=AUTO_MIN_CONFIDENCE,
            top_n=AUTO_TOP_N,
            apply_filter=AUTO_APPLY_FILTER,
            save_csv=False,
            # 自动化专属筛选条件（仅在 apply_filter=True 时生效）
            min_market_cap=AUTO_MIN_MARKET_CAP,
            max_pe=AUTO_MAX_PE,
            min_price=AUTO_MIN_PRICE,
            max_price=AUTO_MAX_PRICE,
            include_st=AUTO_INCLUDE_ST,
        )
        
        # 2. 对每只股票补充 ATR 止损止盈（对齐回测 ml_factor_strategy 中的计算逻辑）
        final_signals = []
        for r in results:
            code = r['stock_code']
            # 获取历史数据计算 ATR（与回测中信号生成时一致）
            data = get_stock_data(DATABASE_PATH, code, days=100)
            if data is not None and len(data) >= FactorConfig.ATR_PERIOD + 1:
                atr_series = talib.ATR(
                    data['high'].values.astype(float),
                    data['low'].values.astype(float),
                    data['close'].values.astype(float),
                    timeperiod=FactorConfig.ATR_PERIOD
                )
                atr = float(atr_series[-1])
                if np.isfinite(atr) and atr > 0:
                    r['stop_loss'] = r['current_price'] - ATR_STOP_MULTIPLIER * atr
                    r['take_profit'] = r['current_price'] + ATR_TARGET_MULTIPLIER * atr
                    logger.info(f"  信号: {code} | 现价: {r['current_price']:.2f} | "
                                f"止损: {r['stop_loss']:.2f} | 止盈: {r['take_profit']:.2f}")
                else:
                    logger.warning(f"  {code}: ATR 无效 ({atr})，跳过止损止盈设置")
            else:
                logger.warning(f"  {code}: 数据不足，无法计算 ATR")
            
            final_signals.append(r)
            
        logger.info(f"信号获取完成，共 {len(final_signals)} 只。")
        return final_signals
    except Exception as e:
        logger.error(f"获取信号失败: {e}", exc_info=True)
        return []

def main_loop():
    """使用 APScheduler 调度的事件驱动主逻辑"""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    trader = AutoTrader()
    controller = ExecutionController(trader)
    
    # 初始化：连接并尝试获取信号
    if not trader.connect():
        logger.error("无法建立交易客户端连接，请确保同花顺等客户端已手动登录并处于解锁状态。")
        if not DRY_RUN: return

    logger.info("=" * 60)
    logger.info("自动交易主调度开始运行 (基于 APScheduler 任务中断模式)")
    logger.info("=" * 60)

    # # 启动时先获取一下今日信号作为兜底
    # signals = get_latest_signals()
    # if signals:
    #     controller.set_buy_signals(signals)

    scheduler = BlockingScheduler()

    def job_get_signals():
        """定时任务：收盘后或盘前获取最新信号"""
        logger.info("=== 触发定时任务：同步实盘并获取最新信号 ===")
        controller.sync_positions()
        sigs = get_latest_signals()
        if sigs:
            controller.set_buy_signals(sigs)

    def job_execute_buys():
        """定时任务：执行买入"""
        if datetime.now().weekday() >= 5: return # 周末不交易
        logger.info("=== 触发定时任务：检查并执行买入 ===")
        controller.execute_buys()

    def job_execute_sells():
        """定时任务：执行卖出（尾盘清理）"""
        if datetime.now().weekday() >= 5: return
        logger.info("=== 触发定时任务：检查并执行尾盘卖出 ===")
        controller.execute_sells()

    def job_heartbeat():
        """心跳日志"""
        now_str = datetime.now().strftime("%H:%M:%S")
        logger.info(f"心跳在线，当前交易客户端状态检查正常。({now_str})")

    # 配置任务调度
    # 1. 每天上午 9:10 重新获取一次今日选股信号
    scheduler.add_job(job_get_signals, CronTrigger(day_of_week='mon-fri', hour=9, minute=00))

    # 2. 从 BUY_WINDOW_START 到 BUY_WINDOW_END 期间，每隔一两分钟尝试买入
    start_buy_h, start_buy_m, _ = map(int, BUY_WINDOW_START.split(':'))
    end_buy_h, end_buy_m, _ = map(int, BUY_WINDOW_END.split(':'))
    # 为了简单起见，设定在指定的起始分钟运行，比如 9点 20-25 分每分钟执行一次
    scheduler.add_job(job_execute_buys, CronTrigger(day_of_week='mon-fri', hour=start_buy_h, minute=f"{start_buy_m}-{end_buy_m}"))

    # 3. 从 SELL_WINDOW_START 到 SELL_WINDOW_END 期间，执行尾盘卖出
    start_sell_h, start_sell_m, _ = map(int, SELL_WINDOW_START.split(':'))
    end_sell_h, end_sell_m, _ = map(int, SELL_WINDOW_END.split(':'))
    scheduler.add_job(job_execute_sells, CronTrigger(day_of_week='mon-fri', hour=start_sell_h, minute=f"{start_sell_m}-{end_sell_m}"))

    # 4. 盘中心跳日志 (每半小时一次)
    scheduler.add_job(job_heartbeat, CronTrigger(day_of_week='mon-fri', hour="9-15", minute="0,30"))

    logger.info("任务调度配置完毕。调度器已启动。")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("用户手动停止调度器。")

if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        logger.error(f"程序异常退出: {e}", exc_info=True)
