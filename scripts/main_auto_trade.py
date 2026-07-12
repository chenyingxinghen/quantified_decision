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
import json
import threading
import ctypes
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.automation.trader_interface import AutoTrader
from core.automation.execution_controller import ExecutionController
from config.automation_config import (
    AUTO_MIN_CONFIDENCE, AUTO_TOP_N, MAX_POSITIONS_AUTO,
    BUY_WINDOW_START, BUY_WINDOW_END, SELL_WINDOW_START, SELL_WINDOW_END,
    DRY_RUN
)
from config.baostock_config import SYSTEM_DATA_DIR

# 信号存档目录
SIGNALS_DIR = os.path.join(SYSTEM_DATA_DIR, "automation", "signals")

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

def save_signals_to_file(signals: List[Dict]):
    """将每日信号持久化到 JSON 文件以供重启恢复，并同步更新统一信号库"""
    if not os.path.exists(SIGNALS_DIR):
        os.makedirs(SIGNALS_DIR, exist_ok=True)
    
    today_str = datetime.now().strftime("%Y%m%d")
    today_date_str = datetime.now().strftime("%Y-%m-%d")
    daily_file_path = os.path.join(SIGNALS_DIR, f"signals_{today_str}.json")
    unified_file_path = os.path.join(SIGNALS_DIR, "signals.json")
    
    try:
        # 1. 保存到日文件
        with open(daily_file_path, 'w', encoding='utf-8') as f:
            json.dump(signals, f, ensure_ascii=False, indent=4)
        logger.info(f"今日信号已存档至: {daily_file_path}")
        
        # 2. 更新统一信号库
        if os.path.exists(unified_file_path):
            try:
                with open(unified_file_path, 'r', encoding='utf-8') as f:
                    unified_data = json.load(f)
            except Exception as e:
                logger.warning(f"读取统一信号库失败，创建新的: {e}")
                unified_data = {"signals": [], "by_stock_code": {}}
        else:
            unified_data = {"signals": [], "by_stock_code": {}}
        
        # 添加新信号
        for signal in signals:
            stock_code = signal["stock_code"]
            # 添加到signals数组
            unified_signal = {
                "stock_code": stock_code,
                "signal_date": today_date_str,
                "confidence": signal["confidence"],
                "current_price": signal["current_price"],
                "stop_loss": signal["stop_loss"],
                "take_profit": signal["take_profit"]
            }
            unified_data["signals"].append(unified_signal)
            
            # 更新by_stock_code索引
            if stock_code not in unified_data["by_stock_code"]:
                unified_data["by_stock_code"][stock_code] = {
                    "latest_signal_date": today_date_str,
                    "signals": []
                }
            else:
                # 更新最新信号日期
                if today_date_str > unified_data["by_stock_code"][stock_code]["latest_signal_date"]:
                    unified_data["by_stock_code"][stock_code]["latest_signal_date"] = today_date_str
            
            # 添加信号记录
            unified_data["by_stock_code"][stock_code]["signals"].append({
                "signal_date": today_date_str,
                "confidence": signal["confidence"],
                "current_price": signal["current_price"],
                "stop_loss": signal["stop_loss"],
                "take_profit": signal["take_profit"]
            })
        
        # 保存更新后的统一信号库
        with open(unified_file_path, 'w', encoding='utf-8') as f:
            json.dump(unified_data, f, ensure_ascii=False, indent=4)
        logger.info(f"统一信号库已更新: {unified_file_path}")
        
    except Exception as e:
        logger.error(f"存档信号失败: {e}")

def load_signals_from_file() -> Optional[List[Dict]]:
    """从本地存档读取今日信号 (当日系统重启恢复用)"""
    today_str = datetime.now().strftime("%Y%m%d")
    file_path = os.path.join(SIGNALS_DIR, f"signals_{today_str}.json")
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                signals = json.load(f)
            logger.info(f"成功从本地存档加载今日信号 ({today_str})，共 {len(signals)} 条。")
            return signals
        except Exception as e:
            logger.error(f"读取存档信号失败: {e}")
    return None

def get_latest_signals() -> List[Dict]:
    """
    获取今日信号。直接复用 MLFactorBacktestStrategy，保证与回测逻辑完全一致。
    """
    from config.automation_config import AUTO_MODEL_PATH, AUTO_TOP_N
    from core.backtest.strategies.ml_factor_strategy import MLFactorBacktestStrategy
    from config.factor_config import TrainingConfig
    from config.baostock_config import DATABASE_PATH

    logger.info("正在获取今日信号")
    logger.info(f"  配置: top_n={AUTO_TOP_N}, min_confidence={AUTO_MIN_CONFIDENCE}")

    # 先刷新因子缓存，确保 select_for_live 使用最新数据
    try:
        from scripts.select_stocks import _update_factor_cache_incremental, get_all_stock_codes
        logger.info("正在增量更新因子缓存...")
        all_codes = get_all_stock_codes(DATABASE_PATH)
        _update_factor_cache_incremental(
            db_path=DATABASE_PATH,
            codes=all_codes,
            cache_dir=TrainingConfig.CACHE_DIR,
        )
        logger.info(f"因子缓存更新完成，覆盖 {len(all_codes)} 只股票。")
    except Exception as e:
        logger.warning(f"因子缓存更新失败，将使用旧缓存继续: {e}", exc_info=True)

    try:
        strategy = MLFactorBacktestStrategy(
            model_path=AUTO_MODEL_PATH,
            min_confidence=AUTO_MIN_CONFIDENCE,
            cache_dir=TrainingConfig.CACHE_DIR,
        )
        strategy.initialize()

        from config import automation_config
        import config.strategy_config as sc
        
        # 设置实盘专用筛选条件（自动化优先、sc 兜底）
        criteria = {
            'min_market_cap': getattr(automation_config, 'AUTO_MIN_MARKET_CAP', None) 
                             if getattr(automation_config, 'AUTO_MIN_MARKET_CAP', None) is not None 
                             else sc.MIN_MARKET_CAP,
            'max_pe': getattr(automation_config, 'AUTO_MAX_PE', None) 
                     if getattr(automation_config, 'AUTO_MAX_PE', None) is not None 
                     else sc.MAX_PE,
            'max_zcfzl': getattr(automation_config, 'AUTO_MAX_ZCFZL', None) 
                        if getattr(automation_config, 'AUTO_MAX_ZCFZL', None) is not None 
                        else sc.MAX_ZCFZL,
            'min_price': getattr(automation_config, 'AUTO_MIN_PRICE', None) 
                        if getattr(automation_config, 'AUTO_MIN_PRICE', None) is not None 
                        else sc.MIN_PRICE,
            'max_price': getattr(automation_config, 'AUTO_MAX_PRICE', None) 
                        if getattr(automation_config, 'AUTO_MAX_PRICE', None) is not None 
                        else sc.MAX_PRICE,
            'include_st': getattr(automation_config, 'AUTO_INCLUDE_ST', None) 
                         if getattr(automation_config, 'AUTO_INCLUDE_ST', None) is not None 
                         else sc.INCLUDE_ST,
            'markets': getattr(automation_config, 'SELECTOR_MARKETS', None) 
                      if getattr(automation_config, 'SELECTOR_MARKETS', None) is not None 
                      else sc.SELECTOR_MARKETS,
            'apply_filter': getattr(automation_config, 'AUTO_APPLY_FILTER', sc.ENABLE_FUNDAMENTAL_FILTER)
        }
        
        results = strategy.select_for_live(
            db_path=DATABASE_PATH,
            top_n=AUTO_TOP_N,
            criteria=criteria
        )
        strategy.cleanup()

        logger.info(f"信号获取完成，共 {len(results)} 只。")
        if results:
            save_signals_to_file(results)
        return results
    except Exception as e:
        logger.error(f"获取信号失败: {e}", exc_info=True)
        return []

def keep_display_awake():
    """
    防止系统锁屏/息屏 + 保持 SendInput 前台权限的后台线程。

    双重保活策略：
    1. SetThreadExecutionState：阻止显示器关闭和系统休眠
    2. 模拟鼠标原地微移：让 Windows 认为持续有用户输入，
       防止 ForegroundLockTimeout 策略回收 SendInput 前台权限，
       避免 pywinauto 报 'no active desktop' 错误。
    """
    ES_CONTINUOUS        = 0x80000000
    ES_SYSTEM_REQUIRED   = 0x00000001
    ES_DISPLAY_REQUIRED  = 0x00000002

    result = ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )
    if result:
        logger.info("已阻止系统锁屏和息屏 (SetThreadExecutionState)。")
    else:
        logger.warning("SetThreadExecutionState 调用失败，请手动设置电源计划防止锁屏。")

    # INPUT 结构体，用于 SendInput 模拟鼠标事件
    # MOUSEINPUT: dx=0, dy=0, mouseData=0, dwFlags=MOUSEEVENTF_MOVE(0x0001), time=0, dwExtraInfo=0
    MOUSEEVENTF_MOVE = 0x0001

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class INPUT(ctypes.Structure):
        class _INPUT(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT)]
        _anonymous_ = ("_input",)
        _fields_ = [("type", ctypes.c_ulong), ("_input", _INPUT)]

    def nudge_mouse():
        """原地微移鼠标 +1 再 -1，视觉上不可见但刷新系统活跃计时器"""
        inp = (INPUT * 2)()
        inp[0].type = 0  # INPUT_MOUSE
        inp[0].mi.dx = 1
        inp[0].mi.dy = 0
        inp[0].mi.dwFlags = MOUSEEVENTF_MOVE
        inp[1].type = 0
        inp[1].mi.dx = -1
        inp[1].mi.dy = 0
        inp[1].mi.dwFlags = MOUSEEVENTF_MOVE
        ctypes.windll.user32.SendInput(2, inp, ctypes.sizeof(INPUT))

    while True:
        time.sleep(60)  # 每分钟保活一次
        try:
            nudge_mouse()
        except Exception:
            pass
        # 同时续期 SetThreadExecutionState
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )


def disable_foreground_lock_timeout():
    """
    将 ForegroundLockTimeout 设为 0，关闭 Windows 前台锁定超时策略。
    该策略默认值为 200000（约 200 秒），超时后后台进程无法通过 SendInput 注入输入事件。
    设为 0 表示永不超时，彻底解决 pywinauto 长时间无操作后报 'no active desktop' 的问题。
    需要管理员权限写入 HKCU，通常不需要 UAC 提权。
    """
    try:
        import winreg
        key_path = r"Control Panel\Desktop"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "ForegroundLockTimeout", 0, winreg.REG_DWORD, 0)
        logger.info("ForegroundLockTimeout 已设为 0，前台锁定策略已关闭。")
    except Exception as e:
        logger.warning(f"无法修改 ForegroundLockTimeout: {e}，将依赖鼠标保活线程。")


def main_loop():
    """使用 APScheduler 调度的事件驱动主逻辑"""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    # 启动防锁屏线程（仅 Windows 有效）
    if sys.platform == 'win32':
        disable_foreground_lock_timeout()
        awake_thread = threading.Thread(target=keep_display_awake, daemon=True, name="KeepAwake")
        awake_thread.start()

    trader = AutoTrader()
    controller = ExecutionController(trader)
    
    # 初始化：连接并尝试获取信号
    if not trader.connect():
        logger.error("无法连接交易客户端，请确保同花顺已登录。")
        if not DRY_RUN: return

    logger.info("-" * 60)

    # 启动时恢复或获取今日信号 (当日系统重启保护)
    signals = load_signals_from_file()
    if not signals:
        # 如果还没存档且已经过了早上 9 点，说明可能错过了定时任务，手动触发一次
        now = datetime.now()
        # 注意：这里判断 9:00 之后，但 15:00 之前（交易时间内）
        if now.hour >= 9 and now.hour < 15:
            logger.info("检测到今日尚未生成信号且已过 09:00，正在手动触发同步与信号生成...")
            controller.sync_positions(cleanup=False)
            signals = get_latest_signals()
        else:
            logger.info("当前未到 09:00 或已收盘，等待定时任务自动生成。")
    
    if signals:
        controller.set_buy_signals(signals)

    scheduler = BlockingScheduler()

    def job_get_signals():
        """定时任务：收盘后或盘前获取最新信号"""
        logger.info("=== 获取今日选股信号 ===")
        sigs = get_latest_signals()
        if sigs:
            controller.set_buy_signals(sigs)

    def job_execute_buys():
        """定时任务：执行买入"""
        if datetime.now().weekday() >= 5: return
        logger.info("=== 执行买入 ===")
        controller.execute_buys()

    def job_execute_sells():
        """定时任务：执行卖出（尾盘）"""
        if datetime.now().weekday() >= 5: return
        logger.info("=== 执行尾盘卖出 ===")
        controller.execute_sells()

    def job_heartbeat():
        """心跳日志"""
        logger.info(f"心跳 | 客户端状态: {'已连接' if trader.is_connected else '未连接'} | {datetime.now().strftime('%H:%M:%S')}")

    def job_update_entry_prices():
        """买入窗口结束后，用实盘成本价修正 entry_price"""
        if datetime.now().weekday() >= 5: return
        logger.info("=== 修正入场价（成本价同步）===")
        controller.update_entry_prices_from_positions()

    # 配置任务调度
    # 1. 每天上午 8:00 重新获取一次今日选股信号
    scheduler.add_job(job_get_signals, CronTrigger(day_of_week='mon-fri', hour=8, minute=00))

    # 2. 从 BUY_WINDOW_START 到 BUY_WINDOW_END 期间，每隔一两分钟尝试买入
    start_buy_h, start_buy_m, _ = map(int, BUY_WINDOW_START.split(':'))
    end_buy_h, end_buy_m, _ = map(int, BUY_WINDOW_END.split(':'))
    # 为了简单起见，设定在指定的起始分钟运行，比如 9点 20-25 分每分钟执行一次
    scheduler.add_job(job_execute_buys, CronTrigger(day_of_week='mon-fri', hour=start_buy_h, minute=f"{start_buy_m}-{end_buy_m}"))

    # 2b. 买入窗口结束后 1 分钟，用实盘成本价修正 entry_price
    scheduler.add_job(job_update_entry_prices, CronTrigger(day_of_week='mon-fri', hour=end_buy_h, minute=end_buy_m + 1))

    # 3. 从 SELL_WINDOW_START 到 SELL_WINDOW_END 期间，执行尾盘卖出
    start_sell_h, start_sell_m, _ = map(int, SELL_WINDOW_START.split(':'))
    end_sell_h, end_sell_m, _ = map(int, SELL_WINDOW_END.split(':'))
    scheduler.add_job(job_execute_sells, CronTrigger(day_of_week='mon-fri', hour=start_sell_h, minute=f"{start_sell_m}-{end_sell_m}"))

    # 4. 盘中心跳日志 (每半小时一次)
    scheduler.add_job(job_heartbeat, CronTrigger(day_of_week='mon-fri', hour="9-15", minute="0,30"))

    # 5. 启动时检查：如果当前就在交易窗口内，则立即触发一次买入/卖出检查
    now_str = datetime.now().strftime("%H:%M:%S")
    if BUY_WINDOW_START <= now_str <= BUY_WINDOW_END:
        logger.info(f"启动时正处于买入窗口 ({now_str})，立即触发买入任务...")
        job_execute_buys()
    elif SELL_WINDOW_START <= now_str <= SELL_WINDOW_END:
        logger.info(f"启动时正处于卖出窗口 ({now_str})，立即触发卖出任务...")
        job_execute_sells()

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
