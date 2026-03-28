import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
import asyncio
import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.deps import get_db_connection, get_paper_db

logger = logging.getLogger("quant_scheduler")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

def auto_fill_paper_trading_prices():
    """
    Find active paper trading positions with NULL buy_price.
    Try to fetch the opening price for their buy_date from daily_data.
    If found, update the positions with the retrieved price.
    """
    logger.info("Starting auto-fill paper trading prices...")
    paper_conn = None
    data_conn = None
    try:
        paper_conn = get_paper_db()
        data_conn = get_db_connection()
        
        # Get active positions with missing buy_price
        cursor = paper_conn.cursor()
        cursor.execute("SELECT id, code, buy_date FROM positions WHERE status='active' AND buy_price IS NULL")
        pending_positions = cursor.fetchall()
        
        filled_count = 0
        for pos in pending_positions:
            pos_id, code, buy_date = pos
            
            # Find open price from daily_data
            data_cursor = data_conn.cursor()
            data_cursor.execute("SELECT open FROM daily_data WHERE code=? AND date=?", (code, buy_date))
            row = data_cursor.fetchone()
            
            if row and row["open"]:
                open_price = row["open"]
                cursor.execute("UPDATE positions SET buy_price=? WHERE id=?", (open_price, pos_id))
                filled_count += 1
                logger.info(f"Filled buy_price {open_price} for {code} on {buy_date}")
        
        paper_conn.commit()
        logger.info(f"Auto-filled {filled_count} prices successfully.")
    except Exception as e:
        logger.error(f"Error in auto_fill_paper_trading_prices: {e}")
    finally:
        if paper_conn: paper_conn.close()
        if data_conn: data_conn.close()

async def daily_data_update_job():
    """
    工作日定时执行的更新任务。
    """
    logger.info("Executing scheduled daily data update...")

    try:
        from scripts.update_daily_data import update_all_stocks
        # 执行增量更新
        logger.info("Starting full incremental update...")
        await asyncio.get_event_loop().run_in_executor(None, lambda: update_all_stocks(incremental=True))
        
        logger.info("Daily data update finished. Triggering paper trading price auto-fill...")
        await asyncio.get_event_loop().run_in_executor(None, auto_fill_paper_trading_prices)
    except Exception as e:
        logger.error(f"Error in scheduled daily update: {e}")


scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")

def start_scheduler():
    # Schedule to run from Monday to Friday at 19:00
    # 市场在周末及节假日通常不更新，这里只针对工作日设定单次触发
    trigger = CronTrigger(day_of_week="mon-fri", hour=18, minute=30, timezone="Asia/Shanghai")
    scheduler.add_job(daily_data_update_job, trigger, id="daily_data_update", replace_existing=True)
    scheduler.start()
    logger.info("APScheduler started: Daily data update scheduled for Mon-Fri at 18:30.")

def stop_scheduler():
    scheduler.shutdown()
    logger.info("APScheduler stopped.")
