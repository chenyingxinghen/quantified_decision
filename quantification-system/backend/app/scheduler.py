import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
import asyncio
import os
import sys
from datetime import datetime, timedelta

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

async def check_yfinance_update():
    """
    试探性获取一只股票 (600000) 的最新 yfinance 数据，
    判断其日期是否大于数据库中的最新日期。
    """
    try:
        from core.data.yfinance_fetcher import YFinanceFetcher
    except ImportError:
        logger.error("Failed to import YFinanceFetcher. Pre-check skipped.")
        return True # Fallback to true to avoid blocking if just an import error

    trial_symbol = "600000"
    
    data_conn = None
    try:
        # 获取数据库记录的最新日期
        data_conn = get_db_connection()
        cursor = data_conn.cursor()
        cursor.execute("SELECT MAX(date) FROM daily_data WHERE code=?", (trial_symbol,))
        row = cursor.fetchone()
        db_last_date = row[0] if row and row[0] else None
    finally:
        if data_conn: data_conn.close()
        
    try:
        if not db_last_date:
            logger.info(f"数据库中未发现 {trial_symbol} 的记录，跳过预检。")
            return True

        logger.info(f"开始 yfinance 更新预检 ({trial_symbol}): 数据库最新日期 {db_last_date}")
        
        yf_fetcher = YFinanceFetcher()
        yf_start = db_last_date.replace("-", "")
        yf_end = (datetime.now()+timedelta(days=1)).strftime("%Y%m%d").replace('-','')
        
        # yf.get_historical_data 是同步的，放到线程池执行
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, lambda: yf_fetcher.get_historical_data(trial_symbol, start_date=yf_start, end_date=yf_end))
        
        if not df.empty:
            yf_last_date = df['日期'].max()
            logger.info(f"yfinance 返回最新日期: {yf_last_date}")
            if yf_last_date > db_last_date:
                return True
            else:
                logger.info(f"yfinance 数据尚未更新 (最新 {yf_last_date} <= 库中 {db_last_date})。")
        else:
            logger.warning(f"yfinance 未返回 {trial_symbol} 的新数据。")
            
    except Exception as e:
        logger.error(f"执行 yfinance 预检时异常: {e}")
        return False # 异常时触发重试
        
    return False

async def daily_data_update_job():
    """
    工作日 19:00 执行的更新任务。
    如果 yfinance 尚未更新，则 30 分钟后重试。
    """
    logger.info("Executing scheduled daily data update...")
    
    # 预检 yfinance 是否已更新
    if not await check_yfinance_update():
        logger.info("yfinance data source not updated yet. Rescheduling update for no more.")
        # next_run = datetime.now() + timedelta(minutes=30)
        # scheduler.add_job(
        #     daily_data_update_job, 
        #     DateTrigger(run_date=next_run), 
        #     id="retry_daily_data_update", 
        #     replace_existing=True
        # )
        return

    try:
        from scripts.update_daily_data import update_all_stocks
        # 执行增量更新
        logger.info("Requirement check passed. Starting full incremental update...")
        await asyncio.get_event_loop().run_in_executor(None, lambda: update_all_stocks(prefer_source="yfinance", incremental=True))
        
        logger.info("Daily data update finished. Triggering paper trading price auto-fill...")
        await asyncio.get_event_loop().run_in_executor(None, auto_fill_paper_trading_prices)
    except Exception as e:
        logger.error(f"Error in scheduled daily update: {e}")


scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")

def start_scheduler():
    # Schedule to run from Monday to Friday at 19:00
    # 市场在周末及节假日通常不更新，这里只针对工作日设定单次触发
    trigger = CronTrigger(day_of_week="mon-fri", hour=17, minute=30, timezone="Asia/Shanghai")
    scheduler.add_job(daily_data_update_job, trigger, id="daily_data_update", replace_existing=True)
    scheduler.start()
    logger.info("APScheduler started: Daily data update scheduled for Mon-Fri at 19:00.")

def stop_scheduler():
    scheduler.shutdown()
    logger.info("APScheduler stopped.")
