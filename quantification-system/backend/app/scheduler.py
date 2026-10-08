import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
import asyncio
import os
import sys
import subprocess
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.deps import get_db_connection, get_paper_db, get_project_root

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

def verify_daily_update():
    """
    更新后校验：统计实际覆盖到目标交易日的股票数，并核对 sync_status 是否撒谎、
    今日 API 配额用量。把“假成功”变成可见的 WARNING 日志，便于第一时间发现数据缺口。
    """
    try:
        import sqlite3
        from config.baostock_config import DATABASE_PATH, META_DB_PATH, COVERAGE_ACTIVE_WINDOW_DAYS
        from datetime import datetime, timedelta

        now = datetime.now()
        target = now.strftime('%Y-%m-%d')
        if now.hour < 17 or (now.hour == 17 and now.minute < 30):
            target = (now - timedelta(days=1)).strftime('%Y-%m-%d')

        # “活跃窗口”起点：仅把仍应保持最新（近 window 天内有过 bar）的股票计入覆盖口径。
        # 长期停牌/退市股 MAX(date) 早已远离 target，混进分母会让健康日也永远报缺失（假报警）。
        window_start = (datetime.strptime(target, '%Y-%m-%d')
                        - timedelta(days=COVERAGE_ACTIVE_WINDOW_DAYS)).strftime('%Y-%m-%d')

        dconn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
        try:
            # daily_data 在主库，sync_status/api_quota 在 meta 库；ATTACH 后统一查询
            dconn.execute(f"ATTACH DATABASE '{META_DB_PATH}' AS meta")
            # 每只股票只算一次 MAX(date)；md 为 ISO 日期串，字典序==时间序。
            # live = 近 window 天内有 bar（应保持最新）；covered = 已到 target。
            tot, live, covered = dconn.execute(
                "SELECT COUNT(*),"
                "       SUM(md >= ?),"
                "       SUM(md >= ?)"
                " FROM (SELECT MAX(date) AS md FROM daily_data GROUP BY code)",
                (window_start, target),
            ).fetchone()
            live, covered = int(live or 0), int(covered or 0)
            missing = live - covered
            stale_excluded = tot - live  # 长期停牌/退市，不计入口径（仅日志说明用）
            # sync_status 谎言：活跃股 状态标到 >= target，但实际 MAX(date) < target。
            # 限定在活跃窗口内，避免死股历史遗留的“已同步”状态继续造成假阳性。
            lied = dconn.execute(
                "SELECT COUNT(*) FROM meta.sync_status s "
                "WHERE s.last_daily_sync >= ? "
                "AND (SELECT MAX(date) FROM daily_data d WHERE d.code = s.code) >= ? "
                "AND (SELECT MAX(date) FROM daily_data d WHERE d.code = s.code) < ?",
                (target, window_start, target),
            ).fetchone()[0]
            qrow = dconn.execute("SELECT count FROM meta.api_quota WHERE date = ?", (target,)).fetchone()
            quota_used = qrow[0] if qrow else 0
        finally:
            dconn.close()

        if missing > 0 or lied > 0:
            logger.warning(
                f"更新校验 ⚠ 目标交易日={target} | 已覆盖 {covered}/{live} 只 | 缺失 {missing} 只 | "
                f"sync_status谎言 {lied} 只 | 今日API {quota_used}"
                + (f" | 剔除长期停牌 {stale_excluded} 只" if stale_excluded else "")
            )
            logger.warning(
                "部分股票未更新到目标交易日（多为 Baostock 返回空/被拉黑）；"
                "下次增量运行会以 MAX(date) 为起点自动回填（前提：当时未被拉黑）。"
            )
        else:
            # 注意：变量名是 tot（total 从未定义过 —— 旧代码在这里会抛 NameError，
            # 于是"覆盖完整"的那天只会看到一条"更新校验失败"，成功路径从未生效过）
            logger.info(
                f"更新校验 ✓ 目标交易日={target} | 已覆盖 {covered}/{live} 只 | 缺失 0 只 | "
                f"今日API {quota_used}"
                + (f" | 剔除长期停牌 {stale_excluded} 只" if stale_excluded else "")
            )
    except Exception as e:
        logger.error(f"更新校验失败（不影响数据写入）: {e}")


def _run_daily_data_update():
    """隔离更新脚本的全局流/socket 设置，并为 Windows worker 提供有效标准句柄。"""
    project_root = get_project_root()
    log_dir = os.path.join(project_root, 'diagnose_output')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'daily_update_{datetime.now().strftime("%Y%m%d")}.log')
    logger.info(f"更新明细日志: {log_path}")
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'
    command = [sys.executable, '-u', os.path.join(project_root, 'scripts', 'update_daily_data.py')]
    # 在创建进程时重定向，而不是修改后端的 fd 1；stderr 和 spawn worker 也落盘。
    # 显式传 stdin，避免后台启动的 Windows 后端继承无效的控制台输入句柄。
    with open(log_path, 'a', encoding='utf-8') as log_file:
        try:
            subprocess.run(command, cwd=project_root, env=env,
                           stdin=subprocess.DEVNULL, stdout=log_file,
                           stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"每日更新进程退出码 {e.returncode}，详见 {log_path}") from e
    return log_path


async def daily_data_update_job():
    """
    工作日定时执行的更新任务。
    """
    started = datetime.now()
    logger.info("Executing scheduled daily data update...")

    try:
        # 执行增量更新；等待独立进程结束后才能校验覆盖和回填成交价。
        logger.info("Starting full incremental update...")
        await asyncio.get_running_loop().run_in_executor(None, _run_daily_data_update)

        elapsed = (datetime.now() - started).total_seconds()
        logger.info(f"Daily data update finished in {elapsed / 60:.1f} min. Verifying coverage...")
        verify_daily_update()
        logger.info("Triggering paper trading price auto-fill...")
        await asyncio.get_running_loop().run_in_executor(None, auto_fill_paper_trading_prices)
    except Exception:
        logger.exception("Error in scheduled daily update")
        raise


scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")

def start_scheduler():
    # Schedule to run from Monday to Friday at 19:00
    # 市场在周末及节假日通常不更新，这里只针对工作日设定单次触发
    trigger = CronTrigger(day_of_week="mon-fri", hour=18, minute=30, timezone="Asia/Shanghai")
    # max_instances=1 是 APScheduler 默认值，这里显式写出来提醒：
    # 一旦某次任务挂起（过去因 baostock 无响应永久阻塞过 12 天），
    # 之后每个交易日的定时都会被 skip。已用 socket 默认超时堵住这个根因，
    # 若再出现长时间挂起，日志里的耗时行会直接暴露。
    scheduler.add_job(daily_data_update_job, trigger, id="daily_data_update",
                      replace_existing=True, max_instances=1, coalesce=True,
                      misfire_grace_time=3600)
    scheduler.start()
    logger.info("APScheduler started: Daily data update scheduled for Mon-Fri at 18:30.")

def stop_scheduler():
    scheduler.shutdown()
    logger.info("APScheduler stopped.")
