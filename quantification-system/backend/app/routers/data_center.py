"""
数据中心 API — 负责数据库状态监控与数据同步
"""
from fastapi import APIRouter
from app.deps import get_db_connection
import sqlite3
from config import META_DB_PATH

router = APIRouter(prefix="/api/data-center", tags=["数据中心"])

@router.get("/status")
async def get_status():
    conn = get_db_connection()
    try:
        # 挂载元数据库
        import os
        meta_db = META_DB_PATH
        if os.path.exists(meta_db):
            conn.execute(f"ATTACH DATABASE '{meta_db}' AS meta")
            # 1. 获取全市场标的总数
            row_total = conn.execute("SELECT COUNT(*) FROM meta.stock_basic").fetchone()
            total_stocks = row_total[0] if row_total else 5500
        else:
            total_stocks = 5500

        # 2. 获取已同步行情数据的股票数
        row_synced = conn.execute("SELECT COUNT(DISTINCT code) FROM daily_data").fetchone()
        synced_stocks = row_synced[0] if row_synced else 0

        # 3. 获取整体同步日期 (从 meta 表获取最新更新记录)
        latest_date = "—"
        if os.path.exists(meta_db):
            row_date = conn.execute("SELECT MAX(update_time) FROM meta.stock_basic").fetchone()
            if row_date and row_date[0]:
                latest_date = row_date[0][:10]  # 取 YYYY-MM-DD

        # 4. 查找行情落后的标的 (随机抽取5个)
        missing_data_info = []

        return {
            "total_stocks": total_stocks,
            "synced_stocks": synced_stocks,
            "latest_date": latest_date,
            "missing_data_info": missing_data_info,
            "status": "online"
        }
    except sqlite3.Error as e:
        print(f"  [ERROR] 数据库访问错误: {e}")
        return {"error": f"数据库访问错误: {str(e)}", "total_stocks": total_stocks, "latest_date": "—", "missing_data_info": []}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"系统内部错误: {str(e)}", "total_stocks": total_stocks, "latest_date": "—", "missing_data_info": []}
    finally:
        conn.close()

@router.post("/update")
async def trigger_update(params: dict):
    # 此处预留数据更新接口，后续对接 scripts/update_daily_data.py
    return {"message": "数据更新功能正在对接中", "params": params}

@router.get("/update-status")
async def get_update_status():
    return {"running": False, "progress": "idle"}
