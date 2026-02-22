"""
数据中心 API
"""

import os, sys, traceback, asyncio, json
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.deps import get_db_path, get_db_connection, get_project_root

router = APIRouter(prefix="/api/data-center", tags=["数据中心"])

# 后台更新任务状态
_update_task = {"running": False, "progress": "", "error": None}


@router.get("/status")
async def database_status():
    """查询数据库状态"""
    conn = get_db_connection()
    try:
        # 总股票数
        total = conn.execute(
            "SELECT COUNT(DISTINCT code) as cnt FROM daily_data"
        ).fetchone()["cnt"]

        # 最新日期
        latest = conn.execute(
            "SELECT MAX(date) as d FROM daily_data"
        ).fetchone()["d"]

        # 最早日期
        earliest = conn.execute(
            "SELECT MIN(date) as d FROM daily_data"
        ).fetchone()["d"]

        # 检查最近 30 天是否有缺失 (简单采样 20 只活跃股票)
        missing_data_info = []
        try:
            today = datetime.now()
            # 找到最近 30 天的所有工作日 (粗略估算)
            sample_stocks = conn.execute(
                "SELECT code FROM daily_data GROUP BY code ORDER BY COUNT(*) DESC LIMIT 20"
            ).fetchall()
            
            for s in sample_stocks:
                code = s["code"]
                last_date_str = conn.execute(
                    "SELECT MAX(date) as d FROM daily_data WHERE code = ?", (code,)
                ).fetchone()["d"]
                if last_date_str:
                    last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
                    diff = (today - last_date).days
                    if diff > 3: # 超过 3 天没更新则标记
                        missing_data_info.append({"code": code, "last_date": last_date_str, "days_ago": diff})
        except Exception:
            pass

        return {
            "total_stocks": total,
            "latest_date": latest,
            "earliest_date": earliest,
            "update_running": _update_task["running"],
            "missing_data_info": missing_data_info[:10], # 只返回前 10 条
        }
    finally:
        conn.close()


class UpdateRequest(BaseModel):
    mode: str = "all"            # single / multiple / all
    symbols: list[str] = []      # mode=single/multiple 时使用
    source: str = "yfinance"     # akshare / yfinance
    incremental: bool = True


@router.post("/update")
async def trigger_update(req: UpdateRequest):
    """手动触发数据更新（后台执行）"""
    global _update_task
    if _update_task["running"]:
        raise HTTPException(status_code=409, detail="已有更新任务在运行中")

    _update_task = {"running": True, "progress": "正在启动...", "error": None}

    async def _run():
        global _update_task
        try:
            from scripts.update_daily_data import (
                update_single_stock, update_multiple_stocks, update_all_stocks,
            )
            if req.mode == "single" and req.symbols:
                _update_task["progress"] = f"更新 {req.symbols[0]}"
                update_single_stock(req.symbols[0], prefer_source=req.source,
                                    incremental=req.incremental)
            elif req.mode == "multiple" and req.symbols:
                _update_task["progress"] = f"批量更新 {len(req.symbols)} 只"
                update_multiple_stocks(req.symbols, prefer_source=req.source,
                                       incremental=req.incremental)
            else:
                _update_task["progress"] = "更新全部股票"
                update_all_stocks(prefer_source=req.source,
                                  incremental=req.incremental)
            _update_task["progress"] = "完成"
        except Exception as e:
            _update_task["error"] = str(e)
            traceback.print_exc()
        finally:
            _update_task["running"] = False

    asyncio.get_event_loop().run_in_executor(None, asyncio.run, _run())
    return {"message": "更新任务已启动", "mode": req.mode}


@router.get("/update-status")
async def update_status():
    """查询更新进度"""
    return _update_task
