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
