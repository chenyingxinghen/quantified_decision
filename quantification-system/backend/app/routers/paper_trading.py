"""
手动实盘验证 API
"""

import os, sys, json, traceback
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Header
from pydantic import BaseModel

from app.deps import get_db_path, get_db_connection, get_paper_db, get_project_root
from app.routers.auth import get_current_user_from_token

router = APIRouter(prefix="/api/paper-trading", tags=["手动实盘验证"])


# ── Pydantic 模型 ─────────────────────────────────────────

class BuyRequest(BaseModel):
    code: str
    name: str = ""
    buy_date: str
    buy_price: float
    quantity: int = 1
    notes: str = ""


class SellRequest(BaseModel):
    position_id: int
    sell_date: str
    sell_price: float
    sell_reason: str = "手动卖出"


# ── 持仓列表 ──────────────────────────────────────────────

@router.get("/positions")
async def list_positions(status: str = Query(default="active", pattern="^(active|closed|all)$"), token: Optional[str] = Header(None)):
    """获取持仓列表"""
    username = get_current_user_from_token(token)
    conn = get_paper_db()
    if status == "all":
        rows = conn.execute("SELECT * FROM positions WHERE username = ? ORDER BY created_at DESC", (username,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM positions WHERE status = ? AND username = ? ORDER BY created_at DESC",
            (status, username)
        ).fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        # 注入最新价格
        if d["status"] == "active":
            d["latest_price"] = _get_latest_price(d["code"])
            if d["latest_price"] and d["buy_price"]:
                d["unrealized_pct"] = round(
                    (d["latest_price"] - d["buy_price"]) / d["buy_price"] * 100, 2
                )
            else:
                d["unrealized_pct"] = None
        results.append(d)
    return {"positions": results}


def _get_latest_price(code: str) -> Optional[float]:
    try:
        conn = get_db_connection()
        row = conn.execute(
            "SELECT close FROM daily_data WHERE code = ? ORDER BY date DESC LIMIT 1",
            (code,)
        ).fetchone()
        conn.close()
        return float(row["close"]) if row else None
    except Exception:
        return None


# ── 买入 ──────────────────────────────────────────────────

@router.post("/buy")
async def buy_stock(req: BuyRequest, token: Optional[str] = Header(None)):
    """记录买入"""
    username = get_current_user_from_token(token)
    conn = get_paper_db()
    conn.execute(
        """INSERT INTO positions (code, name, buy_date, buy_price, quantity, notes, username)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (req.code, req.name, req.buy_date, req.buy_price, req.quantity, req.notes, username),
    )
    conn.commit()
    pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return {"id": pid, "message": "买入记录已创建"}


# ── 卖出 ──────────────────────────────────────────────────

@router.post("/sell")
async def sell_stock(req: SellRequest, token: Optional[str] = Header(None)):
    """记录卖出"""
    username = get_current_user_from_token(token)
    conn = get_paper_db()
    row = conn.execute("SELECT * FROM positions WHERE id = ? AND username = ?", (req.position_id, username)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="持仓不存在或无权限")
    if row["status"] != "active":
        conn.close()
        raise HTTPException(status_code=400, detail="持仓已关闭")

    profit_pct = round((req.sell_price - row["buy_price"]) / row["buy_price"] * 100, 2)
    conn.execute(
        """UPDATE positions SET status='closed', sell_date=?, sell_price=?,
           sell_reason=?, profit_pct=? WHERE id=?""",
        (req.sell_date, req.sell_price, req.sell_reason, profit_pct, req.position_id),
    )
    conn.commit()
    conn.close()
    return {"message": "卖出已记录", "profit_pct": profit_pct}


# ── 检查卖出条件 ──────────────────────────────────────────

@router.get("/check-exit/{code}")
async def check_exit_conditions(
    code: str, 
    buy_price: float = Query(...), 
    buy_date: str = Query(...),
    atr_period: int = Query(14),
    atr_stop_multiplier: float = Query(2.0),
    atr_target_multiplier: float = Query(3.0),
    time_stop_days: int = Query(20),
    time_stop_min_loss_pct: float = Query(-0.05)
):
    """调用回测引擎检测卖出条件是否满足"""
    try:
        import pandas as pd
        import numpy as np
        conn = get_db_connection()
        df = pd.read_sql_query(
            "SELECT * FROM daily_data WHERE code = ? AND date >= ? ORDER BY date ASC",
            conn, params=(code, buy_date),
        )
        conn.close()
        if df.empty:
            raise HTTPException(status_code=404, detail="无数据")

        latest = df.iloc[-1]
        current_price = float(latest["close"])
        holding_days = len(df)

        # ATR 计算
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        tr = np.maximum(highs - lows,
                        np.maximum(np.abs(highs - np.roll(closes, 1)),
                                   np.abs(lows - np.roll(closes, 1))))
        tr[0] = highs[0] - lows[0]
        atr = pd.Series(tr).rolling(atr_period).mean().iloc[-1] if len(tr) >= atr_period else tr[-1]

        stop_loss_price = float(round(buy_price - atr * atr_stop_multiplier, 3))
        take_profit_price = float(round(buy_price + atr * atr_target_multiplier, 3))
        change_pct = float(round((current_price - buy_price) / buy_price * 100, 2))

        conditions = {
            "stop_loss": {
                "triggered": bool(current_price <= stop_loss_price),
                "price": stop_loss_price,
                "label": f"止损线 ({atr_stop_multiplier}×ATR)",
            },
            "take_profit": {
                "triggered": bool(current_price >= take_profit_price),
                "price": take_profit_price,
                "label": f"止盈线 ({atr_target_multiplier}×ATR)",
            },
            "time_stop": {
                "triggered": bool(holding_days >= time_stop_days and change_pct <= time_stop_min_loss_pct * 100),
                "holding_days": int(holding_days),
                "max_days": int(time_stop_days),
                "label": f"时间止损 ({time_stop_days}天)",
            },
        }

        return {
            "code": code,
            "buy_price": float(buy_price),
            "current_price": float(current_price),
            "change_pct": float(change_pct),
            "atr": round(float(atr), 3),
            "conditions": conditions,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── 历史交易 ──────────────────────────────────────────────

@router.get("/history")
async def trade_history(limit: int = Query(default=50, ge=1, le=500), token: Optional[str] = Header(None)):
    """获取历史交易记录"""
    username = get_current_user_from_token(token)
    conn = get_paper_db()
    rows = conn.execute(
        "SELECT * FROM positions WHERE status='closed' AND username = ? ORDER BY sell_date DESC LIMIT ?",
        (username, limit)
    ).fetchall()
    conn.close()
    return {"trades": [dict(r) for r in rows]}
