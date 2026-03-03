"""
手动实盘验证 API
"""

import os, sys, json, traceback
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Header
from pydantic import BaseModel

from app.deps import get_db_path, get_db_connection, get_user_db, get_project_root
from app.routers.auth import get_current_user_from_token

router = APIRouter(prefix="/api/paper-trading", tags=["手动实盘验证"])


# ── Pydantic 模型 ─────────────────────────────────────────

class BuyRequest(BaseModel):
    code: str
    name: str = ""
    buy_date: str
    buy_price: Optional[float] = None
    quantity: int = 1
    notes: str = ""


class SellRequest(BaseModel):
    position_id: int
    sell_date: str
    sell_price: float
    sell_reason: str = "手动卖出"


# ── 持仓列表 ──────────────────────────────────────────────

@router.get("/positions")
async def list_positions(status: str = Query(default="active", pattern="^(active|closed|all)$"), token: Optional[str] = Header(None, alias="Token")):
    """获取持仓列表"""
    username = get_current_user_from_token(token)
    conn = get_user_db()
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
            if d["latest_price"] and d.get("buy_price") is not None:
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
async def buy_stock(req: BuyRequest, token: Optional[str] = Header(None, alias="Token")):
    """记录买入"""
    username = get_current_user_from_token(token)
    conn = get_user_db()
    
    # Check if buy_date is today or future, and if no price provided, set to None.
    # Otherwise if we still want None by default, we just insert.
    buy_price_val = req.buy_price
    
    conn.execute(
        """INSERT INTO positions (code, name, buy_date, buy_price, quantity, notes, username)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (req.code, req.name, req.buy_date, buy_price_val, req.quantity, req.notes, username),
    )
    conn.commit()
    pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    
    msg = "买入记录已创建"
    if buy_price_val is None:
        msg = "买入记录已创建 (等待收市更新开盘价)"
        
    return {"id": pid, "message": msg}


# ── 卖出 ──────────────────────────────────────────────────

@router.post("/sell")
async def sell_stock(req: SellRequest, token: Optional[str] = Header(None, alias="Token")):
    """记录卖出"""
    username = get_current_user_from_token(token)
    conn = get_user_db()
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


# ── 监控管理 ──────────────────────────────────────────────

@router.post("/toggle-monitoring/{position_id}")
async def toggle_monitoring(position_id: int, token: Optional[str] = Header(None, alias="Token")):
    """开启/关闭持仓监控"""
    username = get_current_user_from_token(token)
    conn = get_user_db()
    row = conn.execute("SELECT monitoring FROM positions WHERE id = ? AND username = ?", (position_id, username)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="持仓不存在")
    
    new_status = 0 if row["monitoring"] else 1
    conn.execute("UPDATE positions SET monitoring = ? WHERE id = ?", (new_status, position_id))
    conn.commit()
    conn.close()
    return {"monitoring": bool(new_status)}


@router.delete("/{position_id}")
async def delete_position(position_id: int, token: Optional[str] = Header(None, alias="Token")):
    """删除持仓记录"""
    username = get_current_user_from_token(token)
    conn = get_user_db()
    res = conn.execute("DELETE FROM positions WHERE id = ? AND username = ?", (position_id, username))
    conn.commit()
    count = res.rowcount
    conn.close()
    if count == 0:
        raise HTTPException(status_code=404, detail="持仓不存在或无权限")
    return {"message": "已删除监控记录"}


# ── 检查卖出条件 ──────────────────────────────────────────

@router.get("/check-exit/{code}")
async def check_exit_conditions(
    code: str, 
    buy_price: float = Query(...), 
    buy_date: str = Query(...),
    token: Optional[str] = Header(None, alias="Token"),
    # 允许透传所有策略参数作为查询参数 (使用 Alias 匹配前端大写字段)
    atr_period: Optional[int] = Query(None, alias="ATR_PERIOD"),
    atr_stop_multiplier: Optional[float] = Query(None, alias="ATR_STOP_MULTIPLIER"),
    atr_target_multiplier: Optional[float] = Query(None, alias="ATR_TARGET_MULTIPLIER"),
    time_stop_days: Optional[int] = Query(None, alias="TIME_STOP_DAYS"),
    time_stop_min_loss_pct: Optional[float] = Query(None, alias="TIME_STOP_MIN_LOSS_PCT"),
    enable_stop_loss: Optional[bool] = Query(None, alias="ENABLE_STOP_LOSS_EXIT"),
    enable_take_profit: Optional[bool] = Query(None, alias="ENABLE_TAKE_PROFIT_EXIT"),
    enable_time_stop: Optional[bool] = Query(None, alias="ENABLE_TIME_STOP_EXIT"),
    enable_support_break: Optional[bool] = Query(None, alias="ENABLE_SUPPORT_BREAK_EXIT"),
):
    """
    检查特定股票持仓的退出条件
    优先级：查询参数 > 用户数据库配置 > 全局默认配置
    """
    # 1. 加载全局配置作为基础默认
    from config import strategy_config as sc
    
    # 获取默认值
    def_atr_period = getattr(sc, 'ATR_PERIOD', 14)
    def_atr_stop_multiplier = getattr(sc, 'ATR_STOP_MULTIPLIER', 2.0)
    def_atr_target_multiplier = getattr(sc, 'ATR_TARGET_MULTIPLIER', 3.0)
    def_time_stop_days = getattr(sc, 'TIME_STOP_DAYS', 20)
    def_time_stop_min_loss_pct = getattr(sc, 'TIME_STOP_MIN_LOSS_PCT', -0.05)
    
    def_enable_stop_loss = getattr(sc, 'ENABLE_STOP_LOSS_EXIT', True)
    def_enable_take_profit = getattr(sc, 'ENABLE_TAKE_PROFIT_EXIT', True)
    def_enable_time_stop = getattr(sc, 'ENABLE_TIME_STOP_EXIT', True)
    def_enable_support_break = getattr(sc, 'ENABLE_SUPPORT_BREAK_EXIT', False)

    # 2. 尝试从数据库加载用户个人的覆盖配置
    db_config = {}
    try:
        username = None
        if token and token != 'guest':
            username = get_current_user_from_token(token)
        else:
            # 即使没 token，也可以尝试通过 code/date 找持仓的所有者 (兼顾某些调用场景)
            conn_u = get_user_db()
            pos_row = conn_u.execute("SELECT username FROM positions WHERE code = ? AND buy_date = ? LIMIT 1", (code, buy_date)).fetchone()
            if pos_row:
                username = pos_row["username"]
            conn_u.close()
            
        if username:
            conn_u = get_user_db()
            conf_row = conn_u.execute("SELECT config_json FROM user_configs WHERE username = ?", (username,)).fetchone()
            if conf_row and conf_row["config_json"]:
                db_config = json.loads(conf_row["config_json"])
            conn_u.close()
    except Exception as e:
        print(f"Loading user config from DB failed: {e}")

    # 3. 合并参数优先级 (Query > DB > Global Default)
    # 处理 ATR
    final_atr_period = atr_period if atr_period is not None else db_config.get('ATR_PERIOD', def_atr_period)
    final_atr_stop = atr_stop_multiplier if atr_stop_multiplier is not None else db_config.get('ATR_STOP_MULTIPLIER', def_atr_stop_multiplier)
    final_atr_target = atr_target_multiplier if atr_target_multiplier is not None else db_config.get('ATR_TARGET_MULTIPLIER', def_atr_target_multiplier)
    
    # 处理时间止损
    final_time_days = time_stop_days if time_stop_days is not None else db_config.get('TIME_STOP_DAYS', def_time_stop_days)
    final_time_loss = time_stop_min_loss_pct if time_stop_min_loss_pct is not None else db_config.get('TIME_STOP_MIN_LOSS_PCT', def_time_stop_min_loss_pct)
    
    # 处理启用标志 (注意 Boolean 处理)
    final_en_stop = enable_stop_loss if enable_stop_loss is not None else db_config.get('ENABLE_STOP_LOSS_EXIT', def_enable_stop_loss)
    final_en_profit = enable_take_profit if enable_take_profit is not None else db_config.get('ENABLE_TAKE_PROFIT_EXIT', def_enable_take_profit)
    final_en_time = enable_time_stop if enable_time_stop is not None else db_config.get('ENABLE_TIME_STOP_EXIT', def_enable_time_stop)
    final_en_support = enable_support_break if enable_support_break is not None else db_config.get('ENABLE_SUPPORT_BREAK_EXIT', def_enable_support_break)
    
    # 将计算逻辑中使用的变量替换为 final_xxx
    atr_period = final_atr_period
    atr_stop_multiplier = final_atr_stop
    atr_target_multiplier = final_atr_target
    time_stop_days = final_time_days
    time_stop_min_loss_pct = final_time_loss
    enable_stop_loss = final_en_stop
    enable_take_profit = final_en_profit
    enable_time_stop = final_en_time
    enable_support_break = final_en_support
    
    try:
        import pandas as pd
        import numpy as np
        conn = get_db_connection()
        df = pd.read_sql_query(
            "SELECT * FROM daily_data WHERE code = ? AND date >= ? ORDER BY date ASC",
            conn, params=(code, buy_date),
        )
        conn.close()
        
        # 修复：检测不到数据时，则等待数据更新后再尝试
        if df.empty:
            return {
                "message": "等待行情数据更新...",
                "status": "pending",
                "conditions": {}
            }

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

        # 比例计算防 0
        safe_buy_price = buy_price if buy_price > 0 else current_price
        
        stop_loss_price = float(round(safe_buy_price - atr * atr_stop_multiplier, 3))
        take_profit_price = float(round(safe_buy_price + atr * atr_target_multiplier, 3))
        
        # 修复 ZeroDivisionError
        change_pct = float(round((current_price - safe_buy_price) / safe_buy_price * 100, 2)) if safe_buy_price > 0 else 0

        # Progress Calculation
        def get_progress(curr, start, target):
            if target == start: return 0
            prog = (curr - start) / (target - start) * 100
            return float(round(max(0, min(100, prog)), 1))

        conditions = {}
        
        if enable_stop_loss:
            conditions["stop_loss"] = {
                "triggered": bool(current_price <= stop_loss_price),
                "price": stop_loss_price,
                "label": f"ATR止损监控",
                "progress": get_progress(current_price, safe_buy_price, stop_loss_price)
            }
        
        if enable_take_profit:
            conditions["take_profit"] = {
                "triggered": bool(current_price >= take_profit_price),
                "price": take_profit_price,
                "label": f"ATR止盈监控",
                "progress": get_progress(current_price, safe_buy_price, take_profit_price)
            }
            
        if enable_time_stop:
            conditions["time_stop"] = {
                "triggered": bool(holding_days >= time_stop_days and change_pct <= time_stop_min_loss_pct * 100),
                "holding_days": int(holding_days),
                "max_days": int(time_stop_days),
                "label": f"时间止损监控",
                "progress": get_progress(holding_days, 0, time_stop_days)
            }

        if enable_support_break:
            # 引入趋势线分析
            try:
                from core.analysis.trend_line_analyzer import TrendLineAnalyzer
                analyzer = TrendLineAnalyzer()
                # 需要更多历史数据来分析趋势线
                conn_h = get_db_connection()
                df_h = pd.read_sql_query(
                    "SELECT * FROM daily_data WHERE code = ? ORDER BY date DESC LIMIT 200",
                    conn_h, params=(code,),
                )
                conn_h.close()
                df_h = df_h.sort_values("date").reset_index(drop=True)
                df_h["date"] = pd.to_datetime(df_h["date"])
                df_h = df_h.set_index("date")
                
                analysis = analyzer.analyze(df_h)
                is_broken = analysis.get("broken_support", False)
                
                # 获取支撑位价格 (取短期上升趋势线在当前日期的值)
                support_price = None
                short_up = analysis.get("short_uptrend_line", {})
                if short_up.get("valid"):
                    support_price = float(round(analyzer._get_trendline_value(short_up, df_h.index[-1]), 3))
                
                # 计算跌破支撑的进度: (当前价 - 支撑价) / 支撑价
                # 如果当前价接近支撑价，进度越高
                progress = 0.0
                if support_price and support_price > 0:
                    if is_broken:
                        progress = 100.0
                    else:
                        # 距离支撑位的接近程度 (例如 5% 以内开始显示进度)
                        dist = (current_price - support_price) / support_price
                        if dist <= 0: progress = 100.0
                        elif dist > 0.1: progress = 0.0 # 距离太远不显示进度
                        else: progress = round((0.1 - dist) / 0.1 * 100, 1)

                conditions["support_break"] = {
                    "triggered": bool(is_broken),
                    "label": "跌破支撑卖出",
                    "price": support_price,
                    "progress": progress
                }
            except Exception as e:
                print(f"Support break analysis failed: {e}")

        return {
            "code": code,
            "buy_price": float(buy_price),
            "current_price": float(current_price),
            "change_pct": float(change_pct),
            "atr": round(float(atr), 3),
            "conditions": conditions,
            "status": "success"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── 历史交易 ──────────────────────────────────────────────

@router.get("/history")
async def trade_history(limit: int = Query(default=50, ge=1, le=500), token: Optional[str] = Header(None, alias="Token")):
    """获取历史交易记录"""
    username = get_current_user_from_token(token)
    conn = get_user_db()
    rows = conn.execute(
        "SELECT * FROM positions WHERE status='closed' AND username = ? ORDER BY sell_date DESC LIMIT ?",
        (username, limit)
    ).fetchall()
    conn.close()
    return {"trades": [dict(r) for r in rows]}
