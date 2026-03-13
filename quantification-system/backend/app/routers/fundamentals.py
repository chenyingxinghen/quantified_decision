"""
基本面分析 API — /api/fundamental
"""

import os, traceback
from fastapi import APIRouter, HTTPException, Query

from app.deps import get_db_path, get_db_connection

router = APIRouter(prefix="/api/fundamental", tags=["基本面分析"])


# ── 股票检索 ──────────────────────────────────────────

@router.get("/search")
async def search_stocks(query: str = Query(..., min_length=1)):
    """支持代码 (前缀) 或中文名 (包含) 检索"""
    conn = get_db_connection()
    try:
        db_dir = os.path.dirname(get_db_path())
        meta_db = os.path.join(db_dir, 'stock_meta.db')
        if os.path.exists(meta_db):
            conn.execute(f"ATTACH DATABASE '{meta_db}' AS meta")

            sql = """
                SELECT code, name FROM meta.stock_info_extended
                WHERE code LIKE ? OR name LIKE ?
                LIMIT 15
            """
            rows = conn.execute(sql, (f"{query}%", f"%{query}%")).fetchall()
        else:
            sql = "SELECT DISTINCT code FROM daily_data WHERE code LIKE ? LIMIT 10"
            rows = conn.execute(sql, (f"{query}%",)).fetchall()

        return {"items": [dict(r) for r in rows]}
    except Exception as e:
        traceback.print_exc()
        return {"items": []}
    finally:
        conn.close()


# ── 基本面详情 ──────────────────────────────────────────

@router.get("/{code}")
async def get_stock_fundamental(code: str):
    """获取股票详细基本面信息"""
    db_dir = os.path.dirname(get_db_path())
    conn = get_db_connection()
    try:
        # 挂载 meta 和 finance
        meta_db = os.path.join(db_dir, 'stock_meta.db')
        finance_db = os.path.join(db_dir, 'stock_finance.db')
        if os.path.exists(meta_db):
            conn.execute(f"ATTACH DATABASE '{meta_db}' AS meta")
        if os.path.exists(finance_db):
            conn.execute(f"ATTACH DATABASE '{finance_db}' AS finance")

        # 1. 基础信息
        info = {}
        try:
            row = conn.execute("SELECT * FROM meta.stock_info_extended WHERE code = ?", (code,)).fetchone()
            if row:
                info = dict(row)
        except:
            pass

        # 2. 财务历史 (最近 12 期)
        finance = {}
        try:
            f_rows = conn.execute("""
                SELECT * FROM finance.finance_reports
                WHERE code = ?
                GROUP BY REPORT_DATE
                ORDER BY REPORT_DATE DESC LIMIT 12
            """, (code,)).fetchall()
            if f_rows:
                reports = []
                for r in f_rows:
                    rd = dict(r)
                    # 格式化日期，去时分秒
                    if rd.get("REPORT_DATE"):
                        rd["REPORT_DATE"] = str(rd["REPORT_DATE"]).split(' ')[0]
                    reports.append(rd)

                curr = reports[0]
                prev = reports[1] if len(reports) > 1 else None
                finance = {
                    "current": curr,
                    "previous": prev,
                    "history": reports,
                    "report_date": curr.get("REPORT_DATE")
                }
        except:
            traceback.print_exc()

        # 3. 实时价格计算的 PE/PB
        price_info = {}
        try:
            p_row = conn.execute(
                "SELECT close, date FROM daily_data WHERE code = ? ORDER BY date DESC LIMIT 1",
                (code,)
            ).fetchone()
            if p_row:
                close = p_row["close"]
                price_info = {"close": close, "date": p_row["date"]}
                if finance.get("current"):
                    eps = finance["current"].get("EPSJB")
                    bps = finance["current"].get("BPS")
                    if eps and eps > 0:
                        price_info["pe"] = round(close / eps, 2)
                    if bps and bps > 0:
                        price_info["pb"] = round(close / bps, 2)
        except:
            pass

        return {
            "code": code,
            "info": info,
            "finance": finance,
            "valuation": price_info
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
