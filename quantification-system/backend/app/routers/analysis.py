"""
技术分析 API — K 线、趋势线、形态识别
"""

import os, sys, json, traceback
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from ..deps import get_db_connection

router = APIRouter(prefix="/api/analysis", tags=["技术分析"])


@router.get("/kline/{code}")
async def get_kline(code: str, days: int = Query(default=250, ge=10, le=3000)):
    """获取 K 线 OHLCV 数据"""
    import pandas as pd
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM daily_data WHERE code = ? ORDER BY date DESC LIMIT ?",
            conn, params=(code, days),
        )
    finally:
        conn.close()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"股票 {code} 无数据")

    df = df.sort_values("date").reset_index(drop=True)
    # 构造 ECharts 友好的数据结构
    columns_map = {
        "date": "date", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "volume",
    }
    available = {k: v for k, v in columns_map.items() if k in df.columns}
    result = df[list(available.keys())].rename(columns=available)

    # 转为 list 格式
    dates = result["date"].tolist()
    values = []
    for _, row in result.iterrows():
        values.append([
            float(row.get("open", 0)),
            float(row.get("close", 0)),
            float(row.get("low", 0)),
            float(row.get("high", 0)),
        ])
    volumes = result["volume"].astype(float).tolist() if "volume" in result.columns else []

    return {
        "code": code,
        "dates": dates,
        "values": values,  # [open, close, low, high] per ECharts candlestick
        "volumes": volumes,
    }


@router.get("/trendlines/{code}")
async def get_trendlines(
    code: str, 
    days: int = Query(default=250, ge=30, le=3000),
    long_period: int = Query(default=None),
    short_period: int = Query(default=None)
):
    """获取支撑阻力趋势线"""
    try:
        import pandas as pd
        conn = get_db_connection()
        df = pd.read_sql_query(
            "SELECT * FROM daily_data WHERE code = ? ORDER BY date DESC LIMIT ?",
            conn, params=(code, days),
        )
        conn.close()
        if df.empty:
            raise HTTPException(status_code=404, detail=f"股票 {code} 无数据")

        df = df.sort_values("date").reset_index(drop=True)
        # 确保 date 列为索引（DatetimeIndex）
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        from core.analysis.trend_line_analyzer import TrendLineAnalyzer
        # 构造参数
        analyzer_kwargs = {}
        if long_period: analyzer_kwargs['long_period'] = long_period
        if short_period: analyzer_kwargs['short_period'] = short_period
        
        analyzer = TrendLineAnalyzer(**analyzer_kwargs)
        analysis = analyzer.analyze(df)

        # 将趋势线序列化为前端可用格式
        def serialize_line(line_info):
            if line_info is None:
                return None
            result = {}
            for k, v in line_info.items():
                if isinstance(v, (int, float, bool, str)):
                    result[k] = v
                elif hasattr(v, "isoformat"):
                    result[k] = v.isoformat()
                elif isinstance(v, list):
                    result[k] = [
                        {"date": p[0].isoformat() if hasattr(p[0], "isoformat") else str(p[0]),
                         "price": float(p[1])}
                        if isinstance(p, (list, tuple)) and len(p) >= 2
                        else str(p)
                        for p in v
                    ]
                else:
                    result[k] = str(v)
            return result

        return {
            "code": code,
            "uptrend_line": serialize_line(analysis.get("uptrend_line")),
            "downtrend_line": serialize_line(analysis.get("downtrend_line")),
            "short_uptrend_line": serialize_line(analysis.get("short_uptrend_line")),
            "short_downtrend_line": serialize_line(analysis.get("short_downtrend_line")),
            "broken_support": analysis.get("broken_support", False),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/{code}")
async def get_patterns(code: str, days: int = Query(default=100, ge=10, le=1000)):
    """获取 PA 形态识别结果 + K 线形态"""
    try:
        import pandas as pd
        conn = get_db_connection()
        df = pd.read_sql_query(
            "SELECT * FROM daily_data WHERE code = ? ORDER BY date DESC LIMIT ?",
            conn, params=(code, days),
        )
        conn.close()
        if df.empty:
            raise HTTPException(status_code=404, detail=f"股票 {code} 无数据")

        df = df.sort_values("date").reset_index(drop=True)

        # K 线形态 - 扫描近 90 天的历史信号
        from core.analysis.candlestick_patterns import CandlestickPatterns
        cp = CandlestickPatterns()
        
        # 使用高效的向量化历史扫描
        scan_len = min(len(df), 90)
        history = cp.scan_patterns_history(df, scan_len=scan_len)
        bullish_history = history['bullish']
        bearish_history = history['bearish']

        # PA 市场结构 (针对全量数据)
        from core.analysis.price_action_analyzer import PriceActionAnalyzer
        pa = PriceActionAnalyzer()
        structure = pa.identify_market_structure(df)

        return {
            "code": code,
            "bullish_patterns": bullish_history,
            "bearish_patterns": bearish_history,
            "market_structure": structure,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-structure/{code}")
async def get_market_structure(code: str, days: int = Query(default=250, ge=30, le=2000)):
    """获取市场结构分析"""
    try:
        import pandas as pd
        conn = get_db_connection()
        df = pd.read_sql_query(
            "SELECT * FROM daily_data WHERE code = ? ORDER BY date DESC LIMIT ?",
            conn, params=(code, days),
        )
        conn.close()
        if df.empty:
            raise HTTPException(status_code=404, detail=f"股票 {code} 无数据")

        df = df.sort_values("date").reset_index(drop=True)
        from core.analysis.price_action_analyzer import PriceActionAnalyzer
        pa = PriceActionAnalyzer()
        return {"code": code, "structure": pa.identify_market_structure(df)}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
