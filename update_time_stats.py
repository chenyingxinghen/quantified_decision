#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计数据库中update_time为空或小于今天日期的股票数量
"""
import sqlite3
from datetime import datetime, timedelta

from pandas import to_datetime

from config import DATABASE_PATH


def count_stocks_with_old_update_time():
    """统计update_time为空或小于今天日期的股票数量"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # 获取今天的日期字符串

    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')
    # 查询update_time为空或小于今天日期的股票数量
    query = """
    SELECT COUNT(*) 
    FROM stock_info 
    WHERE update_time IS NULL OR date(update_time) < ?
    """
    
    cursor.execute(query, (today,))
    count = cursor.fetchone()[0]
    
    # 同时查询总股票数量
    cursor.execute("SELECT COUNT(*) FROM stock_info")
    total_count = cursor.fetchone()[0]
    
    # 查询update_time为今天的股票数量
    cursor.execute("SELECT COUNT(*) FROM stock_info WHERE date(update_time) = ?", (today,))
    today_count = cursor.fetchone()[0]
    
    # 查询update_time为今天但是时间不是今天的股票数量（即更新时间是今天但早于当前时间）
    cursor.execute("""
        SELECT COUNT(*) 
        FROM stock_info 
        WHERE date(update_time) = ? AND update_time < datetime('now')
    """, (today,))
    today_not_current_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"📊 股票数据更新统计:")
    print(f"  总股票数量: {total_count}")
    print(f"  update_time为空或小于今天的股票数量: {count}")
    print(f"  update_time为今天的股票数量: {today_count}")
    print(f"  update_time为今天但非最新的股票数量: {today_not_current_count}")
    
    return count, total_count


if __name__ == "__main__":
    count, total = count_stocks_with_old_update_time()
    print(f"\n📈 结果: {count} 只股票的 update_time 为空或小于今天日期")