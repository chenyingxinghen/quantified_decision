#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据库诊断工具"""

import sqlite3
import os
from config import DATABASE_PATH

def check_database():
    """检查数据库状态"""
    print("=" * 60)
    print("数据库诊断工具")
    print("=" * 60)
    
    try:
        # 确保使用项目根目录的数据库路径
        db_path = DATABASE_PATH
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 检查 stock_info 表
        print("\n【股票基本信息表 (stock_info)】")
        cursor.execute("SELECT COUNT(*) FROM stock_info")
        stock_count = cursor.fetchone()[0]
        print(f"总股票数量: {stock_count}")
        
        if stock_count > 0:
            # 显示前5条记录
            cursor.execute("SELECT code, name, market_cap, pe_ratio, pb_ratio FROM stock_info LIMIT 5")
            print("\n前5条记录示例:")
            for row in cursor.fetchall():
                print(f"  代码: {row[0]}, 名称: {row[1]}, 市值: {row[2]}, PE: {row[3]}, PB: {row[4]}")
            
            # 统计有效数据
            cursor.execute("""
                SELECT COUNT(*) FROM stock_info 
                WHERE market_cap IS NOT NULL AND market_cap > 0
            """)
            valid_market_cap = cursor.fetchone()[0]
            print(f"\n有市值数据的股票: {valid_market_cap}")
            
            cursor.execute("""
                SELECT COUNT(*) FROM stock_info 
                WHERE pe_ratio IS NOT NULL AND pe_ratio > 0
            """)
            valid_pe = cursor.fetchone()[0]
            print(f"有市盈率数据的股票: {valid_pe}")
        
        # 检查 daily_data 表
        print("\n【日线数据表 (daily_data)】")
        cursor.execute("SELECT COUNT(*) FROM daily_data")
        daily_count = cursor.fetchone()[0]
        print(f"总记录数: {daily_count}")
        
        if daily_count > 0:
            # 统计有数据的股票数量
            cursor.execute("SELECT COUNT(DISTINCT code) FROM daily_data")
            stocks_with_data = cursor.fetchone()[0]
            print(f"有日线数据的股票数量: {stocks_with_data}")
            
            # 显示数据最多的5只股票
            cursor.execute("""
                SELECT code, COUNT(*) as count 
                FROM daily_data 
                GROUP BY code 
                ORDER BY count DESC 
                LIMIT 5
            """)
            print("\n数据最多的5只股票:")
            for row in cursor.fetchall():
                print(f"  {row[0]}: {row[1]} 条记录")
            
            # 显示最新日期
            cursor.execute("SELECT MAX(date) FROM daily_data")
            latest_date = cursor.fetchone()[0]
            print(f"\n最新数据日期: {latest_date}")
            
            # 显示一条完整记录示例
            cursor.execute("""
                SELECT code, date, open, high, low, close, volume 
                FROM daily_data 
                ORDER BY date DESC 
                LIMIT 1
            """)
            print("\n最新记录示例:")
            row = cursor.fetchone()
            if row:
                print(f"  代码: {row[0]}")
                print(f"  日期: {row[1]}")
                print(f"  开盘: {row[2]}, 最高: {row[3]}, 最低: {row[4]}, 收盘: {row[5]}")
                print(f"  成交量: {row[6]}")
        
        # 检查符合筛选条件的股票
        print("\n【筛选条件检查】")
        from config import SELECTION_CRITERIA
        
        print(f"最小市值: {SELECTION_CRITERIA['min_market_cap']:,}")
        print(f"最大市盈率: {SELECTION_CRITERIA['max_pe']}")
        print(f"最小价格: {SELECTION_CRITERIA['min_price']}")
        print(f"最大价格: {SELECTION_CRITERIA['max_price']}")
        print(f"最小换手率: {SELECTION_CRITERIA['min_turnover_rate']/100:.2%}")
        
        cursor.execute("""
            SELECT COUNT(*) FROM stock_info
            WHERE market_cap >= ? 
            AND pe_ratio > 0 AND pe_ratio <= ?
            AND pb_ratio > 0
        """, (SELECTION_CRITERIA['min_market_cap'], SELECTION_CRITERIA['max_pe']))
        
        qualified_count = cursor.fetchone()[0]
        print(f"\n符合基本面条件的股票数量: {qualified_count}")
        
        if qualified_count > 0:
            cursor.execute("""
                SELECT code, name, market_cap, pe_ratio 
                FROM stock_info
                WHERE market_cap >= ? 
                AND pe_ratio > 0 AND pe_ratio <= ?
                AND pb_ratio > 0
                LIMIT 5
            """, (SELECTION_CRITERIA['min_market_cap'], SELECTION_CRITERIA['max_pe']))
            
            print("\n符合条件的股票示例:")
            for row in cursor.fetchall():
                print(f"  {row[0]} - {row[1]}: 市值={row[2]:,.0f}, PE={row[3]:.2f}")
        
        conn.close()
        
        # 给出建议
        print("\n【诊断建议】")
        if stock_count == 0:
            print("❌ 数据库为空，请先运行菜单选项 1 初始化数据")
        elif daily_count == 0:
            print("❌ 没有日线数据，请先运行菜单选项 1 初始化数据")
        elif qualified_count == 0:
            print("⚠️  没有符合筛选条件的股票，建议:")
            print("   1. 检查 config.py 中的 SELECTION_CRITERIA 是否过于严格")
            print("   2. 确保已运行数据初始化（菜单选项 1）")
            print("   3. 运行增量更新获取最新数据（菜单选项 2）")
        else:
            print("✅ 数据库状态正常")
        
    except Exception as e:
        print(f"❌ 检查数据库时出错: {e}")

if __name__ == "__main__":
    check_database()
