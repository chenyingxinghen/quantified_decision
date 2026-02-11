"""
数据库优化脚本
创建必要的索引以加快查询速度
"""

import sqlite3
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATABASE_PATH


def optimize_database(db_path: str = DATABASE_PATH):
    """
    优化数据库性能
    
    参数:
        db_path: 数据库路径
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("="*80)
    print("数据库优化")
    print("="*80)
    
    # 1. 检查现有索引
    print("\n1. 检查现有索引...")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='daily_data'")
    existing_indices = cursor.fetchall()
    print(f"   现有索引: {[idx[0] for idx in existing_indices]}")
    
    # 2. 创建复合索引（code, date）
    print("\n2. 创建复合索引 (code, date)...")
    try:
        start_time = time.time()
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_data_code_date 
            ON daily_data(code, date)
        """)
        conn.commit()
        elapsed = time.time() - start_time
        print(f"   ✓ 索引创建成功 ({elapsed:.2f}s)")
    except Exception as e:
        print(f"   ✗ 索引创建失败: {e}")
    
    # 3. 创建日期索引
    print("\n3. 创建日期索引...")
    try:
        start_time = time.time()
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_data_date 
            ON daily_data(date)
        """)
        conn.commit()
        elapsed = time.time() - start_time
        print(f"   ✓ 索引创建成功 ({elapsed:.2f}s)")
    except Exception as e:
        print(f"   ✗ 索引创建失败: {e}")
    
    # 4. 分析表统计
    print("\n4. 分析表统计...")
    try:
        cursor.execute("ANALYZE")
        conn.commit()
        print("   ✓ 表统计分析完成")
    except Exception as e:
        print(f"   ✗ 分析失败: {e}")
    
    # 5. 显示表信息
    print("\n5. 表信息统计...")
    cursor.execute("SELECT COUNT(*) FROM daily_data")
    row_count = cursor.fetchone()[0]
    print(f"   总行数: {row_count:,}")
    
    cursor.execute("SELECT COUNT(DISTINCT code) FROM daily_data")
    stock_count = cursor.fetchone()[0]
    print(f"   股票数: {stock_count:,}")
    
    cursor.execute("SELECT MIN(date), MAX(date) FROM daily_data")
    min_date, max_date = cursor.fetchone()
    print(f"   日期范围: {min_date} 至 {max_date}")
    
    # 6. 优化 VACUUM
    print("\n6. 执行 VACUUM 优化...")
    try:
        start_time = time.time()
        cursor.execute("VACUUM")
        conn.commit()
        elapsed = time.time() - start_time
        print(f"   ✓ VACUUM 完成 ({elapsed:.2f}s)")
    except Exception as e:
        print(f"   ✗ VACUUM 失败: {e}")
    
    conn.close()
    
    print("\n" + "="*80)
    print("数据库优化完成！")
    print("="*80)


if __name__ == '__main__':
    optimize_database()
