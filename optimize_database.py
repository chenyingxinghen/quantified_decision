# 数据库优化脚本 - 添加索引提升查询性能

import sqlite3
from config import DATABASE_PATH
import os


def optimize_database():
    """优化数据库性能"""
    
    # 确保使用绝对路径
    db_path = DATABASE_PATH
    if not os.path.isabs(DATABASE_PATH):
        project_root = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(project_root, DATABASE_PATH)
    
    print(f"正在优化数据库: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. 创建索引
    print("\n创建索引...")
    
    indexes = [
        # daily_data表索引
        ("idx_daily_code", "CREATE INDEX IF NOT EXISTS idx_daily_code ON daily_data(code)"),
        ("idx_daily_date", "CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_data(date)"),
        ("idx_daily_code_date", "CREATE INDEX IF NOT EXISTS idx_daily_code_date ON daily_data(code, date)"),
        
        # stock_info表索引
        ("idx_stock_code", "CREATE INDEX IF NOT EXISTS idx_stock_code ON stock_info(code)"),
    ]
    
    for idx_name, sql in indexes:
        try:
            cursor.execute(sql)
            print(f"  ✓ 创建索引: {idx_name}")
        except Exception as e:
            print(f"  ✗ 创建索引失败 {idx_name}: {e}")
    
    # 2. 分析表统计信息
    print("\n分析表统计信息...")
    try:
        cursor.execute("ANALYZE")
        print("  ✓ 统计信息更新完成")
    except Exception as e:
        print(f"  ✗ 分析失败: {e}")
    
    # 3. 清理数据库
    print("\n清理数据库...")
    try:
        cursor.execute("VACUUM")
        print("  ✓ 数据库清理完成")
    except Exception as e:
        print(f"  ✗ 清理失败: {e}")
    
    # 4. 显示数据库统计
    print("\n数据库统计:")
    
    # 股票数量
    cursor.execute("SELECT COUNT(DISTINCT code) FROM daily_data")
    stock_count = cursor.fetchone()[0]
    print(f"  股票数量: {stock_count}")
    
    # 数据记录数
    cursor.execute("SELECT COUNT(*) FROM daily_data")
    record_count = cursor.fetchone()[0]
    print(f"  日线记录数: {record_count:,}")
    
    # 日期范围
    cursor.execute("SELECT MIN(date), MAX(date) FROM daily_data")
    min_date, max_date = cursor.fetchone()
    print(f"  日期范围: {min_date} 至 {max_date}")
    
    # 数据库大小
    cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
    db_size = cursor.fetchone()[0]
    print(f"  数据库大小: {db_size / 1024 / 1024:.2f} MB")
    
    conn.commit()
    conn.close()
    
    print("\n数据库优化完成！")


if __name__ == "__main__":
    try:
        optimize_database()
    except Exception as e:
        print(f"优化失败: {e}")
        import traceback
        traceback.print_exc()
