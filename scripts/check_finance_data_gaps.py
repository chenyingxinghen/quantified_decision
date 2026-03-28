"""
检查 stock_finance.db 中数据不连续的股票

该脚本用于分析财务数据库中各股票的数据连续性，
找出存在数据缺口的股票，帮助识别数据质量问题。
"""

import sqlite3
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.baostock_config import FINANCE_DB_PATH, FINANCE_TABLES


def get_db_connection(db_path):
    """获取数据库连接"""
    conn = sqlite3.connect(db_path)
    return conn


def check_table_continuity(conn, table_name):
    """
    检查单个表的数据连续性
    
    参数:
        conn: 数据库连接
        table_name: 表名
        
    返回:
        DataFrame: 包含数据不连续的股票信息
    """
    print(f"\n检查表：{table_name}")
    
    # 获取该表所有股票代码
    query = f"""
        SELECT DISTINCT code 
        FROM {table_name}
        ORDER BY code
    """
    stock_df = pd.read_sql_query(query, conn)
    
    if stock_df.empty:
        print(f"  表 {table_name} 为空")
        return pd.DataFrame()
    
    results = []
    
    for code in stock_df['code']:
        # 获取该股票的所有统计数据日期
        query = f"""
            SELECT stat_date, pub_date
            FROM {table_name}
            WHERE code = ?
            ORDER BY stat_date ASC
        """
        df = pd.read_sql_query(query, conn, params=(code,))
        
        if df.empty:
            continue
            
        # 转换为日期格式
        df['stat_date'] = pd.to_datetime(df['stat_date'])
        df['pub_date'] = pd.to_datetime(df['pub_date'])
        
        # 分析数据连续性
        gaps = []
        total_expected_periods = 0
        actual_periods = len(df)
        
        if len(df) > 1:
            # 计算相邻报告期之间的间隔
            dates = df['stat_date'].tolist()
            
            for i in range(1, len(dates)):
                prev_date = dates[i-1]
                curr_date = dates[i]
                
                # 计算月份差
                months_diff = (curr_date.year - prev_date.year) * 12 + (curr_date.month - prev_date.month)
                
                # 正常情况下应该是季度报告（间隔 3 个月）
                # 允许一定的误差范围（1-5 个月）
                if months_diff > 5:
                    # 发现数据缺口
                    gap_periods = months_diff // 3 - 1
                    gaps.append({
                        'gap_start': prev_date.strftime('%Y-%m-%d'),
                        'gap_end': curr_date.strftime('%Y-%m-%d'),
                        'missing_periods': max(1, gap_periods),
                        'months_gap': months_diff
                    })
        
        # 计算预期的报告期数量（基于 IPO 时间和当前时间）
        try:
            first_date = dates[0]
            last_date = dates[-1]
            total_months = (last_date.year - first_date.year) * 12 + (last_date.month - first_date.month)
            expected_periods = max(1, total_months // 3) + 1
            total_expected_periods = expected_periods
        except:
            total_expected_periods = actual_periods
        
        # 计算数据完整率
        completeness = (actual_periods / total_expected_periods * 100) if total_expected_periods > 0 else 100
        
        if len(gaps) > 0 or completeness < 80:
            results.append({
                'code': code,
                'table': table_name,
                'total_periods': actual_periods,
                'expected_periods': total_expected_periods,
                'completeness_pct': round(completeness, 2),
                'gap_count': len(gaps),
                'gaps_detail': str(gaps) if gaps else '无重大缺口',
                'first_report_date': df['stat_date'].min().strftime('%Y-%m-%d'),
                'last_report_date': df['stat_date'].max().strftime('%Y-%m-%d')
            })
    
    return pd.DataFrame(results)


def analyze_data_gaps(conn):
    """
    分析所有财务表的数据缺口
    
    参数:
        conn: 数据库连接
        
    返回:
        DataFrame: 汇总结果
    """
    all_results = []
    
    # 检查每个启用的财务表
    for table in FINANCE_TABLES:
        try:
            result_df = check_table_continuity(conn, table)
            if not result_df.empty:
                all_results.append(result_df)
        except Exception as e:
            print(f"检查表 {table} 时出错：{e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def summarize_by_stock(gap_df):
    """
    按股票汇总数据缺口信息
    
    参数:
        gap_df: 包含数据缺口的 DataFrame
        
    返回:
        DataFrame: 按股票汇总的结果
    """
    if gap_df.empty:
        return pd.DataFrame()
    
    summary = gap_df.groupby('code').agg({
        'table': lambda x: list(x),
        'total_periods': 'sum',
        'expected_periods': 'sum',
        'gap_count': 'sum',
        'completeness_pct': 'mean'
    }).reset_index()
    
    summary.columns = ['code', 'affected_tables', 'total_periods', 
                       'expected_periods', 'total_gaps', 'avg_completeness']
    
    # 计算综合完整率
    summary['avg_completeness'] = summary['avg_completeness'].round(2)
    
    # 添加问题严重程度评级
    def rate_severity(row):
        if row['total_gaps'] == 0 and row['avg_completeness'] >= 90:
            return '正常'
        elif row['total_gaps'] <= 2 and row['avg_completeness'] >= 70:
            return '轻微'
        elif row['total_gaps'] <= 5 and row['avg_completeness'] >= 50:
            return '中等'
        else:
            return '严重'
    
    summary['severity'] = summary.apply(rate_severity, axis=1)
    
    return summary.sort_values('avg_completeness', ascending=True)


def main():
    """主函数"""
    print("=" * 80)
    print("股票财务数据连续性检查工具")
    print("=" * 80)
    
    # 检查数据库文件是否存在
    if not os.path.exists(FINANCE_DB_PATH):
        print(f"\n错误：数据库文件不存在：{FINANCE_DB_PATH}")
        return
    
    print(f"\n数据库路径：{FINANCE_DB_PATH}")
    print(f"检查的表：{', '.join(FINANCE_TABLES)}")
    
    # 连接数据库
    conn = get_db_connection(FINANCE_DB_PATH)
    
    try:
        # 分析数据缺口
        gap_df = analyze_data_gaps(conn)
        
        if gap_df.empty:
            print("\n✓ 未发现明显的数据不连续问题")
            return
        
        print(f"\n发现 {len(gap_df)} 条存在数据缺口的记录")
        
        # 按股票汇总
        summary_df = summarize_by_stock(gap_df)
        
        # 显示汇总结果
        print("\n" + "=" * 80)
        print("数据不连续股票汇总表")
        print("=" * 80)
        
        # 按严重程度排序显示
        severity_order = {'严重': 0, '中等': 1, '轻微': 2, '正常': 3}
        summary_df['severity_rank'] = summary_df['severity'].map(severity_order)
        summary_df = summary_df.sort_values(['severity_rank', 'avg_completeness'])
        
        # 显示前 20 个最严重的
        top_n = min(20, len(summary_df))
        print(f"\n显示前 {top_n} 个问题最严重的股票:\n")
        
        display_df = summary_df[['code', 'severity', 'avg_completeness', 
                                  'total_gaps', 'affected_tables']].head(top_n)
        
        # 格式化显示
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        print(display_df.to_string(index=False))
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'data_gap_check_{timestamp}.csv'
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n详细结果已保存到：{output_file}")
        
        # 统计信息
        print("\n" + "=" * 80)
        print("统计摘要")
        print("=" * 80)
        severity_counts = summary_df['severity'].value_counts()
        for severity, count in severity_counts.items():
            print(f"{severity}: {count} 只股票")
        
        avg_completeness_all = summary_df['avg_completeness'].mean()
        print(f"\n平均数据完整率：{avg_completeness_all:.2f}%")
        
    except Exception as e:
        print(f"\n分析过程中出现错误：{e}")
        import traceback
        traceback.print_exc()
    
    finally:
        conn.close()
        print("\n" + "=" * 80)
        print("检查完成")
        print("=" * 80)


if __name__ == '__main__':
    main()
