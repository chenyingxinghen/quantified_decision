"""
检查聚源特征库 (jydb_features.db) 中 jy_fin_* 基本面数据不连续的股票

纯聚源环境下，基本面特征由 LC_MainIndexNew 预处理为 daily_features 的
jy_fin_* 列。本脚本审计这些 PIT 财务序列的报告期连续性，找出存在数据缺口的
股票，帮助识别数据质量问题（替代原 Baostock stock_finance.db 审计）。
"""

import sqlite3
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.jydb_config import JYDB_FEATURE_DB_PATH
from core.data.jydb_feature_store import DEFAULT_TABLE_SPECS

# 聚源基本面表对应的特征列前缀（LC_MainIndexNew -> jy_fin_*）
FIN_TABLE = "LC_MainIndexNew"
FINANCE_PREFIX = DEFAULT_TABLE_SPECS[FIN_TABLE].prefix  # "jy_fin_"


def get_db_connection(db_path):
    """获取数据库连接"""
    conn = sqlite3.connect(db_path)
    return conn


def _fin_columns(conn):
    """返回 daily_features 中 jy_fin_* 列名列表。"""
    rows = conn.execute("PRAGMA table_info(daily_features)").fetchall()
    return [r[1] for r in rows if r[1].startswith(FINANCE_PREFIX)]


def check_table_continuity(conn, code: str, fin_cols: list) -> dict:
    """检查单只股票的 jy_fin_* 报告期连续性（按 available_date 升序）。"""
    cols = ", ".join(f'"{c}"' for c in fin_cols)
    query = (
        f'SELECT date, {cols} FROM daily_features '
        f'WHERE code = ? ORDER BY date ASC'
    )
    df = pd.read_sql_query(query, conn, params=(code,))
    if df.empty:
        return None

    df['date'] = pd.to_datetime(df['date'])
    dates = df['date'].tolist()
    actual_periods = len(df)

    gaps = []
    if len(dates) > 1:
        for i in range(1, len(dates)):
            months_diff = (
                (dates[i].year - dates[i - 1].year) * 12
                + (dates[i].month - dates[i - 1].month)
            )
            # 财报通常季度披露（间隔约 3 个月），间隔 > 5 个月视为缺口
            if months_diff > 5:
                gaps.append({
                    'gap_start': dates[i - 1].strftime('%Y-%m-%d'),
                    'gap_end': dates[i].strftime('%Y-%m-%d'),
                    'missing_periods': max(1, months_diff // 3 - 1),
                    'months_gap': months_diff,
                })

    first_date, last_date = dates[0], dates[-1]
    total_months = (last_date.year - first_date.year) * 12 + (last_date.month - first_date.month)
    expected_periods = max(1, total_months // 3) + 1
    completeness = (actual_periods / expected_periods * 100) if expected_periods > 0 else 100

    if len(gaps) > 0 or completeness < 80:
        return {
            'code': code,
            'total_periods': actual_periods,
            'expected_periods': expected_periods,
            'completeness_pct': round(completeness, 2),
            'gap_count': len(gaps),
            'gaps_detail': str(gaps) if gaps else '无重大缺口',
            'first_report_date': first_date.strftime('%Y-%m-%d'),
            'last_report_date': last_date.strftime('%Y-%m-%d'),
        }
    return None


def analyze_data_gaps(conn):
    """分析所有股票的 jy_fin_* 数据缺口。"""
    fin_cols = _fin_columns(conn)
    if not fin_cols:
        print(f"  daily_features 中无 {FINANCE_PREFIX}* 列，请先运行 build 流程")
        return pd.DataFrame()

    codes = pd.read_sql_query(
        "SELECT DISTINCT code FROM daily_features ORDER BY code", conn
    )['code'].tolist()
    print(f"  审计 {len(codes)} 只股票的 {FINANCE_PREFIX}* 基本面连续性...")

    results = []
    for code in codes:
        rec = check_table_continuity(conn, code, fin_cols)
        if rec:
            results.append(rec)
    return pd.DataFrame(results)


def summarize_by_stock(gap_df):
    """按股票汇总数据缺口信息。"""
    if gap_df.empty:
        return pd.DataFrame()

    summary = gap_df.groupby('code').agg({
        'total_periods': 'sum',
        'expected_periods': 'sum',
        'gap_count': 'sum',
        'completeness_pct': 'mean',
    }).reset_index()
    summary.columns = ['code', 'total_periods', 'expected_periods',
                       'total_gaps', 'avg_completeness']
    summary['avg_completeness'] = summary['avg_completeness'].round(2)

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
    print("聚源基本面数据连续性检查工具 (jy_fin_*)")
    print("=" * 80)

    if not os.path.exists(JYDB_FEATURE_DB_PATH):
        print(f"\n错误：特征库不存在：{JYDB_FEATURE_DB_PATH}")
        return

    print(f"\n特征库路径：{JYDB_FEATURE_DB_PATH}")
    print(f"审计表：{FIN_TABLE} ({FINANCE_PREFIX}*)")

    conn = get_db_connection(JYDB_FEATURE_DB_PATH)
    try:
        gap_df = analyze_data_gaps(conn)
        if gap_df.empty:
            print("\n✓ 未发现明显的数据不连续问题")
            return

        print(f"\n发现 {len(gap_df)} 条存在数据缺口的记录")

        summary_df = summarize_by_stock(gap_df)
        print("\n" + "=" * 80)
        print("数据不连续股票汇总表")
        print("=" * 80)

        severity_order = {'严重': 0, '中等': 1, '轻微': 2, '正常': 3}
        summary_df['severity_rank'] = summary_df['severity'].map(severity_order)
        summary_df = summary_df.sort_values(['severity_rank', 'avg_completeness'])

        top_n = min(20, len(summary_df))
        print(f"\n显示前 {top_n} 个问题最严重的股票:\n")

        display_df = summary_df[['code', 'severity', 'avg_completeness',
                                  'total_gaps']].head(top_n)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        print(display_df.to_string(index=False))

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'data_gap_check_{timestamp}.csv'
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n详细结果已保存到：{output_file}")

        print("\n" + "=" * 80)
        print("统计摘要")
        print("=" * 80)
        severity_counts = summary_df['severity'].value_counts()
        for severity, count in severity_counts.items():
            print(f"{severity}: {count} 只股票")
        print(f"\n平均数据完整率：{summary_df['avg_completeness'].mean():.2f}%")
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
