"""修复 pull_jydb_parallel 的 chunk-overwrite 缺陷后，清理并重新拉取受影响的数据。

先用该脚本清除损坏的 checkpoints 和旧数据，
再运行 pull_jydb_parallel.py 重新拉取。
"""
from __future__ import annotations

import os
import sqlite3
import sys
from contextlib import closing
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.jydb_config import JYDB_RAW_DB_PATH
from core.data.jydb_raw_etl import training_raw_specs
from core.data.jydb_feature_store import iter_date_batches

# 被 chunk-overwrite 缺陷影响的表 (max batch row_count = 50000 = chunksize)
AFFECTED_TABLES = [
    "QT_DailyQuote",
    "LC_DIndicesForValuation",
    "LC_SHSZHSCHoldings",
    "MT_TradingDetail",
    "QT_TradingCapitalFlow",
    "LC_CSIIndustry",
]

# 科创板/北交所等可能从未被拉取的股票前缀
MISSING_PREFIX_TABLES = [
    "QT_DailyQuote",
    "LC_STIBDailyQuote",
]


def fix_data():
    print("=" * 80)
    print(" jydb_raw.db 数据修复工具")
    print("=" * 80)
    print()
    print(" 该脚本将：")
    print("  1. 删除受影响表的 pull_checkpoint（强制重新拉取）")
    print("  2. 删除受影响表的损坏数据")
    print("  3. 重新拉取干净数据")
    print()

    db_path = JYDB_RAW_DB_PATH
    size_mb = os.path.getsize(db_path) / 1024 / 1024
    print(f" 数据库: {db_path}")
    print(f" 当前大小: {size_mb:.1f} MB")
    print()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    # 步骤 1: 统计受影响表
    print("--- 步骤 1: 确认受影响表 ---")
    for table in AFFECTED_TABLES:
        # 获取当前 checkpoint 信息
        pc = conn.execute(
            "SELECT COUNT(*), SUM(row_count) FROM pull_checkpoint WHERE source_table=?",
            (table,),
        ).fetchone()
        total_batches = pc[0]
        total_rows_in_checkpoint = pc[1] or 0

        actual_rows = conn.execute(
            'SELECT COUNT(*) FROM "%s"' % table
        ).fetchone()[0]

        status = "AFFECTED" if total_rows_in_checkpoint >= 50000 else "ok"
        print(f"  {table}: checkpoint={total_rows_in_checkpoint:,}, actual={actual_rows:,} => {status}")

    # 步骤 2: 删除受损 checkpoints
    print("\n--- 步骤 2: 删除受损 checkpoints ---")
    for table in AFFECTED_TABLES:
        deleted = conn.execute(
            "DELETE FROM pull_checkpoint WHERE source_table=? AND row_count > 0", (table,)
        ).rowcount
        # 同时删除 row_count=0 的无效 checkpoint
        zero_deleted = conn.execute(
            "DELETE FROM pull_checkpoint WHERE source_table=? AND row_count = 0", (table,)
        ).rowcount
        print(f"  {table}: 删除 {deleted} 个有效 + {zero_deleted} 个零行 checkpoint")
    conn.commit()
    print("  已提交。")

    # 步骤 3: 删除受损数据
    print("\n--- 步骤 3: 删除受损数据 ---")
    specs = training_raw_specs()
    total_deleted = 0
    for table in AFFECTED_TABLES:
        spec = specs.get(table)
        if not spec:
            print(f"  {table}: 未找到 spec，跳过")
            continue
        # 删除该表所有数据（会被重新拉取覆盖）
        deleted = conn.execute('DELETE FROM "%s"' % table).rowcount
        total_deleted += deleted
        print(f"  {table}: 删除 {deleted:,} 行")
    conn.commit()
    print(f"\n  共删除 {total_deleted:,} 行损坏数据。")

    # 步骤 4: 更新 manifest
    print("\n--- 步骤 4: 重置 raw_etl_manifest ---")
    for table in AFFECTED_TABLES:
        conn.execute(
            "UPDATE raw_etl_manifest SET row_count=0, updated_at=? WHERE source_table=?",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), table),
        )
        print(f"  {table}: row_count 重置为 0")
    conn.commit()

    new_size = os.path.getsize(db_path) / 1024 / 1024
    print(f"\n 数据库大小: {size_mb:.1f} MB -> {new_size:.1f} MB")

    # 步骤 5: 重新拉取
    print("\n--- 步骤 5: 重新拉取数据 ---")
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    tables_str = " ".join(AFFECTED_TABLES)
    print(f"\n 执行命令:")
    print(f"  python scripts/pull_jydb_parallel.py \\")
    print(f"    --start {start_date} --end {end_date} \\")
    print(f"    --tables {tables_str}")
    print()

    # 步骤 6: 重新跑 ETL
    print("--- 步骤 6: 重新运行 ETL（生成 jydb_features.db） ---")
    print(f"  python scripts/update_jydb_data.py --start {start_date} --end {end_date} --incremental")

    conn.close()

    print("\n" + "=" * 80)
    print(" 修复准备完成。请运行步骤 5 中的 pull 命令。")
    print("=" * 80)


if __name__ == "__main__":
    fix_data()
