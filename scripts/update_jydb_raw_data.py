"""拉取模型训练所需的聚源原始数据投影。"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.jydb_config import JYDB_RAW_DB_PATH
from core.data.jydb_raw_etl import JYDBRawETL, JYDBRawStore, training_raw_specs


def main():
    parser = argparse.ArgumentParser(description="拉取训练所需聚源原始数据")
    parser.add_argument("--start", required=True, help="日频训练数据起始日")
    parser.add_argument("--end", required=True, help="数据截止日")
    parser.add_argument(
        "--bootstrap-start", default="1990-01-01",
        help="PIT/状态表历史起点，用于训练起始日状态回溯",
    )
    parser.add_argument("--batch-months", type=int, default=3, help="分批月数")
    parser.add_argument("--output", default=JYDB_RAW_DB_PATH, help="原始仓库路径")
    parser.add_argument(
        "--tables", nargs="*", choices=sorted(training_raw_specs()),
        help="仅抽取指定源表；默认抽取全部训练依赖表",
    )
    args = parser.parse_args()
    if args.batch_months <= 0:
        parser.error("--batch-months 必须为正整数")

    store = JYDBRawStore(args.output)

    def progress(table, batch_start, batch_end, count):
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(
            f"[原始] {table} {batch_start}..{batch_end}: "
            f"{count:,} 行，仓库 {size_mb:,.1f} MB",
            flush=True,
        )

    result = JYDBRawETL(store).run(
        args.start,
        args.end,
        bootstrap_start=args.bootstrap_start,
        tables=args.tables,
        batch_months=args.batch_months,
        progress=progress,
    )
    print(f"聚源原始仓库: {os.path.abspath(args.output)}")
    for table, count in result.items():
        print(f"  {table}: 本次读取 {count:,} 行")


if __name__ == "__main__":
    main()
