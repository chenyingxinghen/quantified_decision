"""抽取聚源数据并更新本地 PIT 特征库。"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.jydb_config import JYDB_FEATURE_DB_PATH
from core.data.jydb_feature_store import DEFAULT_TABLE_SPECS, JYDBETL, JYDBFeatureStore
from core.data.jydb_market_etl import JYDBMarketETL
from config.baostock_config import DATABASE_PATH


def main():
    parser = argparse.ArgumentParser(description="更新聚源 PIT 特征库")
    parser.add_argument("--start", help="起始日期 YYYY-MM-DD；增量模式下为首次回溯起点")
    parser.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--incremental", action="store_true", help="按表水位增量更新")
    parser.add_argument("--overlap-days", type=int, default=5, help="增量更新回看天数")
    parser.add_argument(
        "--tables", nargs="*", choices=sorted(DEFAULT_TABLE_SPECS),
        help="指定抽取表；默认抽取全部已配置表",
    )
    parser.add_argument("--output", default=JYDB_FEATURE_DB_PATH, help="本地特征库路径")
    parser.add_argument("--feature-stocks", nargs="*", help="仅抽取指定股票的特征")
    parser.add_argument("--with-market", action="store_true", help="同时更新 daily_data 和复权因子")
    parser.add_argument(
        "--market-only", action="store_true",
        help="只更新 daily_data、复权因子和 ST 状态，不抽取特征表",
    )
    parser.add_argument("--market-output", default=DATABASE_PATH, help="行情 SQLite 输出路径")
    parser.add_argument("--market-stocks", nargs="*", help="仅抽取指定股票代码")
    parser.add_argument(
        "--batch-months", type=int, default=12,
        help="全量模式按多少个月分批提交（默认 12；设为 0 使用单次长事务）",
    )
    args = parser.parse_args()

    if args.batch_months < 0:
        parser.error("--batch-months 不能为负数")
    if not args.incremental and not args.start:
        parser.error("非增量模式必须提供 --start")

    if not args.market_only:
        store = JYDBFeatureStore(args.output)
        etl = JYDBETL(store)
        if args.incremental:
            result = etl.run_incremental(
                args.end, args.tables,
                fallback_start=args.start or "2000-01-01",
                overlap_days=args.overlap_days,
                stock_codes=args.feature_stocks,
            )
        elif args.batch_months:
            def feature_progress(table, batch_start, batch_end, count):
                size_mb = os.path.getsize(args.output) / 1024 / 1024
                print(
                    f"[特征] {table} {batch_start}..{batch_end}: "
                    f"{count:,} 个非空值，库大小 {size_mb:,.1f} MB",
                    flush=True,
                )

            result = etl.run_batched(
                args.start, args.end, args.tables,
                stock_codes=args.feature_stocks,
                batch_months=args.batch_months,
                progress=feature_progress,
            )
        else:
            result = etl.run(
                args.start, args.end, args.tables, stock_codes=args.feature_stocks
            )
        print(f"聚源 PIT 特征库: {os.path.abspath(args.output)}")
        for table, count in result.items():
            print(f"  {table}: {count:,} 个非空特征值")
    if args.with_market or args.market_only:
        market_start = args.start
        if not market_start:
            parser.error("--with-market 必须提供 --start")
        market_etl = JYDBMarketETL(args.market_output)
        if args.batch_months:
            def market_progress(table, batch_start, batch_end, count):
                size_mb = os.path.getsize(args.market_output) / 1024 / 1024
                print(
                    f"[行情] {table} {batch_start}..{batch_end}: "
                    f"{count:,} 行，库大小 {size_mb:,.1f} MB",
                    flush=True,
                )

            market_result = market_etl.run_batched(
                market_start, args.end, stock_codes=args.market_stocks,
                batch_months=args.batch_months,
                progress=market_progress,
            )
        else:
            market_result = market_etl.run(
                market_start, args.end, stock_codes=args.market_stocks
            )
        print(f"聚源行情库: {os.path.abspath(args.market_output)}")
        for table, count in market_result.items():
            print(f"  {table}: {count:,} 行")


if __name__ == "__main__":
    main()
