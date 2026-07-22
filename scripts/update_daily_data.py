#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票数据增量更新脚本 (纯聚源 / JYDB 版)

流程：
  1) pull_jydb_parallel.py  聚源 SQL Server → 本地 jydb_raw.db（bronze）
  2) build_intermediate_from_raw.py  jydb_raw.db → jydb_features.db + stock_daily.db（silver）
本脚本可选地串联这两步，并提供单步入口。不再依赖 Baostock。
"""
import sys
import os
import argparse
import subprocess

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _run(cmd: list) -> int:
    print(">>> " + " ".join(cmd))
    return subprocess.call([sys.executable, *cmd])


def update_all(incremental: bool, workers: int, start: str, end: str):
    """全市场更新：先拉取聚源原始数据，再构建中间产物。"""
    print(f"\n=== 开始同步全市场数据 (源: 聚源 JYDB) | 模式: {'增量' if incremental else '全量'} ===")

    # 第一步：拉取聚源原始数据到本地 raw 库
    print("\n--- 第一步: 拉取聚源原始数据 ---")
    pull_args = [
        os.path.join("scripts", "pull_jydb_parallel.py"),
        "--start", start, "--end", end, "--workers", str(workers),
    ]
    if _run(pull_args) != 0:
        print("[警告] 聚源原始数据拉取失败，请检查网络/JYDB 连接")

    # 第二步：构建中间产物（特征库 + 行情库）
    print("\n--- 第二步: 构建中间产物（特征 + 行情）---")
    build_args = [
        os.path.join("scripts", "build_intermediate_from_raw.py"),
        "--mode", "both", "--start", start, "--end", end, "--workers", str(workers),
    ]
    if _run(build_args) != 0:
        print("[警告] 中间产物构建失败")

    # 第三步：更新因子缓存
    print("\n--- 第三步: 更新选股因子缓存 ---")
    try:
        from scripts.select_stocks import _update_factor_cache_incremental, get_all_stock_codes
        from config import DATABASE_PATH
        from config.factor_config import TrainingConfig
        cache_dir = os.path.join(PROJECT_ROOT, TrainingConfig.CACHE_DIR)
        all_codes = get_all_stock_codes(DATABASE_PATH)
        _update_factor_cache_incremental(
            db_path=DATABASE_PATH,
            codes=all_codes,
            cache_dir=cache_dir,
            workers=workers,
        )
        print("✓ 因子缓存更新完成")
    except Exception as e:
        import traceback
        print(f"[警告] 因子缓存更新失败，不影响行情/特征数据: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='股票数据增量更新脚本 (纯聚源)')
    parser.add_argument('--full', action='store_true', help='全量更新（从最早可用日期开始）')
    parser.add_argument('--workers', type=int, default=8, help='并行进程数 (默认: 8)')
    parser.add_argument('--start', type=str, default='2010-01-01', help='起始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='结束日期 (YYYY-MM-DD, 默认今天)')
    parser.add_argument('--pull-only', action='store_true', help='仅拉取聚源原始数据')
    parser.add_argument('--build-only', action='store_true', help='仅构建中间产物（假定 raw 库已就绪）')

    args = parser.parse_args()
    incremental = not args.full
    end = args.end or __import__("datetime").datetime.now().strftime('%Y-%m-%d')

    if args.pull_only:
        _run([os.path.join("scripts", "pull_jydb_parallel.py"),
              "--start", args.start, "--end", end, "--workers", str(args.workers)])
    elif args.build_only:
        _run([os.path.join("scripts", "build_intermediate_from_raw.py"),
              "--mode", "both", "--start", args.start, "--end", end,
              "--workers", str(args.workers)])
    else:
        update_all(incremental, args.workers, args.start, end)


if __name__ == "__main__":
    main()
