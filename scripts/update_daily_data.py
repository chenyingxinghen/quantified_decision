#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用混合数据源更新股票数据
支持akshare和yfinance自动切换
"""
import sys
import os


# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import DataFetcher
import config
from datetime import datetime
import argparse
import time
from tqdm import tqdm


def update_single_stock(symbol, incremental=True):
    """
    更新单只股票数据
    
    Args:
        symbol: 股票代码
        incremental: 是否增量更新
    """
    print(f"\n{'='*60}")
    print(f"开始更新股票 {symbol} 的数据")
    print(f"更新模式: {'增量' if incremental else '全量'}")
    print(f"{'='*60}\n")
    
    fetcher = DataFetcher()
    
    try:
        fetcher.update_daily_data(symbol, incremental=incremental)
    finally:
        fetcher.close()


def update_multiple_stocks(symbols, incremental=True):
    """
    批量更新多只股票数据
    
    Args:
        symbols: 股票代码列表
        incremental: 是否增量更新
    """
    print(f"\n{'='*60}")
    print(f"开始批量更新 {len(symbols)} 只股票的数据")
    print(f"更新模式: {'增量' if incremental else '全量'}")
    print(f"{'='*60}\n")
    
    fetcher = DataFetcher()

    success_count = 0
    fail_count = 0
    
    try:
        pbar = tqdm(symbols, desc="更新进度", unit="只")
        for symbol in pbar:
            try:
                fetcher.update_daily_data(symbol, incremental=incremental)
                time.sleep(config.QUEST_INTERVAL)
                success_count += 1
            except Exception as e:
                tqdm.write(f"✗ 更新{symbol}失败: {e}")
                fail_count += 1
        
        print(f"\n{'='*60}")
        print(f"批量更新完成:")
        print(f"  成功: {success_count} 只")
        print(f"  失败: {fail_count} 只")
        print(f"{'='*60}\n")
    finally:
        fetcher.close()


def update_all_stocks(markets=config.DEFAULT_MARKETS, incremental=True):
    """
    更新所有股票数据
    
    Args:
        markets: 市场列表
        incremental: 是否增量更新
    """
    print(f"\n{'='*60}")
    print(f"开始更新所有股票数据")
    print(f"市场: {', '.join(markets)}")
    print(f"更新模式: {'增量' if incremental else '全量'}")
    print(f"{'='*60}\n")

    fetcher = DataFetcher()
    
    try:
        # 获取股票列表
        print("正在获取股票列表...")
        stock_list = fetcher.get_stock_list(markets=markets)
        
        if stock_list.empty:
            print("未获取到股票列表")
            return
        
        print(f"获取到 {len(stock_list)} 只股票\n")
        
        # 批量更新
        symbols = stock_list['code'].tolist()
        update_multiple_stocks(symbols, incremental=incremental)
        
    finally:
        fetcher.close()


def main():
    """主函数"""
    # print(f"当前时间: {datetime.now()}")
    # if datetime.now().time() < datetime.strptime("15:30", "%H:%M").time():
    #     print("规定: 数据获取脚本只能在新一日当天 15:30 以后执行请求接口。")
    #     return

    parser = argparse.ArgumentParser(
        description='使用统一数据源更新股票数据（Yahoo K线 + 东财指标）'
    )
    
    parser.add_argument(
        '--mode',
        choices=['single', 'multiple', 'all'],
        default='all',
        help='运行模式: single(单只股票), multiple(多只股票), all(所有股票)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        help='股票代码（mode=single时使用）'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='股票代码列表（mode=multiple时使用）'
    )
    
    parser.add_argument(
        '--markets',
        type=str,
        nargs='+',
        default=config.DEFAULT_MARKETS,
        help='市场列表（mode=all时使用），可选: sh, sz_main, sz_gem, bj'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='使用全量更新模式（默认为增量更新）'
    )
    
    args = parser.parse_args()
    
    # 默认增量更新，除非指定了 full，或者 config.INCREMENTAL_UPDATE 被设为 False 且没指定增量模式
    incremental = config.INCREMENTAL_UPDATE if not args.full else False
    
    if args.mode == 'single':
        if not args.symbol:
            print("错误: 单只股票模式需要指定 --symbol 参数")
            return
        update_single_stock(args.symbol, incremental=incremental)
    
    elif args.mode == 'multiple':
        if not args.symbols:
            print("错误: 多只股票模式需要指定 --symbols 参数")
            return
        update_multiple_stocks(args.symbols, incremental=incremental)
    
    elif args.mode == 'all':
        update_all_stocks(markets=args.markets, incremental=incremental)


if __name__ == "__main__":
    main()
