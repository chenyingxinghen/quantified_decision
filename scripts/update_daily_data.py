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

from core.data.hybrid_fetcher import HybridDataFetcher
import config
from datetime import datetime
import argparse


def update_single_stock(symbol, prefer_source='yfinance', incremental=True):
    """
    更新单只股票数据
    
    Args:
        symbol: 股票代码
        prefer_source: 优先数据源，'akshare' 或 'yfinance'
        incremental: 是否增量更新
    """
    print(f"\n{'='*60}")
    print(f"开始更新股票 {symbol} 的数据")
    print(f"优先数据源: {prefer_source}")
    print(f"更新模式: {'增量' if incremental else '全量'}")
    print(f"{'='*60}\n")
    
    fetcher = HybridDataFetcher(
        use_proxy=config.USE_PROXY,
        prefer_source=prefer_source
    )
    
    try:
        fetcher.update_daily_data(symbol, incremental=incremental)
        fetcher.print_stats()
    finally:
        fetcher.close()


def update_multiple_stocks(symbols, prefer_source='yfinance', incremental=True):
    """
    批量更新多只股票数据
    
    Args:
        symbols: 股票代码列表
        prefer_source: 优先数据源
        incremental: 是否增量更新
    """
    print(f"\n{'='*60}")
    print(f"开始批量更新 {len(symbols)} 只股票的数据")
    print(f"优先数据源: {prefer_source}")
    print(f"更新模式: {'增量' if incremental else '全量'}")
    print(f"{'='*60}\n")
    
    fetcher = HybridDataFetcher(
        use_proxy=config.USE_PROXY,
        prefer_source=prefer_source
    )
    
    success_count = 0
    fail_count = 0
    
    try:
        for i, symbol in enumerate(symbols, 1):
            # print(f"\n[{i}/{len(symbols)}] 正在更新 {symbol}...")
            if i % 100 == 0:
                print(f"\n[{i}/{len(symbols)}]")
            try:
                fetcher.update_daily_data(symbol, incremental=incremental)
                time.sleep(0.2)
                success_count += 1
            except Exception as e:
                print(f"✗ 更新{symbol}失败: {e}")
                fail_count += 1
        
        print(f"\n{'='*60}")
        print(f"批量更新完成:")
        print(f"  成功: {success_count} 只")
        print(f"  失败: {fail_count} 只")
        print(f"{'='*60}\n")
        
        fetcher.print_stats()
    finally:
        fetcher.close()


def update_all_stocks(markets=['sh', 'sz_main'], prefer_source='yfinance', incremental=True):
    """
    更新所有股票数据
    
    Args:
        markets: 市场列表
        prefer_source: 优先数据源
        incremental: 是否增量更新
    """
    print(f"\n{'='*60}")
    print(f"开始更新所有股票数据")
    print(f"市场: {', '.join(markets)}")
    print(f"优先数据源: {prefer_source}")
    print(f"更新模式: {'增量' if incremental else '全量'}")
    print(f"{'='*60}\n")
    
    fetcher = HybridDataFetcher(
        use_proxy=config.USE_PROXY,
        prefer_source=prefer_source
    )
    
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
        update_multiple_stocks(symbols, prefer_source=prefer_source, incremental=incremental)
        
    finally:
        fetcher.close()


def test_data_sources():
    """
    测试两个数据源的可用性
    """
    print(f"\n{'='*60}")
    print("测试数据源可用性")
    print(f"{'='*60}\n")
    
    test_symbol = "600000"  # 浦发银行
    test_start = "20240101"
    test_end = "20240131"
    
    # 测试akshare
    print("1. 测试akshare数据源...")
    fetcher_ak = HybridDataFetcher(prefer_source='akshare')
    try:
        data_ak = fetcher_ak.get_historical_data(
            test_symbol, 
            start_date=test_start, 
            end_date=test_end,
            fallback=False
        )
        if not data_ak.empty:
            print(f"   ✓ akshare可用，获取到{len(data_ak)}条数据")
        else:
            print(f"   ✗ akshare不可用或无数据")
    except Exception as e:
        print(f"   ✗ akshare测试失败: {e}")
    finally:
        fetcher_ak.close()
    
    # 测试yfinance
    print("\n2. 测试yfinance数据源...")
    fetcher_yf = HybridDataFetcher(prefer_source='yfinance')
    try:
        data_yf = fetcher_yf.get_historical_data(
            test_symbol, 
            start_date=test_start, 
            end_date=test_end,
            fallback=False
        )
        if not data_yf.empty:
            print(f"   ✓ yfinance可用，获取到{len(data_yf)}条数据")
        else:
            print(f"   ✗ yfinance不可用或无数据")
    except Exception as e:
        print(f"   ✗ yfinance测试失败: {e}")
    finally:
        fetcher_yf.close()
    
    print(f"\n{'='*60}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='使用混合数据源更新股票数据（支持akshare和yfinance自动切换）'
    )
    
    parser.add_argument(
        '--mode',
        choices=['single', 'multiple', 'all', 'test'],
        default='all',
        help='运行模式: single(单只股票), multiple(多只股票), all(所有股票), test(测试数据源)'
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
        default=['sh', 'sz_main'],
        help='市场列表（mode=all时使用），可选: sh, sz_main, sz_gem, bj'
    )
    
    parser.add_argument(
        '--source',
        choices=['akshare', 'yfinance'],
        default='yfinance',
        help='优先使用的数据源'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='使用全量更新模式（默认为增量更新）'
    )
    
    args = parser.parse_args()
    
    incremental = not args.full
    
    if args.mode == 'test':
        test_data_sources()
    
    elif args.mode == 'single':
        if not args.symbol:
            print("错误: 单只股票模式需要指定 --symbol 参数")
            return
        update_single_stock(args.symbol, prefer_source=args.source, incremental=incremental)
    
    elif args.mode == 'multiple':
        if not args.symbols:
            print("错误: 多只股票模式需要指定 --symbols 参数")
            return
        update_multiple_stocks(args.symbols, prefer_source=args.source, incremental=incremental)
    
    elif args.mode == 'all':
        update_all_stocks(markets=args.markets, prefer_source=args.source, incremental=incremental)


if __name__ == "__main__":
    main()
