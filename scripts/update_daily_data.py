#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票数据增量更新脚本 (Baostock 版)
"""
import sys
import os
import argparse
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.data.baostock_main import BaostockDataManager
import config

def update_single_stock(symbol, incremental=False, start_date=None, end_date=None):
    """更新单只股票数据"""
    print(f"\n开始更新股票 {symbol} | 模式: {'增量' if incremental else '全量'}")
    manager = BaostockDataManager()
    try:
        # 单只更新时仍保持原有逻辑顺序
        manager.update_stock_data(symbol, incremental=incremental, start_date=start_date, end_date=end_date)
        manager.update_finance_data(symbol, incremental=incremental)
    finally:
        manager.close()
        manager.logout()

def update_multiple_stocks(symbols, incremental=True, workers=None, start_date=None, end_date=None):
    """批量更新多只股票数据"""
    if workers is None:
        workers = getattr(config, 'WORKERS_NUM', 5)
        
    print(f"\n正在批量更新 {len(symbols)} 只股票 (并发数: {workers}) | 模式: {'增量' if incremental else '全量'}")
    manager = BaostockDataManager()
    try:
        # 核心逻辑：先同步所有股票的日频数据
        print("\n--- 第一步: 同步日频行情数据 ---")
        manager.update_specific_stocks(symbols, incremental=incremental, workers=workers, mode='daily', start_date=start_date, end_date=end_date)
        
        # 再同步所有股票的财务数据
        print("\n--- 第二步: 同步财务基本面数据 ---")
        manager.update_specific_stocks(symbols, incremental=incremental, workers=workers, mode='finance')
    finally:
        manager.close()

def update_all_stocks(incremental=True, workers=None, start_date=None, end_date=None):
    """更新所有股票数据"""
    if workers is None:
        workers = getattr(config, 'WORKERS_NUM', 5)
    
    print(f"\n=== 开始同步全市场数据 (源: Baostock) | 模式: {'增量' if incremental else '全量'} ===")
    manager = BaostockDataManager()
    try:
        # 第一步：获取日频数据
        print("\n--- 第一步: 同步全市场日频数据 ---")
        manager.init_all_stocks(incremental=incremental, workers=workers, mode='daily', start_date=start_date, end_date=end_date)
        
        # 第二步：获取财务数据
        print("\n--- 第二步: 同步全市场财务数据 ---")
        manager.init_all_stocks(incremental=incremental, workers=workers, mode='finance')
    finally:
        manager.close()

def main():
    parser = argparse.ArgumentParser(description='股票数据增量更新脚本')
    parser.add_argument('--mode', choices=['single', 'multiple', 'all'], default='all')
    parser.add_argument('--symbol', type=str, help='股票代码 (mode=single)')
    parser.add_argument('--symbols', type=str, nargs='+', help='代码列表 (mode=multiple)')
    parser.add_argument('--full', action='store_true', help='全量更新 (从 10 年前开始)')
    parser.add_argument('--workers', type=int, default=None, help=f'并发线程数 (默认: {config.WORKERS_NUM})')
    parser.add_argument('--start', type=str, default='2010-01-01', help='起始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2030-01-01', help='结束日期 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    incremental = not args.full
    
    if args.mode == 'single' and args.symbol:
        update_single_stock(args.symbol, incremental=incremental, start_date=args.start, end_date=args.end)
    elif args.mode == 'multiple' and args.symbols:
        update_multiple_stocks(args.symbols, incremental=incremental, workers=args.workers, start_date=args.start, end_date=args.end)
    else:
        update_all_stocks(incremental=incremental, workers=args.workers, start_date=args.start, end_date=args.end)

if __name__ == "__main__":
    main()
