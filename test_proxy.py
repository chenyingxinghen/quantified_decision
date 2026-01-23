#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试代理IP功能"""

from data_fetcher import DataFetcher

def test_without_proxy():
    """测试不使用代理"""
    print("=" * 50)
    print("测试不使用代理")
    print("=" * 50)
    
    fetcher = DataFetcher(use_proxy=False)
    
    print("\n测试获取股票列表（不使用代理）...")
    try:
        stock_list = fetcher.get_stock_list(markets=['sh'])
        if not stock_list.empty:
            print(f"✓ 成功获取 {len(stock_list)} 只股票")
            print(f"  示例: {stock_list.head(3)[['code', 'name']].to_dict('records')}")
        else:
            print("✗ 获取股票列表为空")
    except Exception as e:
        print(f"✗ 获取股票列表失败: {e}")
    
    fetcher.close()

def test_with_proxy():
    """测试使用代理"""
    print("\n" + "=" * 50)
    print("测试使用代理")
    print("=" * 50)
    
    fetcher = DataFetcher(use_proxy=True)
    
    # 测试获取代理
    print("\n1. 测试获取代理IP...")
    proxy = fetcher.get_proxy(validate=True)
    if proxy:
        print(f"✓ 代理获取成功: {proxy}")
    else:
        print("✗ 代理获取失败或验证失败")
    
    # 测试获取股票列表（使用代理）
    print("\n2. 测试使用代理获取股票列表...")
    try:
        stock_list = fetcher.get_stock_list(markets=['sh'])
        if not stock_list.empty:
            print(f"✓ 成功获取 {len(stock_list)} 只股票")
            print(f"  示例: {stock_list.head(3)[['code', 'name']].to_dict('records')}")
        else:
            print("✗ 获取股票列表为空")
    except Exception as e:
        print(f"✗ 获取股票列表失败: {e}")
    
    fetcher.close()

if __name__ == "__main__":
    # 先测试不使用代理
    test_without_proxy()
    
    # 再测试使用代理
    test_with_proxy()
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

