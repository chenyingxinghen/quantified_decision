"""
测试动态前复权功能

验证：
1. 不同基准日的复权价格应该不同
2. 复权后的收益率应该一致
3. 不存在未来函数
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data.baostock_main import BaostockDataManager
from core.backtest.baostock_data_handler import BaostockDataHandler


def test_dynamic_adjustment():
    """测试动态前复权"""
    print("=" * 60)
    print("测试动态前复权功能")
    print("=" * 60)
    
    manager = BaostockDataManager()
    
    # 测试股票
    test_code = '000001'
    
    # 获取不同基准日的复权数据
    print(f"\n测试股票: {test_code}")
    print("-" * 60)
    
    # 基准日1: 2023-01-01
    df1 = manager.get_adjusted_kline(
        code=test_code,
        start_date='2022-01-01',
        end_date='2023-12-31',
        adjust_date='2023-01-01'
    )
    
    # 基准日2: 2023-12-31
    df2 = manager.get_adjusted_kline(
        code=test_code,
        start_date='2022-01-01',
        end_date='2023-12-31',
        adjust_date='2023-12-31'
    )
    
    if df1.empty or df2.empty:
        print(f"警告: {test_code} 数据为空，请先运行数据初始化")
        manager.close()
        return
    
    # 比较同一天的复权价格
    test_date = '2022-06-01'
    
    price1 = df1[df1['date'] == test_date]['adj_close'].values
    price2 = df2[df2['date'] == test_date]['adj_close'].values
    
    if len(price1) > 0 and len(price2) > 0:
        print(f"\n{test_date} 的复权价格:")
        print(f"  以 2023-01-01 为基准: {price1[0]:.4f}")
        print(f"  以 2023-12-31 为基准: {price2[0]:.4f}")
        print(f"  差异: {abs(price1[0] - price2[0]):.4f}")
        
        if abs(price1[0] - price2[0]) > 0.01:
            print("  ✓ 不同基准日的复权价格不同（正常）")
        else:
            print("  ✗ 复权价格相同（可能有问题）")
    
    # 验证收益率一致性
    print("\n验证收益率一致性:")
    print("-" * 60)
    
    df1['return'] = df1['adj_close'].pct_change()
    df2['return'] = df2['adj_close'].pct_change()
    
    # 合并数据比较
    merged = pd.merge(
        df1[['date', 'return']],
        df2[['date', 'return']],
        on='date',
        suffixes=('_base1', '_base2')
    )
    
    merged['return_diff'] = abs(merged['return_base1'] - merged['return_base2'])
    max_diff = merged['return_diff'].max()
    
    print(f"收益率最大差异: {max_diff:.6f}")
    if max_diff < 1e-6:
        print("✓ 收益率一致（正常）")
    else:
        print("✗ 收益率不一致（可能有问题）")
    
    manager.close()


def test_backtest_no_future_function():
    """测试回测中不存在未来函数"""
    print("\n" + "=" * 60)
    print("测试回测中的动态复权（无未来函数）")
    print("=" * 60)
    
    handler = BaostockDataHandler('database/stock_daily.db')
    
    # 加载数据
    data = handler.load_data(
        start_date='2023-01-01',
        end_date='2023-12-31',
        parallel=False
    )
    
    if not data:
        print("警告: 没有加载到数据，请先运行数据初始化")
        handler.close()
        return
    
    # 选择一只测试股票
    test_code = list(data.keys())[0]
    print(f"\n测试股票: {test_code}")
    print("-" * 60)
    
    # 模拟回测：在不同日期获取数据
    test_dates = ['2023-03-01', '2023-06-01', '2023-09-01', '2023-12-01']
    
    results = []
    for date in test_dates:
        # 获取截止到该日期的数据（以该日期为基准复权）
        hist_data = handler.get_historical_data(
            test_code,
            end_date=date,
            lookback_days=60,
            adjust_to_date=True
        )
        
        if hist_data is not None and not hist_data.empty:
            # 获取某个历史日期的复权价格
            ref_date = '2023-02-01'
            ref_price = hist_data[hist_data['date'] == ref_date]['adj_close'].values
            
            if len(ref_price) > 0:
                results.append({
                    'base_date': date,
                    'ref_date': ref_date,
                    'adj_close': ref_price[0]
                })
    
    # 显示结果
    print(f"\n{test_code} 在 2023-02-01 的复权价格（不同基准日）:")
    print("-" * 60)
    for r in results:
        print(f"  基准日 {r['base_date']}: {r['adj_close']:.4f}")
    
    # 验证价格应该不同
    if len(results) > 1:
        prices = [r['adj_close'] for r in results]
        if len(set(prices)) > 1:
            print("\n✓ 不同基准日看到的历史价格不同（正确，无未来函数）")
        else:
            print("\n✗ 所有基准日看到的历史价格相同（可能有问题）")
    
    handler.close()


def test_finance_data():
    """测试财务数据获取"""
    print("\n" + "=" * 60)
    print("测试财务数据")
    print("=" * 60)
    
    manager = BaostockDataManager()
    
    # 查询财务数据
    query = '''
        SELECT * FROM finance.profit_ability 
        WHERE code = '000001' 
        ORDER BY pub_date DESC 
        LIMIT 5
    '''
    
    try:
        df = pd.read_sql_query(query, manager.conn)
        
        if not df.empty:
            print("\n盈利能力数据（最近5条）:")
            print("-" * 60)
            print(df[['code', 'pub_date', 'stat_date', 'roeAvg', 'npMargin', 'gpMargin']].to_string(index=False))
            print("\n✓ 财务数据正常")
        else:
            print("\n警告: 没有财务数据，请运行财务数据更新")
    except Exception as e:
        print(f"\n✗ 财务数据查询失败: {e}")
    
    manager.close()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Baostock 动态前复权测试套件")
    print("=" * 60)
    
    try:
        # 测试1: 动态复权
        test_dynamic_adjustment()
        
        # 测试2: 回测无未来函数
        test_backtest_no_future_function()
        
        # 测试3: 财务数据
        test_finance_data()
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
