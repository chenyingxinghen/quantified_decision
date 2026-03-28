"""
测试 Baostock 集成

验证：
1. 数据获取功能
2. 动态前复权功能
3. 财务数据获取
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data.data_fetcher import DataFetcher


def test_basic_fetch():
    """测试基本数据获取"""
    print("=" * 60)
    print("测试1: 基本数据获取")
    print("=" * 60)
    
    fetcher = DataFetcher()
    
    try:
        # 测试更新单只股票
        print("\n更新股票 000001...")
        fetcher.update_daily_data('000001', incremental=False)
        
        # 查询数据
        df = fetcher.get_stock_data('000001', 10)
        print(f"\n最近10天数据:")
        print(df[['code', 'date', 'close', 'volume']].to_string(index=False))
        
        if not df.empty:
            print("\n✓ 基本数据获取成功")
        else:
            print("\n✗ 数据为空")
    
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        fetcher.close()


def test_dynamic_adjustment():
    """测试动态前复权"""
    print("\n" + "=" * 60)
    print("测试2: 动态前复权")
    print("=" * 60)
    
    fetcher = DataFetcher()
    
    try:
        # 获取复权数据
        df1 = fetcher.get_adjusted_kline('000001', '2024-01-01', '2024-12-31', '2024-06-30')
        df2 = fetcher.get_adjusted_kline('000001', '2024-01-01', '2024-12-31', '2024-12-31')
        
        if not df1.empty and not df2.empty:
            # 比较同一天的复权价格
            test_date = '2024-03-01'
            price1 = df1[df1['date'] == test_date]['adj_close'].values
            price2 = df2[df2['date'] == test_date]['adj_close'].values
            
            if len(price1) > 0 and len(price2) > 0:
                print(f"\n{test_date} 的复权价格:")
                print(f"  以 2024-06-30 为基准: {price1[0]:.4f}")
                print(f"  以 2024-12-31 为基准: {price2[0]:.4f}")
                print(f"  差异: {abs(price1[0] - price2[0]):.4f}")
                
                if abs(price1[0] - price2[0]) > 0.01:
                    print("\n✓ 动态前复权功能正常")
                else:
                    print("\n⚠ 复权价格相同（可能该期间无除权除息）")
            else:
                print(f"\n⚠ {test_date} 数据不存在")
        else:
            print("\n✗ 数据为空")
    
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        fetcher.close()


def test_historical_data():
    """测试历史数据获取（带复权）"""
    print("\n" + "=" * 60)
    print("测试3: 历史数据获取")
    print("=" * 60)
    
    fetcher = DataFetcher()
    
    try:
        # 获取不复权数据
        df_no_adj = fetcher.get_historical_data('000001', '2024-01-01', '2024-01-31', adjust=False)
        print(f"\n不复权数据（前5行）:")
        print(df_no_adj[['日期', '开盘', '最高', '最低', '收盘']].head().to_string(index=False))
        
        # 获取复权数据
        df_adj = fetcher.get_historical_data('000001', '2024-01-01', '2024-01-31', adjust=True)
        if 'adj_收盘' in df_adj.columns:
            print(f"\n复权数据（前5行）:")
            print(df_adj[['日期', 'adj_开盘', 'adj_最高', 'adj_最低', 'adj_收盘']].head().to_string(index=False))
            print("\n✓ 历史数据获取成功")
        else:
            print("\n⚠ 复权数据列不存在（可能缺少复权因子）")
    
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        fetcher.close()


def test_finance_data():
    """测试财务数据"""
    print("\n" + "=" * 60)
    print("测试4: 财务数据")
    print("=" * 60)
    
    fetcher = DataFetcher()
    
    try:
        import pandas as pd
        
        # 查询盈利能力数据
        query = '''
            SELECT * FROM finance.profit_ability 
            WHERE code = '000001' 
            ORDER BY pub_date DESC 
            LIMIT 5
        '''
        
        df = pd.read_sql_query(query, fetcher.conn)
        
        if not df.empty:
            print("\n盈利能力数据（最近5条）:")
            print(df[['code', 'pub_date', 'stat_date', 'roeAvg', 'npMargin']].to_string(index=False))
            print("\n✓ 财务数据正常")
        else:
            print("\n⚠ 财务数据为空（可能需要先更新财务数据）")
    
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        fetcher.close()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Baostock 集成测试套件")
    print("=" * 60)
    
    # 测试1: 基本数据获取
    test_basic_fetch()
    
    # 测试2: 动态前复权
    test_dynamic_adjustment()
    
    # 测试3: 历史数据获取
    test_historical_data()
    
    # 测试4: 财务数据
    test_finance_data()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
