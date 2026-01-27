# 回测诊断脚本 - 检查为什么没有交易发生

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from backtest_engine import DataLoader, StrategyInterface
from config import DATABASE_PATH

def diagnose():
    """诊断回测问题"""
    print("=" * 80)
    print("回测诊断工具")
    print("=" * 80)
    
    # 设置测试时间范围（最近3个月）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"\n测试时间范围: {start_date} 至 {end_date}")
    
    # 加载数据
    print("\n1. 加载数据...")
    data_loader = DataLoader()
    all_stocks_data = data_loader.load_all_stocks_data(start_date, end_date)
    trading_dates = data_loader.get_trading_dates(start_date, end_date)
    
    print(f"   - 股票数量: {len(all_stocks_data)}")
    print(f"   - 交易日数量: {len(trading_dates)}")
    
    if len(all_stocks_data) == 0:
        print("\n❌ 错误：没有加载到任何股票数据！")
        print("   请检查数据库是否有数据")
        return
    
    # 测试策略筛选
    print("\n2. 测试策略筛选...")
    strategy = StrategyInterface()
    
    signal_count = 0
    filtered_reasons = {
        'no_uptrend': 0,
        'low_entry_quality': 0,
        'low_trend_strength': 0,
        'above_channel': 0,
        'broken_support': 0,
        'no_core_signals': 0,
        'low_confidence': 0,
        'bad_risk_reward': 0,
        'total_checked': 0
    }
    
    # 检查最近5个交易日
    test_dates = trading_dates[-5:] if len(trading_dates) >= 5 else trading_dates
    
    for test_date in test_dates:
        print(f"\n   检查日期: {test_date}")
        
        # 构建历史数据
        historical_data = {}
        for code, stock_data in all_stocks_data.items():
            date_mask = stock_data['date'] <= test_date
            if date_mask.any():
                hist_data = stock_data[date_mask]
                if len(hist_data) >= 60:
                    historical_data[code] = hist_data
        
        print(f"   - 可用股票: {len(historical_data)}")
        
        # 逐个检查股票
        for stock_code in list(historical_data.keys())[:20]:  # 只检查前20只
            filtered_reasons['total_checked'] += 1
            
            try:
                # 直接调用screen_stock来详细诊断
                result = strategy.smc_v2.screen_stock(stock_code)
                
                if result:
                    signal_count += 1
                    print(f"   ✓ {stock_code}: 信号={result['signal']}, 置信度={result['confidence']:.1f}")
                else:
                    # 尝试诊断为什么被过滤
                    stock_data = historical_data[stock_code]
                    
                    # 检查市场结构
                    market_structure = strategy.smc_v2._check_market_structure_strict(stock_data)
                    if not market_structure['is_uptrend']:
                        filtered_reasons['no_uptrend'] += 1
                        continue
                    
                    # 检查趋势线
                    trend_analysis = strategy.smc_v2.trend_analyzer.analyze(stock_data)
                    if trend_analysis['entry_quality'] < 60:
                        filtered_reasons['low_entry_quality'] += 1
                        continue
                    
                    if trend_analysis['trend_strength'] < 50:
                        filtered_reasons['low_trend_strength'] += 1
                        continue
                    
                    if trend_analysis['current_position'] == 'above_channel':
                        filtered_reasons['above_channel'] += 1
                        continue
                    
                    if trend_analysis['broken_support']:
                        filtered_reasons['broken_support'] += 1
                        continue
                    
                    # 检查核心信号
                    liquidity_grab = strategy.smc_v2._detect_liquidity_grab_strict(stock_data)
                    order_block = strategy.smc_v2._identify_order_block_strict(stock_data)
                    fvg = strategy.smc_v2._detect_fvg_strict(stock_data)
                    
                    if not (liquidity_grab['detected'] or order_block['found'] or fvg['detected']):
                        filtered_reasons['no_core_signals'] += 1
                        continue
                    
                    # 如果到这里还没信号，可能是置信度或风险收益比问题
                    filtered_reasons['low_confidence'] += 1
                    
            except Exception as e:
                print(f"   ✗ {stock_code}: 错误 - {e}")
    
    # 输出诊断结果
    print("\n" + "=" * 80)
    print("诊断结果汇总")
    print("=" * 80)
    print(f"\n总共检查: {filtered_reasons['total_checked']} 只股票")
    print(f"找到信号: {signal_count} 个")
    print(f"\n过滤原因统计:")
    print(f"  - 非上升趋势: {filtered_reasons['no_uptrend']}")
    print(f"  - 入场质量低: {filtered_reasons['low_entry_quality']}")
    print(f"  - 趋势强度低: {filtered_reasons['low_trend_strength']}")
    print(f"  - 在通道上方: {filtered_reasons['above_channel']}")
    print(f"  - 支撑跌破: {filtered_reasons['broken_support']}")
    print(f"  - 无核心信号: {filtered_reasons['no_core_signals']}")
    print(f"  - 置信度/风险收益比不足: {filtered_reasons['low_confidence']}")
    
    # 给出建议
    print("\n" + "=" * 80)
    print("优化建议")
    print("=" * 80)
    
    if filtered_reasons['no_uptrend'] > filtered_reasons['total_checked'] * 0.5:
        print("\n⚠️  超过50%的股票因'非上升趋势'被过滤")
        print("   建议：放宽趋势判断条件")
        print("   - 降低 TREND_STRENGTH_STRONG 阈值（当前70）")
        print("   - 调整 _check_market_structure_strict 中的趋势判断逻辑")
    
    if filtered_reasons['low_entry_quality'] > filtered_reasons['total_checked'] * 0.3:
        print("\n⚠️  超过30%的股票因'入场质量低'被过滤")
        print("   建议：降低 MIN_ENTRY_QUALITY（当前60）")
    
    if filtered_reasons['low_trend_strength'] > filtered_reasons['total_checked'] * 0.3:
        print("\n⚠️  超过30%的股票因'趋势强度低'被过滤")
        print("   建议：降低 MIN_TREND_STRENGTH（当前50）")
    
    if filtered_reasons['no_core_signals'] > filtered_reasons['total_checked'] * 0.5:
        print("\n⚠️  超过50%的股票因'无核心信号'被过滤")
        print("   建议：放宽核心信号检测条件")
        print("   - 降低流动性猎取阈值")
        print("   - 放宽订单块识别条件")
        print("   - 降低FVG最小缺口比例")
    
    if signal_count == 0:
        print("\n❌ 严重问题：完全没有找到任何信号！")
        print("   建议：")
        print("   1. 检查策略参数是否过于严格")
        print("   2. 检查数据质量")
        print("   3. 尝试放宽多个筛选条件")
    
    data_loader.close()
    strategy.close()

if __name__ == "__main__":
    diagnose()
