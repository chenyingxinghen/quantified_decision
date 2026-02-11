"""
回测集体验证
运行一个小型回测，强制使用实时计算，验证策略集成是否正确
"""

import os
import sys
import pandas as pd
from datetime import datetime

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtest.engine import BacktestEngine
from core.backtest.data_handler import DataHandler
from core.backtest.strategies.ml_factor_strategy import MLFactorBacktestStrategy
from config import DATABASE_PATH

def verify_backtest():
    print("\n" + "="*80)
    print("回测集成验证 (强制实时计算)")
    print("="*80)
    
    # 1. 准备策略
    model_path = 'models/xgboost_factor_model.pkl'
    strategy = MLFactorBacktestStrategy(
        model_path=model_path,
        min_confidence=30.0,  # 调低一点以便看到信号
        use_cache=False,      # 强制实时计算以测试 Fallback
    )
    
    # 2. 准备数据处理器
    data_handler = DataHandler(db_path=DATABASE_PATH)
    
    # 3. 初始化回测引擎
    engine = BacktestEngine(
        strategy=strategy,
        data_handler=data_handler,
        initial_capital=100000.0
    )
    
    # 4. 运行回测 (只跑一只股票，几天时间)
    test_codes = ['000001']
    start_date = '2024-01-01'
    end_date = '2024-01-15'
    
    print(f"\n  开始回测: {test_codes} 从 {start_date} 到 {end_date}")
    
    try:
        results = engine.run(
            start_date=start_date,
            end_date=end_date,
            stock_codes=test_codes
        )
        
        print("\n  回测完成!")
        metrics = results['metrics']
        print(f"  最终资产: {metrics['total_return'] + 100000.0:.2f}")
        print(f"  总收益率: {metrics['total_return_pct']:.2g}%")
        print(f"  交易次数: {metrics['total_trades']}")
        
        if metrics['total_trades'] > 0:
            print("  ✓ 成功生成交易信号")
        else:
            print("  ! 未生成交易信号，但这可能是因为行情或置信度原因。只要没报错就说明计算流程通了。")
            
    except Exception as e:
        print(f"\n  [失败] 回测运行报错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    verify_backtest()
