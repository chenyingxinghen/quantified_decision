"""
使用新回测系统运行回测

演示如何使用core.backtest模块进行回测
"""

import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data.data_fetcher import DataFetcher
from core.backtest import BacktestEngine, DataHandler, PerformanceAnalyzer
from core.backtest.strategies import MLFactorBacktestStrategy
from config import DATABASE_PATH,TrainingConfig
from config.strategy_config import (
    ML_FACTOR_MIN_CONFIDENCE,
    ML_FACTOR_MODEL_PATH,
    INITIAL_CAPITAL,
    COMMISSION_RATE,
    MAX_POSITIONS,
    SELECTOR_MARKETS
)
import pandas as pd
import sqlite3
from datetime import datetime, timedelta


def main():
    """主函数"""
    print("=" * 80)
    print("回测系统")
    print("=" * 80)
    
    # 配置参数
    start_date = (datetime.now() - timedelta(days=365*TrainingConfig.YEARS_FOR_BACKTEST)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    initial_capital = INITIAL_CAPITAL
    commission_rate = COMMISSION_RATE
    
    # 模型路径
    model_path = ML_FACTOR_MODEL_PATH
    if not os.path.isabs(model_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, model_path)
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行 train_ml_model.py 训练模型")
        return
    
    # 1. 创建数据处理器
    print("\n初始化数据处理器...")
    data_handler = DataHandler(DATABASE_PATH)
    
    # 2. 创建策略
    print("初始化策略...")
    
    # 缓存目录
    cache_dir = TrainingConfig.CACHE_DIR
    use_cache = os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0
    
    if use_cache:
        print(f"  检测到因子缓存目录: {cache_dir}")
        print(f"  将使用缓存进行回测")
    else:
        print(f"  错误: 未检测到因子缓存！回测需要预计算的因子缓存。")
        print(f"  请先运行 train_ml_model.py --cache-engineered 生成因子缓存")
        return
    
    strategy = MLFactorBacktestStrategy(
        model_path=model_path,
        min_confidence=ML_FACTOR_MIN_CONFIDENCE,
        use_cache=use_cache,
        cache_dir=cache_dir,
        name="ML因子策略",
    )
    
    # 3. 创建回测引擎
    print("初始化回测引擎...")
    engine = BacktestEngine(
        strategy=strategy,
        data_handler=data_handler,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        max_positions=MAX_POSITIONS
    )
    
    # 4. 运行回测
    print("\n开始回测...")
    
    # 提前获取股票代码
    stock_codes = None # 这里可以指定，不指定则从DB获取
    if stock_codes is None:
        df = DataFetcher()
        df = df.get_stock_list(markets=SELECTOR_MARKETS)
        stock_codes = df['code'].tolist()


    # 为了让第1个交易日就有足够的历史数据（DataHandler要求至少30天），
    # 我们加载比回测开始日期更早的数据
    load_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=60)).strftime('%Y-%m-%d')
    data_handler.load_data(load_start_date, end_date, stock_codes)
    
    results = engine.run(
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes,
        verbose=True
    )
    
    # 5. 保存结果
    print("\n保存结果...")
    
    # 自动解析模型元数据以进行归档
    # 预期路径格式: .../models/{weight_status}_{data_volume}/{model_type}_factor_model.pkl
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    
    # 解析归档目录名 (如 weighted_5y_5000s)
    archive_tag = os.path.basename(model_dir)
    if archive_tag == 'models': # 如果直接放在 models 下
        archive_tag = 'default'
        
    # 解析模型类别 (如 xgboost)
    model_category = model_name.split('_')[0]
    
    # 构造回测标识 (包含置信度和日期)
    backtest_tag = f"conf{int(ML_FACTOR_MIN_CONFIDENCE)}_{start_date}_to_{end_date}"
    
    # 创建归档路径: backtest_result/{archive_tag}/{model_category}/{backtest_tag}/
    result_dir = os.path.join('backtest_result', archive_tag, model_category, backtest_tag)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    # 保存交易记录
    PerformanceAnalyzer.save_trades_to_csv(
        results['trades'],
        os.path.join(result_dir, 'backtest_trades.csv')
    )
    
    # 绘制资金曲线
    PerformanceAnalyzer.plot_equity_curve(
        results['equity_curve'],
        title=f"Backtest Equity Curve ({strategy.name})",
        save_path=os.path.join(result_dir, 'backtest_equity.png')
    )
    
    # 绘制置信度与收益率的关系
    PerformanceAnalyzer.plot_confidence_performance(
        results['trades'],
        title=f"Confidence vs. Return ({strategy.name})",
        save_path=os.path.join(result_dir, 'backtest_confidence_analysis.png')
    )
    
    print("\n回测完成！")
    print(f"存档目录: {result_dir}")

    # 6. 清理
    data_handler.close()


if __name__ == '__main__':
    main()
