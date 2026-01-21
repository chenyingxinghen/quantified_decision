# 回测运行脚本 - 主入口

from datetime import datetime, timedelta
from backtest_engine import BacktestEngine
from performance_analyzer import PerformanceAnalyzer, ReportGenerator


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("股票交易策略回测系统")
    print("=" * 80)
    
    # 用户输入
    print("\n请选择回测策略:")
    print("1. SMC流动性猎取策略 (liquidity_grab)")
    print("2. 威科夫Spring反转策略 (wyckoff_spring)")
    
    choice = input("\n请选择 (1-2): ").strip()
    
    if choice == "1":
        strategy_name = "liquidity_grab"
        strategy_display = "SMC流动性猎取策略"
    elif choice == "2":
        strategy_name = "wyckoff_spring"
        strategy_display = "威科夫Spring反转策略"
    else:
        print("无效选择，使用默认策略: SMC流动性猎取")
        strategy_name = "liquidity_grab"
        strategy_display = "SMC流动性猎取策略"
    
    # 设置回测时间范围（默认近3年）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    
    print(f"\n默认回测时间范围: {start_date} 至 {end_date}")
    custom_range = input("是否自定义时间范围？(y/n): ").strip().lower()
    
    if custom_range == 'y':
        start_date = input("请输入开始日期 (YYYY-MM-DD): ").strip()
        end_date = input("请输入结束日期 (YYYY-MM-DD): ").strip()
    
    # 初始化回测引擎
    print(f"\n正在初始化回测引擎...")
    engine = BacktestEngine(initial_capital=1.0, commission_rate=0.01)
    
    # 运行回测
    engine.run(start_date, end_date, strategy_name)
    
    # 获取回测结果
    results = engine.get_results()
    
    # 性能分析
    print("\n正在分析回测结果...")
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_metrics(
        results['trades'],
        results['initial_capital'],
        results['final_capital']
    )
    
    # 生成报告
    report_generator = ReportGenerator()
    report_generator.print_summary(metrics, start_date, end_date, strategy_display)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存交易记录
    csv_path = f"backtest_trades_{strategy_name}_{timestamp}.csv"
    report_generator.save_trades_to_csv(results['trades'], csv_path)
    
    # 绘制收益率曲线
    chart_path = f"backtest_equity_curve_{strategy_name}_{timestamp}.png"
    analyzer.plot_equity_curve(results['equity_curve'], chart_path)
    
    print("\n" + "=" * 80)
    print("回测完成！所有结果已保存。")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n执行出错: {e}")
        import traceback
        traceback.print_exc()
