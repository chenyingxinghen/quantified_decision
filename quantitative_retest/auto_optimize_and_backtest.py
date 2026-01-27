# 自动优化并回测 - 在回测前自动优化参数
# 使用方式: python auto_optimize_and_backtest.py --enable-optimization

import argparse
import sys
import os
from datetime import datetime, timedelta

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantitative_retest.backtest_engine import BacktestEngine
from quantitative_retest.performance_analyzer import PerformanceAnalyzer, ReportGenerator
from quantitative_retest.parameter_optimizer import ParameterOptimizer
import importlib


class AutoOptimizeBacktest:
    """
    自动优化并回测系统
    
    工作流程：
    1. 检查是否启用参数优化
    2. 如果启用，使用历史数据优化参数
    3. 更新策略参数
    4. 运行回测
    5. 生成报告
    """
    
    def __init__(self, enable_optimization: bool = False, 
                 optimization_method: str = 'grid',
                 optimization_iterations: int = 30):
        self.enable_optimization = enable_optimization
        self.optimization_method = optimization_method
        self.optimization_iterations = optimization_iterations
        
        self.optimizer = ParameterOptimizer()
        self.analyzer = PerformanceAnalyzer()
        self.reporter = ReportGenerator()
    
    def run(self, start_date: str, end_date: str, strategy_name: str = 'liquidity_grab'):
        """
        运行自动优化和回测
        
        参数:
            start_date: 回测开始日期
            end_date: 回测结束日期
            strategy_name: 策略名称
        """
        print("=" * 80)
        print("自动优化并回测系统")
        print("=" * 80)
        print(f"参数优化: {'启用' if self.enable_optimization else '禁用'}")
        print(f"回测时间: {start_date} 至 {end_date}")
        print(f"策略名称: {strategy_name}")
        print("=" * 80)
        
        # ========== 第一步：参数优化（可选）==========
        
        if self.enable_optimization:
            print("\n【第一步】参数优化")
            print("-" * 80)
            
            # 定义要优化的参数
            params_to_optimize = [
                'MIN_CONFIDENCE',
                'MIN_ENTRY_QUALITY',
                'MIN_TREND_STRENGTH',
                'ATR_STOP_MULTIPLIER',
                'ATR_TARGET_MULTIPLIER',
            ]
            
            # 定义回测函数
            def backtest_with_params(params: dict) -> dict:
                """使用指定参数运行回测"""
                # 临时更新参数
                self._update_params_temporarily(params)
                
                # 运行回测
                engine = BacktestEngine(initial_capital=1.0, commission_rate=0.01)
                
                # 使用训练集（前80%数据）进行优化
                train_end = self._get_train_end_date(start_date, end_date)
                engine.run(start_date, train_end, strategy_name)
                
                # 获取结果
                results = engine.get_results()
                metrics = self.analyzer.calculate_metrics(
                    results['trades'],
                    results['initial_capital'],
                    results['final_capital']
                )
                
                return metrics
            
            # 运行优化
            if self.optimization_method == 'grid':
                best_result = self.optimizer.grid_search(
                    backtest_func=backtest_with_params,
                    param_names=params_to_optimize,
                    max_iterations=self.optimization_iterations
                )
            elif self.optimization_method == 'bayesian':
                best_result = self.optimizer.bayesian_optimization(
                    backtest_func=backtest_with_params,
                    param_names=params_to_optimize,
                    n_iterations=self.optimization_iterations
                )
            else:
                print(f"未知的优化方法: {self.optimization_method}")
                best_result = None
            
            # 更新策略参数
            if best_result:
                print("\n更新策略参数...")
                self.optimizer.update_strategy_parameters(best_result.parameters)
                
                # 导出优化报告
                self.optimizer.export_optimization_report()
                
                # 重新加载策略模块（使其使用新参数）
                self._reload_strategy_module()
            else:
                print("优化失败，使用默认参数")
        
        # ========== 第二步：完整回测 ==========
        
        print("\n【第二步】完整回测")
        print("-" * 80)
        
        # 运行完整回测
        engine = BacktestEngine(initial_capital=1.0, commission_rate=0.01)
        engine.run(start_date, end_date, strategy_name)
        
        # ========== 第三步：生成报告 ==========
        
        print("\n【第三步】生成报告")
        print("-" * 80)
        
        results = engine.get_results()
        metrics = self.analyzer.calculate_metrics(
            results['trades'],
            results['initial_capital'],
            results['final_capital']
        )
        
        # 打印摘要
        self.reporter.print_summary(metrics, start_date, end_date, strategy_name)
        
        # 保存交易记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trades_file = f"backtest_trades_{strategy_name}_{timestamp}.csv"
        self.reporter.save_trades_to_csv(results['trades'], trades_file)
        
        # 绘制收益曲线
        equity_file = f"backtest_equity_curve_{strategy_name}_{timestamp}.png"
        
        # 获取策略显示名称
        strategy_display = {
            'liquidity_grab': 'SMC流动性猎取策略',
            'wyckoff_spring': '威科夫Spring反转策略'
        }.get(strategy_name, strategy_name)
        
        self.analyzer.plot_equity_curve(results['equity_curve'], equity_file, strategy_display)
        
        print("\n" + "=" * 80)
        print("全部完成！")
        print("=" * 80)
    
    def _update_params_temporarily(self, params: dict):
        """临时更新参数（用于优化过程）"""
        import smc_liquidity_strategy
        
        for param_name, param_value in params.items():
            if hasattr(smc_liquidity_strategy, param_name):
                setattr(smc_liquidity_strategy, param_name, param_value)
    
    def _reload_strategy_module(self):
        """重新加载策略模块"""
        import smc_liquidity_strategy
        importlib.reload(smc_liquidity_strategy)
        print("✓ 策略模块已重新加载")
    
    def _get_train_end_date(self, start_date: str, end_date: str) -> str:
        """
        获取训练集结束日期（前80%数据）
        
        用于参数优化时避免过拟合
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        total_days = (end - start).days
        train_days = int(total_days * 0.8)
        
        train_end = start + timedelta(days=train_days)
        return train_end.strftime('%Y-%m-%d')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自动优化并回测系统')
    
    parser.add_argument('--enable-optimization', action='store_true',
                       help='启用参数优化')
    parser.add_argument('--optimization-method', type=str, default='grid',
                       choices=['grid', 'bayesian'],
                       help='优化方法: grid(网格搜索) 或 bayesian(贝叶斯优化)')
    parser.add_argument('--optimization-iterations', type=int, default=30,
                       help='优化迭代次数')
    parser.add_argument('--start-date', type=str, required=True,
                       help='回测开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                       help='回测结束日期 (YYYY-MM-DD)')
    parser.add_argument('--strategy', type=str, default='liquidity_grab',
                       choices=['liquidity_grab', 'wyckoff_spring'],
                       help='策略名称')
    
    args = parser.parse_args()
    
    # 创建并运行
    system = AutoOptimizeBacktest(
        enable_optimization=args.enable_optimization,
        optimization_method=args.optimization_method,
        optimization_iterations=args.optimization_iterations
    )
    
    system.run(
        start_date=args.start_date,
        end_date=args.end_date,
        strategy_name=args.strategy
    )


if __name__ == '__main__':
    # 示例用法：
    # 
    # 1. 不启用优化，直接回测：
    #    python auto_optimize_and_backtest.py --start-date 2024-01-01 --end-date 2024-12-31
    # 
    # 2. 启用网格搜索优化：
    #    python auto_optimize_and_backtest.py --enable-optimization --start-date 2024-01-01 --end-date 2024-12-31
    # 
    # 3. 启用贝叶斯优化（更高效）：
    #    python auto_optimize_and_backtest.py --enable-optimization --optimization-method bayesian --optimization-iterations 50 --start-date 2024-01-01 --end-date 2024-12-31
    
    main()
