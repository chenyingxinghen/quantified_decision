#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动回测迭代脚本 - 持续优化和验证的闭环系统

核心功能：
1. 自动运行回测
2. 分析结果并识别问题
3. 优化参数
4. 验证优化效果
5. 记录迭代历史
6. 自动决策是否采用新参数

工作流程：
迭代1: 初始回测 → 分析问题 → 优化参数 → 验证效果
迭代2: 使用新参数 → 分析问题 → 继续优化 → 验证效果
...
迭代N: 达到目标或无法继续改进

使用方式：
python auto_backtest_iteration.py --strategy liquidity_grab --max-iterations 5
"""

import sys
import os
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import shutil

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantitative_retest.backtest_engine import BacktestEngine
from quantitative_retest.performance_analyzer import PerformanceAnalyzer, ReportGenerator
from quantitative_retest.parameter_optimizer import ParameterOptimizer
from quantitative_retest.smart_parameter_tuner import SmartParameterTuner


class IterationRecord:
    """迭代记录"""
    def __init__(self, iteration: int, parameters: Dict, metrics: Dict, 
                 problems: List[str], improvements: List[str]):
        self.iteration = iteration
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.parameters = parameters
        self.metrics = metrics
        self.problems = problems
        self.improvements = improvements
        self.score = self._calculate_score(metrics)
    
    def _calculate_score(self, metrics: Dict) -> float:
        """计算综合评分"""
        return (
            metrics.get('total_return', 0) * 30 +
            metrics.get('win_rate', 0) * 30 +
            metrics.get('avg_return', 0) * 20 -
            abs(metrics.get('max_drawdown', 0)) * 20
        )
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'iteration': self.iteration,
            'timestamp': self.timestamp,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'problems': self.problems,
            'improvements': self.improvements,
            'score': self.score
        }


class AutoBacktestIteration:
    """
    自动回测迭代系统
    
    核心思路：
    1. 运行回测 → 获取结果
    2. 智能分析 → 识别问题
    3. 参数优化 → 寻找改进
    4. 验证效果 → 决策是否采用
    5. 记录历史 → 持续改进
    """
    
    def __init__(self, strategy_name: str = 'liquidity_grab',
                 initial_capital: float = 1.0,
                 commission_rate: float = 0.01,
                 output_dir: str = 'quantitative_retest/iterations'):
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 组件
        self.analyzer = PerformanceAnalyzer()
        self.reporter = ReportGenerator()
        self.optimizer = ParameterOptimizer(strategy_name=strategy_name)
        self.tuner = SmartParameterTuner()
        
        # 迭代历史
        self.iteration_history: List[IterationRecord] = []
        self.best_iteration: Optional[IterationRecord] = None
        
        # 目标阈值
        self.target_win_rate = 0.65  # 目标胜率65%
        self.target_return = 0.20    # 目标收益率20%
        self.min_improvement = 0.02  # 最小改进幅度2%
    
    def run(self, start_date: str, end_date: str, 
            max_iterations: int = 5,
            enable_optimization: bool = True,
            optimization_method: str = 'bayesian',
            auto_apply: bool = False):
        """
        运行自动迭代
        
        参数:
            start_date: 回测开始日期
            end_date: 回测结束日期
            max_iterations: 最大迭代次数
            enable_optimization: 是否启用参数优化
            optimization_method: 优化方法 ('grid' 或 'bayesian')
            auto_apply: 是否自动应用改进的参数
        """
        print("=" * 80)
        print("自动回测迭代系统")
        print("=" * 80)
        print(f"策略: {self.strategy_name}")
        print(f"时间范围: {start_date} 至 {end_date}")
        print(f"最大迭代次数: {max_iterations}")
        print(f"参数优化: {'启用' if enable_optimization else '禁用'}")
        print(f"自动应用: {'是' if auto_apply else '否'}")
        print("=" * 80)
        
        # 数据分割（训练集/验证集/测试集）
        train_start, train_end, val_start, val_end, test_start, test_end = \
            self._split_data(start_date, end_date)
        
        print(f"\n数据分割:")
        print(f"  训练集: {train_start} 至 {train_end} (60%)")
        print(f"  验证集: {val_start} 至 {val_end} (20%)")
        print(f"  测试集: {test_start} 至 {test_end} (20%)")
        print()
        
        # 开始迭代
        for iteration in range(1, max_iterations + 1):
            print("\n" + "=" * 80)
            print(f"迭代 {iteration}/{max_iterations}")
            print("=" * 80)
            
            # 步骤1: 运行回测
            print(f"\n【步骤1】运行回测（训练集）")
            print("-" * 80)
            train_results = self._run_backtest(train_start, train_end, iteration, 'train')
            
            if not train_results:
                print("回测失败，终止迭代")
                break
            
            # 步骤2: 分析结果
            print(f"\n【步骤2】分析结果")
            print("-" * 80)
            problems, suggestions = self._analyze_results(train_results, iteration)
            
            # 步骤3: 记录当前迭代
            current_params = self._get_current_parameters()
            current_record = IterationRecord(
                iteration=iteration,
                parameters=current_params,
                metrics=train_results['metrics'],
                problems=problems,
                improvements=suggestions
            )
            self.iteration_history.append(current_record)
            
            # 步骤4: 检查是否达到目标
            if self._check_target_reached(train_results['metrics']):
                print("\n✓ 已达到目标！")
                self._validate_on_test_set(test_start, test_end, iteration)
                break
            
            # 步骤5: 参数优化（如果启用且不是最后一次迭代）
            if enable_optimization and iteration < max_iterations:
                print(f"\n【步骤3】参数优化")
                print("-" * 80)
                
                optimized_params = self._optimize_parameters(
                    train_start, train_end,
                    method=optimization_method,
                    suggestions=suggestions
                )
                
                if optimized_params:
                    # 步骤6: 验证优化效果
                    print(f"\n【步骤4】验证优化效果（验证集）")
                    print("-" * 80)
                    
                    # 临时应用新参数
                    self._apply_parameters_temporarily(optimized_params)
                    val_results = self._run_backtest(val_start, val_end, iteration, 'validation')
                    
                    if val_results:
                        # 对比改进
                        improvement = self._calculate_improvement(
                            train_results['metrics'],
                            val_results['metrics']
                        )
                        
                        print(f"\n改进情况:")
                        print(f"  胜率: {train_results['metrics']['win_rate']*100:.2f}% → "
                              f"{val_results['metrics']['win_rate']*100:.2f}% "
                              f"({improvement['win_rate']:+.2f}%)")
                        print(f"  收益率: {train_results['metrics']['total_return']*100:.2f}% → "
                              f"{val_results['metrics']['total_return']*100:.2f}% "
                              f"({improvement['total_return']:+.2f}%)")
                        
                        # 决策是否采用新参数
                        should_apply = self._should_apply_new_params(
                            improvement, auto_apply
                        )
                        
                        if should_apply:
                            print("\n✓ 采用新参数")
                            self._apply_parameters_permanently(optimized_params)
                            self._backup_strategy_file(iteration)
                        else:
                            print("\n✗ 不采用新参数，恢复原参数")
                            self._restore_parameters()
                    else:
                        print("\n验证失败，恢复原参数")
                        self._restore_parameters()
            
            # 步骤7: 更新最佳迭代
            if self.best_iteration is None or current_record.score > self.best_iteration.score:
                self.best_iteration = current_record
                print(f"\n✓ 发现更优配置！评分: {current_record.score:.2f}")
        
        # 最终测试
        print("\n" + "=" * 80)
        print("最终测试（测试集）")
        print("=" * 80)
        self._validate_on_test_set(test_start, test_end, max_iterations)
        
        # 生成迭代报告
        self._generate_iteration_report()
        
        print("\n" + "=" * 80)
        print("迭代完成！")
        print("=" * 80)
    
    def _split_data(self, start_date: str, end_date: str) -> Tuple[str, str, str, str, str, str]:
        """
        分割数据为训练集/验证集/测试集 (60%/20%/20%)
        
        返回:
            (train_start, train_end, val_start, val_end, test_start, test_end)
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        total_days = (end - start).days
        train_days = int(total_days * 0.6)
        val_days = int(total_days * 0.2)
        
        train_end = start + timedelta(days=train_days)
        val_start = train_end + timedelta(days=1)
        val_end = val_start + timedelta(days=val_days)
        test_start = val_end + timedelta(days=1)
        
        return (
            start.strftime('%Y-%m-%d'),
            train_end.strftime('%Y-%m-%d'),
            val_start.strftime('%Y-%m-%d'),
            val_end.strftime('%Y-%m-%d'),
            test_start.strftime('%Y-%m-%d'),
            end.strftime('%Y-%m-%d')
        )
    
    def _run_backtest(self, start_date: str, end_date: str, 
                     iteration: int, phase: str) -> Optional[Dict]:
        """
        运行回测
        
        返回:
            {'trades': List[Trade], 'metrics': Dict, 'equity_curve': List}
        """
        try:
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate
            )
            
            engine.run(start_date, end_date, self.strategy_name)
            
            results = engine.get_results()
            metrics = self.analyzer.calculate_metrics(
                results['trades'],
                results['initial_capital'],
                results['final_capital']
            )
            
            # 保存交易记录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trades_file = os.path.join(
                self.output_dir,
                f"iter{iteration}_{phase}_trades_{timestamp}.csv"
            )
            self.reporter.save_trades_to_csv(results['trades'], trades_file)
            
            # 保存收益曲线
            equity_file = os.path.join(
                self.output_dir,
                f"iter{iteration}_{phase}_equity_{timestamp}.png"
            )
            
            # 获取策略显示名称
            strategy_display = {
                'liquidity_grab': 'SMC流动性猎取策略',
                'wyckoff_spring': '威科夫Spring反转策略'
            }.get(self.strategy_name, self.strategy_name)
            
            self.analyzer.plot_equity_curve(results['equity_curve'], equity_file, strategy_display)
            
            print(f"✓ 回测完成")
            print(f"  交易次数: {metrics['total_trades']}")
            print(f"  胜率: {metrics['win_rate']*100:.2f}%")
            print(f"  总收益率: {metrics['total_return']*100:.2f}%")
            
            return {
                'trades': results['trades'],
                'metrics': metrics,
                'equity_curve': results['equity_curve'],
                'trades_file': trades_file
            }
        
        except Exception as e:
            print(f"✗ 回测失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _analyze_results(self, results: Dict, iteration: int) -> Tuple[List[str], List[str]]:
        """
        分析回测结果，识别问题和建议
        
        返回:
            (problems, suggestions)
        """
        problems = []
        suggestions = []
        
        metrics = results['metrics']
        
        # 使用智能分析器
        if results.get('trades_file'):
            analysis = self.tuner.analyze_historical_trades(results['trades_file'])
            suggested_params = self.tuner.suggest_parameter_adjustments(analysis)
            
            # 提取问题
            if metrics['win_rate'] < 0.5:
                problems.append(f"胜率偏低 ({metrics['win_rate']*100:.1f}%)")
            
            if metrics['total_return'] < 0.1:
                problems.append(f"收益率偏低 ({metrics['total_return']*100:.1f}%)")
            
            if abs(metrics['max_drawdown']) > 0.15:
                problems.append(f"最大回撤过大 ({abs(metrics['max_drawdown'])*100:.1f}%)")
            
            # 提取建议
            for param_name, param_value in suggested_params.items():
                suggestions.append(f"{param_name} → {param_value}")
        
        if problems:
            print("\n识别的问题:")
            for problem in problems:
                print(f"  ⚠️ {problem}")
        else:
            print("\n✓ 未发现明显问题")
        
        if suggestions:
            print("\n参数调整建议:")
            for suggestion in suggestions:
                print(f"  💡 {suggestion}")
        
        return problems, suggestions
    
    def _optimize_parameters(self, start_date: str, end_date: str,
                            method: str = 'bayesian',
                            suggestions: List[str] = None) -> Optional[Dict]:
        """
        优化参数
        
        返回:
            最优参数字典
        """
        # 定义要优化的参数（根据策略选择）
        if self.strategy_name == 'liquidity_grab':
            param_names = [
                'MIN_CONFIDENCE',
                'MIN_ENTRY_QUALITY',
                'ATR_STOP_MULTIPLIER',
                'ATR_TARGET_MULTIPLIER',
            ]
        elif self.strategy_name == 'wyckoff_spring':
            param_names = [
                'CONFIDENCE_BUY',
                'MA_MID_PERIOD',
                'VOLUME_SURGE_RATIO',
                'TARGET_PROFIT',
            ]
        else:
            print(f"未知策略: {self.strategy_name}")
            return None
        
        # 定义回测函数
        def backtest_with_params(params: Dict) -> Dict:
            self._apply_parameters_temporarily(params)
            
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate
            )
            engine.run(start_date, end_date, self.strategy_name)
            
            results = engine.get_results()
            metrics = self.analyzer.calculate_metrics(
                results['trades'],
                results['initial_capital'],
                results['final_capital']
            )
            
            return metrics
        
        # 运行优化
        try:
            if method == 'grid':
                best_result = self.optimizer.grid_search(
                    backtest_func=backtest_with_params,
                    param_names=param_names,
                    max_iterations=30
                )
            elif method == 'bayesian':
                best_result = self.optimizer.bayesian_optimization(
                    backtest_func=backtest_with_params,
                    param_names=param_names,
                    n_iterations=20
                )
            else:
                print(f"未知优化方法: {method}")
                return None
            
            if best_result:
                return best_result.parameters
            else:
                return None
        
        except Exception as e:
            print(f"优化失败: {e}")
            return None
    
    def _get_current_parameters(self) -> Dict:
        """获取当前参数"""
        params = {}
        
        if self.strategy_name == 'liquidity_grab':
            import smc_liquidity_strategy as strategy
        elif self.strategy_name == 'wyckoff_spring':
            import wyckoff_strategy as strategy
        else:
            return params
        
        # 提取所有大写的参数
        for name in dir(strategy):
            if name.isupper() and not name.startswith('_'):
                params[name] = getattr(strategy, name)
        
        return params
    
    def _apply_parameters_temporarily(self, params: Dict):
        """临时应用参数（用于测试）"""
        if self.strategy_name == 'liquidity_grab':
            import smc_liquidity_strategy as strategy
        elif self.strategy_name == 'wyckoff_spring':
            import wyckoff_strategy as strategy
        else:
            return
        
        for param_name, param_value in params.items():
            if hasattr(strategy, param_name):
                setattr(strategy, param_name, param_value)
    
    def _apply_parameters_permanently(self, params: Dict):
        """永久应用参数（更新文件）"""
        self.optimizer.update_strategy_parameters(params, self.strategy_name)
        
        # 重新加载模块
        if self.strategy_name == 'liquidity_grab':
            import smc_liquidity_strategy
            import importlib
            importlib.reload(smc_liquidity_strategy)
        elif self.strategy_name == 'wyckoff_spring':
            import wyckoff_strategy
            import importlib
            importlib.reload(wyckoff_strategy)
    
    def _restore_parameters(self):
        """恢复原参数"""
        # 从备份恢复
        if self.strategy_name == 'liquidity_grab':
            backup_file = 'smc_liquidity_strategy.py.backup'
            target_file = 'smc_liquidity_strategy.py'
        elif self.strategy_name == 'wyckoff_spring':
            backup_file = 'wyckoff_strategy.py.backup'
            target_file = 'wyckoff_strategy.py'
        else:
            return
        
        if os.path.exists(backup_file):
            shutil.copy(backup_file, target_file)
            print("✓ 已恢复原参数")
    
    def _backup_strategy_file(self, iteration: int):
        """备份策略文件"""
        if self.strategy_name == 'liquidity_grab':
            source_file = 'smc_liquidity_strategy.py'
        elif self.strategy_name == 'wyckoff_spring':
            source_file = 'wyckoff_strategy.py'
        else:
            return
        
        backup_file = os.path.join(
            self.output_dir,
            f"iter{iteration}_{os.path.basename(source_file)}"
        )
        shutil.copy(source_file, backup_file)
        print(f"✓ 已备份策略文件: {backup_file}")
    
    def _calculate_improvement(self, old_metrics: Dict, new_metrics: Dict) -> Dict:
        """计算改进幅度"""
        return {
            'win_rate': (new_metrics['win_rate'] - old_metrics['win_rate']) * 100,
            'total_return': (new_metrics['total_return'] - old_metrics['total_return']) * 100,
            'avg_return': (new_metrics['avg_return'] - old_metrics['avg_return']) * 100,
        }
    
    def _should_apply_new_params(self, improvement: Dict, auto_apply: bool) -> bool:
        """决策是否应用新参数"""
        # 检查是否有显著改进
        has_improvement = (
            improvement['win_rate'] > self.min_improvement * 100 or
            improvement['total_return'] > self.min_improvement * 100
        )
        
        if auto_apply:
            return has_improvement
        else:
            if has_improvement:
                print("\n是否采用新参数？(y/n): ", end='')
                choice = input().strip().lower()
                return choice == 'y'
            else:
                print("\n改进不明显，不建议采用")
                return False
    
    def _check_target_reached(self, metrics: Dict) -> bool:
        """检查是否达到目标"""
        return (
            metrics['win_rate'] >= self.target_win_rate and
            metrics['total_return'] >= self.target_return
        )
    
    def _validate_on_test_set(self, start_date: str, end_date: str, iteration: int):
        """在测试集上验证"""
        print("\n在测试集上验证最终效果...")
        test_results = self._run_backtest(start_date, end_date, iteration, 'test')
        
        if test_results:
            print("\n测试集表现:")
            print(f"  胜率: {test_results['metrics']['win_rate']*100:.2f}%")
            print(f"  总收益率: {test_results['metrics']['total_return']*100:.2f}%")
            print(f"  最大回撤: {abs(test_results['metrics']['max_drawdown'])*100:.2f}%")
    
    def _generate_iteration_report(self):
        """生成迭代报告"""
        report_file = os.path.join(self.output_dir, 'iteration_report.json')
        
        report = {
            'strategy': self.strategy_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_iterations': len(self.iteration_history),
            'best_iteration': self.best_iteration.to_dict() if self.best_iteration else None,
            'iterations': [record.to_dict() for record in self.iteration_history]
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 迭代报告已保存: {report_file}")
        
        # 打印摘要
        print("\n" + "=" * 80)
        print("迭代摘要")
        print("=" * 80)
        print(f"总迭代次数: {len(self.iteration_history)}")
        
        if self.best_iteration:
            print(f"\n最佳迭代: 第{self.best_iteration.iteration}次")
            print(f"  评分: {self.best_iteration.score:.2f}")
            print(f"  胜率: {self.best_iteration.metrics['win_rate']*100:.2f}%")
            print(f"  收益率: {self.best_iteration.metrics['total_return']*100:.2f}%")
        
        print("\n迭代历史:")
        for record in self.iteration_history:
            print(f"  迭代{record.iteration}: 评分={record.score:.2f}, "
                  f"胜率={record.metrics['win_rate']*100:.1f}%, "
                  f"收益={record.metrics['total_return']*100:.1f}%")
        
        print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自动回测迭代系统')
    
    parser.add_argument('--strategy', type=str, default='liquidity_grab',
                       choices=['liquidity_grab', 'wyckoff_spring'],
                       help='策略名称')
    parser.add_argument('--start-date', type=str, required=True,
                       help='回测开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                       help='回测结束日期 (YYYY-MM-DD)')
    parser.add_argument('--max-iterations', type=int, default=5,
                       help='最大迭代次数')
    parser.add_argument('--optimization-method', type=str, default='bayesian',
                       choices=['grid', 'bayesian'],
                       help='优化方法')
    parser.add_argument('--auto-apply', action='store_true',
                       help='自动应用改进的参数（不询问）')
    parser.add_argument('--disable-optimization', action='store_true',
                       help='禁用参数优化（仅分析）')
    parser.add_argument('--output-dir', type=str, 
                       default='quantitative_retest/iterations',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建迭代系统
    system = AutoBacktestIteration(
        strategy_name=args.strategy,
        output_dir=args.output_dir
    )
    
    # 运行迭代
    system.run(
        start_date=args.start_date,
        end_date=args.end_date,
        max_iterations=args.max_iterations,
        enable_optimization=not args.disable_optimization,
        optimization_method=args.optimization_method,
        auto_apply=args.auto_apply
    )


if __name__ == '__main__':
    # 使用示例：
    # 
    # 1. 基础迭代（手动确认）：
    #    python auto_backtest_iteration.py --strategy liquidity_grab --start-date 2024-01-01 --end-date 2024-12-31 --max-iterations 3
    # 
    # 2. 自动迭代（无需确认）：
    #    python auto_backtest_iteration.py --strategy liquidity_grab --start-date 2024-01-01 --end-date 2024-12-31 --max-iterations 5 --auto-apply
    # 
    # 3. 仅分析不优化：
    #    python auto_backtest_iteration.py --strategy liquidity_grab --start-date 2024-01-01 --end-date 2024-12-31 --max-iterations 3 --disable-optimization
    # 
    # 4. 使用网格搜索：
    #    python auto_backtest_iteration.py --strategy wyckoff_spring --start-date 2024-01-01 --end-date 2024-12-31 --max-iterations 3 --optimization-method grid
    
    main()
