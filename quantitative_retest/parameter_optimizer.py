# 参数优化器 - 基于历史回测结果自动优化策略参数
# 支持多种优化算法：网格搜索、贝叶斯优化、遗传算法

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入策略配置以获取默认值
try:
    from strategy_config import *
except ImportError:
    print("警告: 无法导入strategy_config，将使用硬编码的默认值")


@dataclass
class ParameterRange:
    """参数范围定义"""
    name: str
    min_value: float
    max_value: float
    step: float = None  # 网格搜索步长
    current_value: float = None
    param_type: str = 'float'  # 'float', 'int', 'bool'


@dataclass
class OptimizationResult:
    """优化结果"""
    parameters: Dict[str, Any]
    total_return: float
    win_rate: float
    total_trades: int
    sharpe_ratio: float
    max_drawdown: float
    score: float  # 综合评分


class ParameterOptimizer:
    """
    参数优化器
    
    核心功能：
    1. 从历史回测报告中提取性能指标
    2. 定义参数搜索空间
    3. 使用优化算法寻找最优参数组合
    4. 保存优化历史和最优参数
    """
    
    def __init__(self, strategy_name: str = 'liquidity_grab',
                 optimization_history_path: str = 'quantitative_retest/optimization_history.json'):
        self.strategy_name = strategy_name
        self.optimization_history_path = optimization_history_path
        self.optimization_history = self._load_optimization_history()
        
        # 根据策略定义参数搜索空间
        self.parameter_space = self._define_parameter_space(strategy_name)
    
    def _load_optimization_history(self) -> List[Dict]:
        """加载优化历史"""
        if os.path.exists(self.optimization_history_path):
            try:
                with open(self.optimization_history_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_optimization_history(self):
        """保存优化历史"""
        os.makedirs(os.path.dirname(self.optimization_history_path), exist_ok=True)
        with open(self.optimization_history_path, 'w', encoding='utf-8') as f:
            json.dump(self.optimization_history, f, indent=2, ensure_ascii=False)
    
    def _define_parameter_space(self, strategy_name: str = 'liquidity_grab') -> Dict[str, ParameterRange]:
        """
        定义参数搜索空间（支持多策略）
        
        参数:
            strategy_name: 'liquidity_grab' 或 'wyckoff_spring'
        
        核心参数分类：
        1. 信号检测参数（影响信号质量）
        2. 风险管理参数（影响止损止盈）
        3. 趋势判断参数（影响趋势识别）
        4. 置信度阈值（影响筛选严格度）
        """
        if strategy_name == 'liquidity_grab':
            return self._define_smc_parameters()
        elif strategy_name == 'wyckoff_spring':
            return self._define_wyckoff_parameters()
        else:
            raise ValueError(f"未知策略: {strategy_name}")
    
    def _define_smc_parameters(self) -> Dict[str, ParameterRange]:
        """SMC流动性策略参数空间"""
        return {
            # ========== 信号检测参数 ==========
            'LIQUIDITY_SWEEP_THRESHOLD': ParameterRange(
                name='流动性扫荡阈值',
                min_value=0.01,
                max_value=0.03,
                step=0.005,
                current_value=0.02
            ),
            'ORDER_BLOCK_STRENGTH': ParameterRange(
                name='订单块成交量强度',
                min_value=1.2,
                max_value=2.0,
                step=0.2,
                current_value=1.5
            ),
            'FVG_MIN_GAP_RATIO': ParameterRange(
                name='FVG最小缺口比例',
                min_value=0.0005,
                max_value=0.002,
                step=0.0005,
                current_value=0.001
            ),
            
            # ========== 趋势判断参数 ==========
            'TREND_MA_LONG_PERIOD': ParameterRange(
                name='长期均线周期',
                min_value=60,
                max_value=120,
                step=10,
                current_value=90,
                param_type='int'
            ),
            'TREND_MA_MID_PERIOD': ParameterRange(
                name='中期均线周期',
                min_value=20,
                max_value=50,
                step=5,
                current_value=30,
                param_type='int'
            ),
            'TREND_STRENGTH_STRONG': ParameterRange(
                name='强趋势阈值',
                min_value=60,
                max_value=80,
                step=5,
                current_value=70,
                param_type='int'
            ),
            
            # ========== 风险管理参数 ==========
            'ATR_STOP_MULTIPLIER': ParameterRange(
                name='ATR止损倍数',
                min_value=1.0,
                max_value=2.5,
                step=0.25,
                current_value=1.5
            ),
            'ATR_TARGET_MULTIPLIER': ParameterRange(
                name='ATR目标倍数',
                min_value=2.0,
                max_value=4.0,
                step=0.5,
                current_value=3.0
            ),
            'MIN_RISK_REWARD_RATIO': ParameterRange(
                name='最小风险收益比',
                min_value=1.5,
                max_value=3.0,
                step=0.25,
                current_value=2.0
            ),
            
            # ========== 置信度阈值 ==========
            'MIN_CONFIDENCE': ParameterRange(
                name='最低置信度要求',
                min_value=70,
                max_value=90,
                step=5,
                current_value=MIN_CONFIDENCE if 'MIN_CONFIDENCE' in globals() else 80,
                param_type='int'
            ),
            'MIN_ENTRY_QUALITY': ParameterRange(
                name='最低入场质量',
                min_value=75,
                max_value=95,
                step=5,
                current_value=MIN_ENTRY_QUALITY if 'MIN_ENTRY_QUALITY' in globals() else 85,
                param_type='int'
            ),
            'MIN_TREND_STRENGTH': ParameterRange(
                name='最低趋势强度',
                min_value=70,
                max_value=90,
                step=5,
                current_value=MIN_TREND_STRENGTH if 'MIN_TREND_STRENGTH' in globals() else 80,
                param_type='int'
            ),
            
            # ========== 看空信号参数 ==========
            'BEARISH_STRONG_TREND_THRESHOLD': ParameterRange(
                name='强趋势看空阈值',
                min_value=35,
                max_value=55,
                step=5,
                current_value=BEARISH_STRONG_TREND_THRESHOLD if 'BEARISH_STRONG_TREND_THRESHOLD' in globals() else 45,
                param_type='int'
            ),
            'BEARISH_MODERATE_TREND_THRESHOLD': ParameterRange(
                name='中等趋势看空阈值',
                min_value=45,
                max_value=65,
                step=5,
                current_value=BEARISH_MODERATE_TREND_THRESHOLD if 'BEARISH_MODERATE_TREND_THRESHOLD' in globals() else 55,
                param_type='int'
            ),
            'BEARISH_SIGNAL_MEMORY_DAYS': ParameterRange(
                name='看空信号记忆天数',
                min_value=2,
                max_value=5,
                step=1,
                current_value=BEARISH_SIGNAL_MEMORY_DAYS if 'BEARISH_SIGNAL_MEMORY_DAYS' in globals() else 3,
                param_type='int'
            ),
            'BEARISH_MEMORY_DECAY': ParameterRange(
                name='看空信号衰减系数',
                min_value=0.5,
                max_value=0.9,
                step=0.1,
                current_value=BEARISH_MEMORY_DECAY if 'BEARISH_MEMORY_DECAY' in globals() else 0.7
            ),
        }
    
    def _define_wyckoff_parameters(self) -> Dict[str, ParameterRange]:
        """威科夫Spring策略参数空间"""
        return {
            # ========== 均线参数 ==========
            'MA_SHORT_PERIOD': ParameterRange(
                name='短期均线周期',
                min_value=3,
                max_value=10,
                step=1,
                current_value=5,
                param_type='int'
            ),
            'MA_MID_PERIOD': ParameterRange(
                name='中期均线周期',
                min_value=15,
                max_value=30,
                step=5,
                current_value=20,
                param_type='int'
            ),
            'MA_LONG_PERIOD': ParameterRange(
                name='长期均线周期',
                min_value=50,
                max_value=90,
                step=10,
                current_value=60,
                param_type='int'
            ),
            
            # ========== 横盘区域参数 ==========
            'CONSOLIDATION_DAYS': ParameterRange(
                name='横盘区域检测天数',
                min_value=15,
                max_value=30,
                step=5,
                current_value=20,
                param_type='int'
            ),
            'CONSOLIDATION_RANGE': ParameterRange(
                name='横盘区域价格波动范围',
                min_value=0.10,
                max_value=0.20,
                step=0.02,
                current_value=0.15
            ),
            
            # ========== 成交量参数 ==========
            'VOLUME_SHRINK_RATIO': ParameterRange(
                name='成交量萎缩比例',
                min_value=0.6,
                max_value=0.9,
                step=0.1,
                current_value=0.8
            ),
            'VOLUME_SURGE_RATIO': ParameterRange(
                name='Spring成交量放大比例',
                min_value=1.2,
                max_value=2.0,
                step=0.2,
                current_value=1.5
            ),
            
            # ========== Spring形态参数 ==========
            'SPRING_LOOKBACK_DAYS': ParameterRange(
                name='Spring检测回看天数',
                min_value=5,
                max_value=15,
                step=2,
                current_value=10,
                param_type='int'
            ),
            'SPRING_BREAK_THRESHOLD': ParameterRange(
                name='跌破支撑阈值',
                min_value=0.97,
                max_value=0.995,
                step=0.005,
                current_value=0.99
            ),
            
            # ========== RSI参数 ==========
            'RSI_OVERSOLD_MIN': ParameterRange(
                name='RSI超卖下限',
                min_value=20,
                max_value=35,
                step=5,
                current_value=30,
                param_type='int'
            ),
            'RSI_OVERSOLD_MAX': ParameterRange(
                name='RSI超卖上限',
                min_value=45,
                max_value=60,
                step=5,
                current_value=50,
                param_type='int'
            ),
            
            # ========== 置信度阈值 ==========
            'CONFIDENCE_BUY': ParameterRange(
                name='买入信号置信度阈值',
                min_value=50,
                max_value=70,
                step=5,
                current_value=60,
                param_type='int'
            ),
            
            # ========== 风险管理参数 ==========
            'STOP_LOSS_BUFFER': ParameterRange(
                name='止损缓冲',
                min_value=0.01,
                max_value=0.03,
                step=0.005,
                current_value=0.015
            ),
            'TARGET_PROFIT': ParameterRange(
                name='目标收益',
                min_value=0.08,
                max_value=0.15,
                step=0.01,
                current_value=0.12
            ),
        }
    
    def calculate_score(self, metrics: Dict) -> float:
        """
        计算综合评分（用于参数优化）
        
        评分公式：
        Score = 总收益率 * 0.3 + 胜率 * 0.3 + 夏普比率 * 0.2 - 最大回撤 * 0.2
        
        目标：平衡收益、胜率、风险调整收益和回撤
        """
        total_return = metrics.get('total_return', 0)
        win_rate = metrics.get('win_rate', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = abs(metrics.get('max_drawdown', 0))
        
        # 归一化处理
        score = (
            total_return * 30 +  # 总收益率权重30%
            win_rate * 30 +      # 胜率权重30%
            sharpe_ratio * 20 -  # 夏普比率权重20%
            max_drawdown * 20    # 最大回撤惩罚20%
        )
        
        return score
    
    def grid_search(self, backtest_func, param_names: List[str], 
                   max_iterations: int = 100) -> OptimizationResult:
        """
        网格搜索优化
        
        参数:
            backtest_func: 回测函数，接受参数字典，返回性能指标
            param_names: 要优化的参数名称列表
            max_iterations: 最大迭代次数
        
        返回:
            OptimizationResult: 最优参数组合和性能
        """
        print("=" * 80)
        print("开始网格搜索参数优化")
        print("=" * 80)
        print(f"优化参数: {', '.join(param_names)}")
        print(f"最大迭代次数: {max_iterations}")
        print("-" * 80)
        
        # 生成参数组合
        param_combinations = self._generate_grid_combinations(param_names, max_iterations)
        
        best_result = None
        best_score = -float('inf')
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\n[{i}/{len(param_combinations)}] 测试参数组合:")
            for name, value in params.items():
                print(f"  {name}: {value}")
            
            # 运行回测
            try:
                metrics = backtest_func(params)
                
                # 计算评分
                score = self.calculate_score(metrics)
                
                print(f"  结果: 收益率={metrics['total_return']*100:.2f}% "
                      f"胜率={metrics['win_rate']*100:.2f}% "
                      f"评分={score:.2f}")
                
                # 记录到优化历史
                self.optimization_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'parameters': params,
                    'metrics': metrics,
                    'score': score
                })
                
                # 更新最优结果
                if score > best_score:
                    best_score = score
                    best_result = OptimizationResult(
                        parameters=params,
                        total_return=metrics['total_return'],
                        win_rate=metrics['win_rate'],
                        total_trades=metrics['total_trades'],
                        sharpe_ratio=metrics.get('sharpe_ratio', 0),
                        max_drawdown=metrics.get('max_drawdown', 0),
                        score=score
                    )
                    print(f"  ✓ 发现更优参数！当前最佳评分: {best_score:.2f}")
            
            except Exception as e:
                print(f"  ✗ 回测失败: {e}")
                continue
        
        # 保存优化历史
        self._save_optimization_history()
        
        print("\n" + "=" * 80)
        print("网格搜索完成！")
        print("=" * 80)
        
        if best_result:
            print(f"最优评分: {best_result.score:.2f}")
            print(f"最优参数:")
            for name, value in best_result.parameters.items():
                print(f"  {name}: {value}")
            print(f"性能指标:")
            print(f"  总收益率: {best_result.total_return*100:.2f}%")
            print(f"  胜率: {best_result.win_rate*100:.2f}%")
            print(f"  交易次数: {best_result.total_trades}")
        
        return best_result
    
    def _generate_grid_combinations(self, param_names: List[str], 
                                    max_combinations: int) -> List[Dict]:
        """
        生成网格搜索的参数组合
        
        策略：
        1. 如果组合数 <= max_combinations，返回全部组合
        2. 否则，随机采样max_combinations个组合
        """
        # 为每个参数生成候选值
        param_values = {}
        for name in param_names:
            if name not in self.parameter_space:
                print(f"警告: 参数 {name} 不在搜索空间中，跳过")
                continue
            
            param_range = self.parameter_space[name]
            
            if param_range.param_type == 'int':
                values = list(range(
                    int(param_range.min_value),
                    int(param_range.max_value) + 1,
                    int(param_range.step) if param_range.step else 1
                ))
            else:
                values = np.arange(
                    param_range.min_value,
                    param_range.max_value + param_range.step,
                    param_range.step
                ).tolist()
            
            param_values[name] = values
        
        # 生成所有组合
        from itertools import product
        
        keys = list(param_values.keys())
        values = [param_values[k] for k in keys]
        
        all_combinations = []
        for combo in product(*values):
            all_combinations.append(dict(zip(keys, combo)))
        
        # 如果组合数过多，随机采样
        if len(all_combinations) > max_combinations:
            print(f"总组合数 {len(all_combinations)} 超过限制，随机采样 {max_combinations} 个")
            import random
            all_combinations = random.sample(all_combinations, max_combinations)
        
        return all_combinations
    
    def bayesian_optimization(self, backtest_func, param_names: List[str],
                             n_iterations: int = 50) -> OptimizationResult:
        """
        贝叶斯优化（需要安装scikit-optimize）
        
        优势：
        1. 比网格搜索更高效
        2. 利用历史结果指导搜索方向
        3. 适合连续参数空间
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
        except ImportError:
            print("错误: 贝叶斯优化需要安装 scikit-optimize")
            print("请运行: pip install scikit-optimize")
            return None
        
        print("=" * 80)
        print("开始贝叶斯优化")
        print("=" * 80)
        print(f"优化参数: {', '.join(param_names)}")
        print(f"迭代次数: {n_iterations}")
        print("-" * 80)
        
        # 定义搜索空间
        space = []
        for name in param_names:
            param_range = self.parameter_space[name]
            if param_range.param_type == 'int':
                space.append(Integer(
                    int(param_range.min_value),
                    int(param_range.max_value),
                    name=name
                ))
            else:
                space.append(Real(
                    param_range.min_value,
                    param_range.max_value,
                    name=name
                ))
        
        # 定义目标函数（最小化负评分）
        def objective(param_values):
            params = dict(zip(param_names, param_values))
            
            print(f"\n测试参数组合:")
            for name, value in params.items():
                print(f"  {name}: {value}")
            
            try:
                metrics = backtest_func(params)
                score = self.calculate_score(metrics)
                
                print(f"  结果: 收益率={metrics['total_return']*100:.2f}% "
                      f"胜率={metrics['win_rate']*100:.2f}% "
                      f"评分={score:.2f}")
                
                # 记录历史
                self.optimization_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'parameters': params,
                    'metrics': metrics,
                    'score': score
                })
                
                return -score  # 最小化负评分 = 最大化评分
            
            except Exception as e:
                print(f"  ✗ 回测失败: {e}")
                return 0  # 返回最差评分
        
        # 运行优化
        result = gp_minimize(
            objective,
            space,
            n_calls=n_iterations,
            random_state=42,
            verbose=False
        )
        
        # 保存优化历史
        self._save_optimization_history()
        
        # 构建最优结果
        best_params = dict(zip(param_names, result.x))
        best_metrics = backtest_func(best_params)
        best_score = self.calculate_score(best_metrics)
        
        best_result = OptimizationResult(
            parameters=best_params,
            total_return=best_metrics['total_return'],
            win_rate=best_metrics['win_rate'],
            total_trades=best_metrics['total_trades'],
            sharpe_ratio=best_metrics.get('sharpe_ratio', 0),
            max_drawdown=best_metrics.get('max_drawdown', 0),
            score=best_score
        )
        
        print("\n" + "=" * 80)
        print("贝叶斯优化完成！")
        print("=" * 80)
        print(f"最优评分: {best_result.score:.2f}")
        print(f"最优参数:")
        for name, value in best_result.parameters.items():
            print(f"  {name}: {value}")
        
        return best_result
    
    def get_best_parameters_from_history(self) -> Optional[Dict]:
        """从优化历史中获取最优参数"""
        if not self.optimization_history:
            return None
        
        # 找到评分最高的记录
        best_record = max(self.optimization_history, key=lambda x: x.get('score', -float('inf')))
        return best_record.get('parameters')
    
    def update_strategy_parameters(self, best_params: Dict, 
                                   strategy_name: str = 'liquidity_grab'):
        """
        更新策略文件中的参数
        
        参数:
            best_params: 最优参数字典
            strategy_name: 策略名称
        """
        print("\n" + "=" * 80)
        print("更新策略参数")
        print("=" * 80)
        
        # 根据策略选择文件
        if strategy_name == 'liquidity_grab':
            strategy_file = 'smc_liquidity_strategy.py'
        elif strategy_name == 'wyckoff_spring':
            strategy_file = 'wyckoff_strategy.py'
        else:
            print(f"错误: 未知策略 {strategy_name}")
            return
        
        # 读取策略文件
        with open(strategy_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 更新参数
        updated_count = 0
        for i, line in enumerate(lines):
            for param_name, param_value in best_params.items():
                if line.strip().startswith(f"{param_name} ="):
                    # 保留注释
                    comment_idx = line.find('#')
                    comment = line[comment_idx:] if comment_idx != -1 else '\n'
                    
                    # 更新参数值
                    lines[i] = f"{param_name} = {param_value}  {comment}"
                    updated_count += 1
                    print(f"✓ 更新 {param_name} = {param_value}")
        
        # 写回文件
        with open(strategy_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"\n共更新 {updated_count} 个参数")
        print("=" * 80)
    
    def export_optimization_report(self, output_path: str = 'quantitative_retest/optimization_report.csv'):
        """导出优化报告"""
        if not self.optimization_history:
            print("没有优化历史记录")
            return
        
        # 转换为DataFrame
        records = []
        for record in self.optimization_history:
            row = {
                '时间': record['timestamp'],
                '评分': record['score'],
                '总收益率': f"{record['metrics']['total_return']*100:.2f}%",
                '胜率': f"{record['metrics']['win_rate']*100:.2f}%",
                '交易次数': record['metrics']['total_trades'],
            }
            # 添加参数
            for param_name, param_value in record['parameters'].items():
                row[param_name] = param_value
            
            records.append(row)
        
        df = pd.DataFrame(records)
        df = df.sort_values('评分', ascending=False)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"✓ 优化报告已保存到: {output_path}")


# ========== 使用示例 ==========

if __name__ == '__main__':
    # 示例：如何使用参数优化器
    
    optimizer = ParameterOptimizer()
    
    # 定义回测函数（需要根据实际情况实现）
    def run_backtest_with_params(params: Dict) -> Dict:
        """
        使用指定参数运行回测
        
        返回性能指标字典
        """
        # 这里需要实际实现回测逻辑
        # 1. 更新策略参数
        # 2. 运行回测
        # 3. 返回性能指标
        
        # 示例返回值
        return {
            'total_return': 0.15,
            'win_rate': 0.65,
            'total_trades': 50,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08
        }
    
    # 方法1: 网格搜索
    best_result = optimizer.grid_search(
        backtest_func=run_backtest_with_params,
        param_names=['MIN_CONFIDENCE', 'MIN_ENTRY_QUALITY', 'ATR_STOP_MULTIPLIER'],
        max_iterations=50
    )
    
    # 方法2: 贝叶斯优化（更高效）
    # best_result = optimizer.bayesian_optimization(
    #     backtest_func=run_backtest_with_params,
    #     param_names=['MIN_CONFIDENCE', 'MIN_ENTRY_QUALITY', 'ATR_STOP_MULTIPLIER'],
    #     n_iterations=30
    # )
    
    # 更新策略参数
    if best_result:
        optimizer.update_strategy_parameters(best_result.parameters)
        optimizer.export_optimization_report()
