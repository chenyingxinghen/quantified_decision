"""
因子计算参数自动并行调优脚本

功能：
1. 并行评估所有因子计算参数（周期、阈值等）
2. 使用 Rank IC (Spearman 相关系数) 作为优化目标
3. 采用贪婪坐标下降算法 (Greedy Coordinate Descent) 进行多轮优化
4. 自动更新 config/factor_config.py 文件
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from joblib import Parallel, delayed
import scipy.stats as stats
import re

# 尝试导入 tqdm
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.factors.comprehensive_factor_calculator import ComprehensiveFactorCalculator
from core.factors.quantitative_factors import QuantitativeFactors
from core.factors.candlestick_pattern_factors import CandlestickPatternFactors
from config import DATABASE_PATH, FactorConfig, TrainingConfig

# 忽略警告
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 搜索空间定义
# ============================================================================

SEARCH_SPACE = {
    # 动量类因子
    'RSI_PERIOD': [6, 9, 14, 21, 28, 35],
    'ROC_PERIOD': [5, 10, 12, 20, 30, 40],
    'MTM_PERIOD': [5, 10, 12, 20, 30, 40],
    'CMO_PERIOD': [9, 14, 21, 28, 35],
    'STOCHRSI_PERIOD': [9, 14, 21, 28, 35],
    'RVI_PERIOD': [5, 10, 14, 20, 30],
    
    # 趋势类因子
    'MACD_FAST': [5, 10, 12, 15, 20],
    'MACD_SLOW': [20, 26, 30, 40, 60],
    'MACD_SIGNAL': [5, 7, 9, 12, 15],
    'ADX_PERIOD': [7, 10, 14, 21, 28],
    'DMI_PERIOD': [7, 10, 14, 21, 28],
    'AROON_PERIOD': [14, 20, 25, 30, 50],
    'TRIX_PERIOD': [15, 20, 30, 45, 60],
    'MA_RATIO_PERIOD': [5, 10, 20, 30, 60],
    'MA_SLOPE_PERIOD': [5, 10, 20, 30, 60],
    
    # 波动率因子
    'ATR_PERIOD': [5, 10, 14, 20, 30],
    'NATR_PERIOD': [5, 7, 10, 14, 21],
    'BB_PERIOD': [10, 15, 20, 25, 30, 50],
    'BB_STD': [1.0, 1.5, 2.0, 2.5, 3.0],
    'CCI_PERIOD': [7, 10, 14, 21, 28],
    'ULCER_PERIOD': [7, 10, 14, 21, 28],
    'PRICE_VAR_PERIOD': [10, 20, 30, 60],
    
    # 成交量因子
    'VOLUME_MA_PERIOD': [5, 10, 20, 30, 60],
    'VOLUME_STD_PERIOD': [5, 10, 20, 30, 60],
    'AMOUNT_MA_PERIOD': [5, 10, 20, 30, 60],
    'AMOUNT_STD_PERIOD': [5, 10, 20, 30, 60],
    'MFI_PERIOD': [7, 10, 14, 21, 28],
    'VR_PERIOD': [13, 20, 26, 39],
    'VROC_PERIOD': [6, 12, 18, 24],
    'VRSI_PERIOD': [3, 6, 9, 12, 15],
    'VMACD_FAST': [6, 12, 15, 20],
    'VMACD_SLOW': [13, 26, 30, 45],
    'VMACD_SIGNAL': [5, 7, 9, 12, 15],
    'ADOSC_FAST': [2, 3, 5, 7],
    'ADOSC_SLOW': [7, 10, 15, 20],
    
    # 摆动指标参数
    'KDJ_N': [5, 9, 14, 21],
    'WILLR_PERIOD': [7, 10, 14, 21, 28],
    'BIAS_PERIOD': [3, 6, 12, 24],
    'PSY_PERIOD': [6, 12, 18, 24],
    'AR_BR_PERIOD': [13, 20, 26, 39],
    'CR_PERIOD': [13, 20, 26, 39],
    
    # K线形态参数
    'BODY_SIZE_THRESHOLD_LARGE': [0.01, 0.015, 0.02, 0.03],
    'BODY_SIZE_THRESHOLD_SMALL': [0.002, 0.003, 0.005, 0.008],
    'HAMMER_LOWER_SHADOW_RATIO': [1.5, 2.0, 2.5, 3.0],
    'HAMMER_UPPER_SHADOW_RATIO': [0.5, 0.8, 1.0, 1.5]
}

# 形态因子对参数的映射
PARAM_TO_METHOD_MAP = {
    'RSI_PERIOD': ('tech', 'calculate_rsi'),
    'ROC_PERIOD': ('tech', 'calculate_roc'),
    'MTM_PERIOD': ('tech', 'calculate_mtm'),
    'CMO_PERIOD': ('tech', 'calculate_cmo'),
    'STOCHRSI_PERIOD': ('tech', 'calculate_stochrsi'),
    'RVI_PERIOD': ('tech', 'calculate_rvi'),
    'MACD_FAST': ('tech', 'calculate_macd'),
    'MACD_SLOW': ('tech', 'calculate_macd'),
    'MACD_SIGNAL': ('tech', 'calculate_macd'),
    'ADX_PERIOD': ('tech', 'calculate_adx'),
    'DMI_PERIOD': ('tech', 'calculate_dmi'),
    'AROON_PERIOD': ('tech', 'calculate_aroon'),
    'TRIX_PERIOD': ('tech', 'calculate_trix'),
    'MA_RATIO_PERIOD': ('tech', 'calculate_ma_ratio'),
    'MA_SLOPE_PERIOD': ('tech', 'calculate_ma_slope'),
    'ATR_PERIOD': ('tech', 'calculate_atr'),
    'NATR_PERIOD': ('tech', 'calculate_natr'),
    'BB_PERIOD': ('tech', 'calculate_bollinger_bands'),
    'BB_STD': ('tech', 'calculate_bollinger_bands'),
    'CCI_PERIOD': ('tech', 'calculate_cci'),
    'ULCER_PERIOD': ('tech', 'calculate_ulcer_index'),
    'PRICE_VAR_PERIOD': ('tech', 'calculate_price_variance'),
    'VOLUME_MA_PERIOD': ('tech', 'calculate_volume_ma'),
    'VOLUME_STD_PERIOD': ('tech', 'calculate_volume_std'),
    'AMOUNT_MA_PERIOD': ('tech', 'calculate_amount_ma'),
    'AMOUNT_STD_PERIOD': ('tech', 'calculate_amount_std'),
    'MFI_PERIOD': ('tech', 'calculate_mfi'),
    'VR_PERIOD': ('tech', 'calculate_vr'),
    'VROC_PERIOD': ('tech', 'calculate_vroc'),
    'VRSI_PERIOD': ('tech', 'calculate_vrsi'),
    'VMACD_FAST': ('tech', 'calculate_vmacd'),
    'VMACD_SLOW': ('tech', 'calculate_vmacd'),
    'VMACD_SIGNAL': ('tech', 'calculate_vmacd'),
    'ADOSC_FAST': ('tech', 'calculate_adosc'),
    'ADOSC_SLOW': ('tech', 'calculate_adosc'),
    'KDJ_N': ('tech', 'calculate_kdj'),
    'WILLR_PERIOD': ('tech', 'calculate_willr'),
    'BIAS_PERIOD': ('tech', 'calculate_bias'),
    'PSY_PERIOD': ('tech', 'calculate_psy'),
    'AR_BR_PERIOD': ('tech', 'calculate_ar_br'),
    'CR_PERIOD': ('tech', 'calculate_cr'),
    'BODY_SIZE_THRESHOLD_LARGE': ('candle', 'calculate_all_candlestick_patterns'),
    'BODY_SIZE_THRESHOLD_SMALL': ('candle', 'calculate_all_candlestick_patterns'),
    'HAMMER_LOWER_SHADOW_RATIO': ('candle', 'calculate_all_candlestick_patterns'),
    'HAMMER_UPPER_SHADOW_RATIO': ('candle', 'calculate_all_candlestick_patterns')
}

class FactorTuner:
    """因子超参数调优器"""
    
    def __init__(self, n_stocks: int = 300, years: int = 3, n_jobs: int = -1):
        self.n_stocks = n_stocks
        self.years = years
        self.n_jobs = n_jobs
        self.db_path = DATABASE_PATH
        self.current_config = self._load_current_config()
        self.stocks_data = {}
        self.forward_returns = {}
        
    def _load_current_config(self) -> Dict[str, Any]:
        """从 FactorConfig 类中获取当前参数值"""
        config_dict = {}
        for key in SEARCH_SPACE.keys():
            if hasattr(FactorConfig, key):
                config_dict[key] = getattr(FactorConfig, key)
        return config_dict

    def load_data(self):
        """加载用于优化的股票数据"""
        print(f"正在加载 {self.n_stocks} 只活跃股票的数据 (最近 {self.years} 年)...")
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365 * self.years)).strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        
        # 选取 N 只活跃股票
        stock_query = f'''
            SELECT code, COUNT(*) as cnt
            FROM daily_data
            WHERE date >= ?
            GROUP BY code
            HAVING cnt > 200
            ORDER BY RANDOM()
            LIMIT ?
        '''
        stock_df = pd.read_sql_query(stock_query, conn, params=(start_date, self.n_stocks))
        codes = stock_df['code'].tolist()
        
        if not codes:
            print("错误: 未能在数据库中找到符合条件的股票")
            conn.close()
            return

        placeholders = ','.join(['?' for _ in codes])
        data_query = f'''
            SELECT code, date, open, high, low, close, volume, amount
            FROM daily_data
            WHERE code IN ({placeholders}) AND date >= ?
            ORDER BY code, date ASC
        '''
        all_data = pd.read_sql_query(data_query, conn, params=codes + [start_date])
        conn.close()
        
        # 预处理数据和未来收益率
        for code in codes:
            df = all_data[all_data['code'] == code].copy()
            if len(df) < 200: continue
            
            # N日未来收益率
            df['target'] = df['close'].pct_change(TrainingConfig.FUTURE_DAYS).shift(-TrainingConfig.FUTURE_DAYS)
            
            # 移除 NaN 目标行，但保留原始数据用于因子计算
            # 我们需要完整的序列来计算因子，但在计算 IC 时只取有 target 的行
            self.stocks_data[code] = df
            self.forward_returns[code] = df['target']
                
        print(f"成功加载 {len(self.stocks_data)} 只股票的数据")

    def evaluate_parameter(self, param_name: str, value: Any) -> float:
        """评估特定参数值的平均 Rank IC"""
        
        # 创建临时配置对象
        class TempConfig(FactorConfig):
            pass
        
        # 设置所有当前最优值
        for k, v in self.current_config.items():
            setattr(TempConfig, k, v)
        # 覆盖待测试的值
        setattr(TempConfig, param_name, value)
        
        # 获取计算方法映射
        method_info = PARAM_TO_METHOD_MAP.get(param_name)
        if not method_info:
            return 0.0
            
        calc_type, method_name = method_info
        
        def process_stock(code, data):
            try:
                # 针对不同类型的因子选择不同的计算器
                if calc_type == 'tech':
                    calc = QuantitativeFactors(config=TempConfig)
                else:
                    calc = CandlestickPatternFactors(config=TempConfig)
                
                method = getattr(calc, method_name)
                
                # 执行计算
                if calc_type == 'tech':
                    # 大多数方法只需要 data，有些可能需要额外参数
                    # 检查参数数量
                    import inspect
                    sig = inspect.signature(method)
                    if 'period' in sig.parameters:
                        res = method(data, period=getattr(TempConfig, param_name) if 'PERIOD' in param_name else 14)
                    elif 'n' in sig.parameters:
                        res = method(data, n=getattr(TempConfig, 'KDJ_N'))
                    else:
                        res = method(data)
                else:
                    res = method(data)
                
                # 处理返回结果（可能是单个 array 或 tuple）
                factor_results = []
                if isinstance(res, tuple):
                    for item in res:
                        if isinstance(item, np.ndarray) or isinstance(item, pd.Series):
                            factor_results.append(item)
                else:
                    factor_results.append(res)
                
                v_target = self.forward_returns[code].values
                stock_ics = []
                
                for factor_values in factor_results:
                    # 确保是 array 且长度一致
                    if isinstance(factor_values, pd.Series):
                        v_factor = factor_values.values
                    else:
                        v_factor = factor_values
                        
                    if len(v_factor) != len(v_target):
                        continue
                        
                    # 移除无效值
                    mask = np.isfinite(v_factor) & np.isfinite(v_target)
                    if np.sum(mask) > 100:
                        v_f = v_factor[mask]
                        v_t = v_target[mask]
                        if np.ptp(v_f) > 0 and np.ptp(v_t) > 0:
                            ic, _ = stats.spearmanr(v_f, v_t)
                            if np.isfinite(ic):
                                stock_ics.append(abs(ic))
                
                return stock_ics
            except Exception as e:
                return []

        # 并行计算
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_stock)(code, data) 
            for code, data in self.stocks_data.items()
        )
        
        all_ics = [ic for sublist in results for ic in sublist]
        return np.mean(all_ics) if all_ics else 0.0

    def optimize(self, max_rounds: int = 1):
        """采用贪婪坐标下降进行优化"""
        print("\n" + "="*80)
        print("开始因子参数并行调优流程")
        print("="*80)
        
        for round_idx in range(max_rounds):
            print(f"\n[第 {round_idx + 1} 轮优化]")
            
            params_to_opt = list(SEARCH_SPACE.keys())
            # 随机打乱以提高鲁棒性
            np.random.shuffle(params_to_opt)
            
            pbar = tqdm(params_to_opt, desc="优化进度")
            for param_name in pbar:
                values = SEARCH_SPACE[param_name]
                current_val = self.current_config[param_name]
                
                pbar.set_description(f"正在优化 {param_name}")
                
                best_score = -1.0
                best_val = current_val
                
                # 首先评估当前值作为基准
                base_score = self.evaluate_parameter(param_name, current_val)
                best_score = base_score
                
                for val in values:
                    if val == current_val:
                        continue
                        
                    score = self.evaluate_parameter(param_name, val)
                    
                    if score > best_score * 1.001: # 只有提升超过 0.1% 才更新，避免微小波动
                        best_score = score
                        best_val = val
                
                if best_val != current_val:
                    tqdm.write(f"  ★ {param_name}: {current_val} -> {best_val} (IC: {base_score:.6f} -> {best_score:.6f})")
                    self.current_config[param_name] = best_val
                else:
                    # tqdm.write(f"  ✓ {param_name} 保持原值: {current_val}")
                    pass

    def update_config_file(self):
        """将优化的参数写回 config/factor_config.py"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'factor_config.py')
        
        if not os.path.exists(config_path):
            print(f"错误: 配置文件不存在 {config_path}")
            return

        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        modified = False
        new_lines = []
        
        # 查找 FactorConfig 类定义
        in_factor_config = False
        
        for line in lines:
            if 'class FactorConfig:' in line:
                in_factor_config = True
                new_lines.append(line)
                continue
            
            if in_factor_config and line.startswith('class '): # 进入了下一个类
                in_factor_config = False
            
            if in_factor_config:
                # 匹配 param = value
                match = re.match(r'^(\s+)([A-Z_0-9]+)(\s*=\s*)([\d\.]+)', line)
                if match:
                    indent, param, eq, old_val = match.groups()
                    if param in self.current_config:
                        new_val = self.current_config[param]
                        if str(old_val) != str(new_val):
                            line = f"{indent}{param}{eq}{new_val}\n"
                            modified = True
            
            new_lines.append(line)
            
        if modified:
            with open(config_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"\n✓ 成功将优化参数写入 {config_path}")
        else:
            print("\n! 未检测到参数变化，无需更新文件")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='因子计算超参数并行调优')
    parser.add_argument('--stocks', type=int, default=200, help='使用的股票数量')
    parser.add_argument('--years', type=int, default=1, help='历史数据年数')
    parser.add_argument('--rounds', type=int, default=1, help='优化轮数')
    parser.add_argument('--n-jobs', type=int, default=-1, help='并行任务数')
    
    args = parser.parse_args()
    
    tuner = FactorTuner(n_stocks=args.stocks, years=args.years, n_jobs=args.n_jobs)
    tuner.load_data()
    
    if not tuner.stocks_data:
        print("错误: 未能加载有效数据")
        return
        
    tuner.optimize(max_rounds=args.rounds)
    tuner.update_config_file()
    
    print("\n调优流程结束！")

if __name__ == '__main__':
    main()
