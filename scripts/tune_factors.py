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
from typing import Dict, List, Any, Tuple
from joblib import Parallel, delayed
import scipy.stats as stats
import re

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.factors.comprehensive_factor_calculator import ComprehensiveFactorCalculator
from config import DATABASE_PATH, FactorConfig, TrainingConfig

# 忽略警告
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 搜索空间定义
# ============================================================================

SEARCH_SPACE = {
    # 动量类因子
    'RSI_PERIOD': [6, 9, 14, 21, 28],
    'ROC_PERIOD': [5, 10, 12, 20, 30],
    'MTM_PERIOD': [5, 10, 12, 20, 30],
    'CMO_PERIOD': [9, 14, 21, 28],
    'STOCHRSI_PERIOD': [9, 14, 21, 28],
    'RVI_PERIOD': [5, 10, 14, 20],
    
    # 趋势类因子
    'MACD_FAST': [5, 10, 12, 15],
    'MACD_SLOW': [20, 26, 30, 40],
    'MACD_SIGNAL': [5, 9, 12],
    'ADX_PERIOD': [7, 14, 21, 28],
    'DMI_PERIOD': [7, 14, 21, 28],
    'AROON_PERIOD': [14, 25, 30, 50],
    'TRIX_PERIOD': [15, 30, 45, 60],
    'MA_RATIO_PERIOD': [5, 10, 20, 30, 60],
    'MA_SLOPE_PERIOD': [5, 10, 20, 30, 60],
    
    # 波动率因子
    'ATR_PERIOD': [5, 10, 14, 20, 30],
    'NATR_PERIOD': [7, 14, 21],
    'BB_PERIOD': [10, 20, 25, 30, 50],
    'BB_STD': [1.5, 2.0, 2.5],
    'CCI_PERIOD': [7, 14, 21, 28],
    'ULCER_PERIOD': [7, 14, 21, 28],
    'PRICE_VAR_PERIOD': [10, 20, 30, 60],
    
    # 成交量因子
    'VOLUME_MA_PERIOD': [5, 10, 20, 30, 60],
    'VOLUME_STD_PERIOD': [5, 10, 20, 30, 60],
    'AMOUNT_MA_PERIOD': [5, 10, 20, 30, 60],
    'AMOUNT_STD_PERIOD': [5, 10, 20, 30, 60],
    'MFI_PERIOD': [7, 14, 21, 28],
    'VR_PERIOD': [13, 26, 39],
    'VROC_PERIOD': [6, 12, 18, 24],
    'VRSI_PERIOD': [3, 6, 9, 12],
    'VMACD_FAST': [6, 12, 15],
    'VMACD_SLOW': [13, 26, 30],
    'VMACD_SIGNAL': [5, 9, 12],
    'ADOSC_FAST': [2, 3, 5],
    'ADOSC_SLOW': [7, 10, 15],
    
    # 摆动指标参数
    'KDJ_N': [5, 9, 14, 21],
    'WILLR_PERIOD': [7, 14, 21, 28],
    'BIAS_PERIOD': [3, 6, 12, 24],
    'PSY_PERIOD': [6, 12, 18, 24],
    'AR_BR_PERIOD': [13, 26, 39],
    'CR_PERIOD': [13, 26, 39],
    
    # K线形态参数
    'BODY_SIZE_THRESHOLD_LARGE': [0.015, 0.02, 0.03],
    'BODY_SIZE_THRESHOLD_SMALL': [0.003, 0.005, 0.008],
    'HAMMER_LOWER_SHADOW_RATIO': [1.5, 2.0, 2.5, 3.0],
    'HAMMER_UPPER_SHADOW_RATIO': [0.5, 1.0, 1.5]
}

# 形态因子对参数的映射（优化评估时只计算相关的因子）
PARAM_TO_FACTOR_MAP = {
    'RSI_PERIOD': ['rsi_'],
    'ROC_PERIOD': ['roc_'],
    'MTM_PERIOD': ['mtm_'],
    'CMO_PERIOD': ['cmo_'],
    'STOCHRSI_PERIOD': ['stochrsi_k', 'stochrsi_d'],
    'RVI_PERIOD': ['rvi_'],
    'MACD_FAST': ['macd'],
    'MACD_SLOW': ['macd'],
    'MACD_SIGNAL': ['macd'],
    'ADX_PERIOD': ['adx_'],
    'DMI_PERIOD': ['plus_di', 'minus_di'],
    'AROON_PERIOD': ['aroon_up', 'aroon_down'],
    'TRIX_PERIOD': ['trix_'],
    'MA_RATIO_PERIOD': ['ma_ratio_'],
    'MA_SLOPE_PERIOD': ['ma_slope_'],
    'ATR_PERIOD': ['atr_'],
    'NATR_PERIOD': ['natr_'],
    'BB_PERIOD': ['bb_width', 'bb_position'],
    'BB_STD': ['bb_width', 'bb_position'],
    'CCI_PERIOD': ['cci_'],
    'ULCER_PERIOD': ['ulcer_'],
    'PRICE_VAR_PERIOD': ['price_var_'],
    'VOLUME_MA_PERIOD': ['vol_ma_'],
    'VOLUME_STD_PERIOD': ['vol_std_'],
    'AMOUNT_MA_PERIOD': ['amount_ma_'],
    'AMOUNT_STD_PERIOD': ['amount_std_'],
    'MFI_PERIOD': ['mfi_'],
    'VR_PERIOD': ['vr_'],
    'VROC_PERIOD': ['vroc_'],
    'VRSI_PERIOD': ['vrsi_'],
    'VMACD_FAST': ['vmacd'],
    'VMACD_SLOW': ['vmacd'],
    'VMACD_SIGNAL': ['vmacd'],
    'ADOSC_FAST': ['adosc'],
    'ADOSC_SLOW': ['adosc'],
    'KDJ_N': ['kdj_k', 'kdj_d', 'kdj_j'],
    'WILLR_PERIOD': ['willr_'],
    'BIAS_PERIOD': ['bias_'],
    'PSY_PERIOD': ['psy_'],
    'AR_BR_PERIOD': ['ar_', 'br_'],
    'CR_PERIOD': ['cr_'],
    'BODY_SIZE_THRESHOLD_LARGE': ['white_candle', 'black_candle'],
    'BODY_SIZE_THRESHOLD_SMALL': ['doji', 'marubozu'],
    'HAMMER_LOWER_SHADOW_RATIO': ['hammer', 'hanging_man', 'shooting_star', 'inverted_hammer'],
    'HAMMER_UPPER_SHADOW_RATIO': ['hammer', 'hanging_man', 'shooting_star', 'inverted_hammer']
}

class FactorTuner:
    """因子超参数调优器"""
    
    def __init__(self, n_stocks: int = 100, years: int = 3, n_jobs: int = -1):
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
        
        # 选取成交量最大的前 N 只股票
        stock_query = f'''
            SELECT code, SUM(volume) as total_vol
            FROM daily_data
            WHERE date >= ?
            GROUP BY code
            ORDER BY total_vol DESC
            LIMIT ?
        '''
        stock_df = pd.read_sql_query(stock_query, conn, params=(start_date, self.n_stocks))
        codes = stock_df['code'].tolist()
        
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
            
            # 5日未来收益率
            df['target'] = df['close'].pct_change(5).shift(-5)
            # 移除 NaN 目标行
            df = df.dropna(subset=['target'])
            
            if not df.empty:
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
        
        # 获取受此参数影响的因子前缀
        target_prefixes = PARAM_TO_FACTOR_MAP.get(param_name, [])
        
        def process_stock(code, data):
            try:
                # 使用临时配置初始化计算器
                calculator = ComprehensiveFactorCalculator(config=TempConfig)
                
                # 获取受此参数影响的因子前缀
                target_prefixes = PARAM_TO_FACTOR_MAP.get(param_name, [])
                
                # 决定计算哪一类因子
                # 如果前缀属于技术指标
                factors = pd.DataFrame(index=data.index)
                
                # 获取所有因子（基础因子，不含特征工程）
                # 为了简化和通用，我们直接计算技术指标和K线形态，这是调优的重点
                tech_factors = calculator.factor_calculator.calculate_all_factors(data)
                candle_factors = calculator.candlestick_calculator.calculate_all_candlestick_patterns(data)
                
                # 合并
                factors = pd.concat([tech_factors, candle_factors], axis=1)
                
                # 如果指定了目标特征前缀，只筛选匹配的列以节省后续 IC 计算
                if target_prefixes:
                    relevant_cols = [c for c in factors.columns if any(c.startswith(p) for p in target_prefixes)]
                    if not relevant_cols:
                        return []
                    factors = factors[relevant_cols]
                
                # 计算与目标的 Rank IC
                stock_ics = []
                # 预先获取目标的数值
                v_target = self.forward_returns[code].values
                
                for col in factors.columns:
                    # 确保没有 NaN
                    v_factor = factors[col].values
                    
                    # 移除无效值
                    mask = np.isfinite(v_factor) & np.isfinite(v_target)
                    if np.sum(mask) > 100: # 至少需要100个有效样本
                        # 检查输入是否为常量，避免 spearmanr 产生 ConstantInputWarning
                        v_f = v_factor[mask]
                        v_t = v_target[mask]
                        if np.ptp(v_f) > 0 and np.ptp(v_t) > 0:
                            ic, _ = stats.spearmanr(v_f, v_t)
                            if np.isfinite(ic):
                                stock_ics.append(abs(ic)) # 使用绝对值，因为方向不重要，相关性才重要
                        else:
                            # 如果是常量，则相关性为 0
                            stock_ics.append(0.0)
                
                return stock_ics
            except Exception as e:
                # print(f"Error processing {code}: {e}")
                return []

        # 并行计算所有股票的 IC
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_stock)(code, data) 
            for code, data in self.stocks_data.items()
        )
        
        # 合并所有 IC 并取平均
        all_ics = [ic for sublist in results for ic in sublist]
        return np.mean(all_ics) if all_ics else 0.0

    def optimize(self, max_rounds: int = 1):
        """采用贪婪坐标下降进行优化"""
        print("\n" + "="*80)
        print("开始因子参数并行调优流程")
        print("="*80)
        
        for round_idx in range(max_rounds):
            print(f"\n[第 {round_idx + 1} 轮优化]")
            
            # 打乱参数顺序以增加鲁棒性
            params_to_opt = list(SEARCH_SPACE.keys())
            np.random.shuffle(params_to_opt)
            
            for param_name in params_to_opt:
                values = SEARCH_SPACE[param_name]
                current_val = self.current_config[param_name]
                
                print(f"正在优化 {param_name} (当前值: {current_val})...")
                
                best_score = -1.0
                best_val = current_val
                
                for val in values:
                    score = self.evaluate_parameter(param_name, val)
                    print(f"  - 测试值: {val:4}, Mean Abs Rank IC: {score:.6f}")
                    
                    if score > best_score:
                        best_score = score
                        best_val = val
                
                if best_val != current_val:
                    print(f"  ★ 发现更好的值: {current_val} -> {best_val} (得分: {best_score:.6f})")
                    self.current_config[param_name] = best_val
                else:
                    print(f"  ✓ 保持原值: {current_val}")

    def update_config_file(self):
        """将优化的参数写回 config/factor_config.py"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'factor_config.py')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        modified_content = content
        
        # 使用正则替换 FactorConfig 类中的属性
        # 这里假设 FactorConfig 类定义在一块集中的区域
        for param, value in self.current_config.items():
            # 匹配形如 PARAM_NAME = VALUE 的行
            # 支持整数、浮点数
            pattern = rf"^(\s+{param}\s*=\s*)([\d\.]+)"
            replacement = rf"\g<1>{value}"
            modified_content = re.sub(pattern, replacement, modified_content, flags=re.MULTILINE)
            
        if modified_content != content:
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
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
