"""
模型优化器 - 特征选择、超参数优化、集成学习优化

功能：
1. 特征选择（多种方法）
2. 特征工程超参数优化
3. 集成学习权重优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    f_regression, mutual_info_regression,
    RFE, RFECV
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score
import warnings
from joblib import Parallel, delayed
import multiprocessing
warnings.filterwarnings('ignore')
from config.factor_config import TrainingConfig

# ---------------------------------------------------------
# 全局优化设置：防止异常情况下大数据矩阵刷屏
# ---------------------------------------------------------
np.set_printoptions(threshold=1000, edgeitems=3, precision=6, suppress=True)


try:
    from scipy.optimize import minimize, differential_evolution
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告: scipy未安装，部分优化功能不可用")

from config.factor_config import OptimizationConfig, TrainingConfig


class FeatureSelector:
    """特征选择器"""
    
    def __init__(self, method: str = 'importance'):
        """
        初始化特征选择器
        
        参数:
            method: 选择方法
                - 'importance': 基于模型特征重要性
                - 'correlation': 基于相关性
                - 'mutual_info': 基于互信息
                - 'rfe': 递归特征消除
                - 'hybrid': 混合方法
        """
        self.method = method
        self.selected_features = []
        self.feature_scores = {}
    
    def select_by_importance(self, model, feature_names: List[str], 
                            threshold: float = None) -> List[str]:
        """基于模型特征重要性选择"""
        if threshold is None:
            threshold = OptimizationConfig.FEATURE_IMPORTANCE_THRESHOLD
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("模型不支持特征重要性")
        
        importances = model.feature_importances_
        self.feature_scores = dict(zip(feature_names, importances))
        
        # 选择重要性超过阈值的特征
        selected = [
            name for name, imp in self.feature_scores.items()
            if imp >= threshold
        ]
        
        print(f"基于重要性选择: {len(selected)}/{len(feature_names)} 个特征")
        return selected
    
    def select_by_correlation(self, X: np.ndarray, y: np.ndarray,
                             feature_names: List[str],
                             threshold: float = None) -> List[str]:
        """基于与目标的相关性选择 (向量化)"""
        if threshold is None:
            threshold = OptimizationConfig.CORRELATION_THRESHOLD_LOW
        
        # 向量化计算相关性
        X_centered = X - np.mean(X, axis=0)
        y_centered = y - np.mean(y)
        
        # 避免除以 0
        std_x = np.std(X, axis=0)
        std_y = np.std(y)
        
        if std_y < 1e-8:
            print("警告: 目标变量 y 几乎没有变化，相关性选择可能无效")
            return feature_names
            
        # 并行/向量化计算相关系数
        # corr = Cov(X, Y) / (std(X) * std(Y))
        correlations = np.abs(np.dot(X_centered.T, y_centered) / (len(y) * std_x * std_y + 1e-8))
        
        self.feature_scores = dict(zip(feature_names, correlations))
        
        # 选择相关性超过阈值的特征
        selected = [
            name for name, corr in self.feature_scores.items()
            if corr >= threshold
        ]
        
        print(f"基于相关性选择: {len(selected)}/{len(feature_names)} 个特征")
        return selected
    
    def select_by_mutual_info(self, X: np.ndarray, y: np.ndarray,
                              feature_names: List[str],
                              n_features: int = 30) -> List[str]:
        """基于互信息选择 (并行优化版)"""
        print(f"执行并行互信息计算 (样本量: {len(y)}, 特征数: {len(feature_names)})...")
        
        # 使用统一的并行 MI 计算逻辑
        mi_scores = self._compute_mi_parallel(X, y, feature_names)
        self.feature_scores = dict(zip(feature_names, mi_scores))
        
        # 选择Top N特征
        sorted_features = sorted(
            self.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        selected = [name for name, _ in sorted_features[:n_features]]
        
        print(f"基于互信息选择: {len(selected)}/{len(feature_names)} 个特征")
        return selected

    def _compute_mi_parallel(self, X: np.ndarray, y: np.ndarray, 
                            feature_names: List[str]) -> np.ndarray:
        """内部通用并行互信息计算逻辑"""
        n_jobs = multiprocessing.cpu_count()
        
        # 数据量过大时科学抽样 (10k-100k 是互信息的黄金平衡点)
        X_mi, y_mi = X, y
        if len(y) > 100000:
            np.random.seed(42)
            indices = np.random.choice(len(y), 100000, replace=False)
            X_mi, y_mi = X[indices], y[indices]
            print(f"  采样计算: 样本量由 {len(y)} 压缩至 100,000 以激活多核高性能计算")

        def compute_mi_chunk(chunk_indices):
            # discrete_features=False 显式声明连续值以加速
            return mutual_info_regression(X_mi[:, chunk_indices], y_mi, discrete_features=False, random_state=42)
        
        # 分块数建议为核心数的 2-4 倍，保证负载均衡
        n_chunks = min(len(feature_names), n_jobs * 2)
        indices_chunks = np.array_split(np.arange(len(feature_names)), n_chunks)
        
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(compute_mi_chunk)(chunk) for chunk in indices_chunks
        )
        return np.concatenate(results)
    
    def select_by_rfe(self, model, X: np.ndarray, y: np.ndarray,
                     feature_names: List[str],
                     n_features: int = 30) -> List[str]:
        """递归特征消除"""
        rfe = RFE(estimator=model, n_features_to_select=n_features, step=5)
        rfe.fit(X, y)
        
        selected = [
            feature_names[i] for i in range(len(feature_names))
            if rfe.support_[i]
        ]
        
        # 记录排名
        self.feature_scores = dict(zip(feature_names, rfe.ranking_))
        
        print(f"递归特征消除选择: {len(selected)}/{len(feature_names)} 个特征")
        return selected
    
    def select_hybrid(self, model, X: np.ndarray, y: np.ndarray,
                     feature_names: List[str],
                     n_features: int = 30) -> List[str]:
        """混合方法 - 并行化综合多种方法的结果 (深度重构)"""
        n_cores = multiprocessing.cpu_count()
        print(f"启动混合特征选择引擎 (分配核心: {n_cores}, 算法: Importance + Correlation + MutualInfo)")
        
        def get_importance():
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            return np.zeros(len(feature_names))

        def get_correlation():
            # 向量化矩阵运算充分利用 CPU 指令集
            X_centered = X - np.mean(X, axis=0)
            y_centered = y - np.mean(y)
            std_x = np.std(X, axis=0)
            std_y = np.std(y)
            return np.abs(np.dot(X_centered.T, y_centered) / (len(y) * std_x * std_y + 1e-8))

        # 1. 首先并行获取三种独立得分
        # 注意：MI 内部已高度并行，这里外层用 Threading 触发以防 MI 内部 Loky 递归锁定
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_imp = executor.submit(get_importance)
            future_corr = executor.submit(get_correlation)
            future_mi = executor.submit(self._compute_mi_parallel, X, y, feature_names)
            
            imp_scores = future_imp.result()
            correlations = future_corr.result()
            mi_scores = future_mi.result()
        
        # 2. 计算各维度排名并实施 Borda 投票
        imp_rank = np.argsort(imp_scores)[::-1]
        mi_rank = np.argsort(mi_scores)[::-1]
        corr_rank = np.argsort(correlations)[::-1]
        
        combined_rank = np.zeros(len(feature_names))
        for i, idx in enumerate(imp_rank): combined_rank[idx] += i
        for i, idx in enumerate(mi_rank): combined_rank[idx] += i
        for i, idx in enumerate(corr_rank): combined_rank[idx] += i
        
        # 选择综合排名最高的特征
        top_indices = np.argsort(combined_rank)[:n_features]
        selected = [feature_names[i] for i in top_indices]
        
        # 记录得分
        self.feature_scores = dict(zip(feature_names, -combined_rank))
        
        print(f"混合方法优化完成: 共分析 {len(feature_names)} 个维度，最终精选出 {len(selected)} 个核心因子")
        return selected
    
    def remove_correlated_features(self, X: np.ndarray, feature_names: List[str],
                                   threshold: float = None) -> List[str]:
        """移除高度相关的特征 (向量化优化)"""
        if threshold is None:
            threshold = OptimizationConfig.CORRELATION_THRESHOLD
        
        # 计算相关系数矩阵
        corr_matrix = np.abs(np.corrcoef(X.T))
        
        # 找到高度相关的特征对
        # 只取上三角矩阵，避免重复比较和自比较
        upper = np.triu(corr_matrix, k=1)
        
        # 找到所有大于阈值的索引
        rows, cols = np.where(upper > threshold)
        
        to_remove = set()
        for row, col in zip(rows, cols):
            # 对于每一对高度相关的特征，移除一个
            f1, f2 = feature_names[row], feature_names[col]
            
            if f1 in to_remove or f2 in to_remove:
                continue
                
            # 如果有分数，保留分数高的（已经在 self.feature_scores 中计算过）
            s1 = self.feature_scores.get(f1, -np.inf)
            s2 = self.feature_scores.get(f2, -np.inf)
            
            if s1 < s2:
                to_remove.add(f1)
            else:
                to_remove.add(f2)
        
        selected = [f for f in feature_names if f not in to_remove]
        print(f"移除高度相关特征: 完成。共移除 {len(to_remove)} 个，保留 {len(selected)} 个")
        return selected
    
    def select(self, model, X: np.ndarray, y: np.ndarray,
              feature_names: List[str], **kwargs) -> List[str]:
        """执行特征选择"""
        if self.method == 'importance':
            selected = self.select_by_importance(
                model, feature_names,
                threshold=kwargs.get('threshold', 0.001)
            )
        elif self.method == 'correlation':
            selected = self.select_by_correlation(
                X, y, feature_names,
                threshold=kwargs.get('threshold', 0.05)
            )
        elif self.method == 'mutual_info':
            selected = self.select_by_mutual_info(
                X, y, feature_names,
                n_features=kwargs.get('n_features', 30)
            )
        elif self.method == 'rfe':
            selected = self.select_by_rfe(
                model, X, y, feature_names,
                n_features=kwargs.get('n_features', 30)
            )
        elif self.method == 'hybrid':
            selected = self.select_hybrid(
                model, X, y, feature_names,
                n_features=kwargs.get('n_features', 30)
            )
        else:
            raise ValueError(f"未知的选择方法: {self.method}")
        
        # 可选：移除高度相关的特征
        if kwargs.get('remove_correlated', True):
            X_selected = X[:, [feature_names.index(f) for f in selected]]
            selected = self.remove_correlated_features(
                X_selected, selected,
                threshold=kwargs.get('corr_threshold', 0.95)
            )
        
        self.selected_features = selected
        return selected


class FeatureEngineeringOptimizer:
    """特征工程超参数优化器"""
    
    def __init__(self):
        """初始化优化器"""
        self.best_params = {}
        self.optimization_history = []
    
    def optimize_factor_periods(self, data: pd.DataFrame,
                                code: str,
                                factor_calculator_class,
                                y_true: pd.DataFrame,
                                model_class=None) -> Dict:
        """
        优化因子计算周期
        
        参数:
            data: 原始数据
            code: 股票代码
            factor_calculator_class: ComprehensiveFactorCalculator 类
            y_true: 真实标签 (含日期索引)
            model_class: 模型类 (可选)
        
        返回:
            最优参数字典
        """
        print("\n优化因子计算周期...")
        print("=" * 80)
        
        # 定义搜索空间 (基于 FactorConfig 的属性)
        search_space = {
            'RSI_PERIOD': [6, 9, 14],
            'MACD_FAST': [10, 12, 15],
            'MACD_SLOW': [20, 26, 30],
            'BB_PERIOD': [15, 20, 25],
            'ATR_PERIOD': [10, 14, 20],
            'BODY_SIZE_THRESHOLD_SMALL': [0.005, 0.01, 0.015],
            'HAMMER_LOWER_SHADOW_RATIO': [1.5, 2.0, 2.5]
        }
        
        # 初始基准参数
        from config.factor_config import FactorConfig
        
        best_score = -np.inf
        best_config_vals = {}
        
        # 简化搜索：依次优化每个参数 (Coordinate Descent 思想)
        current_params = {
            'RSI_PERIOD': FactorConfig.RSI_PERIOD,
            'MACD_FAST': FactorConfig.MACD_FAST,
            'MACD_SLOW': FactorConfig.MACD_SLOW,
            'BB_PERIOD': FactorConfig.BB_PERIOD,
            'ATR_PERIOD': FactorConfig.ATR_PERIOD,
            'BODY_SIZE_THRESHOLD_SMALL': FactorConfig.BODY_SIZE_THRESHOLD_SMALL,
            'HAMMER_LOWER_SHADOW_RATIO': FactorConfig.HAMMER_LOWER_SHADOW_RATIO
        }
        
        def evaluate_config(params):
            # 创建临时配置类
            class TempConfig(FactorConfig):
                pass
            
            for k, v in params.items():
                setattr(TempConfig, k, v)
                
            # 计算因子
            calc = factor_calculator_class(config=TempConfig)
            factors = calc.calculate_all_factors(code, data, apply_feature_engineering=False)
            
            if factors.empty:
                return -1.0
                
            # 计算 IC (信息系数) 作为评分标准
            # 简单起见，取前几个主要因子的绝对相关性均值
            scores = []
            target_cols = [
                f'rsi_{TempConfig.RSI_PERIOD}',
                'macd',
                'bb_width',
                f'atr_{TempConfig.ATR_PERIOD}'
            ]
            for col in target_cols:
                if col in factors.columns:
                    # 对齐
                    common_idx = factors.index.intersection(y_true.index)
                    if len(common_idx) > 10:
                        corr = np.corrcoef(factors.loc[common_idx, col], y_true.loc[common_idx])[0, 1]
                        if np.isfinite(corr):
                            scores.append(abs(corr))
            
            return np.mean(scores) if scores else 0.0

        print("正在进行贪婪参数搜索...")
        for param_name, values in search_space.items():
            param_best_score = -np.inf
            param_best_val = current_params[param_name]
            
            for val in values:
                test_params = current_params.copy()
                test_params[param_name] = val
                score = evaluate_config(test_params)
                
                print(f"  测试 {param_name}={val}, 得分 (Mean IC): {score:.4f}")
                
                if score > param_best_score:
                    param_best_score = score
                    param_best_val = val
            
            current_params[param_name] = param_best_val
            if param_best_score > best_score:
                best_score = param_best_score
                
        print(f"\n最优因子参数: {current_params}")
        print(f"最终得分: {best_score:.4f}")
        
        return current_params


class EnsembleOptimizer:
    """集成学习优化器"""
    
    def __init__(self):
        """初始化优化器"""
        self.best_weights = None
        self.optimization_history = []
    
    def optimize_weights_grid_search(self, models: List,
                                     X_val: np.ndarray,
                                     y_val: np.ndarray) -> List[float]:
        """
        网格搜索优化集成权重
        
        参数:
            models: 模型列表
            X_val: 验证集特征
            y_val: 验证集标签
        
        返回:
            最优权重列表
        """
        print("\n使用网格搜索优化集成权重...")
        print("=" * 80)
        
        n_models = len(models)
        
        # 获取每个模型的预测
        predictions = []
        for i, model in enumerate(models):
            # 内存优化：直接使用 numpy 数组进行预测，由 model 内部处理
            pred = model.predict(X_val)
            predictions.append(pred)
            print(f"模型 {i+1} 预测范围: [{pred.min():.4f}, {pred.max():.4f}]")
        
        predictions = np.array(predictions)
        
        # 准备权重候选集
        resolution = OptimizationConfig.ENSEMBLE_GRID_RESOLUTION if hasattr(OptimizationConfig, 'ENSEMBLE_GRID_RESOLUTION') else 11
        steps = np.linspace(0, 1, resolution)
        
        # 递归生成权重组合 (和为1)
        def generate_weights(n, current_sum=0):
            if n == 1:
                yield [1.0 - current_sum]
            else:
                for w in steps[steps <= 1.0 - current_sum + 1e-9]:
                    for rest in generate_weights(n - 1, current_sum + w):
                        yield [w] + rest

        # 如果模型较多，限制组合数量
        if n_models > 3:
            print(f"模型数量({n_models})较多，限制搜索粒度...")
            steps = np.linspace(0, 1, 6) # 降级粒度
            
        # 准备权重组合列表
        all_weights = list(generate_weights(n_models))
        print(f"开始权重网格搜索 (组合总数: {len(all_weights)}, 验证集: {X_val.shape[0]} 样本)...")
        
        from tqdm import tqdm
        
        # 定义评估函数供并行调用
        def eval_weight(w):
            ensemble_pred = np.average(predictions, axis=0, weights=w)
            return roc_auc_score(y_val, ensemble_pred)
        
        # 使用多核并行加速 (倾向于使用 threading 后端，因为 predictions 很大且 numpy 操作释放 GIL)
        scores = Parallel(n_jobs=-1, backend="threading")(
            delayed(eval_weight)(w) for w in tqdm(all_weights, desc="权重优化进度")
        )
        
        # 记录结果
        for w, score in zip(all_weights, scores):
            if score > best_score:
                best_score = score
                best_weights = w
            
            self.optimization_history.append({
                'weights': w,
                'auc': score
            })
        
        self.best_weights = best_weights
        print(f"\n最优权重: {[f'{w:.3f}' for w in best_weights]}")
        print(f"最优AUC: {best_score:.4f}")
        
        return best_weights
    
    def optimize_weights_scipy(self, models: List,
                              X_val: np.ndarray,
                              y_val: np.ndarray) -> List[float]:
        """
        使用scipy优化集成权重
        
        参数:
            models: 模型列表
            X_val: 验证集特征
            y_val: 验证集标签
        
        返回:
            最优权重列表
        """
        if not HAS_SCIPY:
            print("scipy未安装，使用网格搜索代替")
            return self.optimize_weights_grid_search(models, X_val, y_val)
        
        print("\n使用scipy优化集成权重...")
        print("=" * 80)
        
        n_models = len(models)
        
        # 获取每个模型的预测
        predictions = []
        for model in models:
            # 内存优化：直接使用 numpy 数组
            pred = model.predict(X_val)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # 定义目标函数（最小化负AUC）
        def objective(weights):
            # 归一化权重
            weights = np.abs(weights)
            weights = weights / weights.sum()
            
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            auc = roc_auc_score(y_val, ensemble_pred)
            return -auc  # 最小化负AUC
        
        # 约束：权重和为1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # 边界：每个权重在[0, 1]之间
        bounds = [(0, 1) for _ in range(n_models)]
        
        # 初始权重：平均
        initial_weights = np.array([1.0 / n_models] * n_models)
        
        # 优化
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        best_weights = result.x
        best_weights = best_weights / best_weights.sum()  # 归一化
        best_score = -result.fun
        
        self.best_weights = best_weights.tolist()
        print(f"\n最优权重: {[f'{w:.3f}' for w in best_weights]}")
        print(f"最优AUC: {best_score:.4f}")
        
        return self.best_weights
    
    def optimize_stacking(self, models: List,
                         X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Stacking集成优化
        
        参数:
            models: 基模型列表
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
        
        返回:
            Stacking模型配置
        """
        print("\n优化Stacking集成...")
        print("=" * 80)
        
        from sklearn.linear_model import LogisticRegression
        
        # 生成元特征
        meta_features_train = []
        meta_features_val = []
        
        for i, model in enumerate(models):
            # 内存优化：直接使用 numpy 数组
            pred_train = model.predict(X_train)
            pred_val = model.predict(X_val)
            
            meta_features_train.append(pred_train)
            meta_features_val.append(pred_val)
        
        meta_features_train = np.column_stack(meta_features_train)
        meta_features_val = np.column_stack(meta_features_val)
        
        # 训练元模型
        meta_model = LogisticRegression(random_state=42)
        meta_model.fit(meta_features_train, y_train)
        
        # 评估
        pred_val = meta_model.predict_proba(meta_features_val)[:, 1]
        auc = roc_auc_score(y_val, pred_val)
        
        print(f"Stacking AUC: {auc:.4f}")
        print(f"元模型权重: {meta_model.coef_[0]}")
        
        return {
            'meta_model': meta_model,
            'auc': auc,
            'weights': meta_model.coef_[0].tolist()
        }


class ModelOptimizer:
    """模型优化器 - 整合所有优化功能"""
    
    def __init__(self):
        """初始化优化器"""
        self.feature_selector = None
        self.fe_optimizer = FeatureEngineeringOptimizer()
        self.ensemble_optimizer = EnsembleOptimizer()
        self.optimization_results = {}
    
    def run_full_optimization(self, models: Dict, X: np.ndarray, y: np.ndarray,
                             feature_names: List[str],
                             X_val: np.ndarray = None,
                             y_val: np.ndarray = None,
                             dates: np.ndarray = None,
                             returns: np.ndarray = None) -> Dict:
        """
        运行完整优化流程
        
        参数:
            models: 模型字典 {'xgboost': model1, 'lightgbm': model2, ...}
            X: 训练集特征 (或全量特征，取决于是否传了 X_val)
            y: 训练集标签
            feature_names: 特征名称列表
            X_val: 验证集特征（可选）
            y_val: 验证集标签（可选）
            dates: 日期序列 (用于时间序列分割和排序分组)
            returns: 收益率序列 (用于精准评估)
        
        返回:
            优化结果字典
        """
        print("\n" + "=" * 80)
        print("开始深度模型优化流程")
        print("=" * 80)
        
        # 0. 统一数据集划分：优先使用时间序列分割
        if X_val is None or y_val is None:
            if dates is not None:
                # 遵循 TrainingConfig 比例进行时间序列切分
                split_idx = int(len(X) * TrainingConfig.TRAIN_TEST_SPLIT)
                # 寻找日期边界，防止同一天的样本被切分到不同集合
                split_date = dates[split_idx]
                actual_split_idx = np.searchsorted(dates, split_date, side='left')
                
                print(f"执行时间序列分割: 训练集截止 {dates[actual_split_idx-1]}, 验证集起始 {dates[actual_split_idx]}")
                X_train, X_val = X[:actual_split_idx], X[actual_split_idx:]
                y_train, y_val = y[:actual_split_idx], y[actual_split_idx:]
                
                # 同步切分辅助变量
                dates_val = dates[actual_split_idx:]
                returns_val = returns[actual_split_idx:] if returns is not None else None
            else:
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                dates_val = None
                returns_val = None
            
            print(f"数据划分完成: 训练样本 {len(X_train)}, 验证样本 {len(X_val)}")
        else:
            X_train, X_val = X, X_val
            y_train, y_val = y, y_val
            dates_val = None
            returns_val = None

        # 1. 特征选择 (在训练集或全量数据上执行)
        print("\n[步骤 1/3] 执行核心特征精选")
        print("-" * 80)
        
        self.feature_selector = FeatureSelector(method='hybrid')
        
        # 【关键修复】确保加载模型的重要性分值能正确对应到当前的 factor_names
        # 即使加载的模型特征顺序不同，通过此映射逻辑可以精准提取贡献度
        base_model_wrapper = list(models.values())[-1] # 使用最后一个模型(通常是LightGBM)作为基准
        
        if hasattr(base_model_wrapper, 'feature_importance') and base_model_wrapper.feature_importance:
            print(f"  正在从基准模型 {base_model_wrapper.model_type} 提取并对齐特征贡献度...")
            scores_dict = base_model_wrapper.feature_importance
            aligned_scores = np.array([scores_dict.get(name, 0.0) for name in feature_names])
            
            # 创建模拟模型对象传递分值
            class MockModel:
                def __init__(self, imp): self.feature_importances_ = imp
            eval_model = MockModel(aligned_scores)
        else:
            eval_model = base_model_wrapper.model

        selected_features = self.feature_selector.select(
            eval_model, X_train, y_train, feature_names,
            n_features=OptimizationConfig.N_FEATURES_TO_SELECT,
            remove_correlated=True,
            corr_threshold=0.95
        )
        
        self.optimization_results['selected_features'] = selected_features
        self.optimization_results['feature_scores'] = self.feature_selector.feature_scores
        
        # 2. 模型重定义与全量训练
        print(f"\n[步骤 2/3] 使用精选的 {len(selected_features)} 个特征重塑模型")
        print("-" * 80)
        
        selected_indices = [feature_names.index(f) for f in selected_features]
        # 注意：此处我们需要通过全量 X, y 重新训练，因为模型内部会处理 validation_split
        X_selected = X[:, selected_indices]
        X_val_selected = X_val[:, selected_indices]
        
        retrained_models = {}
        for name, model in models.items():
            print(f"\n正在重构并训练 {name.upper()}...")
            
            # 为训练准备必要的 Kwargs
            train_kwargs = {
                'validation_split': 1.0 - TrainingConfig.TRAIN_TEST_SPLIT,
                'feature_names': selected_features,
                'returns': returns,
                'dates': dates,
                'split_idx': actual_split_idx # 强制对齐分割索引
            }
            
            # 【关键修复】针对 Ranking 任务通过 dates 动态还原分组信息
            if getattr(model, 'task', '') == 'ranking' and dates is not None:
                # 重新计算分组 (Query Counts)
                train_dates_sub = dates[:actual_split_idx]
                val_dates_sub = dates[actual_split_idx:]
                
                _, group_train = np.unique(train_dates_sub, return_counts=True)
                _, group_val = np.unique(val_dates_sub, return_counts=True)
                
                train_kwargs['group'] = group_train
                train_kwargs['eval_group'] = group_val
                print(f"  ✓ 已同步排序任务分组信息 (Train Groups: {len(group_train)}, Val Groups: {len(group_val)})")

            # 执行重新训练
            model.train(X_selected, y, **train_kwargs)
            retrained_models[name] = model
        
        # 3. 集成学习权重优化
        print("\n[步骤 3/3] 集成学习权重深度搜索")
        print("-" * 80)
        
        model_list = list(retrained_models.values())
        
        # 【关键修复】predict 接口需要 DataFrame 类型以进行特征过滤，解决 IndexError
        X_val_df = pd.DataFrame(X_val_selected, columns=selected_features)
        
        # 方法1: 网格搜索 (使用更新后的验证集指标)
        best_weights_grid = self.ensemble_optimizer.optimize_weights_grid_search(
            model_list, X_val_df, y_val
        )
        
        # 方法2: scipy优化（如果可用）
        if HAS_SCIPY:
            best_weights_scipy = self.ensemble_optimizer.optimize_weights_scipy(
                model_list, X_val_selected, y_val
            )
            self.optimization_results['ensemble_weights_scipy'] = best_weights_scipy
        
        self.optimization_results['ensemble_weights_grid'] = best_weights_grid
        self.optimization_results['model_names'] = list(retrained_models.keys())
        
        # 方法3: Stacking（可选）
        try:
            stacking_result = self.ensemble_optimizer.optimize_stacking(
                model_list, X_selected, y, X_val_selected, y_val
            )
            self.optimization_results['stacking'] = stacking_result
        except Exception as e:
            print(f"Stacking优化失败: {e}")
        
        # 总结
        print("\n" + "=" * 80)
        print("优化完成")
        print("=" * 80)
        print(f"选择的特征数: {len(selected_features)}")
        print(f"集成权重: {best_weights_grid}")
        
        return self.optimization_results
    
    def save_results(self, filepath: str = 'models/optimization_results.json'):
        """保存优化结果"""
        import json
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 转换为可序列化的格式
        results_to_save = {
            'selected_features': self.optimization_results.get('selected_features', []),
            'ensemble_weights': self.optimization_results.get('ensemble_weights_grid', []),
            'model_names': self.optimization_results.get('model_names', []),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"\n优化结果已保存到: {filepath}")
