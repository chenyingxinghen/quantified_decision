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
    RFE, RFECV
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize, differential_evolution
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("警告: scipy未安装，部分优化功能不可用")

from config.factor_config import OptimizationConfig


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
        """基于与目标的相关性选择"""
        if threshold is None:
            threshold = OptimizationConfig.CORRELATION_THRESHOLD_LOW
        
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr))
        
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
        """基于互信息选择"""
        mi_scores = mutual_info_classif(X, y, random_state=42)
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
        """混合方法 - 综合多种方法的结果"""
        print("使用混合方法进行特征选择...")
        
        # 方法1: 特征重要性
        if hasattr(model, 'feature_importances_'):
            imp_scores = model.feature_importances_
            imp_rank = np.argsort(imp_scores)[::-1]
        else:
            imp_rank = np.arange(len(feature_names))
        
        # 方法2: 互信息
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_rank = np.argsort(mi_scores)[::-1]
        
        # 方法3: 相关性
        correlations = []
        for i in range(X.shape[1]):
            corr = abs(np.corrcoef(X[:, i], y)[0, 1])
            correlations.append(corr)
        corr_rank = np.argsort(correlations)[::-1]
        
        # 综合排名（Borda count）
        combined_rank = np.zeros(len(feature_names))
        for i, idx in enumerate(imp_rank):
            combined_rank[idx] += i
        for i, idx in enumerate(mi_rank):
            combined_rank[idx] += i
        for i, idx in enumerate(corr_rank):
            combined_rank[idx] += i
        
        # 选择综合排名最高的特征
        top_indices = np.argsort(combined_rank)[:n_features]
        selected = [feature_names[i] for i in top_indices]
        
        # 记录综合得分
        self.feature_scores = dict(zip(feature_names, -combined_rank))
        
        print(f"混合方法选择: {len(selected)}/{len(feature_names)} 个特征")
        return selected
    
    def remove_correlated_features(self, X: np.ndarray, feature_names: List[str],
                                   threshold: float = None) -> List[str]:
        """移除高度相关的特征"""
        if threshold is None:
            threshold = OptimizationConfig.CORRELATION_THRESHOLD
        
        corr_matrix = np.corrcoef(X.T)
        
        to_remove = set()
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if abs(corr_matrix[i, j]) > threshold:
                    # 保留重要性更高的特征
                    if feature_names[i] in self.feature_scores and \
                       feature_names[j] in self.feature_scores:
                        if self.feature_scores[feature_names[i]] < self.feature_scores[feature_names[j]]:
                            to_remove.add(feature_names[i])
                        else:
                            to_remove.add(feature_names[j])
                    else:
                        to_remove.add(feature_names[j])
        
        selected = [f for f in feature_names if f not in to_remove]
        print(f"移除高度相关特征: {len(to_remove)} 个")
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
            # 创建DataFrame，使用模型的特征名称
            X_val_df = pd.DataFrame(X_val, columns=model.feature_names)
            pred = model.predict(X_val_df)
            predictions.append(pred)
            print(f"模型 {i+1} 预测范围: [{pred.min():.4f}, {pred.max():.4f}]")
        
        predictions = np.array(predictions)
        
        # 网格搜索权重
        best_score = -np.inf
        best_weights = None
        
        if n_models == 2:
            # 两个模型：搜索 w1 从 0 到 1
            for w1 in np.linspace(0, 1, 21):
                weights = [w1, 1 - w1]
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
                score = roc_auc_score(y_val, ensemble_pred)
                
                if score > best_score:
                    best_score = score
                    best_weights = weights
                
                self.optimization_history.append({
                    'weights': weights,
                    'auc': score
                })
        
        elif n_models == 3:
            # 三个模型：搜索 w1, w2（w3 = 1 - w1 - w2）
            for w1 in np.linspace(0, 1, 11):
                for w2 in np.linspace(0, 1 - w1, 11):
                    w3 = 1 - w1 - w2
                    weights = [w1, w2, w3]
                    ensemble_pred = np.average(predictions, axis=0, weights=weights)
                    score = roc_auc_score(y_val, ensemble_pred)
                    
                    if score > best_score:
                        best_score = score
                        best_weights = weights
                    
                    self.optimization_history.append({
                        'weights': weights,
                        'auc': score
                    })
        
        else:
            # 多个模型：使用简化的搜索
            print("模型数量较多，使用简化搜索...")
            # 平均权重作为基准
            weights = [1.0 / n_models] * n_models
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            best_score = roc_auc_score(y_val, ensemble_pred)
            best_weights = weights
        
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
            # 创建DataFrame，使用模型的特征名称
            X_val_df = pd.DataFrame(X_val, columns=model.feature_names)
            pred = model.predict(X_val_df)
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
            # 创建DataFrame，使用模型的特征名称
            X_train_df = pd.DataFrame(X_train, columns=model.feature_names)
            X_val_df = pd.DataFrame(X_val, columns=model.feature_names)
            
            pred_train = model.predict(X_train_df)
            pred_val = model.predict(X_val_df)
            
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
                             y_val: np.ndarray = None) -> Dict:
        """
        运行完整优化流程
        
        参数:
            models: 模型字典 {'xgboost': model1, 'lightgbm': model2, ...}
            X: 训练集特征
            y: 训练集标签
            feature_names: 特征名称列表
            X_val: 验证集特征（可选）
            y_val: 验证集标签（可选）
        
        返回:
            优化结果字典
        """
        print("\n" + "=" * 80)
        print("开始模型优化流程")
        print("=" * 80)
        
        # 如果没有验证集，从训练集中分割
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print(f"训练集: {len(X)}, 验证集: {len(X_val)}")
        
        # 1. 特征选择
        print("\n[步骤 1/3] 特征选择")
        print("-" * 80)
        
        # 使用混合方法
        self.feature_selector = FeatureSelector(method='hybrid')
        
        # 使用第一个模型进行特征选择
        first_model = list(models.values())[0]
        selected_features = self.feature_selector.select(
            first_model.model, X, y, feature_names,
            n_features=30,
            remove_correlated=True,
            corr_threshold=0.95
        )
        
        self.optimization_results['selected_features'] = selected_features
        self.optimization_results['feature_scores'] = self.feature_selector.feature_scores
        
        # 使用选择的特征重新训练模型
        selected_indices = [feature_names.index(f) for f in selected_features]
        X_selected = X[:, selected_indices]
        X_val_selected = X_val[:, selected_indices]
        
        print(f"\n使用选择的 {len(selected_features)} 个特征重新训练模型...")
        
        retrained_models = {}
        for name, model in models.items():
            print(f"\n重新训练 {name}...")
            # 重新训练模型使用选择的特征
            model.train(X_selected, y, feature_names=selected_features)
            retrained_models[name] = model
        
        # 2. 特征工程超参数优化（可选，耗时较长）
        print("\n[步骤 2/3] 特征工程超参数优化")
        print("-" * 80)
        print("跳过（需要重新计算因子，耗时较长）")
        # best_fe_params = self.fe_optimizer.optimize_factor_periods(...)
        # self.optimization_results['best_fe_params'] = best_fe_params
        
        # 3. 集成学习权重优化
        print("\n[步骤 3/3] 集成学习权重优化")
        print("-" * 80)
        
        model_list = list(retrained_models.values())
        
        # 方法1: 网格搜索
        best_weights_grid = self.ensemble_optimizer.optimize_weights_grid_search(
            model_list, X_val_selected, y_val
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
