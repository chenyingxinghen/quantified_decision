"""
机器学习因子模型
使用机器学习算法学习量化因子与未来价格走势的关系

支持的模型：
1. XGBoost - 梯度提升树
2. LightGBM - 轻量级梯度提升
3. Random Forest - 随机森林
4. Neural Network - 神经网络

特征：
- 自动特征工程
- 因子重要性分析
- 在线学习支持
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pickle
import os
from datetime import datetime

# 机器学习库
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("警告: XGBoost未安装，部分功能不可用")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("警告: LightGBM未安装，部分功能不可用")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from config.factor_config import TrainingConfig, ModelConfig


class MLFactorModel:
    """机器学习因子模型"""
    
    def __init__(self, model_type: str = 'xgboost', task: str = 'classification'):
        """
        初始化模型
        
        参数:
            model_type: 模型类型 ('xgboost', 'lightgbm', 'random_forest')
            task: 任务类型 ('classification' 或 'regression')
        """
        self.model_type = model_type
        self.task = task  # 'classification', 'regression', 'ranking'
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = {}
        self.is_trained = False
        self.optimal_threshold = 0.5  # 默认阈值
        
        # 初始化模型
        self._init_model()
    
    def _init_model(self):
        """初始化机器学习模型"""
        # 获取模型参数
        model_params = ModelConfig.get_model_params(self.model_type)
        
        # 确保 random_state 和 n_jobs 被设置 (如果配置中没有)
        if 'random_state' not in model_params:
            model_params['random_state'] = 42
        if 'n_jobs' not in model_params:
            model_params['n_jobs'] = -1
            
        if self.model_type == 'xgboost' and HAS_XGB:
            if self.task == 'classification':
                self.model = xgb.XGBClassifier(**model_params)
            elif self.task == 'ranking':
                # 为 Ranking 设置合适的 objective
                if 'objective' not in model_params or model_params['objective'] == 'binary:logistic':
                    model_params['objective'] = 'rank:ndcg'
                if 'eval_metric' in model_params and model_params['eval_metric'] == 'auc':
                    model_params['eval_metric'] = 'ndcg'
                self.model = xgb.XGBRanker(**model_params)
            else:
                # 回归模型的客观函数不同
                if 'objective' in model_params:
                    model_params['objective'] = 'reg:squarederror'
                if 'eval_metric' in model_params:
                    del model_params['eval_metric'] # 回归通常不需要 auc
                    
                self.model = xgb.XGBRegressor(**model_params)
        
        elif self.model_type == 'lightgbm' and HAS_LGB:
            if self.task == 'classification':
                self.model = lgb.LGBMClassifier(**model_params)
            elif self.task == 'ranking':
                if 'objective' not in model_params or model_params['objective'] == 'binary':
                    model_params['objective'] = 'lambdarank'
                if 'metric' in model_params and model_params['metric'] == 'auc':
                    model_params['metric'] = 'ndcg'
                self.model = lgb.LGBMRanker(**model_params)
            else:
                if 'objective' in model_params:
                    model_params['objective'] = 'regression'
                if 'metric' in model_params:
                    del model_params['metric']
                    
                self.model = lgb.LGBMRegressor(**model_params)
        
        elif self.model_type == 'random_forest':
            if self.task == 'classification':
                self.model = RandomForestClassifier(**model_params)
            else:
                self.model = RandomForestRegressor(**model_params)
        
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def prepare_training_data(self, factors_df: pd.DataFrame, price_data: pd.DataFrame,
                            forward_days: int = None, threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        参数:
            factors_df: 因子数据DataFrame
            price_data: 价格数据DataFrame (包含close列)
            forward_days: 预测未来N天的收益（默认使用配置）
            threshold: 分类阈值（涨幅超过此值为正样本，默认使用配置）
        
        返回:
            X: 特征矩阵
            y: 标签向量
        """
        # 使用配置中的默认值
        if forward_days is None:
            forward_days = TrainingConfig.FUTURE_DAYS
        if threshold is None:
            threshold = TrainingConfig.RETURN_THRESHOLD
        
        # 计算未来收益率
        future_returns = price_data['close'].pct_change(forward_days).shift(-forward_days)
        
        # 对齐数据
        valid_idx = ~(factors_df.isna().any(axis=1) | future_returns.isna())
        X = factors_df[valid_idx].values
        
        if self.task == 'classification':
            # 分类任务：涨幅超过阈值为1，否则为0
            y = (future_returns[valid_idx] > threshold).astype(int).values
        elif self.task == 'ranking':
            # 排序任务：通常直接使用连续的收益率作为相关性得分
            y = future_returns[valid_idx].values
        else:
            # 回归任务：直接预测收益率
            y = future_returns[valid_idx].values
        
        # 保存特征名称
        self.feature_names = factors_df.columns.tolist()
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             validation_split: float = 0.2, use_time_series_split: bool = True,
             feature_names: List[str] = None,
             sample_weight: Optional[np.ndarray] = None,
             groups: Optional[np.ndarray] = None) -> Dict:
        """
        训练模型
        
        参数:
            X: 特征矩阵
            y: 标签向量
            validation_split: 验证集比例
            use_time_series_split: 是否使用时间序列分割
            feature_names: 特征名称列表
            sample_weight: 样本权重
            groups: 分组信息（用于排序任务，表示每个查询包含的样本数）
        
        返回:
            训练结果字典
        """
        print(f"开始训练 {self.model_type} 模型...")
        print(f"样本数量: {len(X)}, 特征数量: {X.shape[1]}")
        
        # 数据验证和清理（在StandardScaler之前）
        X = X.astype(np.float64)
        
        # 检查并替换NaN值
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            print(f"  警告: 发现 {nan_count} 个NaN值，进行替换")
            X = np.nan_to_num(X, nan=0.0)
        
        # 检查并替换无穷大值
        if np.isinf(X).any():
            inf_count = np.isinf(X).sum()
            print(f"  警告: 发现 {inf_count} 个无穷大值，进行替换")
            X[np.isinf(X)] = 0.0
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 转换为 DataFrame 以保留特征名称
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_scaled.shape[1])]
        
        # 保存特征名称
        self.feature_names = feature_names
        
        X_df = pd.DataFrame(X_scaled, columns=feature_names)
        
        # 分割训练集和验证集
        if use_time_series_split:
            # 时间序列分割（保持时间顺序）
            split_idx = int(len(X_df) * (1 - validation_split))
            X_train, X_val = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            w_train = sample_weight[:split_idx] if sample_weight is not None else None
            w_val = sample_weight[split_idx:] if sample_weight is not None else None
            # 注意：groups 在排序中通常是每个时间点的样本数，分割需要特殊处理
            g_train = None
            g_val = None
            if groups is not None:
                # 假设 groups 已经按照 X 的顺序计算好了
                # 排序任务通常需要按组分割
                cumulative_groups = np.cumsum(groups)
                train_groups_idx = np.searchsorted(cumulative_groups, split_idx)
                g_train = groups[:train_groups_idx]
                g_val = groups[train_groups_idx:]
                # 修正分割点的样本对齐
                split_idx = int(cumulative_groups[train_groups_idx-1]) if train_groups_idx > 0 else 0
                X_train, X_val = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                w_train = sample_weight[:split_idx] if sample_weight is not None else None
                w_val = sample_weight[split_idx:] if sample_weight is not None else None
        else:
            # 随机分割 (对于排序任务不推荐随机分割)
            X_train, X_val, y_train, y_val = train_test_split(
                X_df, y, test_size=validation_split, random_state=42
            )
            w_train, w_val = None, None
            g_train, g_val = None, None
            if sample_weight is not None or groups is not None:
                print("  警告: 非时间序列分割下未处理权重和分组")
        
        print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")
        
        # 训练模型
        if self.model_type == 'xgboost':
            fit_params = {
                'eval_set': [(X_val, y_val)],
                'verbose': False
            }
            if w_train is not None:
                fit_params['sample_weight'] = w_train
                fit_params['sample_weight_eval_set'] = [w_val]
            if g_train is not None:
                fit_params['group'] = g_train
                fit_params['eval_group'] = [g_val]
                
            self.model.fit(X_train, y_train, **fit_params)
            
        elif self.model_type == 'lightgbm':
            from lightgbm import early_stopping
            fit_params = {
                'eval_set': [(X_val, y_val)],
                'callbacks': [early_stopping(stopping_rounds=20)],
                'eval_metric': 'ndcg' if self.task == 'ranking' else ('auc' if self.task == 'classification' else 'rmse')
            }
            if w_train is not None:
                fit_params['sample_weight'] = w_train
            if g_train is not None:
                fit_params['group'] = g_train
                fit_params['eval_group'] = [g_val]

            self.model.fit(X_train, y_train, **fit_params)
        else:
            fit_params = {}
            if w_train is not None:
                fit_params['sample_weight'] = w_train
            self.model.fit(X_train, y_train, **fit_params)
        
        # 评估模型
        train_metrics = self._evaluate(X_train, y_train, "训练集")
        val_metrics = self._evaluate(X_val, y_val, "验证集")
        
        # 对于分类任务，优化决策阈值以提高召回率
        if self.task == 'classification':
            self._optimize_threshold(X_val, y_val)
        
        # 计算特征重要性
        self._calculate_feature_importance()
        
        self.is_trained = True
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importance': self.feature_importance
        }
    
    def _evaluate(self, X: Any, y: np.ndarray, dataset_name: str) -> Dict:
        """评估模型性能"""
        # 确保输入是 DataFrame 以包含特征名称，避免警告
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
            
        y_pred = self.model.predict(X)
        
        if self.task == 'classification':
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0),
                'auc': roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0
            }
            
            print(f"\n{dataset_name}性能:")
            print(f"  准确率: {metrics['accuracy']:.4f}")
            print(f"  精确率: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  F1分数: {metrics['f1']:.4f}")
            print(f"  AUC: {metrics['auc']:.4f}")
        
        elif self.task == 'ranking':
            # 排序任务评估简单化：计算相关性
            correlation = pd.Series(y).corr(pd.Series(y_pred))
            metrics = {'correlation': correlation}
            print(f"\n{dataset_name}性能:")
            print(f"  预测值与收益率相关性: {correlation:.4f}")
        
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2': r2_score(y, y_pred)
            }
            
            print(f"\n{dataset_name}性能:")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def _calculate_feature_importance(self):
        """计算特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.feature_importance = dict(zip(self.feature_names, importances))
            
            # 排序并打印前20个重要特征
            sorted_features = sorted(self.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            print("\n前20个重要因子:")
            for i, (name, importance) in enumerate(sorted_features[:20], 1):
                print(f"  {i}. {name}: {importance:.4f}")
    
    def _optimize_threshold(self, X_val: Any, y_val: np.ndarray):
        """
        优化分类阈值以提高召回率
        
        参数:
            X_val: 验证集特征
            y_val: 验证集标签
        """
        # 确保输入是 DataFrame
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val, columns=self.feature_names)
            
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # 尝试不同的阈值，找到最优的 F1 分数
        best_f1 = 0
        best_threshold = 0.5
        
        print("\n阈值优化:")
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if len(np.unique(y_pred)) > 1:  # 确保有正负样本
                f1 = f1_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                precision = precision_score(y_val, y_pred, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                
                if threshold % 0.1 < 0.01:  # 每隔0.1打印一次
                    print(f"  阈值 {threshold:.2f}: F1={f1:.4f}, 精确率={precision:.4f}, 召回率={recall:.4f}")
        
        self.optimal_threshold = best_threshold
        print(f"\n最优阈值: {best_threshold:.2f} (F1分数: {best_f1:.4f})")
    
    def predict(self, factors: pd.DataFrame) -> np.ndarray:
        """
        预测
        
        参数:
            factors: 因子DataFrame
        
        返回:
            预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 检查特征是否存在
        missing_features = [f for f in self.feature_names if f not in factors.columns]
        if missing_features:
            raise ValueError(f"缺少特征: {missing_features}")
        
        # 按照训练时的特征顺序选择
        X = factors[self.feature_names].values
        
        # 检查是否有NaN值
        if np.isnan(X).any():
            raise ValueError("特征中包含NaN值")
        
        X_scaled = self.scaler.transform(X)
        
        # 转换为 DataFrame 以保留特征名称，避免 LightGBM/XGBoost 的有效特征名称警告
        X_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        if self.task == 'classification':
            # 返回正类概率
            return self.model.predict_proba(X_df)[:, 1]
        else:
            # 返回预测值
            return self.model.predict(X_df)
    
    def predict_signal(self, factors: pd.DataFrame, threshold: float = None) -> Dict:
        """
        生成交易信号
        
        参数:
            factors: 因子DataFrame（单个样本）
            threshold: 分类阈值（如果为None则使用优化后的阈值）
        
        返回:
            信号字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        if threshold is None:
            threshold = self.optimal_threshold
        
        try:
            prediction = self.predict(factors)
        except Exception as e:
            # 预测失败，返回hold信号
            print(f"预测失败: {e}")
            return {
                'signal': 'hold',
                'confidence': 0,
                'prediction': 0
            }
        
        if self.task == 'classification':
            prob = prediction[0]
            signal = 'buy' if prob >= threshold else 'hold'
            confidence = prob * 100
        else:
            expected_return = prediction[0]
            signal = 'buy' if expected_return > TrainingConfig.EXPECTED_RETURN_THRESHOLD else 'hold'
            confidence = min(abs(expected_return) * 1000, 100)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'prediction': float(prediction[0])
        }
    
    def get_top_factors(self, n: int = 20) -> List[Tuple[str, float]]:
        """获取最重要的N个因子"""
        if not self.feature_importance:
            return []
        
        sorted_features = sorted(self.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        return sorted_features[:n]
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'task': self.task,
            'is_trained': self.is_trained,
            'optimal_threshold': self.optimal_threshold
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data['model_type']
        self.task = model_data['task']
        self.is_trained = model_data['is_trained']
        self.optimal_threshold = model_data.get('optimal_threshold', 0.5)
        
        print(f"模型已从 {filepath} 加载")
        print(f"模型类型: {self.model_type}, 任务: {self.task}")
        print(f"特征数量: {len(self.feature_names)}")
        print(f"最优阈值: {self.optimal_threshold:.2f}")


