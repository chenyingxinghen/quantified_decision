"""
机器学习因子模型
使用机器学习算法 learn 量化因子与未来价格走势的关系
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

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import spearmanr, rankdata
from config.factor_config import TrainingConfig, ModelConfig

class MLFactorModel:
    """机器学习因子模型"""
    
    def __init__(self, model_type: str = 'xgboost', task: str = 'classification'):
        self.model_type = model_type
        
        # 固定任务类型：LGBM 使用排序，XGBoost 使用回归（拟合软化标签）
        if self.model_type == 'lightgbm':
            self.task = 'ranking'
        elif self.model_type == 'xgboost':
            self.task = 'regression'
        else:
            self.task = task
            
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names = []
        self.feature_importance = {}
        self.is_trained = False
        self.optimal_threshold = 0.5
        self._init_model()
    
    def _init_model(self):
        model_params = ModelConfig.get_model_params(self.model_type)
        
        if 'random_state' not in model_params: model_params['random_state'] = 42
        if 'n_jobs' not in model_params: model_params['n_jobs'] = -1
            
        if self.model_type == 'xgboost' and HAS_XGB:
            if self.task == 'regression':
                # 回归方案优化：使用 reg:logistic 确保输出在 0-1 之间
                if 'objective' not in model_params: model_params['objective'] = 'reg:logistic'
                self.model = xgb.XGBRegressor(**model_params)
            else:
                self.model = xgb.XGBClassifier(**model_params)
        elif self.model_type == 'lightgbm' and HAS_LGB:
            if self.task == 'ranking':
                # 排序模式
                # 确保使用排序相关的 objective，如果配置中没有则设为默认的 lambdarank
                if 'objective' not in model_params:
                    model_params['objective'] = 'lambdarank'
                
                # 关键Ranking参数配置 (如果配置未提供则使用默认值)
                # label_gain: 每个相关等级的增益. 增大Top等级的权重，支持 0-9 共 10 个等级
                if 'label_gain' not in model_params:
                    model_params['label_gain'] = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511]
                
                # 扩大关注的排名范围，改善整体排序质量(IC)
                if 'lambdarank_truncation_level' not in model_params:
                    model_params['lambdarank_truncation_level'] = 15
                
                # 调整 early_stopping_rounds 为更保守的值，Ranking 任务通常需要更多轮次收敛
                self.early_stopping_rounds = model_params.pop('early_stopping_rounds', 50)
                
                self.model = lgb.LGBMRanker(**model_params)
            elif self.task == 'regression':
                self.model = lgb.LGBMRegressor(**model_params)
            else:
                self.model = lgb.LGBMClassifier(**model_params)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def train(self, X: np.ndarray, y: np.ndarray, 
             validation_split: float = 0.2, feature_names: List[str] = None,
             sample_weight: Optional[np.ndarray] = None,
             returns: Optional[np.ndarray] = None,
             use_time_series_split: bool = True,
             **kwargs) -> Dict:
        """
        训练模型
        """
        # 1. 预处理
        # 优化：由 float64 改为 float32，避免内存翻倍。且预先在 prepare_dataset 中已完成填充，此处仅做校验。
        if not X.flags.c_contiguous: X = np.ascontiguousarray(X)
        X = np.nan_to_num(X.astype(np.float32), copy=False, nan=0.0)
        
        # 标签清理：确保没有 NaN 或 Inf (针对 XGBoost 报错)
        if np.isnan(y).any() or np.isinf(y).any():
            print(f"  [WARNING] 检测到标签中包含 NaN/Inf，正在自动填充为 0")
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
        self.feature_names = feature_names if feature_names else [f'f{i}' for i in range(X.shape[1])]

        # 如果是排序任务，使用相关度分数进行分档
        if self.task == 'ranking':
            # 优先使用 y 作为相关度分数进行分档（y 可能包含路径质量等复合指标，比原始收益率更稳健）
            # 只有当 y 已经是离散整数时，才跳过分档；否则无论是 y (软标签) 还是 returns 都要分档
            has_discrete_y = np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) > 2
            
            if not has_discrete_y:
                # 确定分档目标：优先用 y (Soft Label)，没 y 用 returns
                is_using_path_score = not np.issubdtype(y.dtype, np.integer)
                target_score = y if is_using_path_score else returns
                
                if target_score is not None:
                    src_name = "路径质量评分(Y)" if is_using_path_score else "原始收益率"
                    print(f"  [INFO] 排序任务：使用{src_name}进行组内百分位分档")
                    
                    n_bins = len(ModelConfig.LIGHTGBM_PARAMS.get('label_gain', []))
                    if n_bins == 0: n_bins = 21
                    thresholds = np.linspace(1.0/n_bins, 1.0 - 1.0/n_bins, n_bins - 1)
                    
                    if 'group' in kwargs:
                        # 按组内百分位分档
                        group_sizes = kwargs['group']
                        if 'eval_group' in kwargs:
                            all_group_sizes = np.concatenate([group_sizes, kwargs['eval_group']])
                        else:
                            all_group_sizes = group_sizes
                        y_ranked = np.zeros_like(target_score, dtype=np.int32)
                        offset = 0
                        for g_size in all_group_sizes:
                            g_size = int(g_size)
                            g_scores = target_score[offset:offset + g_size]
                            if len(g_scores) > 0:
                                pct_rank = rankdata(g_scores, method='average') / len(g_scores)
                                labels = np.zeros(len(g_scores), dtype=np.int32)
                                for i, thresh in enumerate(thresholds):
                                    labels[pct_rank > thresh] = i + 1
                                y_ranked[offset:offset + g_size] = labels
                            offset += g_size
                        y = y_ranked
                    else:
                        raise ValueError('ranking分组失败')
                    
                    # print(f"  相关度标签分布 (0-{n_bins-1}): {dict(zip(*np.unique(y, return_counts=True)))}")
                else:
                    print(f"  [WARNING] 排序任务缺少分档目标(y/returns)，将直接使用原始标签")
        
        
        # 2. 划分数据集
        if use_time_series_split:
            # 排序任务：使用对齐到日期边界的 split_idx，避免拆分同一天的 group
            if 'split_idx' in kwargs:
                split_idx = kwargs['split_idx']
            else:
                split_idx = int(len(X) * (1 - validation_split))
            X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            w_train = sample_weight[:split_idx] if sample_weight is not None else None
            r_val = returns[split_idx:] if returns is not None else None
            r_train = returns[:split_idx] if returns is not None else None
            # 提取 dates 用于按组评估
            dates = kwargs.get('dates', None)
            dates_train = dates[:split_idx] if dates is not None else None
            dates_val = dates[split_idx:] if dates is not None else None
        else:
            X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
            w_train = None
            r_val = None
            r_train = None
            dates_train = None
            dates_val = None

        # 3. 标准化
        # 已在数据准备阶段做了按日横截面 Z-Score 或者百分位处理，
        # 此处如果再加全局 RobustScaler 会破坏横向对比信息且多余，因此移除原先的标准化步骤
        X_train = pd.DataFrame(X_train_raw, columns=self.feature_names)
        X_val = pd.DataFrame(X_val_raw, columns=self.feature_names)
        
        # 4. 模型拟合
        # 改进：排序任务支持样本权重，有助于通过权重惩罚（如涨停、停牌）引导模型避开不可买入样本
        fit_params = {'sample_weight': w_train}
        
        # 处理排序任务的分组信息
        if self.task == 'ranking':
            if 'group' in kwargs: fit_params['group'] = kwargs['group']
            if 'eval_group' in kwargs: 
                # LGBMRanker 的 eval_set 需要对应的 group
                eval_params = {'eval_set': [(X_val, y_val)], 'eval_group': [kwargs['eval_group']]}
                fit_params.update(eval_params)
            else:
                fit_params['eval_set'] = [(X_val, y_val)]
        else:
            fit_params['eval_set'] = [(X_val, y_val)]

        if self.model_type == 'xgboost':
            self.model.fit(X_train, y_train, verbose=False, **fit_params)
        elif self.model_type == 'lightgbm':
            from lightgbm import early_stopping, log_evaluation
            # 提高 Ranking 任务的早停容忍度，增加稳定性
            es_rounds = getattr(self, 'early_stopping_rounds', 100)
            callbacks = [early_stopping(stopping_rounds=es_rounds, first_metric_only=True)]
            
            if self.task == 'ranking':
                # 排序任务：打印训练进度
                callbacks.append(log_evaluation(period=50))
                # 严重 BUG 修复: LGBM lambdarank 的 sample_weight 会被解析为 group 权重。
                if 'sample_weight' in fit_params:
                    del fit_params['sample_weight']

            self.model.fit(X_train, y_train, callbacks=callbacks, **fit_params)
        else:
            self.model.fit(X_train, y_train, sample_weight=w_train)
            
        self._calculate_feature_importance()
        self.is_trained = True
        
        # 5. 评估
        return {
            'train_metrics': self._evaluate(X_train, y_train, "训练集", returns=r_train, dates=dates_train, sample_ratio=0.1),
            'val_metrics': self._evaluate(X_val, y_val, "验证集", returns=r_val, dates=dates_val, sample_ratio=0.5)
        }

    def _get_predict_proba(self, X: Any) -> np.ndarray:
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X, columns=self.feature_names)
        
        # 对于分类器，返回正类概率
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        
        # 对于回归器或排序器，预测值即为分数/概率
        preds = self.model.predict(X)
            
        # 自动纠正任务类型：如果模型是 LGBMRanker 但任务标记不是 ranking，强制按 ranking 处理
        current_task = self.task
        if 'LGBMRanker' in str(type(self.model)) and current_task != 'ranking':
            current_task = 'ranking'

        if current_task == 'ranking':
            # 排序模型 (LGBMRanker) 的输出是相对分数 (Raw Score)
            return 1.0 / (1.0 + np.exp(-preds * 1.0))
        
        if current_task == 'regression':
            # 回归任务（如软标签）
            return np.clip(preds, 0.0, 1.0)
        
        return preds

    def _evaluate(self, X: Any, y: np.ndarray, dataset_name: str, returns: np.ndarray = None, 
                 dates: np.ndarray = None, sample_ratio: float = 1.0) -> Dict:
        y_prob = self._get_predict_proba(X)
        
        # 1. 基础指标计算 (针对类别/回归)
        # y 可能包含软标签，因此先进行二元化处理
        y_true_bin = (y >= 0.5).astype(int)
        y_pred = (y_prob >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true_bin, y_pred) if len(np.unique(y_true_bin)) > 1 else 1.0,
            'auc': roc_auc_score(y_true_bin, y_prob) if len(np.unique(y_true_bin)) > 1 else 0.5,
            'precision': precision_score(y_true_bin, y_pred, zero_division=0),
            'recall': recall_score(y_true_bin, y_pred, zero_division=0),
            'f1': f1_score(y_true_bin, y_pred, zero_division=0)
        }
        
        # 2. 核心选股指标 (Top-N 精度 & 按组 Rank IC)
        # 无论什么任务，只要提供了日期信息，都按交易日分组评估，这反映了真实的选股能力
        reference = returns if returns is not None else y.astype(float)
        
        if dates is not None:
            # 获取日期分组
            unique_dates = np.unique(dates)
            
            # 抽样逻辑：减少非验证集的计算开销
            if sample_ratio < 1.0:
                n_sample = max(1, int(len(unique_dates) * sample_ratio))
                unique_dates = np.random.choice(unique_dates, size=n_sample, replace=False)
                eval_type = f"随机抽样 {sample_ratio:.0%}"
            else:
                eval_type = "全量"

            rank_ics = []
            top1_hits = []
            top5_hits = []
            
            for d in unique_dates:
                mask = dates == d
                if mask.sum() < 10:  # 样本太少的日期跳过
                    continue
                
                g_prob = y_prob[mask]
                g_ref = reference[mask]
                
                # A. 组内 Rank IC
                if len(np.unique(g_ref)) > 1:
                    ic, _ = spearmanr(g_prob, g_ref)
                    if not np.isnan(ic):
                        rank_ics.append(ic)
                
                # B. Top-1 精度：模型选的第 1 名是否在组内真实表现前 5%
                if len(g_ref) >= 10:
                    top1_idx = np.argmax(g_prob)
                    top5pct_threshold = np.percentile(g_ref, 95)
                    top1_hits.append(1.0 if g_ref[top1_idx] >= top5pct_threshold else 0.0)
                    
                    # C. Top-5 精度：模型选的前 5 名中，有多少在组内前 20%
                    n_top = min(5, len(g_ref))
                    top5_idx = np.argsort(g_prob)[-n_top:]
                    top20pct_threshold = np.percentile(g_ref, 80)
                    top5_precision = np.mean(g_ref[top5_idx] >= top20pct_threshold)
                    top5_hits.append(top5_precision)
            
            metrics['rank_ic'] = np.mean(rank_ics) if rank_ics else 0.0
            metrics['rank_ic_std'] = np.std(rank_ics) if rank_ics else 0.0
            metrics['top1_precision'] = np.mean(top1_hits) if top1_hits else 0.0
            metrics['top5_precision'] = np.mean(top5_hits) if top5_hits else 0.0
            
            # 辅助统计：预测区分度
            prob_std = np.std(y_prob)
            unique_probs = len(np.unique(np.round(y_prob, 6)))
            
            print(f"  [{dataset_name}] {eval_type}评估 ({len(unique_dates)} 个交易日):")
            print(f"    预测区分度: Std={prob_std:.4f}, Unique={unique_probs}")
            print(f"    Rank IC: {metrics['rank_ic']:.4f} ± {metrics['rank_ic_std']:.4f}")
            print(f"    Top-1 精度(命中前5%): {metrics['top1_precision']:.2%}")
            print(f"    Top-5 精度(命中前20%): {metrics['top5_precision']:.2%}")
        else:
            # 没有日期信息，退化为全局计算
            metrics['rank_ic'], _ = spearmanr(y_prob, reference)
            metrics['top1_precision'] = 0.0
            metrics['top5_precision'] = 0.0
            
            prob_std = np.std(y_prob)
            unique_probs = len(np.unique(np.round(y_prob, 6)))
            
            print(f"  [{dataset_name}] 全局评估 (注意: 无日期分组，评价选股能力可能不准确):")
            print(f"    预测区分度: Std={prob_std:.4f}, Unique={unique_probs}")
            print(f"    全局 Rank IC: {metrics['rank_ic']:.4f}")
        
        return metrics

    def _calculate_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))

    def get_top_factors(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        获取重要性最高的 Top-N 因子
        """
        if not self.feature_importance:
            return []
        # 按重要性降序排列
        sorted_factors = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_factors[:n]

    def predict(self, factors: pd.DataFrame) -> np.ndarray:
        if not self.is_trained: raise ValueError("未训练")
        X = np.nan_to_num(factors[self.feature_names].values, nan=0.0)
        return self._get_predict_proba(X)

    def predict_signal(self, factors: pd.DataFrame, threshold: float = 0.5) -> Dict:
        prob = self.predict(factors)[0]
        return {'signal': 'buy' if prob >= threshold else 'hold', 'confidence': float(prob * 100), 'prediction': float(prob)}

    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f: pickle.dump(self.__dict__, f)

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f: self.__dict__.update(pickle.load(f))

class EnsembleFactorModel:
    """
    集成因子模型
    将多个 MLFactorModel 的预测结果进行加权平均
    """
    def __init__(self, models: List[MLFactorModel], weights: List[float]):
        if not models:
            raise ValueError("models 列表不能为空")
        if len(models) != len(weights):
            raise ValueError("模型的数量与权重的数量不匹配")
        
        # 归一化权重
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        self.models = models
        # Ensemble model is considered trained if all its constituent models are trained
        self.is_trained = all(model.is_trained for model in models) 
        
        print(f"集成模型初始化完成，包含 {len(models)} 个模型，权重: {self.weights}")

    def predict(self, factors: pd.DataFrame) -> np.ndarray:
        """
        获取集成模型的预测结果
        """
        if not self.is_trained:
            raise ValueError("集成模型中的所有子模型必须先经过训练")
            
        all_predictions = []
        for model in self.models:
            model_factors = factors[model.feature_names] if hasattr(model, 'feature_names') and model.feature_names else factors
            all_predictions.append(model.predict(model_factors))
            
        # 加权平均
        ensemble_pred = np.average(np.array(all_predictions), axis=0, weights=self.weights)
        return ensemble_pred

    def save_model(self, filepath: str):
        """
        保存集成模型（保存子模型和权重）
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # 将集成模型自身的状态保存到一个字典中
        ensemble_state = {
            'weights': self.weights,
            'model_states': [model.__dict__ for model in self.models]
        }
        with open(filepath, 'wb') as f:
            pickle.dump(ensemble_state, f)
        print(f"集成模型已保存到: {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        """
        加载集成模型
        """
        with open(filepath, 'rb') as f:
            ensemble_state = pickle.load(f)
            
        # 从保存的状态中重构子模型
        models = []
        for model_state in ensemble_state['model_states']:
            reconstructed_model = MLFactorModel(
                model_type=model_state.get('model_type', 'xgboost'),
                task=model_state.get('task', 'classification')
            )
            reconstructed_model.__dict__.update(model_state)
            models.append(reconstructed_model)
            
        return cls(models, ensemble_state['weights'])