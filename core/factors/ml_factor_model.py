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
            try:
                if self.task == 'regression':
                    # 回归方案优化：使用 reg:logistic 确保输出在 0-1 之间
                    if 'objective' not in model_params: model_params['objective'] = 'reg:logistic'
                    self.model = xgb.XGBRegressor(**model_params)
                else:
                    self.model = xgb.XGBClassifier(**model_params)
            except Exception as e:
                # GPU 初始化失败回退到 CPU
                if 'gpu' in str(model_params.get('tree_method', '')) or 'cuda' in str(model_params.get('device', '')):
                    print(f"  [WARNING] XGBoost GPU 初始化失败: {e}，正在尝试回退到 CPU...")
                    model_params.pop('tree_method', None)
                    model_params.pop('device', None)
                    model_params.pop('predictor', None)
                    if self.task == 'regression':
                        self.model = xgb.XGBRegressor(**model_params)
                    else:
                        self.model = xgb.XGBClassifier(**model_params)
                else:
                    raise e
        elif self.model_type == 'lightgbm' and HAS_LGB:
            try:
                if self.task == 'ranking':
                    # 排序模式
                    if 'objective' not in model_params:
                        model_params['objective'] = 'lambdarank'
                    
                    if 'label_gain' not in model_params:
                        model_params['label_gain'] = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511]
                    
                    if 'lambdarank_truncation_level' not in model_params:
                        model_params['lambdarank_truncation_level'] = 15
                    
                    self.early_stopping_rounds = model_params.pop('early_stopping_rounds', 50)
                    self.model = lgb.LGBMRanker(**model_params)
                elif self.task == 'regression':
                    self.model = lgb.LGBMRegressor(**model_params)
                else:
                    self.model = lgb.LGBMClassifier(**model_params)
            except Exception as e:
                # GPU 初始化失败回退到 CPU
                if model_params.get('device') == 'gpu':
                    print(f"  [WARNING] LightGBM GPU 初始化失败: {e}，正在尝试回退到 CPU...")
                    model_params['device'] = 'cpu'
                    if self.task == 'ranking':
                        self.model = lgb.LGBMRanker(**model_params)
                    elif self.task == 'regression':
                        self.model = lgb.LGBMRegressor(**model_params)
                    else:
                        self.model = lgb.LGBMClassifier(**model_params)
                else:
                    raise e
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

        # ---------------------------------------------------------------------
        # 3. 特征缩放 (RobustScaler)
        # ---------------------------------------------------------------------
        # 修复问题3: 特征缩放应该在横截面归一化之后进行
        # 注意：由于在 train_models 中已经进行了横截面归一化，
        # print(f"  [INFO] 模型训练准备：正在进行特征缩放 (RobustScaler)...")
        # X_train_raw = self.scaler.fit_transform(X_train_raw).astype(np.float32)
        # X_val_raw = self.scaler.transform(X_val_raw).astype(np.float32)

        # ---------------------------------------------------------------------
        # 4. 样本打乱 (Shuffle) —— 解决分批训练导致的分布漂移关键
        # ---------------------------------------------------------------------
        # 风险：原始数据是按日期严格排序的，分批次训练时 Batch 0 全是旧数据。
        # 解决：在训练集内部进行随机打乱，使每个 Batch 都能代表全量分布。
        # 注意：仅针对回归/分类任务 (XGBoost)，排序任务 (LightGBM) 需要在 group 内部有序（这里维持现状）。
        if self.task != 'ranking':
            print(f"  [INFO] 正在对训练样本进行随机打乱，以消除分批训练的分布漂移...")
            shuffle_idx = np.arange(len(X_train_raw))
            np.random.seed(42)
            np.random.shuffle(shuffle_idx)
            X_train_raw = X_train_raw[shuffle_idx]
            y_train = y_train[shuffle_idx]
            if w_train is not None: w_train = w_train[shuffle_idx]
            if r_train is not None: r_train = r_train[shuffle_idx]
            if dates_train is not None: dates_train = dates_train[shuffle_idx]

        # 5. 内存优化与分批训练
        # 如果启用内存优化且在 GPU 上，使用 XGBoost 的 DataIter 或 LightGBM 的 Dataset 优化
        use_gpu = TrainingConfig.USE_GPU
        mem_efficient = getattr(TrainingConfig, 'MEMORY_EFFICIENT', True)
        batch_size = getattr(TrainingConfig, 'GPU_BATCH_SIZE', 1000000)

        # ---------------------------------------------------------------------
        # 情况 A: XGBoost 分批训练 (DataIter)
        # ---------------------------------------------------------------------
        if self.model_type == 'xgboost' and mem_efficient and len(X_train_raw) > batch_size:
            print(f"  [INFO] XGBoost 启动分批训练模式 (样本数: {len(X_train_raw)}, Batch: {batch_size})")
            
            # 使用更加通用的 DataIter 基类 (兼容不同版本)
            BaseDataIter = getattr(xgb, 'DataIter', xgb.core.DataIter)
            
            class XGBDataIter(BaseDataIter):
                def __init__(self, X, y, w=None, b_size=1000000):
                    self.X = X
                    self.y = y
                    self.w = w
                    self.b_size = b_size
                    self.it = 0
                    super().__init__(cache_prefix=None)

                def next(self, input_data):
                    if self.it >= len(self.X):
                        return 0
                    end = min(self.it + self.b_size, len(self.X))
                    
                    batch_data = self.X[self.it:end]
                    batch_label = self.y[self.it:end]
                    
                    # 关键修复：不要把 None 作为 weight 传入，以免触发 XGBoost 的参数冲突检查
                    if self.w is not None:
                        batch_weight = self.w[self.it:end]
                        input_data(data=batch_data, label=batch_label, weight=batch_weight)
                    else:
                        input_data(data=batch_data, label=batch_label)
                    
                    self.it = end
                    return 1

                def reset(self):
                    self.it = 0

            # 创建训练集迭代器
            it = XGBDataIter(X_train_raw, y_train, w_train, batch_size)
            
            # 创建训练集和验证集的 DMatrix
            # 为避免某些版本中 feature_names 触发 DataIter 冲突检查，在此处先不传入 feature_names
            dtrain = xgb.QuantileDMatrix(it)
            dval = xgb.DMatrix(X_val_raw, label=y_val)
            
            # 统一设置特征名
            dtrain.feature_names = self.feature_names
            dval.feature_names = self.feature_names
            
            params = ModelConfig.get_model_params('xgboost')
            # 兼容 scikit-learn 参数名到 native 参数名
            xgb_params = {
                'tree_method': params.get('tree_method', 'hist'),
                'device': params.get('device', 'cuda'),
                'learning_rate': params.get('learning_rate', 0.02),
                'max_depth': params.get('max_depth', 6),
                'min_child_weight': params.get('min_child_weight', 1),
                'subsample': params.get('subsample', 1),
                'colsample_bytree': params.get('colsample_bytree', 1),
                'reg_alpha': params.get('reg_alpha', 0),
                'reg_lambda': params.get('reg_lambda', 1),
                'objective': params.get('objective', 'reg:logistic'),
                'eval_metric': params.get('eval_metric', 'auc'),
                'nthread': params.get('n_jobs', -1)
            }
            
            # 使用 native train 接口
            self.model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=params.get('n_estimators', 1000),
                evals=[(dval, 'validation')],
                early_stopping_rounds=params.get('early_stopping_rounds', 50),
                verbose_eval=params.get('verbosity', 1) > 0
            )
            
            # 内存回收
            del dtrain, dval
            X_train = X_train_raw # 仅用于后续评估
            X_val = X_val_raw

        # ---------------------------------------------------------------------
        # 情况 B: LightGBM 内存优化
        # ---------------------------------------------------------------------
        elif self.model_type == 'lightgbm' and mem_efficient:
            X_train = X_train_raw
            X_val = X_val_raw
            
            # LightGBM 不直接通过 DataIter 分批，但可以通过 Dataset 构造函数级优化
            # 这里我们不再使用 scikit-learn API，而是转向更加省内存的 Native API 或优化参数
            fit_params = {'sample_weight': w_train}
            if self.task == 'ranking':
                if 'group' in kwargs: fit_params['group'] = kwargs['group']
                if 'eval_group' in kwargs: 
                    fit_params.update({'eval_set': [(X_val, y_val)], 'eval_group': [kwargs['eval_group']]})
                else:
                    fit_params['eval_set'] = [(X_val, y_val)]
            else:
                fit_params['eval_set'] = [(X_val, y_val)]

            from lightgbm import early_stopping, log_evaluation
            es_rounds = getattr(self, 'early_stopping_rounds', 100)
            callbacks = [early_stopping(stopping_rounds=es_rounds, first_metric_only=True)]
            
            if self.task == 'ranking':
                callbacks.append(log_evaluation(period=50))
                if 'sample_weight' in fit_params: del fit_params['sample_weight']

            # 注意：LGBM 即使是用 numpy array 训练，内部也会转 Dataset。
            # 开启 histogram_pool_size (在 config 中已添加) 是 6G 显存的关键。
            self.model.fit(X_train, y_train, callbacks=callbacks, feature_name=self.feature_names, **fit_params)

        # ---------------------------------------------------------------------
        # 情况 C: 标准流程 (不符合分批条件或非优化模式)
        # ---------------------------------------------------------------------
        else:
            # 尽量使用 numpy 直接训练，避免 DataFrame 拷贝
            if len(X_train_raw) > 500000:
                X_train = X_train_raw
                X_val = X_val_raw
                feature_name_param = self.feature_names
            else:
                X_train = pd.DataFrame(X_train_raw, columns=self.feature_names)
                X_val = pd.DataFrame(X_val_raw, columns=self.feature_names)
                feature_name_param = 'auto'

            fit_params = {'sample_weight': w_train}
            if self.task == 'ranking':
                if 'group' in kwargs: fit_params['group'] = kwargs['group']
                if 'eval_group' in kwargs: 
                    fit_params.update({'eval_set': [(X_val, y_val)], 'eval_group': [kwargs['eval_group']]})
                else:
                    fit_params['eval_set'] = [(X_val, y_val)]
            else:
                fit_params['eval_set'] = [(X_val, y_val)]

            if self.model_type == 'xgboost':
                self.model.fit(X_train, y_train, verbose=False, **fit_params)
            elif self.model_type == 'lightgbm':
                from lightgbm import early_stopping, log_evaluation
                es_rounds = getattr(self, 'early_stopping_rounds', 100)
                callbacks = [early_stopping(stopping_rounds=es_rounds, first_metric_only=True)]
                if self.task == 'ranking':
                    callbacks.append(log_evaluation(period=50))
                    if 'sample_weight' in fit_params: del fit_params['sample_weight']
                self.model.fit(X_train, y_train, callbacks=callbacks, **fit_params)
            else:
                self.model.fit(X_train, y_train, sample_weight=w_train)
            
        # 4. 后处理与评估
        self._calculate_feature_importance()
        self.is_trained = True
        
        import gc
        gc.collect()
        
        return {
            'train_metrics': self._evaluate(X_train, y_train, "训练集", returns=r_train, dates=dates_train, sample_ratio=0.05),
            'val_metrics': self._evaluate(X_val, y_val, "验证集", returns=r_val, dates=dates_val, sample_ratio=0.3)
        }

    def _get_predict_proba(self, X: Any) -> np.ndarray:
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X, columns=self.feature_names)
        
        # 对于分类器，返回正类概率
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        
        # 对于回归器或排序器，预测值即为分数/概率
        if self.model_type == 'xgboost' and HAS_XGB:
            # 兼容原生 Booster 和 Scikit-learn 接口
            is_booster = isinstance(self.model, xgb.Booster)
            
            if is_booster:
                # 确保针对 Booster 的 DMatrix 包含正确的特征名
                # 如果 X 是 DataFrame 且列名正确，DMatrix 会自动提取；
                # 如果是 numpy，则显式指定。
                if isinstance(X, pd.DataFrame):
                    dmat = xgb.DMatrix(X)
                else:
                    dmat = xgb.DMatrix(X, feature_names=self.feature_names)
                preds = self.model.predict(dmat)
            else:
                # scikit-learn 包装类
                device = self.model.get_params().get('device', 'cpu')
                is_gpu = device == 'cuda' or 'gpu' in str(self.model.get_params().get('tree_method', ''))
                
                if is_gpu:
                    # GPU 模式下的预测加速与防止显存碎片
                    dmat = xgb.DMatrix(X, feature_names=self.feature_names) if not isinstance(X, pd.DataFrame) else xgb.DMatrix(X)
                    preds = self.model.get_booster().predict(dmat)
                else:
                    preds = self.model.predict(X)
        else:
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
        # 核心改进：优先使用 soft label y 作为评估基准 (y 已经过板块中性化排名处理)
        # 这样评估出的 IC 才是真实的“在同板块内选出龙头”的能力
        reference = y if y is not None else (returns if returns is not None else y_prob)
        
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
        elif isinstance(self.model, xgb.Booster):
            # 处理原生 Booster
            score = self.model.get_score(importance_type='gain')
            
            # 兼容性处理：检查返回的是特征名还是默认索引 (f0, f1...)
            if score:
                first_key = list(score.keys())[0]
                if first_key.startswith('f') and first_key[1:].isdigit() and first_key not in self.feature_names:
                    # 如果返回的是 f0, f1... 则手动映射回特征名
                    self.feature_importance = {}
                    for k, v in score.items():
                        idx = int(k[1:])
                        if idx < len(self.feature_names):
                            self.feature_importance[self.feature_names[idx]] = v
                else:
                    # 返回的是实际特征名
                    self.feature_importance = {name: score.get(name, 0) for name in self.feature_names}
            else:
                self.feature_importance = {name: 0 for name in self.feature_names}

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
        # 预测阶段必须使用训练阶段拟合好的 scaler
        X_scaled = self.scaler.transform(X).astype(np.float32)
        return self._get_predict_proba(X_scaled)

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