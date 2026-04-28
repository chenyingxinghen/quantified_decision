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
        self._evals_result = {}  # 训练曲线，用于过拟合诊断
        self._init_model()
    
    def _init_model(self):
        model_params = ModelConfig.get_model_params(self.model_type)
        
        if 'random_state' not in model_params: model_params['random_state'] = 42
        if 'n_jobs' not in model_params: model_params['n_jobs'] = -1
            
        if self.model_type == 'xgboost' and HAS_XGB:
            try:
                if self.task == 'regression':
                    # 回归方案：使用 reg:squarederror 拟合连续排名标签
                    if 'objective' not in model_params: model_params['objective'] = 'reg:squarederror'
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

        注意：当 use_time_series_split=True 且 kwargs 中传入 split_idx 时，
        validation_split 参数会被忽略，实际划分由 split_idx 控制。
        split_idx 由外层 train_models 按日期边界对齐后传入，确保同一天的样本不被拆分。
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
            # 优先判定：如果 y 是浮点数且不全是整数，则视为需要分档的软标签或原始分值
            # 由于外层训练器（train_ml_model.py）已将 returns 作为 y 传入，
            # 这里的 y 实际上就是排序的标准。
            
            # 检查 y 是否已经是离散整数标签 (0, 1, 2...)
            # 如果 y 是 int 类型且非二元，则视为已经处理好的离散相关度分值
            has_discrete_y = np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) > 2
            
            if not has_discrete_y:
                # 确定分档源：外层已将排序锚点（可能是 returns 或 path_score）赋值给 y
                # 我们这里统一将其视为 target_score
                target_score = y
                
                # 判定来源名称用于日志输出
                # 如果 target_score 和 returns 极其接近，则认为是原始收益率
                is_returns = False
                if returns is not None and len(returns) == len(target_score):
                    # 容差检查
                    is_returns = np.allclose(target_score[:100], returns[:100], atol=1e-5)
                
                src_name = "原始收益率" if is_returns else "路径质量/逻辑分数"
                print(f"  [INFO] 排序任务：使用 {src_name} 进行组内百分位分档 (Labels: 0-{len(ModelConfig.LIGHTGBM_PARAMS.get('label_gain', []))-1})")
                
                if target_score is not None:
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
                                # 计算组内排名
                                pct_rank = rankdata(g_scores, method='average') / (len(g_scores) + 1)
                                labels = np.zeros(len(g_scores), dtype=np.int32)
                                for i, thresh in enumerate(thresholds):
                                    labels[pct_rank > thresh] = i + 1
                                y_ranked[offset:offset + g_size] = labels
                            offset += g_size
                        y = y_ranked
                    else:
                        raise ValueError('ranking任务必须提供 group 分组信息')
                else:
                    print(f"  [WARNING] 排序任务缺少分档目标值")
        
        
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
        # ---------------------------------------------------------------------
        # 情况 A: XGBoost 分批训练 (DataIter)
        # ---------------------------------------------------------------------
        # 优化：通过 DataIter 分批向 GPU 供弹，核心在于 batch_size 需足够大以遮掩 PCIe 延迟
        batch_size = getattr(TrainingConfig, 'GPU_BATCH_SIZE', 100000)
        if self.model_type == 'xgboost' and mem_efficient and len(X_train_raw) > batch_size:
            print(f"  [INFO] XGBoost 启动 QuantileDMatrix 训练模式 (样本数: {len(X_train_raw)})")
            
            # 优化：直接传整个数组给 QuantileDMatrix，避免 DataIter 多次遍历的队列等待瓶颈。
            # QuantileDMatrix 内部会自动分块处理，GPU/CPU 利用率更均衡。
            params = ModelConfig.get_model_params('xgboost')
            xgb_params = {
                'tree_method': params.get('tree_method', 'hist'),
                'device':           params.get('device', 'cuda'),
                'learning_rate':    params.get('learning_rate', 0.03),
                'max_depth':        params.get('max_depth', 6),
                'min_child_weight': params.get('min_child_weight', 50),
                'subsample':        params.get('subsample', 0.8),
                'colsample_bytree': params.get('colsample_bytree', 0.8),
                'colsample_bylevel':params.get('colsample_bylevel', 0.8),
                'gamma':            params.get('gamma', 0.0),
                'reg_alpha':        params.get('reg_alpha', 0.1),
                'reg_lambda':       params.get('reg_lambda', 1.0),
                'objective':        params.get('objective', 'reg:squarederror'),
                'eval_metric':      params.get('eval_metric', 'rmse'),
                'nthread':          params.get('n_jobs', -1),
                'verbosity':        params.get('verbosity', 0),
            }

            dtrain = xgb.QuantileDMatrix(
                X_train_raw, label=y_train, weight=w_train,
                feature_names=self.feature_names,
            )
            dval = xgb.QuantileDMatrix(
                X_val_raw, label=y_val,
                feature_names=self.feature_names,
                ref=dtrain,
            )

            # 用小样本训练集做过拟合监控，避免全量推理
            monitor_size = min(50000, len(X_train_raw))
            monitor_idx = np.random.choice(len(X_train_raw), monitor_size, replace=False)
            dtrain_monitor = xgb.QuantileDMatrix(
                X_train_raw[monitor_idx], label=y_train[monitor_idx],
                feature_names=self.feature_names, ref=dtrain,
            )

            evals_result = {}
            verbose_eval = 50
            self.model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=params.get('n_estimators', 3000),
                evals=[(dtrain_monitor, 'train_monitor'), (dval, 'validation')],
                evals_result=evals_result,
                early_stopping_rounds=params.get('early_stopping_rounds', 20),
                verbose_eval=verbose_eval,
            )
            self._evals_result = evals_result

            # 内存回收
            del dtrain, dval, dtrain_monitor
            X_train = X_train_raw  # 仅用于后续评估
            X_val = X_val_raw

        # ---------------------------------------------------------------------
        # 情况 B: LightGBM 内存优化
        # ---------------------------------------------------------------------
        elif self.model_type == 'lightgbm' and mem_efficient:
            X_train = X_train_raw
            X_val = X_val_raw
            
            fit_params = {'sample_weight': w_train}
            # 抽样训练集用于过拟合监控（仅非ranking任务，ranking需要group信息无法简单抽样）
            monitor_size = min(50000, len(X_train))
            monitor_idx = np.random.choice(len(X_train), monitor_size, replace=False)
            X_train_monitor = X_train[monitor_idx]
            y_train_monitor = y_train[monitor_idx]
            if self.task == 'ranking':
                if 'group' in kwargs: fit_params['group'] = kwargs['group']
                if 'eval_group' in kwargs:
                    fit_params.update({
                        'eval_set': [(X_val, y_val)],
                        'eval_group': [kwargs['eval_group']],
                    })
                else:
                    fit_params['eval_set'] = [(X_val, y_val)]
            else:
                fit_params.update({
                    'eval_set': [(X_train_monitor, y_train_monitor), (X_val, y_val)],
                    'eval_names': ['train_monitor', 'valid'],
                })

            from lightgbm import early_stopping, log_evaluation, record_evaluation
            es_rounds = getattr(self, 'early_stopping_rounds', 100)
            lgb_evals_result = {}
            callbacks = [
                early_stopping(stopping_rounds=es_rounds, first_metric_only=True),
                log_evaluation(50),
                record_evaluation(lgb_evals_result),
            ]
            
            self.model.fit(X_train, y_train, callbacks=callbacks, feature_name=self.feature_names, **fit_params)
            self._evals_result = lgb_evals_result

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
                from lightgbm import early_stopping, log_evaluation, record_evaluation
                es_rounds = getattr(self, 'early_stopping_rounds', 100)
                lgb_evals_result = {}
                if self.task != 'ranking':
                    monitor_size = min(50000, len(X_train))
                    monitor_idx = np.random.choice(len(X_train), monitor_size, replace=False)
                    X_train_monitor = X_train[monitor_idx] if isinstance(X_train, np.ndarray) else X_train.iloc[monitor_idx]
                    y_train_monitor = y_train[monitor_idx]
                    fit_params.update({
                        'eval_set': [(X_train_monitor, y_train_monitor), (X_val, y_val)],
                        'eval_names': ['train_monitor', 'valid'],
                    })
                callbacks = [
                    early_stopping(stopping_rounds=es_rounds, first_metric_only=True),
                    record_evaluation(lgb_evals_result),
                ]
                if self.task == 'ranking':
                    callbacks.append(log_evaluation(period=50))
                self.model.fit(X_train, y_train, callbacks=callbacks, **fit_params)
                self._evals_result = lgb_evals_result
            else:
                self.model.fit(X_train, y_train, sample_weight=w_train)
            
        # 4. 后处理与评估
        self._calculate_feature_importance()
        self.is_trained = True
        
        import gc
        gc.collect()

        # 从训练曲线提取过拟合诊断（零推理开销）
        train_metrics = self._overfitting_diagnosis()
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': self._evaluate(X_val, y_val, "验证集", returns=r_val, dates=dates_val, sample_ratio=1.0)
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

    def _overfitting_diagnosis(self) -> Dict:
        """从训练曲线直接读取过拟合诊断，零推理开销"""
        evals = getattr(self, '_evals_result', {})
        if not evals:
            return {}

        # XGBoost: {'train_monitor': {'rmse': [...]}, 'validation': {'rmse': [...]}}
        # LightGBM: {'train_monitor': {'ndcg@1': [...]}, 'valid': {'ndcg@1': [...]}}
        train_key = next((k for k in evals if 'train' in k.lower()), None)
        val_key   = next((k for k in evals if 'train' not in k.lower()), None)

        metrics = {}

        # 只有验证集曲线时（ranking 任务 group 信息缺失导致训练集监控被跳过）
        if val_key and not train_key:
            val_curves  = evals[val_key]
            metric_name = next(iter(val_curves))
            val_series  = val_curves[metric_name]
            is_loss     = any(x in metric_name for x in ('loss', 'rmse', 'error', 'logloss'))
            best_val    = min(val_series) if is_loss else max(val_series)
            final_val   = val_series[-1]
            best_round  = val_series.index(best_val) + 1
            degradation = abs(final_val - best_val) / (abs(best_val) + 1e-8)

            print(f"\n  [过拟合诊断 - 训练曲线] 指标: {metric_name}  (仅验证集，ranking任务)")
            print(f"    验证集最终值:  {final_val:.5f}")
            print(f"    验证集最优值:  {best_val:.5f} (第 {best_round} 轮，共 {len(val_series)} 轮)")
            print(f"    Early Stop 后退化: {degradation:.2%}")
            metrics = {'val_final': final_val, 'best_val': best_val, 'best_round': best_round,
                       'metric_name': metric_name, 'rank_ic': 0.0}
            return metrics

        if not (train_key and val_key):
            return {}

        train_curves = evals[train_key]
        val_curves   = evals[val_key]
        metric_name  = next(iter(train_curves))
        # val 可能用不同 metric key，取第一个
        val_metric   = next(iter(val_curves))

        train_series = train_curves[metric_name]
        val_series   = val_curves[val_metric]
        is_loss      = any(x in metric_name for x in ('loss', 'rmse', 'error', 'logloss'))

        train_final = train_series[-1]
        val_final   = val_series[-1]
        best_val    = min(val_series) if is_loss else max(val_series)
        best_round  = val_series.index(best_val) + 1
        gap         = abs(train_final - val_final)
        overfit_ratio = gap / (abs(train_final) + 1e-8)

        print(f"\n  [过拟合诊断 - 训练曲线] 指标: {metric_name}")
        print(f"    训练集最终值:  {train_final:.5f}")
        print(f"    验证集最终值:  {val_final:.5f}")
        print(f"    验证集最优值:  {best_val:.5f} (第 {best_round} 轮，共 {len(val_series)} 轮)")
        print(f"    Train/Val 差距: {gap:.5f}  (过拟合比率: {overfit_ratio:.2%})")
        if overfit_ratio > 0.15:
            print(f"    ⚠️  过拟合风险较高，建议增大正则化或减少树深度")
        else:
            print(f"    ✅  Train/Val 差距在合理范围内")

        metrics = {
            'train_final': train_final,
            'val_final': val_final,
            'best_val': best_val,
            'best_round': best_round,
            'overfit_ratio': overfit_ratio,
            'metric_name': metric_name,
            'rank_ic': 0.0,
        }
        return metrics

    def _evaluate(self, X: Any, y: np.ndarray, dataset_name: str, returns: np.ndarray = None, 
                 dates: np.ndarray = None, sample_ratio: float = 1.0) -> Dict:
        y_prob = self._get_predict_proba(X)
        if not TrainingConfig.SAMPLE_EVAL:
            sample_ratio = 1.0
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
        # 评估基准优先用 returns（真实收益率），避免 y 因归一化方式变化导致 percentile 阈值偏移
        # 例如验证集用 z-score->sigmoid 归一化后，y 不再是排名分布，Top-N 精度会失真
        if returns is not None:
            reference = returns
        elif y is not None:
            reference = y
        else:
            reference = y_prob
        if isinstance(reference, pd.Series):
            reference = reference.to_numpy()
        if isinstance(reference, np.ndarray) and reference.dtype != np.float32:
            reference = reference.astype(np.float32)
        
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
                if len(np.unique(g_ref)) > 1 and len(np.unique(g_prob)) > 1:
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
            if len(np.unique(y_prob)) > 1 and len(np.unique(reference)) > 1:
                metrics['rank_ic'], _ = spearmanr(y_prob, reference)
            else:
                metrics['rank_ic'] = 0.0
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
        # X_scaled = self.scaler.transform(X).astype(np.float32)
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