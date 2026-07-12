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
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from scipy.stats import spearmanr, rankdata
from config.factor_config import TrainingConfig, ModelConfig

class MLFactorModel:
    """机器学习因子模型"""
    
    def __init__(self, model_type: str = 'xgboost', task: str = None):
        self.model_type = model_type
        
        if task is None:
            task = getattr(TrainingConfig, 'TASK', 'ranking')
        
        # 处理 hybrid 任务：拆解为具体模型对应的底层任务
        if task == 'hybrid':
            if self.model_type == 'lightgbm':
                self.task = 'ranking'
            else:
                self.task = 'regression'
        elif task in ['ranking', 'regression']:
            self.task = task
        else:
            print(f"警告: 任务类型 '{task}' 无效，默认使用 'ranking'")
            self.task = 'ranking'
            
        self.model = None
        self.feature_names = []
        self.feature_importance = {}
        self.is_trained = False
        self.optimal_threshold = 0.5
        self._evals_result = {}  # 训练曲线，用于过拟合诊断
        self._init_model()
    
    def _init_model(self):
        """
        根据 model_type 和 task 初始化底层模型实例。

        设计原则：
        - 所有超参数从 ModelConfig 集中读取，_init_model 不做二次默认值填充。
        - ranking 专用参数（early_stopping_rounds, eval_at）在此处 pop 并保存为实例属性，
          避免在各训练路径（fit / xgb.train）中重复处理。
        - GPU 初始化失败时自动回退到 CPU，保证鲁棒性。
        """
        model_params = ModelConfig.get_model_params(self.model_type)

        # XGBoost 的 sklearn wrapper 不使用 random_state，用 seed；
        # LightGBM 用 random_state，但已在 LIGHTGBM_PARAMS 中未设置，依赖框架默认值。
        # 统一：如果用户在 ModelConfig 中没有显式设置，则不注入，避免与框架冲突。

        if self.model_type == 'xgboost' and HAS_XGB:
            self._init_xgboost(model_params)
        elif self.model_type == 'lightgbm' and HAS_LGB:
            self._init_lightgbm(model_params)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}，"
                             f"或所需库未安装 (HAS_XGB={HAS_XGB}, HAS_LGB={HAS_LGB})")

    def _init_xgboost(self, model_params: dict):
        """
        初始化 XGBoost 模型。

        Ranking 任务使用 XGBRanker (rank:ndcg)，标签为离散整数档位。
        参数已在 ModelConfig.XGBOOST_PARAMS 中完整定义，此处仅做 pop 处理。
        """
        # early_stopping_rounds 由训练器（xgb.train 或 fit）管理，不传入构造器
        self.early_stopping_rounds = model_params.pop('early_stopping_rounds', None)

        if model_params.get('objective') == 'regression':
            model_params.pop('ndcg_exp_gain', None)
            model_params.pop('lambdarank_num_pair_per_sample', None)
            model_params.pop('lambdarank_pair_method', None)
            model_params.pop('ndcg_exp_gain', None)

        def _build_xgb(params: dict):
            if model_params.get('objective') == 'rank:ndcg':
                return xgb.XGBRanker(**params)
            elif model_params.get('objective') == 'reg:squarederror':
                return xgb.XGBRegressor(**params)
            else:
                return xgb.XGBClassifier(**params)

        try:
            self.model = _build_xgb(model_params)
        except Exception as e:
            is_gpu = 'gpu' in str(model_params.get('tree_method', '')) or \
                     'cuda' in str(model_params.get('device', ''))
            if is_gpu:
                print(f"  [WARNING] XGBoost GPU 初始化失败: {e}，正在尝试回退到 CPU...")
                model_params.pop('tree_method', None)
                model_params.pop('device', None)
                model_params.pop('predictor', None)
                self.model = _build_xgb(model_params)
            else:
                raise

    def _init_lightgbm(self, model_params: dict):
        """
        初始化 LightGBM 模型。

        Ranking 任务使用 LGBMRanker (lambdarank)，标签档位数由 label_gain 决定。
        参数已在 ModelConfig.LIGHTGBM_PARAMS 中完整定义，此处仅做 pop 处理。
        """
        # pop ranking 专用参数，避免传入 LGBMRanker 构造器时冲突
        self.early_stopping_rounds = model_params.pop('early_stopping_rounds', None)
        # eval_at：用于 fit 时指定 ndcg 的截断位置
        self.eval_at = model_params.pop('eval_at', [10])

        if model_params.get('objective') == 'regression':
            model_params.pop('label_gain', None)
            model_params.pop('lambdarank_truncation_level', None)

        def _build_lgb(params: dict):
            if model_params.get('objective') == 'lambdarank':
                return lgb.LGBMRanker(**params)
            elif model_params.get('objective') == 'regression':
                return lgb.LGBMRegressor(**params)
            else:
                return lgb.LGBMClassifier(**params)

        try:
            self.model = _build_lgb(model_params)
        except Exception as e:
            if model_params.get('device') == 'gpu':
                print(f"  [WARNING] LightGBM GPU 初始化失败: {e}，正在尝试回退到 CPU...")
                model_params['device'] = 'cpu'
                self.model = _build_lgb(model_params)
            else:
                raise

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
        # if self.task == 'ranking':
        #     has_discrete_y = np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) > 2
        #     if not has_discrete_y:
        #         raise ValueError(f"排序任务需要使用离散整数标签 (0, 1, 2...), 但传入的标签 y 的 dtype 为 {y.dtype}")
        
        
        # 2. 划分数据集
        if use_time_series_split:
            # 排序任务：使用对齐到日期边界的 split_idx，避免拆分同一天的 group
            if 'split_idx' in kwargs:
                split_idx = kwargs['split_idx']
            else:
                split_idx = int(len(X) * (1 - validation_split))

            # 支持外部验证集直接传入（X_val_external / y_val_external）
            # 优先使用外部验证集做 early stopping，时间分布与真实推理场景一致
            X_val_ext = kwargs.get('X_val_external', None)
            y_val_ext = kwargs.get('y_val_external', None)

            if X_val_ext is not None and y_val_ext is not None:
                # 使用全量训练集 + 外部验证集
                X_train_raw = X[:split_idx]
                y_train     = y[:split_idx]
                X_val_raw   = X_val_ext
                y_val       = y_val_ext
                w_train     = sample_weight[:split_idx] if sample_weight is not None else None
                r_train     = returns[:split_idx] if returns is not None else None
                r_val       = None
                dates       = kwargs.get('dates', None)
                dates_train = dates[:split_idx] if dates is not None else None
                # 外部验证集的日期由调用方通过 dates_val_external 传入
                dates_val   = kwargs.get('dates_val_external', None)
            else:
                # 原有逻辑：内部切分
                val_start_idx = kwargs.get('val_start_idx', split_idx)
                X_train_raw, X_val_raw = X[:split_idx], X[val_start_idx:]
                y_train, y_val = y[:split_idx], y[val_start_idx:]
                w_train = sample_weight[:split_idx] if sample_weight is not None else None
                r_val = returns[val_start_idx:] if returns is not None else None
                r_train = returns[:split_idx] if returns is not None else None
                dates = kwargs.get('dates', None)
                dates_train = dates[:split_idx] if dates is not None else None
                dates_val = dates[val_start_idx:] if dates is not None else None

            # ── 排序任务：关键修复 ──
            # 由于内部执行了 split_idx 划分，必须同步将 group (query counts) 也切分为
            # 内部训练组和内部验证组，否则 LightGBM 会报 "Sum of query counts differs from data length"
            if self.task == 'ranking' and 'group' in kwargs:
                group_train = np.array(kwargs['group'])
                # eval_group 由外层直接传入（已按 val_start_idx 对齐），优先使用
                if 'eval_group' in kwargs and kwargs['eval_group'] is not None:
                    group_val = np.array(kwargs['eval_group'])
                    # group_train 已由调用方按 split_idx 对齐，直接使用，无需再切分
                else:
                    # 没有外部 eval_group：按 split_idx 从 group 中切分
                    group = np.array(kwargs['group'])
                    group_cumsum = np.cumsum(group)
                    split_group_count = np.searchsorted(group_cumsum, split_idx, side='right')
                    group_train = group[:split_group_count]
                    group_val = group[split_group_count:]

                # 校验：sum(group_train) 必须等于 split_idx
                actual_train_len = np.sum(group_train)
                if actual_train_len != split_idx:
                    print(f"  [WARNING] Ranking group split misalignment: "
                          f"sum(group_train)={actual_train_len}, expected={split_idx}. "
                          f"可能原因是 split_idx 未对齐到日期边界。")
            else:
                group_train = None
                group_val = None
        else:
            X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
            w_train = None
            r_val = None
            r_train = None
            dates_train = None
            dates_val = None


        # ---------------------------------------------------------------------
        # 3. 样本打乱 (Shuffle) —— 解决分批训练导致的分布漂移关键
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
        # QuantileDMatrix 一次性将数据驻留 GPU，避免迭代间反复 PCIe 搬运
        if self.model_type == 'xgboost' and mem_efficient:
            print(f"  [INFO] XGBoost 启动 QuantileDMatrix 训练模式 (样本数: {len(X_train_raw)})")
            
            # 直接传整个数组给 QuantileDMatrix，避免 DataIter 多次遍历的队列等待瓶颈。
            # QuantileDMatrix 内部会自动分块处理，GPU/CPU 利用率更均衡。
            params = ModelConfig.get_model_params('xgboost')
            xgb_params = params.copy()
            # XGBoost 原生接口（xgb.train）参数名转换：sklearn 参数名 → 原生参数名
            if 'n_jobs' in xgb_params:
                xgb_params['nthread'] = xgb_params.pop('n_jobs')
            # n_estimators / early_stopping_rounds 由 xgb.train 的 num_boost_round / early_stopping_rounds 管理
            xgb_params.pop('n_estimators', None)
            xgb_params.pop('early_stopping_rounds', None)

            # 回归模式：清除 ranking 专属参数，设置回归目标与评估指标
            if self.task == 'regression':
                xgb_params.pop('ndcg_exp_gain', None)
                xgb_params.pop('lambdarank_num_pair_per_sample', None)
                xgb_params.pop('lambdarank_pair_method', None)

            # ranking 模式不传 weight（由 group 承担），回归/分类模式传入样本权重
            _w_for_dtrain = None if self.task == 'ranking' else w_train

            dtrain = xgb.QuantileDMatrix(
                X_train_raw, label=y_train, weight=_w_for_dtrain,
                feature_names=self.feature_names,
            )
            if self.task == 'ranking' and group_train is not None:
                dtrain.set_group(group_train)

            dval = xgb.QuantileDMatrix(
                X_val_raw, label=y_val,
                feature_names=self.feature_names,
                ref=dtrain,
            )
            if self.task == 'ranking' and group_val is not None:
                dval.set_group(group_val)

            # 所有任务均加入训练集采样监控，用于过拟合诊断（训练集 vs 验证集曲线对比）
            # ranking 任务：按日期采样完整 group，保证 NDCG 计算有意义
            _dates_train_xgb = kwargs.get('dates', None)
            if _dates_train_xgb is not None:
                _dates_train_xgb = _dates_train_xgb[:split_idx]
            if self.task == 'ranking' and _dates_train_xgb is not None and group_train is not None:
                _unique_td = np.unique(_dates_train_xgb)
                _n_mon_days = max(50, int(len(_unique_td) * 0.2))
                _rng = np.random.default_rng(42)
                _mon_dates = _rng.choice(_unique_td, size=min(_n_mon_days, len(_unique_td)), replace=False)
                _mon_mask = np.isin(_dates_train_xgb, _mon_dates)
                _X_mon = X_train_raw[_mon_mask]
                _y_mon = y_train[_mon_mask]
                _, _mon_group = np.unique(_dates_train_xgb[_mon_mask], return_counts=True)
            else:
                _mon_size = min(50000, len(X_train_raw))
                _mon_idx = np.random.choice(len(X_train_raw), _mon_size, replace=False)
                _X_mon = X_train_raw[_mon_idx]
                _y_mon = y_train[_mon_idx]
                _mon_group = None

            dtrain_monitor = xgb.QuantileDMatrix(
                _X_mon, label=_y_mon,
                feature_names=self.feature_names, ref=dtrain,
            )
            if self.task == 'ranking':
                dtrain_monitor.set_group(_mon_group if _mon_group is not None else np.ones(len(_y_mon), dtype=np.int32))
            evals_list = [(dtrain_monitor, 'train_monitor'), (dval, 'validation')]

            evals_result = {}
            self.model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=params.get('n_estimators', 3000),
                evals=evals_list,
                evals_result=evals_result,
                # 使用 _init_model 中从配置 pop 出来的值，保证与 ModelConfig 一致
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=50,
            )
            self._evals_result = evals_result

            # 内存回收
            del dtrain_monitor
            del dtrain, dval
            X_train = X_train_raw  # 仅用于后续评估
            X_val = X_val_raw



        # ---------------------------------------------------------------------
        # 情况 B: LightGBM 内存优化
        # ---------------------------------------------------------------------
        elif self.model_type == 'lightgbm' and mem_efficient:
            X_train = X_train_raw
            X_val = X_val_raw

            fit_params = {'sample_weight': w_train}
            # 训练集监控：按日期采样完整 group，保证 NDCG 计算有意义（训练集 vs 验证集曲线对比）
            if self.task == 'ranking':
                if group_train is not None:
                    fit_params['group'] = group_train

                eval_at = getattr(self, 'eval_at', [10])
                # ranking 训练集监控：从 dates_train 中随机采样若干天，保留每天完整的 group 结构
                if dates_train is not None and group_train is not None:
                    unique_train_dates = np.unique(dates_train)
                    n_monitor_days = max(50, int(len(unique_train_dates) * 0.2))
                    rng = np.random.default_rng(42)
                    monitor_dates = rng.choice(unique_train_dates, size=min(n_monitor_days, len(unique_train_dates)), replace=False)
                    monitor_date_mask = np.isin(dates_train, monitor_dates)
                    X_train_monitor = X_train[monitor_date_mask]
                    y_train_monitor = y_train[monitor_date_mask]
                    _, monitor_group = np.unique(dates_train[monitor_date_mask], return_counts=True)
                else:
                    # 无日期信息时回退：随机采样，每行独立查询
                    monitor_size = min(50000, len(X_train))
                    monitor_idx = np.random.choice(len(X_train), monitor_size, replace=False)
                    X_train_monitor = X_train[monitor_idx]
                    y_train_monitor = y_train[monitor_idx]
                    monitor_group = np.ones(len(y_train_monitor), dtype=np.int32)

                val_group = group_val if group_val is not None else np.ones(len(y_val), dtype=np.int32)
                fit_params.update({
                    'eval_set': [(X_train_monitor, y_train_monitor), (X_val, y_val)],
                    'eval_names': ['train_monitor', 'valid'],
                    'eval_group': [monitor_group, val_group],
                    'eval_at': eval_at,
                })
            else:
                monitor_size = min(50000, len(X_train))
                monitor_idx = np.random.choice(len(X_train), monitor_size, replace=False)
                X_train_monitor = X_train[monitor_idx]
                y_train_monitor = y_train[monitor_idx]
                fit_params.update({
                    'eval_set': [(X_train_monitor, y_train_monitor), (X_val, y_val)],
                    'eval_names': ['train_monitor', 'valid'],
                })

            from lightgbm import early_stopping, log_evaluation, record_evaluation
            es_rounds = getattr(self, 'early_stopping_rounds', None)
            lgb_evals_result = {}

            # 防御性检查：确保 eval_set 已设置，否则 early_stopping callback 会报错
            if 'eval_set' not in fit_params:
                raise RuntimeError(
                    f"LightGBM ranking: fit_params 缺少 eval_set。"
                    f"group_val={group_val is not None}, X_val shape={X_val.shape}, y_val len={len(y_val)}, "
                    f"fit_params keys={list(fit_params.keys())}"
                )
            # 校验 eval_group 与 eval_set 长度一致（每个 eval_set 对应一个 group）
            if 'eval_group' in fit_params:
                for i, (eg, (_, ev)) in enumerate(zip(fit_params['eval_group'], fit_params['eval_set'])):
                    if np.sum(eg) != len(ev):
                        raise RuntimeError(
                            f"LightGBM eval_group[{i}] sum ({np.sum(eg)}) != eval_set[{i}] y len ({len(ev)})"
                        )

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

            fit_params = {}
            if w_train is not None and not (self.model_type == 'xgboost' and self.task == 'ranking'):
                fit_params['sample_weight'] = w_train
            if self.task == 'ranking':
                if group_train is not None:
                    fit_params['group'] = group_train

                if group_val is not None:
                    fit_params.update({
                        'eval_set': [(X_val, y_val)],
                        'eval_group': [group_val],
                    })
                else:
                    fit_params.update({
                        'eval_set': [(X_val, y_val)],
                    })
                    
                if self.model_type == 'lightgbm':
                    fit_params['eval_at'] = getattr(self, 'eval_at', [10])
            else:
                fit_params['eval_set'] = [(X_val, y_val)]

            if self.model_type == 'xgboost':
                self.model.fit(X_train, y_train, verbose=50, **fit_params)
                if hasattr(self.model, 'evals_result'):
                    _raw = self.model.evals_result()
                    if _raw:
                        _keys = list(_raw.keys())
                        if len(_keys) == 1:
                            self._evals_result = {k: v for k, v in _raw.items()}
                        else:
                            renamed = {}
                            for _i, _k in enumerate(_keys):
                                if _i == 0:
                                    renamed['train_monitor'] = _raw[_k]
                                elif _i == 1:
                                    renamed['valid'] = _raw[_k]
                                else:
                                    renamed[_k] = _raw[_k]
                            self._evals_result = renamed
            elif self.model_type == 'lightgbm':
                from lightgbm import early_stopping, log_evaluation, record_evaluation
                es_rounds = getattr(self, 'early_stopping_rounds', None)
                lgb_evals_result = {}
                # 所有任务均加入训练集采样监控，用于过拟合诊断（训练集 vs 验证集曲线对比）
                if self.task == 'ranking':
                    eval_at = getattr(self, 'eval_at', [10])
                    # ranking：按日期采样完整 group，保证 NDCG 计算有意义
                    if dates_train is not None and group_train is not None:
                        unique_train_dates = np.unique(dates_train)
                        n_monitor_days = max(50, int(len(unique_train_dates) * 0.2))
                        rng = np.random.default_rng(42)
                        monitor_dates = rng.choice(unique_train_dates, size=min(n_monitor_days, len(unique_train_dates)), replace=False)
                        monitor_date_mask = np.isin(dates_train, monitor_dates)
                        X_train_arr = X_train if isinstance(X_train, np.ndarray) else X_train.values
                        X_train_monitor = X_train_arr[monitor_date_mask]
                        y_train_monitor = y_train[monitor_date_mask]
                        _, monitor_group = np.unique(dates_train[monitor_date_mask], return_counts=True)
                    else:
                        monitor_size = min(50000, len(X_train))
                        monitor_idx = np.random.choice(len(X_train), monitor_size, replace=False)
                        X_train_monitor = X_train[monitor_idx] if isinstance(X_train, np.ndarray) else X_train.iloc[monitor_idx]
                        y_train_monitor = y_train[monitor_idx]
                        monitor_group = np.ones(len(y_train_monitor), dtype=np.int32)
                    existing_eval_group = fit_params.pop('eval_group', None)
                    val_group = existing_eval_group[0] if existing_eval_group else np.ones(len(y_val), dtype=np.int32)
                    fit_params.update({
                        'eval_set': [(X_train_monitor, y_train_monitor), (X_val, y_val)],
                        'eval_names': ['train_monitor', 'valid'],
                        'eval_group': [monitor_group, val_group],
                        'eval_at': eval_at,
                    })
                else:
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
                    log_evaluation(period=50),
                ]
                self.model.fit(X_train, y_train, callbacks=callbacks, **fit_params)
                self._evals_result = lgb_evals_result
            else:
                self.model.fit(X_train, y_train, sample_weight=w_train)
            
        # 4. 后处理
        self._calculate_feature_importance()
        self.is_trained = True
        
        import gc
        gc.collect()

        return {
            'train_metrics': {},
            'val_metrics': {},
        }

    @staticmethod
    def _slice_rows(data: Any, mask: np.ndarray) -> Any:
        if data is None:
            return None
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.iloc[np.flatnonzero(mask)]
        try:
            return data[mask]
        except TypeError:
            return np.asarray(data)[mask]

    @staticmethod
    def _get_xgb_best_iteration(booster: Any) -> Optional[int]:
        best_iteration = None
        try:
            best_iteration = getattr(booster, 'best_iteration')
        except Exception:
            best_iteration = None

        if best_iteration is None and hasattr(booster, 'attr'):
            try:
                best_iteration = booster.attr('best_iteration')
            except Exception:
                best_iteration = None

        try:
            best_iteration = int(best_iteration)
        except (TypeError, ValueError):
            return None

        return best_iteration if best_iteration >= 0 else None

    def _predict_xgb_booster(self, booster: Any, dmat: Any) -> np.ndarray:
        best_iteration = self._get_xgb_best_iteration(booster)
        if best_iteration is None:
            return booster.predict(dmat)

        try:
            return booster.predict(dmat, iteration_range=(0, best_iteration + 1))
        except TypeError:
            best_ntree_limit = best_iteration + 1
            try:
                best_ntree_limit = int(getattr(booster, 'best_ntree_limit'))
            except Exception:
                pass
            try:
                return booster.predict(dmat, ntree_limit=best_ntree_limit)
            except TypeError:
                return booster.predict(dmat)

    def _get_predict_proba(self, X: Any) -> np.ndarray:
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
                preds = self._predict_xgb_booster(self.model, dmat)
            else:
                # scikit-learn 包装类
                device = self.model.get_params().get('device', 'cpu')
                is_gpu = device == 'cuda' or 'gpu' in str(self.model.get_params().get('tree_method', ''))
                
                if is_gpu:
                    # GPU 模式下的预测加速与防止显存碎片
                    dmat = xgb.DMatrix(X, feature_names=self.feature_names) if not isinstance(X, pd.DataFrame) else xgb.DMatrix(X)
                    preds = self._predict_xgb_booster(self.model.get_booster(), dmat)
                else:
                    if not isinstance(X, pd.DataFrame) and self.feature_names:
                        X = pd.DataFrame(X, columns=self.feature_names)
                    preds = self.model.predict(X)
        else:
            if not isinstance(X, pd.DataFrame) and self.feature_names:
                X = pd.DataFrame(X, columns=self.feature_names)
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

    def _primary_metric_name(self, curves: Dict) -> str:
        """
        从 evals_result 的某个数据集曲线字典中，选出主监控指标名。

        优先级：
        1. 与 eval_at[0] 对应的 ndcg@N（early stopping 实际监控的指标）
        2. 任意包含 'ndcg' 的指标
        3. 字典第一个 key（兜底）
        """
        if not curves:
            return ''
        eval_at = getattr(self, 'eval_at', None)
        if eval_at:
            primary_k = f'ndcg@{eval_at[0]}'
            if primary_k in curves:
                return primary_k
        ndcg_keys = [k for k in curves if 'ndcg' in k.lower()]
        if ndcg_keys:
            return ndcg_keys[0]
        return next(iter(curves))

    def _overfitting_diagnosis(self) -> Dict:
        """
        从训练曲线直接读取过拟合诊断，零推理开销。

        训练集监控（train_monitor）使用按日期采样的完整 group 子集，
        与验证集使用相同的 NDCG 指标，可直接对比学习能力与泛化能力。
        """
        evals = getattr(self, '_evals_result', {})
        if not evals:
            return {}

        # key 约定：
        #   train_monitor → 训练集采样监控（含 'train'）
        #   valid / validation → 验证集（不含 'train'）
        train_key = next((k for k in evals if 'train' in k.lower()), None)
        val_key   = next((k for k in evals if 'train' not in k.lower()), None)

        metrics = {}

        # 仅有验证集曲线（兜底分支，正常情况不应触发）
        if val_key and not train_key:
            val_curves  = evals[val_key]
            metric_name = self._primary_metric_name(val_curves)
            val_series  = val_curves[metric_name]
            is_loss     = any(x in metric_name for x in ('loss', 'rmse', 'error', 'logloss'))
            best_val    = min(val_series) if is_loss else max(val_series)
            final_val   = val_series[-1]
            best_round  = val_series.index(best_val) + 1
            degradation = abs(final_val - best_val) / (abs(best_val) + 1e-8)

            print(f"\n  [训练曲线诊断] 指标: {metric_name}  (仅验证集)")
            print(f"    验证集最优值:  {best_val:.5f} (第 {best_round} 轮，共 {len(val_series)} 轮)")
            print(f"    验证集最终值:  {final_val:.5f}  (Early Stop 后退化: {degradation:.2%})")
            metrics = {'val_final': final_val, 'best_val': best_val, 'best_round': best_round,
                       'metric_name': metric_name, 'rank_ic': 0.0}
            return metrics

        if not (train_key and val_key):
            return {}

        train_curves = evals[train_key]
        val_curves   = evals[val_key]
        metric_name  = self._primary_metric_name(train_curves)
        val_metric   = self._primary_metric_name(val_curves)

        train_series = train_curves[metric_name]
        val_series   = val_curves[val_metric]
        is_loss      = any(x in metric_name for x in ('loss', 'rmse', 'error', 'logloss'))

        train_final = train_series[-1]
        val_final   = val_series[-1]
        best_val    = min(val_series) if is_loss else max(val_series)
        best_round  = val_series.index(best_val) + 1
        gap         = abs(train_final - val_final)
        overfit_ratio = gap / (abs(train_final) + 1e-8)

        # 学习趋势：训练集从初始到最终的提升幅度
        train_init  = train_series[0]
        train_gain  = train_final - train_init if not is_loss else train_init - train_final

        print(f"\n  [训练曲线诊断] 指标: {metric_name}")
        print(f"    训练集: {train_init:.5f} → {train_final:.5f}  (提升 {train_gain:+.5f})")
        print(f"    验证集: 最优 {best_val:.5f} (第 {best_round} 轮) → 最终 {val_final:.5f}  "
              f"(退化 {abs(val_final - best_val)/(abs(best_val)+1e-8):.2%})")
        print(f"    Train/Val 差距: {gap:.5f}  (过拟合度: {overfit_ratio:.2%})")

        if overfit_ratio > 0.20:
            print(f"    ⚠️  [严重] 过拟合风险极高！Train 远强于 Val。建议减小 depth 或增加 lambda。")
        elif overfit_ratio > 0.10:
            print(f"    ⚠️  [中度] 存在过拟合。")
        else:
            print(f"    ✅  [良好] 泛化性能控制在合理范围内。")

        metrics = {
            'train_final': train_final,
            'train_init': train_init,
            'train_gain': train_gain,
            'val_final': val_final,
            'best_val': best_val,
            'best_round': best_round,
            'overfit_ratio': overfit_ratio,
            'metric_name': metric_name,
        }
        return metrics

    def _evaluate(self, X: Any, y: np.ndarray, dataset_name: str, returns: np.ndarray = None, 
                 dates: np.ndarray = None, sample_ratio: float = 0.2) -> Dict:
        eval_type = "全量"
        eval_dates = None
        dates_eval = None

        if dates is not None:
            dates = np.asarray(dates)
            all_dates = np.unique(dates)
            eval_dates = all_dates

            if sample_ratio < 1.0:
                n_sample = max(1, int(len(all_dates) * sample_ratio))
                rng = np.random.default_rng(42)
                eval_dates = rng.choice(all_dates, size=n_sample, replace=False)
                eval_type = f"随机抽样 {sample_ratio:.0%}"

            sample_mask = np.isin(dates, eval_dates)
            X = self._slice_rows(X, sample_mask)
            y = self._slice_rows(y, sample_mask)
            returns = self._slice_rows(returns, sample_mask)
            dates_eval = dates[sample_mask]

        y_prob = self._get_predict_proba(X)
        y = np.asarray(y)
        # 1. 基础指标计算 (针对类别/回归)
        # y 可能包含软标签，因此先进行二元化处理
        y_true_bin = (y >= 0.5).astype(int)
        y_pred = (y_prob >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true_bin, y_pred) if len(np.unique(y_true_bin)) > 1 else 1.0,
            'auc': roc_auc_score(y_true_bin, y_prob) if len(np.unique(y_true_bin)) > 1 else 0.5,
            'precision': precision_score(y_true_bin, y_pred, zero_division=0),
            'recall': recall_score(y_true_bin, y_pred, zero_division=0),
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
        
        if dates_eval is not None:
            rank_ics = []
            top1_hits = []
            top5_hits = []
            
            for d in eval_dates:
                mask = dates_eval == d
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
                    
                    # D. 绝对胜率 (Win Rate)：Top-1 的真实收益是否大于 0
                    # 注意：如果 reference 是收益率 (returns)，则判断 > 0；如果是归一化后的 y，则判断是否大于中性值
                    is_win = (g_ref[top1_idx] > 0)
                    metrics.setdefault('win_rates', []).append(1.0 if is_win else 0.0)
            
            metrics['rank_ic'] = np.mean(rank_ics) if rank_ics else 0.0
            metrics['rank_ic_std'] = np.std(rank_ics) if rank_ics else 0.0
            metrics['top1_precision'] = np.mean(top1_hits) if top1_hits else 0.0
            metrics['top5_precision'] = np.mean(top5_hits) if top5_hits else 0.0
            metrics['win_rate'] = np.mean(metrics.pop('win_rates')) if 'win_rates' in metrics else 0.0
            
            # 辅助统计：预测区分度
            prob_std = np.std(y_prob)
            unique_probs = len(np.unique(np.round(y_prob, 6)))
            
            print(f"  [{dataset_name}] {eval_type}评估 ({len(eval_dates)} 个交易日):")
            print(f"    预测区分度: Std={prob_std:.4f}, Unique={unique_probs}")
            print(f"    Rank IC: {metrics['rank_ic']:.4f} ± {metrics['rank_ic_std']:.4f}")
            print(f"    Top-1 胜率 (收益>0): {metrics['win_rate']:.2%}")
            print(f"    Top-1 精度 (命中前5%): {metrics['top1_precision']:.2%}")
            print(f"    Top-5 精度 (命中前20%): {metrics['top5_precision']:.2%}")
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
        """
        获取模型预测结果，自动处理特征对齐。

        注意：本方法不再内部执行横截面归一化。
        调用方（如 generate_signals）负责在调用前完成归一化，
        以确保归一化逻辑与训练时完全一致（跳过规则、列范围等）。
        内部二次归一化会破坏调用方已正确处理的特征分布。
        """
        if not self.is_trained: raise ValueError("未训练")
        
        # 1. 提取特征，对齐训练时的特征顺序，处理缺失值与边界
        if isinstance(factors, pd.DataFrame):
            X_arr = factors[self.feature_names].values.astype(np.float32)
        elif isinstance(factors, np.ndarray):
            X_arr = factors.astype(np.float32)
        else:
            X_arr = np.asarray(factors, dtype=np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.5, posinf=1.0, neginf=0.0)

        # 保留列名包装为 DataFrame，避免 sklearn 警告 "X does not have valid feature names"
        X = pd.DataFrame(X_arr, columns=self.feature_names) if self.feature_names else X_arr

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
