"""
神经网络模型配置（NeuralConfig）

仅包含神经网络专属超参数。数据范围、缓存目录、多目标权重、横截面归一化规则
等全部复用 core.factors 下的既有配置（config.factor_config.TrainingConfig），
以保证"共用同一数据基础设施"。
"""
from typing import Dict, Any, Tuple
import os


class NeuralConfig:
    """神经网络训练超参数。"""

    # ── 网络结构 ──────────────────────────────────────────────────────────
    HIDDEN_DIMS: Tuple[int, ...] = (256, 128, 64, 32)  # 隐藏层维度
    DROPOUT: float = 0.25                         # Dropout 比例
    BATCHNORM: bool = True                        # 是否使用 BatchNorm1d

    # ── 训练超参 ───────────────────────────────────────────────────────────
    LEARNING_RATE: float = 1e-3
    BATCH_SIZE: int = 32768                       # GPU 训练主杠杆：数据常驻 GPU 后调大可进一步提升 CUDA 利用率（4096→16384→32768；显存够可试 65536）
    EPOCHS: int = 60
    WEIGHT_DECAY: float = 1e-5
    PATIENCE: int = 12                            # Early stopping 容忍轮数
    MIN_DELTA: float = 1e-4                       # 早停最小改善阈值（低于此值视为无改善）

    # ── 设备 ───────────────────────────────────────────────────────────────
    # None 表示自动（有 CUDA 用 cuda，否则 cpu）。
    DEVICE: Any = None

    # ── 随机种子 ───────────────────────────────────────────────────────────
    SEED: int = 42

    # ── 实验标签（用于模型目录命名/可读性）────────────────────────────────
    TAG: str = "neural"

    # ── 多目标权重（风险调整版）──────────────────────────────────────────
    # 与 GBM 的 TrainingConfig.MULTI_OBJECTIVE_WEIGHTS 解耦：NN 默认强调
    # 风险调整收益（sharpe）与回撤/下行波动，而非单纯预测高收益。
    # 键名必须与 multi_objective_labels 产出的列一致（含 y_sharpe_*）。
    MULTI_OBJECTIVE_WEIGHTS: Dict[str, float] = {
        "y_sharpe_20d": 0.40,   # 核心：20日收益 / |20日最大回撤|
        "y_sharpe_60d": 0.15,   # 60日风险调整收益
        "y_ret_20d": 0.10,      # 纯收益（保留部分权重）
        "y_ret_60d": 0.10,
        "y_mdd_20d": 0.12,      # 直接奖励低回撤
        "y_downvol_20d": 0.08,  # 奖励低下行波动
        "y_illiq_20d": 0.03,    # 奖励流动性
        "y_tradable_20d": 0.02, # 奖励可交易性
    }

    # 方向需反转的风险列（越小越好 -> 排名时反转）
    RISK_COLS = {"y_downvol_20d", "y_illiq_20d"}

    # ── v2 多目标权重：剔除退化目标 illiq/tradable（与 GBM v2 对齐），重新归一化 ──
    MULTI_OBJECTIVE_WEIGHTS_V2: Dict[str, float] = {
        "y_sharpe_20d": 0.40,
        "y_sharpe_60d": 0.16,
        "y_ret_20d": 0.12,
        "y_ret_60d": 0.12,
        "y_mdd_20d": 0.12,
        "y_downvol_20d": 0.08,
    }

    # ── v2 网络/训练超参：针对训练内即过拟合（val_loss 第 1 轮即上升）加强正则 ──
    DROPOUT_V2: float = 0.4            # 0.25 → 0.4
    WEIGHT_DECAY_V2: float = 1e-3      # 1e-5 → 1e-3
    BATCH_SIZE_V2: int = 16384         # 32768 → 16384（更小批量 = 更多随机性 = 正则）
    HIDDEN_DIMS_V2: Tuple[int, ...] = (256, 128, 64)  # 去掉最后一层 32，降容量
    PATIENCE_V2: int = 15

    @classmethod
    def to_model_kwargs(cls) -> Dict[str, Any]:
        """导出为传给 NeuralNetFactorModel 的关键字参数。"""
        return {
            "hidden_dims": cls.HIDDEN_DIMS,
            "dropout": cls.DROPOUT,
            "batchnorm": cls.BATCHNORM,
            "lr": cls.LEARNING_RATE,
            "batch_size": cls.BATCH_SIZE,
            "epochs": cls.EPOCHS,
            "weight_decay": cls.WEIGHT_DECAY,
            "patience": cls.PATIENCE,
            "min_delta": cls.MIN_DELTA,
            "device": cls.DEVICE,
            "seed": cls.SEED,
        }

    @classmethod
    def current_weights(cls) -> Dict[str, float]:
        """按 QD_MODEL_VERSION 返回多目标权重（默认 v1）。"""
        if os.getenv('QD_MODEL_VERSION') == 'v2':
            return dict(cls.MULTI_OBJECTIVE_WEIGHTS_V2)
        return dict(cls.MULTI_OBJECTIVE_WEIGHTS)

    @classmethod
    def current_model_kwargs(cls) -> Dict[str, Any]:
        """按 QD_MODEL_VERSION 返回模型超参（v2 叠加正则化加强项）。"""
        kw = cls.to_model_kwargs()
        if os.getenv('QD_MODEL_VERSION') == 'v2':
            kw.update(
                dropout=cls.DROPOUT_V2,
                weight_decay=cls.WEIGHT_DECAY_V2,
                batch_size=cls.BATCH_SIZE_V2,
                hidden_dims=cls.HIDDEN_DIMS_V2,
                patience=cls.PATIENCE_V2,
                min_delta=cls.MIN_DELTA,
            )
        return kw
