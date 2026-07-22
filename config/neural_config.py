"""
神经网络模型配置（NeuralConfig）

仅包含神经网络专属超参数。数据范围、缓存目录、多目标权重、横截面归一化规则
等全部复用 core.factors 下的既有配置（config.factor_config.TrainingConfig），
以保证"共用同一数据基础设施"。
"""
from typing import Dict, Any, Tuple


class NeuralConfig:
    """神经网络训练超参数。"""

    # ── 网络结构 ──────────────────────────────────────────────────────────
    HIDDEN_DIMS: Tuple[int, ...] = (256, 128, 64, 32)  # 隐藏层维度
    DROPOUT: float = 0.25                         # Dropout 比例
    BATCHNORM: bool = True                        # 是否使用 BatchNorm1d

    # ── 训练超参 ───────────────────────────────────────────────────────────
    LEARNING_RATE: float = 1e-3
    BATCH_SIZE: int = 4096
    EPOCHS: int = 60
    WEIGHT_DECAY: float = 1e-5
    PATIENCE: int = 12                            # Early stopping 容忍轮数

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
            "device": cls.DEVICE,
            "seed": cls.SEED,
        }
