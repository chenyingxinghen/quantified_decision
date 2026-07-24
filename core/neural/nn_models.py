"""
神经网络因子模型实现（PyTorch）。

对外接口与 core.factors.ml_factor_model 中的 MLFactorModel /
MultiObjectiveFactorModel 保持一致，便于被回测与实盘选股直接消费：

- NeuralNetFactorModel：单目标神经网络，实现
      predict(factors) -> np.ndarray(float32, [0,1])
      feature_names / is_trained / save_model / load_model / get_top_factors
- MultiObjectiveNeuralModel：多目标加权组合，实现
      models / weights / feature_names / predict / predict_components
      save_model / load_model

注意：本模块在导入时会 import torch，因此仅在真正使用神经网络模型时才会
被加载（例如回测加载到神经网络模型文件时）。torch 未安装时不影响既有 GBM 流程。
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================================
# 1. 网络结构
# ============================================================================
class NeuralNet(nn.Module):
    """多层感知机（MLP），面向表格因子预测。

    结构：Linear -> (BatchNorm) -> ReLU -> Dropout 的若干隐藏层，
    末层 Linear(1) -> Sigmoid，直接输出 [0,1] 区间的分数（软标签回归语义）。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (256, 128, 64),
        dropout: float = 0.2,
        batchnorm: bool = True,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim 必须为正数，收到 {input_dim}")
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ============================================================================
# 2. 单目标神经网络模型（API 兼容 MLFactorModel）
# ============================================================================
class NeuralNetFactorModel:
    """单目标神经网络因子模型。

    与 MLFactorModel 的差异：
    - 训练器为 PyTorch（MLFactorModel 为 XGBoost/LightGBM）。
    - 损失采用软标签 MSE（标签已在区间 [0,1] 且方向统一：越大越好）。
    - 特征重要性采用首层权重幅值近似（非树模型增益）。
    预测接口与输出语义与 MLFactorModel 一致（sigmoid 后的 [0,1] 分数）。
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dims: tuple = (256, 128, 64),
        dropout: float = 0.2,
        batchnorm: bool = True,
        task: str = "ranking",
        feature_names: Optional[List[str]] = None,
        lr: float = 1e-3,
        batch_size: int = 4096,
        epochs: int = 60,
        weight_decay: float = 1e-5,
        patience: int = 12,
        min_delta: float = 1e-6,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.input_dim = int(input_dim) if input_dim is not None else None
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = float(dropout)
        self.batchnorm = bool(batchnorm)
        self.task = task
        self.feature_names = list(feature_names) if feature_names else []
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(seed)

        self.model: Optional[NeuralNet] = None
        self.is_trained = False
        self._history: Dict[str, Any] = {}
        self.feature_importance: Dict[str, float] = {}

        if self.input_dim is not None:
            self._init_model()

    def _init_model(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.model = NeuralNet(
            self.input_dim, self.hidden_dims, self.dropout, self.batchnorm
        ).to(self.device)

    # ------------------------------------------------------------------ 训练
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """训练神经网络。X/y 应为已横截面归一化的 float32 数组，y 取值 [0,1]。"""
        if self.input_dim is None:
            self.input_dim = X_train.shape[1]
            self._init_model()
        if self.model is None:
            self._init_model()

        Xtr = np.ascontiguousarray(X_train, dtype=np.float32)
        ytr = np.ascontiguousarray(y_train, dtype=np.float32).reshape(-1)
        Xtr = np.nan_to_num(Xtr, nan=0.5, posinf=1.0, neginf=0.0)
        ytr = np.nan_to_num(ytr, nan=0.5)

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv = np.ascontiguousarray(X_val, dtype=np.float32)
            yv = np.ascontiguousarray(y_val, dtype=np.float32).reshape(-1)
            Xv = np.nan_to_num(Xv, nan=0.5, posinf=1.0, neginf=0.0)
            yv = np.nan_to_num(yv, nan=0.5)

        # 可选样本权重（回归软标签场景简单按权重重采样成本较高，这里用
        # WeightedRandomSampler 仅在有明显权重差异时启用）。
        use_weighted = sample_weight is not None and np.std(sample_weight) > 1e-6
        if use_weighted:
            w = np.asarray(sample_weight, dtype=np.float64)
            w = w / (w.sum() + 1e-12)
            sampler = torch.utils.data.WeightedRandomSampler(
                w, num_samples=len(w), replacement=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        use_cuda = "cuda" in str(self.device)
        pin_memory = False
        # 显存占用低时，把整份训练/验证数据一次性搬上 GPU，消除训练循环里
        # 逐批的 CPU→GPU 小拷贝（这是低利用率的主要拖累）。显存不足则回退
        # 到 CPU 常驻 + pin_memory 异步搬运，保证不 OOM。两种路径数值结果一致。
        if use_cuda:
            torch.backends.cudnn.benchmark = True
            try:
                Xtr_t = torch.from_numpy(Xtr).to(self.device)
                ytr_t = torch.from_numpy(ytr).to(self.device)
                if has_val:
                    Xv_t = torch.from_numpy(Xv).to(self.device)
                    yv_t = torch.from_numpy(yv).to(self.device)
                # 整份数据已在 GPU，无需 pin_memory / 异步搬运
            except RuntimeError:
                # 显存不够（如 6GB 卡跑全市场）：回退 CPU + pin_memory
                Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
                if has_val:
                    Xv_t, yv_t = torch.from_numpy(Xv), torch.from_numpy(yv)
                pin_memory = True
        else:
            Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
            if has_val:
                Xv_t, yv_t = torch.from_numpy(Xv), torch.from_numpy(yv)
        train_ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=shuffle,
            sampler=sampler, drop_last=False, pin_memory=pin_memory,
        )
        if has_val:
            val_ds = torch.utils.data.TensorDataset(Xv_t, yv_t)
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False,
                pin_memory=pin_memory,
            )

        loss_fn = nn.MSELoss(reduction="mean")
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(3, self.patience // 2)
        )

        self.model.train()
        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        train_losses, val_losses = [], []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running = torch.zeros((), device=self.device)
            n = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=pin_memory)
                yb = yb.to(self.device, non_blocking=pin_memory)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                # 累计用张量、不在每 batch 触发 CPU↔GPU 同步（结果中性）
                running += loss * xb.size(0)
                n += xb.size(0)
            train_loss = (running / max(n, 1)).item()
            train_losses.append(train_loss)

            if has_val:
                self.model.eval()
                v_running = torch.zeros((), device=self.device)
                v_n = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device, non_blocking=pin_memory)
                        yb = yb.to(self.device, non_blocking=pin_memory)
                        v_running += loss_fn(self.model(xb), yb) * xb.size(0)
                        v_n += xb.size(0)
                val_loss = (v_running / max(v_n, 1)).item()
                val_losses.append(val_loss)
                scheduler.step(val_loss)
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if verbose:
                    print(f"  [NN epoch {epoch:03d}] "
                          f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                          f"(best={best_val_loss:.5f}, no_improve={epochs_no_improve})")
                if epochs_no_improve >= self.patience:
                    if verbose:
                        print(f"  [NN] Early stopping at epoch {epoch}")
                    break
            else:
                if verbose:
                    print(f"  [NN epoch {epoch:03d}] train_loss={train_loss:.5f}")
                # 无验证集时以自身为最优（不早停，跑满 epochs）

        # 还原最优权重
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.is_trained = True
        self._history = {"train_loss": train_losses, "val_loss": val_losses}
        self._compute_feature_importance()
        return {"train_metrics": {"loss": train_losses[-1] if train_losses else None},
                "val_metrics": {"loss": val_losses[-1] if val_losses else None}}

    def _compute_feature_importance(self):
        """首层权重幅值近似特征重要性（与树模型增益语义不同，仅作参考）。"""
        if self.model is None or not self.feature_names:
            return
        first_linear = None
        for m in self.model.net:
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        if first_linear is None:
            return
        w = first_linear.weight.detach().abs().sum(dim=0).cpu().numpy()
        if len(w) == len(self.feature_names):
            total = w.sum() + 1e-12
            self.feature_importance = {
                name: float(v / total) for name, v in zip(self.feature_names, w)
            }

    # ------------------------------------------------------------------ 预测
    def predict(self, factors: Any) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("神经网络模型尚未训练")
        if isinstance(factors, pd.DataFrame):
            X = factors[self.feature_names].values.astype(np.float32)
        elif isinstance(factors, np.ndarray):
            X = factors.astype(np.float32)
        else:
            X = np.asarray(factors, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.5, posinf=1.0, neginf=0.0)
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(X)).to(self.device)
            out = self.model(t).cpu().numpy().astype(np.float32)
        return out

    def predict_signal(self, factors: Any, threshold: float = 0.5) -> Dict:
        prob = float(self.predict(factors)[0])
        return {"signal": "buy" if prob >= threshold else "hold",
                "confidence": prob * 100, "prediction": prob}

    def get_top_factors(self, n: int = 10) -> List[tuple]:
        if not self.feature_importance:
            return []
        return sorted(self.feature_importance.items(),
                      key=lambda x: x[1], reverse=True)[:n]

    # -------------------------------------------------------------- 序列化
    def _to_state_dict(self) -> Dict[str, Any]:
        return {
            "format": "neural_factor_model",
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "batchnorm": self.batchnorm,
            "task": self.task,
            "feature_names": self.feature_names,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "device": self.device,
            "seed": self.seed,
            "is_trained": self.is_trained,
            "history": self._history,
            "state_dict": (
                self.model.state_dict() if self.model is not None else None
            ),
        }

    @classmethod
    def _from_state_dict(cls, state: Dict[str, Any]) -> "NeuralNetFactorModel":
        if state.get("format") != "neural_factor_model":
            raise ValueError("不是神经网络因子模型文件")
        obj = cls(
            input_dim=state["input_dim"],
            hidden_dims=state["hidden_dims"],
            dropout=state["dropout"],
            batchnorm=state["batchnorm"],
            task=state.get("task", "ranking"),
            feature_names=state.get("feature_names", []),
            lr=state.get("lr", 1e-3),
            batch_size=state.get("batch_size", 4096),
            epochs=state.get("epochs", 60),
            weight_decay=state.get("weight_decay", 1e-5),
            patience=state.get("patience", 12),
            device=state.get("device"),
            seed=state.get("seed", 42),
        )
        sd = state.get("state_dict")
        if sd is not None and obj.model is not None:
            obj.model.load_state_dict(sd)
            obj.is_trained = bool(state.get("is_trained", True))
        obj._history = state.get("history", {})
        return obj

    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self._to_state_dict(), f)

    @classmethod
    def load_model(cls, filepath: str) -> "NeuralNetFactorModel":
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        return cls._from_state_dict(state)


# ============================================================================
# 3. 多目标神经网络组合模型（API 兼容 MultiObjectiveFactorModel）
# ============================================================================
class MultiObjectiveNeuralModel:
    """将多个单目标神经网络模型组合为统一选股分数。

    与 MultiObjectiveFactorModel 完全等价：每个子模型预测方向统一后的
    desirability 分数（越大越好），回测/实盘只需消费加权总分，同时可审计各目标分量。
    """

    FORMAT_VERSION = 1

    def __init__(self, models: Dict[str, NeuralNetFactorModel],
                 weights: Dict[str, float]):
        if not models:
            raise ValueError("models 不能为空")
        if set(models) != set(weights):
            raise ValueError("models 与 weights 的目标名称必须一致")
        if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("多目标权重必须非负且总和大于 0")
        total = float(sum(weights.values()))
        self.models = dict(models)
        self.weights = {name: float(weights[name]) / total for name in models}
        self.is_trained = all(
            getattr(m, "is_trained", False) for m in models.values()
        )
        ordered: List[str] = []
        seen = set()
        for m in models.values():
            for name in getattr(m, "feature_names", []) or []:
                if name not in seen:
                    ordered.append(name)
                    seen.add(name)
        self.feature_names = ordered

    def predict_components(self, factors: pd.DataFrame) -> Dict[str, np.ndarray]:
        if not self.is_trained:
            raise ValueError("多目标子模型尚未全部训练")
        result = {}
        for objective, model in self.models.items():
            model_input = factors.reindex(columns=model.feature_names, fill_value=0.5)
            result[objective] = np.asarray(model.predict(model_input), dtype=np.float32)
        return result

    def predict(self, factors: pd.DataFrame) -> np.ndarray:
        components = self.predict_components(factors)
        output = np.zeros(len(factors), dtype=np.float32)
        for objective, values in components.items():
            output += values * self.weights[objective]
        return output

    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        # 每个子模型序列化到内存缓冲，避免产生额外磁盘文件
        model_states: Dict[str, bytes] = {}
        for name, model in self.models.items():
            model_states[name] = pickle.dumps(model._to_state_dict())
        state = {
            "format": "multi_objective_neural_model",
            "version": self.FORMAT_VERSION,
            "weights": self.weights,
            "model_states": model_states,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_model(cls, filepath: str) -> "MultiObjectiveNeuralModel":
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        if state.get("format") != "multi_objective_neural_model":
            raise ValueError("不是多目标神经网络模型文件")
        models: Dict[str, NeuralNetFactorModel] = {}
        for name, raw in state["model_states"].items():
            models[name] = NeuralNetFactorModel._from_state_dict(pickle.loads(raw))
        return cls(models, state["weights"])
