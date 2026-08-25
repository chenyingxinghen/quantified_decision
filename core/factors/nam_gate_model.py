"""
NAMGateModel —— 神经可加模型（NAM）专家 + 市场状态门控（Regime Gate）
=====================================================================

设计动机
--------
生产基线（XGBoost LambdaRank）在 T027–T038 卡在两个结构性问题上：

1. **中长期市场状态学不到**：全市场情绪列在日截面零方差，被迫剔除；
   只能靠手工 ``{mkt}_regime_{stock}`` 乘积注入，编码低效。
2. **可解释性只有 feature_importance**：说得出"哪个因子重要"，
   说不出"这个因子怎么起作用""什么行情下它才起作用"。

本模型把两件事一次解决：

    score = β0 + Σ_k  w_t[k] · Σ_{i ∈ group k} f_i(x_i)
            └────────────┘   └──────────────────────┘
             市场状态门控          单因子形状函数（NAM）

- ``f_i(x_i)``：每个因子一条独立曲线，**可直接画出来**（线性？阈值？U 型？）
- ``w_t[k]``：由**多时间尺度市场状态向量** ``m_t`` 生成的因子族权重，
  **可直接画成热力图**（横轴时间、纵轴因子族），回答"什么行情放大哪类因子"
- 两者相乘即"regime conditioning"，但是**学出来的**，不是手工乘出来的

工程要点
--------
- 227 个专家用 **batched einsum** 一次算完，不用 227 个 nn.Module（否则 Python 循环致命）
- 门控在**因子族**（K≈12）而非单因子上输出权重：227 维 softmax 必然坍塌
- 三重防坍塌：负载均衡损失（CV²）+ 熵正则 + 温度 warmup；兜底可切乘性调节模式
- 接口对齐 ``MLFactorModel``（``predict`` / ``save_model`` / ``load_model`` /
  ``feature_names`` / ``feature_importance``），回测侧改动最小
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
    _Module = nn.Module
except Exception:  # pragma: no cover - 允许无 torch 环境导入本模块（回测分派会跳过）
    torch = None
    nn = None
    _TORCH_OK = False
    _Module = object


# ---------------------------------------------------------------------------
# 网络组件
# ---------------------------------------------------------------------------

class FactorExpertBank(_Module):
    """
    N 个单因子形状函数 ``f_i: R -> R``，用 batched 权重并行计算。

    每个专家结构：``1 -> H -> H -> 1``（ReLU）。刻意做小（H=16）：
    单变量函数不需要容量，容量过大只会在低信噪比的量化数据上过拟合。

    参数量 ≈ N × (2H + H² + 2H + 1)，N=227/H=16 时约 7.4 万，比 GBDT 小两个量级。
    """

    def __init__(self, n_factors: int, hidden: int = 16, dropout: float = 0.0):
        super().__init__()
        self.n_factors = n_factors
        self.hidden = hidden

        def _p(*shape, scale):
            return nn.Parameter(torch.randn(*shape) * scale)

        # 第 1 层: [N,1,H]
        self.w1 = _p(n_factors, 1, hidden, scale=1.0)
        self.b1 = nn.Parameter(torch.zeros(n_factors, hidden))
        # 第 2 层: [N,H,H]
        self.w2 = _p(n_factors, hidden, hidden, scale=(1.0 / hidden) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(n_factors, hidden))
        # 输出层: [N,H,1]
        self.w3 = _p(n_factors, hidden, 1, scale=(1.0 / hidden) ** 0.5)
        self.b3 = nn.Parameter(torch.zeros(n_factors, 1))

        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """x: [B, N] -> contributions: [B, N]"""
        h = x.unsqueeze(-1)                                   # [B,N,1]
        h = self.act(torch.einsum('bni,nih->bnh', h, self.w1) + self.b1)
        if self.drop is not None:
            h = self.drop(h)
        h = self.act(torch.einsum('bnh,nhg->bng', h, self.w2) + self.b2)
        out = torch.einsum('bnh,nho->bno', h, self.w3) + self.b3   # [B,N,1]
        return out.squeeze(-1)

    def shape_curve(self, factor_idx: int, grid: "torch.Tensor") -> "torch.Tensor":
        """取单个因子的形状函数曲线（可解释性出图用）。grid: [G] -> [G]"""
        with torch.no_grad():
            g = grid.shape[0]
            x = torch.zeros(g, self.n_factors, device=grid.device, dtype=grid.dtype)
            x[:, factor_idx] = grid
            return self.forward(x)[:, factor_idx]


class RegimeGate(_Module):
    """
    市场状态 → 因子族权重。

    ``mode='softmax'``：``w = K · softmax(logits / τ)``，权重和恒为 K（均值 1），
    保证总体尺度稳定，纯粹表达"资源在族间如何再分配"。
    ``mode='multiplicative'``：``w = lo + (hi-lo)·sigmoid(logits)``，
    各族独立缩放、互不竞争 —— 观察到坍塌时的兜底方案。
    ``mode='scalar'``（E6/T073）：**不用 MLP**。取单一 regime 列 s（由
    ``scalar_idx`` 指定，滚动分位归一化后已在 [-1,1]、0=历史中位），
    令 ``logits = a · s``，``a ∈ R^{n_groups}`` 是**唯一**的门控参数
    （13 族 = 13 个参数），再走与 softmax 模式相同的 ``K·softmax``。
    立论：T046 的 oracle IC +128% 说明条件结构存在，但可实现增益≈0 的原因是
    在约 90 个独立块上学不动 (64,32) MLP 的 2000+ 参数（台账 E6 节）。
    参数量降三个量级后才有胜算；起点 a=0 → 均匀权重，与 softmax 模式同起点。
    ``mode='dual_scalar'``（scheme B，2026-08-24）：单列 scalar 的 2 轴推广。
    取两列 s1、s2（``scalar_idx`` / ``scalar_idx2``），``logits = a·s1 + b·s2``，
    ``a, b ∈ R^{n_groups}`` 共 26 个参数（2×13 族）。用于 phase0 市场级列的
    PCA-2 主成分双轴门控（如利率/流动性轴 × 风险偏好/恐慌轴），仍远低于
    softmax MLP，保持小样本可估；a=b=0 起点 → 均匀权重 → 安全护栏不变。
    """

    def __init__(self, d_regime: int, n_groups: int, hidden: Sequence[int] = (64, 32),
                 mode: str = 'softmax', bound: Tuple[float, float] = (0.2, 2.0),
                 dropout: float = 0.1, scalar_idx: int = 0, scalar_idx2: int = 0):
        super().__init__()
        self.mode = mode
        self.n_groups = n_groups
        self.lo, self.hi = bound
        self.scalar_idx = int(scalar_idx)
        self.scalar_idx2 = int(scalar_idx2)

        if mode in ('scalar', 'dual_scalar'):
            # 唯一参数：每族对该标量信号的敏感度。零初始化 → 起步权重均匀。
            # dual_scalar 再叠加第二个敏感度向量 b（26 = 2×13 参数）。
            self.a = nn.Parameter(torch.zeros(n_groups))
            self.b = nn.Parameter(torch.zeros(n_groups)) if mode == 'dual_scalar' else None
            self.net = None
            return

        layers: List = []
        prev = d_regime
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.Tanh()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, n_groups)]
        self.net = nn.Sequential(*layers)
        # 初始输出接近 0 → softmax 近似均匀 / sigmoid 近似中值，避免起步就偏科
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, m: "torch.Tensor", temperature: float = 1.0) -> "torch.Tensor":
        if self.mode in ('scalar', 'dual_scalar'):
            # regime 矩阵已由 _rolling_pct_normalize 映射到 [-1,1]、0=历史中位，
            # 直接用即可。2026-08-14 修复：原先在这里又做了一次 2s-1，复合成
            # 4r-3 —— 均匀锚点被推到 75 分位、低波半区杠杆是高波半区的 3 倍、
            # 缺失日填 0 变成满档倾斜。E6 的判定建立在这个畸变实现上。
            s1 = m[:, self.scalar_idx:self.scalar_idx + 1]       # [B,1] ∈ [-1,1]
            if self.mode == 'dual_scalar':
                s2 = m[:, self.scalar_idx2:self.scalar_idx2 + 1] # [B,1]
                logits = self.a.unsqueeze(0) * s1 + self.b.unsqueeze(0) * s2
            else:
                logits = self.a.unsqueeze(0) * s1                # [B,K]
            return torch.softmax(logits / max(temperature, 1e-3), dim=-1) * self.n_groups
        logits = self.net(m)
        if self.mode == 'softmax':
            return torch.softmax(logits / max(temperature, 1e-3), dim=-1) * self.n_groups
        s = torch.sigmoid(logits / max(temperature, 1e-3))
        return self.lo + (self.hi - self.lo) * s


class NAMGateNet(_Module):
    """NAM 专家 + Regime 门控的完整打分网络。"""

    def __init__(self, n_factors: int, group_ids: np.ndarray, n_groups: int,
                 d_regime: int, expert_hidden: int = 16,
                 gate_hidden: Sequence[int] = (64, 32),
                 gate_mode: str = 'softmax', expert_dropout: float = 0.0,
                 disable_gate: bool = False, gate_scalar_idx: int = 0,
                 gate_scalar_idx2: int = 0,
                 group_norm: bool = False):
        super().__init__()
        self.experts = FactorExpertBank(n_factors, expert_hidden, expert_dropout)
        self.gate = RegimeGate(d_regime, n_groups, gate_hidden, mode=gate_mode,
                               scalar_idx=gate_scalar_idx, scalar_idx2=gate_scalar_idx2)
        self.n_groups = n_groups
        self.disable_gate = bool(disable_gate)
        # T119：族输出截面标准化。动机是乘积 w_k·S_k 在 (w_k→c·w_k, S_k→S_k/c) 下不变
        # ⇒ 损失曲面有平坦方向 ⇒ 猜测门控权重会沿它随噪声漂移。标准化把 f 的幅度锁死，
        # 门控只能调"方向/相对重要性"，无法被专家吸收。
        # ⛔ 该动机已被 T119 自己证伪：锁死平坦方向后，门学到的敏感度向量 a 与不锁时
        #    **逐种子余弦 +0.991~+1.000**，门控行为分毫不动 ⇒ 平坦方向不是成因。
        #    开关保留只为复现台账里的 T119/T125 臂，**生产恒为 --disable-gate**，两条路径
        #    逐位相同。真实成因见台账 T127（|a_k| ∝ n_k^−1.45，权重分配由族边界决定），机制未查明。
        # 减均值本身无副作用：ListNet 按日 softmax 平移不变。
        # 仅作用于门控通路；--disable-gate 时不启用，保证纯加性基线逐位不变。
        self.group_norm = bool(group_norm)
        self.register_buffer('group_ids', torch.as_tensor(group_ids, dtype=torch.long))
        # one-hot 分组矩阵 [N, K]，用矩阵乘做分组求和（比 scatter_add 更快且可导）
        onehot = torch.zeros(n_factors, n_groups)
        onehot[torch.arange(n_factors), torch.as_tensor(group_ids, dtype=torch.long)] = 1.0
        self.register_buffer('group_onehot', onehot)
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _std_groups(group_sums: "torch.Tensor") -> "torch.Tensor":
        """按当日截面对每族求和做标准化（batch 维即一个交易日的全部股票）。"""
        mu = group_sums.mean(dim=0, keepdim=True)
        sd = group_sums.std(dim=0, unbiased=False, keepdim=True)
        return (group_sums - mu) / (sd + 1e-6)

    def forward(self, x: "torch.Tensor", m: "torch.Tensor", temperature: float = 1.0):
        """
        x: [B, N] 个股因子（已做当日截面归一化）
        m: [B, d_m] 该样本所属交易日的市场状态向量
        返回 (score[B], gate_weights[B, K], group_sums[B, K])
        """
        contrib = self.experts(x)                       # [B, N]
        group_sums = contrib @ self.group_onehot        # [B, K]
        if self.disable_gate:
            w = torch.ones(group_sums.shape[0], self.n_groups, device=group_sums.device)
            score = group_sums.sum(dim=-1) + self.bias
            return score, w, group_sums
        w = self.gate(m, temperature)                   # [B, K]
        gs = self._std_groups(group_sums) if self.group_norm else group_sums
        score = (gs * w).sum(dim=-1) + self.bias
        return score, w, group_sums

    def forward_day(self, x: "torch.Tensor", m_row: "torch.Tensor", temperature: float = 1.0):
        """
        单交易日快路径：同一天所有股票共享一个市场状态，门控只需算一次。

        x: [B, N]，m_row: [d_m]（一维）。相比 ``forward`` 省掉 B 份重复的门控前向；
        按日分组训练（每日一个 query）时这是主要加速点。
        """
        contrib = self.experts(x)                        # [B, N]
        group_sums = contrib @ self.group_onehot         # [B, K]
        if self.disable_gate:
            w = torch.ones(self.n_groups, device=group_sums.device)
            score = group_sums.sum(dim=-1) + self.bias
            return score, w, group_sums
        w = self.gate(m_row.unsqueeze(0), temperature)   # [1, K]
        gs = self._std_groups(group_sums) if self.group_norm else group_sums
        score = (gs * w).sum(dim=-1) + self.bias
        return score, w.squeeze(0), group_sums


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------

def listnet_loss(pred: "torch.Tensor", y: "torch.Tensor",
                 temp: float = 1.0, y_scale: float = 10.0) -> "torch.Tensor":
    """按日分组的 ListNet 列表损失（沿用 T036 设置，y_scale 放大头部权重）。"""
    p = torch.softmax(pred / temp, dim=0)
    t = torch.softmax(y * y_scale / temp, dim=0)
    return -(t * torch.log(p + 1e-8)).sum()


def load_balance_loss(w: "torch.Tensor") -> "torch.Tensor":
    """
    负载均衡损失：批内各族平均权重的变异系数平方 ``CV²``。

    为 0 表示各族被均等使用；越大表示门控越偏科。作为软约束加入总损失，
    防止 MoE 经典的"专家坍塌"（少数族吃掉全部权重，其余梯度饿死）。
    """
    mean_w = w.mean(dim=0)
    return mean_w.var(unbiased=False) / (mean_w.mean() ** 2 + 1e-8)


class GateUsageTracker:
    """
    跨日累积门控使用率的 EMA。

    单日 batch 上的 ``load_balance_loss`` 有个致命副作用：同一交易日内市场状态
    向量 ``m`` 恒定 → 当日各股票的门控权重完全一致，于是"批内 CV²"惩罚的正是
    *当天的族间差异*，而这恰恰是我们希望模型学到的东西（结果就是门控被压成
    均匀分布的死门控）。

    正确的约束是：**允许单日偏科，只要求长期平均均衡**。本类维护跨日 EMA，
    惩罚项作用在 EMA 上；梯度只经由当前步的贡献回传（标准 MoE 做法）。
    同时提供 ``diversity`` 项——奖励当日权重偏离其长期均值，直接对抗死门控。
    """

    def __init__(self, n_groups: int, momentum: float = 0.02, device=None):
        self.momentum = float(momentum)
        self.ema = torch.ones(n_groups, device=device)

    def update(self, w_day: "torch.Tensor"):
        """w_day: [B, K]（同日各行相同）。返回 (balance_loss, diversity_bonus)。"""
        cur = w_day.mean(dim=0)
        prev = self.ema.detach()
        self.ema = (1.0 - self.momentum) * prev + self.momentum * cur
        balance = self.ema.var(unbiased=False) / (self.ema.mean() ** 2 + 1e-8)
        diversity = ((cur - prev) ** 2).mean()
        return balance, diversity


def gate_entropy(w: "torch.Tensor") -> "torch.Tensor":
    """门控权重的平均熵（softmax 模式下监控坍塌；越接近 ln K 越健康）。"""
    p = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
    return -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# 对外模型包装（接口对齐 MLFactorModel）
# ---------------------------------------------------------------------------

class NAMGateModel:
    """
    可被回测/选股直接加载的模型包装。

    与 ``MLFactorModel`` 的差异只有一点：打分需要**当日市场状态**。
    调用方在 ``predict`` 前调用 ``set_context_date(date)`` 即可；
    未设置时回退到不晚于"最后一次见过的日期"的最近一行（并给出一次性警告）。
    """

    model_type = 'nam_gate'
    task = 'ranking'

    def __init__(self,
                 feature_names: Optional[List[str]] = None,
                 group_names: Optional[List[str]] = None,
                 group_ids: Optional[np.ndarray] = None,
                 regime_cols: Optional[List[str]] = None,
                 expert_hidden: int = 16,
                 gate_hidden: Sequence[int] = (64, 32),
                 gate_mode: str = 'softmax',
                 gate_scalar_col: str = 'vol_expand',
                 gate_scalar_col2: str = '',
                 device: str = 'auto'):
        self.feature_names: List[str] = list(feature_names or [])
        self.group_names: List[str] = list(group_names or [])
        self.group_ids: Optional[np.ndarray] = None if group_ids is None else np.asarray(group_ids)
        self.regime_cols: List[str] = list(regime_cols or [])
        self.expert_hidden = expert_hidden
        self.gate_hidden = tuple(gate_hidden)
        self.gate_mode = gate_mode
        # gate_mode='scalar'/'dual_scalar' 时驱动门控的 regime 列名（其余模式忽略）
        self.gate_scalar_col = str(gate_scalar_col)
        self.gate_scalar_col2 = str(gate_scalar_col2)

        self.net: Optional[NAMGateNet] = None
        self.feature_importance: Dict[str, float] = {}
        self.is_trained = False
        self.regime_matrix: Optional[pd.DataFrame] = None   # 逐日市场状态
        self.input_mean: Optional[np.ndarray] = None        # 特征标准化统计
        self.input_std: Optional[np.ndarray] = None
        # scheme B：mkt_* 市场级列的 PCA-2 投影（训练段拟合，随模型持久化，
        # 推理时对传入的 regime 矩阵做同构投影 → mkt_pc1/mkt_pc2）
        self.mkt_pca: Optional[Dict[str, np.ndarray]] = None
        self._context_date = None
        self._warned_missing_date = False

        self.device = self._pick_device(device)

    # -- 基础设施 ---------------------------------------------------------

    @staticmethod
    def _pick_device(device: str) -> str:
        if not _TORCH_OK:
            return 'cpu'
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def build(self, d_regime: int, expert_dropout: float = 0.0,
               disable_gate: bool = False, group_norm: bool = False) -> NAMGateNet:
        if not _TORCH_OK:
            raise RuntimeError('PyTorch 未安装，无法构建 NAMGateModel')
        n_factors = len(self.feature_names)
        n_groups = len(self.group_names)
        # scalar/dual_scalar 门控需要把列名解析成索引；缺列必须硬失败（静默回退到
        # 第 0 列会让不同实验的"标量信号"其实不是同一个信号，属 T043/T048 同类口径事故）。
        scalar_idx = 0
        scalar_idx2 = 0
        if self.gate_mode in ('scalar', 'dual_scalar'):
            _need = [self.gate_scalar_col]
            if self.gate_mode == 'dual_scalar':
                _need.append(self.gate_scalar_col2)
            _miss = [c for c in _need if c not in self.regime_cols]
            if _miss:
                raise ValueError(
                    f"gate_mode={self.gate_mode!r} 需要的 regime 列不存在: {_miss}\n"
                    f"  可用列: {self.regime_cols}")
            scalar_idx = self.regime_cols.index(self.gate_scalar_col)
            if self.gate_mode == 'dual_scalar':
                scalar_idx2 = self.regime_cols.index(self.gate_scalar_col2)
        self.net = NAMGateNet(
            n_factors=n_factors, group_ids=self.group_ids, n_groups=n_groups,
            d_regime=d_regime, expert_hidden=self.expert_hidden,
            gate_hidden=self.gate_hidden, gate_mode=self.gate_mode,
            expert_dropout=expert_dropout, disable_gate=disable_gate,
            gate_scalar_idx=scalar_idx, gate_scalar_idx2=scalar_idx2,
            group_norm=group_norm,
        ).to(self.device)
        self.group_norm = bool(group_norm)
        return self.net

    def set_context_date(self, date) -> None:
        """回测/选股在打分前告知当前交易日，用于查市场状态向量。"""
        self._context_date = pd.Timestamp(date)

    def attach_regime(self, regime_matrix: pd.DataFrame) -> None:
        if regime_matrix is None or regime_matrix.empty:
            self.regime_matrix = regime_matrix
            self.regime_cols = list(getattr(regime_matrix, 'columns', []) or [])
            return
        if self.mkt_pca is not None:
            # scheme B：传入矩阵若含 mkt_* 原始列，先投影成 mkt_pc1/mkt_pc2
            # （训练段拟合的 PCA 方向），再挂载 —— 保证推理与训练同构。
            _need = [c for c in self.mkt_pca['mkt_cols'] if c not in regime_matrix.columns]
            if _need:
                raise ValueError(
                    f'模型绑定 mkt PCA 但 regime 矩阵缺列: {_need}\n'
                    f'回测/推理需用 build_regime_matrix(include_mkt=True)')
            regime_matrix = self._apply_mkt_pca(regime_matrix)
        self.regime_matrix = regime_matrix
        self.regime_cols = list(regime_matrix.columns)

    def _apply_mkt_pca(self, mat: pd.DataFrame) -> pd.DataFrame:
        """把 mkt_* 列投影成 2 个主成分（mkt_pc1/mkt_pc2），替换原列。"""
        pca = self.mkt_pca
        mkt_cols = list(pca['mkt_cols'])
        mean = pca['mean']                       # [1, n_mkt]
        proj = pca['proj']                       # [2, n_mkt]
        X = mat[mkt_cols].to_numpy(dtype=np.float64) - mean
        pc = (X @ proj.T).astype(np.float32)     # [n, 2]
        rest = mat.drop(columns=mkt_cols)
        out = rest.copy()
        out['mkt_pc1'] = pc[:, 0]
        out['mkt_pc2'] = pc[:, 1]
        return out

    # -- 推理 -------------------------------------------------------------

    def _regime_vector(self, n_rows: int) -> np.ndarray:
        d = len(self.regime_cols)
        if self.regime_matrix is None or self.regime_matrix.empty:
            return np.zeros((n_rows, d), dtype=np.float32)

        idx = self.regime_matrix.index
        if self._context_date is None:
            if not self._warned_missing_date:
                print('  [NAMGateModel] 警告: 未设置 context_date，回退使用最后一行市场状态')
                self._warned_missing_date = True
            row = self.regime_matrix.iloc[-1].to_numpy(dtype=np.float32)
        else:
            pos = idx.searchsorted(self._context_date, side='right') - 1
            if pos < 0:
                row = np.zeros(d, dtype=np.float32)
            else:
                row = self.regime_matrix.iloc[pos].to_numpy(dtype=np.float32)
        return np.repeat(row[None, :], n_rows, axis=0)

    def predict(self, factors) -> np.ndarray:
        """
        与 ``MLFactorModel.predict`` 同签名：输入已完成当日截面归一化的因子矩阵。

        接受 DataFrame（按 ``feature_names`` 取列）或 ndarray（假定已按序对齐）。
        返回 ``float32[n]`` 打分（越大越好），语义与 ranker 输出一致。
        """
        if not self.is_trained or self.net is None:
            raise ValueError('NAMGateModel 未训练')

        if isinstance(factors, pd.DataFrame):
            X = factors[self.feature_names].to_numpy(dtype=np.float32)
        else:
            X = np.asarray(factors, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.5, posinf=1.0, neginf=0.0)

        if self.input_mean is not None:
            X = (X - self.input_mean) / self.input_std

        M = self._regime_vector(len(X))

        self.net.eval()
        with torch.no_grad():
            xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            mt = torch.as_tensor(M, dtype=torch.float32, device=self.device)
            score, _, _ = self.net(xt, mt)
        return score.detach().cpu().numpy().astype(np.float32)

    # -- 可解释性 ---------------------------------------------------------

    def compute_feature_importance(self, X: np.ndarray, batch: int = 8192) -> Dict[str, float]:
        """
        用形状函数输出的方差 ``Var(f_i(x_i))`` 作为因子重要性。

        语义与 GBDT 的 gain importance 可比（都度量"该因子造成的分数变动幅度"），
        便于与基线横向对照。
        """
        if self.net is None:
            return {}
        if self.input_mean is not None:
            X = (X - self.input_mean) / self.input_std
        self.net.eval()
        acc_sum = np.zeros(len(self.feature_names), dtype=np.float64)
        acc_sq = np.zeros(len(self.feature_names), dtype=np.float64)
        n = 0
        with torch.no_grad():
            for i in range(0, len(X), batch):
                chunk = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=self.device)
                c = self.net.experts(chunk).cpu().numpy().astype(np.float64)
                acc_sum += c.sum(axis=0)
                acc_sq += (c ** 2).sum(axis=0)
                n += len(c)
        if n == 0:
            return {}
        var = acc_sq / n - (acc_sum / n) ** 2
        self.feature_importance = {name: float(max(v, 0.0))
                                   for name, v in zip(self.feature_names, var)}
        return self.feature_importance

    def get_top_factors(self, n: int = 20) -> List[Tuple[str, float]]:
        if not self.feature_importance:
            return []
        return sorted(self.feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def shape_curves(self, factor_names: Sequence[str], n_points: int = 101,
                     lo: float = 0.0, hi: float = 1.0) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        导出指定因子的形状函数曲线 ``{name: (grid, f_i(grid))}``。

        输入域默认 ``[0,1]`` —— 训练前做过当日截面 rank 归一化，因子取值本就在此区间。
        """
        if self.net is None:
            return {}
        grid = np.linspace(lo, hi, n_points, dtype=np.float32)
        if self.input_mean is not None:
            pass  # grid 在原始 rank 空间；下方逐因子做标准化
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.net.eval()
        name_to_idx = {n: i for i, n in enumerate(self.feature_names)}
        with torch.no_grad():
            for name in factor_names:
                if name not in name_to_idx:
                    continue
                i = name_to_idx[name]
                g = grid.copy()
                if self.input_mean is not None:
                    g = (g - self.input_mean[i]) / self.input_std[i]
                x = torch.zeros(n_points, len(self.feature_names), device=self.device)
                x[:, i] = torch.as_tensor(g, device=self.device)
                y = self.net.experts(x)[:, i].cpu().numpy()
                out[name] = (grid.copy(), y.astype(np.float32))
        return out

    def gate_weights_over_time(self, dates: Optional[Sequence] = None) -> pd.DataFrame:
        """
        导出逐交易日的因子族门控权重（热力图数据源）。

        返回 index=交易日、columns=因子族 的 DataFrame。
        """
        if self.net is None or self.regime_matrix is None or self.regime_matrix.empty:
            return pd.DataFrame()
        mat = self.regime_matrix if dates is None else \
            self.regime_matrix.reindex(pd.DatetimeIndex(pd.to_datetime(list(dates))).unique()).ffill()
        self.net.eval()
        with torch.no_grad():
            m = torch.as_tensor(mat.to_numpy(dtype=np.float32), device=self.device)
            w = self.net.gate(m).cpu().numpy()
        return pd.DataFrame(w, index=mat.index, columns=self.group_names)

    # -- 持久化 -----------------------------------------------------------

    def save_model(self, filepath: str) -> None:
        """
        保存为单个 ``.pkl``（与 ``MLFactorModel`` 一致），内含 state_dict 字节流。

        约定文件名以 ``_factor_model.pkl`` 结尾，才能被回测的目录扫描逻辑发现。
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        state = None
        if self.net is not None:
            state = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        payload = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'group_names': self.group_names,
            'group_ids': None if self.group_ids is None else np.asarray(self.group_ids),
            'regime_cols': self.regime_cols,
            'expert_hidden': self.expert_hidden,
            'gate_hidden': list(self.gate_hidden),
            'gate_mode': self.gate_mode,
            'gate_scalar_col': getattr(self, 'gate_scalar_col', 'vol_expand'),
            'gate_scalar_col2': getattr(self, 'gate_scalar_col2', ''),
            'disable_gate': bool(getattr(self, 'disable_gate', False)),
            'group_norm': bool(getattr(self, 'group_norm', False)),
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained,
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'mkt_pca': self.mkt_pca,
            'regime_matrix': self.regime_matrix,
            'state_dict': state,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)

    def load_model(self, filepath: str) -> 'NAMGateModel':
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        if payload.get('model_type') != self.model_type:
            raise ValueError(f"不是 NAMGateModel 存档: {payload.get('model_type')}")

        self.feature_names = list(payload['feature_names'])
        self.group_names = list(payload['group_names'])
        self.group_ids = payload['group_ids']
        self.regime_cols = list(payload['regime_cols'])
        self.expert_hidden = payload['expert_hidden']
        self.gate_hidden = tuple(payload['gate_hidden'])
        self.gate_mode = payload['gate_mode']
        self.gate_scalar_col = str(payload.get('gate_scalar_col', 'vol_expand'))
        self.gate_scalar_col2 = str(payload.get('gate_scalar_col2', ''))
        self.disable_gate = bool(payload.get('disable_gate', False))
        self.group_norm = bool(payload.get('group_norm', False))
        self.feature_importance = payload.get('feature_importance', {})
        self.input_mean = payload.get('input_mean')
        self.input_std = payload.get('input_std')
        self.mkt_pca = payload.get('mkt_pca')
        self.regime_matrix = payload.get('regime_matrix')
        self.is_trained = payload.get('is_trained', False)

        if payload.get('state_dict') is not None:
            self.build(d_regime=len(self.regime_cols), disable_gate=self.disable_gate,
                       group_norm=self.group_norm)
            self.net.load_state_dict(payload['state_dict'])
            self.net.to(self.device).eval()
        return self

    @classmethod
    def is_nam_gate_archive(cls, filepath: str) -> bool:
        """轻量嗅探：判断某个 pkl 是否为 NAMGateModel 存档（回测加载分派用）。"""
        try:
            with open(filepath, 'rb') as f:
                payload = pickle.load(f)
            return isinstance(payload, dict) and payload.get('model_type') == cls.model_type
        except Exception:
            return False


def save_sidecar_metadata(model: NAMGateModel, out_dir: str) -> None:
    """把特征名/分组/regime 列另存为 JSON，便于人工检查与跨语言复用。"""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'feature_names.json'), 'w', encoding='utf-8') as f:
        json.dump(model.feature_names, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, 'factor_groups.json'), 'w', encoding='utf-8') as f:
        mapping = {g: [] for g in model.group_names}
        for name, gid in zip(model.feature_names, np.asarray(model.group_ids).tolist()):
            mapping[model.group_names[gid]].append(name)
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, 'regime_cols.json'), 'w', encoding='utf-8') as f:
        json.dump(model.regime_cols, f, ensure_ascii=False, indent=2)
