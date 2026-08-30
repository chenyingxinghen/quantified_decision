# -*- coding: utf-8 -*-
"""
双轴 NAM 模型（T139 架构方向）：纵截面市场时序调制横截面个股打分。

横截面轴（与 T045 生产基线逐位相同）：
    contrib = FactorExpertBank(x)            # [B, 228]，每个单因子一个 shape 函数
纵截面轴（新）：
    LSTM(market_window[20, 30]) -> h_t       # 市场状态时序表示
    m_t = 1 + tanh(MLP(h_t))                 # [228]，逐因子调制权重
    **零初始化保证 m_t ≡ 1 起步 → 训练起点退化为纯加性 NAM（安全网）**

合并：
    score_i = Σ_k m_t,k · contrib_i,k + bias

与 gate 家族的区别：
    - 调制粒度：228 个单因子权重（gate 是 13 个族权重）
    - 输入表示：LSTM 20 日时序（gate 是当日静态 regime 向量）
    - 通路：m_t 逐日共享但不含 softmax 归一化，安全网零初始化

成败判据（训练后检查）：
    - m_t 的时间变异度 std_t(m_t)：接近 0 → 学成静态重加权，未利用市场信息 → 判负
    - OOS 增益（熊市主判 + 随机零假设）相对 T045 基线
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core.factors.nam_gate_model import FactorExpertBank


class DualAxisNAM(nn.Module):
    """双轴 NAM 网络：横截面专家 + 纵截面 LSTM 调制。

    Parameters
    ----------
    n_factors : 单因子数（228）
    d_regime  : 市场状态向量维度（30）
    expert_hidden : 专家隐藏层宽度（与生产一致 16）
    hidden    : LSTM 隐藏层宽度（32）
    window    : 市场时序窗口（交易日，20）
    """

    def __init__(self, n_factors: int, d_regime: int, expert_hidden: int = 16,
                 hidden: int = 32, window: int = 20):
        super().__init__()
        self.n_factors = n_factors
        self.d_regime = d_regime
        self.hidden = hidden
        self.window = window

        # 横截面轴：与生产 T045 完全相同的专家库
        self.experts = FactorExpertBank(n_factors, expert_hidden)

        # 纵截面轴：市场时序 → 逐因子调制权重
        self.lstm = nn.LSTM(d_regime, hidden, batch_first=True)
        self.mod = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_factors),
        )
        # 安全网：输出层零初始化 → m_t = 1 + tanh(0) = 1
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """x: [B, N] 个股因子（当日截面归一化）；w: [window, d_regime] 该日市场窗口。

        Returns
        -------
        score : [B]
        m     : [N] 逐因子调制权重（每日一个，供变异度诊断）
        contrib : [B, N] 专家贡献（供分解诊断）
        """
        contrib = self.experts(x)                      # [B, N]
        _, (h, _) = self.lstm(w.unsqueeze(0))          # [1, 1, hidden]
        m = 1.0 + torch.tanh(self.mod(h[0, 0]))        # [N]
        score = contrib @ m + self.bias                # [B]
        return score, m, contrib


class DualAxisModel:
    """回测/推理可加载的双轴模型包装（接口对齐 NAMGateModel）。

    保存内容（目录下并存）：
        dual_axis_model.pkl      — torch 完整状态（含 feature_names/regime_cols/norm_stats）
        feature_names.json       — 与生产一致的 sidecar
        norm_stats.pkl           — 截面归一化统计量（回测端校验加载）
        regime_cols.json         — 市场状态列顺序（回测端重建窗口用）
    """

    def __init__(self, net: DualAxisNAM, feature_names: List[str],
                 regime_cols: List[str], norm_stats: Optional[dict] = None):
        self.net = net
        self.feature_names = list(feature_names)
        self.regime_cols = list(regime_cols)
        self.norm_stats = norm_stats
        self.regime_matrix: Optional[pd.DataFrame] = None
        self._context_date = None

    # ---- 回测接口 ----
    def set_context_date(self, date) -> None:
        """回测日循环显式告知当前交易日（与 NAMGateModel 接口对齐）。"""
        self._context_date = pd.Timestamp(date)

    # ---- 市场窗口 ----
    def attach_regime(self, regime_matrix: pd.DataFrame) -> None:
        """回测端注入逐日市场状态矩阵（DatetimeIndex，列序必须与 regime_cols 一致）。"""
        if self.regime_cols and not set(self.regime_cols).issubset(regime_matrix.columns):
            miss = [c for c in self.regime_cols if c not in regime_matrix.columns]
            raise ValueError(f'regime 矩阵缺列: {miss}')
        self.regime_matrix = regime_matrix[self.regime_cols] if self.regime_cols \
            else regime_matrix

    def _day_window(self, day) -> np.ndarray:
        """取 day 及前 window-1 个交易日的 regime 窗口 [window, d_regime]。"""
        rm = self.regime_matrix
        pos = rm.index.get_indexer([pd.Timestamp(day)], method='pad')[0]
        if pos < 0:
            raise ValueError(f'regime 矩阵无 {day} 当日或更早数据')
        win = self.net.window
        start = max(0, pos - win + 1)
        arr = rm.iloc[start:pos + 1].to_numpy(dtype=np.float32)
        if arr.shape[0] < win:                       # 训练起始段不足窗口 → 首行前填
            arr = np.concatenate([np.repeat(arr[:1], win - arr.shape[0], axis=0), arr])
        return arr

    # ---- 打分 ----
    def _score_block(self, x: torch.Tensor, w: np.ndarray) -> torch.Tensor:
        """x: [B, N] -> score [B]（单日共享调制向量）"""
        contrib = self.net.experts(x)                 # [B, N]
        with torch.no_grad():
            _, (h, _) = self.net.lstm(
                torch.from_numpy(w).unsqueeze(0))     # [1, 1, hidden]
            m = 1.0 + torch.tanh(self.net.mod(h[0, 0]))
        return contrib @ m + self.net.bias

    def predict(self, factors: np.ndarray,
                dates: Optional[Sequence] = None) -> np.ndarray:
        """factors: [n, N]（已截面归一化）；dates: [n] 或 None（用最近窗口）。"""
        if self.regime_matrix is None or self.regime_matrix.empty:
            raise RuntimeError('双轴模型推理需要 attach_regime(regime_matrix)')
        self.net.eval()
        X = np.asarray(factors, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float64)
        if dates is None:
            day = self._context_date if self._context_date is not None \
                else self.regime_matrix.index[-1]
            w = self._day_window(day)
            with torch.no_grad():
                out[:] = self._score_block(torch.from_numpy(X.copy()), w).numpy()
            return out
        dts = pd.to_datetime(pd.Series(list(dates)))
        uniq = pd.DatetimeIndex(dts.unique()).sort_values()
        for d in uniq:
            mask = (dts == d).to_numpy()
            w = self._day_window(d)
            with torch.no_grad():
                out[mask] = self._score_block(
                    torch.from_numpy(X[mask]), w).numpy()
        return out

    # ---- 保存 / 加载 ----
    def save(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        pkl = os.path.join(out_dir, 'dual_axis_model.pkl')
        payload = {
            'state_dict': self.net.state_dict(),
            'n_factors': self.net.n_factors,
            'd_regime': self.net.d_regime,
            'expert_hidden': self.net.experts.hidden,
            'hidden': self.net.hidden,
            'window': self.net.window,
            'feature_names': self.feature_names,
            'regime_cols': self.regime_cols,
            'norm_stats': self.norm_stats,
        }
        torch.save(payload, pkl)
        with open(os.path.join(out_dir, 'feature_names.json'), 'w', encoding='utf-8') as f:
            json.dump({'features': self.feature_names}, f)
        with open(os.path.join(out_dir, 'regime_cols.json'), 'w', encoding='utf-8') as f:
            json.dump({'regime_cols': self.regime_cols, 'window': self.net.window}, f)
        if self.norm_stats is not None:
            import pickle
            with open(os.path.join(out_dir, 'norm_stats.pkl'), 'wb') as f:
                pickle.dump(self.norm_stats, f)
        return pkl

    @classmethod
    def load(cls, pkl_path: str) -> 'DualAxisModel':
        payload = torch.load(pkl_path, map_location='cpu', weights_only=False)
        net = DualAxisNAM(payload['n_factors'], payload['d_regime'],
                          expert_hidden=payload.get('expert_hidden', 16),
                          hidden=payload.get('hidden', 32),
                          window=payload.get('window', 20))
        net.load_state_dict(payload['state_dict'])
        return cls(net, payload['feature_names'], payload['regime_cols'],
                   norm_stats=payload.get('norm_stats'))
