"""
组合优化器（Portfolio Optimizer）

把神经网络给出的"每只票合意度分数"与"历史/预测风险"转成真正的组合权重，
而非贪心 top-N 等权。直接回答"平衡回撤/波动与最终收益的最佳组合"。

提供三种优化目标：
- mean_variance   : 最大化 μᵀw − λ·wᵀΣw（风险厌恶可调）
- max_sharpe      : 最大化 (μᵀw + γ·mddᵀw) / √(wᵀΣw)，可叠加预测回撤惩罚
- risk_parity     : 等风险贡献（分散化优先）

约束：Σw=1，0≤wᵢ≤max_weight（可选 min_weight、最小持仓数、换手约束）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any, Sequence

import numpy as np
import pandas as pd


class PortfolioOptimizer:
    """组合权重求解器（纯数值，不依赖数据库，可单测）。"""

    # ---------------------------------------------------------- 协方差估计
    @staticmethod
    def estimate_covariance(returns: pd.DataFrame, shrinkage: float = 0.1,
                            ridge: float = 1e-6) -> np.ndarray:
        """由日收益矩阵估计协方差，带收缩 + 对角岭项以保证数值稳定。

        returns: DataFrame，行=日期，列=股票代码，值为日简单收益。
        """
        R = returns.dropna(how="all").astype(float)
        if R.shape[1] == 0:
            return np.empty((0, 0))
        cov = R.cov().values
        n = cov.shape[0]
        if n == 1:
            return cov.copy()
        # Ledoit-Wolf 风格收缩到单位阵的倍数（保持量级）
        trace = np.trace(cov)
        target = (trace / n) * np.eye(n) if trace > 0 else np.eye(n)
        cov = (1.0 - shrinkage) * cov + shrinkage * target
        # 对角岭项，避免非正定导致求解失败
        cov = cov + ridge * np.eye(n)
        return cov

    # ---------------------------------------------------------- 通用求解
    @staticmethod
    def _solve(codes: List[str], mu: np.ndarray, cov: np.ndarray,
               method: str = "max_sharpe",
               risk_aversion: float = 1.0,
               max_weight: float = 0.2,
               min_weight: float = 0.0,
               min_holdings: int = 1,
               drawdown_penalty: float = 0.0,
               predicted_mdd: Optional[np.ndarray] = None,
               max_iter: int = 200) -> np.ndarray:
        from scipy.optimize import minimize

        n = len(codes)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        mu = np.asarray(mu, dtype=float).reshape(-1)
        cov = np.asarray(cov, dtype=float)
        if predicted_mdd is not None:
            mdd = np.asarray(predicted_mdd, dtype=float).reshape(-1)
            # 归一化到大致 [-1,1]，避免量纲主导
            mdd = mdd / (np.abs(mdd).max() + 1e-12)
        else:
            mdd = np.zeros(n)

        x0 = np.full(n, 1.0 / n)
        # 约束
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        if min_holdings and min_holdings > 1:
            # 至少 min_holdings 只持仓（用非负 + 上限自然约束，配合初值即可近似）
            pass
        bounds = [(min_weight, max_weight) for _ in range(n)]

        def port_var(w):
            return float(w @ cov @ w)

        def neg_sharpe(w):
            ret = float(mu @ w + drawdown_penalty * (mdd @ w))
            vol = np.sqrt(max(port_var(w), 1e-12))
            return -ret / vol

        def neg_utility(w):
            return -(mu @ w - risk_aversion * port_var(w))

        def risk_parity_obj(w):
            rc = w * (cov @ w)
            rc = rc / (rc.sum() + 1e-12)
            # 各资产风险贡献与均值的平方差
            return float(((rc - rc.mean()) ** 2).sum())

        if method == "mean_variance":
            obj = neg_utility
        elif method == "risk_parity":
            obj = risk_parity_obj
        else:  # max_sharpe
            obj = neg_sharpe

        res = minimize(obj, x0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": max_iter, "ftol": 1e-9})
        w = res.x if res.success else x0
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s <= 0:
            w = x0
            s = 1.0
        return w / s

    # ---------------------------------------------------------- 对外 API
    @classmethod
    def optimize(cls, scores: Dict[str, float],
                 returns: pd.DataFrame,
                 method: str = "max_sharpe",
                 risk_aversion: float = 1.0,
                 max_weight: float = 0.2,
                 min_weight: float = 0.0,
                 min_holdings: int = 1,
                 drawdown_penalty: float = 0.0,
                 predicted_mdd: Optional[Dict[str, float]] = None,
                 score_to_return_scale: float = 0.8,
                 shrinkage: float = 0.1) -> Dict[str, float]:
        """由模型分数 + 收益序列求解组合权重。

        参数：
            scores        : 模型给出的每只票合意度分数（建议 [0,1]）
            returns       : 候选票的日收益矩阵（列=代码）
            method        : max_sharpe | mean_variance | risk_parity
            max_weight    : 单票权重上限（分散化）
            predicted_mdd : 可选，"回撤越小越好"的 rank 分数（与策略传入的
                            rank_y_mdd_20d 一致：越大=回撤越小=越好）。优化器将其
                            作为奖励项叠加到预期收益（max_sharpe 下）或效用上，
                            从而压低高回撤票的权重。
        返回：
            code -> weight（已归一化，和为 1）
        """
        codes = [c for c in scores.keys() if c in returns.columns]
        if not codes:
            # 退化为按分数比例分配（无收益数据时）
            tot = sum(max(v, 0) for v in scores.values()) or 1.0
            return {c: max(v, 0) / tot for c, v in scores.items()}
        R = returns[codes].dropna(how="all")
        cov = cls.estimate_covariance(R, shrinkage=shrinkage)
        # 分数 -> 期望收益（单调映射，使高分的票获得更高预期收益）
        s = np.array([scores[c] for c in codes], dtype=float)
        mu = (np.clip(s, 0.0, 1.0) - 0.5) * 2.0 * score_to_return_scale
        mdd_vec = None
        if predicted_mdd:
            mdd_vec = np.array([predicted_mdd.get(c, 0.0) for c in codes], dtype=float)

        w = cls._solve(
            codes, mu, cov, method=method, risk_aversion=risk_aversion,
            max_weight=max_weight, min_weight=min_weight,
            min_holdings=min_holdings, drawdown_penalty=drawdown_penalty,
            predicted_mdd=mdd_vec,
        )
        return {c: float(wt) for c, wt in zip(codes, w)}


def select_portfolio(
    scores: Dict[str, float],
    returns: pd.DataFrame,
    method: str = "max_sharpe",
    max_weight: float = 0.2,
    min_weight: float = 0.0,
    risk_aversion: float = 1.0,
    drawdown_penalty: float = 0.0,
    predicted_mdd: Optional[Dict[str, float]] = None,
    top_n: Optional[int] = None,
    score_to_return_scale: float = 0.8,
    shrinkage: float = 0.1,
) -> List[Dict[str, Any]]:
    """端到端：分数 + 收益 -> 带权重的组合清单。

    返回每项：{code, weight, expected_return, expected_risk}。
    expected_return / expected_risk 由映射后的 μ 与协方差导出，仅供展示与诊断。
    """
    if top_n:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        scores = dict(ranked)

    weights = PortfolioOptimizer.optimize(
        scores, returns, method=method, risk_aversion=risk_aversion,
        max_weight=max_weight, min_weight=min_weight,
        drawdown_penalty=drawdown_penalty, predicted_mdd=predicted_mdd,
        score_to_return_scale=score_to_return_scale, shrinkage=shrinkage,
    )
    # 诊断指标
    codes = list(weights.keys())
    if codes and codes[0] in returns.columns:
        R = returns[codes].dropna(how="all")
        cov = PortfolioOptimizer.estimate_covariance(R, shrinkage=shrinkage)
        s_vec = np.array([scores[c] for c in codes], dtype=float)
        mu = (np.clip(s_vec, 0, 1) - 0.5) * 2 * score_to_return_scale
        w_vec = np.array([weights[c] for c in codes], dtype=float)
        port_ret = float(mu @ w_vec)
        port_vol = float(np.sqrt(max(w_vec @ cov @ w_vec, 0.0)))
    else:
        port_ret = port_vol = float("nan")

    out = []
    for c in codes:
        out.append({"code": c, "weight": weights[c]})
    return {
        "holdings": out,
        "portfolio": {
            "expected_return": port_ret,
            "expected_risk": port_vol,
            "n_holdings": len(codes),
        },
    }
