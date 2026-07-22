"""
PortfolioOptimizer / select_portfolio 单元测试

不依赖数据库与 torch，纯数值验证组合优化器的核心性质：
- 权重非负、和为 1、受 max_weight 约束
- max_sharpe 下高分票权重更高（方向正确）
- 无收益数据时退化为按分数比例分配
- risk_parity 下风险贡献大致均衡
- select_portfolio 返回结构正确（holdings / portfolio 诊断字段）
"""
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neural.portfolio import PortfolioOptimizer, select_portfolio


def _make_returns(n=8, days=200, seed=0):
    """构造日收益矩阵：前几只票预期收益更高、相互有一定相关性。"""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.01, size=days)
    codes = [f"{i:06d}" for i in range(n)]
    R = pd.DataFrame(index=range(days), columns=codes, dtype=float)
    for i, c in enumerate(codes):
        # 票 i 的 alpha 随 i 递减，使预期收益有区分度
        alpha = 0.0008 * (n - i) / n
        # 与基准的 beta 随 i 变化，制造相关结构
        beta = 0.5 + 0.5 * (i / max(n - 1, 1))
        noise = rng.normal(0, 0.008, size=days)
        R[c] = alpha + beta * base + noise
    return R


def _scores(n=8):
    # 票 0 分数最高，票 n-1 最低
    return {f"{i:06d}": 0.9 - 0.9 * i / max(n - 1, 1) for i in range(n)}


def test_weights_basic_properties():
    R = _make_returns()
    scores = _scores()
    w = PortfolioOptimizer.optimize(scores, R, method="max_sharpe", max_weight=0.2)
    codes = list(w.keys())
    arr = np.array([w[c] for c in codes])
    # 非负
    assert np.all(arr >= -1e-9), "权重应非负"
    # 和为 1
    assert abs(arr.sum() - 1.0) < 1e-6, f"权重和应为 1，实际 {arr.sum()}"
    # 受上限约束
    assert arr.max() <= 0.2 + 1e-9, f"单票权重应 <= max_weight，实际 {arr.max()}"


def test_max_sharpe_prefers_higher_score():
    R = _make_returns()
    scores = _scores()
    w = PortfolioOptimizer.optimize(scores, R, method="max_sharpe", max_weight=0.2)
    codes = list(scores.keys())
    w0 = w[codes[0]]   # 分数最高
    wlast = w[codes[-1]]  # 分数最低
    assert w0 > wlast, f"max_sharpe 下高分票权重应更高: w0={w0}, wlast={wlast}"


def test_single_asset():
    w = PortfolioOptimizer.optimize({"000001": 0.9}, pd.DataFrame(), method="max_sharpe")
    assert abs(w["000001"] - 1.0) < 1e-9


def test_fallback_without_returns():
    """无收益数据时退化为按分数比例分配，且和为 1。"""
    scores = {"a": 0.8, "b": 0.2, "c": 0.0}
    w = PortfolioOptimizer.optimize(scores, pd.DataFrame())
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["a"] > w["b"] > w["c"], "无收益数据应按分数比例"
    assert w["c"] == 0.0, "零分票不应分配权重"


def test_risk_parity_balances_contrib():
    R = _make_returns(n=6, days=300, seed=3)
    scores = {c: 0.5 for c in R.columns}  # 分数相同 -> 纯分散化
    w = PortfolioOptimizer.optimize(scores, R, method="risk_parity", max_weight=0.5)
    cov = PortfolioOptimizer.estimate_covariance(R)
    wv = np.array([w[c] for c in R.columns])
    rc = wv * (cov @ wv)
    rc = rc / (rc.sum() + 1e-12)
    # 风险贡献应相对均衡（与均值的最大偏离不应过大）
    assert rc.max() - rc.min() < 0.25, f"risk_parity 风险贡献应均衡: {rc}"


def test_select_portfolio_structure():
    R = _make_returns()
    scores = _scores()
    out = select_portfolio(scores, R, method="max_sharpe", max_weight=0.2, top_n=5)
    assert "holdings" in out and "portfolio" in out
    assert isinstance(out["holdings"], list)
    assert len(out["holdings"]) <= 5, "top_n 应限制持仓数"
    for h in out["holdings"]:
        assert set(["code", "weight"]).issubset(h.keys())
    port = out["portfolio"]
    assert "expected_return" in port and "expected_risk" in port
    assert "n_holdings" in port
    total = sum(h["weight"] for h in out["holdings"])
    assert abs(total - 1.0) < 1e-6, "组合权重和应为 1"


def test_drawdown_penalty_lowers_high_mdd_weight():
    """drawdown_penalty 越大，预测回撤越大的票权重应越低（方向正确）。

    用近独立的收益序列使各票风险近似相同，把"回撤惩罚"从"风险分散"效应中
    隔离出来。直接比对单一阈值容易被 SLSQP 局部极小干扰，因此改为扫描惩罚
    系数，断言高回撤票（c0）权重随惩罚单调不增、且在强惩罚下显著低于无惩罚。
    """
    rng = np.random.default_rng(7)
    n = 8
    days = 600
    R = pd.DataFrame(
        rng.normal(0, 0.01, size=(days, n)),
        columns=[f"{i:06d}" for i in range(n)],
    )
    scores = {c: 0.8 for c in R.columns}  # 分数相同，仅回撤不同
    # 注意：predicted_mdd 的语义是"回撤越小越好的 rank"（与策略传入的
    # rank_y_mdd_20d 一致）：值越大=回撤越小=越好。这里让 c0 拥有最低 rank
    # 值（即预测回撤最大/最差），验证惩罚会压低它的权重。
    predicted_mdd = {c: (0.9 if i != 0 else 0.1) for i, c in enumerate(R.columns)}
    c0 = R.columns[0]

    weights_by_penalty = {}
    for pen in [0.0, 1.0, 5.0, 20.0]:
        w = PortfolioOptimizer.optimize(
            scores, R, method="max_sharpe", max_weight=0.2,
            drawdown_penalty=pen, predicted_mdd=predicted_mdd,
        )
        weights_by_penalty[pen] = w[c0]

    # 单调不增
    pens = [0.0, 1.0, 5.0, 20.0]
    for a, b in zip(pens, pens[1:]):
        assert weights_by_penalty[a] >= weights_by_penalty[b] - 1e-9, (
            f"回撤惩罚增大时 c0 权重应不增: pen={a}->{b}, "
            f"w={weights_by_penalty[a]} -> {weights_by_penalty[b]}"
        )
    # 强惩罚下显著低于无惩罚
    assert weights_by_penalty[20.0] < weights_by_penalty[0.0] - 1e-3, (
        f"强惩罚应显著降低高回撤票权重: {weights_by_penalty[0.0]} -> {weights_by_penalty[20.0]}"
    )


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
