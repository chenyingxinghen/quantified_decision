import unittest

import numpy as np
import pandas as pd

from core.factors.multi_objective_labels import (
    MultiObjectiveLabelBuilder,
    cross_sectional_rank_targets,
    orthogonalize_labels,
    diagnose_label_orthogonality,
)


def _make_random_walk(n: int, seed: int, start: float = 10.0) -> pd.DataFrame:
    """生成独立同分布日收益的价格序列（无结构性相关，便于检验正交性）。"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, size=n)
    close = start * np.cumprod(1.0 + rets)
    open_ = np.empty(n)
    open_[0] = start
    open_[1:] = close[:-1]  # 当日开盘≈前收（简化）
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n).astype(str),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(1_000, 10_000, size=n).astype(float),
        "amount": rng.integers(1_000_000, 10_000_000, size=n).astype(float),
    })


class MultiObjectiveLabelTests(unittest.TestCase):
    def test_labels_start_at_next_open_and_do_not_use_current_close(self):
        data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=8).astype(str),
            "open": [10, 20, 21, 22, 23, 24, 25, 26],
            "high": [11, 21, 22, 23, 24, 25, 26, 27],
            "low": [9, 19, 20, 21, 22, 23, 24, 25],
            "close": [100, 20, 22, 21, 24, 23, 26, 27],
            "volume": [100] * 8,
            "amount": [1000] * 8,
        })
        labels = MultiObjectiveLabelBuilder((2,), risk_horizon=3).build(data)
        # t=0 的入口必须是 t+1 open=20，退出是 t+2 close=22。
        self.assertAlmostEqual(labels.loc[0, "y_ret_2d"], 0.10, places=6)
        self.assertTrue(np.isnan(labels.loc[7, "y_ret_2d"]))

    def test_risk_and_tradability_labels(self):
        data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=7).astype(str),
            "open": [10, 10, 10, 10, 10, 10, 10],
            "high": [10, 10, 10, 10, 10, 10, 10],
            "low": [10, 10, 8, 9, 10, 10, 10],
            "close": [10, 10, 8, 9, 10, 10, 10],
            "volume": [100, 100, 0, 100, 100, 100, 100],
            "amount": [1000] * 7,
        })
        labels = MultiObjectiveLabelBuilder((2,), risk_horizon=3).build(data)
        self.assertAlmostEqual(labels.loc[0, "y_mdd_3d"], -0.2, places=6)
        self.assertEqual(labels.loc[0, "y_tradable_3d"], 0.0)
        self.assertGreaterEqual(labels.loc[0, "y_downvol_3d"], 0.0)

    def test_cross_sectional_risk_direction_is_reversed(self):
        labels = pd.DataFrame({
            "date": ["2024-01-01"] * 3,
            "y_ret": [0.01, 0.02, 0.03],
            "y_vol": [0.3, 0.2, 0.1],
        })
        ranked = cross_sectional_rank_targets(
            labels, ["y_ret", "y_vol"], risk_cols=["y_vol"]
        )
        self.assertGreater(ranked.loc[2, "rank_y_ret"], ranked.loc[0, "rank_y_ret"])
        self.assertGreater(ranked.loc[2, "rank_y_vol"], ranked.loc[0, "rank_y_vol"])


class OrthogonalityTests(unittest.TestCase):
    def test_orthogonal_legs_are_constructed(self):
        data = _make_random_walk(80, seed=1)
        labels = MultiObjectiveLabelBuilder((5, 20, 60), risk_horizon=20).build(data)
        for col in ("y_ret_leg_1_5d", "y_ret_leg_6_20d", "y_ret_leg_21_60d"):
            self.assertIn(col, labels.columns)

    def test_orthogonal_legs_have_near_zero_cross_correlation(self):
        """非重叠前向收益腿在独立收益下应近似正交（|corr| 很小）。"""
        rng = np.random.default_rng(42)
        # 单只股票的长期独立收益序列 → 不同时窗收益互不相关。
        n = 2000
        rets = rng.normal(0.0003, 0.02, size=n)
        close = 10.0 * np.cumprod(1.0 + rets)
        data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n).astype(str),
            "open": np.concatenate([[10.0], close[:-1]]),
            "high": np.maximum(np.concatenate([[10.0], close[:-1]]), close) * 1.01,
            "low": np.minimum(np.concatenate([[10.0], close[:-1]]), close) * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 10_000, n).astype(float),
            "amount": rng.integers(1_000_000, 10_000_000, n).astype(float),
        })
        labels = MultiObjectiveLabelBuilder((5, 20, 60), risk_horizon=20).build(data)
        legs = ["y_ret_leg_1_5d", "y_ret_leg_6_20d", "y_ret_leg_21_60d"]
        corr = labels[legs].corr().to_numpy()
        off = corr[~np.eye(len(legs), dtype=bool)]
        self.assertLess(np.nanmax(np.abs(off)), 0.05,
                        msg=f"正交收益腿相关过大: {off}")

    def test_nested_cumulative_returns_are_correlated(self):
        """对照：嵌套累积收益（旧目标）相关性强，证明需要正交化。"""
        rng = np.random.default_rng(7)
        n = 2000
        rets = rng.normal(0.0003, 0.02, size=n)
        close = 10.0 * np.cumprod(1.0 + rets)
        data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n).astype(str),
            "open": np.concatenate([[10.0], close[:-1]]),
            "high": np.maximum(np.concatenate([[10.0], close[:-1]]), close) * 1.01,
            "low": np.minimum(np.concatenate([[10.0], close[:-1]]), close) * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 10_000, n).astype(float),
            "amount": rng.integers(1_000_000, 10_000_000, n).astype(float),
        })
        labels = MultiObjectiveLabelBuilder((5, 20, 60), risk_horizon=20).build(data)
        corr = labels[["y_ret_5d", "y_ret_20d", "y_ret_60d"]].corr().abs().to_numpy()
        off = corr[~np.eye(3, dtype=bool)]
        self.assertGreater(np.nanmax(off), 0.3,
                           msg="嵌套收益本应高相关，作为正交化对照基准")

    def test_orthogonalize_labels_makes_objectives_orthogonal(self):
        """多股票截面下，Gram-Schmidt 残差化后目标近似正交。"""
        n_stocks, n_days = 60, 120
        stocks = {}
        for s in range(n_stocks):
            stocks[f"{s:06d}.SZ"] = _make_random_walk(n_days, seed=100 + s)
        builder = MultiObjectiveLabelBuilder((5, 20, 60), risk_horizon=20)
        universe = builder.build_universe(stocks)
        target_cols = ["y_ret_5d", "y_ret_20d", "y_ret_60d", "y_mdd_20d"]
        diag_before = diagnose_label_orthogonality(universe, target_cols)
        orth = orthogonalize_labels(universe, target_cols)
        orth_cols = [f"orth_{c}" for c in target_cols]
        diag_after = diagnose_label_orthogonality(orth, orth_cols)
        # 正交化前存在强共线；正交化后条件数应显著下降、最大相关趋近 0。
        self.assertGreater(diag_before["condition_number"], diag_after["condition_number"])
        self.assertLess(diag_after["max_abs_offdiag_corr"], 0.1,
                        msg=f"正交化后仍有强相关: {diag_after['max_abs_offdiag_corr']}")


class NoLeakageTests(unittest.TestCase):
    def test_verify_no_lookahead_passes(self):
        data = _make_random_walk(200, seed=3)
        builder = MultiObjectiveLabelBuilder((5, 20, 60), risk_horizon=20)
        self.assertTrue(builder.verify_no_lookahead(data))

    def test_sharpe_uses_matching_horizon_drawdown(self):
        """y_sharpe_5d 的分母应是 5 日回撤，而非全局 20 日回撤（修复泄露）。"""
        # 构造：前 5 日大幅回撤后收复，使 5 日回撤 >> 20 日回撤。
        # 价格序列（日收盘），t=0 起。前 5 日下挫 20%，之后反弹，
        # 但在第 12 日（属 20 日窗口、超出 5 日窗口）再次出现更深回撤，
        # 使 20 日最大回撤明显大于 5 日最大回撤。
        close = np.array([10, 8, 8, 8, 8, 12, 13, 14, 15, 16,
                          17, 18, 9, 18, 19, 19.5, 20, 20.5, 21, 21.5, 22],
                         dtype=float)
        # 让 t=1..5 明显下挫（5 日回撤大），之后持续上涨（20 日回撤小）。
        open_ = np.empty_like(close)
        open_[0] = 10.0
        open_[1:] = close[:-1]
        data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=len(close)).astype(str),
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": np.full(len(close), 100.0),
            "amount": np.full(len(close), 1000.0),
        })
        b_match = MultiObjectiveLabelBuilder((5, 20, 60), risk_horizon=20,
                                             use_matching_risk_for_sharpe=True).build(data)
        b_global = MultiObjectiveLabelBuilder((5, 20, 60), risk_horizon=20,
                                              use_matching_risk_for_sharpe=False).build(data)
        # 同长分母与全局分母不一致 → 两个 sharpe_5d 不相等，证明已切换为同长回撤。
        self.assertNotAlmostEqual(
            float(b_match.loc[0, "y_sharpe_5d"]),
            float(b_global.loc[0, "y_sharpe_5d"]),
            places=4,
            msg="y_sharpe_5d 未从固定 20 日回撤切换为同长回撤（仍存在跨期泄露）",
        )

    def test_rank_targets_require_date_for_leak_safety(self):
        labels = pd.DataFrame({
            "y_ret": [0.01, 0.02, 0.03],
            "y_vol": [0.3, 0.2, 0.1],
        })
        with self.assertRaises(ValueError):
            cross_sectional_rank_targets(labels, ["y_ret"], verify_leak_free=True)


if __name__ == "__main__":
    unittest.main()
