import unittest

import numpy as np
import pandas as pd

from core.factors.multi_objective_labels import (
    MultiObjectiveLabelBuilder,
    cross_sectional_rank_targets,
)


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


if __name__ == "__main__":
    unittest.main()
