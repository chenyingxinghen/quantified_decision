import unittest

import numpy as np

from core.factors.feature_selector import CrossSectionalFeatureSelector
from core.factors.train_ml_model import _rank_finite_to_unit_interval
from config.factor_config import OptimizationConfig


class FeatureSelectorTests(unittest.TestCase):
    def test_feature_limit_is_200(self):
        self.assertEqual(OptimizationConfig.N_FEATURES_TO_SELECT, 200)

    def test_selector_prefers_predictive_and_removes_redundant_features(self):
        rng = np.random.default_rng(7)
        y = rng.normal(size=500)
        predictive = y + rng.normal(scale=0.05, size=500)
        duplicate = predictive * 1.01
        noise = rng.normal(size=500)
        constant = np.ones(500)
        X = np.column_stack([predictive, duplicate, noise, constant])
        names = ["predictive", "duplicate", "noise", "constant"]

        selector = CrossSectionalFeatureSelector(max_features=2, corr_threshold=0.95)
        _, selected = selector.fit_transform(X, names, y)

        self.assertIn("predictive", selected)
        self.assertNotIn("duplicate", selected)
        self.assertNotIn("constant", selected)

    def test_multi_target_weights_change_priority(self):
        rng = np.random.default_rng(11)
        y1 = rng.normal(size=400)
        y2 = rng.normal(size=400)
        X = np.column_stack([y1, y2])
        names = ["return_factor", "risk_factor"]
        targets = np.column_stack([y1, y2])

        selector = CrossSectionalFeatureSelector(max_features=1, corr_threshold=0.99)
        _, selected = selector.fit_transform(
            X, names, targets, target_weights=[0.9, 0.1]
        )
        self.assertEqual(selected, ["return_factor"])

    def test_coverage_is_measured_before_neutral_imputation(self):
        rng = np.random.default_rng(19)
        y = rng.normal(size=300)
        X = np.column_stack([
            y + rng.normal(scale=0.1, size=300),
            y + rng.normal(scale=0.01, size=300),
        ])
        selector = CrossSectionalFeatureSelector(
            max_features=2, min_coverage=0.20, corr_threshold=0.99,
        )
        _, selected = selector.fit_transform(
            X, ["covered", "sparse_but_imputed"], y,
            feature_coverage=[1.0, 0.05],
        )

        self.assertEqual(selected, ["covered"])
        self.assertEqual(
            selector.report_.dropped_low_coverage, ["sparse_but_imputed"]
        )

    def test_missing_values_do_not_participate_in_cross_sectional_rank(self):
        ranked = _rank_finite_to_unit_interval(
            np.array([1.0, np.nan, 3.0], dtype=np.float32)
        )
        np.testing.assert_allclose(ranked, [1 / 3, 0.5, 2 / 3], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
