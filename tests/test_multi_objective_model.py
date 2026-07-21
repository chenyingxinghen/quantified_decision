import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from core.factors.ml_factor_model import MultiObjectiveFactorModel


class DummyModel:
    def __init__(self, feature_names, values):
        self.feature_names = feature_names
        self.values = np.asarray(values, dtype=np.float32)
        self.is_trained = True

    def predict(self, factors):
        return self.values[:len(factors)]


class MultiObjectiveModelTests(unittest.TestCase):
    def test_weighted_prediction_and_feature_union(self):
        model = MultiObjectiveFactorModel(
            models={
                "return": DummyModel(["a", "b"], [0.8, 0.2]),
                "risk": DummyModel(["b", "c"], [0.4, 0.6]),
            },
            weights={"return": 0.75, "risk": 0.25},
        )
        factors = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        prediction = model.predict(factors)
        np.testing.assert_allclose(prediction, [0.7, 0.3], rtol=1e-6)
        self.assertEqual(model.feature_names, ["a", "b", "c"])

    def test_invalid_objective_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            MultiObjectiveFactorModel(
                {"return": DummyModel(["a"], [0.5])},
                {"risk": 1.0},
            )


if __name__ == "__main__":
    unittest.main()
