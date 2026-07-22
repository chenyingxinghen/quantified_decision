import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from config.automation_config import AUTO_MODEL_PATH, AUTO_NORM_STATS_PATH
from core.backtest.data_handler import _prepare_adjusted_stock_data
from core.exit_rules import evaluate_exit
from core.factors.train_ml_model import MLModelTrainer
from scripts.select_stocks import _update_factor_cache_incremental


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingLabelTests(unittest.TestCase):
    def test_unbuyable_samples_are_forced_to_daily_bottom(self):
        scores = np.array([0.2, 9.0, 0.5, 8.0], dtype=np.float64)
        returns = np.array([0.01, 0.20, 0.03, 0.18], dtype=np.float32)
        dates = np.array(['2026-01-05'] * 4)
        unbuyable = np.array([False, True, False, True])

        ranked, penalized_scores, penalized_returns = (
            MLModelTrainer._apply_unbuyable_penalty(
                scores, returns, dates, unbuyable
            )
        )

        self.assertTrue(np.all(penalized_scores[unbuyable] < penalized_scores[~unbuyable]))
        self.assertTrue(np.all(penalized_returns[unbuyable] < penalized_returns[~unbuyable]))
        self.assertTrue(np.all(ranked[unbuyable] < ranked[~unbuyable].min()))


class ExitRuleTests(unittest.TestCase):
    def test_intraday_touch_does_not_trigger_tail_exit(self):
        decision = evaluate_exit(
            current_price=10.2,
            entry_price=10.0,
            holding_days=3,
            stop_loss=9.0,
            take_profit=12.0,
            enable_stop_loss=True,
            enable_take_profit=True,
            enable_time_stop=False,
            time_stop_days=7,
            time_stop_max_return_pct=0.15,
        )
        self.assertFalse(decision.should_exit)

    def test_tail_price_stop_loss_triggers(self):
        decision = evaluate_exit(
            current_price=8.9,
            entry_price=10.0,
            holding_days=3,
            stop_loss=9.0,
            take_profit=12.0,
            enable_stop_loss=True,
            enable_take_profit=True,
            enable_time_stop=False,
            time_stop_days=7,
            time_stop_max_return_pct=0.15,
        )
        self.assertEqual(decision.reason, 'stop_loss')


class AdjustmentTests(unittest.TestCase):
    def test_adjusted_prices_capture_corporate_action_return(self):
        raw = pd.DataFrame({
            'date': ['2024-08-05', '2024-08-06'],
            'open': [6.82, 6.79],
            'high': [6.95, 6.84],
            'low': [6.80, 6.67],
            'close': [6.81, 6.73],
            'preclose': [6.84, 6.70],
            'fore_adjust_factor': [np.nan, 0.977396],
            'back_adjust_factor': [np.nan, 1.081856],
        })
        adjusted = _prepare_adjusted_stock_data(
            raw,
            prior_fore_factor=0.961608,
            prior_back_factor=1.064381,
        )

        adjusted_return = adjusted.loc[1, 'close'] / adjusted.loc[0, 'close'] - 1
        self.assertAlmostEqual(adjusted_return, 0.004478, places=5)
        self.assertAlmostEqual(adjusted.loc[0, 'raw_close'], 6.81, places=6)


class ArtifactAndCacheTests(unittest.TestCase):
    def test_automation_model_has_bound_normalization_artifact(self):
        model_path = PROJECT_ROOT / AUTO_MODEL_PATH
        norm_path = PROJECT_ROOT / AUTO_NORM_STATS_PATH
        archived_model = norm_path.parent / 'lightgbm_factor_model.pkl'
        self.assertTrue(model_path.exists())
        self.assertTrue(norm_path.exists())
        self.assertTrue(archived_model.exists())

        def sha256(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        self.assertEqual(sha256(model_path), sha256(archived_model))

    @patch('core.data.market_sentiment_calculator.MarketSentimentCalculator')
    @patch('scripts.select_stocks.MLModelTrainer')
    def test_incremental_cache_honors_requested_directory(self, trainer_cls, sentiment_cls):
        trainer = MagicMock()
        trainer.load_training_data.return_value = {}
        trainer_cls.return_value = trainer
        sentiment_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            requested = os.path.join(temp_dir, 'custom-cache')
            _update_factor_cache_incremental(
                db_path='unused.db',
                codes=['000001'],
                cache_dir=requested,
                workers=1,
            )
            self.assertEqual(trainer.factors_cache_dir, os.path.abspath(requested))


if __name__ == '__main__':
    unittest.main()
