import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from config.automation_config import AUTO_MODEL_PATH, AUTO_NORM_STATS_PATH
from core.backtest.baostock_data_handler import _prepare_adjusted_stock_data
from core.backtest.strategies.ml_factor_strategy import MLFactorBacktestStrategy
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

    def test_recency_weights_follow_half_life_and_preserve_daily_groups(self):
        dates = np.array([
            '2020-01-01', '2020-01-01',
            '2021-01-01', '2021-01-01',
        ])
        weights = MLModelTrainer._calculate_recency_weights(
            dates, half_life_years=1.0, min_weight=0.01
        )

        self.assertAlmostEqual(weights[0], weights[1], places=6)
        self.assertAlmostEqual(weights[2], weights[3], places=6)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertAlmostEqual(weights[0] / weights[2], 0.5, delta=0.002)

    def test_recency_weights_validate_configuration(self):
        with self.assertRaises(ValueError):
            MLModelTrainer._calculate_recency_weights(
                np.array(['2020-01-01']), half_life_years=0
            )


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


class BasicRiskFilterTests(unittest.TestCase):
    def test_strategy_defaults_enable_independent_risk_controls(self):
        strategy = MLFactorBacktestStrategy(model_path='unused.pkl')
        self.assertEqual(strategy.risk_min_price, 1.0)
        self.assertTrue(strategy.risk_exclude_st)

    def test_rejects_low_price_stock(self):
        self.assertFalse(
            MLFactorBacktestStrategy._passes_basic_risk_filter(
                {'raw_close': 4.99, 'close': 10.0, 'is_st': 0},
                min_price=5.0,
                exclude_st=False,
            )
        )

    def test_rejects_st_stock(self):
        self.assertFalse(
            MLFactorBacktestStrategy._passes_basic_risk_filter(
                {'raw_close': 12.0, 'is_st': 1},
                min_price=5.0,
                exclude_st=True,
            )
        )

    def test_accepts_normal_stock(self):
        self.assertTrue(
            MLFactorBacktestStrategy._passes_basic_risk_filter(
                {'raw_close': 12.0, 'is_st': 0},
                min_price=5.0,
                exclude_st=True,
            )
        )

    def test_filter_and_existing_positions_do_not_change_prediction_cross_section(self):
        class RecordingModel:
            def __init__(self):
                self.last_input = None

            def predict(self, values):
                self.last_input = np.asarray(values).copy()
                return np.array([0.99, 0.90, 0.80], dtype=np.float32)

        class MarketData:
            def __init__(self):
                self._codes = ['cheap', 'eligible', 'held']
                self._bars = {
                    'cheap': {'raw_close': 0.5, 'close': 0.5, 'is_st': 0},
                    'eligible': {'raw_close': 10.0, 'close': 10.0, 'is_st': 0},
                    'held': {'raw_close': 20.0, 'close': 20.0, 'is_st': 0},
                }
                self._history = pd.DataFrame({
                    'high': np.full(15, 10.5),
                    'low': np.full(15, 9.5),
                    'close': np.full(15, 10.0),
                })

            def keys(self):
                return self._codes

            def get_bar(self, code):
                return self._bars.get(code)

            def __getitem__(self, code):
                return self._history

        strategy = object.__new__(MLFactorBacktestStrategy)
        strategy.min_confidence = 0.0
        strategy.risk_min_price = 1.0
        strategy.risk_exclude_st = False
        strategy.norm_stats = None
        strategy.model = RecordingModel()
        strategy._get_model_feature_names = lambda: ['factor']
        factor_values = {
            'cheap': np.array([0.0], dtype=np.float32),
            'eligible': np.array([10.0], dtype=np.float32),
            'held': np.array([20.0], dtype=np.float32),
        }
        strategy._get_factor_row_array = lambda code, _: factor_values[code]
        strategy._calculate_atr = lambda _: 0.0

        signals = strategy.generate_signals(
            current_date='2026-01-05',
            market_data=MarketData(),
            portfolio_state={
                'positions': {'held': object()},
                'available_slots': 3,
            },
        )

        np.testing.assert_allclose(
            strategy.model.last_input[:, 0],
            np.array([0.25, 0.50, 0.75], dtype=np.float32),
        )
        self.assertEqual([signal.stock_code for signal in signals], ['eligible'])


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
