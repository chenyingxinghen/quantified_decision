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
from core.factors.advanced_factors import RiskFactors
from core.factors.cache_manifest import (
    bind_model_to_cache, resolve_ensemble_cache, resolve_model_cache,
    write_cache_manifest,
)
from core.factors.train_ml_model import MLModelTrainer, _fast_rankdata_1d, _scan_cache_file
from scripts.migrate_downside_risk_cache import _migrate_one
from scripts.diag_random_null import simulate
from scripts.exp_nam_gate import _topk_excess
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







class NamSelectionMetricTests(unittest.TestCase):
    def test_topk_excess_remains_a_diagnostic_metric(self):
        pred = np.array([4.0, 3.0, 2.0, 1.0])
        returns = np.array([0.04, 0.03, -0.02, -0.01])
        value = _topk_excess(pred, returns, [(0, 4)], k=2)
        # 少于10只的截面按训练实现跳过，防止小样本头部统计进入评估。
        self.assertEqual(value, float('-inf'))

    def test_best_epoch_index_contract_is_zero_based(self):
        source = (PROJECT_ROOT / 'scripts' / 'exp_nam_gate.py').read_text(encoding='utf-8')
        self.assertIn('best_epoch_idx = len(history) - 1', source)
        self.assertNotIn('best_epoch_idx = len(history)\n', source)

    def test_rank_ic_remains_default_selection_metric(self):
        source = (PROJECT_ROOT / 'scripts' / 'exp_nam_gate.py').read_text(encoding='utf-8')
        self.assertIn("select_metric='rank_ic'", source)
        self.assertIn("default='rank_ic'", source)


class RandomNullTests(unittest.TestCase):
    def test_random_null_uses_distinct_concurrent_holdings(self):
        dates = ['2026-01-02', '2026-01-05', '2026-01-06']
        rets = pd.DataFrame({'a': [0.0, 0.1, 0.0], 'b': [0.0, 0.0, 0.2]}, index=dates)
        eligible = pd.DataFrame(True, index=dates, columns=['a', 'b'])
        trades = pd.DataFrame([
            {'stock_code': '000001', 'buy_date': dates[0], 'sell_date': dates[2], 'capital_weight': 0.5},
            {'stock_code': '000002', 'buy_date': dates[0], 'sell_date': dates[2], 'capital_weight': 0.5},
        ])
        result = simulate(trades, rets, eligible, 0.0, 0.0, n_sims=1, seed=1)
        # 两条并发腿必须分别抽中 a/b，因此组合收益恒为 0.5*10% + 0.5*20%。
        self.assertAlmostEqual(float(result['sims'][0]), 0.15, places=6)

class BacktestNamingTests(unittest.TestCase):
    def test_ensemble_member_order_is_canonical(self):
        import hashlib
        members_a = sorted(os.path.normcase(os.path.abspath(p)) for p in ['models/b', 'models/a'])
        members_b = sorted(os.path.normcase(os.path.abspath(p)) for p in ['models/a', 'models/b'])
        self.assertEqual(members_a, members_b)
        self.assertEqual(
            hashlib.sha256('\0'.join(members_a).encode('utf-8')).hexdigest()[:8],
            hashlib.sha256('\0'.join(members_b).encode('utf-8')).hexdigest()[:8],
        )

class PortfolioCostTests(unittest.TestCase):
    def test_buy_and_sell_costs_are_not_double_counted(self):
        from core.backtest.portfolio import Portfolio
        p = Portfolio(1.0, max_positions=1, buy_cost_rate=0.0008, sell_cost_rate=0.0013)
        pos = p.open_position("000001.SZ", "2026-01-02", 10.0, capital_allocation=1.0)
        self.assertIsNotNone(pos)
        self.assertLessEqual(p.cash, 1e-8)
        trade = p.close_position("000001.SZ", "2026-01-05", 10.0, "test")
        self.assertIsNotNone(trade)
        self.assertAlmostEqual(trade.commission, 0.0008 + 0.0013 / 1.0008, places=4)
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
        strategy.regime_filter = 'off'
        strategy.max_rel_atr = None
        strategy.norm_stats = None
        strategy.model = RecordingModel()
        strategy.ensemble_models = [strategy.model]
        strategy.risk_penalty_lambda = 0.0
        strategy.risk_penalty_feature = 'max_drawdown_20'
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


class RiskPenaltyTests(unittest.TestCase):
    def test_fast_rank_treats_nan_as_neutral_percentile(self):
        values = np.array([100.0, np.nan, 900.0], dtype=np.float32)
        ranks = _fast_rankdata_1d(values) / (len(values) + 1)
        np.testing.assert_allclose(ranks, np.array([1.0 / 3.0, 0.50, 2.0 / 3.0]))

    def test_downside_risk_is_defined_with_mixed_return_signs(self):
        # 交替涨跌的常见窗口也必须有下行风险；旧实现会因不足 20 个负收益而输出 0。
        returns = np.array(([0.01, -0.02] * 15), dtype=float)
        close = 10.0 * np.cumprod(1.0 + returns)
        factors = RiskFactors.calculate_risk_features(pd.DataFrame({'close': close}))
        self.assertGreater(float(factors['downside_risk'].iloc[-1]), 0.0)

    def test_risk_direction_low_penalizes_deeper_drawdown(self):
        adjusted = MLFactorBacktestStrategy._risk_adjusted_scores(
            probs=np.array([0.5, 0.5]),
            risk_values=np.array([-0.5, -0.1]),
            penalty_lambda=0.2,
            direction='low',
        )
        self.assertLess(adjusted[0], adjusted[1])

    def test_constant_risk_cross_section_fails_loudly(self):
        with self.assertRaisesRegex(RuntimeError, '横截面已退化'):
            MLFactorBacktestStrategy._risk_adjusted_scores(
                probs=np.array([0.1, 0.9]),
                risk_values=np.array([0.5, 0.5]),
                penalty_lambda=0.15,
                direction='high',
            )


class FactorCacheManifestTests(unittest.TestCase):
    def test_new_cache_can_be_bound_and_resolved(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir = os.path.join(root, 'cache-v2')
            model_dir = os.path.join(root, 'model')
            write_cache_manifest(cache_dir)
            bind_model_to_cache(model_dir, cache_dir)
            self.assertEqual(resolve_model_cache(model_dir, 'legacy'), os.path.abspath(cache_dir))

    def test_legacy_model_without_manifest_uses_legacy_cache(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = os.path.join(root, 'legacy-model')
            os.makedirs(model_dir)
            legacy = os.path.join(root, 'legacy-cache')
            self.assertEqual(resolve_model_cache(model_dir, legacy), os.path.abspath(legacy))

    def test_explicit_unversioned_nonempty_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir = os.path.join(root, 'bad-cache')
            os.makedirs(cache_dir)
            Path(cache_dir, 'dummy.parquet').write_bytes(b'not-a-real-parquet')
            with self.assertRaisesRegex(RuntimeError, '缺少版本清单'):
                MLModelTrainer(db_path='unused.db', cache_dir=cache_dir)

    @patch('scripts.migrate_downside_risk_cache._calculate_downside_risk')
    def test_strict_migration_changes_only_downside_risk(self, calculate):
        calculate.return_value = pd.DataFrame({
            'date': ['2020-01-02', '2020-01-03'],
            'downside_risk': np.array([0.1, 0.2], dtype=np.float32),
        })
        with tempfile.TemporaryDirectory() as root:
            source_dir = os.path.join(root, 'source')
            target_dir = os.path.join(root, 'target')
            os.makedirs(source_dir)
            source_path = os.path.join(source_dir, '000001_factors.parquet')
            original = pd.DataFrame({
                'date': ['2020-01-02', '2020-01-03'],
                'factor': np.array([1.0, 2.0], dtype=np.float32),
                'downside_risk': np.array([0.0, 0.0], dtype=np.float32),
            })
            original.to_parquet(source_path, index=False)
            name, ok, status = _migrate_one(source_path, target_dir)
            migrated = pd.read_parquet(os.path.join(target_dir, name))
            self.assertTrue(ok)
            self.assertEqual(status, 'written')
            self.assertTrue(original[['date', 'factor']].equals(migrated[['date', 'factor']]))
            np.testing.assert_allclose(migrated['downside_risk'], [0.1, 0.2])

    def test_ensemble_rejects_different_bound_caches(self):
        with tempfile.TemporaryDirectory() as root:
            cache_a = os.path.join(root, 'cache-a')
            cache_b = os.path.join(root, 'cache-b')
            model_a = os.path.join(root, 'model-a')
            model_b = os.path.join(root, 'model-b')
            write_cache_manifest(cache_a)
            write_cache_manifest(cache_b)
            bind_model_to_cache(model_a, cache_a)
            bind_model_to_cache(model_b, cache_b)
            with self.assertRaisesRegex(RuntimeError, '不同因子缓存'):
                resolve_ensemble_cache([model_a, model_b], os.path.join(root, 'legacy'))

    def test_cache_scan_requires_full_date_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, '000001_factors.parquet')
            pd.DataFrame({
                'date': ['2020-01-02', '2020-01-03'],
                'factor': [1.0, 2.0],
            }).to_parquet(path, index=False)
            _, covered = _scan_cache_file(
                ('000001', '2020-01-02', '2020-01-03', path, ['factor'])
            )
            _, missing_history = _scan_cache_file(
                ('000001', '2019-12-31', '2020-01-03', path, ['factor'])
            )
            _, missing_future = _scan_cache_file(
                ('000001', '2020-01-02', '2020-01-06', path, ['factor'])
            )
            self.assertFalse(covered)
            self.assertTrue(missing_history)
            self.assertTrue(missing_future)

    def test_generic_model_archive_binds_versioned_cache(self):
        class RecordingModel:
            feature_names = []
            model = None

            def save_model(self, path):
                Path(path).write_bytes(b'model')

        with tempfile.TemporaryDirectory() as root:
            cache_dir = os.path.join(root, 'cache-v2')
            save_dir = os.path.join(root, 'models')
            trainer = MLModelTrainer(db_path='unused.db', cache_dir=cache_dir)
            trainer.models = {'xgboost': RecordingModel()}
            trainer.task = 'ranking'
            trainer._save_training_config = MagicMock()
            archive_dir = trainer.save_models(
                save_dir=save_dir,
                years=1,
                stocks=1,
                update_latest=False,
                training_results={},
            )
            self.assertEqual(
                resolve_model_cache(archive_dir, os.path.join(root, 'legacy')),
                os.path.abspath(cache_dir),
            )


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


class PerformanceForecastSchemaTests(unittest.TestCase):
    """业绩预告接口的字段名和其它财务接口不一样，改名逻辑错了会静默落 0 行。"""

    def _fake_rs(self, fields, rows):
        rs = MagicMock()
        rs.fields = fields
        it = iter(rows)
        holder = {}

        def _next():
            try:
                holder['row'] = next(it)
                return True
            except StopIteration:
                return False

        rs.next.side_effect = _next
        rs.get_row_data.side_effect = lambda: holder['row']
        return rs

    def test_forecast_pub_date_is_mapped_from_exp_pub_date(self):
        # baostock query_forecast_report 真实字段名，没有 pubDate / statDate
        fields = ['code', 'profitForcastExpPubDate', 'profitForcastExpStatDate',
                  'profitForcastType', 'profitForcastAbstract',
                  'profitForcastChgPctUp', 'profitForcastChgPctDwn']
        rows = [['sz.000001', '2010-01-27', '2009-12-31', '预增', '摘要', '0', '700']]
        with patch('core.data.baostock_fetcher_methods._bs_query',
                   return_value=self._fake_rs(fields, rows)):
            from core.data.baostock_fetcher_methods import fetch_performance_forecast
            df = fetch_performance_forecast('000001', '2010-01-01', '2026-08-14')
        self.assertIn('pub_date', df.columns)
        self.assertIn('stat_date', df.columns)
        # PIT 纪律：pub_date 必须是预告披露日，不是报告期
        self.assertEqual(df['pub_date'].iloc[0], '2010-01-27')
        self.assertEqual(df['stat_date'].iloc[0], '2009-12-31')
        # 落库前的 dropna(subset=['pub_date']) 不能把行清空
        self.assertEqual(len(df.dropna(subset=['pub_date'])), 1)


if __name__ == '__main__':
    unittest.main()

