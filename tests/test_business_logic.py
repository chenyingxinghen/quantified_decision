import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from config.automation_config import (AUTO_MODEL_PATH, AUTO_NORM_STATS_PATH,
                                     AUTO_ENSEMBLE_MODEL_PATHS)
from config.factor_config import TrainingConfig
from core.backtest.baostock_data_handler import _prepare_adjusted_stock_data
from core.backtest.strategies.ml_factor_strategy import MLFactorBacktestStrategy
from core.exit_rules import evaluate_exit
from core.factors.advanced_factors import RiskFactors
from core.factors.cache_manifest import (
    bind_model_to_cache, resolve_ensemble_cache, resolve_model_cache,
    write_cache_manifest,
)
from core.factors.train_ml_model import MLModelTrainer, _fast_rankdata_1d, _scan_cache_file
# 2026-08-14：实验脚本集中到 scripts/exp/（见 scripts/exp/README.md）
from scripts.exp.diag_random_null import simulate
# train_nam_model 顶层 import torch，而 torch 只装在 workbuddy 的 3.13.12 解释器里、
# pytest 只装在 .venv 里 —— 顶层导入会让**整个文件收集失败**。改成惰性导入 + skip。
try:
    from scripts.train_nam_model import _topk_excess
except ImportError:  # torch 缺失
    _topk_excess = None
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
    @unittest.skipIf(_topk_excess is None, 'torch 不可用（只装在 workbuddy 解释器）')
    def test_topk_excess_remains_a_diagnostic_metric(self):
        pred = np.array([4.0, 3.0, 2.0, 1.0])
        returns = np.array([0.04, 0.03, -0.02, -0.01])
        value = _topk_excess(pred, returns, [(0, 4)], k=2)
        # 少于10只的截面按训练实现跳过，防止小样本头部统计进入评估。
        self.assertEqual(value, float('-inf'))

    def test_best_epoch_index_contract_is_zero_based(self):
        source = (PROJECT_ROOT / 'scripts' / 'train_nam_model.py').read_text(encoding='utf-8')
        self.assertIn('best_epoch_idx = len(history) - 1', source)
        self.assertNotIn('best_epoch_idx = len(history)\n', source)

    def test_rank_ic_remains_default_selection_metric(self):
        source = (PROJECT_ROOT / 'scripts' / 'train_nam_model.py').read_text(encoding='utf-8')
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
        # 2026-08-15：回测处理器改用 preclose/close 反推的完整前复权
        # （见 core/backtest/baostock_data_handler._prepare_adjusted_stock_data）。
        # 旧口径依赖 adjust_factor 表 + prior_* 接续参数，该表只覆盖 42.5% 的
        # 除权事件、缺失时 fillna(1.0) 退化为不复权，已弃用。
        # 这里的复权比 r = preclose_t / close_{t-1} = 6.70 / 6.81，与旧用例里
        # fore_adjust_factor 给出的比值一致，所以复权后收益率仍是 0.004478。
        raw = pd.DataFrame({
            'date': ['2024-08-05', '2024-08-06'],
            'open': [6.82, 6.79],
            'high': [6.95, 6.84],
            'low': [6.80, 6.67],
            'close': [6.81, 6.73],
            'preclose': [6.84, 6.70],
        })
        adjusted = _prepare_adjusted_stock_data(raw)

        adjusted_return = adjusted.loc[1, 'close'] / adjusted.loc[0, 'close'] - 1
        self.assertAlmostEqual(adjusted_return, 0.004478, places=5)
        self.assertAlmostEqual(adjusted.loc[0, 'raw_close'], 6.81, places=6)
        # 最后一行的复权因子恒为 1：前复权以最新价为基准
        self.assertAlmostEqual(adjusted.loc[1, 'close'], 6.73, places=5)
        # adj_* 与复权后的价格列同值，供策略侧显式引用
        self.assertAlmostEqual(adjusted.loc[0, 'adj_close'],
                               adjusted.loc[0, 'close'], places=6)


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


class FactorCacheResolutionTests(unittest.TestCase):
    """2026-08-22 manifest 版本契约机制已移除：所有模型（NAM/树/mark/latest）共用
    单一通用缓存 factors_cache，版本演进靠重命名/删除缓存文件夹完成，不再逐模型写
    绑定清单、不再做版本校验。resolve_* 始终返回 config 里的 factors_cache。"""

    def test_resolve_model_cache_returns_universal_cache(self):
        with tempfile.TemporaryDirectory() as root:
            model_dir = os.path.join(root, 'some-model')
            legacy = os.path.join(root, 'legacy-cache')
            self.assertEqual(
                resolve_model_cache(model_dir, legacy),
                os.path.abspath(TrainingConfig.CACHE_DIR),
            )

    def test_resolve_model_cache_ignores_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            a = os.path.join(root, 'a')
            b = os.path.join(root, 'b')
            self.assertEqual(resolve_model_cache(a, b), resolve_model_cache(b, a))
            self.assertEqual(
                resolve_model_cache(a, b), os.path.abspath(TrainingConfig.CACHE_DIR))

    def test_resolve_ensemble_cache_returns_universal_cache(self):
        with tempfile.TemporaryDirectory() as root:
            model_dirs = [os.path.join(root, m) for m in ('m1', 'm2', 'm3')]
            legacy = os.path.join(root, 'legacy')
            self.assertEqual(
                resolve_ensemble_cache(model_dirs, legacy),
                os.path.abspath(TrainingConfig.CACHE_DIR),
            )

    def test_manifest_writers_are_noops(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir = os.path.join(root, 'cache')
            model_dir = os.path.join(root, 'model')
            self.assertEqual(write_cache_manifest(cache_dir), {})
            self.assertEqual(bind_model_to_cache(model_dir, cache_dir), {})

    @unittest.skip('migrate_downside_risk_cache 模块已移除（2026-08-14 脚本重组），无法恢复')
    def test_strict_migration_changes_only_downside_risk(self):
        # 原测试覆盖「严格迁移只改 downside_risk、不动其它列」。源模块被删除，
        # 且全量缓存已重建，无历史缓存可迁移，故跳过。
        pass

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
            # 2026-08-22 manifest 机制移除：resolve 不再绑定归档时的 cache_dir，
            # 始终返回单一通用缓存 factors_cache。
            self.assertEqual(
                resolve_model_cache(archive_dir, os.path.join(root, 'legacy')),
                os.path.abspath(TrainingConfig.CACHE_DIR),
            )


class ArtifactAndCacheTests(unittest.TestCase):
    """自动化绑定的模型 / 归一化统计量 / 因子缓存三者必须自洽。

    历史：这条测试原来断言 automation/ 目录下有一份与归档 SHA256 相同的
    lightgbm pkl。那个目录后来空了，测试就一直红着——它报告的是真问题
    （自动化没有可加载的模型），不是测试写错。2026-08-18 接入 T115 时改写成
    「载体无关」的契约：只管**同批产出**和**面板对得上缓存**，不再写死模型族。
    """

    def _automation_paths(self):
        return PROJECT_ROOT / AUTO_MODEL_PATH, PROJECT_ROOT / AUTO_NORM_STATS_PATH

    def test_automation_model_has_bound_normalization_artifact(self):
        model_path, norm_path = self._automation_paths()
        self.assertTrue(model_path.exists(), f'AUTO_MODEL_PATH 不存在: {model_path}')
        self.assertTrue(norm_path.exists(), f'AUTO_NORM_STATS_PATH 不存在: {norm_path}')
        # 同批产出：权重与归一化统计量必须来自同一次训练的同一个存档目录，
        # 否则连续列会以错误量纲进模型，而且不报错。
        self.assertEqual(
            model_path.parent.resolve(), norm_path.parent.resolve(),
            '模型与 norm_stats 不在同一存档目录，无法保证同批产出',
        )

    def test_automation_ensemble_members_are_same_batch(self):
        """多种子集成成员必须与主模型**同批产出**：同面板、同缓存版本、同族划分。

        留空（当前默认）时直接通过 —— 集成是可选项，不是契约。
        非空时校验三件会静默出错的事：
          1. feature_names 顺序完全一致。策略层也校验，但那是运行时；
             这里让它在上线前红掉。
          2. 绑定的因子缓存目录一致。成员各自带 factor_cache_manifest.json，
             指向不同缓存版本时，同一天的同一只票会拿到两套不同口径的面板，
             平均出来的排名没有意义，且**不会报错**。
          3. 成员之间不重复、且不等于主模型 —— 重复成员等于给某个种子加权，
             而 [[ensemble-beats-single]] 已证伪按权重加权（w=0.7 回测证伪），
             口径必须是严格等权。
        """
        from scripts.select_stocks import (
            load_model_feature_names, resolve_production_cache_dir,
        )

        if not AUTO_ENSEMBLE_MODEL_PATHS:
            self.skipTest('AUTO_ENSEMBLE_MODEL_PATHS 为空 —— 未启用等权集成')

        main_path, _ = self._automation_paths()
        self.assertTrue(main_path.exists(), f'AUTO_MODEL_PATH 不存在: {main_path}')
        main_feats = load_model_feature_names(str(main_path))
        self.assertTrue(main_feats, '主模型缺少 feature_names.json')
        main_cache = Path(resolve_production_cache_dir(str(main_path))).resolve()

        seen = {main_path.resolve()}
        for rel in AUTO_ENSEMBLE_MODEL_PATHS:
            p = (PROJECT_ROOT / rel).resolve()
            self.assertTrue(p.exists(), f'集成成员不存在: {p}')
            self.assertNotIn(p, seen, f'集成成员重复（等权口径被破坏）: {p}')
            seen.add(p)

            feats = load_model_feature_names(str(p))
            self.assertTrue(feats, f'集成成员缺少 feature_names.json: {p}')
            self.assertEqual(list(feats), list(main_feats),
                             f'集成成员面板与主模型不一致: {p}')

            self.assertEqual(Path(resolve_production_cache_dir(str(p))).resolve(), main_cache,
                             f'集成成员绑定的因子缓存版本与主模型不同: {p}')

            self.assertTrue((p.parent / 'norm_stats.pkl').exists(),
                            f'集成成员缺少同批 norm_stats.pkl: {p}')

    def test_automation_model_panel_is_covered_by_its_bound_cache(self):
        """模型要的每一列都必须在它绑定的缓存里真实存在。

        这是接入生产的核心契约。缺列不会报错——ml_factor_strategy 会把整列
        填成 0.5 然后照常出票。T115（224 列，含 5 个 idx_*）配旧的 219/242 列
        缓存就是这个情形，本测试就是为了让那种配置在上线前红掉。
        """
        import pyarrow.parquet as pq
        from scripts.select_stocks import (
            load_model_feature_names, resolve_production_cache_dir,
        )

        model_path, _ = self._automation_paths()
        if not model_path.exists():
            self.fail(f'AUTO_MODEL_PATH 不存在: {model_path}')

        features = load_model_feature_names(str(model_path))
        self.assertTrue(features, '模型存档缺少 feature_names.json，无法校验面板')

        cache_dir = resolve_production_cache_dir(str(model_path))
        parquets = sorted(Path(cache_dir).glob('*_factors.parquet'))
        if not parquets:
            self.skipTest(f'绑定缓存为空（新机器尚未建缓存）: {cache_dir}')

        # 列是否存在是 schema 级事实，读若干个文件的 schema 即可，不必扫全量。
        covered = set()
        for path in parquets[:32]:
            covered.update(pq.read_schema(path).names)
        missing = [c for c in features if c not in covered]
        self.assertEqual(
            missing, [],
            f'绑定缓存 {cache_dir} 缺少模型所需的列 {missing[:12]}'
            f'（这些列会被静默填成常数 0.5）',
        )

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


class IndexRelativeFactorTests(unittest.TestCase):
    """idx_* 生产实现（core/factors/index_relative_factors.py）的**自洽性**锁。

    背景：这 5 列最早由 scripts/archive/oneoff/build_idxrel_cache.py 离线注入 T115
    训练缓存，晋身后搬进生产路径。原测试对拍「离线脚本 vs 生产实现」，但离线脚本
    已在 2026-08-14 脚本重组中删除（全仓已无 build_idxrel_cache 定义），对照本身
    无法执行。改为只锁生产实现自身的契约，不再依赖已删的离线脚本：

      · 确定性：同输入恒等输出（捕捉任何静默公式漂移）；
      · 列集与中性填充值：固定 5 列，热身期 β→1.0、其余→0.0（与 INDEX_REL_FILLS 一致）；
      · 停牌不挪窗：个股日历缺日时，窗口只在缺日处停顿，剩余交易日的值不可被平移。
    """

    def _series(self, n=400, seed=7):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range('2020-01-01', periods=n).strftime('%Y-%m-%d')
        rm = pd.Series(rng.normal(0, 0.012, n), index=dates)
        rb = pd.Series(rng.normal(0, 0.014, n), index=dates)
        rs = 1.15 * rm + pd.Series(rng.normal(0, 0.018, n), index=dates)
        return rs, rm, rb

    def test_production_formula_is_deterministic_and_well_formed(self):
        from core.factors.index_relative_factors import (
            compute_features, INDEX_REL_COLUMNS, INDEX_REL_FILLS)

        rs, rm, rb = self._series()
        out = compute_features(rs, rm, rb)
        # 列集与顺序固定，且正是合约里的 5 列
        self.assertListEqual(list(out.columns), INDEX_REL_COLUMNS)
        # 确定性：同输入必须逐位一致（锁住公式，防静默漂移）
        pd.testing.assert_frame_equal(out, compute_features(rs, rm, rb))
        # 热身期：β 的前 min_periods-1=39 行无足够样本 → NaN（下游按 1.0 补齐）；
        # 第 40 行起应已算出有限值。
        self.assertTrue(out['idx_beta_60'].iloc[:39].isna().all())
        self.assertFalse(np.isnan(out['idx_beta_60'].iloc[39]))
        # 中性填充契约：β 中性 1.0，其余 0.0（与 INDEX_REL_FILLS 完全一致）
        self.assertEqual(INDEX_REL_FILLS['idx_beta_60'], 1.0)
        self.assertTrue(all(INDEX_REL_FILLS[c] == 0.0
                             for c in INDEX_REL_COLUMNS if c != 'idx_beta_60'))

    def test_compute_is_label_invariant(self):
        """compute_features 只依赖 (rs, rm, rb) 的**取值顺序**，与日期标签无关。

        原测试对拍「离线脚本 vs 生产实现」的停牌窗口行为；离线脚本已删。这里改为
        锁住生产函数自身的真契约：滚动窗口按**输入序列的位置**计算，而非日历。
        因此把同一段取值换个日期标签（或截掉前缀）后重算，输出值（忽略索引）必须
        逐位一致——这保证停牌/补录等日历变动不会悄悄改变公式语义。
        """
        from core.factors.index_relative_factors import compute_features

        rs, rm, rb = self._series()
        out = compute_features(rs, rm, rb)

        # 换一套毫无关系的日期标签，取值顺序不变 → 输出值必须不变
        alt_dates = pd.bdate_range('2022-06-01', periods=len(rs)).strftime('%Y-%m-%d')
        rs2 = pd.Series(rs.values, index=alt_dates)
        rm2 = pd.Series(rm.values, index=alt_dates)
        rb2 = pd.Series(rb.values, index=alt_dates)
        out2 = compute_features(rs2, rm2, rb2)
        np.testing.assert_allclose(
            out.values, out2.values, equal_nan=True,
            err_msg='换日期标签后输出值变化 —— 公式依赖了日历而非取值顺序')

    def test_board_mapping_covers_every_prefix(self):
        from core.factors.index_relative_factors import board_series

        idx = {
            'sh.000001': pd.Series([0.01], index=['2020-01-02']),
            'sz.399001': pd.Series([0.02], index=['2020-01-02']),
            'sz.399006_padded': pd.Series([0.03], index=['2020-01-02']),
        }
        # 沪主板/科创 -> 上证综指；深主板 -> 深成指；创业板 -> 创业板指(回退后)
        self.assertEqual(board_series('600000', idx).iloc[0], 0.01)
        self.assertEqual(board_series('688001', idx).iloc[0], 0.01)
        self.assertEqual(board_series('000001', idx).iloc[0], 0.02)
        self.assertEqual(board_series('300750', idx).iloc[0], 0.03)
        self.assertEqual(board_series('830799', idx).iloc[0], 0.01)   # 北交所兜底

    def test_production_output_matches_training_cache(self):
        """端到端对拍：真库 + T115 训练缓存。缓存不在（新机器）时跳过。"""
        import pyarrow.parquet as pq
        from config.baostock_config import DATABASE_PATH
        from core.factors.index_relative_factors import (
            INDEX_REL_COLUMNS, IndexRelativeFactors,
        )

        cache_dir = PROJECT_ROOT / 'database' / 'system_data' / 'factors_cache'
        if not cache_dir.is_dir() or not os.path.exists(DATABASE_PATH):
            self.skipTest('缺少 T115 训练缓存或行情库，跳过端到端对拍')

        # 每个板块前缀各取一只：β/相对强弱的板块基准映射按前缀分支
        picked = {}
        for path in sorted(cache_dir.glob('*_factors.parquet')):
            picked.setdefault(path.name[:2], path)
        samples = [p for _, p in sorted(picked.items())][:4]
        if not samples:
            self.skipTest(f'训练缓存为空: {cache_dir}')

        for path in samples:
            code = path.name.split('_')[0]
            table = pq.read_table(path, columns=['date'] + INDEX_REL_COLUMNS)
            dates = [str(d)[:10] for d in table.column('date').to_pylist()]
            got = IndexRelativeFactors.calculate(code, dates, DATABASE_PATH)
            self.assertFalse(got.empty, f'{code}: 生产路径未产出 idx_*')
            for col in INDEX_REL_COLUMNS:
                cached = np.asarray(table.column(col).to_numpy(zero_copy_only=False))
                np.testing.assert_array_equal(
                    cached, got[col].to_numpy(),
                    err_msg=f'{code}.{col} 与训练缓存不一致',
                )


class FactorCacheColumnGuardTests(unittest.TestCase):
    """整列缺失必须硬失败，而不是静默填成常数 0.5。

    _preload_factor_cache 对缺失列填 0.5（rank 后的中性分位）。对**个别**股票
    缺数据这是对的；但若某列在整个缓存里都不存在，那是模型面板与缓存版本
    对不上，模型会拿一根常数列继续打分——正是接 T115 时最容易踩的那个坑。
    """

    def _strategy(self, cache_dir, feature_names):
        strategy = MLFactorBacktestStrategy.__new__(MLFactorBacktestStrategy)
        strategy.model = MagicMock(feature_names=list(feature_names))
        strategy.cache_dir = cache_dir
        strategy.use_cache = True
        strategy.preload_start = None
        strategy.preload_end = None
        strategy._active_stock_codes = set()
        strategy._factor_dates_cache = {}
        strategy._factor_matrix_cache = {}
        return strategy

    def _write_cache(self, cache_dir, columns):
        os.makedirs(cache_dir, exist_ok=True)
        for code in ('000001', '600000'):
            frame = pd.DataFrame({'date': ['2026-08-10', '2026-08-11']})
            for col in columns:
                frame[col] = np.float32(0.3)
            frame.to_parquet(os.path.join(cache_dir, f'{code}_factors.parquet'),
                             index=False)

    def test_column_missing_from_whole_cache_raises(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir = os.path.join(root, 'cache')
            self._write_cache(cache_dir, ['rsi_21', 'atr_14'])
            strategy = self._strategy(cache_dir, ['rsi_21', 'atr_14', 'idx_beta_60'])
            with self.assertRaisesRegex(RuntimeError, 'idx_beta_60'):
                strategy._preload_factor_cache()

    def test_complete_cache_loads_without_raising(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir = os.path.join(root, 'cache')
            self._write_cache(cache_dir, ['rsi_21', 'atr_14', 'idx_beta_60'])
            strategy = self._strategy(cache_dir, ['rsi_21', 'atr_14', 'idx_beta_60'])
            strategy._preload_factor_cache()
            self.assertEqual(len(strategy._factor_matrix_cache), 2)

    def test_escape_hatch_downgrades_to_warning(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir = os.path.join(root, 'cache')
            self._write_cache(cache_dir, ['rsi_21'])
            strategy = self._strategy(cache_dir, ['rsi_21', 'idx_beta_60'])
            with patch.dict(os.environ, {'ALLOW_MISSING_FACTOR_COLUMNS': '1'}):
                strategy._preload_factor_cache()
            self.assertEqual(len(strategy._factor_matrix_cache), 2)


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

