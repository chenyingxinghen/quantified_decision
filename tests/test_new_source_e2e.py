import os
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from core.backtest.strategies.ml_factor_strategy import MLFactorBacktestStrategy
from core.data.jydb_feature_store import JYDBFeatureStore
from core.factors.external_source_factors import ExternalSourceFactors
from core.factors.comprehensive_factor_calculator import ComprehensiveFactorCalculator
from core.factors.factor_filler import FactorFiller
from core.factors.train_ml_model import (
    FACTOR_CACHE_PREPROCESSING_VERSION,
    _read_factor_cache_version,
    _write_factor_cache,
)
from core.factors.feature_selector import CrossSectionalFeatureSelector
from core.factors.multi_objective_labels import MultiObjectiveLabelBuilder


class _ModelContract:
    feature_names = ["jy_fin_ROE", "return_1d"]


class NewSourceEndToEndTests(unittest.TestCase):
    def test_factor_cache_records_preprocessing_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = os.path.join(temp_dir, "factors.parquet")
            _write_factor_cache(
                pd.DataFrame({"date": ["2024-01-01"], "jy_fin_ROE": [np.nan]}),
                cache_file,
            )
            self.assertEqual(
                _read_factor_cache_version(cache_file),
                FACTOR_CACHE_PREPROCESSING_VERSION,
            )

    def test_external_missing_values_survive_factor_preprocessing(self):
        calculator = ComprehensiveFactorCalculator.__new__(ComprehensiveFactorCalculator)
        calculator.filler = FactorFiller()
        base = pd.DataFrame({
            "jy_fin_ROE": [np.nan, 12.0],
            "rsi_21": [np.nan, 55.0],
        })
        calculator._calculate_base_factors = lambda *args, **kwargs: base.copy()
        calculator._apply_feature_engineering = lambda frame, verbose=False: frame

        result = calculator.calculate_all_factors(
            "000001", pd.DataFrame(index=base.index),
            apply_feature_engineering=False,
        )

        self.assertTrue(np.isnan(result.loc[0, "jy_fin_ROE"]))
        self.assertEqual(float(result.loc[0, "rsi_21"]), 0.0)

    def test_backtest_strategy_allows_standalone_market_db_without_meta_tables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            market_db = os.path.join(temp_dir, "daily.db")
            sqlite3.connect(market_db).close()
            strategy = MLFactorBacktestStrategy(
                model_path=os.path.join(temp_dir, "unused.pkl"),
                use_cache=False,
                db_path=market_db,
            )
            strategy._precompute_pit_data()
            self.assertEqual(strategy._meta_map, {})
            self.assertTrue(strategy._all_finance_df.empty)

    def test_etl_to_factor_selection_and_backtest_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "jydb_features.db")
            cache_dir = os.path.join(temp_dir, "factors")
            os.makedirs(cache_dir)
            store = JYDBFeatureStore(db_path)
            store.upsert_wide_frame(
                pd.DataFrame({
                    "code": ["000001", "000001"],
                    "available_date": ["2024-01-05", "2024-01-15"],
                    "end_date": ["2023-12-31", "2024-03-31"],
                    "ROE": [8.0, 12.0],
                }),
                source_table="LC_MainIndexNew",
                feature_cols=["ROE"],
                prefix="jy_fin_",
            )
            dates = pd.date_range("2024-01-01", periods=30, freq="D")
            close = np.linspace(10.0, 13.0, len(dates))
            daily = pd.DataFrame({
                "date": dates.strftime("%Y-%m-%d"),
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(len(dates), 1000.0),
                "amount": close * 1000.0,
            })

            external = ExternalSourceFactors(db_path).calculate_series("000001", daily)
            factors = pd.DataFrame({
                "date": daily["date"],
                "jy_fin_ROE": external["jy_fin_ROE"],
                "return_1d": daily["close"].pct_change(),
            })
            # 公告日前绝不能看到 ROE。
            self.assertTrue(factors.loc[3, "jy_fin_ROE"] != factors.loc[3, "jy_fin_ROE"])
            self.assertEqual(factors.loc[4, "jy_fin_ROE"], 8.0)
            self.assertEqual(factors.loc[14, "jy_fin_ROE"], 12.0)

            labels = MultiObjectiveLabelBuilder((5,), risk_horizon=5).build(daily)
            valid = factors[["jy_fin_ROE", "return_1d"]].notna().all(axis=1) & labels["y_ret_5d"].notna()
            selector = CrossSectionalFeatureSelector(
                max_features=2, min_coverage=0.1, corr_threshold=0.99
            )
            _, selected = selector.fit_transform(
                factors.loc[valid, ["jy_fin_ROE", "return_1d"]].to_numpy(),
                ["jy_fin_ROE", "return_1d"],
                labels.loc[valid, "y_ret_5d"].to_numpy(),
            )
            self.assertTrue(selected)

            factors.to_parquet(
                os.path.join(cache_dir, "000001_factors.parquet"), index=False
            )
            strategy = MLFactorBacktestStrategy.__new__(MLFactorBacktestStrategy)
            strategy.cache_dir = cache_dir
            strategy._factors_cache = {}
            strategy.model = _ModelContract()
            row = strategy._get_factors("000001", None, "2024-01-10")
            self.assertEqual(float(row.iloc[0]["jy_fin_ROE"]), 8.0)
            # 回测按 as-of 日期取缓存，不能读取 1 月 15 日才生效的 12.0。
            self.assertNotEqual(float(row.iloc[0]["jy_fin_ROE"]), 12.0)


if __name__ == "__main__":
    unittest.main()
