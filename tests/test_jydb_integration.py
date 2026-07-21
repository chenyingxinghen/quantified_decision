import os
import tempfile
import unittest
from contextlib import closing

import numpy as np
import pandas as pd

from core.data.jydb_feature_store import (
    DEFAULT_TABLE_SPECS, JYDBETL, JYDBFeatureStore, iter_date_batches,
)
from core.data.jydb_market_etl import DAILY_QUOTE_SQL, JYDBMarketETL
from core.data.jydb_raw_etl import JYDBRawStore, RawQuerySpec, training_raw_specs
from core.factors.external_source_factors import ExternalSourceFactors


class JYDBFeatureStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "jydb_features.db")
        self.store = JYDBFeatureStore(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_wide_etl_cleans_and_persists_numeric_features(self):
        source = pd.DataFrame({
            "code": ["000001.SZ", "bad-code"],
            "available_date": ["2024-04-30 18:00:00", "2024-04-30"],
            "end_date": ["2024-03-31", "2024-03-31"],
            "ROE": [12.5, 99.0],
            "PE": ["8.2", "invalid"],
            "JSID": [1, 2],
        })
        count = self.store.upsert_wide_frame(
            source,
            source_table="sample",
            feature_cols=["ROE", "PE"],
            prefix="jy_",
        )
        self.assertEqual(count, 2)
        self.assertEqual(self.store.get_feature_names(), ["jy_PE", "jy_ROE"])

    def test_pit_alignment_never_backfills_future_values(self):
        source = pd.DataFrame({
            "code": ["000001", "000001"],
            "available_date": ["2024-04-30", "2024-08-30"],
            "end_date": ["2024-03-31", "2024-06-30"],
            "ROE": [10.0, 15.0],
        })
        self.store.upsert_wide_frame(
            source, source_table="LC_MainIndexNew",
            feature_cols=["ROE"], prefix="jy_fin_",
        )
        dates = pd.Series(["2024-04-29", "2024-04-30", "2024-08-29", "2024-08-30"])
        aligned = self.store.get_pit_series("000001", dates)

        self.assertTrue(np.isnan(aligned.loc[0, "jy_fin_ROE"]))
        self.assertEqual(aligned.loc[1, "jy_fin_ROE"], 10.0)
        self.assertEqual(aligned.loc[2, "jy_fin_ROE"], 10.0)
        self.assertEqual(aligned.loc[3, "jy_fin_ROE"], 15.0)

    def test_higher_revision_wins_without_changing_availability_date(self):
        base = pd.DataFrame({
            "code": ["000001"], "available_date": ["2024-04-30"],
            "end_date": ["2024-03-31"], "ROE": [10.0],
        })
        self.store.upsert_wide_frame(
            base, source_table="LC_MainIndexNew", feature_cols=["ROE"],
            prefix="jy_fin_", revision=0,
        )
        revised = base.assign(ROE=11.0)
        self.store.upsert_wide_frame(
            revised, source_table="LC_MainIndexNew", feature_cols=["ROE"],
            prefix="jy_fin_", revision=1,
        )
        aligned = self.store.get_pit_series(
            "000001", pd.Series(["2024-04-30", "2024-05-01"])
        )
        self.assertEqual(aligned["jy_fin_ROE"].tolist(), [11.0, 11.0])

    def test_external_factor_adapter_preserves_daily_index(self):
        self.store.upsert_wide_frame(
            pd.DataFrame({
                "code": ["000001"], "available_date": ["2024-01-02"],
                "PE": [8.5],
            }),
            source_table="LC_DIndicesForValuation",
            end_date_col=None,
            feature_cols=["PE"],
            prefix="jy_val_",
        )
        daily = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "close": [10.0, 10.1, 10.2],
        }, index=[10, 11, 12])
        factors = ExternalSourceFactors(self.db_path).calculate_series("000001", daily)
        self.assertEqual(factors.index.tolist(), [10, 11, 12])
        self.assertTrue(np.isnan(factors.loc[10, "jy_val_PE"]))
        self.assertEqual(factors.loc[11, "jy_val_PE"], 8.5)
        self.assertEqual(factors.loc[12, "jy_val_PE"], 8.5)

    def test_daily_wide_storage_preserves_low_cardinality_dimensions(self):
        count = self.store.upsert_daily_wide_frame(
            pd.DataFrame({
                "code": ["000001", "000001"],
                "available_date": ["2024-01-02", "2024-01-02"],
                "ValueRange": [1, 2],
                "BuyValue": [100.0, 250.0],
                "SellValue": [80.0, 200.0],
            }),
            feature_cols=["BuyValue", "SellValue"],
            dimension_cols=["ValueRange"],
            prefix="jy_flow_",
        )
        self.assertEqual(count, 4)
        result = self.store.get_daily_series(
            "000001", pd.Series(["2024-01-02"])
        )
        self.assertEqual(result.loc[0, "jy_flow_BuyValue__ValueRange_1"], 100.0)
        self.assertEqual(result.loc[0, "jy_flow_BuyValue__ValueRange_2"], 250.0)

    def test_incremental_etl_uses_watermark_with_overlap(self):
        self.store.set_watermark("LC_MainIndexNew", "2024-03-10")

        class RecordingETL(JYDBETL):
            def __init__(self, store):
                super().__init__(store)
                self.calls = []

            def extract_table(self, spec, start_date, end_date, chunksize=100_000, stock_codes=None):
                self.calls.append((spec.name, start_date, end_date))
                return 1

        etl = RecordingETL(self.store)
        result = etl.run_incremental(
            "2024-03-20", tables=["LC_MainIndexNew"], overlap_days=5
        )
        self.assertEqual(result["LC_MainIndexNew"], 1)
        self.assertEqual(
            etl.calls, [("LC_MainIndexNew", "2024-03-05", "2024-03-20")]
        )

    def test_date_batches_are_closed_contiguous_intervals(self):
        self.assertEqual(
            list(iter_date_batches("2024-01-15", "2024-03-02", 1)),
            [
                ("2024-01-15", "2024-02-14"),
                ("2024-02-15", "2024-03-02"),
            ],
        )

    def test_batched_feature_etl_updates_each_committed_interval(self):
        class RecordingETL(JYDBETL):
            def __init__(self, store):
                super().__init__(store)
                self.calls = []

            def extract_table(self, spec, start_date, end_date, chunksize=100_000, stock_codes=None):
                self.calls.append((spec.name, start_date, end_date))
                return 2

        etl = RecordingETL(self.store)
        result = etl.run_batched(
            "2024-01-01", "2025-01-01",
            tables=["LC_MainIndexNew"], batch_months=12,
        )
        self.assertEqual(result["LC_MainIndexNew"], 4)
        self.assertEqual(
            etl.calls,
            [
                ("LC_MainIndexNew", "2024-01-01", "2024-12-31"),
                ("LC_MainIndexNew", "2025-01-01", "2025-01-01"),
            ],
        )

    def test_first_phase_structured_metadata_specs_are_registered(self):
        expected = {
            "LC_PerformanceLetters", "LC_AuditOpinion", "LC_ShareStru",
            "LC_SharesFloatingSchedule", "LC_ActualController",
            "LC_SHSZHSCHoldings", "MT_TradingDetail", "LC_CSIIndustry",
            "LC_ReserveReportDate", "LC_AShareSeasonedNewIssue",
            "LC_ASharePlacement", "LC_IndexComponentsWeight",
        }
        self.assertTrue(expected.issubset(DEFAULT_TABLE_SPECS))
        self.assertEqual(DEFAULT_TABLE_SPECS["LC_SHSZHSCHoldings"].storage, "daily")
        self.assertIn(
            "OpinionType", DEFAULT_TABLE_SPECS["LC_AuditOpinion"].dimension_cols
        )

    def test_market_etl_schema_migrates_for_existing_backtest_data_handler(self):
        market_db = os.path.join(self.temp_dir.name, "market.db")
        import sqlite3
        with closing(sqlite3.connect(market_db)) as conn, conn:
            conn.execute(
                "CREATE TABLE daily_data(code TEXT, date TEXT, PRIMARY KEY(code,date))"
            )
        JYDBMarketETL(market_db).initialize()
        with closing(sqlite3.connect(market_db)) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(daily_data)")}
        self.assertTrue(
            {"tradestatus", "pctChg", "psTTM", "pcfNcfTTM"}.issubset(columns)
        )

    def test_market_etl_unifies_main_board_and_star_market_quotes(self):
        self.assertIn("QT_DailyQuote", DAILY_QUOTE_SQL)
        self.assertIn("LC_STIBDailyQuote", DAILY_QUOTE_SQL)

    def test_training_raw_specs_include_all_market_and_feature_sources(self):
        specs = training_raw_specs()
        self.assertTrue(
            {"SecuMain", "QT_DailyQuote", "LC_STIBDailyQuote",
             "QT_AdjustingFactor", "LC_SpecialTrade"}.issubset(specs)
        )
        self.assertTrue(set(DEFAULT_TABLE_SPECS).issubset(specs))

    def test_raw_store_replaces_overlapping_date_batch(self):
        raw_path = os.path.join(self.temp_dir.name, "jydb_raw.db")
        store = JYDBRawStore(raw_path)
        spec = RawQuerySpec("SampleRaw", "SELECT 1", "available_date")
        first = pd.DataFrame({
            "code": ["000001", "000002"],
            "available_date": ["2024-01-01", "2024-01-02"],
            "value": [1.0, 2.0],
        })
        second = pd.DataFrame({
            "code": ["000001"],
            "available_date": ["2024-01-01"],
            "value": [3.0],
        })
        store.replace_batch(spec, [first], "2024-01-01", "2024-01-02")
        store.replace_batch(spec, [second], "2024-01-01", "2024-01-01")
        import sqlite3
        with closing(sqlite3.connect(raw_path)) as conn:
            rows = conn.execute(
                "SELECT code,substr(available_date,1,10),value "
                "FROM SampleRaw ORDER BY code"
            ).fetchall()
        self.assertEqual(
            rows,
            [("000001", "2024-01-01", 3.0), ("000002", "2024-01-02", 2.0)],
        )


if __name__ == "__main__":
    unittest.main()
