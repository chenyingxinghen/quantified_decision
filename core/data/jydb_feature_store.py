"""聚源数据的本地点时（PIT）特征存储与 ETL。

远端表先被标准化成长表并写入 SQLite。所有下游模块只按
``available_date <= 交易日`` 读取，避免按报告期回填造成前视偏差。
"""
from __future__ import annotations

import os
import sqlite3
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TECHNICAL_COLUMNS = {
    "id", "jsid", "xgrq", "updatetime", "inserttime", "rowversion"
}


def iter_date_batches(
    start_date: str, end_date: str, batch_months: int = 12
) -> Iterator[Tuple[str, str]]:
    """按自然月切分闭区间，便于大批量抽取分段提交和断点续跑。"""
    if batch_months <= 0:
        raise ValueError("batch_months 必须为正整数")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date 不能晚于 end_date")
    cursor = start
    while cursor <= end:
        batch_end = min(cursor + pd.DateOffset(months=batch_months) - pd.Timedelta(days=1), end)
        yield cursor.strftime("%Y-%m-%d"), batch_end.strftime("%Y-%m-%d")
        cursor = batch_end + pd.Timedelta(days=1)


class JYDBFeatureStore:
    """统一的聚源 PIT 特征库。"""

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)

    def connect(self) -> sqlite3.Connection:
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=120000")
        conn.execute("PRAGMA cache_size=-80000")       # 80 MB page cache
        conn.execute("PRAGMA wal_autocheckpoint=0")    # 禁用自动 checkpoint；由主进程手动 TRUNCATE
        return conn

    @contextmanager
    def session(self):
        """事务结束后显式关闭连接，避免 Windows 文件句柄残留。"""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.session() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS pit_features (
                    code TEXT NOT NULL,
                    available_date TEXT NOT NULL,
                    end_date TEXT NOT NULL DEFAULT '',
                    source_table TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    loaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (
                        code, available_date, end_date,
                        source_table, feature_name, revision
                    )
                );
                CREATE INDEX IF NOT EXISTS idx_pit_features_code_date
                    ON pit_features(code, available_date);
                CREATE INDEX IF NOT EXISTS idx_pit_features_name_date
                    ON pit_features(feature_name, available_date);
                CREATE TABLE IF NOT EXISTS etl_watermarks (
                    source_table TEXT PRIMARY KEY,
                    watermark TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS daily_features (
                    code TEXT NOT NULL,
                    date TEXT NOT NULL,
                    PRIMARY KEY(code, date)
                );
                CREATE INDEX IF NOT EXISTS idx_daily_features_date
                    ON daily_features(date);
                """
            )

    @staticmethod
    def _date_string(values: pd.Series) -> pd.Series:
        return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%d")

    def upsert_wide_frame(
        self,
        frame: pd.DataFrame,
        *,
        source_table: str,
        code_col: str = "code",
        available_date_col: str = "available_date",
        end_date_col: Optional[str] = "end_date",
        feature_cols: Optional[Sequence[str]] = None,
        dimension_cols: Sequence[str] = (),
        prefix: str = "jy_",
        revision: int = 0,
    ) -> int:
        """清洗宽表并写入长表，返回写入的非空特征值数量。"""
        if frame.empty:
            return 0
        required = {code_col, available_date_col}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{source_table} 缺少必要字段: {sorted(missing)}")

        data = frame.copy()
        data[code_col] = data[code_col].astype(str).str.extract(r"(\d{6})", expand=False)
        data[available_date_col] = self._date_string(data[available_date_col])
        if end_date_col and end_date_col in data.columns:
            data[end_date_col] = self._date_string(data[end_date_col]).fillna("")
        else:
            end_date_col = "__end_date"
            data[end_date_col] = ""
        data = data.dropna(subset=[code_col, available_date_col])

        dimension_cols = [col for col in dimension_cols if col in data.columns]
        excluded = {code_col, available_date_col, end_date_col, *dimension_cols}
        if feature_cols is None:
            feature_cols = [
                col for col in data.columns
                if col not in excluded and col.lower() not in TECHNICAL_COLUMNS
            ]
        feature_cols = [col for col in feature_cols if col in data.columns]
        if not feature_cols:
            return 0

        for col in feature_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        long_df = data.melt(
            id_vars=[code_col, available_date_col, end_date_col, *dimension_cols],
            value_vars=feature_cols,
            var_name="feature_name",
            value_name="feature_value",
        ).dropna(subset=["feature_value"])
        if long_df.empty:
            return 0
        long_df["feature_name"] = prefix + long_df["feature_name"].str.strip()
        if dimension_cols:
            def _dimension_suffix(row):
                parts = []
                for col in dimension_cols:
                    value = row[col]
                    if pd.isna(value):
                        token = "na"
                    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
                        token = str(int(value))
                    else:
                        token = str(value).strip()
                    token = "".join(ch if ch.isalnum() else "_" for ch in token)
                    parts.append(f"__{col}_{token}")
                return "".join(parts)
            long_df["feature_name"] += long_df.apply(_dimension_suffix, axis=1)
        long_df["source_table"] = source_table
        long_df["revision"] = int(revision)
        rows = list(long_df[[
            code_col, available_date_col, end_date_col, "source_table",
            "feature_name", "feature_value", "revision",
        ]].itertuples(index=False, name=None))

        self.initialize()
        with self.session() as conn:
            conn.executemany(
                """
                INSERT INTO pit_features (
                    code, available_date, end_date, source_table,
                    feature_name, feature_value, revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    code, available_date, end_date,
                    source_table, feature_name, revision
                ) DO UPDATE SET
                    feature_value=excluded.feature_value,
                    loaded_at=CURRENT_TIMESTAMP
                """,
                rows,
            )
        return len(rows)

    @staticmethod
    def _safe_column_name(name: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
        if not cleaned or cleaned[0].isdigit():
            cleaned = "f_" + cleaned
        return cleaned

    def upsert_daily_wide_frame(
        self,
        frame: pd.DataFrame,
        *,
        code_col: str = "code",
        date_col: str = "available_date",
        feature_cols: Sequence[str],
        dimension_cols: Sequence[str] = (),
        prefix: str = "jy_",
    ) -> int:
        """把高频数据按证券—交易日保存为宽表，返回有效单元格数。"""
        if frame.empty:
            return 0
        data = frame.copy()
        data[code_col] = data[code_col].astype(str).str.extract(r"(\d{6})", expand=False)
        data[date_col] = self._date_string(data[date_col])
        data = data.dropna(subset=[code_col, date_col])
        dimension_cols = [col for col in dimension_cols if col in data.columns]
        feature_cols = [col for col in feature_cols if col in data.columns]
        for col in feature_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        long_df = data.melt(
            id_vars=[code_col, date_col, *dimension_cols],
            value_vars=feature_cols,
            var_name="feature_name",
            value_name="feature_value",
        ).dropna(subset=["feature_value"])
        if long_df.empty:
            return 0
        long_df["feature_name"] = prefix + long_df["feature_name"].astype(str)
        for dim in dimension_cols:
            values = long_df[dim]
            tokens = values.map(
                lambda value: "na" if pd.isna(value)
                else str(int(value)) if isinstance(value, (float, np.floating)) and float(value).is_integer()
                else str(value).strip()
            ).map(lambda value: "".join(ch if ch.isalnum() else "_" for ch in value))
            long_df["feature_name"] += "__" + dim + "_" + tokens
        long_df["feature_name"] = long_df["feature_name"].map(self._safe_column_name)
        wide = long_df.pivot_table(
            index=[code_col, date_col], columns="feature_name",
            values="feature_value", aggfunc="last",
        ).reset_index()
        feature_names = [col for col in wide.columns if col not in (code_col, date_col)]
        if not feature_names:
            return 0

        self.initialize()
        with self.session() as conn:
            existing = {row[1] for row in conn.execute("PRAGMA table_info(daily_features)")}
            for col in feature_names:
                if col not in existing:
                    conn.execute(f'ALTER TABLE daily_features ADD COLUMN "{col}" REAL')
            insert_cols = ["code", "date", *feature_names]
            quoted = ",".join(f'"{col}"' for col in insert_cols)
            placeholders = ",".join("?" for _ in insert_cols)
            updates = ",".join(
                f'"{col}"=COALESCE(excluded."{col}", daily_features."{col}")'
                for col in feature_names
            )
            sql = (
                f"INSERT INTO daily_features ({quoted}) VALUES ({placeholders}) "
                f"ON CONFLICT(code,date) DO UPDATE SET {updates}"
            )
            rows = []
            for row in wide[[code_col, date_col, *feature_names]].itertuples(index=False, name=None):
                rows.append(tuple(None if pd.isna(value) else value for value in row))
            conn.executemany(sql, rows)
        return int(long_df["feature_value"].notna().sum())

    def get_feature_names(self) -> List[str]:
        if not os.path.exists(self.db_path):
            return []
        self.initialize()
        with self.session() as conn:
            rows = conn.execute(
                "SELECT DISTINCT feature_name FROM pit_features ORDER BY feature_name"
            ).fetchall()
            daily = [
                row[1] for row in conn.execute("PRAGMA table_info(daily_features)")
                if row[1] not in ("code", "date")
            ]
        return sorted({row[0] for row in rows}.union(daily))

    def get_daily_series(
        self,
        code: str,
        dates: pd.Series,
        feature_names: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=dates.index)
        if len(dates) == 0 or not os.path.exists(self.db_path):
            return result
        self.initialize()
        target_dates = pd.to_datetime(dates, errors="coerce")
        valid_dates = target_dates.dropna()
        if valid_dates.empty:
            return result
        with self.session() as conn:
            available = {
                row[1] for row in conn.execute("PRAGMA table_info(daily_features)")
                if row[1] not in ("code", "date")
            }
            selected = list(feature_names) if feature_names else sorted(available)
            selected = [name for name in selected if name in available]
            if not selected:
                return result
            cols = ",".join(f'"{name}"' for name in selected)
            raw = pd.read_sql_query(
                f"SELECT date,{cols} FROM daily_features "
                "WHERE code=? AND date>=? AND date<=? ORDER BY date",
                conn,
                params=(
                    str(code), valid_dates.min().strftime("%Y-%m-%d"),
                    valid_dates.max().strftime("%Y-%m-%d"),
                ),
            )
        if raw.empty:
            return result
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.drop_duplicates("date", keep="last").set_index("date")
        aligned = raw.reindex(target_dates.values)
        aligned.index = dates.index
        return aligned.replace([np.inf, -np.inf], np.nan)

    def get_pit_series(
        self,
        code: str,
        dates: pd.Series,
        feature_names: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """将每个交易日对齐到当时最新可知的各项特征。"""
        result = pd.DataFrame(index=dates.index)
        if len(dates) == 0 or not os.path.exists(self.db_path):
            return result
        self.initialize()
        target_dates = pd.to_datetime(dates, errors="coerce")
        max_date = target_dates.max()
        if pd.isna(max_date):
            return result

        params: List[object] = [str(code), max_date.strftime("%Y-%m-%d")]
        feature_filter = ""
        if feature_names:
            placeholders = ",".join("?" for _ in feature_names)
            feature_filter = f" AND feature_name IN ({placeholders})"
            params.extend(feature_names)
        query = f"""
            SELECT available_date, end_date, source_table, feature_name,
                   feature_value, revision, loaded_at
            FROM pit_features
            WHERE code = ? AND available_date <= ? {feature_filter}
            ORDER BY available_date, end_date, revision, loaded_at
        """
        with self.session() as conn:
            raw = pd.read_sql_query(query, conn, params=params)
        if raw.empty:
            return result

        raw["available_date"] = pd.to_datetime(raw["available_date"])
        # 同一特征同一可用日保留最高 revision；相同 revision 保留最后加载版本。
        raw = raw.drop_duplicates(
            ["available_date", "feature_name"], keep="last"
        )
        wide = raw.pivot(
            index="available_date", columns="feature_name", values="feature_value"
        ).sort_index()
        # 将特征变更日与目标交易日合并后向前填充，严格禁止向后填充。
        union_index = wide.index.union(pd.DatetimeIndex(target_dates.dropna())).sort_values()
        aligned = wide.reindex(union_index).ffill().reindex(target_dates.values)
        aligned.index = dates.index
        if feature_names:
            aligned = aligned.reindex(columns=list(feature_names))
        return aligned.replace([np.inf, -np.inf], np.nan)

    def set_watermark(self, source_table: str, watermark: str) -> None:
        self.initialize()
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO etl_watermarks(source_table, watermark)
                VALUES (?, ?)
                ON CONFLICT(source_table) DO UPDATE SET
                    watermark=excluded.watermark,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (source_table, str(watermark)),
            )

    def get_watermark(self, source_table: str) -> Optional[str]:
        if not os.path.exists(self.db_path):
            return None
        self.initialize()
        with self.session() as conn:
            row = conn.execute(
                "SELECT watermark FROM etl_watermarks WHERE source_table=?",
                (source_table,),
            ).fetchone()
        return row[0] if row else None


@dataclass(frozen=True)
class JYDBTableSpec:
    name: str
    sql: str
    available_date_col: str
    end_date_col: Optional[str]
    feature_cols: Sequence[str]
    prefix: str
    dimension_cols: Sequence[str] = ()
    storage: str = "pit"
    parameter_multiplier: int = 1
    earliest_date: Optional[str] = None


VALUATION_FIELDS = (
    "TotalMV", "TotalMV2", "TotalASMV", "NegotiableMV", "PELYR", "PE",
    "PETTMCut", "ForwardPEQA", "ForwardPEHR", "PEG", "PB", "PCF",
    "PCFTTM", "PCFS", "PCFSTTM", "PS", "PSTTM", "DividendRatioLYR",
    "DividendRatio", "EnterpriseValueW", "EnterpriseValueN", "EVToEBITDA",
    "EVToOR", "EVToFCFF", "EVToCFO", "EVToGP", "MVA", "FloatMVA",
)

FINANCIAL_INDEX_FIELDS = (
    "EPS", "BasicEPS", "DilutedEPS", "NetAssetPS", "ROE", "ROECut",
    "ROEWeighted", "ROA", "ROIC", "GrossIncomeRatio", "NetProfitRatio",
    "OperatingProfitRatio", "EBIT", "EBITDA", "OperatingRevenueGrowRate",
    "NetProfitGrowRate", "TotalAssetGrowRate", "NetAssetGrowRate",
    "CurrentRatio", "QuickRatio", "SuperQuickRatio", "DebtAssetsRatio",
    "DebtEquityRatio", "InterestCover", "TotalAssetTRate", "ARTRate",
    "ARTDays", "InventoryTRate", "InventoryTDays", "OperCycle",
    "NetOperateCashFlow", "FreeCashFlow", "OperCashFlowPS",
    "NetProfitCashCover", "OperatingRevenueCashCover",
)

FORECAST_FIELDS = (
    "EValueFloor", "EValueCeiling",
    "EGrowRateFloorC", "EGrowthRateCeilC", "EEarningFloor", "EEarningCeiling",
    "LastEarning", "EProfitFloor", "EProfitCeiling", "LastProfit",
    "EEPSFloor", "EEPSCeiling", "LastEPS", "NPYOYConsistentForecast",
)

FREE_FLOAT_FIELDS = (
    "TotalAShare", "AFloats", "InactiveFloats", "GovInactFloats",
    "NonGovInactFLoats", "ForgnInactFloats", "NaturPersnInactFloats",
    "FreeFloats", "FreeFloatRatio", "AdjFreeFloatRatio", "AdjFreeFloats",
)

SHAREHOLDER_NUMBER_FIELDS = (
    "SHNum", "AverageHoldSum", "HoldProportionPAccount", "ProportionChange",
    "AvgHoldSumGRQuarter", "ProportionGRQuarter", "AvgHoldSumGRHalfAYear",
    "ProportionGRHalfAYear", "ASHNum", "AAverageHoldSum",
    "AFAverageHoldSum", "AFProportionChange", "AFAvgHoldSumGRQuarter",
)

PLEDGE_FIELDS = (
    "AccuFPSharesCalc", "AccuProportionCalc",
)

CAPITAL_FLOW_FIELDS = (
    "BuyVolume", "SellVolume", "BuyValue", "SellValue",
)

DIVIDEND_FIELDS = (
    "IfDividend", "EPS", "BonusShareRatio", "TranAddShareRaio", "CashDiviRMB",
    "ActualCashDiviRMB", "DiviBase", "TotalCashDiviComRMB", "DistributeTimes",
    "MainForm", "IFSchemeChange",
)

BUYBACK_FIELDS = (
    "VolumeFloor", "VolumeCeiling", "BuybackSum", "Percentage", "PriceFloor",
    "PriceCeiling", "BuybackPrice", "ValueFloor", "ValueCeiling",
    "BuybackMoney", "EventProcedure", "BuybackModeCode",
)

PERFORMANCE_LETTER_FIELDS = (
    "OperatingRevenue", "GrossProfit", "OperatingProfit", "TotalProfit",
    "NPParentCompanyOwners", "NetProfitCut", "NetOperateCashFlow",
    "TotalAssets", "SEWithoutMI", "TotalShares", "BasicEPS", "EPSCut",
    "ROE", "ROECut", "NetOperateCashFlowPS", "NetAssetPS",
    "OperatingRevenueYOY", "GrossProfitYOY", "OperatingProfitYOY",
    "TotalProfitYOY", "NPParentCompanyOwnersYOY", "NetProfitCutYOY",
    "NetOperateCashFlowYOY", "TotalAssetsToOpening", "SEWithoutMIToOpening",
    "BasicEPSYOY", "EPSCutYOY", "ROEYOY", "ROECutYOY",
    "NetOperateCashFlowPSYOY", "NetAssetPSToOpening",
)

SHARE_STRUCTURE_FIELDS = (
    "TotalShares", "Ashares", "AFloats", "AFloatListed", "FloatShare",
    "NonRestrictedShares", "RestrictedShares", "RestrictedAShares",
    "StateHolding", "SLegalPersonHolding", "DLegalPersonHolding",
    "DNaturalPersonHolding", "ForeignHolding", "ForeignHoldingAshares",
    "ManagementShares", "MutualFundShares",
)

FLOATING_SCHEDULE_FIELDS = (
    "NewMarketableAShares", "Proportion1", "AccuMarketableAShares",
    "NonMarketableAShares", "TotalAShares", "Proportion2", "TotalShares",
    "DaysToFloatingAtAnnouncement",
)

NORTHBOUND_HOLDING_FIELDS = (
    "SharesHolding", "Holdratio", "AdjustedHoldratio", "AdjustedSharesHolding",
)

MARGIN_TRADING_FIELDS = (
    "FinanceValue", "FinanceBuyValue", "FinanceRefundValue",
    "SecurityVolume", "SecuritySellVolume", "SecurityRefundVolume",
    "SecurityValue", "TradingValue", "FinaInTotalRatio", "SecuInTotalRatio",
)

INDUSTRY_CODE_FIELDS = (
    "FstIndustryCSI", "SndIndustryCSI", "TrdIndustryCSI", "FthIndustryCSI",
    "FstIndustryCSRS", "SndIndustryCSRS",
)

SEASONED_ISSUE_FIELDS = (
    "IssuePriceCeiling", "IssuePriceFloor", "IssueVolCeiling", "IssueVolFloor",
    "PlannedProceeds", "AdjustedIssuePrice", "AdjustedIssueVol",
    "AdjustedVolFloor", "AdjustedPriceCeiling", "DiscountRatePerShare",
)

PLACEMENT_FIELDS = (
    "PlaRatioPlanned", "PlaPriceCeiling", "PlaPriceFloor", "BaseShares",
    "PlannedPlaVol", "PlannedPlaProceeds", "AdjustedPlaRatioPl",
    "AdjustedPlaVolPl", "PlannedBaseShares",
)

A_SHARE_FILTER = """
          AND s.SecuMarket IN (18, 83, 90)
          AND (
              s.SecuCode LIKE '00%' OR s.SecuCode LIKE '30%'
              OR s.SecuCode LIKE '60%' OR s.SecuCode LIKE '68%'
              OR s.SecuCode LIKE '43%' OR s.SecuCode LIKE '83%'
              OR s.SecuCode LIKE '87%' OR s.SecuCode LIKE '92%'
          )
"""


def _select_sql(table: str, date_col: str, fields: Sequence[str], company_join: bool) -> str:
    join_key = "CompanyCode" if company_join else "InnerCode"
    selected = ", ".join(f"t.{field}" for field in fields)
    return f"""
        SELECT s.SecuCode AS code, t.{date_col} AS available_date,
               {('t.EndDate AS end_date, ' if company_join else '')}{selected}
        FROM {table} t
        JOIN SecuMain s ON t.{join_key} = s.{join_key}
        WHERE t.{date_col} >= ? AND t.{date_col} <= ?
          AND s.SecuCategory = 1
          {A_SHARE_FILTER}
    """


def _event_sql(
    table: str,
    date_col: str,
    fields: Sequence[str],
    *,
    join_key: str,
    end_date_col: Optional[str] = None,
) -> str:
    selected = ", ".join(f"t.{field}" for field in fields)
    end_select = f"t.{end_date_col} AS end_date, " if end_date_col else ""
    return f"""
        SELECT s.SecuCode AS code, t.{date_col} AS available_date,
               {end_select}{selected}
        FROM {table} t
        JOIN SecuMain s ON t.{join_key} = s.{join_key}
        WHERE t.{date_col} >= ? AND t.{date_col} <= ?
          AND s.SecuCategory = 1
          {A_SHARE_FILTER}
    """


DEFAULT_TABLE_SPECS: Dict[str, JYDBTableSpec] = {
    "LC_DIndicesForValuation": JYDBTableSpec(
        name="LC_DIndicesForValuation",
        sql=_select_sql("LC_DIndicesForValuation", "TradingDay", VALUATION_FIELDS, False),
        available_date_col="available_date",
        end_date_col=None,
        feature_cols=VALUATION_FIELDS,
        prefix="jy_val_",
        storage="daily",
    ),
    "LC_MainIndexNew": JYDBTableSpec(
        name="LC_MainIndexNew",
        # InfoPublDate 是市场可知日期；绝不能使用 EndDate 作为生效日。
        sql=_select_sql("LC_MainIndexNew", "InfoPublDate", FINANCIAL_INDEX_FIELDS, True),
        available_date_col="available_date",
        end_date_col="end_date",
        feature_cols=FINANCIAL_INDEX_FIELDS,
        prefix="jy_fin_",
        earliest_date="1991-03-01",
    ),
    "LC_PerformanceForecast": JYDBTableSpec(
        name="LC_PerformanceForecast",
        sql=_event_sql(
            "LC_PerformanceForecast", "InfoPublDate",
            ("ForcastType", "ForecastObject") + FORECAST_FIELDS,
            join_key="CompanyCode", end_date_col="EndDate",
        ),
        available_date_col="available_date", end_date_col="end_date",
        feature_cols=FORECAST_FIELDS, prefix="jy_forecast_",
        dimension_cols=("ForcastType", "ForecastObject"),
        earliest_date="1999-01-01",
    ),
    "LC_FreeFloat": JYDBTableSpec(
        name="LC_FreeFloat",
        sql=_event_sql(
            "LC_FreeFloat", "InfoPublDate", FREE_FLOAT_FIELDS,
            join_key="InnerCode", end_date_col="ChangeDate",
        ),
        available_date_col="available_date", end_date_col="end_date",
        feature_cols=FREE_FLOAT_FIELDS, prefix="jy_float_",
        earliest_date="1992-05-01",
    ),
    "LC_SHNumber": JYDBTableSpec(
        name="LC_SHNumber",
        sql=_event_sql(
            "LC_SHNumber", "InfoPublDate", SHAREHOLDER_NUMBER_FIELDS,
            join_key="CompanyCode", end_date_col="EndDate",
        ),
        available_date_col="available_date", end_date_col="end_date",
        feature_cols=SHAREHOLDER_NUMBER_FIELDS, prefix="jy_holder_",
        earliest_date="1991-10-01",
    ),
    # 该统计表没有独立公告日，EndDate 只能作为保守可用日。生产接入时若能从
    # 公告关联表取得发布时间，应在源查询中替换 available_date。
    "LC_ShareFPSta": JYDBTableSpec(
        name="LC_ShareFPSta",
        sql=f"""
            SELECT s.SecuCode AS code, t.EndDate AS available_date,
                   t.EndDate AS end_date, t.Category,
                   SUM(COALESCE(t.AccuFPSharesCalc, t.AccuFPShares)) AS AccuFPSharesCalc,
                   SUM(COALESCE(t.AccuProportionCalc, t.AccuProportion)) AS AccuProportionCalc
            FROM LC_ShareFPSta t
            JOIN SecuMain s ON t.CompanyCode = s.CompanyCode
            WHERE t.EndDate >= ? AND t.EndDate <= ?
              AND s.SecuCategory = 1
              {A_SHARE_FILTER}
            GROUP BY s.SecuCode, t.EndDate, t.Category
        """,
        available_date_col="available_date", end_date_col="end_date",
        feature_cols=PLEDGE_FIELDS, prefix="jy_pledge_",
        dimension_cols=("Category",),
        earliest_date="2002-10-01",
    ),
    "QT_TradingCapitalFlow": JYDBTableSpec(
        name="QT_TradingCapitalFlow",
        sql=f"""
            SELECT s.SecuCode AS code, t.TradingDate AS available_date,
                   t.ValueRange, t.BuyVolume, t.SellVolume, t.BuyValue, t.SellValue
            FROM QT_TradingCapitalFlow t
            JOIN SecuMain s ON t.InnerCode = s.InnerCode
            JOIN QT_DailyQuote q
              ON t.InnerCode = q.InnerCode AND t.TradingDate = q.TradingDay
            WHERE t.TradingDate >= ? AND t.TradingDate <= ?
              AND s.SecuCategory = 1
              {A_SHARE_FILTER}
        """,
        available_date_col="available_date", end_date_col=None,
        feature_cols=CAPITAL_FLOW_FIELDS, prefix="jy_flow_",
        dimension_cols=("ValueRange",),
        storage="daily",
    ),
    "LC_Dividend": JYDBTableSpec(
        name="LC_Dividend",
        sql=_event_sql(
            "LC_Dividend", "AdvanceDate", DIVIDEND_FIELDS,
            join_key="InnerCode", end_date_col="EndDate",
        ),
        available_date_col="available_date", end_date_col="end_date",
        feature_cols=DIVIDEND_FIELDS, prefix="jy_dividend_",
        earliest_date="1991-01-01",
    ),
    "LC_Buyback": JYDBTableSpec(
        name="LC_Buyback",
        sql=_event_sql(
            "LC_Buyback", "FirstPublDate", BUYBACK_FIELDS,
            join_key="CompanyCode",
        ),
        available_date_col="available_date", end_date_col=None,
        feature_cols=BUYBACK_FIELDS, prefix="jy_buyback_",
        earliest_date="1994-06-01",
    ),
    "LC_PerformanceLetters": JYDBTableSpec(
        name="LC_PerformanceLetters",
        sql=_event_sql(
            "LC_PerformanceLetters", "InfoPublDate", PERFORMANCE_LETTER_FIELDS,
            join_key="CompanyCode", end_date_col="EndDate",
        ),
        available_date_col="available_date", end_date_col="end_date",
        feature_cols=PERFORMANCE_LETTER_FIELDS, prefix="jy_letter_",
        earliest_date="2004-02-01",
    ),
    "LC_AuditOpinion": JYDBTableSpec(
        name="LC_AuditOpinion",
        sql=f"""
            SELECT s.SecuCode AS code, t.InfoPublDate AS available_date,
                   t.EndDate AS end_date, t.OpinionType, t.AuditReportsType,
                   CAST(1 AS float) AS AuditEvent
            FROM LC_AuditOpinion t
            JOIN SecuMain s ON t.CompanyCode = s.CompanyCode
            WHERE t.InfoPublDate >= ? AND t.InfoPublDate <= ?
              AND s.SecuCategory = 1
              {A_SHARE_FILTER}
        """,
        available_date_col="available_date", end_date_col="end_date",
        feature_cols=("AuditEvent",), prefix="jy_audit_",
        dimension_cols=("OpinionType", "AuditReportsType"),
        earliest_date="1991-03-01",
    ),
    "LC_ShareStru": JYDBTableSpec(
        name="LC_ShareStru",
        sql=_event_sql(
            "LC_ShareStru", "InfoPublDate", SHARE_STRUCTURE_FIELDS,
            join_key="CompanyCode", end_date_col="EndDate",
        ),
        available_date_col="available_date", end_date_col="end_date",
        feature_cols=SHARE_STRUCTURE_FIELDS, prefix="jy_share_",
        earliest_date="1989-03-01",
    ),
    "LC_SharesFloatingSchedule": JYDBTableSpec(
        name="LC_SharesFloatingSchedule",
        sql=f"""
            SELECT s.SecuCode AS code, t.InfoPublDate AS available_date,
                   t.StartDateForFloating AS end_date, t.SourceType,
                   t.NewMarketableAShares, t.Proportion1, t.AccuMarketableAShares,
                   t.NonMarketableAShares, t.TotalAShares, t.Proportion2,
                   t.TotalShares,
                   DATEDIFF(day, t.InfoPublDate, t.StartDateForFloating)
                       AS DaysToFloatingAtAnnouncement
            FROM LC_SharesFloatingSchedule t
            JOIN SecuMain s ON t.InnerCode = s.InnerCode
            WHERE t.InfoPublDate >= ? AND t.InfoPublDate <= ?
              AND s.SecuCategory = 1
              {A_SHARE_FILTER}
        """,
        available_date_col="available_date", end_date_col="end_date",
        feature_cols=FLOATING_SCHEDULE_FIELDS, prefix="jy_unlock_",
        dimension_cols=("SourceType",),
        earliest_date="2005-11-01",
    ),
    "LC_ActualController": JYDBTableSpec(
        name="LC_ActualController",
        sql=f"""
            SELECT s.SecuCode AS code, t.InfoPublDate AS available_date,
                   t.EndDate AS end_date, t.ControllerNature, t.EconomicNature,
                   t.NationalityCode, CAST(1 AS float) AS ControllerEvent
            FROM LC_ActualController t
            JOIN SecuMain s ON t.CompanyCode = s.CompanyCode
            WHERE t.InfoPublDate >= ? AND t.InfoPublDate <= ?
              AND s.SecuCategory = 1
              {A_SHARE_FILTER}
        """,
        available_date_col="available_date", end_date_col="end_date",
        feature_cols=("ControllerEvent",), prefix="jy_controller_",
        dimension_cols=(),
        earliest_date="1998-03-01",
    ),
    "LC_SHSZHSCHoldings": JYDBTableSpec(
        name="LC_SHSZHSCHoldings",
        sql=f"""
            SELECT s.SecuCode AS code, t.EndDate AS available_date,
                   t.TradingType, t.SharesHolding, t.Holdratio,
                   t.AdjustedHoldratio, t.AdjustedSharesHolding
            FROM LC_SHSZHSCHoldings t
            JOIN SecuMain s ON t.InnerCode = s.InnerCode
            WHERE t.EndDate >= ? AND t.EndDate <= ?
              AND s.SecuCategory = 1
              {A_SHARE_FILTER}
        """,
        available_date_col="available_date", end_date_col=None,
        feature_cols=NORTHBOUND_HOLDING_FIELDS, prefix="jy_north_",
        dimension_cols=("TradingType",), storage="daily",
        earliest_date="2016-12-01",
    ),
    "MT_TradingDetail": JYDBTableSpec(
        name="MT_TradingDetail",
        sql=_select_sql("MT_TradingDetail", "TradingDay", MARGIN_TRADING_FIELDS, False),
        available_date_col="available_date", end_date_col=None,
        feature_cols=MARGIN_TRADING_FIELDS, prefix="jy_margin_", storage="daily",
        earliest_date="2010-03-31",
    ),
    "LC_CSIIndustry": JYDBTableSpec(
        name="LC_CSIIndustry",
        sql=_event_sql(
            "LC_CSIIndustry", "EndDate", INDUSTRY_CODE_FIELDS,
            join_key="InnerCode", end_date_col="EndDate",
        ),
        available_date_col="available_date", end_date_col="end_date",
        feature_cols=INDUSTRY_CODE_FIELDS, prefix="jy_industry_",
        earliest_date="2012-11-01",
    ),
    "LC_ReserveReportDate": JYDBTableSpec(
        name="LC_ReserveReportDate",
        sql=f"""
            SELECT s.SecuCode AS code, t.InfoPublDate AS available_date,
                   t.EndDate AS end_date, t.BulletinType, t.NoticeType,
                   CAST(COALESCE(t.CorrectNum, 0) AS float) AS CorrectNum,
                   DATEDIFF(day, t.InfoPublDate, t.NoticeStartDate) AS DaysToNoticeStart,
                   DATEDIFF(day, t.InfoPublDate, t.NoticeEndDate) AS DaysToNoticeEnd,
                   CAST(COALESCE(t.IfEffected, 0) AS float) AS IfEffected
            FROM LC_ReserveReportDate t
            JOIN SecuMain s ON t.InnerCode = s.InnerCode
            WHERE t.InfoPublDate >= ? AND t.InfoPublDate <= ?
              AND s.SecuCategory = 1
              {A_SHARE_FILTER}
        """,
        available_date_col="available_date", end_date_col="end_date",
        feature_cols=("CorrectNum", "DaysToNoticeStart", "DaysToNoticeEnd", "IfEffected"),
        prefix="jy_disclosure_", dimension_cols=("BulletinType", "NoticeType"),
        earliest_date="1990-12-01",
    ),
    "LC_AShareSeasonedNewIssue": JYDBTableSpec(
        name="LC_AShareSeasonedNewIssue",
        sql=f"""
            SELECT s.SecuCode AS code,
                   COALESCE(t.LatestAdvanceDate, t.AdvanceDate, t.InitialInfoPublDate)
                       AS available_date,
                   t.PricingBaseDate AS end_date, t.IssueType, t.EventProcedureCode,
                   {', '.join('t.' + field for field in SEASONED_ISSUE_FIELDS)}
            FROM LC_AShareSeasonedNewIssue t
            JOIN SecuMain s ON t.InnerCode = s.InnerCode
            WHERE COALESCE(t.LatestAdvanceDate, t.AdvanceDate, t.InitialInfoPublDate) >= ?
              AND COALESCE(t.LatestAdvanceDate, t.AdvanceDate, t.InitialInfoPublDate) <= ?
              AND s.SecuCategory = 1
              {A_SHARE_FILTER}
        """,
        available_date_col="available_date", end_date_col="end_date",
        feature_cols=SEASONED_ISSUE_FIELDS, prefix="jy_seasoned_",
        dimension_cols=("IssueType",),
        earliest_date="1991-07-01",
    ),
    "LC_ASharePlacement": JYDBTableSpec(
        name="LC_ASharePlacement",
        sql=f"""
            SELECT s.SecuCode AS code,
                   COALESCE(t.LatestInfoPublDate, t.AdvanceDate, t.InitialInfoPublDate)
                       AS available_date,
                   t.PlaBaseDate AS end_date, t.IssueMethod, t.EventProcedureCode,
                   {', '.join('t.' + field for field in PLACEMENT_FIELDS)}
            FROM LC_ASharePlacement t
            JOIN SecuMain s ON t.InnerCode = s.InnerCode
            WHERE COALESCE(t.LatestInfoPublDate, t.AdvanceDate, t.InitialInfoPublDate) >= ?
              AND COALESCE(t.LatestInfoPublDate, t.AdvanceDate, t.InitialInfoPublDate) <= ?
              AND s.SecuCategory = 1
              {A_SHARE_FILTER}
        """,
        available_date_col="available_date", end_date_col="end_date",
        feature_cols=PLACEMENT_FIELDS, prefix="jy_placement_",
        dimension_cols=(),
        earliest_date="1991-03-01",
    ),
    "LC_IndexComponentsWeight": JYDBTableSpec(
        name="LC_IndexComponentsWeight",
        sql=f"""
            SELECT s.SecuCode AS code, t.EndDate AS available_date,
                   t.EndDate AS end_date, idx.SecuCode AS IndexCode, t.Weight
            FROM LC_IndexComponentsWeight t
            JOIN SecuMain s ON t.InnerCode = s.InnerCode
            JOIN SecuMain idx ON t.IndexCode = idx.InnerCode
            WHERE t.EndDate >= ? AND t.EndDate <= ?
              AND idx.SecuCode IN ('000016', '000300', '000905', '000852', '399006')
              AND idx.SecuMarket IN (83, 90)
              AND s.SecuCategory = 1
              {A_SHARE_FILTER}
        """,
        available_date_col="available_date", end_date_col="end_date",
        feature_cols=("Weight",), prefix="jy_index_",
        dimension_cols=("IndexCode",),
        earliest_date="1997-01-01",
    ),
}


class JYDBETL:
    """从聚源 SQL Server 抽取选定宽表并加载到本地 PIT 特征库。"""

    def __init__(self, store: JYDBFeatureStore, connection_factory=None):
        self.store = store
        self.connection_factory = connection_factory

    def _connect(self):
        if self.connection_factory is not None:
            return self.connection_factory()
        try:
            import pyodbc
        except ImportError:
            pyodbc = None
        if pyodbc is not None:
            from config.jydb_config import build_connection_string
            return pyodbc.connect(build_connection_string(), timeout=60)
        try:
            import adodbapi
        except ImportError as exc:
            raise RuntimeError("运行聚源抽取需要安装 pyodbc 或 adodbapi") from exc
        from config.jydb_config import build_ado_connection_string
        return adodbapi.connect(build_ado_connection_string())

    def extract_table(
        self, spec: JYDBTableSpec, start_date: str, end_date: str,
        chunksize: int = 100_000,
        stock_codes: Optional[Sequence[str]] = None,
    ) -> int:
        conn = self._connect()
        total = 0
        try:
            sql = spec.sql
            params: List[object] = [start_date, end_date] * spec.parameter_multiplier
            if stock_codes:
                placeholders = ",".join("?" for _ in stock_codes)
                predicate = f" AND s.SecuCode IN ({placeholders}) "
                group_pos = sql.upper().find("GROUP BY")
                sql = sql[:group_pos] + predicate + sql[group_pos:] if group_pos >= 0 else sql + predicate
                params.extend(str(code) for code in stock_codes)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="pandas only supports SQLAlchemy connectable.*",
                    category=UserWarning,
                )
                chunks = pd.read_sql_query(
                    sql, conn, params=tuple(params), chunksize=chunksize
                )
            for chunk in chunks:
                if spec.storage == "daily":
                    total += self.store.upsert_daily_wide_frame(
                        chunk,
                        date_col=spec.available_date_col,
                        feature_cols=spec.feature_cols,
                        dimension_cols=spec.dimension_cols,
                        prefix=spec.prefix,
                    )
                else:
                    total += self.store.upsert_wide_frame(
                        chunk,
                        source_table=spec.name,
                        available_date_col=spec.available_date_col,
                        end_date_col=spec.end_date_col,
                        feature_cols=spec.feature_cols,
                        dimension_cols=spec.dimension_cols,
                        prefix=spec.prefix,
                    )
        finally:
            conn.close()
        self.store.set_watermark(spec.name, end_date)
        return total

    def run(
        self, start_date: str, end_date: str,
        tables: Optional[Iterable[str]] = None,
        stock_codes: Optional[Sequence[str]] = None,
    ) -> Dict[str, int]:
        selected = list(tables or DEFAULT_TABLE_SPECS.keys())
        unknown = sorted(set(selected).difference(DEFAULT_TABLE_SPECS))
        if unknown:
            raise ValueError(f"未知聚源表配置: {unknown}")
        return {
            name: self.extract_table(
                DEFAULT_TABLE_SPECS[name], start_date, end_date,
                stock_codes=stock_codes,
            )
            for name in selected
        }

    def run_batched(
        self,
        start_date: str,
        end_date: str,
        tables: Optional[Iterable[str]] = None,
        stock_codes: Optional[Sequence[str]] = None,
        batch_months: int = 12,
        progress: Optional[Callable[[str, str, str, int], None]] = None,
    ) -> Dict[str, int]:
        """按日期分批抽取，每批独立提交并更新表水位。"""
        selected = list(tables or DEFAULT_TABLE_SPECS.keys())
        unknown = sorted(set(selected).difference(DEFAULT_TABLE_SPECS))
        if unknown:
            raise ValueError(f"未知聚源表配置: {unknown}")
        batches = list(iter_date_batches(start_date, end_date, batch_months))
        results: Dict[str, int] = {}
        for name in selected:
            total = 0
            for batch_start, batch_end in batches:
                count = self.extract_table(
                    DEFAULT_TABLE_SPECS[name], batch_start, batch_end,
                    stock_codes=stock_codes,
                )
                total += count
                if progress is not None:
                    progress(name, batch_start, batch_end, count)
            results[name] = total
        return results

    def run_incremental(
        self,
        end_date: str,
        tables: Optional[Iterable[str]] = None,
        fallback_start: str = "2000-01-01",
        overlap_days: int = 5,
        stock_codes: Optional[Sequence[str]] = None,
    ) -> Dict[str, int]:
        """按各表水位增量更新，并回看数日捕获源表修订。"""
        selected = list(tables or DEFAULT_TABLE_SPECS.keys())
        unknown = sorted(set(selected).difference(DEFAULT_TABLE_SPECS))
        if unknown:
            raise ValueError(f"未知聚源表配置: {unknown}")
        end_ts = pd.Timestamp(end_date)
        results = {}
        for name in selected:
            watermark = self.store.get_watermark(name)
            if watermark:
                start_ts = pd.Timestamp(watermark) - pd.Timedelta(days=max(0, overlap_days))
                start_ts = max(start_ts, pd.Timestamp(fallback_start))
            else:
                start_ts = pd.Timestamp(fallback_start)
            if start_ts > end_ts:
                results[name] = 0
                continue
            results[name] = self.extract_table(
                DEFAULT_TABLE_SPECS[name],
                start_ts.strftime("%Y-%m-%d"),
                end_ts.strftime("%Y-%m-%d"),
                stock_codes=stock_codes,
            )
        return results
