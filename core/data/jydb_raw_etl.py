"""训练所需聚源原始输入的本地落地层。"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import warnings
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Callable, Dict, Iterable, Optional, Sequence

import pandas as pd

from core.data.jydb_feature_store import (
    A_SHARE_FILTER,
    DEFAULT_TABLE_SPECS,
    JYDBETL,
    iter_date_batches,
)


@dataclass(frozen=True)
class RawQuerySpec:
    name: str
    sql: str
    date_col: Optional[str]
    parameter_multiplier: int = 1
    history_kind: str = "pit"


SECU_MAIN_SQL = f"""
    SELECT s.ID, s.InnerCode, s.CompanyCode, s.SecuCode AS code,
           s.ChiName, s.ChiNameAbbr, s.EngName, s.EngNameAbbr,
           s.SecuAbbr, s.ChiSpelling, s.SecuMarket, s.SecuCategory,
           s.ListedDate, s.ListedSector, s.ListedState, s.XGRQ, s.JSID,
           s.ISIN, s.ExtendedAbbr, s.ExtendedSpelling
    FROM SecuMain s
    WHERE s.SecuCategory = 1 {A_SHARE_FILTER}
"""


def _quote_sql(table: str) -> str:
    return f"""
        SELECT s.SecuCode AS code, q.ID, q.InnerCode, q.TradingDay,
               q.PrevClosePrice, q.OpenPrice, q.HighPrice, q.LowPrice,
               q.ClosePrice, q.TurnoverVolume, q.TurnoverValue,
               q.TurnoverDeals, q.JSID
        FROM {table} q
        JOIN SecuMain s ON q.InnerCode = s.InnerCode
        WHERE q.TradingDay >= ? AND q.TradingDay <= ?
          AND s.SecuCategory = 1 {A_SHARE_FILTER}
    """


ADJUSTING_FACTOR_RAW_SQL = f"""
    SELECT s.SecuCode AS code, a.ID, a.ExDiviDate, a.InnerCode,
           a.AdjustingFactor, a.AdjustingConst, a.RatioAdjustingFactor,
           a.XGRQ, a.JSID, a.AccuCashDivi, a.AccuBonusShareRatio
    FROM QT_AdjustingFactor a
    JOIN SecuMain s ON a.InnerCode = s.InnerCode
    WHERE a.ExDiviDate >= ? AND a.ExDiviDate <= ?
      AND s.SecuCategory = 1 {A_SHARE_FILTER}
"""


SPECIAL_TRADE_RAW_SQL = f"""
    SELECT s.SecuCode AS code, t.ID, t.InnerCode, t.InfoPublDate,
           t.SecurityAbbr, t.SpecialTradeType, t.SpecialTradeTime,
           t.SpecialTradeExplain, t.SpecialTradeReason,
           t.XGRQ, t.JSID, t.InsertTime
    FROM LC_SpecialTrade t
    JOIN SecuMain s ON t.InnerCode = s.InnerCode
    WHERE t.SpecialTradeTime >= ? AND t.SpecialTradeTime <= ?
      AND s.SecuCategory = 1 {A_SHARE_FILTER}
"""


BASE_RAW_SPECS: Dict[str, RawQuerySpec] = {
    "SecuMain": RawQuerySpec("SecuMain", SECU_MAIN_SQL, None, 0, "snapshot"),
    "QT_DailyQuote": RawQuerySpec(
        "QT_DailyQuote", _quote_sql("QT_DailyQuote"), "TradingDay", 1, "daily"
    ),
    "LC_STIBDailyQuote": RawQuerySpec(
        "LC_STIBDailyQuote", _quote_sql("LC_STIBDailyQuote"),
        "TradingDay", 1, "daily",
    ),
    "QT_AdjustingFactor": RawQuerySpec(
        "QT_AdjustingFactor", ADJUSTING_FACTOR_RAW_SQL,
        "ExDiviDate", 1, "bootstrap",
    ),
    "LC_SpecialTrade": RawQuerySpec(
        "LC_SpecialTrade", SPECIAL_TRADE_RAW_SQL,
        "SpecialTradeTime", 1, "bootstrap",
    ),
}


def training_raw_specs() -> Dict[str, RawQuerySpec]:
    """返回行情基础表与已配置训练特征源表的原始投影。"""
    result = dict(BASE_RAW_SPECS)
    for name, spec in DEFAULT_TABLE_SPECS.items():
        result[name] = RawQuerySpec(
            name=name,
            sql=spec.sql,
            date_col=spec.available_date_col,
            parameter_multiplier=spec.parameter_multiplier,
            history_kind="daily" if spec.storage == "daily" else "pit",
        )
    return result


class JYDBRawStore:
    """把源查询结果按源表名原样保存到单一 SQLite 仓库。"""

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)

    def connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=120)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def initialize(self) -> None:
        with closing(self.connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_etl_manifest (
                    source_table TEXT PRIMARY KEY,
                    history_kind TEXT NOT NULL,
                    date_column TEXT,
                    extracted_start TEXT,
                    extracted_end TEXT,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    sql_sha256 TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _quote(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    @staticmethod
    def normalize(frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        if "code" in data.columns:
            data["code"] = data["code"].astype(str).str.extract(
                r"(\d{6})", expand=False
            )
        for column in data.columns:
            lower = column.lower()
            if (
                lower.endswith("date") or lower.endswith("day")
                or lower.endswith("time") or lower in {"available_date", "end_date", "xgrq"}
            ):
                converted = pd.to_datetime(data[column], errors="coerce")
                if converted.notna().any():
                    data[column] = converted.dt.strftime("%Y-%m-%d %H:%M:%S")
            elif data[column].dtype == object:
                data[column] = data[column].map(
                    lambda value: float(value) if isinstance(value, Decimal) else value
                )
        return data.where(pd.notna(data), None)

    def replace_batch(
        self,
        spec: RawQuerySpec,
        frames: Iterable[pd.DataFrame],
        batch_start: Optional[str],
        batch_end: Optional[str],
    ) -> int:
        self.initialize()
        table = self._quote(spec.name)
        inserted = 0
        with closing(self.connect()) as conn, conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (spec.name,),
            ).fetchone()
            if exists:
                if spec.date_col and batch_start and batch_end:
                    date_col = self._quote(spec.date_col)
                    conn.execute(
                        f"DELETE FROM {table} WHERE substr({date_col},1,10) BETWEEN ? AND ?",
                        (batch_start, batch_end),
                    )
                elif spec.date_col is None:
                    conn.execute(f"DELETE FROM {table}")
            for frame in frames:
                data = self.normalize(frame)
                if data.empty:
                    continue
                data.to_sql(spec.name, conn, if_exists="append", index=False)
                inserted += len(data)
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] if (
                inserted or exists
            ) else 0
            sql_hash = hashlib.sha256(spec.sql.encode("utf-8")).hexdigest()
            conn.execute(
                """
                INSERT INTO raw_etl_manifest (
                    source_table, history_kind, date_column,
                    extracted_start, extracted_end, row_count,
                    sql_sha256, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_table) DO UPDATE SET
                    history_kind=excluded.history_kind,
                    date_column=excluded.date_column,
                    extracted_start=CASE
                        WHEN raw_etl_manifest.extracted_start IS NULL THEN excluded.extracted_start
                        WHEN excluded.extracted_start IS NULL THEN raw_etl_manifest.extracted_start
                        ELSE MIN(raw_etl_manifest.extracted_start, excluded.extracted_start)
                    END,
                    extracted_end=CASE
                        WHEN raw_etl_manifest.extracted_end IS NULL THEN excluded.extracted_end
                        WHEN excluded.extracted_end IS NULL THEN raw_etl_manifest.extracted_end
                        ELSE MAX(raw_etl_manifest.extracted_end, excluded.extracted_end)
                    END,
                    row_count=excluded.row_count,
                    sql_sha256=excluded.sql_sha256,
                    updated_at=excluded.updated_at
                """,
                (
                    spec.name, spec.history_kind, spec.date_col,
                    batch_start, batch_end, row_count, sql_hash,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
        return inserted


class JYDBRawETL:
    """抽取训练依赖的聚源原始投影，不生成派生特征或行情表。"""

    def __init__(self, store: JYDBRawStore, connection_factory=None):
        self.store = store
        self.source = JYDBETL(store=None, connection_factory=connection_factory)

    def extract_batch(
        self,
        spec: RawQuerySpec,
        batch_start: Optional[str],
        batch_end: Optional[str],
        chunksize: int = 50_000,
    ) -> int:
        conn = self.source._connect()
        try:
            sql = spec.sql
            params = ()
            if spec.parameter_multiplier:
                start_literal = pd.Timestamp(batch_start).strftime("%Y-%m-%d")
                end_literal = pd.Timestamp(batch_end).strftime("%Y-%m-%d")
                for value in [start_literal, end_literal] * spec.parameter_multiplier:
                    sql = sql.replace("?", f"CAST('{value}' AS date)", 1)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="pandas only supports SQLAlchemy connectable.*",
                    category=UserWarning,
                )
                frames = pd.read_sql_query(
                    sql, conn, params=params, chunksize=chunksize
                )
                return self.store.replace_batch(
                    spec, frames, batch_start, batch_end
                )
        finally:
            conn.close()

    def run(
        self,
        start_date: str,
        end_date: str,
        bootstrap_start: str = "1990-01-01",
        tables: Optional[Sequence[str]] = None,
        batch_months: int = 3,
        progress: Optional[Callable[[str, str, str, int], None]] = None,
    ) -> Dict[str, int]:
        specs = training_raw_specs()
        selected = list(tables or specs.keys())
        unknown = sorted(set(selected).difference(specs))
        if unknown:
            raise ValueError(f"未知聚源原始表: {unknown}")
        results: Dict[str, int] = {}
        for name in selected:
            spec = specs[name]
            if spec.history_kind == "snapshot":
                count = self.extract_batch(spec, None, None)
                results[name] = count
                if progress:
                    progress(name, "snapshot", "snapshot", count)
                continue
            table_start = (
                bootstrap_start
                if spec.history_kind in {"pit", "bootstrap"}
                else start_date
            )
            total = 0
            for batch_start, batch_end in iter_date_batches(
                table_start, end_date, batch_months
            ):
                count = self.extract_batch(spec, batch_start, batch_end)
                total += count
                if progress:
                    progress(name, batch_start, batch_end, count)
            results[name] = total
        return results


__all__ = [
    "BASE_RAW_SPECS", "JYDBRawETL", "JYDBRawStore", "RawQuerySpec",
    "training_raw_specs",
]
