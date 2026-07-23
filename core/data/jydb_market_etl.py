"""将聚源行情转换为项目既有的 stock_daily.db schema。"""
from __future__ import annotations

import os
import sqlite3
import warnings
from contextlib import closing
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from core.data.jydb_feature_store import (
    A_SHARE_FILTER, JYDBETL, iter_date_batches,
    _build_fast, _apply_build_pragmas,
)


DAILY_QUOTE_SQL = f"""
    SELECT * FROM (
        SELECT s.SecuCode AS code, q.TradingDay AS date,
               q.OpenPrice AS [open], q.HighPrice AS high, q.LowPrice AS low,
               q.ClosePrice AS [close], q.PrevClosePrice AS preclose,
               q.TurnoverVolume AS volume, q.TurnoverValue AS amount,
               q.TurnoverDeals AS turnover_deals,
               CAST(1 AS int) AS tradestatus,
               CASE WHEN q.PrevClosePrice IS NULL OR q.PrevClosePrice = 0 THEN NULL
                    ELSE (q.ClosePrice / q.PrevClosePrice - 1) * 100 END AS pctChg,
               v.PE AS peTTM, v.PB AS pbMRQ, v.PSTTM AS psTTM,
               v.PCFTTM AS pcfNcfTTM
        FROM QT_DailyQuote q
        JOIN SecuMain s ON q.InnerCode = s.InnerCode
        LEFT JOIN LC_DIndicesForValuation v
          ON q.InnerCode = v.InnerCode AND q.TradingDay = v.TradingDay
        WHERE q.TradingDay >= ? AND q.TradingDay <= ?
          AND s.SecuCategory = 1
          {A_SHARE_FILTER}
        UNION ALL
        SELECT s.SecuCode AS code, q.TradingDay AS date,
               q.OpenPrice AS [open], q.HighPrice AS high, q.LowPrice AS low,
               q.ClosePrice AS [close], q.PrevClosePrice AS preclose,
               q.TurnoverVolume AS volume, q.TurnoverValue AS amount,
               q.TurnoverDeals AS turnover_deals,
               CAST(1 AS int) AS tradestatus,
               CASE WHEN q.PrevClosePrice IS NULL OR q.PrevClosePrice = 0 THEN NULL
                    ELSE (q.ClosePrice / q.PrevClosePrice - 1) * 100 END AS pctChg,
               v.PE AS peTTM, v.PB AS pbMRQ, v.PSTTM AS psTTM,
               v.PCFTTM AS pcfNcfTTM
        FROM LC_STIBDailyQuote q
        JOIN SecuMain s ON q.InnerCode = s.InnerCode
        LEFT JOIN LC_DIndicesForValuation v
          ON q.InnerCode = v.InnerCode AND q.TradingDay = v.TradingDay
        WHERE q.TradingDay >= ? AND q.TradingDay <= ?
          AND s.SecuCategory = 1
          {A_SHARE_FILTER}
    ) d
    WHERE 1=1
"""

ADJUST_FACTOR_SQL = f"""
    SELECT s.SecuCode AS code, a.ExDiviDate AS effective_date,
           a.RatioAdjustingFactor AS factor
    FROM QT_AdjustingFactor a
    JOIN SecuMain s ON a.InnerCode = s.InnerCode
    WHERE a.ExDiviDate <= ?
      AND s.SecuCategory = 1
      {A_SHARE_FILTER}
    ORDER BY s.SecuCode, a.ExDiviDate
"""

SPECIAL_TRADE_SQL = f"""
    SELECT s.SecuCode AS code, t.SpecialTradeTime AS effective_date,
           t.SecurityAbbr, t.SpecialTradeType
    FROM LC_SpecialTrade t
    JOIN SecuMain s ON t.InnerCode = s.InnerCode
    WHERE t.SpecialTradeTime <= ?
      AND s.SecuCategory = 1
      {A_SHARE_FILTER}
    ORDER BY s.SecuCode, t.SpecialTradeTime, t.InfoPublDate
"""


class JYDBMarketETL:
    """聚源行情 ETL；输出可直接被 DataHandler/MLModelTrainer 使用。"""

    def __init__(self, db_path: str, connection_factory=None):
        self.db_path = os.path.abspath(db_path)
        self.source = JYDBETL(store=None, connection_factory=connection_factory)

    def initialize(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with closing(sqlite3.connect(self.db_path, timeout=60)) as conn, conn:
            _apply_build_pragmas(conn)
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS daily_data (
                    code TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, preclose REAL,
                    volume REAL, amount REAL, turnover_deals REAL,
                    turnover_rate REAL, tradestatus INTEGER NOT NULL DEFAULT 1,
                    pctChg REAL, is_st INTEGER NOT NULL DEFAULT 0,
                    peTTM REAL, pbMRQ REAL, psTTM REAL, pcfNcfTTM REAL,
                    PRIMARY KEY(code, date)
                );
                CREATE INDEX IF NOT EXISTS idx_daily_data_date ON daily_data(date);
                CREATE TABLE IF NOT EXISTS adjust_factor (
                    code TEXT NOT NULL,
                    date TEXT NOT NULL,
                    fore_adjust_factor REAL,
                    back_adjust_factor REAL,
                    PRIMARY KEY(code, date)
                );
                """
            )
            existing = {row[1] for row in conn.execute("PRAGMA table_info(daily_data)")}
            # 完整期望列；既有表（可能由 baostock 建立）缺少任何列都补齐，
            # 避免 INSERT 时报 'no column named ...'。
            desired_columns = {
                "code": "TEXT NOT NULL",
                "date": "TEXT NOT NULL",
                "open": "REAL", "high": "REAL", "low": "REAL", "close": "REAL",
                "preclose": "REAL", "volume": "REAL", "amount": "REAL",
                "turnover_deals": "REAL", "turnover_rate": "REAL",
                "tradestatus": "INTEGER NOT NULL DEFAULT 1",
                "pctChg": "REAL", "is_st": "INTEGER NOT NULL DEFAULT 0",
                "peTTM": "REAL", "pbMRQ": "REAL", "psTTM": "REAL", "pcfNcfTTM": "REAL",
            }
            for column, definition in desired_columns.items():
                if column not in existing:
                    conn.execute(f'ALTER TABLE daily_data ADD COLUMN "{column}" {definition}')

    @staticmethod
    def _read_chunks(conn, sql, params, chunksize):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="pandas only supports SQLAlchemy connectable.*",
                category=UserWarning,
            )
            return pd.read_sql_query(sql, conn, params=params, chunksize=chunksize)

    # 幂等 upsert 语句：供 SQL Server 直连与本地源两条路径共用。
    _DAILY_UPSERT_SQL = """
        INSERT INTO daily_data (
            code,date,open,high,low,close,preclose,volume,amount,
            turnover_deals,turnover_rate,tradestatus,pctChg,is_st,
            peTTM,pbMRQ,psTTM,pcfNcfTTM
        ) VALUES (?,?,?,?,?,?,?,?,?,?,NULL,?,?,0,?,?,?,?)
        ON CONFLICT(code,date) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, preclose=excluded.preclose,
            volume=excluded.volume, amount=excluded.amount,
            turnover_deals=excluded.turnover_deals,
            tradestatus=excluded.tradestatus, pctChg=excluded.pctChg,
            peTTM=excluded.peTTM, pbMRQ=excluded.pbMRQ,
            psTTM=excluded.psTTM, pcfNcfTTM=excluded.pcfNcfTTM
    """

    @staticmethod
    def _daily_numeric_cols() -> Sequence[str]:
        return (
            "open", "high", "low", "close", "preclose", "volume",
            "amount", "turnover_deals", "tradestatus", "pctChg",
            "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
        )

    def write_daily_chunks(self, chunks) -> int:
        """把行情结果分块写入 daily_data（幂等 upsert），返回写入行数。

        与数据来源无关：``chunks`` 可来自 SQL Server 直连游标，
        也可来自本地 raw 库读取——只要每块含标准列即可。
        """
        self.initialize()
        total = 0
        numeric = self._daily_numeric_cols()
        with closing(sqlite3.connect(self.db_path, timeout=60)) as target, target:
            _apply_build_pragmas(target)
            for chunk in chunks:
                chunk = chunk.copy()
                chunk["code"] = chunk["code"].astype(str).str.extract(r"(\d{6})", expand=False)
                chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                chunk = chunk.dropna(subset=["code", "date"])
                for col in numeric:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
                rows = list(chunk[["code", "date", *numeric]].itertuples(index=False, name=None))
                target.executemany(self._DAILY_UPSERT_SQL, rows)
                total += len(rows)
        return total

    def extract_daily_quotes(
        self, start_date: str, end_date: str, chunksize: int = 100_000,
        stock_codes: Optional[Sequence[str]] = None,
    ) -> int:
        source_conn = self.source._connect()
        try:
            sql = DAILY_QUOTE_SQL
            params = [start_date, end_date, start_date, end_date]
            if stock_codes:
                placeholders = ",".join("?" for _ in stock_codes)
                sql += f" AND d.code IN ({placeholders})"
                params.extend(str(code) for code in stock_codes)
            chunks = self._read_chunks(source_conn, sql, tuple(params), chunksize)
            return self.write_daily_chunks(chunks)
        finally:
            source_conn.close()

    def rebuild_adjust_factors(
        self, start_date: str, end_date: str,
        stock_codes: Optional[Sequence[str]] = None,
        events: Optional[pd.DataFrame] = None,
    ) -> int:
        """把除权事件累计因子扩展为逐交易日因子。

        ``events`` 为空时从 SQL Server 拉取除权事件；本地源管线可直接注入
        含 (code, effective_date, factor) 的事件表，避免二次连接远端。
        """
        self.initialize()
        if events is None:
            source_conn = self.source._connect()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="pandas only supports SQLAlchemy connectable.*",
                        category=UserWarning,
                    )
                    source_sql = ADJUST_FACTOR_SQL
                    source_params = [end_date]
                    if stock_codes:
                        placeholders = ",".join("?" for _ in stock_codes)
                        order_pos = source_sql.upper().find("ORDER BY")
                        predicate = f" AND s.SecuCode IN ({placeholders}) "
                        source_sql = source_sql[:order_pos] + predicate + source_sql[order_pos:]
                        source_params.extend(str(code) for code in stock_codes)
                    events = pd.read_sql_query(
                        source_sql, source_conn, params=tuple(source_params)
                    )
            finally:
                source_conn.close()
        else:
            events = events.copy()
        if not events.empty:
            events["effective_date"] = pd.to_datetime(events["effective_date"], errors="coerce")
            events["factor"] = pd.to_numeric(events["factor"], errors="coerce")
            events = events.dropna(subset=["code", "effective_date", "factor"])

        with closing(sqlite3.connect(self.db_path, timeout=60)) as conn, conn:
            _apply_build_pragmas(conn)
            daily_sql = "SELECT code,date FROM daily_data WHERE date>=? AND date<=?"
            daily_params = [start_date, end_date]
            if stock_codes:
                placeholders = ",".join("?" for _ in stock_codes)
                daily_sql += f" AND code IN ({placeholders})"
                daily_params.extend(str(code) for code in stock_codes)
            daily_sql += " ORDER BY code,date"
            daily = pd.read_sql_query(daily_sql, conn, params=tuple(daily_params))
            if daily.empty:
                return 0
            daily["date_dt"] = pd.to_datetime(daily["date"])
            outputs = []
            event_groups = {
                str(code): group.sort_values("effective_date")
                for code, group in events.groupby("code", sort=False)
            } if not events.empty else {}
            for code, group in daily.groupby("code", sort=False):
                group = group.sort_values("date_dt")
                code_events = event_groups.get(str(code))
                if code_events is None or code_events.empty:
                    factors = np.ones(len(group), dtype=float)
                else:
                    merged = pd.merge_asof(
                        group[["date_dt"]],
                        code_events[["effective_date", "factor"]],
                        left_on="date_dt", right_on="effective_date",
                        direction="backward",
                    )
                    factors = merged["factor"].fillna(1.0).to_numpy(dtype=float)
                part = pd.DataFrame({
                    "code": str(code), "date": group["date"].values,
                    "fore_adjust_factor": factors,
                    "back_adjust_factor": factors,
                })
                outputs.append(part)
            result = pd.concat(outputs, ignore_index=True)
            conn.executemany(
                """
                INSERT INTO adjust_factor(code,date,fore_adjust_factor,back_adjust_factor)
                VALUES (?,?,?,?)
                ON CONFLICT(code,date) DO UPDATE SET
                    fore_adjust_factor=excluded.fore_adjust_factor,
                    back_adjust_factor=excluded.back_adjust_factor
                """,
                result.itertuples(index=False, name=None),
            )
        return len(result)

    def rebuild_st_status(
        self, start_date: str, end_date: str,
        stock_codes: Optional[Sequence[str]] = None,
        events: Optional[pd.DataFrame] = None,
    ) -> int:
        """根据历史特别处理实施日生成逐日 ST 状态。

        ``events`` 为空时从 SQL Server 拉取特别处理事件；本地源管线可直接注入
        含 (code, effective_date, SecurityAbbr) 的事件表。
        """
        if events is None:
            source_conn = self.source._connect()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="pandas only supports SQLAlchemy connectable.*",
                        category=UserWarning,
                    )
                    source_sql = SPECIAL_TRADE_SQL
                    source_params = [end_date]
                    if stock_codes:
                        placeholders = ",".join("?" for _ in stock_codes)
                        order_pos = source_sql.upper().find("ORDER BY")
                        predicate = f" AND s.SecuCode IN ({placeholders}) "
                        source_sql = source_sql[:order_pos] + predicate + source_sql[order_pos:]
                        source_params.extend(str(code) for code in stock_codes)
                    events = pd.read_sql_query(
                        source_sql, source_conn, params=tuple(source_params)
                    )
            finally:
                source_conn.close()
        else:
            events = events.copy()
        if not events.empty:
            events["effective_date"] = pd.to_datetime(events["effective_date"], errors="coerce")
            events["is_st"] = events["SecurityAbbr"].fillna("").astype(str).str.upper().str.contains("ST").astype(int)
            events = events.dropna(subset=["code", "effective_date"])

        st_rows = []
        with closing(sqlite3.connect(self.db_path, timeout=60)) as conn, conn:
            _apply_build_pragmas(conn)
            daily_sql = "SELECT code,date FROM daily_data WHERE date>=? AND date<=?"
            daily_params = [start_date, end_date]
            if stock_codes:
                placeholders = ",".join("?" for _ in stock_codes)
                daily_sql += f" AND code IN ({placeholders})"
                daily_params.extend(str(code) for code in stock_codes)
            daily_sql += " ORDER BY code,date"
            daily = pd.read_sql_query(daily_sql, conn, params=tuple(daily_params))
            daily["date_dt"] = pd.to_datetime(daily["date"])
            event_groups = {
                str(code): group.sort_values("effective_date")
                for code, group in events.groupby("code", sort=False)
            } if not events.empty else {}
            for code, group in daily.groupby("code", sort=False):
                code_events = event_groups.get(str(code))
                if code_events is None or code_events.empty:
                    states = np.zeros(len(group), dtype=int)
                else:
                    merged = pd.merge_asof(
                        group.sort_values("date_dt")[["date", "date_dt"]],
                        code_events[["effective_date", "is_st"]],
                        left_on="date_dt", right_on="effective_date",
                        direction="backward",
                    )
                    states = merged["is_st"].fillna(0).astype(int).to_numpy()
                    group = group.sort_values("date_dt")
                st_rows.extend(
                    (str(code), date, int(state))
                    for state, date in zip(states, group["date"].values)
                )
            # 集合式更新：写入临时表后单条 UPDATE...FROM 连接，避免逐行 UPDATE
            conn.execute("DROP TABLE IF EXISTS temp_st_update")
            conn.execute(
                "CREATE TEMP TABLE temp_st_update "
                "(code TEXT, date TEXT, is_st INTEGER, PRIMARY KEY(code,date))"
            )
            conn.executemany(
                "INSERT INTO temp_st_update(code,date,is_st) VALUES (?,?,?)", st_rows
            )
            conn.execute(
                "UPDATE daily_data SET is_st = t.is_st "
                "FROM temp_st_update t "
                "WHERE daily_data.code = t.code AND daily_data.date = t.date"
            )
            conn.execute("DROP TABLE IF EXISTS temp_st_update")
        return len(st_rows)

    @staticmethod
    def get_stock_list(
        db_path: str,
        as_of_date: Optional[str] = None,
        limit: Optional[int] = None,
        market_prefixes: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """从 daily_data 取 DISTINCT code 作为股票列表（纯聚源，无 Baostock 依赖）。

        - ``as_of_date``：只保留在该日期（含）之前有过交易的股票，剔除长期停牌/未上市代码。
        - ``market_prefixes``：可选的市场前缀过滤（如 ["60", "00", "30"]）。
        - 返回按 code 升序排列的列表。
        """
        db_path = os.path.abspath(db_path)
        if not os.path.exists(db_path):
            return []
        query = "SELECT DISTINCT code FROM daily_data"
        params: list = []
        clauses = []
        if as_of_date:
            clauses.append("date <= ?")
            params.append(as_of_date)
        if market_prefixes:
            like_parts = " OR ".join("code LIKE ?" for _ in market_prefixes)
            clauses.append(f"({like_parts})")
            params.extend(f"{p}%" for p in market_prefixes)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY code"
        if limit:
            query += " LIMIT ?"
            params.append(int(limit))
        with closing(sqlite3.connect(db_path, timeout=30)) as conn:
            rows = conn.execute(query, params).fetchall()
        return [r[0] for r in rows]

    def run(
        self, start_date: str, end_date: str,
        stock_codes: Optional[Sequence[str]] = None,
    ) -> dict:
        quotes = self.extract_daily_quotes(
            start_date, end_date, stock_codes=stock_codes
        )
        factors = self.rebuild_adjust_factors(start_date, end_date, stock_codes=stock_codes)
        st_rows = self.rebuild_st_status(start_date, end_date, stock_codes=stock_codes)
        return {"daily_data": quotes, "adjust_factor": factors, "st_status": st_rows}

    def run_batched(
        self,
        start_date: str,
        end_date: str,
        stock_codes: Optional[Sequence[str]] = None,
        batch_months: int = 12,
        progress: Optional[Callable[[str, str, str, int], None]] = None,
    ) -> dict:
        """按日期分批提交行情、复权和 ST 状态，避免首次全量长事务不可见。"""
        totals = {"daily_data": 0, "adjust_factor": 0, "st_status": 0}
        for batch_start, batch_end in iter_date_batches(
            start_date, end_date, batch_months
        ):
            steps = (
                ("daily_data", self.extract_daily_quotes),
                ("adjust_factor", self.rebuild_adjust_factors),
                ("st_status", self.rebuild_st_status),
            )
            for name, operation in steps:
                count = operation(
                    batch_start, batch_end, stock_codes=stock_codes
                )
                totals[name] += count
                if progress is not None:
                    progress(name, batch_start, batch_end, count)
        return totals
