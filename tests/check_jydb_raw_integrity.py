"""jydb_raw.db 最近四年数据完整性检查（批量优化版）。"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.jydb_config import JYDB_RAW_DB_PATH

END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=4 * 365 + 1)).strftime("%Y-%m-%d")

DAILY_TABLES = [
    "QT_DailyQuote",
    "LC_STIBDailyQuote",
    "MT_TradingDetail",
    "QT_TradingCapitalFlow",
    "LC_DIndicesForValuation",
    "LC_SHSZHSCHoldings",
]

PIT_TABLES = [
    "LC_MainIndexNew",
    "LC_FreeFloat",
    "LC_SHNumber",
    "LC_ShareFPSta",
    "LC_Dividend",
    "LC_Buyback",
    "LC_PerformanceForecast",
    "LC_PerformanceLetters",
    "LC_AuditOpinion",
    "LC_ShareStru",
    "LC_SharesFloatingSchedule",
    "LC_ActualController",
    "LC_AShareSeasonedNewIssue",
    "LC_ASharePlacement",
    "LC_IndexComponentsWeight",
    "LC_CSIIndustry",
    "LC_ReserveReportDate",
    "LC_SpecialTrade",
    "QT_AdjustingFactor",
]

SNAPSHOT_TABLES = ["SecuMain"]


def fmt(s):
    return str(s or "-")


def get_conn():
    conn = sqlite3.connect(JYDB_RAW_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA cache_size=-800000")
    return conn


def build_trading_calendar(conn):
    t0 = time.time()
    rows = conn.execute(
        "SELECT DISTINCT substr(TradingDay,1,10) FROM QT_DailyQuote "
        "WHERE substr(TradingDay,1,10) >= ? AND substr(TradingDay,1,10) <= ? ORDER BY 1",
        (START_DATE, END_DATE),
    ).fetchall()
    cal = [r[0] for r in rows]
    print("  [%.1fs] Loaded %d trading days" % (time.time() - t0, len(cal)))
    return cal


def get_all_stock_codes(conn):
    """获取 A 股股票清单（与 pull_jydb_parallel 的 A_SHARE_FILTER 一致）。"""
    rows = conn.execute(
        "SELECT code, ChiName, SecuMarket FROM SecuMain "
        "WHERE SecuMarket IN (18, 83, 90) ORDER BY code"
    ).fetchall()
    # SQL 中的 LIKE 条件用 Python 过滤（避免了参数化多个 LIKE 的麻烦）
    allowed_prefixes = ("00", "30", "60", "68", "43", "83", "87", "92")
    return [r for r in rows if r[0].startswith(allowed_prefixes)]


def check_daily_table(conn, table, date_col, trading_calendar, all_codes):
    t0 = time.time()
    total_expected = len(trading_calendar)
    expected_codes = {r[0] for r in all_codes}

    dc = date_col

    sql = (
        "SELECT code, COUNT(*) AS cnt, "
        "MIN(substr({dc},1,10)), MAX(substr({dc},1,10)) "
        "FROM \"{tbl}\" "
        "WHERE substr({dc},1,10) >= ? AND substr({dc},1,10) <= ? "
        "GROUP BY code ORDER BY code"
    ).format(dc=dc, tbl=table)

    cur = conn.execute(sql, (START_DATE, END_DATE))
    code_stats = {}
    for r in cur.fetchall():
        code_stats[r[0]] = {"rows": r[1], "first": r[2], "last": r[3]}

    covered = set(code_stats.keys())
    missing_codes = expected_codes - covered

    severe_gap_stocks = []
    threshold_80 = int(total_expected * 0.8)
    for code in sorted(covered):
        cs = code_stats[code]
        if cs["rows"] < threshold_80:
            severe_gap_stocks.append((code, cs["rows"], total_expected))

    print("  [%.1fs] Stats collected, %d stocks, %d missing, %d severe gaps" %
          (time.time() - t0, len(covered), len(missing_codes), len(severe_gap_stocks)))
    return {
        "table": table,
        "total_rows": sum(s["rows"] for s in code_stats.values()),
        "stocks_covered": len(covered),
        "stocks_expected": len(expected_codes),
        "missing_count": len(missing_codes),
        "missing_rate_pct": round(len(missing_codes) / len(expected_codes) * 100, 2) if expected_codes else 0,
        "date_range": (min((s["first"] for s in code_stats.values()), default="-"),
                       max((s["last"] for s in code_stats.values()), default="-")),
        "severe_gap_count": len(severe_gap_stocks),
        "severe_gap_top": severe_gap_stocks[:10],
        "missing_stocks": sorted(missing_codes)[:10],
    }


def check_pit_table(conn, table, date_col, expected_codes):
    t0 = time.time()
    dc = date_col
    sql = (
        "SELECT code, COUNT(*), MIN(substr({dc},1,10)), MAX(substr({dc},1,10)) "
        "FROM \"{tbl}\" "
        "WHERE substr({dc},1,10) >= ? AND substr({dc},1,10) <= ? "
        "GROUP BY code ORDER BY code"
    ).format(dc=dc, tbl=table)
    cur = conn.execute(sql, (START_DATE, END_DATE))
    code_stats = {}
    for r in cur.fetchall():
        code_stats[r[0]] = {"rows": r[1], "first": r[2], "last": r[3]}
    covered = len(code_stats)
    total_rows = sum(s["rows"] for s in code_stats.values())
    missing = sorted(expected_codes - set(code_stats.keys()))
    print("  [%.1fs] %d stocks, %d rows" % (time.time() - t0, covered, total_rows))
    return {
        "table": table,
        "total_rows": total_rows,
        "stocks_covered": covered,
        "stocks_expected": len(expected_codes),
        "missing_count": len(missing),
        "date_range": (min((s["first"] for s in code_stats.values()), default="-"),
                       max((s["last"] for s in code_stats.values()), default="-")),
        "avg_rows_per_stock": round(total_rows / covered, 1) if covered else 0,
    }


def check_daily_value_sanity(conn):
    t0 = time.time()
    bad_rows = conn.execute(
        "SELECT code, TradingDay, OpenPrice, HighPrice, LowPrice, ClosePrice, "
        "TurnoverVolume, TurnoverValue "
        "FROM QT_DailyQuote "
        "WHERE substr(TradingDay,1,10) >= ? AND substr(TradingDay,1,10) <= ? "
        "AND (ClosePrice <= 0 OR OpenPrice <= 0 OR HighPrice <= 0 OR LowPrice <= 0 "
        "  OR HighPrice < LowPrice OR TurnoverVolume < 0 OR TurnoverValue < 0)",
        (START_DATE, END_DATE),
    ).fetchall()
    anomaly_types = defaultdict(int)
    for r in bad_rows:
        if r[5] is not None and r[5] <= 0:
            anomaly_types["ClosePrice<=0"] += 1
        if r[3] is not None and r[4] is not None and r[3] < r[4]:
            anomaly_types["High<Low"] += 1
        if r[6] is not None and r[6] < 0:
            anomaly_types["Volume<0"] += 1
        if r[7] is not None and r[7] < 0:
            anomaly_types["Value<0"] += 1
        if r[2] is not None and r[2] <= 0:
            anomaly_types["OpenPrice<=0"] += 1
    print("  [%.1fs] Checked value sanity" % (time.time() - t0))
    return {
        "total_anomalies": len(bad_rows),
        "anomaly_types": dict(anomaly_types),
        "samples": [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in bad_rows[:10]],
    }


def check_batch_completeness(conn):
    rows = conn.execute(
        "SELECT source_table, COUNT(*) as batches, "
        "MIN(batch_start), MAX(batch_end), "
        "MIN(completed_at), MAX(completed_at) "
        "FROM pull_checkpoint "
        "WHERE batch_end >= ? "
        "GROUP BY source_table ORDER BY source_table",
        (START_DATE,),
    ).fetchall()
    return {
        r[0]: {
            "batches": r[1],
            "batch_range": (r[2], r[3]),
            "completed_range": (r[4], r[5]),
        } for r in rows
    }


def _c(n):
    if n is None:
        return "-"
    return "{:,}".format(n)

def print_header(title):
    print()
    print("=" * 88)
    print(" " + title)
    print("=" * 88)


def print_subheader(title):
    print()
    print("--- " + title + " ---")


def check_integrity():
    overall_t0 = time.time()
    conn = get_conn()
    db_size = os.path.getsize(JYDB_RAW_DB_PATH) / 1024 / 1024

    print_header("jydb_raw.db Data Integrity Check Report (%s ~ %s)" % (START_DATE, END_DATE))
    print("  Database: %s" % JYDB_RAW_DB_PATH)
    print("  Size:     %.1f MB" % db_size)
    total_tables = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
    ).fetchone()[0]
    print("  Tables:   %d" % total_tables)

    print_subheader("raw_etl_manifest Summary")
    rows = conn.execute(
        "SELECT source_table, history_kind, extracted_start, extracted_end, row_count "
        "FROM raw_etl_manifest ORDER BY source_table"
    ).fetchall()
    print("  %-30s %-12s %-14s %-14s %10s" % ("Table", "Type", "Start", "End", "Rows"))
    print("  " + "-" * 88)
    total_all_rows = 0
    for r in rows:
        print("  %-30s %-12s %-14s %-14s %10s" % (r[0], r[1], fmt(r[2]), fmt(r[3]), _c(r[4])))
        total_all_rows += r[4]
    print("  " + "-" * 88)
    print("  %-30s %-12s %-14s %-14s %10s" % ("TOTAL", "", "", "", _c(total_all_rows)))

    print_subheader("Trading Calendar & Stock Scope")
    trading_calendar = build_trading_calendar(conn)
    all_codes = get_all_stock_codes(conn)
    print("  A-shares in SecuMain: %d" % len(all_codes))
    market_counts = defaultdict(int)
    for r in all_codes:
        market_counts[r[2]] += 1
    for mk, cnt in sorted(market_counts.items()):
        print("    Market %s: %d" % (mk, cnt))

    print_header("1. Daily Tables - Continuity Check")
    daily_results = []
    for table in DAILY_TABLES:
        info = conn.execute(
            "SELECT date_column, history_kind FROM raw_etl_manifest WHERE source_table=?",
            (table,),
        ).fetchone()
        if not info:
            print("\n  [SKIP] %s: not in manifest" % table)
            continue
        date_col = info[0]
        if table == "QT_DailyQuote":
            date_col = "TradingDay"
        print("\n  >> %s" % table)
        r = check_daily_table(conn, table, date_col, trading_calendar, all_codes)
        daily_results.append(r)
        print("    Rows (4yr):   %s" % _c(r["total_rows"]))
        print("    Stocks:       %s / %s (missing %s, %.1f%%)" %
              (_c(r["stocks_covered"]), _c(r["stocks_expected"]),
               _c(r["missing_count"]), r["missing_rate_pct"]))
        print("    Date range:   %s ~ %s" % r["date_range"])
        print("    Severe gaps   (<80%% days): %s stocks" % _c(r["severe_gap_count"]))
        if r["missing_stocks"]:
            print("    Missing sample: %s" % ", ".join(r["missing_stocks"][:10]))
        if r["severe_gap_top"]:
            print("    Worst gaps (stock, actual, expected):")
            for item in r["severe_gap_top"]:
                print("      %s: %s / %s" % (item[0], _c(item[1]), _c(item[2])))

    print_header("2. PIT / Bootstrap Tables - Coverage Check")
    pit_results = []
    for table in PIT_TABLES:
        info = conn.execute(
            "SELECT date_column, history_kind FROM raw_etl_manifest WHERE source_table=?",
            (table,),
        ).fetchone()
        if not info:
            continue
        date_col = info[0]
        if not date_col:
            continue
        print("\n  >> %s" % table)
        expected_codes_set = {s[0] for s in all_codes}
        r = check_pit_table(conn, table, date_col, expected_codes_set)
        pit_results.append(r)
        print("    Rows (4yr):   %s" % _c(r["total_rows"]))
        print("    Stocks:       %s / %s (missing %s)" %
              (_c(r["stocks_covered"]), _c(r["stocks_expected"]), _c(r["missing_count"])))
        print("    Date range:   %s ~ %s" % r["date_range"])
        print("    Avg/stock:    %.1f" % r["avg_rows_per_stock"])

    print_header("3. Snapshot Tables")
    for table in SNAPSHOT_TABLES:
        cnt = conn.execute("SELECT COUNT(*) FROM \"%s\"" % table).fetchone()[0]
        print("  >> %s: %s rows" % (table, _c(cnt)))

    print_header("4. Value Sanity Check (QT_DailyQuote)")
    sanity = check_daily_value_sanity(conn)
    print("  Total anomalies: %s" % _c(sanity["total_anomalies"]))
    if sanity["anomaly_types"]:
        for atype, acnt in sorted(sanity["anomaly_types"].items()):
            print("    %s: %s" % (atype, _c(acnt)))
    if sanity["samples"]:
        print("  Samples:")
        for s in sanity["samples"][:5]:
            print("    %s %s: O=%s H=%s L=%s C=%s" % s)

    print_header("5. Batch Completeness (pull_checkpoint)")
    batch_info = check_batch_completeness(conn)
    if batch_info:
        print("  %-30s %6s  %-14s %-14s  %-16s" %
              ("Table", "Batches", "From", "To", "Completed"))
        print("  " + "-" * 88)
        for table, bi in sorted(batch_info.items()):
            print("  %-30s %6d  %-14s %-14s  %s" %
                  (table, bi["batches"],
                   bi["batch_range"][0], bi["batch_range"][1],
                   bi["completed_range"][0][:16]))

    print_header("6. Overall Assessment")
    total_issues = 0
    for dr in daily_results:
        total_issues += dr["severe_gap_count"]
        total_issues += dr["missing_count"]
    total_issues += sanity["total_anomalies"]
    print("  Total issues found: %d" % total_issues)
    if daily_results:
        avg_cover = sum(
            dr["stocks_covered"] / dr["stocks_expected"] * 100
            for dr in daily_results if dr["stocks_expected"]
        ) / len(daily_results)
        print("  Avg daily stock coverage: %.1f%%" % avg_cover)
    if sanity["total_anomalies"] == 0:
        print("  Value sanity: PASS")
    else:
        print("  Value sanity: %s anomalies" % _c(sanity["total_anomalies"]))

    print("  (Note: 'Missing stocks' includes delisted/suspended stocks without recent trading data)")
    if total_issues == 0:
        print("\n  Conclusion: PASS - Data is complete and healthy.")
    elif total_issues < 50:
        print("\n  Conclusion: MINOR - %d issues, mostly expected (delisted stocks, etc.)." % total_issues)
    else:
        print("\n  Conclusion: WARN - %d issues (many are delisted/suspended, check severity breakdown)." % total_issues)

    elapsed = time.time() - overall_t0
    print("\n  Total time: %.1fs" % elapsed)
    conn.close()


if __name__ == "__main__":
    check_integrity()
