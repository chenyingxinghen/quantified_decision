"""pull_jydb_parallel.py 增量更新效率分析。"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.jydb_config import JYDB_RAW_DB_PATH
from core.data.jydb_raw_etl import training_raw_specs
from core.data.jydb_feature_store import iter_date_batches


def get_conn():
    conn = sqlite3.connect(JYDB_RAW_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def analyze_pull_efficiency():
    print("=" * 88)
    print(" pull_jydb_parallel.py Incremental Update Efficiency Analysis")
    print("=" * 88)

    db_size = os.path.getsize(JYDB_RAW_DB_PATH) / 1024 / 1024
    print("\n  Database size: %.1f MB" % db_size)
    print("  Database: %s" % JYDB_RAW_DB_PATH)

    conn = get_conn()

    # 1. pull_checkpoint statistics
    print("\n--- 1. pull_checkpoint Batch Statistics ---")
    total_batches = conn.execute("SELECT COUNT(*) FROM pull_checkpoint").fetchone()[0]
    print("  Total checkpoints: %d" % total_batches)

    # Per-table stats (row_count from raw_etl_manifest for accuracy, not pull_checkpoint which has overlap)
    rows = conn.execute(
        "SELECT pc.source_table, pc.batches, m.row_count, "
        "pc.c_min, pc.c_max, "
        "pc.bs_min, pc.be_max "
        "FROM ("
        "  SELECT source_table, COUNT(*) as batches, "
        "  MIN(completed_at) as c_min, MAX(completed_at) as c_max, "
        "  MIN(batch_start) as bs_min, MAX(batch_end) as be_max "
        "  FROM pull_checkpoint GROUP BY source_table"
        ") pc "
        "LEFT JOIN raw_etl_manifest m ON pc.source_table = m.source_table "
        "ORDER BY pc.source_table"
    ).fetchall()

    total_rows = 0
    table_stats = {}
    for r in rows:
        cnt = r[1]
        rc = r[2] or 0
        c_min = r[3]
        c_max = r[4]
        bs = r[5]
        be = r[6]
        total_rows += rc

        dur = None
        if c_min and c_max:
            t_min = datetime.strptime(c_min[:19], "%Y-%m-%d %H:%M:%S")
            t_max = datetime.strptime(c_max[:19], "%Y-%m-%d %H:%M:%S")
            dur = (t_max - t_min).total_seconds()

        table_stats[r[0]] = {
            "batches": cnt,
            "rows": rc,
            "time_range": (c_min, c_max),
            "duration": dur,
            "batch_range": (bs, be),
        }

    print("  %-30s %6s %12s %10s" % ("Table", "Batches", "Rows", "Span"))
    print("  " + "-" * 88)
    for t, s in sorted(table_stats.items()):
        dur_str = "%.0fs" % s["duration"] if s["duration"] else "-"
        print("  %-30s %6d %12s %10s" %
              (t, s["batches"], _c(s["rows"]), dur_str))
    print("  " + "-" * 88)
    total_tables = len(table_stats)
    print("  %-30s %6d %12s" % ("TOTAL (%d tables)" % total_tables, total_batches, _c(total_rows)))

    # 2. Analyze incremental update efficiency
    print("\n--- 2. Incremental Update Overhead Analysis ---")

    specs = training_raw_specs()

    # scenario: daily update from last batch end to today
    last_batch_end = conn.execute(
        "SELECT MAX(batch_end) FROM pull_checkpoint WHERE batch_end != '__snapshot__'"
    ).fetchone()[0]
    print("  Last pulled batch end: %s" % (last_batch_end or "-"))

    today_str = datetime.now().strftime("%Y-%m-%d")
    if last_batch_end and last_batch_end < today_str:
        print("  Pending data: %s ~ %s (%d days behind)" %
              (last_batch_end, today_str,
               (datetime.now() - datetime.strptime(last_batch_end, "%Y-%m-%d")).days))

    # Check how many batches are NOT done for a new run ending today
    print("\n  Simulating a re-run with --start 2020-01-01 --end %s ..." % today_str)
    new_end = today_str
    # Use a smaller start to check overall skip ratio
    check_start = "2020-01-01"

    store = ParallelRawStoreCheck(conn)
    all_items = _build_items_check(store, specs, check_start, new_end, "1990-01-01", 3)
    done_count = sum(1 for item in all_items if item["done"])
    pending_count = sum(1 for item in all_items if not item["done"])
    total_count = len(all_items)
    skip_rate = done_count / total_count * 100 if total_count else 0
    print("  Total batch-range combinations checked: %d" % total_count)
    print("  Already completed (skipped): %d (%.1f%%)" % (done_count, skip_rate))
    print("  Pending (would be pulled):   %d (%.1f%%)" % (pending_count, 100 - skip_rate))

    # 3. Performance of is_batch_done check
    print("\n--- 3. Checkpoint Scan Performance ---")
    # Time how long it takes to scan all checkpoints
    t0 = time.time()
    pc_count = conn.execute("SELECT COUNT(*) FROM pull_checkpoint").fetchone()[0]
    print("  pull_checkpoint table rows: %d" % pc_count)

    # Time the is_batch_done check for all items
    t0 = time.time()
    for item in all_items[:100]:
        s = specs[item["table"]]
        store.is_batch_done(item["table"], item["bs"], item["be"], s.sql)
    t1 = time.time()
    per_check = (t1 - t0) / min(100, len(all_items)) * 1000
    print("  Per-check time (is_batch_done): %.2f ms" % per_check)
    estimated_all = per_check * total_count / 1000
    print("  Estimated total check time (all %d items): %.1fs" % (total_count, estimated_all))

    # 4. SQLite performance
    print("\n--- 4. SQLite Write Performance Estimate ---")
    conn.execute("PRAGMA page_count")
    pages = conn.execute("PRAGMA page_count").fetchone()[0]
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    print("  SQLite pages: %d, page size: %d bytes" % (pages, page_size))
    print("  WAL mode: enabled")
    print("  Synchronous: NORMAL (balanced safety/performance)")

    # Estimate write cost per row
    approx_bytes_per_row = db_size * 1024 * 1024 / total_rows if total_rows else 0
    print("  Approx bytes per row: %.0f" % approx_bytes_per_row)

    # 5. Recommendations
    print("\n--- 5. Recommendations for Incremental Update ---")

    current_approach = (
        "Current approach: checkpoint-based skip (pull_checkpoint table). "
        "On re-run with wide date range, all batches are enumerated and checked "
        "against pull_checkpoint. Completed batches are skipped."
    )
    print("  " + current_approach)

    if pending_count > 0:
        print("\n  >>> %d pending batches would be pulled." % pending_count)
        print("      Estimated new data volume: ~%s rows" % _c(_estimate_pending_rows(pending_count, table_stats)))
        print("      Estimated write time: ~%.0f min" % (pending_count * 0.5))  # rough: 30s per batch
    else:
        print("\n  >>> All batches are up to date. A re-run would complete in ~%.1fs (checkpoint checks only)." %
              estimated_all)

    # Detect patterns
    print("\n  Optimization opportunities:")
    optimizations = []

    # Check if incremental start date mode would help
    if pending_count == 0:
        optimizations.append(
            "1. SHORT-TERM: Reduce --start to recent date (e.g., --start %s) to "
            "skip early-batch enumeration overhead on future runs." %
            (last_batch_end[:7] + "-01" if last_batch_end else "2026-01-01")
        )

    optimizations.append(
        "2. MEDIUM-TERM: Add --incremental flag to pull_jydb_parallel.py "
        "(similar to update_jydb_data.py --incremental). Use raw_etl_manifest's "
        "extracted_end as the start boundary + overlap_days for revision capture, "
        "avoiding full-range batch enumeration."
    )

    # Check for overlapping batches (indicates batch_months changes)
    overlap_count = _count_overlapping_batches(conn)
    if overlap_count > 0:
        optimizations.append(
            "3. HOUSEKEEPING: %d overlapping batch ranges found in pull_checkpoint "
            "(caused by batch_months change between runs). Run a cleanup: "
            "DELETE FROM pull_checkpoint WHERE row_count=0; VACUUM;" %
            overlap_count
        )

    current_batch_span = _get_current_batch_span(conn)
    if current_batch_span > 12 * 3:  # more than 36 months
        optimizations.append(
            "4. LONG-TERM: Consider archiving old pull_checkpoint entries (>3 years). "
            "For re-pull of old data, a fresh pull is needed anyway."
        )

    for opt in optimizations:
        print("  " + opt)

    if not optimizations:
        print("  (No significant optimization opportunities identified)")

    conn.close()
    print("\n  Analysis complete.")


def _get_current_batch_span(conn):
    row = conn.execute(
        "SELECT MIN(batch_start), MAX(batch_end) FROM pull_checkpoint "
        "WHERE batch_start != '__snapshot__'"
    ).fetchone()
    if row and row[0] and row[1]:
        try:
            d1 = datetime.strptime(row[0], "%Y-%m-%d")
            d2 = datetime.strptime(row[1], "%Y-%m-%d")
            return (d2 - d1).days / 30
        except ValueError:
            pass
    return 0


def _count_overlapping_batches(conn):
    """Detect overlapping batch ranges (indicates batch_months config changes)."""
    rows = conn.execute(
        "SELECT source_table, batch_start, batch_end FROM pull_checkpoint "
        "ORDER BY source_table, batch_start"
    ).fetchall()
    overlap = 0
    prev_table = None
    prev_end = None
    for r in rows:
        t = r[0]
        s = r[1]
        e = r[2]
        if t != prev_table:
            prev_table = t
            prev_end = e
            continue
        if s < prev_end:
            overlap += 1
        if e > prev_end:
            prev_end = e
    return overlap


def _estimate_pending_rows(pending_count, table_stats):
    avg_rows_per_batch = sum(s["rows"] for s in table_stats.values()) / max(1, len(table_stats))
    avg_batches_per_table = sum(s["batches"] for s in table_stats.values()) / max(1, len(table_stats))
    est_pending_batches = pending_count
    avg_batch_row_count = avg_rows_per_batch / max(1, avg_batches_per_table)
    return int(est_pending_batches * avg_batch_row_count)


class ParallelRawStoreCheck:
    """Minimal checkpoint checker (no schema init, read-only)."""

    def __init__(self, conn):
        self.conn = conn

    def is_batch_done(self, table, batch_start, batch_end, sql) -> bool:
        sql_hash = hashlib.sha256(sql.encode()).hexdigest()
        row = self.conn.execute(
            "SELECT sql_sha256 FROM pull_checkpoint "
            "WHERE source_table=? AND batch_start=? AND batch_end=?",
            (table, batch_start, batch_end),
        ).fetchone()
        return row is not None and row[0] == sql_hash


def _build_items_check(store, specs, start_date, end_date, bootstrap_start, batch_months):
    items = []
    for name, spec in sorted(specs.items()):
        if spec.history_kind == "snapshot":
            done = store.is_batch_done(name, "__snapshot__", "__snapshot__", spec.sql)
            items.append({"table": name, "bs": "__snapshot__", "be": "__snapshot__", "done": done})
            continue
        table_start = bootstrap_start if spec.history_kind in {"pit", "bootstrap"} else start_date
        if spec.earliest_date and spec.earliest_date > table_start:
            table_start = spec.earliest_date
        for bs, be in iter_date_batches(table_start, end_date, batch_months):
            done = store.is_batch_done(name, bs, be, spec.sql)
            items.append({"table": name, "bs": bs, "be": be, "done": done})
    return items


def _c(n):
    if n is None:
        return "-"
    return "{:,}".format(n)


if __name__ == "__main__":
    analyze_pull_efficiency()
