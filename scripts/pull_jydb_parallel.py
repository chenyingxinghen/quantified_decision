"""多线程聚源原始数据拉取脚本（替代 update_jydb_raw_data.py）。

特性：
- 多线程并行抽取多张表 / 多个时间批次
- 随时可停（Ctrl+C 优雅退出，已完成的批次不丢失）
- 中断续传（重启后自动跳过已完成的批次）
- 进度实时显示
- 输出与 JYDBRawStore 完全兼容（raw_etl_manifest + 相同表结构）

连接配置通过环境变量读取（与 config/jydb_config.py 一致）：
    JYDB_SERVER, JYDB_DATABASE, JYDB_USERNAME, JYDB_PASSWORD, JYDB_DRIVER

用法：
    python scripts/pull_jydb_parallel.py --start 2020-01-01 --end 2025-07-22
    python scripts/pull_jydb_parallel.py --start 2020-01-01 --end 2025-07-22 --workers 6
    python scripts/pull_jydb_parallel.py --start 2020-01-01 --end 2025-07-22 --tables QT_DailyQuote LC_MainIndexNew
"""
from __future__ import annotations

import argparse
import hashlib
import os
import signal
import sqlite3
import sys
import time
import random
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass

from config.jydb_config import JYDB_RAW_DB_PATH, build_connection_string
from core.data.jydb_raw_etl import RawQuerySpec, training_raw_specs
from core.data.jydb_feature_store import iter_date_batches

# ─── 全局停止信号（由主进程初始化） ──────────────────────────────────────────────────
_manager = None
_stop_event = None

def _init_globals():
    global _manager, _stop_event
    _manager = multiprocessing.Manager()
    _stop_event = _manager.Event()

def _signal_handler(signum, frame):
    print("\n[中断] 收到停止信号，等待当前批次完成后退出...")
    if _stop_event:
        _stop_event.set()

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ─── 连接工厂（复用 config/jydb_config 统一入口）─────────────────────────────────
def _connect_source():
    """创建到聚源 SQL Server 的新连接，使用项目统一配置。"""
    import pyodbc
    conn = pyodbc.connect(build_connection_string(), timeout=60)
    # 查询执行超时（秒）。仅 timeout=60 只管登录阶段；不设 cnxn.timeout 时，
    # 聚源侧查询卡死会让进程在 C 层 recv 永久挂起（进程存活但零吞吐、零提交）。
    # 设此值后，卡死的查询会在 300s 后抛 pyodbc.OperationalError，
    # 被主循环 try/except 捕获→标记批次失败→watcher 自愈重试。
    conn.timeout = 300
    return conn


# ─── 本地 SQLite 仓库（兼容 JYDBRawStore schema）─────────────────────────────────
class ParallelRawStore:
    """线程安全的 SQLite 原始数据仓库。

    数据表和 raw_etl_manifest 与 JYDBRawStore 完全兼容；
    额外维护 pull_checkpoint 表实现批次级断点续传。
    """

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=120)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=300000")
        return conn

    def _init_schema(self):
        with closing(self._connect()) as conn, conn:
            # 兼容 JYDBRawStore 的 manifest（下游代码可能读取）
            conn.execute("""
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
            """)
            # 批次级断点续传（本脚本专用）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pull_checkpoint (
                    source_table TEXT NOT NULL,
                    batch_start TEXT NOT NULL,
                    batch_end TEXT NOT NULL,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    sql_sha256 TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    PRIMARY KEY (source_table, batch_start, batch_end)
                )
            """)

    def is_batch_done(self, table: str, batch_start: str, batch_end: str, sql: str) -> bool:
        """检查某批次是否已完成（用于续传跳过）。"""
        sql_hash = hashlib.sha256(sql.encode()).hexdigest()
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT sql_sha256 FROM pull_checkpoint "
                "WHERE source_table=? AND batch_start=? AND batch_end=?",
                (table, batch_start, batch_end),
            ).fetchone()
        return row is not None and row[0] == sql_hash

    @staticmethod
    def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
        """与 JYDBRawStore.normalize 保持一致的清洗逻辑，使用矢量化加速。"""
        data = frame.copy()
        if "code" in data.columns:
            data["code"] = data["code"].astype(str).str.extract(r"(\d{6})", expand=False)
        for col in data.columns:
            lower = col.lower()
            if (lower.endswith("date") or lower.endswith("day")
                    or lower.endswith("time") or lower in {"available_date", "end_date", "xgrq"}):
                data[col] = pd.to_datetime(data[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            elif data[col].dtype == object:
                # 矢量化加速 Decimal 及其他 Object 数字类型的转换
                try:
                    data[col] = data[col].astype("float64")
                except (ValueError, TypeError):
                    # 若存在无法转换的字符串则回退
                    data[col] = data[col].map(lambda v: float(v) if isinstance(v, Decimal) else v)
        return data.where(pd.notna(data), None)

    def _upsert_checkpoint(self, table: str, batch_start: str, batch_end: str,
                           row_count: int, sql_hash: str) -> None:
        """写入/更新批次级 checkpoint（续传用）。空批次也会记录，避免断点续传
        把 0 行结果的历史批次误判为未完成、反复重拉。"""
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO pull_checkpoint "
                "(source_table, batch_start, batch_end, row_count, sql_sha256, completed_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(source_table, batch_start, batch_end) DO UPDATE SET "
                "row_count=excluded.row_count, sql_sha256=excluded.sql_sha256, "
                "completed_at=excluded.completed_at",
                (table, batch_start, batch_end, row_count, sql_hash,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )

    def save_batch(
        self, spec: RawQuerySpec, frame: pd.DataFrame,
        batch_start: str, batch_end: str,
    ) -> int:
        """多进程安全地写入一个批次数据，同时更新 raw_etl_manifest。"""
        data = self._normalize(frame)
        sql_hash = hashlib.sha256(spec.sql.encode()).hexdigest()
        if data.empty:
            # 空结果批次也要记录 checkpoint，否则断点续传会一直认为它未完成
            self._upsert_checkpoint(spec.name, batch_start, batch_end, 0, sql_hash)
            return 0
        table_quoted = '"' + spec.name.replace('"', '""') + '"'

        max_retries = 30
        for attempt in range(max_retries):
            try:
                with closing(self._connect()) as conn, conn:
                    # 删除该批次旧数据（幂等）
                    exists = conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (spec.name,),
                    ).fetchone()
                    if exists and spec.date_col:
                        date_col_q = '"' + spec.date_col.replace('"', '""') + '"'
                        conn.execute(
                            f"DELETE FROM {table_quoted} "
                            f"WHERE substr({date_col_q},1,10) BETWEEN ? AND ?",
                            (batch_start, batch_end),
                        )
                    elif exists and spec.date_col is None:
                        conn.execute(f"DELETE FROM {table_quoted}")

                    data.to_sql(spec.name, conn, if_exists="append", index=False)

                    # 更新 raw_etl_manifest（兼容 JYDBRawStore 格式）
                    row_count = conn.execute(
                        f"SELECT COUNT(*) FROM {table_quoted}"
                    ).fetchone()[0]
                    # snapshot 表日期字段为 None，与 JYDBRawStore 行为一致
                    manifest_start = batch_start if spec.date_col else None
                    manifest_end = batch_end if spec.date_col else None
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
                        (spec.name, spec.history_kind, spec.date_col,
                         manifest_start, manifest_end, row_count, sql_hash,
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    )

                    # 记录批次级 checkpoint（续传用）
                    self._upsert_checkpoint(spec.name, batch_start, batch_end, len(data), sql_hash)
                break  # 成功退出重试循环
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(random.uniform(2.0, 10.0))
                else:
                    raise
        return len(data)

    def get_stats(self) -> Dict[str, int]:
        """获取各表已拉取行数统计（从 raw_etl_manifest 读取）。"""
        with closing(self._connect()) as conn:
            try:
                rows = conn.execute(
                    "SELECT source_table, row_count FROM raw_etl_manifest"
                ).fetchall()
                return {r[0]: r[1] for r in rows}
            except Exception:
                return {}


# ─── 工作单元 ─────────────────────────────────────────────────────────────────────
@dataclass
class WorkItem:
    spec: RawQuerySpec
    batch_start: Optional[str]
    batch_end: Optional[str]
    label: str


def _build_work_items(
    specs: Dict[str, RawQuerySpec],
    start_date: str,
    end_date: str,
    bootstrap_start: str,
    batch_months: int,
    store: ParallelRawStore,
    selected_tables: Optional[Sequence[str]] = None,
) -> List[WorkItem]:
    """构建待处理工作列表，跳过已完成的批次。"""
    items: List[WorkItem] = []
    selected = list(selected_tables or specs.keys())

    for name in selected:
        spec = specs[name]
        if spec.history_kind == "snapshot":
            if not store.is_batch_done(name, "__snapshot__", "__snapshot__", spec.sql):
                items.append(WorkItem(spec, None, None, f"{name} [snapshot]"))
            continue

        table_start = bootstrap_start if spec.history_kind in {"pit", "bootstrap"} else start_date
        if spec.earliest_date and spec.earliest_date > table_start:
            table_start = spec.earliest_date
        for bs, be in iter_date_batches(table_start, end_date, batch_months):
            if not store.is_batch_done(name, bs, be, spec.sql):
                items.append(WorkItem(spec, bs, be, f"{name} [{bs}..{be}]"))

    return items


# ─── 抽取逻辑 ─────────────────────────────────────────────────────────────────────
def _extract_one(item: WorkItem, store: ParallelRawStore, stop_evt: multiprocessing.Event, chunksize: int = 50_000) -> Tuple[str, int]:
    """抽取单个工作单元，返回 (label, row_count)。

    注意：必须先将所有 chunk 收集合并后再调用 save_batch，
    因为 save_batch 内部会 DELETE 该批次已有数据再 INSERT。
    若逐 chunk 调用，前一个 chunk 的数据会被后一个覆盖，
    导致最终只保留最后一个 chunk（即 chunk-overwrite 缺陷）。
    """
    spec = item.spec
    conn = _connect_source()
    try:
        sql = spec.sql
        if spec.parameter_multiplier and item.batch_start and item.batch_end:
            start_lit = pd.Timestamp(item.batch_start).strftime("%Y-%m-%d")
            end_lit = pd.Timestamp(item.batch_end).strftime("%Y-%m-%d")
            for value in [start_lit, end_lit] * spec.parameter_multiplier:
                sql = sql.replace("?", f"CAST('{value}' AS date)", 1)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pandas only supports SQLAlchemy connectable.*",
                category=UserWarning,
            )
            frames = pd.read_sql_query(sql, conn, chunksize=chunksize)

        all_parts: List[pd.DataFrame] = []
        for frame in frames:
            if stop_evt.is_set():
                break
            all_parts.append(frame)

        total = 0
        if all_parts and not stop_evt.is_set():
            combined = pd.concat(all_parts, ignore_index=True)
            total = store.save_batch(spec, combined, item.batch_start or "__snapshot__",
                                     item.batch_end or "__snapshot__")
        return item.label, total
    finally:
        conn.close()


# ─── 主流程 ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="多线程拉取聚源原始数据（支持中断续传）")
    parser.add_argument("--start", required=True, help="日频数据起始日 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="数据截止日 YYYY-MM-DD")
    parser.add_argument("--bootstrap-start", default="1990-01-01",
                        help="PIT/状态表历史起点（默认 1990-01-01）")
    parser.add_argument("--batch-months", type=int, default=3, help="分批月数（默认 3）")
    parser.add_argument("--workers", type=int, default=12, help="并行进程数（默认 12）")
    parser.add_argument("--output", default=JYDB_RAW_DB_PATH,
                        help="本地 SQLite 输出路径（默认 database/jydb_raw.db）")
    parser.add_argument("--tables", nargs="*", help="仅抽取指定表；默认全部")
    parser.add_argument("--chunksize", type=int, default=50_000, help="每次读取行数")
    args = parser.parse_args()

    # 初始化全局信号和多进程管家
    _init_globals()

    if args.batch_months <= 0:
        parser.error("--batch-months 必须为正整数")

    specs = training_raw_specs()
    if args.tables:
        unknown = sorted(set(args.tables) - set(specs.keys()))
        if unknown:
            parser.error(f"未知表: {unknown}\n可选: {sorted(specs.keys())}")

    store = ParallelRawStore(args.output)
    items = _build_work_items(
        specs, args.start, args.end, args.bootstrap_start,
        args.batch_months, store, args.tables,
    )

    total_batches = len(items)
    if total_batches == 0:
        print("[完成] 所有批次已拉取，无需重复执行。")
        _print_summary(store)
        return

    # 统计已完成数量
    done_before = _count_done(specs, args.start, args.end, args.bootstrap_start,
                              args.batch_months, store, args.tables)
    total_all = done_before + total_batches

    # 从 config 读取服务器信息用于显示
    from config.jydb_config import JYDB_SERVER, JYDB_DATABASE
    print(f"{'='*60}")
    print(f" 聚源数据并行拉取")
    print(f" 服务器: {JYDB_SERVER}/{JYDB_DATABASE}")
    print(f" 输出: {os.path.abspath(args.output)}")
    print(f" 进程数: {args.workers} (多进程并发)")
    print(f" 待处理: {total_batches} 批 (已完成 {done_before}, 总计 {total_all})")
    print(f" 按 Ctrl+C 随时停止，下次运行自动续传")
    print(f"{'='*60}\n")

    completed = 0
    failed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {}
        for item in items:
            if _stop_event.is_set():
                break
            future = executor.submit(_extract_one, item, store, _stop_event, args.chunksize)
            future_map[future] = item

        for future in as_completed(future_map):
            if _stop_event.is_set():
                for f in future_map:
                    f.cancel()
                break
            item = future_map[future]
            try:
                label, count = future.result()
                completed += 1
                elapsed = time.time() - start_time
                speed = completed / elapsed * 60 if elapsed > 0 else 0
                size_mb = os.path.getsize(args.output) / 1024 / 1024 if os.path.exists(args.output) else 0
                print(
                    f"  [{completed + done_before}/{total_all}] {label}: "
                    f"{count:,} 行 | 库 {size_mb:,.0f} MB | {speed:.1f} 批/min"
                )
            except Exception as e:
                failed += 1
                print(f"  [失败] {item.label}: {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    if _stop_event.is_set():
        print(f" 已中断 | 本次完成 {completed} 批, 失败 {failed} 批, 耗时 {elapsed:.1f}s")
        print(f" 下次运行将自动续传剩余批次")
    else:
        print(f" 全部完成 | 本次 {completed} 批, 失败 {failed} 批, 耗时 {elapsed:.1f}s")
    print(f"{'='*60}")
    _print_summary(store)

    if failed > 0:
        sys.exit(1)


def _count_done(specs, start_date, end_date, bootstrap_start, batch_months, store, tables):
    """统计已完成的批次数。"""
    count = 0
    selected = list(tables or specs.keys())
    for name in selected:
        spec = specs[name]
        if spec.history_kind == "snapshot":
            if store.is_batch_done(name, "__snapshot__", "__snapshot__", spec.sql):
                count += 1
            continue
        table_start = bootstrap_start if spec.history_kind in {"pit", "bootstrap"} else start_date
        if spec.earliest_date and spec.earliest_date > table_start:
            table_start = spec.earliest_date
        for bs, be in iter_date_batches(table_start, end_date, batch_months):
            if store.is_batch_done(name, bs, be, spec.sql):
                count += 1
    return count


def _print_summary(store: ParallelRawStore):
    stats = store.get_stats()
    if stats:
        print(f"\n 本地仓库统计 ({os.path.basename(store.db_path)}):")
        for table, rows in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"   {table}: {rows:,} 行")
        total = sum(stats.values())
        print(f"   合计: {total:,} 行")


if __name__ == "__main__":
    main()
