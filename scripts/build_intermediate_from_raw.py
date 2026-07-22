"""从本地 raw 库（jydb_raw.db）流水线式生成中间产物。

管线定位（三段式 bronze→silver 架构）：

    聚源 SQL Server ──[pull_jydb_parallel]──▶ jydb_raw.db          (获取 / bronze)
    jydb_raw.db     ──[本脚本]────────────▶ jydb_features.db      (处理 / silver)
                                            + stock_daily.db

与 update_jydb_parallel.py 的区别：
- update_jydb_parallel.py 直连聚源 SQL Server 抽取（受网络 IO 制约，CPU 利用率低）；
- 本脚本只读本地 jydb_raw.db，无网络等待，CPU 可跑满，且可离线重跑。

raw 库中各表已是「投影后」schema（code / available_date / [end_date] / 特征列），
因此可直接喂给 core.data 既有的清洗-透视-写库逻辑（upsert_wide_frame /
upsert_daily_wide_frame / JYDBMarketETL），保证产物 schema 与直连版完全一致。

流水线（producer-consumer）：
- 特征：以 (表 × 日期批次) 为工作单元，ProcessPoolExecutor 并发
  「读 raw 切片 → 透视 → 写特征库」，各单元互不阻塞。
- 行情：daily_data 按日期批次并发写入；复权因子 / ST 状态依赖完整
  daily_data，串行重建（事件表从本地 raw 库一次性读入后注入）。

用法：
    python scripts/build_intermediate_from_raw.py --mode both
    python scripts/build_intermediate_from_raw.py --mode feature --workers 12
    python scripts/build_intermediate_from_raw.py --mode market --clear-market
    # 区间默认自动对齐 raw 库 (jydb_raw.db) 实际覆盖范围；命令行 --start/--end 仅作收窄。
"""
from __future__ import annotations

import argparse
import os
import signal
import sqlite3
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.jydb_config import (
    JYDB_FEATURE_DB_PATH, JYDB_RAW_DB_PATH, DATABASE_PATH,
)
from core.data.jydb_feature_store import DEFAULT_TABLE_SPECS, iter_date_batches

# ─── 优雅停止 ─────────────────────────────────────────────────────────────────────
_stop_requested = False


def _signal_handler(signum, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[中断] 收到停止信号，等待已提交批次结束后退出（已完成批次不丢失）...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ─── 本地 raw 库读取 ──────────────────────────────────────────────────────────────
def _open_raw(raw_db: str) -> sqlite3.Connection:
    """只读方式打开本地 raw 库；多进程并发读安全。"""
    conn = sqlite3.connect(f"file:{raw_db}?mode=ro", uri=True, timeout=120)
    conn.execute("PRAGMA query_only=1")
    return conn


def _raw_coverage(raw_db: str, mode: str) -> Tuple[Optional[str], Optional[str]]:
    """读取 raw 库 raw_etl_manifest 中相关表的实际覆盖范围（真相源）。

    mode='feature' 只看 DEFAULT_TABLE_SPECS 的表；mode='market' 只看行情基础表
    （QT_DailyQuote / LC_STIBDailyQuote）；mode='both' 看全部。
    返回 (min_start, max_end)，无数据则为 (None, None)。
    """
    if not os.path.exists(raw_db):
        return None, None
    try:
        with closing(_open_raw(raw_db)) as conn:
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_etl_manifest'"
            ).fetchone():
                return None, None
            if mode == "feature":
                names = list(DEFAULT_TABLE_SPECS.keys())
            elif mode == "market":
                names = ["QT_DailyQuote", "LC_STIBDailyQuote"]
            else:
                names = list(DEFAULT_TABLE_SPECS.keys()) + [
                    "QT_DailyQuote", "LC_STIBDailyQuote"
                ]
            placeholders = ",".join("?" for _ in names)
            row = conn.execute(
                f"SELECT MIN(extracted_start), MAX(extracted_end) "
                f"FROM raw_etl_manifest WHERE source_table IN ({placeholders})",
                names,
            ).fetchone()
            return (row[0], row[1]) if row else (None, None)
    except Exception:
        return None, None


def _resolve_build_range(raw_db: str, mode: str, start: str, end: str) -> Tuple[str, str]:
    """把命令行区间对齐到 raw 库实际覆盖区间。

    规则（解决时间区间碎片化）：
    - 默认/请求区间不得超过 raw 库实际覆盖；超出部分被裁剪，避免 build 到
      没有源数据的区间（特征缺失/不一致）。
    - 请求区间可以比 raw 覆盖更窄（用户主动裁剪），此时尊重用户意图。
    - 若 raw 库无 manifest 信息，则原样返回请求区间（保留旧行为）。
    """
    cov_start, cov_end = _raw_coverage(raw_db, mode)
    if not cov_start or not cov_end:
        return start, end
    eff_start = start if pd.Timestamp(start) > pd.Timestamp(cov_start) else cov_start
    eff_end = end if pd.Timestamp(end) < pd.Timestamp(cov_end) else cov_end
    if eff_start != start or eff_end != end:
        print(f"  [区间对齐] 请求 {start}..{end} → 对齐 raw 覆盖 {eff_start}..{eff_end}")
    return eff_start, eff_end


def _daily_data_coverage(market_db: str) -> Tuple[Optional[str], Optional[str]]:
    """读取 daily_data 实际覆盖区间（下游派生的真实边界）。"""
    if not os.path.exists(market_db):
        return None, None
    try:
        with closing(sqlite3.connect(market_db, timeout=30)) as conn:
            row = conn.execute("SELECT MIN(date), MAX(date) FROM daily_data").fetchone()
            return (row[0], row[1]) if row and row[0] else (None, None)
    except Exception:
        return None, None


def _raw_has_rows(raw_db: str, table: str) -> bool:
    with closing(_open_raw(raw_db)) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            return False
        return conn.execute(f'SELECT 1 FROM "{table}" LIMIT 1').fetchone() is not None


def _read_raw_slice(
    raw_db: str, table: str, date_col: str,
    start: str, end: str, chunksize: int,
):
    """从 raw 库按 available_date 区间分块读取一张表的切片。"""
    conn = _open_raw(raw_db)
    quoted = '"' + table.replace('"', '""') + '"'
    dc = '"' + date_col.replace('"', '""') + '"'
    sql = (
        f"SELECT * FROM {quoted} "
        f"WHERE substr({dc},1,10) >= ? AND substr({dc},1,10) <= ?"
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="pandas only supports SQLAlchemy connectable.*",
                category=UserWarning,
            )
            yield from pd.read_sql_query(
                sql, conn, params=(start, end), chunksize=chunksize
            )
    finally:
        conn.close()


# ─── 进程内工作函数（顶层定义，Windows spawn 可 pickle）──────────────────────────
def _feature_worker(
    table: str, start: str, end: str,
    raw_db: str, feature_db: str, chunksize: int,
) -> Tuple[str, str, str, int]:
    """读 raw 切片 → 透视 → 写特征库；一个 (表 × 日期批次) 工作单元。

    特征写入为 upsert，幂等可安全重跑。
    """
    from core.data.jydb_feature_store import JYDBFeatureStore

    spec = DEFAULT_TABLE_SPECS[table]
    store = JYDBFeatureStore(feature_db)
    total = 0
    for chunk in _read_raw_slice(
        raw_db, table, spec.available_date_col, start, end, chunksize
    ):
        if chunk.empty:
            continue
        if spec.storage == "daily":
            total += store.upsert_daily_wide_frame(
                chunk,
                date_col=spec.available_date_col,
                feature_cols=spec.feature_cols,
                dimension_cols=spec.dimension_cols,
                prefix=spec.prefix,
            )
        else:
            total += store.upsert_wide_frame(
                chunk,
                source_table=spec.name,
                available_date_col=spec.available_date_col,
                end_date_col=spec.end_date_col,
                feature_cols=spec.feature_cols,
                dimension_cols=spec.dimension_cols,
                prefix=spec.prefix,
            )
    return table, start, end, total


def _market_worker(
    start: str, end: str, raw_db: str, market_db: str, chunksize: int,
) -> Tuple[str, str, int]:
    """读 raw 日线切片 → 组装标准列 → 写 daily_data；一个日期批次。

    日线写入为 upsert，幂等可安全重跑。
    """
    from core.data.jydb_market_etl import JYDBMarketETL

    etl = JYDBMarketETL(market_db)
    chunks = _read_daily_quote_slices(raw_db, start, end, chunksize)
    count = etl.write_daily_chunks(chunks)
    return start, end, count


# ─── 行情：从 raw 库组装标准 daily_data 列 ─────────────────────────────────────────
def _read_daily_quote_slices(raw_db: str, start: str, end: str, chunksize: int):
    """把 raw 库的 QT_DailyQuote + LC_STIBDailyQuote 行情与 LC_DIndicesForValuation
    估值 LEFT JOIN，产出与 DAILY_QUOTE_SQL 同口径的标准列，逐块 yield。

    估值表若为空（未拉取），PE/PB/PS/PCF 置空——不阻塞行情写入。
    """
    conn = _open_raw(raw_db)
    try:
        has_val = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='LC_DIndicesForValuation'"
        ).fetchone() is not None
        val_join = ""
        val_select = "NULL AS peTTM, NULL AS pbMRQ, NULL AS psTTM, NULL AS pcfNcfTTM"
        if has_val:
            val_join = (
                "LEFT JOIN LC_DIndicesForValuation v "
                "ON q.code = v.code AND substr(q.TradingDay,1,10) = substr(v.available_date,1,10)"
            )
            val_select = (
                "v.PE AS peTTM, v.PB AS pbMRQ, v.PSTTM AS psTTM, v.PCFTTM AS pcfNcfTTM"
            )
        parts = []
        for quote_tbl in ("QT_DailyQuote", "LC_STIBDailyQuote"):
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (quote_tbl,),
            ).fetchone()
            if not exists:
                continue
            parts.append(f"""
                SELECT q.code AS code, q.TradingDay AS date,
                       q.OpenPrice AS [open], q.HighPrice AS high, q.LowPrice AS low,
                       q.ClosePrice AS [close], q.PrevClosePrice AS preclose,
                       q.TurnoverVolume AS volume, q.TurnoverValue AS amount,
                       q.TurnoverDeals AS turnover_deals,
                       1 AS tradestatus,
                       CASE WHEN q.PrevClosePrice IS NULL OR q.PrevClosePrice = 0 THEN NULL
                            ELSE (q.ClosePrice / q.PrevClosePrice - 1) * 100 END AS pctChg,
                       {val_select}
                FROM {quote_tbl} q
                {val_join}
                WHERE substr(q.TradingDay,1,10) >= ? AND substr(q.TradingDay,1,10) <= ?
            """)
        if not parts:
            return
        sql = " UNION ALL ".join(parts)
        params = tuple(p for _ in parts for p in (start, end))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="pandas only supports SQLAlchemy connectable.*",
                category=UserWarning,
            )
            yield from pd.read_sql_query(sql, conn, params=params, chunksize=chunksize)
    finally:
        conn.close()


def _read_adjust_events(raw_db: str, end: str) -> pd.DataFrame:
    """从 raw 库 QT_AdjustingFactor 读除权累计比例因子事件。"""
    with closing(_open_raw(raw_db)) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='QT_AdjustingFactor'"
        ).fetchone():
            return pd.DataFrame(columns=["code", "effective_date", "factor"])
        return pd.read_sql_query(
            "SELECT code, ExDiviDate AS effective_date, "
            "RatioAdjustingFactor AS factor "
            "FROM QT_AdjustingFactor "
            "WHERE substr(ExDiviDate,1,10) <= ? "
            "ORDER BY code, ExDiviDate",
            conn, params=(end,),
        )


def _read_st_events(raw_db: str, end: str) -> pd.DataFrame:
    """从 raw 库 LC_SpecialTrade 读特别处理事件。"""
    with closing(_open_raw(raw_db)) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='LC_SpecialTrade'"
        ).fetchone():
            return pd.DataFrame(columns=["code", "effective_date", "SecurityAbbr"])
        return pd.read_sql_query(
            "SELECT code, SpecialTradeTime AS effective_date, "
            "SecurityAbbr, SpecialTradeType "
            "FROM LC_SpecialTrade "
            "WHERE substr(SpecialTradeTime,1,10) <= ? "
            "ORDER BY code, SpecialTradeTime, InfoPublDate",
            conn, params=(end,),
        )


# ─── 工作单元构建 ─────────────────────────────────────────────────────────────────
def _get_watermarks(feature_db: str) -> Dict[str, str]:
    """读取特征库中各表的水位线。"""
    if not os.path.exists(feature_db):
        return {}
    from core.data.jydb_feature_store import JYDBFeatureStore
    store = JYDBFeatureStore(feature_db)
    return {
        name: wm for name in DEFAULT_TABLE_SPECS
        if (wm := store.get_watermark(name)) is not None
    }


def _feature_work_items(
    tables: Sequence[str], start: str, end: str,
    batch_months: int, raw_db: str,
    feature_db: str = "", overlap_days: int = 5,
) -> List[Tuple[str, str, str]]:
    """构建 (表, 批次起, 批次止) 列表。
    
    增量模式（feature_db 非空时）：读取各表水位线，跳过已处理区间，
    并通过 overlap_days 回看以捕获源数据修订。
    """
    items: List[Tuple[str, str, str]] = []
    skipped: List[str] = []
    watermarks = _get_watermarks(feature_db) if feature_db else {}
    for name in tables:
        spec = DEFAULT_TABLE_SPECS[name]
        if not _raw_has_rows(raw_db, name):
            skipped.append(name)
            continue
        table_start = start
        if spec.earliest_date and spec.earliest_date > table_start:
            table_start = spec.earliest_date

        wm = watermarks.get(name)
        if wm:
            if wm >= end:
                skipped.append(f"{name}(水位{wm})")
                continue
            wm_back = (pd.Timestamp(wm) - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")
            if wm_back > table_start:
                table_start = wm_back

        for bs, be in iter_date_batches(table_start, end, batch_months):
            items.append((name, bs, be))
    if skipped:
        print(f"  [增量跳过] {len(skipped)} 个表: {', '.join(skipped)}")
    return items


# ─── 特征并行 ETL ─────────────────────────────────────────────────────────────────
def run_features(
    tables: Sequence[str], start: str, end: str,
    raw_db: str, feature_db: str, workers: int, batch_months: int, chunksize: int,
    overlap_days: int = 5,
) -> None:
    from core.data.jydb_feature_store import JYDBFeatureStore

    items = _feature_work_items(
        tables, start, end, batch_months, raw_db,
        feature_db, overlap_days,
    )
    print(f"\n{'='*64}")
    print(f" 特征 ETL（本地源，多进程）→ {os.path.abspath(feature_db)}")
    print(f" 源: {os.path.abspath(raw_db)}")
    print(f" 工作单元: {len(items)} 个 | 进程数: {workers}")
    print(f"{'='*64}", flush=True)
    if not items:
        print(" 无待处理工作单元。")
        return

    store = JYDBFeatureStore(feature_db)  # 主进程初始化 schema
    store.initialize()

    completed = failed = 0
    table_max_end: Dict[str, str] = {}
    t0 = time.time()
    _last_ckpt = time.time()
    _CKPT_INTERVAL = 15  # WAL checkpoint 间隔（秒）

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for (name, bs, be) in items:
            if _stop_requested:
                break
            fut = executor.submit(
                _feature_worker, name, bs, be, raw_db, feature_db, chunksize
            )
            future_map[fut] = (name, bs, be)

        for fut in as_completed(future_map):
            name, bs, be = future_map[fut]
            if _stop_requested:
                for f in future_map:
                    f.cancel()
                break
            try:
                tbl, b_start, b_end, count = fut.result()
                completed += 1
                prev = table_max_end.get(tbl)
                if prev is None or b_end > prev:
                    table_max_end[tbl] = b_end
                elapsed = time.time() - t0
                size_mb = os.path.getsize(feature_db) / 1024 / 1024 if os.path.exists(feature_db) else 0
                speed = completed / elapsed * 60 if elapsed > 0 else 0
                print(f"  [{completed}/{len(items)}] {tbl} [{b_start}..{b_end}]: "
                      f"{count:,} 值 | 库 {size_mb:,.0f} MB | {speed:.1f} 批/min", flush=True)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  [失败] {name} [{bs}..{be}]: {e}", file=sys.stderr, flush=True)

            if time.time() - _last_ckpt > _CKPT_INTERVAL:
                try:
                    with closing(sqlite3.connect(feature_db, timeout=60)) as _ckpt:
                        _ckpt.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    _last_ckpt = time.time()
                except Exception:
                    pass

    for tbl, max_end in table_max_end.items():
        try:
            store.set_watermark(tbl, max_end)
        except Exception as e:  # noqa: BLE001
            print(f"  [警告] 写水位失败 {tbl}: {e}", file=sys.stderr)

    try:
        with closing(sqlite3.connect(feature_db, timeout=120)) as _ckpt:
            _ckpt.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass

    print(f"\n 特征 ETL {'中断' if _stop_requested else '完成'}: "
          f"成功 {completed} 批, 失败 {failed} 批, 耗时 {time.time() - t0:.1f}s", flush=True)
    if failed:
        print(" 提示: 失败批次可重跑本脚本补齐（upsert 幂等）。", file=sys.stderr)


# ─── 行情并行 ETL ─────────────────────────────────────────────────────────────────
def run_market(
    start: str, end: str, raw_db: str, market_db: str,
    workers: int, batch_months: int, chunksize: int, clear: bool,
    overlap_days: int = 5,
) -> None:
    from core.data.jydb_market_etl import JYDBMarketETL

    print(f"\n{'='*64}")
    print(f" 行情 ETL（本地源，多进程）→ {os.path.abspath(market_db)}")
    print(f" 源: {os.path.abspath(raw_db)} | 区间: {start}..{end} | 进程数: {workers}")
    print(f"{'='*64}", flush=True)

    market_etl = JYDBMarketETL(market_db)
    market_etl.initialize()

    if clear:
        print("  [--clear-market] 清空 daily_data / adjust_factor...")
        with closing(sqlite3.connect(market_db, timeout=60)) as conn, conn:
            conn.execute("PRAGMA busy_timeout=120000")
            conn.execute("DELETE FROM daily_data")
            conn.execute("DELETE FROM adjust_factor")

    batches = list(iter_date_batches(start, end, batch_months))
    if not clear:
        try:
            with closing(sqlite3.connect(market_db, timeout=10)) as _chk:
                _row = _chk.execute("SELECT MAX(date) FROM daily_data").fetchone()
            if _row and _row[0]:
                cutoff = (pd.Timestamp(_row[0]) - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")
                filtered = [(b, e) for b, e in batches if e > cutoff]
                print(f"  [增量跳过] daily_data 最新日期 {_row[0]}，"
                      f"保留 {len(filtered)}/{len(batches)} 个批次", flush=True)
                batches = filtered
        except Exception:
            pass

    print(f" 日线工作单元: {len(batches)} 个日期批次", flush=True)

    completed = failed = total_rows = 0
    t0 = time.time()
    _last_ckpt = time.time()
    _CKPT_INTERVAL = 15

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for (bs, be) in batches:
            if _stop_requested:
                break
            fut = executor.submit(_market_worker, bs, be, raw_db, market_db, chunksize)
            future_map[fut] = (bs, be)

        for fut in as_completed(future_map):
            bs, be = future_map[fut]
            if _stop_requested:
                for f in future_map:
                    f.cancel()
                break
            try:
                b_start, b_end, count = fut.result()
                completed += 1
                total_rows += count
                elapsed = time.time() - t0
                size_mb = os.path.getsize(market_db) / 1024 / 1024 if os.path.exists(market_db) else 0
                speed = completed / elapsed * 60 if elapsed > 0 else 0
                print(f"  [{completed}/{len(batches)}] daily_data [{b_start}..{b_end}]: "
                      f"{count:,} 行 | 库 {size_mb:,.0f} MB | {speed:.1f} 批/min", flush=True)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  [失败] daily_data [{bs}..{be}]: {e}", file=sys.stderr, flush=True)

            if time.time() - _last_ckpt > _CKPT_INTERVAL:
                try:
                    with closing(sqlite3.connect(market_db, timeout=60)) as _ckpt:
                        _ckpt.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    _last_ckpt = time.time()
                except Exception:
                    pass

    print(f"\n 日线写入 {'中断' if _stop_requested else '完成'}: "
          f"成功 {completed} 批, 失败 {failed} 批, 共 {total_rows:,} 行, "
          f"耗时 {time.time() - t0:.1f}s", flush=True)

    if _stop_requested or failed:
        print(" 提示: 日线未全部完成，复权因子与 ST 状态将在下次完整运行后重建。", file=sys.stderr)
        return

    # 复权因子与 ST 状态依赖完整 daily_data，事件从本地 raw 库一次性读入后注入串行重建。
    # 关键：无论本次 build 的区间多窄，这两个派生表都必须按 daily_data 的全量覆盖重建，
    # 否则窄区间会让早期交易日缺少复权因子 / ST 标记（开天窗），造成前复权错位与非 ST 误判。
    rebuild_start, rebuild_end = _daily_data_coverage(market_db) or (start, end)
    if (rebuild_start, rebuild_end) != (start, end):
        print(f"  [全量重建] 复权/ST 范围对齐 daily_data 全量: {rebuild_start}..{rebuild_end}", flush=True)

    print("\n 重建复权因子（adjust_factor）...", flush=True)
    t1 = time.time()
    adj_events = _read_adjust_events(raw_db, rebuild_end)
    factors = market_etl.rebuild_adjust_factors(rebuild_start, rebuild_end, events=adj_events)
    print(f"  adjust_factor: {factors:,} 行, 耗时 {time.time() - t1:.1f}s", flush=True)

    print(" 重建逐日 ST 状态（daily_data.is_st）...", flush=True)
    t1 = time.time()
    st_events = _read_st_events(raw_db, rebuild_end)
    st_rows = market_etl.rebuild_st_status(rebuild_start, rebuild_end, events=st_events)
    print(f"  st_status: {st_rows:,} 行, 耗时 {time.time() - t1:.1f}s", flush=True)

    # 市场情绪因子（涨跌家数比、涨停比、市场广度等）由完整 daily_data 汇总而来，
    # 是聚源行情的下游派生特征。放在行情重建之后统一预计算，写入 stock_meta.db
    # 的 market_sentiment 表，供训练/回测直接读取，避免运行时再触发懒计算。
    print("\n 计算全市场情绪指标（market_sentiment）...", flush=True)
    t1 = time.time()
    from core.data.market_sentiment_calculator import MarketSentimentCalculator
    sentiment_calc = MarketSentimentCalculator(market_db)
    sentiment_calc.check_and_update()
    print(f"  market_sentiment 更新完成, 耗时 {time.time() - t1:.1f}s", flush=True)

    try:
        with closing(sqlite3.connect(market_db, timeout=60)) as _ckpt:
            _ckpt.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass


# ─── 主流程 ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="从本地 raw 库流水线式生成中间产物（特征库 + 行情库）"
    )
    parser.add_argument("--mode", choices=["feature", "market", "both"], default="both",
                        help="处理目标：feature=特征库, market=行情库, both=两者（默认）")
    parser.add_argument("--start", default='2022-01-01', help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", default='2027-01-01', help="结束日期 YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=8, help="并行进程数（默认 8）")
    parser.add_argument("--batch-months", type=int, default=3,
                        help="日期分批月数（默认 3；越小并行度越高但单批越小）")
    parser.add_argument("--tables", nargs="*", help="仅处理指定特征表；默认全部")
    parser.add_argument("--raw-db", default=JYDB_RAW_DB_PATH, help="本地 raw 库路径（数据源）")
    parser.add_argument("--feature-db", default=JYDB_FEATURE_DB_PATH, help="特征库输出路径")
    parser.add_argument("--market-db", default=DATABASE_PATH, help="行情库输出路径")
    parser.add_argument("--chunksize", type=int, default=100_000, help="每次读取行数")
    parser.add_argument("--overlap-days", type=int, default=5,
                        help="增量回看天数以捕获源表修订（默认 5）")
    parser.add_argument("--clear-market", action="store_true",
                        help="行情模式前先清空 daily_data/adjust_factor")
    args = parser.parse_args()

    if args.batch_months <= 0:
        parser.error("--batch-months 必须为正整数")
    if not os.path.exists(args.raw_db):
        parser.error(f"raw 库不存在: {args.raw_db}")

    tables = list(args.tables or DEFAULT_TABLE_SPECS.keys())
    unknown = sorted(set(tables) - set(DEFAULT_TABLE_SPECS))
    if unknown:
        parser.error(f"未知特征表: {unknown}\n可选: {sorted(DEFAULT_TABLE_SPECS)}")

    # 区间治理：把命令行区间对齐到 raw 库实际覆盖范围，避免 build 到无源数据区间。
    eff_start, eff_end = _resolve_build_range(
        args.raw_db, args.mode, args.start, args.end
    )

    print(f"本地源中间产物流水线 | 数据源: {os.path.abspath(args.raw_db)}")
    print(f"请求区间 {args.start}..{args.end} | 对齐后 {eff_start}..{eff_end} | "
          f"模式 {args.mode} | 进程 {args.workers} | 批 {args.batch_months} 月", flush=True)

    if args.mode in ("feature", "both"):
        run_features(tables, eff_start, eff_end, args.raw_db, args.feature_db,
                     args.workers, args.batch_months, args.chunksize,
                     args.overlap_days)

    if args.mode in ("market", "both") and not _stop_requested:
        run_market(eff_start, eff_end, args.raw_db, args.market_db,
                   args.workers, args.batch_months, args.chunksize, args.clear_market,
                   args.overlap_days)

    # 区间一致性自检：raw 覆盖 vs features 水位 vs daily_data 覆盖
    _self_check_coverage(args.raw_db, args.feature_db, args.market_db)

    print("\n=== 流程结束 ===", flush=True)


def _self_check_coverage(raw_db: str, feature_db: str, market_db: str) -> None:
    """build 结束后做区间一致性自检，打印告警。"""
    cov_s, cov_e = _raw_coverage(raw_db, "both")
    d_s, d_e = _daily_data_coverage(market_db)
    print("\n--- 区间一致性自检 ---")
    print(f"  raw 覆盖:    {cov_s} .. {cov_e}")
    print(f"  daily_data:  {d_s} .. {d_e}")
    if cov_s and d_s and pd.Timestamp(d_s) < pd.Timestamp(cov_s):
        print("  [警告] daily_data 早于 raw 覆盖起始，可能行情来自历史 Baostock 残留，"
              "建议 --clear-market 后全量重建")
    if cov_e and d_e and pd.Timestamp(d_e) > pd.Timestamp(cov_e):
        print(f"  [警告] daily_data 超出 raw 覆盖结束 ({cov_e})，超出部分缺少源数据支撑")
    print("----------------------")


if __name__ == "__main__":
    main()
