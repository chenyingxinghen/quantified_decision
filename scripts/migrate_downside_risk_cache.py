"""从历史因子缓存创建严格单变量副本：仅替换 downside_risk 列。"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.baostock_config import DATABASE_PATH
from core.factors.cache_manifest import (
    load_manifest,
    validate_cache_manifest,
    write_cache_manifest,
)


def _calculate_downside_risk(code: str, first_date: str, last_date: str) -> pd.DataFrame:
    # 源缓存首日未必是上市首日；为 rolling(20) 读取 PIT 暖启动历史，
    # 但最终只映射回源 parquet 原有日期域，不新增或删除任何行。
    query_start = (
        datetime.strptime(first_date, '%Y-%m-%d') - timedelta(days=60)
    ).strftime('%Y-%m-%d')
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        prices = pd.read_sql_query(
            """
            SELECT k.date, k.close, a.fore_adjust_factor
            FROM daily_data k
            LEFT JOIN adjust_factor a ON k.code = a.code AND k.date = a.date
            WHERE k.code = ? AND k.date >= ? AND k.date <= ?
            ORDER BY k.date
            """,
            conn,
            params=[code, query_start, last_date],
        )
    finally:
        conn.close()
    if prices.empty:
        raise RuntimeError(f'{code} 无法加载迁移所需行情')
    prices['date'] = prices['date'].astype(str).str[:10]
    close = pd.to_numeric(prices['close'], errors='coerce')
    factor = pd.to_numeric(prices['fore_adjust_factor'], errors='coerce').bfill().ffill().fillna(1.0)
    base = float(factor.iloc[-1]) if len(factor) else 1.0
    if not np.isfinite(base) or base == 0:
        base = 1.0
    adjusted_close = close * (factor / base)
    downside = adjusted_close.pct_change().clip(upper=0.0)
    risk = np.sqrt(downside.pow(2).rolling(20).mean()).fillna(0.0).astype(np.float32)
    return pd.DataFrame({'date': prices['date'], 'downside_risk': risk})


def _migrate_one(source_file: str, target_dir: str, force: bool = False) -> tuple[str, bool, str]:
    source_file = os.path.abspath(source_file)
    target_file = os.path.join(target_dir, os.path.basename(source_file))
    if os.path.exists(target_file) and not force:
        return os.path.basename(source_file), True, 'skipped'
    try:
        frame = pd.read_parquet(source_file)
        if frame.empty or 'date' not in frame.columns or 'downside_risk' not in frame.columns:
            raise RuntimeError('源缓存缺少 date/downside_risk 或为空')
        dates = frame['date'].astype(str).str[:10]
        code = os.path.basename(source_file).split('_factors.parquet')[0]
        replacement = _calculate_downside_risk(code, dates.iloc[0], dates.iloc[-1])
        replacement_map = replacement.set_index('date')['downside_risk']
        new_values = dates.map(replacement_map)
        if new_values.isna().any():
            missing = int(new_values.isna().sum())
            raise RuntimeError(f'有 {missing} 行日期无法匹配行情')
        # 严格单变量：DataFrame 其余列不重算、不转换、不填充。
        frame['downside_risk'] = new_values.to_numpy(dtype=np.float32)
        os.makedirs(target_dir, exist_ok=True)
        temp_file = target_file + f'.tmp-{os.getpid()}'
        frame.to_parquet(temp_file, index=False)
        os.replace(temp_file, target_file)
        return os.path.basename(source_file), True, 'written'
    except Exception as exc:
        return os.path.basename(source_file), False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description='严格单变量迁移 downside_risk 因子缓存')
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--target-dir', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None, help='仅迁移前 N 个文件，用于验收')
    parser.add_argument('--force', action='store_true', help='原子重写目标目录中已存在的同名缓存')
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source_dir)
    target_dir = os.path.abspath(args.target_dir)
    if source_dir == target_dir:
        raise ValueError('源目录与目标目录不能相同，禁止原地覆盖历史缓存')
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(source_dir)

    os.makedirs(target_dir, exist_ok=True)
    manifest = load_manifest(target_dir)
    existing_parquet = list(Path(target_dir).glob('*.parquet'))
    if manifest is None and existing_parquet:
        raise RuntimeError('目标目录已有 parquet 但缺少版本清单，拒绝继续')
    if manifest is None:
        write_cache_manifest(target_dir)
    else:
        validate_cache_manifest(target_dir)

    files = sorted(str(path) for path in Path(source_dir).glob('*_factors.parquet'))
    if args.limit is not None:
        files = files[:args.limit]
    print(f'严格单变量迁移: {len(files)} 个缓存文件')
    written = skipped = failed = 0
    workers = max(1, min(args.workers, len(files) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_migrate_one, path, target_dir, args.force): path
            for path in files
        }
        for index, future in enumerate(as_completed(futures), 1):
            name, ok, status = future.result()
            if ok and status == 'written':
                written += 1
            elif ok:
                skipped += 1
            else:
                failed += 1
                print(f'  [失败] {name}: {status}')
            if index % 250 == 0 or index == len(files):
                print(f'  进度 {index}/{len(files)} | 写入 {written} 跳过 {skipped} 失败 {failed}')
    if failed:
        raise RuntimeError(f'迁移存在 {failed} 个失败文件')
    print(f'迁移完成: 写入 {written}, 跳过 {skipped}, 失败 {failed}')


if __name__ == '__main__':
    main()
