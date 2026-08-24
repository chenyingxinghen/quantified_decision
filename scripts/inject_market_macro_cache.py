#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段0：把市场级外生数据（market_macro_daily）注入独立因子缓存
================================================================

对 ``factors_cache_phase0`` 下每只股票的 parquet 追加 ``mkt_*`` 列：

    mkt_margin_balance   两融余额（亿元）
    mkt_margin_fin_buy   融资买入额（亿元）
    mkt_basis_if         IF 主力升贴水（1.0=100%，负=贴水）
    mkt_basis_ih         IH 主力升贴水
    mkt_basis_ic         IC 主力升贴水
    mkt_shibor_on        SHIBOR 隔夜（%）
    mkt_shibor_1w        SHIBOR 1W（%）
    mkt_shibor_3m        SHIBOR 3M（%）
    mkt_lpr_1y           LPR 1Y（%）
    mkt_lpr_5y           LPR 5Y（%）
    mkt_cn10y            中国国债 10Y 收益率（%）

设计要点
--------
1. **独立缓存，不碰共享 factors_cache**（T052 铁律）：注入只发生在
   ``--cache-dir`` 指定的副本上；训练用 ``--cache-dir .../factors_cache_phase0``，
   模型绑定该缓存，回测/推理按绑定读取。
2. **PIT 安全**：所有列 T 日可得（两融/期指/SHIBOR/国债盘后发布，LPR 发布日
   可见），按 date 左连接 + 前向填充（月度 LPR 自然摊平），前导缺失填 0
   （与缓存现有 fill 约定一致）。
3. **归一化路径**：``mkt_`` 前缀命中 ``TrainingConfig.should_skip_rank``
   （market_ 前缀规则），走 B/C 类 robust-sigmoid 跨时间全局统计 —— 与
   ``up_ratio``/``mean_return`` 等市场列同一分支，无需改归一化代码。
4. **幂等**：parquet 已含任一 ``mkt_`` 列则跳过（--force 重写）。

用法:
  python scripts/inject_market_macro_cache.py \
      --cache-dir database/system_data/factors_cache_phase0 [--force] [--workers 8]
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.baostock_config import META_DB_PATH

# 注入列：{目标列名: market_macro_daily 源列名}
INJECT_COLS = {
    'mkt_margin_balance': 'margin_balance',
    'mkt_margin_fin_buy': 'margin_fin_buy',
    'mkt_basis_if': 'basis_if',
    'mkt_basis_ih': 'basis_ih',
    'mkt_basis_ic': 'basis_ic',
    'mkt_shibor_on': 'shibor_on',
    'mkt_shibor_1w': 'shibor_1w',
    'mkt_shibor_3m': 'shibor_3m',
    'mkt_lpr_1y': 'lpr_1y',
    'mkt_lpr_5y': 'lpr_5y',
    'mkt_cn10y': 'cn10y',
}


def load_macro() -> pd.DataFrame:
    """读 market_macro_daily 全表，date 字符串索引，只留注入用列。"""
    conn = sqlite3.connect(META_DB_PATH, timeout=60.0)
    try:
        src_cols = list(dict.fromkeys(INJECT_COLS.values()))
        df = pd.read_sql_query(
            f'SELECT date, {",".join(src_cols)} FROM market_macro_daily '
            'ORDER BY date ASC', conn)
    finally:
        conn.close()
    df['date'] = df['date'].astype(str)
    df = df.set_index('date')
    return df


def process_one(path: str, macro: pd.DataFrame, force: bool) -> tuple:
    """给单个 parquet 注入 mkt_* 列。返回 (path, status)。"""
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
        table = pq.read_table(path)
        names = set(table.schema.names)
        if not force and any(c in names for c in INJECT_COLS):
            return path, 'skip'

        df = table.to_pandas()
        if 'date' not in df.columns:
            return path, 'err:no-date'
        dates = df['date'].astype(str)
        # 左连接 + 前向填充 + 前导填 0
        joined = dates.to_frame('date').join(macro, on='date', how='left')
        joined = joined.ffill().fillna(0.0)
        for target, src in INJECT_COLS.items():
            df[target] = joined[src].astype(np.float32).values

        # 重写 parquet（保持原列顺序，mkt_* 追加在后）
        for c in INJECT_COLS:
            if c in table.schema.names:
                df = df.drop(columns=[c])
        table_new = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table_new, path, compression='snappy')
        return path, 'ok'
    except Exception as e:
        return path, f'err:{type(e).__name__}:{e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir',
                    default='database/system_data/factors_cache_phase0')
    ap.add_argument('--force', action='store_true', help='已注入也重写')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    if not os.path.isdir(a.cache_dir):
        print(f'[ABORT] 缓存目录不存在: {a.cache_dir}')
        sys.exit(1)

    macro = load_macro()
    print(f'market_macro_daily: {len(macro)} 行, 列 {list(macro.columns)}')

    files = sorted(glob.glob(os.path.join(a.cache_dir, '*_factors.parquet')))
    print(f'待处理 parquet: {len(files)}')

    t0 = time.time()
    ok = skip = err = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (path, status) in enumerate(
                ex.map(lambda p: process_one(p, macro, a.force), files), 1):
            if status == 'ok':
                ok += 1
            elif status == 'skip':
                skip += 1
            else:
                err += 1
                print(f'  [ERR] {os.path.basename(path)}: {status}')
            if i % 500 == 0:
                el = time.time() - t0
                print(f'  {i}/{len(files)}  ok={ok} skip={skip} err={err}  '
                      f'{el:.0f}s ({el / i:.2f}s/个)')

    el = time.time() - t0
    print(f'\n完成: ok={ok} skip={skip} err={err}  耗时 {el:.0f}s')
    if err:
        print(f'[WARN] {err} 个文件注入失败，需人工检查')
        sys.exit(2)


if __name__ == '__main__':
    main()
