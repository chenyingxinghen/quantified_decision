"""
指数相对因子（``idx_*``，族 index_rel）—— T115 晋级后的生产实现。

来历
----
这 5 列最早由 ``scripts/build_idxrel_cache.py`` **离线注入**进实验缓存
（复制 242 列基础缓存 + 按 date 左连接 5 列），目的是避免为一次未定判的实验
重算数小时因子。T115 判定通过（2026-08-18 按族层面证据晋级为新基线）后，
公式必须回到生产路径，否则实盘的增量缓存更新只会产出 219/242 列，模型的
5 个新列会被 ``ml_factor_strategy._preload_factor_cache`` 静默填成常数 0.5。

本模块就是那次离线注入的**逐字移植**。任何公式改动都必须同时改这里和
``build_idxrel_cache.py``，否则训练缓存与实盘输入会分叉——
``tests/test_business_logic.py::IndexRelativeFactorTests`` 对拍这两条路径。

口径（与离线脚本完全一致，逐条对应）
------------------------------------
- 收益一律取 ``daily_data.pctChg / 100``（交易所口径，preclose 已含分红除权
  → **不需要**再做前复权；用复权后的 close.pct_change() 反而会双重调整）。
- 市场基准固定沪深300 ``sh.000300``；板块基准按代码前缀映射：
  60/68 → 上证综指，00 → 深成指，30 → 创业板指（2010-06 之前回退深成指），
  北交所 4/8/9 → 上证综指。
- 统计在**个股自身交易日历**上滚动（停牌日不参与），再 reindex 到目标日期
  并 ffill，最后按中性值补齐热身期（β→1.0，其余→0.0）。

PIT 安全性：全部是滚动窗口内的历史收益，无未来信息；指数当日收益与个股当日
收益同为 T 日收盘后可得，与其余量价因子的可见性一致。
"""
from __future__ import annotations

import os
import sqlite3
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

# 5 列的固定顺序（写缓存时的列序，对拍时按名取值，与顺序无关）
INDEX_REL_COLUMNS = [
    'idx_beta_60',
    'idx_corr_20',
    'idx_idio_vol_20',
    'idx_rs_20',
    'idx_rs_60',
]

# 热身期/缺失的中性填充值。β 的中性值是 1.0（等市场暴露），
# 相关/相对强弱/特异波动的中性值是 0.0。
INDEX_REL_FILLS: Dict[str, float] = {
    'idx_beta_60': 1.0,
    'idx_corr_20': 0.0,
    'idx_idio_vol_20': 0.0,
    'idx_rs_20': 0.0,
    'idx_rs_60': 0.0,
}

MARKET_INDEX = 'sh.000300'

# 进程内指数收益缓存。因子计算走 ProcessPoolExecutor，每个子进程各加载一次
# （30192 行，可忽略）；用 db_path 做键，避免测试库与生产库串味。
_INDEX_CACHE: Dict[str, Dict[str, pd.Series]] = {}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def load_index_returns(db_path: str) -> Optional[Dict[str, pd.Series]]:
    """
    读取指数日收益（date 字符串索引）。表不存在时返回 None ——
    调用方据此让整组列**缺席**而不是造中性假列：缺席会被下游的
    列完整性护栏抓住并硬失败，假列则会无声地退化成常数。
    """
    key = os.path.abspath(db_path)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key] or None

    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        if not _table_exists(conn, 'index_daily'):
            _INDEX_CACHE[key] = {}
            return None
        df = pd.read_sql_query(
            'SELECT code, date, pctChg FROM index_daily ORDER BY date ASC', conn)
    finally:
        conn.close()

    if df.empty:
        _INDEX_CACHE[key] = {}
        return None

    out: Dict[str, pd.Series] = {}
    for code, g in df.groupby('code'):
        out[code] = pd.to_numeric(
            g.set_index('date')['pctChg'], errors='coerce') / 100.0

    # 创业板指 2010-06 才成立，之前的日期用深成指顶上，避免 30 开头的票
    # 在 2010 年前整段没有板块基准。
    if 'sz.399006' in out and 'sz.399001' in out:
        gem = out['sz.399006'].reindex(out['sz.399001'].index)
        out['sz.399006_padded'] = gem.fillna(out['sz.399001'])

    if MARKET_INDEX not in out:
        _INDEX_CACHE[key] = {}
        return None

    _INDEX_CACHE[key] = out
    return out


def board_series(code: str, idx: Dict[str, pd.Series]) -> pd.Series:
    """板块基准映射。code 为不带交易所前缀的 6 位代码。"""
    if code.startswith(('60', '68')):
        return idx['sh.000001']
    if code.startswith('30'):
        return idx.get('sz.399006_padded', idx['sz.399001'])
    if code.startswith('00'):
        return idx['sz.399001']
    return idx['sh.000001']


def compute_features(r_s: pd.Series, r_m: pd.Series, r_b: pd.Series) -> pd.DataFrame:
    """
    在个股自身交易日历上算 5 列；r_* 均以 date 字符串为索引。

    与 ``build_idxrel_cache.compute_features`` 必须逐字等价。
    """
    df = pd.DataFrame({'rs': r_s})
    df['rm'] = r_m.reindex(df.index)
    df['rb'] = r_b.reindex(df.index)
    # 停牌/指数缺失日：市场收益缺失时该日不参与统计（rolling min_periods 兜底）
    cov = df['rs'].rolling(60, min_periods=40).cov(df['rm'])
    var = df['rm'].rolling(60, min_periods=40).var()
    beta = cov / var.replace(0.0, np.nan)
    resid = df['rs'] - beta * df['rm']
    out = pd.DataFrame(index=df.index)
    out['idx_beta_60'] = beta
    out['idx_corr_20'] = df['rs'].rolling(20, min_periods=15).corr(df['rm'])
    out['idx_idio_vol_20'] = resid.rolling(20, min_periods=15).std()
    rel = df['rs'] - df['rb']
    out['idx_rs_20'] = rel.rolling(20, min_periods=15).sum()
    out['idx_rs_60'] = rel.rolling(60, min_periods=40).sum()
    return out.replace([np.inf, -np.inf], np.nan)


class IndexRelativeFactors:
    """指数相对因子计算器。与其余因子子模块一样，由综合计算器持有。"""

    @staticmethod
    def calculate(code: str,
                  dates: Sequence,
                  db_path: str) -> pd.DataFrame:
        """
        返回 index 与 ``dates`` 等长的 5 列 DataFrame（float32，无 NaN）。

        **收益序列取自该股在 daily_data 里的全量历史，而不是传入的行情窗口。**
        增量更新只传最近 300 行，若在窗口上滚动，头部 60 行的 β/相对强弱会
        退化成热身期中性值，与全量重算出来的缓存对不上——同一日期在两次运行里
        取到不同的值，是最难查的那类漂移。多一次小查询换取窗口无关性，值。

        表缺失（未跑 update_daily_data.py）时返回空 DataFrame，
        由调用方决定是让列缺席还是报错，本函数不静默造列。
        """
        n = len(dates)
        idx = load_index_returns(db_path)
        if idx is None or n == 0:
            return pd.DataFrame()

        date_keys = np.array([str(d)[:10] for d in dates], dtype=object)

        conn = sqlite3.connect(db_path, timeout=60.0)
        try:
            d = pd.read_sql_query(
                'SELECT date, pctChg FROM daily_data WHERE code=? ORDER BY date ASC',
                conn, params=(code,))
        finally:
            conn.close()

        if d.empty:
            feats = pd.DataFrame(index=date_keys, columns=INDEX_REL_COLUMNS,
                                 dtype=float)
        else:
            r_s = pd.to_numeric(d.set_index('date')['pctChg'],
                                errors='coerce') / 100.0
            feats = compute_features(r_s, idx[MARKET_INDEX],
                                     board_series(code, idx))
            feats = feats.reindex(date_keys).ffill()

        out = pd.DataFrame(index=pd.RangeIndex(n))
        for col, fill in INDEX_REL_FILLS.items():
            out[col] = pd.to_numeric(feats[col], errors='coerce') \
                .fillna(fill).astype(np.float32).values
        return out
