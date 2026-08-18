"""涨跌停限额的**实测**查表（替代静态 `config.MARKET_LIMITS`）。

为什么需要它
------------
`MARKET_LIMITS` 是按代码前缀的静态字典（main 0.098 / gem_star 0.198 / bj 0.295 /
st 0.05），没有时间维度。但限额在历史上变过：创业板 2020-08-24 由 10% 改 20%、
主板 ST 2026-07-06 由 5% 改 10%（用户 2026-08-18 指出）。静态表用错限额会：

  1. **误判 T+1 一字涨停** ⇒ 样本剔除错误 ⇒ 训练/验证集构成本身就错
     （创业板 2020-08 前真实一字板是 +10%，代码按 19.8% 判 ⇒ 该删的没删；
      ST 现为 10% 而代码写 5% ⇒ 5% 就触发 ⇒ 误删能买的票）
  2. 污染 `scores` 标签的除法归一化（`f_returns_raw / limits`）
  3. 让 `market_type` 特征把「板块身份」与「当期限额」混为一谈

限额表由 `scripts/build_price_limit_history.py` 从行情**实测**得到（涨停价是
`round(preclose×(1+L), 2)` 这个精确值），规则变更自动跟随，不需要维护规则表。

开关
----
`TrainingConfig.USE_EMPIRICAL_PRICE_LIMITS`，**默认 False** —— 打开会改变样本
剔除结果，使新结果与 T113/T115 基线不可配对。等下次本就要建新基线时再开。
表缺失时自动回退静态值并给出一次性告警（不静默）。
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd

from config import DATABASE_PATH, MARKET_LIMITS

_CACHE: Optional[dict] = None
_WARNED = False


def board_of(code: str) -> str:
    """与 build_price_limit_history.py 保持一致的板块划分。"""
    if code.startswith(('300', '301')):
        return 'gem'
    if code.startswith(('688', '689')):
        return 'star'
    if code.startswith(('43', '83', '87', '88')):
        return 'bj'
    return 'main'


def _static_for(board: str, is_st: bool) -> float:
    if board == 'gem' or board == 'star':
        return MARKET_LIMITS['gem_star']
    if board == 'bj':
        return MARKET_LIMITS['bj']
    return MARKET_LIMITS['st'] if is_st else MARKET_LIMITS['main']


def load_history(db_path: Optional[str] = None) -> dict:
    """{(board, is_st): Series(index=date_str, value=limit)}；表不存在时返回 {}。"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = db_path or DATABASE_PATH
    _CACHE = {}
    if not os.path.exists(path):
        return _CACHE
    conn = sqlite3.connect(path, timeout=60.0)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='price_limit_history'")
        if not cur.fetchone():
            return _CACHE
        df = pd.read_sql_query(
            'SELECT date, board, is_st, limit_pct FROM price_limit_history '
            'ORDER BY board, is_st, date', conn)
    finally:
        conn.close()
    for (bd, st), g in df.groupby(['board', 'is_st']):
        _CACHE[(bd, int(st))] = pd.Series(
            g['limit_pct'].to_numpy(dtype=np.float64),
            index=g['date'].astype(str).to_numpy())
    return _CACHE


def resolve(code: str, dates, is_st=None, db_path: Optional[str] = None) -> np.ndarray:
    """逐日限额 float32[n]。

    ``dates``：字符串日期序列；``is_st``：同长度 0/1 序列（None 视为全非 ST）。
    实测表缺该 (board, is_st) 组合的日期时，用**该组合最近的历史值**前向填充；
    整组缺失才回退静态值。
    """
    global _WARNED
    n = len(dates)
    board = board_of(code)
    st_arr = (np.zeros(n, dtype=np.int8) if is_st is None
              else np.asarray(pd.Series(is_st).fillna(0)).astype(np.int8))
    hist = load_history(db_path)
    if not hist:
        if not _WARNED:
            print('  [price_limits] 警告: price_limit_history 表不存在，回退静态 MARKET_LIMITS。'
                  '先跑 scripts/build_price_limit_history.py')
            _WARNED = True
        out = np.empty(n, dtype=np.float32)
        for st in (0, 1):
            m = st_arr == st
            if m.any():
                out[m] = _static_for(board, bool(st))
        return out

    d = pd.Index(pd.Series(dates).astype(str))
    out = np.empty(n, dtype=np.float32)
    for st in (0, 1):
        m = st_arr == st
        if not m.any():
            continue
        s = hist.get((board, st))
        if s is None or s.empty:
            # 该板块该 ST 状态从未观测到涨停（如 bj 库内无数据）→ 静态兜底
            out[m] = _static_for(board, bool(st))
            continue
        # reindex + ffill：限额是分段常数，取不晚于当日的最近一次观测
        aligned = s.reindex(s.index.union(d[m])).ffill().reindex(d[m])
        vals = aligned.to_numpy(dtype=np.float64)
        # 早于该组合首次观测的日期仍为 NaN → 静态兜底
        vals = np.where(np.isnan(vals), _static_for(board, bool(st)), vals)
        out[m] = vals.astype(np.float32)
    return out


def resolve_or_static(code: str, dates, is_st, enabled: bool,
                      db_path: Optional[str] = None) -> np.ndarray:
    """按开关选择实测表或静态值；静态分支与历史行为逐位一致。"""
    n = len(dates)
    if enabled:
        return resolve(code, dates, is_st, db_path)
    board = board_of(code)
    out = np.full(n, _static_for(board, False), dtype=np.float32)
    if is_st is not None and board == 'main':
        st_arr = np.asarray(pd.Series(is_st).fillna(0)).astype(np.int8)
        out[st_arr == 1] = MARKET_LIMITS['st']
    return out
