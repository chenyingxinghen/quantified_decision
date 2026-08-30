"""
换手率时序族 + 估值动量族（``turn_*`` / ``val_*``）—— 生产实现
=============================================================

来历
----
这 11 列最早由 ``scripts/inject_turnover_value_cache.py`` **离线注入**进因子缓存
（对每个 parquet 追加列），目的是避免为一次未定判的实验重算数小时因子。
与 ``index_relative_factors`` 走过的路完全一样：一旦公式只存在于注入脚本里，
实时计算器（``ComprehensiveFactorCalculator``）产出的列集就会比缓存少一截，
于是出现「树线 246 列 / 缓存 347 列」这种同一份缓存两套面板的分叉。

本模块是那次离线注入的**逐字移植**，注入脚本改为调用这里，公式只有一处。

口径
----
- 数据源 ``daily_data``：``close/pctChg/volume/turnover_rate/peTTM/pbMRQ``，
  全部是 T 日收盘后可得的当日值 → PIT 安全。
- 统计在**个股自身交易日历**上滚动（停牌日不在表里，天然不参与），再 reindex
  到目标日期并 ffill，最后热身期/缺失补 0.0。
- **窗口无关**：收益/换手/估值序列取自该股在 ``daily_data`` 的全量历史，而不是
  传入的行情窗口。增量更新只传最近 300 行，若在窗口上滚动，头部的
  ``turn_ma20`` / ``val_mom_pb20`` 会退化成热身值，与全量重算的缓存对不上。

中性填充
--------
全部 0.0。这一族没有"1.0 才中性"的量（对比 ``idx_beta_60``）：换手率水平、
估值动量、相关系数的中性值都是 0。
"""
from __future__ import annotations

import os
import sqlite3
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from core.factors.rolling_stats import align_to_dates, rolling_slope

# 11 列的固定顺序（写缓存时的列序；对拍按名取值，与顺序无关）
TURNOVER_VALUE_COLUMNS: List[str] = [
    # 估值动量
    'val_mom_pb20', 'val_mom_pe20', 'val_pe_turn_corr20', 'val_accel_pe5',
    # 换手率时序族
    'turn_ma5', 'turn_ma20', 'turn_std20', 'turn_chg20',
    'turn_slope5', 'turn_vol_corr20', 'turn_price_corr20',
]

# 中性填充值：全 0.0
TURNOVER_VALUE_FILLS: Dict[str, float] = {c: 0.0 for c in TURNOVER_VALUE_COLUMNS}

_SQL = ('SELECT date, close, pctChg, volume, turnover_rate, peTTM, pbMRQ '
        'FROM daily_data WHERE code=? ORDER BY date ASC')


def compute_features(d: pd.DataFrame) -> pd.DataFrame:
    """在个股自身日历上算 11 列。``d`` 需含 ``_SQL`` 的全部列，索引任意。

    与原 ``inject_turnover_value_cache.compute_turn_value`` 的计算段逐字等价。
    """
    d = d.copy()
    d['date'] = d['date'].astype(str)
    d = d.drop_duplicates('date').set_index('date').sort_index()

    pc = pd.to_numeric(d['pctChg'], errors='coerce').astype(float)
    vol = d['volume'].astype(float)
    to = pd.to_numeric(d['turnover_rate'], errors='coerce')
    pe = pd.to_numeric(d['peTTM'], errors='coerce')
    pb = pd.to_numeric(d['pbMRQ'], errors='coerce')

    with np.errstate(invalid='ignore', divide='ignore'):
        out = pd.DataFrame(index=d.index)
        out['val_mom_pb20'] = pb.pct_change(20)
        out['val_mom_pe20'] = pe.pct_change(20)
        out['val_pe_turn_corr20'] = pe.rolling(20, min_periods=10).corr(to)
        out['val_accel_pe5'] = pe.pct_change(5).diff()
        out['turn_ma5'] = to.rolling(5, min_periods=3).mean()
        out['turn_ma20'] = to.rolling(20, min_periods=10).mean()
        out['turn_std20'] = to.rolling(20, min_periods=10).std()
        out['turn_chg20'] = to.pct_change(20)
        out['turn_slope5'] = rolling_slope(to, 5)
        out['turn_vol_corr20'] = to.rolling(20, min_periods=10).corr(vol)
        out['turn_price_corr20'] = to.rolling(20, min_periods=10).corr(pc)
    return out[TURNOVER_VALUE_COLUMNS]


def load_stock_frame(code: str, db_path: str) -> pd.DataFrame:
    """读该股在 ``daily_data`` 的全量历史（窗口无关性的来源）。"""
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        return pd.read_sql_query(_SQL, conn, params=(code,))
    finally:
        conn.close()


class TurnoverValueFactors:
    """换手率/估值动量计算器。与其余因子子模块一样，由综合计算器持有。"""

    @staticmethod
    def calculate(code: str, dates: Sequence, db_path: str) -> pd.DataFrame:
        """返回 index 与 ``dates`` 等长的 11 列 DataFrame（float32，无 NaN）。

        该股在 ``daily_data`` 无记录时返回**全中性列**（而不是空表）：与
        ``idx_*`` 不同，这一族的数据源就是个股自身日线，没有"整表缺失"这种
        全局失效场景——``daily_data`` 若整表没了，上游连行情都读不到。因此
        单股缺记录属于个体情况，按中性值补齐即可，不需要让列缺席去触发护栏。
        """
        if len(dates) == 0:
            return pd.DataFrame()
        d = load_stock_frame(code, db_path)
        feats = pd.DataFrame() if d.empty else compute_features(d)
        return align_to_dates(feats, dates,
                              TURNOVER_VALUE_COLUMNS, TURNOVER_VALUE_FILLS)
