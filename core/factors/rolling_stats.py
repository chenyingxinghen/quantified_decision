"""
滚动统计原子公式（跨因子族共享）
================================

为什么单独一个模块
------------------
``rolling_slope`` 同时被三处用到：市场敏感度族的形态派生（``msens_*_slope5_corr``）、
换手率时序族（``turn_slope5``）、以及离线诊断脚本。它最早在
``scripts/inject_market_sensitivity_cache.py`` 与 ``scripts/inject_turnover_value_cache.py``
里各抄了一份——两份逐字相同，但**任何一侧改动都会让缓存与生产路径静默分叉**，
而这类分叉在数值上表现为"同一日期两次运行取到不同的值"，是最难查的那类漂移。

因此把它提到共享层：生产因子模块与注入脚本都从这里 import，公式只有一处。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def rolling_slope(y: pd.Series, L: int) -> pd.Series:
    """序列 ``y`` 最近 ``L`` 期的滚动 OLS 斜率（对时间 t=0..L-1 回归）。

    - 窗口内有效样本数不足 ``max(3, L // 2)`` 时输出 NaN（热身/缺失由调用方按
      中性值补齐），**不做插值**。
    - ``np.nansum`` 口径：窗口内的 NaN 按 0 计入分子求和。这不是最严谨的缺失
      处理，但它是原离线注入脚本落进 14GB 缓存的既有口径，**必须逐位保留**，
      否则已物化的 ``msens_*_slope*`` / ``turn_slope5`` 全部作废。
    """
    arr = y.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n < L:
        return pd.Series(out, index=y.index)
    w = sliding_window_view(arr, L)            # (n-L+1, L)
    t = np.arange(L, dtype=float)
    St = t.sum()
    Stt = (t * t).sum()
    D = L * Stt - St * St
    valid = np.sum(~np.isnan(w), axis=1) >= max(3, L // 2)
    sy = np.nansum(w, axis=1)
    sty = np.nansum(w * t, axis=1)
    slope = (L * sty - St * sy) / D
    out[L - 1:] = np.where(valid, slope, np.nan)
    return pd.Series(out, index=y.index)


def align_to_dates(feats: pd.DataFrame, dates, columns, fills) -> pd.DataFrame:
    """把「个股自身日历上算好的特征」对齐到目标日期序列。

    统一 ``reindex → ffill → 中性填充 → float32`` 四步，是 ``idx_*`` / ``msens_*`` /
    ``turn_*`` 三族共用的收尾口径：

    - ``reindex`` 用日期字符串（截到 10 位）做键，与 parquet 里的 ``date`` 列同型；
    - ``ffill`` 覆盖停牌日（个股日历缺该日，取最近一次有效值）；
    - 剩余 NaN（热身期、整列缺源数据）按 ``fills`` 填中性值；
    - 输出 ``RangeIndex``，长度恒等于 ``len(dates)``，便于按位置拼回因子表。

    参数
    ----
    feats: 以日期字符串为索引的特征表，列可以缺（缺列按中性值整列补齐）
    dates: 目标日期序列
    columns: 输出列名与顺序（**契约**，不随 feats 实际列变化）
    fills: ``{列名: 中性值}``；缺键按 0.0
    """
    n = len(dates)
    date_keys = np.array([str(d)[:10] for d in dates], dtype=object)
    if feats is None or feats.empty:
        aligned = pd.DataFrame(index=date_keys, columns=list(columns), dtype=float)
    else:
        aligned = feats.reindex(date_keys).ffill()

    out = pd.DataFrame(index=pd.RangeIndex(n))
    for col in columns:
        fill = float(fills.get(col, 0.0)) if fills else 0.0
        if col in aligned.columns:
            series = pd.to_numeric(aligned[col], errors='coerce')
        else:
            series = pd.Series(np.nan, index=aligned.index)
        out[col] = series.fillna(fill).astype(np.float32).values
    return out
