"""
多时间尺度市场状态（Regime）特征
==================================

动机
----
全市场情绪列（up_ratio / breadth_ma20 / mean_return ...）当日**全市场同值**，
在日截面上方差为 0，作为个股特征进入 ranker 毫无区分度，因此在
``train_ml_model._RAW_MARKET_SENTIMENT_COLS`` 中被显式剔除。
代价是"中长期时代特征"在日频训练里彻底失声，只能靠
``feature_engineering.py`` 里手工的 ``{mkt}_regime_{stock}`` 乘积项间接注入 ——
特征数膨胀、编码低效、树仍难学到"什么行情下哪类因子有效"。

本模块的定位
------------
把市场状态从"个股特征矩阵 X 的一列"升级为**独立的上下文矩阵 M**：
- ``X[n_samples, n_factors]``：个股截面因子（保持不变）
- ``M[n_samples, d_m]``：该样本所属交易日的市场状态向量

M 不进入截面排序，而是喂给门控网络（RegimeGate），由门控输出因子族权重去
调节个股得分。这样"截面零方差"不再是问题 —— 门控本来就是逐日（而非逐股）生效的。

防泄露
------
1. 所有滚动窗口只使用 ``t`` 及之前的数据（pandas rolling 默认右闭）。
2. 归一化用**滚动历史分位**（``rolling(W).rank(pct=True)``）而非全样本 z-score，
   避免用未来分布信息标准化历史。
3. ``lag_days`` 参数可把整张表整体后移 N 天，用于验证"当日情绪是否算前视"。
   默认 0，与现有生产管线（同日情绪列）保持一致。
"""

from __future__ import annotations

import os
import sqlite3
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from config import DATABASE_PATH
except Exception:  # pragma: no cover - 允许脱离项目环境单测
    DATABASE_PATH = ''


# 归一化滚动窗口：3 年 ≈ 750 个交易日，覆盖一轮完整牛熊
_RANK_WINDOW = 750
_RANK_MIN_PERIODS = 120


# ---------------------------------------------------------------------------
# 原始数据读取
# ---------------------------------------------------------------------------

def load_market_sentiment(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    从 ``stock_meta.db`` 的 ``market_sentiment`` 表读取全市场逐日情绪指标。

    返回索引为 ``DatetimeIndex``、按日期升序的 DataFrame；表不存在时返回空表。
    """
    db_path = db_path or DATABASE_PATH
    meta_db = os.path.join(os.path.dirname(db_path), 'stock_meta.db')
    if not os.path.exists(meta_db):
        return pd.DataFrame()

    conn = sqlite3.connect(meta_db, timeout=30)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_sentiment'")
        if not cur.fetchone():
            return pd.DataFrame()
        df = pd.read_sql_query("SELECT * FROM market_sentiment ORDER BY date ASC", conn)
    finally:
        conn.close()

    if df.empty:
        return df

    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['date'], keep='last').set_index('date').sort_index()
    return df.astype(np.float64)


# ---------------------------------------------------------------------------
# 多时间尺度派生
# ---------------------------------------------------------------------------

def _safe(df: pd.DataFrame, col: str) -> pd.Series:
    """取列；不存在时返回全 0 序列（保证列结构稳定）。"""
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce')
    return pd.Series(0.0, index=df.index, dtype=np.float64)


def _build_raw_regime(sent: pd.DataFrame) -> pd.DataFrame:
    """
    由逐日情绪构造多时间尺度的**原始**（未归一化）市场状态列。

    覆盖四个语义轴：
      breadth（广度）/ trend（趋势）/ vol（波动）/ flow（资金与情绪强度）
    每个轴都给出短(5d) - 中(20d) - 长(60d~250d) 三档，让门控自己挑时间尺度。
    """
    out: Dict[str, pd.Series] = {}

    up = _safe(sent, 'up_ratio')
    strong_up = _safe(sent, 'strong_up_ratio')
    down = _safe(sent, 'down_ratio')
    lu = _safe(sent, 'limit_up_ratio')
    ld = _safe(sent, 'limit_down_ratio')
    mret = _safe(sent, 'mean_return')
    tvol = _safe(sent, 'total_volume')
    advol = _safe(sent, 'adv_vol_ratio')
    breadth = _safe(sent, 'breadth_ma20')

    # --- 合成市场指数：等权全市场日均收益累乘 ---
    idx = (1.0 + mret.fillna(0.0)).cumprod()

    # ---- 轴 1: 广度（多少股票在中期均线上方）----
    out['bd_level'] = breadth
    out['bd_ma5'] = breadth.rolling(5, min_periods=2).mean()
    out['bd_ma20'] = breadth.rolling(20, min_periods=5).mean()
    out['bd_ma60'] = breadth.rolling(60, min_periods=15).mean()
    out['bd_chg20'] = breadth - breadth.shift(20)
    # 广度背离：短期广度 vs 长期广度（顶背离/底背离的粗代理）
    out['bd_div'] = out['bd_ma5'] - out['bd_ma60']

    # ---- 轴 2: 趋势（合成指数的多尺度动量与位置）----
    for n in (5, 20, 60, 120):
        out[f'trend_ret{n}'] = idx.pct_change(n)
    out['trend_ma20_dev'] = idx / idx.rolling(20, min_periods=5).mean() - 1.0
    out['trend_ma60_dev'] = idx / idx.rolling(60, min_periods=15).mean() - 1.0
    # 距 250 日高点的回撤（<=0），刻画"熊市有多深"
    out['trend_dd250'] = idx / idx.rolling(250, min_periods=60).max() - 1.0
    # 距 250 日低点的涨幅（>=0），刻画"反弹走了多远"
    out['trend_up250'] = idx / idx.rolling(250, min_periods=60).min() - 1.0
    # MA20 斜率（20 日均线自身的 20 日变化率）
    ma20 = idx.rolling(20, min_periods=5).mean()
    out['trend_ma20_slope'] = ma20.pct_change(20)

    # ---- 轴 3: 波动（regime 切换最关键的维度）----
    vol20 = mret.rolling(20, min_periods=5).std()
    vol60 = mret.rolling(60, min_periods=15).std()
    out['vol_20'] = vol20
    out['vol_60'] = vol60
    # 波动扩张比：>1 表示波动正在放大（危险信号）
    out['vol_expand'] = vol20 / vol60.replace(0, np.nan)
    # 下行半波动：只统计负收益日，区分"上涨的高波动"与"下跌的高波动"
    neg = mret.where(mret < 0, 0.0)
    out['vol_down20'] = neg.rolling(20, min_periods=5).std()

    # ---- 轴 4: 资金与情绪强度 ----
    out['flow_up5'] = up.rolling(5, min_periods=2).mean()
    out['flow_up20'] = up.rolling(20, min_periods=5).mean()
    out['flow_up60'] = up.rolling(60, min_periods=15).mean()
    out['flow_strongup20'] = strong_up.rolling(20, min_periods=5).mean()
    out['flow_down20'] = down.rolling(20, min_periods=5).mean()
    # 涨跌停温度：投机情绪的直接读数
    out['flow_limitup20'] = lu.rolling(20, min_periods=5).mean()
    out['flow_limitdown20'] = ld.rolling(20, min_periods=5).mean()
    out['flow_limit_net'] = (lu - ld).rolling(20, min_periods=5).mean()
    out['flow_advvol20'] = advol.rolling(20, min_periods=5).mean()
    # 成交量的短长期比（放量/缩量）
    v5 = tvol.rolling(5, min_periods=2).mean()
    v60 = tvol.rolling(60, min_periods=15).mean()
    out['flow_vol_ratio'] = v5 / v60.replace(0, np.nan)

    return pd.DataFrame(out, index=sent.index)


def _rolling_pct_normalize(df: pd.DataFrame,
                           window: int = _RANK_WINDOW,
                           min_periods: int = _RANK_MIN_PERIODS) -> pd.DataFrame:
    """
    滚动历史分位归一化：把每列映射到 ``[-1, 1]``。

    ``rolling(window).rank(pct=True)`` 只看窗口内（含当日、不含未来）的历史值，
    因此不存在前视偏差。热身期（不足 ``min_periods``）填 0（= 中性）。
    """
    ranked = df.rolling(window, min_periods=min_periods).rank(pct=True)
    normed = (ranked * 2.0 - 1.0).astype(np.float32)
    return normed.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _load_macro_m1m2_gap(db_path: Optional[str],
                         index: pd.DatetimeIndex) -> Optional[pd.Series]:
    """
    M1-M2 同比剪刀差（T116 标量门终检的外生条件化信号）。

    PIT 纪律：statMonth 的读数按**次月 15 日**才可见（央行金融统计数据实际
    发布在次月 10~15 日，取保守端），可见日之前一律用上一期。月度序列在
    日频索引上前向填充成阶梯；归一化与其余 regime 列共用滚动分位管线。

    表不存在（未跑 update_daily_data.py，或跑时带了 --skip-money）时返回 None ——
    列整体缺席。scalar 门控按列名解析、缺列硬失败，不会静默退化。
    """
    db_path = db_path or DATABASE_PATH
    meta_db = os.path.join(os.path.dirname(db_path), 'stock_meta.db')
    if not os.path.exists(meta_db):
        return None
    conn = sqlite3.connect(meta_db, timeout=30)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='macro_money_supply'")
        if not cur.fetchone():
            return None
        df = pd.read_sql_query(
            'SELECT statMonth, m1YOY, m2YOY FROM macro_money_supply '
            'WHERE m1YOY IS NOT NULL AND m2YOY IS NOT NULL '
            'ORDER BY statMonth ASC', conn)
    finally:
        conn.close()
    if df.empty:
        return None
    stat = pd.to_datetime(df['statMonth'], format='%Y-%m')
    visible = stat + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
    gap = pd.Series((df['m1YOY'] - df['m2YOY']).to_numpy(dtype=np.float64),
                    index=pd.DatetimeIndex(visible)).sort_index()
    gap = gap[~gap.index.duplicated(keep='last')]
    return gap.reindex(gap.index.union(index)).ffill().reindex(index)


# ---------------------------------------------------------------------------
# 对外主接口
# ---------------------------------------------------------------------------

def build_regime_matrix(db_path: Optional[str] = None,
                        dates: Optional[Sequence] = None,
                        normalize: bool = True,
                        lag_days: int = 0,
                        return_raw: bool = False) -> pd.DataFrame:
    """
    构造多时间尺度市场状态矩阵 ``M``。

    参数
    ----
    db_path : 主库路径（用于定位同目录的 ``stock_meta.db``）
    dates   : 若提供，则只返回这些交易日的行（缺失日 ffill 后再补 0）
    normalize : 是否做滚动历史分位归一化（默认 True，门控输入建议开启）
    lag_days  : 整体后移 N 个交易日。默认 0（与生产管线同日口径一致）；
                设为 1 可做"严格 T-1 信息"敏感性检验
    return_raw : True 时返回未归一化的原始列（画图/体检用）

    返回
    ----
    以 ``DatetimeIndex`` 为索引的 DataFrame，列即市场状态向量各维度。
    """
    sent = load_market_sentiment(db_path)
    if sent.empty:
        return pd.DataFrame()

    raw = _build_raw_regime(sent)
    # T116：外生宏观列（M1-M2 剪刀差，已按发布日 PIT 对齐）。放在归一化之前，
    # 与内生列共用同一滚动分位管线；表未落库时列缺席（不填 0 假列）。
    _gap = _load_macro_m1m2_gap(db_path, raw.index)
    if _gap is not None:
        raw['macro_m1m2_gap'] = _gap
    if lag_days > 0:
        raw = raw.shift(lag_days)

    if return_raw or not normalize:
        mat = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).astype(np.float32)
    else:
        mat = _rolling_pct_normalize(raw)

    if dates is not None:
        want = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates)).unique())).sort_values()
        mat = mat.reindex(mat.index.union(want)).ffill().reindex(want).fillna(0.0)

    return mat.astype(np.float32)


def get_regime_feature_names(db_path: Optional[str] = None) -> List[str]:
    """返回市场状态向量的列名顺序（供模型保存/加载时对齐）。"""
    mat = build_regime_matrix(db_path)
    return list(mat.columns)


def align_regime_to_samples(sample_dates: np.ndarray,
                            regime_matrix: pd.DataFrame) -> np.ndarray:
    """
    把逐日的市场状态矩阵展开成逐样本的 ``M[n_samples, d_m]``。

    ``sample_dates`` 与训练集 ``X`` 行一一对应（同一交易日的多只股票共享一行状态）。
    未覆盖的日期用最近的历史值前向填充，仍缺失则填 0（中性）。
    """
    if regime_matrix is None or regime_matrix.empty:
        return np.zeros((len(sample_dates), 0), dtype=np.float32)

    s = pd.to_datetime(pd.Series(sample_dates))
    uniq = pd.DatetimeIndex(s.unique()).sort_values()
    aligned = regime_matrix.reindex(regime_matrix.index.union(uniq)).ffill().reindex(uniq).fillna(0.0)

    lookup = {d: i for i, d in enumerate(uniq)}
    row_idx = s.map(lookup).to_numpy()
    return aligned.to_numpy(dtype=np.float32)[row_idx]


if __name__ == '__main__':  # 体检：直接运行本文件即可查看矩阵概况
    import sys
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from config import DATABASE_PATH as _DB  # noqa: E402
    DATABASE_PATH = _DB

    m = build_regime_matrix(_DB)
    print(f"regime matrix: {m.shape[0]} 交易日 × {m.shape[1]} 维")
    print(f"日期范围: {m.index.min().date()} ~ {m.index.max().date()}")
    print(m.tail(3).T.to_string())
