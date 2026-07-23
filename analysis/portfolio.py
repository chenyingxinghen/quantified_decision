"""
analysis/portfolio.py — 投资组合检验

覆盖指标:
  - 分组收益:       按模型得分十分位分组，各组平均收益与累计收益曲线
  - 多空组合:       头部组 − 尾部组 的每日收益与累计净值、Sharpe、最大回撤
  - 换手率:         多头组合（头部组）相邻交易日换仓比例时间序列
  - 交易成本检验:   扣除双边换手成本后的多空组合净收益
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _add_groups(df: pd.DataFrame, q: int) -> pd.DataFrame:
    """按交易日对得分做百分位分组（保证 q 组）。"""
    size = df.groupby("date")["score"].transform("size")
    rank = df.groupby("date")["score"].rank(method="first")
    df = df.copy()
    df["grp"] = (((rank - 1) / size * q).astype(int)).clip(0, q - 1)
    return df


def grouping_returns(
    scores: np.ndarray, returns: np.ndarray, dates: np.ndarray, q: int = 10
):
    """分组收益分析。

    返回 dict:
        group_mean_ret: 各组平均日收益 (Series, index 0..q-1)
        group_cum_ret:   各组累计净值 (DataFrame, index=date, columns=grp)
        ls_daily:        多空每日收益 (Series, index=date)
        ls_cum:          多空累计净值 (Series)
    """
    df = pd.DataFrame({
        "date": pd.Series(dates).astype(str).str[:10],
        "score": np.asarray(scores, dtype=float),
        "ret": np.asarray(returns, dtype=float),
    })
    df = _add_groups(df, q)
    grp_daily = df.groupby(["date", "grp"])["ret"].mean().unstack("grp").sort_index()
    group_mean_ret = grp_daily.mean()
    group_cum_ret = (1.0 + grp_daily.fillna(0.0)).cumprod()

    top, bot = q - 1, 0
    ls_daily = (grp_daily[top] - grp_daily[bot]).fillna(0.0)
    ls_cum = (1.0 + ls_daily).cumprod()
    return {
        "group_mean_ret": group_mean_ret,
        "group_cum_ret": group_cum_ret,
        "ls_daily": ls_daily,
        "ls_cum": ls_cum,
        "q": q,
        "top_group": top,
        "bottom_group": bot,
    }


def _long_book_turnover(df: pd.DataFrame, q: int) -> pd.Series:
    """多头组合（头部组）相邻交易日换仓比例。

    turnover_t = 1 − |A_t ∩ A_{t-1}| / |A_t|  （等权假设下 0.5·Σ|w_t−w_{t-1}|）
    """
    top = q - 1
    daily_top = (
        df[df["grp"] == top]
        .groupby("date")["code"].apply(lambda s: set(s))
        .sort_index()
    )
    turnover = {}
    prev = None
    for d, codes in daily_top.items():
        if prev is None or len(codes) == 0:
            turnover[d] = 0.0
        else:
            inter = len(codes & prev)
            turnover[d] = 1.0 - inter / len(codes) if len(codes) else 0.0
        prev = codes
    return pd.Series(turnover)


def turnover_analysis(
    scores: np.ndarray, returns: np.ndarray, dates: np.ndarray,
    codes: np.ndarray, q: int = 10,
) -> dict:
    """换手率分析（多头腿 + 双边）。"""
    df = pd.DataFrame({
        "date": pd.Series(dates).astype(str).str[:10],
        "score": np.asarray(scores, dtype=float),
        "ret": np.asarray(returns, dtype=float),
        "code": np.asarray(codes),
    })
    df = _add_groups(df, q)
    to_long = _long_book_turnover(df, q)
    # 双边换手约为多头腿的 2 倍（多空两头同时换仓）
    to_total = to_long * 2.0
    return {
        "turnover_long": to_long,
        "turnover_total": to_total,
        "mean_turnover_long": float(to_long.mean()),
        "mean_turnover_total": float(to_total.mean()),
    }


def _series_stats(s: pd.Series, periods_per_year: int = 252) -> dict:
    s = s.dropna()
    n = len(s)
    if n == 0:
        return {"n": 0}
    mean_d, std_d = s.mean(), s.std(ddof=1)
    cum = float((1.0 + s).prod() - 1.0)
    ann_ret = float((1.0 + cum) ** (periods_per_year / n) - 1.0) if n > 0 else float("nan")
    sharpe = float(mean_d / std_d * np.sqrt(periods_per_year)) if std_d > 0 else float("nan")
    # 最大回撤
    equity = (1.0 + s).cumprod()
    peak = equity.cummax()
    mdd = float((equity / peak - 1.0).min())
    return {
        "n_days": int(n),
        "total_return": cum,
        "ann_return": ann_ret,
        "ann_vol": float(std_d * np.sqrt(periods_per_year)),
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "daily_mean": float(mean_d),
    }


def transaction_cost_analysis(
    ls_daily: pd.Series, turnover_total: pd.Series,
    cost_per_trade: float = 0.001,
) -> dict:
    """扣除交易成本后的多空组合。

    net_ls_t = ls_t − turnover_total_t × cost_rate
    """
    aligned = pd.concat([ls_daily, turnover_total], axis=1, join="inner").dropna()
    aligned.columns = ["ls", "to"]
    net = aligned["ls"] - aligned["to"] * cost_per_trade
    gross_stats = _series_stats(ls_daily)
    net_stats = _series_stats(net)
    net_cum = (1.0 + net).cumprod()
    return {
        "cost_per_trade": cost_per_trade,
        "ls_daily": ls_daily,
        "net_daily": net,
        "net_cum": net_cum,
        "gross": gross_stats,
        "net": net_stats,
        "cost_drag_ann": float(gross_stats.get("ann_return", float("nan"))
                               - net_stats.get("ann_return", float("nan"))),
    }


def run_portfolio(
    scores: np.ndarray, returns: np.ndarray, dates: np.ndarray,
    codes: np.ndarray, q: int = 10, cost_per_trade: float = 0.001,
) -> dict:
    """端到端投资组合检验。"""
    grp = grouping_returns(scores, returns, dates, q=q)
    to = turnover_analysis(scores, returns, dates, codes, q=q)
    tcost = transaction_cost_analysis(grp["ls_daily"], to["turnover_total"],
                                      cost_per_trade=cost_per_trade)
    return {
        "grouping": grp,
        "turnover": to,
        "tcost": tcost,
        "group_stats": _series_stats(grp["ls_daily"]),
    }
