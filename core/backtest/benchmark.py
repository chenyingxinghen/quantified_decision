"""
回测基准与超额收益口径

背景：此前回测只输出策略自身的绝对收益，无法区分 alpha 与 beta。
同一模型在 2022-09→2024-08 为 −41%、在 2024-08→2026-08 为 +23%，
在没有基准对照的情况下无法判断是「信号失效」还是「多头 beta 在熊市的暴露」。

本模块提供：
1. `build_equal_weight_benchmark` —— 从 daily_data 构造全市场等权日收益序列。
   对「从全市场挑 N 只、等权持有」的策略而言，全市场等权是最贴合的 beta 口径
   （市值加权指数会引入策略本身并不承担的规模暴露）。
2. `compute_relative_metrics` —— 给定策略资金曲线与基准日收益，输出
   年化收益/波动/夏普、beta、年化 alpha、跟踪误差、信息比率、相对回撤。
"""

from __future__ import annotations

import os
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _resolve_db_path(db_path: Optional[str] = None) -> str:
    if db_path:
        return db_path
    import sys

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    from config.baostock_config import DATABASE_PATH  # noqa: WPS433

    return DATABASE_PATH


def build_equal_weight_benchmark(start_date: str,
                                 end_date: str,
                                 exclude_st: bool = True,
                                 min_stocks: int = 100,
                                 db_path: Optional[str] = None) -> pd.Series:
    """构造全市场等权日收益序列（decimal，非百分比）。

    只取 tradestatus=1（正常交易）的个股，按日做横截面等权平均。
    停牌股当日不计入——这与「策略持有停牌股时净值不变」的处理一致。
    """
    path = _resolve_db_path(db_path)
    where = ["date >= ?", "date <= ?", "tradestatus = 1", "pctChg IS NOT NULL"]
    params: List[object] = [start_date, end_date]
    if exclude_st:
        where.append("COALESCE(is_st, 0) = 0")
    sql = (
        "SELECT date, AVG(pctChg) AS ret, COUNT(*) AS n "
        "FROM daily_data WHERE " + " AND ".join(where) + " GROUP BY date ORDER BY date"
    )
    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        return pd.Series(dtype=float)

    df = df[df['n'] >= min_stocks]
    s = pd.Series(df['ret'].to_numpy() / 100.0, index=df['date'].astype(str))
    s.name = 'benchmark_ret'
    return s


def build_buy_hold_equal_weight_benchmark(start_date: str,
                                          end_date: str,
                                          exclude_st: bool = True,
                                          min_history_ratio: float = 0.5,
                                          db_path: Optional[str] = None) -> pd.Series:
    """构造「期初等权买入并持有」的日收益序列（decimal）。

    为什么必须与日频再平衡口径区分开：
      日频再平衡等权组合每天把全市场 ~5000 只股票拉回等权，隐含每天 100% 换手。
      这个动作会系统性地卖出涨的、买入跌的，机械收割日频反转与买卖价差噪声，
      产生一块**不可实现的再平衡溢价**（本项目实测 2024-08→2026-08 达 +27pp）。
      更要命的是它在零成本假设下才成立：按真实 0.21% 往返成本，
      每天全额换手两年要付掉 100pp 以上，根本不是可投资标的。

    因此：
      - 日频再平衡等权 = 横截面平均收益的统计量，用于算 beta / 相关性；
      - 期初等权买入持有 = **可投资的 beta 基准**，用于判断策略是否真的创造价值。
    判断「策略有没有 alpha」必须用后者。
    """
    path = _resolve_db_path(db_path)
    where = ["date >= ?", "date <= ?", "pctChg IS NOT NULL"]
    params: List[object] = [start_date, end_date]
    if exclude_st:
        where.append("COALESCE(is_st, 0) = 0")
    sql = (
        "SELECT date, code, pctChg FROM daily_data WHERE "
        + " AND ".join(where) + " ORDER BY date"
    )
    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        return pd.Series(dtype=float)

    df['date'] = df['date'].astype(str)
    all_dates = np.sort(df['date'].unique())
    if len(all_dates) < 2:
        return pd.Series(dtype=float)

    # 只保留期初就存在的股票（避免把后来上市的新股算进「期初买入」）
    first_day_codes = set(df.loc[df['date'] == all_dates[0], 'code'])
    # 同时要求覆盖率达标，剔除中途长期停牌/退市造成的伪持仓
    cov = df.groupby('code')['date'].nunique()
    eligible = {c for c in first_day_codes if cov.get(c, 0) >= len(all_dates) * min_history_ratio}
    if len(eligible) < 100:
        eligible = first_day_codes

    sub = df[df['code'].isin(eligible)]
    wide = sub.pivot_table(index='date', columns='code', values='pctChg', aggfunc='last')
    wide = wide.reindex(all_dates)
    # 停牌日按 0 收益处理（净值不变），与策略端持有停牌股的处理一致
    wide = wide.fillna(0.0) / 100.0
    cum = (1.0 + wide).cumprod()
    port = cum.mean(axis=1)          # 期初等权 -> 权重随后自然漂移
    ret = port.pct_change().dropna()
    ret.name = 'benchmark_ret'
    return ret


def _equity_to_returns(equity_curve: Sequence[Tuple[str, float]]) -> pd.Series:
    """资金曲线 -> 日收益序列（decimal）。"""
    if not equity_curve:
        return pd.Series(dtype=float)
    dates = [str(d) for d, _ in equity_curve]
    vals = np.asarray([float(v) for _, v in equity_curve], dtype=float)
    eq = pd.Series(vals, index=dates).sort_index()
    eq = eq[~eq.index.duplicated(keep='last')]
    ret = eq.pct_change().dropna()
    ret.name = 'strategy_ret'
    return ret


def _max_drawdown(cum: pd.Series) -> float:
    if len(cum) == 0:
        return 0.0
    running = cum.cummax()
    return float(((cum - running) / running).min() * 100.0)


def _ann_stats(ret: pd.Series) -> Dict[str, float]:
    if len(ret) == 0:
        return {'total_return_pct': 0.0, 'annual_return_pct': 0.0,
                'annual_vol_pct': 0.0, 'sharpe': 0.0, 'max_drawdown_pct': 0.0}
    cum = (1.0 + ret).cumprod()
    total = float(cum.iloc[-1] - 1.0)
    years = len(ret) / TRADING_DAYS
    ann = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 and total > -1 else float('nan')
    vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if ret.std(ddof=1) > 0 else 0.0
    return {
        'total_return_pct': total * 100.0,
        'annual_return_pct': float(ann) * 100.0,
        'annual_vol_pct': vol * 100.0,
        'sharpe': sharpe,
        'max_drawdown_pct': _max_drawdown(cum),
    }


def compute_relative_metrics(equity_curve: Sequence[Tuple[str, float]],
                             bench_ret: pd.Series) -> Dict[str, object]:
    """对齐策略与基准日收益，输出绝对 + 相对口径指标。

    关键区分：
      - `alpha_annual_pct` 由 CAPM 式回归 r_s = α + β·r_b 得到，剔除 beta 暴露；
      - `excess_*` 为 α 口径之外的简单主动收益（r_s − r_b），信息比率基于它计算。
    """
    sret = _equity_to_returns(equity_curve)
    if len(sret) == 0 or len(bench_ret) == 0:
        return {'available': False, 'reason': '资金曲线或基准序列为空'}

    df = pd.concat([sret, bench_ret.rename('benchmark_ret')], axis=1, join='inner').dropna()
    if len(df) < 20:
        return {'available': False, 'reason': f'对齐后仅 {len(df)} 个交易日，样本不足'}

    rs = df['strategy_ret']
    rb = df['benchmark_ret']

    s_stats = _ann_stats(rs)
    b_stats = _ann_stats(rb)

    # CAPM 回归（无风险利率按 0 处理，与既有夏普口径一致）
    var_b = float(rb.var(ddof=1))
    beta = float(rs.cov(rb) / var_b) if var_b > 0 else float('nan')
    alpha_daily = float(rs.mean() - beta * rb.mean()) if np.isfinite(beta) else float('nan')
    alpha_annual = alpha_daily * TRADING_DAYS if np.isfinite(alpha_daily) else float('nan')
    resid = rs - (alpha_daily + beta * rb) if np.isfinite(beta) else rs * np.nan
    resid_vol = float(resid.std(ddof=1) * np.sqrt(TRADING_DAYS)) if np.isfinite(beta) else float('nan')
    appraisal = alpha_annual / resid_vol if np.isfinite(resid_vol) and resid_vol > 0 else float('nan')
    corr = float(rs.corr(rb))

    # 主动收益（等权多空口径：策略 − 基准）
    active = rs - rb
    te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ir = float(active.mean() * TRADING_DAYS / te) if te > 0 else float('nan')

    rel_cum = (1.0 + rs).cumprod() / (1.0 + rb).cumprod()
    rel_total = float(rel_cum.iloc[-1] - 1.0)
    rel_dd = _max_drawdown(rel_cum)

    return {
        'available': True,
        'n_days': int(len(df)),
        'date_range': [str(df.index[0]), str(df.index[-1])],
        'strategy': s_stats,
        'benchmark': b_stats,
        'beta': beta,
        'corr': corr,
        'alpha_annual_pct': alpha_annual * 100.0 if np.isfinite(alpha_annual) else None,
        'residual_vol_pct': resid_vol * 100.0 if np.isfinite(resid_vol) else None,
        'appraisal_ratio': appraisal if np.isfinite(appraisal) else None,
        'excess_total_pct': (s_stats['total_return_pct'] - b_stats['total_return_pct']),
        'relative_total_pct': rel_total * 100.0,
        'relative_max_drawdown_pct': rel_dd,
        'tracking_error_pct': te * 100.0,
        'information_ratio': ir if np.isfinite(ir) else None,
    }


def print_relative_summary(rel: Dict[str, object], bench_name: str = '全市场等权') -> None:
    if not rel.get('available'):
        print(f"\n【基准对照】不可用：{rel.get('reason')}")
        return
    s = rel['strategy']
    b = rel['benchmark']
    print("\n" + "=" * 80)
    print(f"【基准对照 · {bench_name}】{rel['date_range'][0]} 至 {rel['date_range'][1]}"
          f"（{rel['n_days']} 个交易日）")
    print("=" * 80)
    print(f"{'':<14}{'总收益':>10}{'年化':>10}{'年化波动':>10}{'夏普':>8}{'最大回撤':>10}")
    for label, d in (('策略', s), (bench_name, b)):
        print(f"{label:<14}{d['total_return_pct']:>9.2f}%{d['annual_return_pct']:>9.2f}%"
              f"{d['annual_vol_pct']:>9.2f}%{d['sharpe']:>8.2f}{d['max_drawdown_pct']:>9.2f}%")
    print("-" * 80)
    print(f"  beta            : {rel['beta']:.3f}   (与基准相关系数 {rel['corr']:.3f})")
    ap = rel['alpha_annual_pct']
    print(f"  年化 alpha      : {ap:.2f}%" if ap is not None else "  年化 alpha      : n/a")
    ar = rel['appraisal_ratio']
    print(f"  信息比(残差口径): {ar:.3f}" if ar is not None else "  信息比(残差口径): n/a")
    print(f"  主动收益(策略−基准): {rel['excess_total_pct']:+.2f}%   "
          f"几何相对收益 {rel['relative_total_pct']:+.2f}%")
    print(f"  跟踪误差        : {rel['tracking_error_pct']:.2f}%   "
          f"信息比(主动口径) {rel['information_ratio']:.3f}"
          if rel['information_ratio'] is not None else
          f"  跟踪误差        : {rel['tracking_error_pct']:.2f}%")
    print(f"  相对净值最大回撤: {rel['relative_max_drawdown_pct']:.2f}%")
    print("=" * 80)
