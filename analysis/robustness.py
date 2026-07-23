"""
analysis/robustness.py — 稳健性检验：统计性检验 + 牛熊震荡异质性

覆盖指标:
  - RankIC:        每日截面 Spearman(模型得分, 实际收益)
  - ICIR:          RankIC 均值 / 标准差
  - t 检验:        RankIC 是否显著异于 0 (ttest_1samp)
  - F 检验:        不同市场状态下 RankIC 分布是否存在显著差异 (f_oneway)
  - 牛/熊/震荡异质性: 各状态下 IC 与分组收益表现
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_daily_ic(
    scores: np.ndarray, returns: np.ndarray, dates: np.ndarray
) -> pd.Series:
    """逐交易日计算截面 RankIC（Spearman 相关系数）。"""
    df = pd.DataFrame({
        "date": pd.Series(dates).astype(str).str[:10],
        "score": np.asarray(scores, dtype=float),
        "ret": np.asarray(returns, dtype=float),
    })
    out = {}
    for d, g in df.groupby("date"):
        s = g["score"].to_numpy()
        r = g["ret"].to_numpy()
        mask = np.isfinite(s) & np.isfinite(r)
        if mask.sum() < 10:
            continue
        rho, _ = stats.spearmanr(s[mask], r[mask])
        if np.isfinite(rho):
            out[d] = rho
    return pd.Series(out).sort_index()


def rankic_stats(daily_ic: pd.Series) -> dict:
    """汇总 RankIC 统计量与显著性检验。"""
    ic = daily_ic.to_numpy(dtype=float)
    ic = ic[np.isfinite(ic)]
    n = len(ic)
    mean = float(np.mean(ic)) if n else float("nan")
    std = float(np.std(ic, ddof=1)) if n > 1 else float("nan")
    icir = mean / std if std and std > 0 else float("nan")
    # t 检验: H0: IC 均值 = 0
    t_stat, p_val = (float("nan"), float("nan"))
    if n > 1:
        t_stat, p_val = stats.ttest_1samp(ic, 0.0)
    ic_pos_ratio = float(np.mean(ic > 0)) if n else float("nan")
    return {
        "n_days": n,
        "rankic_mean": mean,
        "rankic_std": std,
        "icir": icir,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "ic_positive_ratio": ic_pos_ratio,
        "significant_5pct": bool(p_val < 0.05) if np.isfinite(p_val) else False,
    }


def regime_heterogeneity(
    scores: np.ndarray, returns: np.ndarray, dates: np.ndarray,
    regime_per_sample: np.ndarray,
) -> dict:
    """分市场状态计算 RankIC 与分组收益异质性。"""
    df = pd.DataFrame({
        "date": pd.Series(dates).astype(str).str[:10],
        "score": np.asarray(scores, dtype=float),
        "ret": np.asarray(returns, dtype=float),
        "regime": np.asarray(regime_per_sample),
    })
    result: dict = {"by_regime": {}, "ic_by_regime": {}}
    ic_series = {}
    for regime in ["bull", "bear", "sideways"]:
        sub = df[df["regime"] == regime]
        if len(sub) < 50:
            result["by_regime"][regime] = {"n": int(len(sub)), "note": "样本不足"}
            continue
        # 该状态下的每日 IC
        daily_ic = compute_daily_ic(sub["score"].to_numpy(), sub["ret"].to_numpy(),
                                    sub["date"].to_numpy())
        ic_series[regime] = daily_ic
        # 该状态下分组收益（十分位）
        grp_ret, grp_spread = _group_returns(sub["score"].to_numpy(),
                                             sub["ret"].to_numpy(), q=10)
        result["by_regime"][regime] = {
            "n": int(len(sub)),
            "rankic_mean": float(np.mean(daily_ic.to_numpy())) if len(daily_ic) else float("nan"),
            "icir": (float(np.mean(daily_ic.to_numpy()) / np.std(daily_ic.to_numpy(), ddof=1))
                     if len(daily_ic) > 1 and np.std(daily_ic.to_numpy(), ddof=1) > 0 else float("nan")),
            "top_group_return": float(grp_ret[-1]),
            "bottom_group_return": float(grp_ret[0]),
            "top_minus_bottom": float(grp_spread),
        }
        result["ic_by_regime"][regime] = daily_ic
    # F 检验: 三状态 RankIC 分布是否显著不同
    samples = [s.to_numpy() for s in ic_series.values() if len(s) > 1]
    if len(samples) >= 2:
        f_stat, f_p = stats.f_oneway(*samples)
        result["f_test"] = {
            "f_stat": float(f_stat),
            "p_value": float(f_p),
            "significant_5pct": bool(f_p < 0.05),
        }
    else:
        result["f_test"] = {"note": "状态样本不足，未做 F 检验"}
    return result


def _group_returns(scores: np.ndarray, returns: np.ndarray, q: int = 10):
    """按得分十分位分组，返回每组平均收益与头尾差。"""
    order = np.argsort(np.argsort(scores))  # 0..n-1 秩
    denom = len(scores) / q
    grp = np.minimum((order / denom).astype(int), q - 1)
    grp_ret = np.array([np.nanmean(returns[grp == k]) for k in range(q)])
    return grp_ret, grp_ret[-1] - grp_ret[0]


def run_robustness(
    scores: np.ndarray, returns: np.ndarray, dates: np.ndarray,
    regime_per_sample: np.ndarray,
) -> dict:
    """端到端运行统计性检验 + 状态异质性。"""
    daily_ic = compute_daily_ic(scores, returns, dates)
    stats_summary = rankic_stats(daily_ic)
    hetero = regime_heterogeneity(scores, returns, dates, regime_per_sample)
    return {
        "daily_ic": daily_ic,
        "stats": stats_summary,
        "regime": hetero,
    }
