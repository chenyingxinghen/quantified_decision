"""多目标横截面选股标签构造。"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


class MultiObjectiveLabelBuilder:
    """从复权 OHLCV 构造收益、风险、流动性和可交易性标签。

    特征在 t 日收盘后生成，标签统一从 t+1 开盘开始，避免把无法获得的
    t 日收盘成交价作为执行价格。
    """

    def __init__(self, return_horizons: Sequence[int] = (5, 20, 60), risk_horizon: int = 20):
        horizons = sorted({int(h) for h in return_horizons})
        if not horizons or min(horizons) <= 0 or risk_horizon <= 1:
            raise ValueError("标签周期必须为正，风险周期必须大于 1")
        self.return_horizons = tuple(horizons)
        self.risk_horizon = int(risk_horizon)

    @staticmethod
    def _future_windows(values: np.ndarray, horizon: int, n_rows: int) -> np.ndarray:
        if len(values) <= horizon:
            return np.empty((0, horizon), dtype=float)
        return sliding_window_view(values[1:], horizon)[: n_rows - horizon]

    def build(self, data: pd.DataFrame, code: str | None = None) -> pd.DataFrame:
        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"构造多目标标签缺少字段: {sorted(missing)}")
        n = len(data)
        result = pd.DataFrame({"date": data["date"].values}, index=data.index)
        if code is not None:
            result["code"] = str(code)

        open_ = pd.to_numeric(data["open"], errors="coerce").to_numpy(dtype=float)
        close = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
        volume = pd.to_numeric(data["volume"], errors="coerce").to_numpy(dtype=float)
        amount = pd.to_numeric(data.get("amount", pd.Series(np.nan, index=data.index)), errors="coerce").to_numpy(dtype=float)
        entry = np.roll(open_, -1)
        entry[-1] = np.nan

        for horizon in self.return_horizons:
            label = np.full(n, np.nan, dtype=np.float32)
            windows = self._future_windows(close, horizon, n)
            valid_n = len(windows)
            if valid_n:
                label[:valid_n] = (windows[:, -1] / entry[:valid_n] - 1).astype(np.float32)
            result[f"y_ret_{horizon}d"] = label

        h = self.risk_horizon
        close_windows = self._future_windows(close, h, n)
        volume_windows = self._future_windows(volume, h, n)
        amount_windows = self._future_windows(amount, h, n)
        valid_n = len(close_windows)
        mdd = np.full(n, np.nan, dtype=np.float32)
        downvol = np.full(n, np.nan, dtype=np.float32)
        illiq = np.full(n, np.nan, dtype=np.float32)
        tradable = np.full(n, np.nan, dtype=np.float32)
        if valid_n:
            paths = np.concatenate([entry[:valid_n, None], close_windows], axis=1)
            running_max = np.maximum.accumulate(paths, axis=1)
            drawdowns = paths / np.where(running_max == 0, np.nan, running_max) - 1
            mdd[:valid_n] = np.nanmin(drawdowns, axis=1).astype(np.float32)

            daily_returns = paths[:, 1:] / paths[:, :-1] - 1
            negative_returns = np.where(daily_returns < 0, daily_returns, np.nan)
            neg_count = np.sum(np.isfinite(negative_returns), axis=1)
            neg_mean = np.nansum(negative_returns, axis=1) / np.maximum(neg_count, 1)
            neg_var = np.nansum((negative_returns - neg_mean[:, None]) ** 2, axis=1) / np.maximum(neg_count - 1, 1)
            downvol_values = np.sqrt(neg_var) * np.sqrt(252.0)
            downvol_values[neg_count < 2] = 0.0
            downvol[:valid_n] = downvol_values.astype(np.float32)

            with np.errstate(divide="ignore", invalid="ignore"):
                impact = np.abs(daily_returns) / amount_windows
            impact[~np.isfinite(impact)] = np.nan
            impact_count = np.sum(np.isfinite(impact), axis=1)
            impact_mean = np.nansum(impact, axis=1) / np.maximum(impact_count, 1)
            impact_mean[impact_count == 0] = np.nan
            illiq[:valid_n] = impact_mean.astype(np.float32)
            tradable[:valid_n] = np.all(
                np.isfinite(volume_windows) & (volume_windows > 0), axis=1
            ).astype(np.float32)

        result[f"y_mdd_{h}d"] = mdd
        result[f"y_downvol_{h}d"] = downvol
        result[f"y_illiq_{h}d"] = illiq
        result[f"y_tradable_{h}d"] = tradable

        # 风险调整收益（前瞻 Sharpe 类）：未来收益 / |最大回撤|，越大越好。
        # 直接把"收益"与"回撤"绑成单一目标，使模型学习"每单位风险的收益"，
        # 而非单纯预测谁收益高。分母用同一风险窗口的最大回撤。
        for horizon in self.return_horizons:
            ret_vals = result[f"y_ret_{horizon}d"].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                s = ret_vals / (np.abs(mdd) + 1e-3)
            s[~np.isfinite(s)] = np.nan
            result[f"y_sharpe_{horizon}d"] = np.clip(s, -10.0, 10.0).astype(np.float32)
        return result

    def build_universe(
        self,
        stocks_data: dict[str, pd.DataFrame],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        parts = []
        for code, data in stocks_data.items():
            if data is None or data.empty:
                continue
            labels = self.build(data, code=code)
            if start_date is not None:
                labels = labels[labels["date"].astype(str) >= str(start_date)]
            if end_date is not None:
                labels = labels[labels["date"].astype(str) <= str(end_date)]
            parts.append(labels)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)


def cross_sectional_rank_targets(
    labels: pd.DataFrame,
    target_cols: Iterable[str],
    *,
    risk_cols: Iterable[str] = (),
) -> pd.DataFrame:
    """把连续目标转换为同日 [0,1] desirability rank。

    风险列方向反转，使所有输出统一为“越大越好”。最大回撤列通常为负数，
    原值越大已经越好，因此不应放入 risk_cols；波动和非流动性应反转。
    """
    result = labels.copy()
    risk_set = set(risk_cols)
    for col in target_cols:
        ascending = col not in risk_set
        result[f"rank_{col}"] = result.groupby("date")[col].rank(
            pct=True, ascending=ascending, method="average"
        )
    return result
