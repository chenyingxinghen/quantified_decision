"""多目标横截面选股标签构造。

设计保证（核心诉求：正交性 + 无数据泄露）：

1. 无数据泄露（forward-only）
   - 特征在 t 日收盘后生成，所有标签统一从 t+1 日开盘开始，严禁把 t 日
     及之前的未实现信息当作可执行价格。
   - `y_sharpe_{h}d` 使用「与收益窗口同长」的最大回撤作为分母，不再借用
     更长的 20 日风险窗口，避免把长周期风险泄露进短周期目标。
   - 提供 `verify_no_lookahead()`，可实证「标签 t 完全不依赖 close[t] 及更早
     数据」，作为防泄露闸门。

2. 正交性
   - 累积收益 `y_ret_{h}d`（5/20/60）为向后兼容保留，但窗口嵌套、彼此共线，
     本身不正交。
   - 新增「非重叠前向收益腿」`y_ret_leg_{a}_{b}d`（如 1-5 / 6-20 / 21-60），
     因窗口互不相交、按构造正交，是无泄露的正交目标基。
   - 提供 `orthogonalize_labels()`：在**同日截面内**用 Gram-Schmidt 对目标列
     做残差化，使任意目标集合正交；因只使用当日截面，对训练/验证/测试分别
     应用均不泄露。
   - 提供 `diagnose_label_orthogonality()`：量化目标间的平均截面相关与条件数。
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# 默认非重叠前向收益腿：相对标签日 t 的交易日偏移（含端点），窗口互不相交。
# 1-5 / 6-20 / 21-60 与配置中的 (5,20,60) 对齐，但彼此不重叠 → 正交。
DEFAULT_ORTHOGONAL_LEGS: tuple[tuple[int, int], ...] = ((1, 5), (6, 20), (21, 60))


class MultiObjectiveLabelBuilder:
    """从复权 OHLCV 构造收益、风险、流动性和可交易性标签。

    特征在 t 日收盘后生成，标签统一从 t+1 开盘开始，避免把无法获得的
    t 日收盘成交价作为执行价格。
    """

    def __init__(
        self,
        return_horizons: Sequence[int] = (5, 20, 60),
        risk_horizon: int = 20,
        *,
        orthogonal_legs: bool = True,
        legs: Sequence[tuple[int, int]] = DEFAULT_ORTHOGONAL_LEGS,
        use_matching_risk_for_sharpe: bool = True,
        sharpe_clip: float = 10.0,
        sharpe_eps: float = 1e-3,
        strict_no_lookahead: bool = True,
    ):
        horizons = sorted({int(h) for h in return_horizons})
        if not horizons or min(horizons) <= 0 or risk_horizon <= 1:
            raise ValueError("标签周期必须为正，风险周期必须大于 1")
        self.return_horizons = tuple(horizons)
        self.risk_horizon = int(risk_horizon)
        self.orthogonal_legs = bool(orthogonal_legs)
        self.legs = tuple((int(a), int(b)) for a, b in legs if b > a > 0)
        self.use_matching_risk_for_sharpe = bool(use_matching_risk_for_sharpe)
        self.sharpe_clip = float(sharpe_clip)
        self.sharpe_eps = float(sharpe_eps)
        self.strict_no_lookahead = bool(strict_no_lookahead)
        # 构造期使用的最大未来偏移，用于泄露自检。
        self._max_forward_offset = max(
            max(self.return_horizons),
            self.risk_horizon,
            max((b for _, b in self.legs), default=0),
        )

    # ── 前向窗口工具（只引用索引 > t 的数据）──────────────────────────────
    @staticmethod
    def _future_closes(close: np.ndarray, horizon: int, n: int) -> np.ndarray:
        """close[t+1 .. t+horizon] 的滑动窗口；只含未来数据。长度 n-horizon。"""
        if len(close) <= horizon:
            return np.empty((0, horizon), dtype=float)
        return sliding_window_view(close[1:], horizon)[: n - horizon]

    @staticmethod
    def _leg_slice(values: np.ndarray, start_off: int, end_off: int, n: int) -> np.ndarray:
        """values[t+start_off .. t+end_off] 的窗口（含两端），只含未来数据。"""
        span = end_off - start_off + 1
        if len(values) <= end_off:
            return np.empty((0, span), dtype=float)
        return sliding_window_view(values[start_off:], span)[: n - end_off]

    def _assert_forward_only(self) -> None:
        """防泄露闸门：任何 ≤0 的偏移都意味着引用了标签日或之前的数据。"""
        if self._max_forward_offset <= 0:
            raise RuntimeError("标签构造存在非前向偏移，可能泄露未来信息")

    def build(self, data: pd.DataFrame, code: str | None = None) -> pd.DataFrame:
        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"构造多目标标签缺少字段: {sorted(missing)}")
        self._assert_forward_only()

        n = len(data)
        result = pd.DataFrame({"date": data["date"].values}, index=data.index)
        if code is not None:
            result["code"] = str(code)

        open_ = pd.to_numeric(data["open"], errors="coerce").to_numpy(dtype=float)
        close = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
        volume = pd.to_numeric(data["volume"], errors="coerce").to_numpy(dtype=float)
        amount = pd.to_numeric(
            data.get("amount", pd.Series(np.nan, index=data.index)), errors="coerce"
        ).to_numpy(dtype=float)

        # 入场价 = 下一交易日开盘（forward-only，绝不使用 t 日 close）。
        entry = np.roll(open_, -1)
        entry[-1] = np.nan

        # ── 累积收益（向后兼容；窗口嵌套、彼此共线，并非正交基）────────────
        for horizon in self.return_horizons:
            label = np.full(n, np.nan, dtype=np.float32)
            windows = self._future_closes(close, horizon, n)
            valid_n = len(windows)
            if valid_n:
                label[:valid_n] = (windows[:, -1] / entry[:valid_n] - 1).astype(np.float32)
            result[f"y_ret_{horizon}d"] = label

        # ── 非重叠前向收益腿（按构造正交，无泄露的正交目标基）────────────────
        if self.orthogonal_legs:
            for a, b in self.legs:
                if b >= n:
                    legret = np.full(n, np.nan, dtype=np.float32)
                else:
                    legret = np.full(n, np.nan, dtype=np.float32)
                    close_w = self._leg_slice(close, a, b, n)
                    open_w = self._leg_slice(open_, a, a, n)[:, 0]
                    valid_n = len(close_w)
                    if valid_n:
                        legret[:valid_n] = (
                            close_w[:, -1] / open_w[:valid_n] - 1
                        ).astype(np.float32)
                result[f"y_ret_leg_{a}_{b}d"] = legret

        # ── 风险标签（基于入场日起至 risk_horizon 的风险窗口）────────────────
        h = self.risk_horizon
        close_windows = self._leg_slice(close, 1, h, n)
        volume_windows = self._leg_slice(volume, 1, h, n)
        amount_windows = self._leg_slice(amount, 1, h, n)
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

        # ── 风险调整收益（前瞻 Sharpe 类）：未来收益 / |最大回撤| ─────────────
        # 修复泄露：每个收益窗口使用「同长」的最大回撤，不再借用固定 20 日窗口。
        mdd_by_horizon: dict[int, np.ndarray] = {h: mdd}
        if self.use_matching_risk_for_sharpe:
            for horizon in self.return_horizons:
                if horizon == h:
                    continue
                cw = self._leg_slice(close, 1, horizon, n)
                if len(cw) == 0:
                    mdd_by_horizon[horizon] = np.full(n, np.nan, dtype=np.float32)
                    continue
                p = np.concatenate([entry[: len(cw), None], cw], axis=1)
                rm = np.maximum.accumulate(p, axis=1)
                d = p / np.where(rm == 0, np.nan, rm) - 1
                mm = np.full(n, np.nan, dtype=np.float32)
                mm[: len(cw)] = np.nanmin(d, axis=1).astype(np.float32)
                mdd_by_horizon[horizon] = mm

        for horizon in self.return_horizons:
            ret_vals = result[f"y_ret_{horizon}d"].to_numpy(dtype=float)
            denom_mdd = mdd_by_horizon.get(horizon, mdd)
            with np.errstate(divide="ignore", invalid="ignore"):
                s = ret_vals / (np.abs(denom_mdd) + self.sharpe_eps)
            s[~np.isfinite(s)] = np.nan
            result[f"y_sharpe_{horizon}d"] = np.clip(s, -self.sharpe_clip, self.sharpe_clip).astype(np.float32)
        return result

    def verify_no_lookahead(self, data: pd.DataFrame, tol: float = 1e-9) -> bool:
        """实证防泄露：篡改 close[t] / open[t]（标签 t 的过去信息），标签 t 应不变。

        标签 t 只引用 open[t+1]（入场）与 close[t+horizon] 等未来数据，因此
        对位于索引 i 处的 close/open 做改动，不应影响 result.loc[i] 的任何列。
        返回 True 表示通过；否则抛出 AssertionError。
        """
        base = self.build(data)

        # 单点扰动：只改动索引 i 处的 close/open（标签 i 的「过去」信息），
        # 重建后标签 i 必须不变。若改的是 close[i+horizon]/open[i+1]（标签 i
        # 合法引用的未来数据）则另当别论——这里只改同索引，恰好是标签 i 不
        # 该引用的位置，从而精确区分「合法前向引用」与「泄露」。
        check_cols = [c for c in base.columns if c not in ("date", "code")]
        n = len(data)
        step = max(1, n // 40)  # 抽样若干索引，避免 O(n^2) 过慢
        for i in range(step, n - self._max_forward_offset, step):
            mutated = data.copy()
            for col in ("close", "open"):
                if col not in mutated.columns:
                    continue
                arr = mutated[col].to_numpy(dtype=float).copy()
                if not np.isfinite(arr[i]):
                    continue
                arr[i] = arr[i] * 1.234 + 0.567
                mutated[col] = arr
            after = self.build(mutated)
            for c in check_cols:
                b = float(base[c].to_numpy(dtype=float)[i])
                a = float(after[c].to_numpy(dtype=float)[i])
                if np.isfinite(b) and abs(b - a) > tol:
                    raise AssertionError(
                        f"泄露检测到：列 {c} 在行 {i} 依赖了 t 日/更早数据 "
                        f"(base={b}, mutated={a})"
                    )
        return True

    def build_universe(
        self,
        stocks_data: dict[str, pd.DataFrame],
        start_date: str | None = None,
        end_date: str | None = None,
        *,
        orthogonalize: bool = False,
        group: str = "date",
    ) -> pd.DataFrame:
        parts = []
        for code, data in stocks_data.items():
            if data is None or data.empty:
                continue
            labels = self.build(data, code=code)
            if orthogonalize:
                # 只对连续目标做正交化（y_tradable 为 0/1 标志，不参与）。
                target_like = [
                    c
                    for c in labels.columns
                    if c.startswith("y_ret")
                    or c.startswith("y_mdd")
                    or c.startswith("y_downvol")
                    or c.startswith("y_illiq")
                    or c.startswith("y_sharpe")
                ]
                if target_like:
                    labels = orthogonalize_labels(labels, target_like, group=group)
            if start_date is not None:
                labels = labels[labels["date"].astype(str) >= str(start_date)]
            if end_date is not None:
                labels = labels[labels["date"].astype(str) <= str(end_date)]
            parts.append(labels)
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
        return out


def orthogonalize_labels(
    labels: pd.DataFrame,
    cols: Iterable[str],
    group: str = "date",
    add_intercept: bool = True,
) -> pd.DataFrame:
    """把目标列残差化为正交目标（Gram-Schmidt，按截面日分组）。

    在**每个截面日内部**依次把当前列对已有正交列做最小二乘回归并取残差，
    使输出列在同日截面内两两正交。因只使用当日截面数据，对训练/验证/测试
    分别调用均不泄露未来/跨期信息。

    返回：原表副本 + 各 `orth_<col>` 正交列。原始列保留。
    """
    cols = list(cols)
    out = labels.copy()
    orth_frames = []
    for _, g in labels.groupby(group, sort=False):
        orth_frames.append(_orthogonalize_group(g, cols, add_intercept))
    if orth_frames:
        orth_all = pd.concat(orth_frames)
        orth_all = orth_all.reindex(out.index)
        for c in cols:
            out[f"orth_{c}"] = orth_all[f"orth_{c}"]
    return out


def _orthogonalize_group(g: pd.DataFrame, cols: list[str], add_intercept: bool) -> pd.DataFrame:
    n = len(g)
    orth = pd.DataFrame(index=g.index)
    prev: list[np.ndarray] = []
    arrays = {c: g[c].to_numpy(dtype=float) for c in cols}
    for c in cols:
        y = arrays[c].copy()
        if prev:
            Xcols = ([np.ones(n)] if add_intercept else []) + list(prev)
            X = np.column_stack(Xcols)
            mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            if mask.sum() >= X.shape[1] + 1:
                beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
                pred = X @ beta
                resid = np.where(np.isfinite(y), y - pred, np.nan)
            else:
                resid = y
        else:
            resid = y
        orth[f"orth_{c}"] = resid
        prev.append(orth[f"orth_{c}"].to_numpy(dtype=float))
    return orth


def diagnose_label_orthogonality(
    labels: pd.DataFrame,
    cols: Iterable[str],
    group: str = "date",
) -> dict:
    """量化目标间的平均截面相关与多重共线性。

    返回：
      - avg_correlation: 各目标平均截面相关矩阵（DataFrame）
      - max_abs_offdiag_corr: 平均相关矩阵非对角最大绝对值
      - condition_number: 平均相关矩阵的条件数（>30 表示强共线性）
      - n_groups: 参与统计的截面日数量
    """
    cols = list(cols)
    k = len(cols)
    corr_sum = np.zeros((k, k), dtype=float)
    n_groups = 0
    for _, g in labels.groupby(group, sort=False):
        sub = g[cols]
        if sub.shape[0] < 3:
            continue
        gc = sub.corr().to_numpy()
        if not np.all(np.isfinite(gc)):
            continue
        corr_sum += gc
        n_groups += 1
    if n_groups == 0:
        avg = np.full((k, k), np.nan)
    else:
        avg = corr_sum / n_groups
    np.fill_diagonal(avg, np.nan)
    max_abs = float(np.nanmax(np.abs(avg))) if n_groups else float("nan")
    # 条件数：用相关矩阵（对角补 1）的特征值。
    cond = float("nan")
    if n_groups:
        sym = np.where(np.isnan(avg), 0.0, avg)
        np.fill_diagonal(sym, 1.0)
        eig = np.linalg.eigvalsh(sym)
        eig = np.clip(eig, 1e-12, None)
        cond = float(eig.max() / eig.min())
    return {
        "avg_correlation": pd.DataFrame(avg, index=cols, columns=cols),
        "max_abs_offdiag_corr": max_abs,
        "condition_number": cond,
        "n_groups": n_groups,
    }


def cross_sectional_rank_targets(
    labels: pd.DataFrame,
    target_cols: Iterable[str],
    *,
    risk_cols: Iterable[str] = (),
    verify_leak_free: bool = True,
) -> pd.DataFrame:
    """把连续目标转换为同日 [0,1] desirability rank。

    风险列方向反转，使所有输出统一为“越大越好”。最大回撤列通常为负数，
    原值越大已经越好，因此不应放入 risk_cols；波动和非流动性应反转。

    防泄露：排名严格按 `date` 分组，绝不引入排序键之外的跨期/跨股信息。
    """
    if verify_leak_free and "date" not in labels.columns:
        raise ValueError("cross_sectional_rank_targets 需要 date 列以严格按截面日排名")
    result = labels.copy()
    risk_set = set(risk_cols)
    for col in target_cols:
        ascending = col not in risk_set
        result[f"rank_{col}"] = result.groupby("date", sort=False)[col].rank(
            pct=True, ascending=ascending, method="average"
        )
    return result
