"""Shared exit rules for backtest and live execution."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    reason: Optional[str] = None


def evaluate_exit(
    *,
    current_price: float,
    entry_price: float,
    holding_days: int,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    enable_stop_loss: bool,
    enable_take_profit: bool,
    enable_time_stop: bool,
    time_stop_days: int,
    time_stop_max_return_pct: float,
) -> ExitDecision:
    """Evaluate the close/tail-price exit rules used by both runtimes.

    The strategy executes exits near the close. Consequently stop-loss and
    take-profit decisions must use the price observed at that execution time,
    rather than the day's high/low, which are unavailable to the live tail
    process as path-dependent triggers.
    """
    if current_price <= 0 or entry_price <= 0:
        return ExitDecision(False)

    if enable_stop_loss and stop_loss is not None and current_price <= stop_loss:
        return ExitDecision(True, "stop_loss")

    if enable_take_profit and take_profit is not None and current_price >= take_profit:
        return ExitDecision(True, "take_profit")

    return_pct = (current_price - entry_price) / entry_price
    if (
        enable_time_stop
        and holding_days >= time_stop_days
        and return_pct <= time_stop_max_return_pct
    ):
        return ExitDecision(True, "time_stop")

    return ExitDecision(False)
