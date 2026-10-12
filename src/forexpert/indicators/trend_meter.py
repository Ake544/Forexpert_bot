"""Port of the "Trend meter by Lij_MC" script (Pine v4) - green/red state only.

Licence note: the pasted source carries no licence header. Porting it for
personal use is fine; do not redistribute it.

The Lij_MC script is a dashboard of MANY independent states, so "the trend meter
is green" is ambiguous. Three readings are supported, selected by
``TrendMeterParams.mode``:

    crossover        Trend Bar colour with the default "MA Crossover" setting:
                     fast_ma > slow_ma. With slow_ma = SMA(150) this is the
                     "Slow MA 150, SMA" configuration from the strategy spec
                     (Pine lines 239 / 448 / 682).
    slow_direction   Trend Bar colour with "MA Direction - Slow MA" selected:
                     slow_ma[t] > slow_ma[t-1] (Pine line 247 / 553).
    three_meters     The script's own "All 3 Trend Meters align" state:
                     RSI(5) > 50 AND RSI(13) > 50 AND fast MACD histogram > 0
                     with defaults 8/21/5 (Pine lines 317 / 401 / 433 / 494).

The script's Wave Trend, Heikin-Ashi trend candles, Top Dog DAD, MACD 12/26/9,
RSI-signal-line and the background fills are NOT part of any supported mode.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .pine_math import ema, moving_average, rsi


@dataclass(frozen=True)
class TrendMeterResult:
    fast: pd.Series
    slow: pd.Series
    is_bull: pd.Series  # tri-state: True / False / None while warming up
    mode: str


def compute_trend_meter(
    close: pd.Series,
    mode: str = "crossover",
    fast_type: str = "EMA",
    fast_length: int = 5,
    slow_type: str = "SMA",
    slow_length: int = 150,
) -> TrendMeterResult:
    """Compute the configured Trend Meter state.

    Args:
        close: Close series (the script reads ``close`` for every MA).
        mode: 'crossover' | 'slow_direction' | 'three_meters'.
        fast_type/fast_length: Fast MA inputs of Trend Bar 1 (Pine lines 195-196).
        slow_type/slow_length: Slow MA inputs of Trend Bar 1 (Pine lines 197-198).
    """
    normalized = mode.strip().lower()
    if normalized not in {"crossover", "slow_direction", "three_meters"}:
        raise ValueError(f"Unsupported trend meter mode: {mode!r}")

    if normalized == "three_meters":
        fast_ema = ema(close, 8)
        slow_ema = ema(close, 21)
        signal_ema = ema(fast_ema - slow_ema, 5)
        fast_macd_hist_positive = (fast_ema - slow_ema - signal_ema) > 0
        rsi5_positive = rsi(close, 5) > 50
        rsi13_positive = rsi(close, 13) > 50
        aligned = fast_macd_hist_positive & rsi5_positive & rsi13_positive
        ready = rsi(close, 13).notna()
        return TrendMeterResult(
            fast=fast_ema,
            slow=slow_ema,
            is_bull=aligned.where(ready),
            mode=normalized,
        )

    fast = moving_average(close, fast_length, fast_type)
    slow = moving_average(close, slow_length, slow_type)
    if normalized == "crossover":
        comparison = fast > slow
    else:
        comparison = slow > slow.shift(1)
    ready = fast.notna() & slow.notna()
    if normalized == "slow_direction":
        ready = slow.notna() & slow.shift(1).notna()
    return TrendMeterResult(
        fast=fast, slow=slow, is_bull=comparison.where(ready), mode=normalized
    )