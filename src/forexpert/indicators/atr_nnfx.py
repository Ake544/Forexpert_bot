"""Port of 'NNFX ATR' by sueun123 (Pine v4).

Licence: the original Pine is Mozilla Public License 2.0. This port is a
derivative and stays under MPL-2.0.

Pine source of truth (line numbers from Indicators.txt):

    atr = int(rma(tr(true), atrLength) * 1e5) / 1e5                  # 1133
    bAtrSell = close - atr * atrLoss                                 # 1137  (buy SL)
    bAtrBuy  = close + atr * atrProfit                               # 1136  (buy TP)
    sAtrBuy  = close - atr * atrProfit                               # 1143  (sell TP)
    sAtrSell = close + atr * atrLoss                                 # 1144  (sell SL)

Two details are easy to lose in a port:

  * ``tr(true)`` uses ``handle_na = true``, so the first bar's true range is
    ``high - low`` instead of na.
  * ``formula(number, 5)`` TRUNCATES the ATR to 5 decimals. On a 5-digit FX
    instrument this repeatedly shaves up to 1e-5 off the value; it is part of
    the indicator's behaviour, so it is reproduced (set ``decimals`` to match
    your instrument, or to a large number to disable it).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .pine_math import ema, moving_average, rma, sma, true_range, truncate, wma


@dataclass(frozen=True)
class AtrResult:
    tr: pd.Series
    smoothed: pd.Series
    atr: pd.Series

    def levels(
        self, close: pd.Series, direction: str, sl_multiplier: float, tp_multiplier: float
    ) -> tuple[pd.Series, pd.Series]:
        """Return (stop_loss, take_profit) series for a hypothetical fill at ``close``."""
        if direction == "buy":
            return close - sl_multiplier * self.atr, close + tp_multiplier * self.atr
        if direction == "sell":
            return close + sl_multiplier * self.atr, close - tp_multiplier * self.atr
        raise ValueError("direction must be 'buy' or 'sell'")


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    smoothing: str = "RMA",
    decimals: int = 5,
) -> AtrResult:
    """Port of NNFX ATR.

    Args:
        high/low/close: OHLC series.
        length: 'X' input (14).
        smoothing: 'RMA' | 'SMA' | 'EMA' | 'WMA'.
        decimals: decimal places the Pine truncates the ATR to (5).
    """
    tr = true_range(high, low, close, handle_na=True)
    normalized = smoothing.strip().upper()
    if normalized == "RMA":
        smoothed = rma(tr, length)
    elif normalized == "SMA":
        smoothed = sma(tr, length)
    elif normalized == "EMA":
        smoothed = ema(tr, length)
    elif normalized == "WMA":
        smoothed = wma(tr, length)
    else:
        raise ValueError(f"Unsupported ATR smoothing: {smoothing!r}")

    atr = truncate(smoothed, decimals)
    return AtrResult(tr=tr, smoothed=smoothed, atr=atr.rename("atr"))


def at_risk_levels(
    atr_value: float, direction: str, sl_multiplier: float = 1.5, tp_multiplier: float = 2.25
) -> tuple[float, float]:
    """ATR distances for a fill at price 0.0, used to build levels around an entry."""
    if direction == "buy":
        return -sl_multiplier * atr_value, tp_multiplier * atr_value
    if direction == "sell":
        return sl_multiplier * atr_value, -tp_multiplier * atr_value
    raise ValueError("direction must be 'buy' or 'sell'")