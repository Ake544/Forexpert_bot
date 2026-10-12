"""Port of 'Volume-Supported Linear Regression Trend' (VSLRT) by LonesomeTheBlue.

Licence: the original Pine is Mozilla Public License 2.0. This port is a
derivative and stays under MPL-2.0.

Pine source of truth (line numbers from Indicators.txt):

    slope_price    = linreg(src, len1, 0) - linreg(src, len1, 1)      # 851
    slope_price_lt = linreg(src, len2, 0) - linreg(src, len2, 1)      # 852
    plot(slope_price    * len1, style = columns)                      # 883
    plot(slope_price_lt * len2, linewidth = 3)                        # 894

Because ``linreg(src, n, 0) - linreg(src, n, 1)`` reduces to the OLS slope of
the trailing window (see pine_math.linreg), the plotted series are simply the
windowed regression slope scaled by the window length.

The strategy's confirmation is "the VSLRT line is above/below", which maps to
the only LINE plot in the script: ``slope_price_lt * len2`` compared to zero.

Volume is deliberately NOT an input: in the Pine, volume only selects the
*colour* of the plots (lines 866-880). The values we need are close-only.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .pine_math import ols_slope


@dataclass(frozen=True)
class VslrtResult:
    slope_short: pd.Series
    slope_long: pd.Series
    short_line: pd.Series
    long_line: pd.Series
    is_bull: pd.Series  # tri-state: True / False / None while warming up

    def line(self, which: str = "long") -> pd.Series:
        if which == "long":
            return self.long_line
        if which == "short":
            return self.short_line
        raise ValueError(f"Unknown VSLRT line: {which!r} (expected 'short' or 'long')")


def compute_vslrt(
    close: pd.Series, short_length: int = 20, long_length: int = 64
) -> VslrtResult:
    """Port of VSLRT.

    Args:
        close: Source (the strategy fixes source = close).
        short_length: "Short Term Length".
        long_length: "Long Term Length".
    """
    if short_length < 2 or long_length < 2:
        raise ValueError("VSLRT lengths must be >= 2")

    slope_short = ols_slope(close, short_length)
    slope_long = ols_slope(close, long_length)
    short_line = slope_short * short_length
    long_line = slope_long * long_length

    return VslrtResult(
        slope_short=slope_short,
        slope_long=slope_long,
        short_line=short_line,
        long_line=long_line,
        is_bull=(long_line > 0).where(long_line.notna()),
    )