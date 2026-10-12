"""Port of 'VWAP Bands with ML [CryptoSea]' (Pine v6).

Pine source of truth (line numbers from Indicators.txt):

    vwap          = ta.vwap(source)                                  # 766
    ml_base       = ta.sma(source, window)                           # 770
    ml_adjusted   = ml_base + sigma * (vwap - ml_base)               # 771
    ml_output     = ml_adjusted[forecast]                            # 772
    std_dev       = ta.stdev(close, window)                          # 776
    upper_band*   = vwap ± mult * std_dev                            # 777-781  (DEAD CODE)
    upper_ml_band = ml_output ± mult * std_dev                       # 784-787
    colour        = vwap > ml_output ? bull : bear                   # 808-809

Two facts that are easy to get wrong:

1. GREEN does NOT mean price above VWAP. It means ``vwap > ml_output``. With
   sigma = 0.1 that is a comparison of VWAP against a lagged blend that is
   ~90% an SMA(window), so the filter is mostly an SMA(300) regime test.
2. The band multipliers only affect the *plotted* bands and the script's own
   crossover alerts, never the green/red state. They are irrelevant to this
   strategy; they are implemented only for visual parity.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .pine_math import session_vwap, sma, stdev


@dataclass(frozen=True)
class MlVwapResult:
    """Every series the Pine script plots, plus its green/red state."""

    vwap: pd.Series
    ml_base: pd.Series
    ml_adjusted: pd.Series
    ml_output: pd.Series
    std_dev: pd.Series
    upper_band_1: pd.Series
    lower_band_1: pd.Series
    upper_band_2: pd.Series
    lower_band_2: pd.Series
    is_bull: pd.Series  # tri-state: True / False / None while warming up

    def level(self, which: str) -> pd.Series:
        """The reference line used by the strategy's above/below and touch rules."""
        if which == "vwap":
            return self.vwap
        if which == "ml_output":
            return self.ml_output
        raise ValueError(f"Unknown touch line: {which!r} (expected 'vwap' or 'ml_output')")


def compute_ml_vwap(
    close: pd.Series,
    volume: pd.Series,
    window: int = 300,
    forecast: int = 1,
    sigma: float = 0.1,
    band1_multiplier: float = 2.0,
    band2_multiplier: float = 3.0,
) -> MlVwapResult:
    """Port of the CryptoSea ML VWAP indicator.

    Args:
        close: Source (the strategy fixes source = close).
        volume: Bar volume. The Pine VWAP is volume weighted, so this cannot be
            faked. Callers must enforce a volume policy upstream. The VWAP is
            session anchored (Pine's default ``timeframe.change("1D")``), so it
            resets at each UTC trading day and equals ``close`` on daily bars.
        window: ML Window (SMA length used for the ML base and for the stdev).
        forecast: Shifts the ML output FORWARD by N bars (``ml_adjusted[forecast]``).
        sigma: Weight of VWAP inside the ML blend.
        band1_multiplier: Band 1 multiplier (plotted only).
        band2_multiplier: Band 2 multiplier (plotted only).
    """
    if window <= 1:
        raise ValueError("window must be > 1")
    if forecast < 0:
        raise ValueError("forecast must be >= 0")

    vwap = session_vwap(close, volume)
    ml_base = sma(close, window)
    ml_adjusted = ml_base + sigma * (vwap - ml_base)
    ml_output = ml_adjusted.shift(forecast)
    deviation = stdev(close, window)
    green = (vwap > ml_output).where(vwap.notna() & ml_output.notna())

    return MlVwapResult(
        vwap=vwap,
        ml_base=ml_base,
        ml_adjusted=ml_adjusted,
        ml_output=ml_output,
        std_dev=deviation,
        upper_band_1=ml_output + band1_multiplier * deviation,
        lower_band_1=ml_output - band1_multiplier * deviation,
        upper_band_2=ml_output + band2_multiplier * deviation,
        lower_band_2=ml_output - band2_multiplier * deviation,
        is_bull=green,
    )