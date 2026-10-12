"""Aggregate: run every indicator over a closed-candle frame and snapshot the last bar.

Input contract (enforced here so no caller can pass leaky data):
  * index/column 'time' is a UTC datetime, strictly increasing
  * only CLOSED candles (the caller drops the in-progress bar)
  * columns: open, high, low, close, volume
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from ..settings import IndicatorParams
from .atr_nnfx import AtrResult, compute_atr
from .ml_vwap import MlVwapResult, compute_ml_vwap
from .pine_math import truncate
from .trend_meter import TrendMeterResult, compute_trend_meter
from .vslrt import VslrtResult, compute_vslrt

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
Direction = Literal["buy", "sell"]


class DataContractError(ValueError):
    """Raised when the candle frame violates the closed-candle OHLCV contract."""


def validate_candles(frame: pd.DataFrame, require_volume: bool = True) -> pd.DataFrame:
    """Validate the OHLCV frame, returning it unchanged when it is well formed."""
    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise DataContractError(f"Candle frame is missing columns: {missing}")
    if frame.index.name != "time" and "time" not in frame.columns:
        raise DataContractError("Candle frame must be indexed by a UTC 'time' index")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataContractError("Candle frame index must be a DatetimeIndex")
    if frame.index.tz is None:
        raise DataContractError("Candle frame index must be timezone-aware (UTC)")
    if not frame.index.is_monotonic_increasing:
        raise DataContractError("Candle frame index must be sorted ascending")
    if frame.index.has_duplicates:
        raise DataContractError("Candle frame index must not contain duplicates")
    if len(frame) and (
        (frame["high"] < frame["low"]).any()
        or (frame["high"] < frame[["open", "close"]].max(axis=1)).any()
        or (frame["low"] > frame[["open", "close"]].min(axis=1)).any()
    ):
        raise DataContractError("Candle frame violates high >= max(open, close) >= min(open, close) >= low")
    if require_volume and len(frame) and frame["volume"].isna().any():
        raise DataContractError("Candle frame contains rows without volume")
    return frame


@dataclass(frozen=True)
class IndicatorFrame:
    """All indicator outputs for a validated frame, aligned on the candle index."""

    candles: pd.DataFrame
    ml_vwap: MlVwapResult
    trend_meter: TrendMeterResult
    vslrt: VslrtResult
    atr: AtrResult

    @property
    def reference_level(self) -> pd.Series:
        return self.ml_vwap.level(self.touch_line)

    touch_line: str = "vwap"

    def level_frame(self) -> pd.DataFrame:
        """The columns the state machine reads, with stable names."""
        level = self.reference_level
        return pd.DataFrame(
            {
                "open": self.candles["open"],
                "high": self.candles["high"],
                "low": self.candles["low"],
                "close": self.candles["close"],
                "level": level,
                "vwap_bull": self.ml_vwap.is_bull,
                "trend_bull": self.trend_meter.is_bull,
                "vslrt_bull": self.vslrt.is_bull,
                "vslrt_long_line": self.vslrt.long_line,
                "atr": self.atr.atr,
            },
            index=self.candles.index,
        )


@dataclass(frozen=True)
class Snapshot:
    """The values on the single most recent CLOSED candle."""

    bar_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    level: float | None
    vwap_bull: bool | None
    trend_bull: bool | None
    vslrt_bull: bool | None
    vslrt_long_line: float | None
    atr: float | None

    def is_buy_ready(self) -> bool:
        return bool(self.vwap_bull) and bool(self.trend_bull)


def compute_indicators(
    frame: pd.DataFrame, params: IndicatorParams, touch_line: str = "vwap"
) -> IndicatorFrame:
    """Run all ported indicators over a validated closed-candle frame."""
    validate_candles(frame)
    if params.kro.enabled:
        from .kro import kro_confirmation

        kro_confirmation()

    ml = compute_ml_vwap(
        close=frame["close"],
        volume=frame["volume"],
        window=params.ml_vwap.window,
        forecast=params.ml_vwap.forecast,
        sigma=params.ml_vwap.sigma,
        band1_multiplier=params.ml_vwap.band1_multiplier,
        band2_multiplier=params.ml_vwap.band2_multiplier,
    )
    trend = compute_trend_meter(
        close=frame["close"],
        mode=params.trend_meter.mode,
        fast_type=params.trend_meter.fast_type,
        fast_length=params.trend_meter.fast_length,
        slow_type=params.trend_meter.slow_type,
        slow_length=params.trend_meter.slow_length,
    )
    vslrt = compute_vslrt(
        close=frame["close"],
        short_length=params.vslrt.short_length,
        long_length=params.vslrt.long_length,
    )
    atr = compute_atr(
        high=frame["high"],
        low=frame["low"],
        close=frame["close"],
        length=params.atr.length,
        smoothing=params.atr.smoothing,
        decimals=params.atr.decimals,
    )
    return IndicatorFrame(
        candles=frame,
        ml_vwap=ml,
        trend_meter=trend,
        vslrt=vslrt,
        atr=atr,
        touch_line=touch_line,
    )


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


def _opt_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return bool(value)


def snapshot(indicators: IndicatorFrame) -> Snapshot:
    """Extract the last closed candle's values, mapping na to None."""
    if len(indicators.candles) == 0:
        raise DataContractError("Cannot snapshot an empty frame")
    levels = indicators.level_frame()
    last = levels.iloc[-1]
    candle = indicators.candles.iloc[-1]
    return Snapshot(
        bar_time=pd.Timestamp(levels.index[-1]),
        open=float(candle["open"]),
        high=float(candle["high"]),
        low=float(candle["low"]),
        close=float(candle["close"]),
        volume=float(candle["volume"]),
        level=_opt_float(last["level"]),
        vwap_bull=_opt_bool(last["vwap_bull"]),
        trend_bull=_opt_bool(last["trend_bull"]),
        vslrt_bull=_opt_bool(last["vslrt_bull"]),
        vslrt_long_line=_opt_float(last["vslrt_long_line"]),
        atr=_opt_float(last["atr"]),
    )


def stop_loss_take_profit(
    entry_price: float,
    atr_value: float,
    direction: Direction,
    sl_multiplier: float,
    tp_multiplier: float,
) -> tuple[float, float] | None:
    """Apply the NNFX ATR distances around an entry price.

    Returns None when ATR is unavailable (still warming up) so the caller can
    refuse to publish a signal instead of publishing a zero-distance one.
    """
    if not _opt_float(atr_value) or atr_value <= 0:
        return None
    distance_sl = atr_value * sl_multiplier
    distance_tp = atr_value * tp_multiplier
    if direction == "buy":
        return entry_price - distance_sl, entry_price + distance_tp
    return entry_price + distance_sl, entry_price - distance_tp


def last_atr(indicators: IndicatorFrame, decimals: int | None = None) -> float | None:
    """Last ATR value, optionally re-truncated to the instrument's decimal places."""
    value = indicators.atr.atr.iloc[-1]
    truncated = truncate(value, decimals) if decimals is not None else value
    return _opt_float(truncated)