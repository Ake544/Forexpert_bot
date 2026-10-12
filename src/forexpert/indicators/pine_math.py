"""TradingView Pine built-in maths, ported to match Pine semantics exactly.

The differences from the obvious pandas equivalents matter and are the usual
source of silent parity failures:

  * ``ta.ema`` / ``ta.rma`` are seeded with ``ta.sma(source, length)`` on the
    first non-na bar. ``pandas.Series.ewm`` seeds with the *first value* and
    ``rolling.mean`` cannot seed at all, so both are wrong for parity work.
  * ``ta.stdev`` is the **population** (biased, divide by N) standard deviation.
    ``rolling.std`` defaults to ``ddof=1``.
  * ``ta.linreg(source, length, offset)`` fits the line over the *current*
    ``length`` bars and evaluates it ``offset`` bars back. Writing the regression
    with local coordinates x = 0..length-1 (oldest to newest) this is
    ``sma(source, length) - offset * slope``, so
    ``linreg(src, n, 0) - linreg(src, n, 1)`` is exactly the OLS slope of the
    window. VSLRT relies on that identity.
  * ``ta.vwap(source)`` with no anchor argument is anchored to the start of the
    trading day (Pine's default is ``timeframe.change("1D")``), so it resets
    every session. On a daily chart every bar is its own session and the result
    is simply ``source``. See :func:`session_vwap`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

Series = pd.Series


def _as_float_series(source: Series) -> Series:
    if isinstance(source, pd.Series):
        return source.astype("float64")
    return pd.Series(np.asarray(source, dtype="float64"))


def sma(source: Series, length: int) -> Series:
    """``ta.sma``: simple moving average, na for the first ``length - 1`` bars."""
    s = _as_float_series(source)
    if length <= 0:
        raise ValueError("length must be positive")
    return s.rolling(window=length, min_periods=length).mean()


def _seeded_recursive(source: Series, length: int, alpha: float) -> Series:
    """Pine's EMA/RMA recursion: seed with SMA(length), then y = a*x + (1-a)*y_prev."""
    s = _as_float_series(source)
    values = s.to_numpy(dtype="float64", copy=True)
    out = np.full(values.shape, np.nan, dtype="float64")
    if values.size < length:
        return pd.Series(out, index=s.index, name=s.name)

    seed_window = values[:length]
    if np.isnan(seed_window).any():
        return pd.Series(out, index=s.index, name=s.name)
    out[length - 1] = seed_window.mean()

    one_minus_alpha = 1.0 - alpha
    previous = out[length - 1]
    for i in range(length, values.size):
        current = values[i]
        if np.isnan(current) or np.isnan(previous):
            out[i] = np.nan
            previous = np.nan
            continue
        previous = alpha * current + one_minus_alpha * previous
        out[i] = previous
    return pd.Series(out, index=s.index, name=s.name)


def ema(source: Series, length: int) -> Series:
    """``ta.ema``: alpha = 2 / (length + 1), seeded with SMA(length)."""
    if length <= 0:
        raise ValueError("length must be positive")
    return _seeded_recursive(source, length, 2.0 / (length + 1.0))


def rma(source: Series, length: int) -> Series:
    """``ta.rma``: Wilder's smoothing, alpha = 1 / length, seeded with SMA(length)."""
    if length <= 0:
        raise ValueError("length must be positive")
    return _seeded_recursive(source, length, 1.0 / length)


def wma(source: Series, length: int) -> Series:
    """``ta.wma``: linearly weighted moving average, weights 1..length."""
    s = _as_float_series(source)
    weights = np.arange(1, length + 1, dtype="float64")
    weights /= weights.sum()
    return (
        s.rolling(window=length, min_periods=length)
        .apply(lambda w: float(np.dot(w, weights)), raw=True)
        .rename(s.name)
    )


def stdev(source: Series, length: int) -> Series:
    """``ta.stdev``: POPULATION standard deviation (ddof=0), na until ``length`` bars."""
    s = _as_float_series(source)
    return s.rolling(window=length, min_periods=length).std(ddof=0).rename(s.name)


def moving_average(source: Series, length: int, ma_type: str) -> Series:
    """Dispatch helper for the MA types the Pine scripts allow ('EMA'/'SMA'/'RMA'/'WMA')."""
    key = ma_type.strip().upper()
    if key == "SMA":
        return sma(source, length)
    if key == "EMA":
        return ema(source, length)
    if key == "RMA":
        return rma(source, length)
    if key == "WMA":
        return wma(source, length)
    raise ValueError(f"Unsupported MA type: {ma_type}")


def ols_slope(source: Series, length: int) -> Series:
    """OLS slope of the line fitted over the trailing ``length`` bars.

    Uses the closed form slope = (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2) which is exact
    for the fixed coordinate sequence x = 0..n-1.
    """
    s = _as_float_series(source)
    values = s.to_numpy(dtype="float64", copy=False)
    out = np.full(values.shape, np.nan, dtype="float64")
    if values.size < length:
        return pd.Series(out, index=s.index, name=s.name)

    x = np.arange(length, dtype="float64")
    sum_x = x.sum()
    sum_x2 = float((x * x).sum())
    denom = length * sum_x2 - sum_x * sum_x

    windows = sliding_window_view(values, length)
    if windows.shape[0]:
        sum_y = windows.sum(axis=1)
        sum_xy = (windows * x).sum(axis=1)
        slopes = (length * sum_xy - sum_x * sum_y) / denom
        out[length - 1 :] = slopes
    return pd.Series(out, index=s.index, name=s.name)


def linreg(source: Series, length: int, offset: int = 0) -> Series:
    """``ta.linreg``: regression line over the current window, evaluated ``offset`` bars back."""
    return (sma(source, length) - offset * ols_slope(source, length)).rename(source.name)


def true_range(
    high: Series, low: Series, close: Series, handle_na: bool = True
) -> Series:
    """``ta.tr(handle_na)``: max(high-low, |high-close[1]|, |low-close[1]|).

    With ``handle_na=True`` (which NNFX ATR uses) the first bar, where the
    previous close is na, falls back to ``high - low``.
    """
    h, low_, c = _as_float_series(high), _as_float_series(low), _as_float_series(close)
    previous_close = c.shift(1)
    tr = pd.concat(
        [h - low_, (h - previous_close).abs(), (low_ - previous_close).abs()], axis=1
    ).max(axis=1)
    if handle_na:
        fallback = (h - low_).where(previous_close.isna())
        tr = tr.where(previous_close.notna(), fallback)
    return tr.rename("tr")


def rsi(source: Series, length: int) -> Series:
    """``ta.rsi``: Wilder's RSI (up/down averages seeded with SMA, alpha = 1/length)."""
    s = _as_float_series(source)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = rma(gain.fillna(0.0), length)
    avg_loss = rma(loss.fillna(0.0), length)
    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))
    result = result.where(~(avg_loss == 0.0), np.where(avg_gain > 0.0, 100.0, np.nan))
    result.iloc[:length] = np.nan
    return result.rename("rsi")


def _session_ids(
    index: pd.Index, session_tz: str = "UTC", session_start_hour: float = 0.0
) -> Series | None:
    """Group a bar index into session days, or None when it carries no dates.

    A bar belongs to the session whose start is the previous ``session_start_hour``
    local midnight in ``session_tz``. Defaults (UTC / hour 0) are plain UTC
    calendar days. Using an IANA ``session_tz`` keeps the boundary aligned across
    daylight-saving changes, which a fixed UTC offset cannot.
    """
    if not isinstance(index, pd.DatetimeIndex):
        return None
    aware = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    local = aware.tz_convert(session_tz)
    start_hour = float(session_start_hour) % 24.0
    shifted = local - pd.Timedelta(start_hour, unit="h") if start_hour else local
    return pd.Series(shifted.normalize(), index=index)


def session_vwap(
    source: Series,
    volume: Series,
    session_tz: str = "UTC",
    session_start_hour: float = 0.0,
) -> Series:
    """``ta.vwap(source)`` with no anchor: the daily session VWAP.

    Pine's default anchor is ``timeframe.change("1D")``: the volume and
    price*volume sums are reset at the first bar of each trading day on the
    exchange clock. On a daily chart each bar is therefore its own session and
    the result equals ``source``.

    Sessions default to UTC calendar days. ``session_tz`` and
    ``session_start_hour`` move the boundary for feeds whose daily bar opens off
    midnight UTC (many FX brokers): e.g. a broker on ``Europe/Athens`` server
    time opens its day at local 00:00, and TradingView's FX sessions start at
    17:00 ``America/New_York``. An index without datetime information has no
    session boundaries, so the whole series counts as one session instead of
    guessing.
    """
    s, v = _as_float_series(source), _as_float_series(volume)
    sessions = _session_ids(s.index, session_tz, session_start_hour)
    weighted = s * v
    if sessions is None:
        cumulative_volume = v.cumsum()
        cumulative_weighted = weighted.cumsum()
    else:
        cumulative_volume = v.groupby(sessions, sort=False, dropna=False).cumsum()
        cumulative_weighted = weighted.groupby(sessions, sort=False, dropna=False).cumsum()
    with np.errstate(divide="ignore", invalid="ignore"):
        result = cumulative_weighted / cumulative_volume
    return result.where(cumulative_volume > 0).rename("vwap")


def truncate(value: float | Series, decimals: int) -> float | Series:
    """NNFX ATR's ``formula``: ``int(value * 10^d) / 10^d`` (truncate toward zero)."""
    factor = 10.0**decimals
    if isinstance(value, pd.Series):
        return (np.trunc(value * factor) / factor).rename(value.name)
    return float(np.trunc(float(value) * factor) / factor)