"""Parity checks for the 'VWAP Bands with ML [CryptoSea]' port."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forexpert.indicators.ml_vwap import compute_ml_vwap


def frame(close: list[float], volume: list[float] | None = None) -> tuple[pd.Series, pd.Series]:
    index = pd.date_range("2024-01-01", periods=len(close), freq="h", tz="UTC")
    vols = pd.Series(volume if volume is not None else [1.0] * len(close), index=index)
    return pd.Series(close, index=index), vols


def test_vwap_is_cumulative_within_a_session() -> None:
    close, volume = frame([2.0, 4.0, 6.0], [1.0, 1.0, 2.0])
    result = compute_ml_vwap(close, volume, window=10)
    assert result.vwap.iloc[2] == pytest.approx((2 * 1 + 4 * 1 + 6 * 2) / 4)


def test_daily_bars_make_the_session_vwap_equal_close() -> None:
    """Pine's no-anchor default resets every trading day, so a 1D chart is degenerate."""
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    close = pd.Series([10.0, 30.0, 20.0], index=index)
    volume = pd.Series([7.0, 1.0, 9.0], index=index)
    result = compute_ml_vwap(close, volume, window=2)
    pd.testing.assert_series_equal(result.vwap, close.rename("vwap"), check_exact=False)


def test_intraday_vwap_resets_at_the_utc_day_boundary() -> None:
    index = pd.to_datetime(["2024-01-01 20:00", "2024-01-02 00:00"], utc=True)
    close = pd.Series([10.0, 40.0], index=index)
    volume = pd.Series([1.0, 1.0], index=index)
    result = compute_ml_vwap(close, volume, window=2)
    assert result.vwap.iloc[1] == pytest.approx(40.0)


def test_ml_output_is_the_previous_bar_ml_adjusted_value() -> None:
    close, volume = frame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = compute_ml_vwap(close, volume, window=3, forecast=1, sigma=0.1)
    assert np.isnan(result.ml_output.iloc[0])
    assert result.ml_output.iloc[3] == pytest.approx(result.ml_adjusted.iloc[2])
    assert result.ml_output.iloc[5] == pytest.approx(result.ml_adjusted.iloc[4])


def test_ml_adjusted_is_the_sigma_blend_of_sma_and_vwap() -> None:
    close, volume = frame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = compute_ml_vwap(close, volume, window=3, forecast=1, sigma=0.1)
    assert result.ml_base.iloc[2] == pytest.approx(2.0)
    assert result.ml_adjusted.iloc[2] == pytest.approx(2.0 + 0.1 * (result.vwap.iloc[2] - 2.0))


def test_green_state_is_vwap_above_ml_output_not_price_above_vwap() -> None:
    """Green is a VWAP-vs-ML comparison, which can disagree with price-vs-VWAP.

    A flat base then a jump up: price is far above VWAP yet VWAP is also above the
    lagged ML line, so green. A flat base then a drop: price sits below VWAP and
    the state is red. There is no bar in either series where "price above VWAP"
    and "green" disagree, because the lagged SMA dominates - which is the point.
    """
    up_close, up_volume = frame([10.0, 10.0, 10.0, 30.0])
    up = compute_ml_vwap(up_close, up_volume, window=3, forecast=1, sigma=0.1)
    assert up.vwap.iloc[-1] == pytest.approx(15.0)
    assert up.ml_output.iloc[-1] == pytest.approx(10.0)
    assert bool(up.is_bull.iloc[-1]) is True

    down_close, down_volume = frame([30.0, 10.0, 10.0, 10.0])
    down = compute_ml_vwap(down_close, down_volume, window=3, forecast=1, sigma=0.1)
    assert down.vwap.iloc[-1] == pytest.approx(15.0)
    assert bool(down.is_bull.iloc[-1]) is False


def test_green_state_is_unknown_while_the_ml_line_is_warming_up() -> None:
    """Tri-state matters: na must not silently read as 'red' and arm a short setup."""
    close, volume = frame([1.0, 2.0, 3.0, 4.0])
    result = compute_ml_vwap(close, volume, window=300, forecast=1)
    assert result.is_bull.iloc[0] is None or pd.isna(result.is_bull.iloc[0])
    pd.testing.assert_series_equal(
        result.is_bull,
        (result.vwap > result.ml_output).where(result.vwap.notna() & result.ml_output.notna()),
        check_names=False,
    )


def test_bands_are_built_on_the_shifted_ml_line_with_population_stdev() -> None:
    close, volume = frame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = compute_ml_vwap(close, volume, window=3, forecast=1, sigma=0.1,
                             band1_multiplier=2.0, band2_multiplier=3.0)
    std = float(np.std([2.0, 3.0, 4.0], ddof=0))
    assert result.std_dev.iloc[3] == pytest.approx(std)
    assert result.upper_band_1.iloc[3] == pytest.approx(result.ml_output.iloc[3] + 2.0 * std)
    assert result.lower_band_2.iloc[3] == pytest.approx(result.ml_output.iloc[3] - 3.0 * std)


def test_level_selector_rejects_unknown_line() -> None:
    close, volume = frame([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    result = compute_ml_vwap(close, volume, window=2)
    pd.testing.assert_series_equal(result.level("vwap"), result.vwap)
    pd.testing.assert_series_equal(result.level("ml_output"), result.ml_output)
    with pytest.raises(ValueError):
        result.level("lower_band_1")