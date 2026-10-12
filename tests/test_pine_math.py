"""Parity checks for the Pine maths primitives.

Every expected value below is derived by hand from the Pine semantics, not from
the implementation, so a regression in the port shows up as a failure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forexpert.indicators import pine_math as pm


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC"))


def test_sma_seeds_after_length_bars() -> None:
    result = pm.sma(series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_ema_is_seeded_with_sma_not_first_value() -> None:
    result = pm.ema(series([1.0, 2.0, 3.0, 4.0]), 3)
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)          # seed = sma(1,2,3)
    assert result.iloc[3] == pytest.approx(0.5 * 4 + 0.5 * 2.0)


def test_rma_is_seeded_with_sma_and_uses_alpha_one_over_n() -> None:
    result = pm.rma(series([1.0, 2.0, 3.0, 4.0]), 3)
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(4.0 / 3.0 + 2.0 * 2.0 / 3.0)


def test_stdev_is_population_not_sample() -> None:
    result = pm.stdev(series([1.0, 2.0, 3.0, 4.0]), 4)
    expected = np.sqrt(((2.5 - 1) ** 2 + (2.5 - 2) ** 2 + (2.5 - 3) ** 2 + (2.5 - 4) ** 2) / 4)
    assert result.iloc[3] == pytest.approx(expected)
    assert result.iloc[3] != pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))


def test_ols_slope_matches_numpy_polyfit_on_a_noisy_window() -> None:
    rng = np.random.default_rng(7)
    values = series(list(rng.normal(size=80)))
    slopes = pm.ols_slope(values, 20)
    assert np.isnan(slopes.iloc[18])
    window = values.to_numpy()[60:80]
    assert slopes.iloc[79] == pytest.approx(np.polyfit(np.arange(20), window, 1)[0])


def test_linreg_offset_difference_equals_slope() -> None:
    rng = np.random.default_rng(11)
    values = series(list(rng.normal(size=60)))
    at_zero = pm.linreg(values, 20, 0)
    at_one = pm.linreg(values, 20, 1)
    assert (at_zero - at_one).iloc[-1] == pytest.approx(pm.ols_slope(values, 20).iloc[-1])


def test_linreg_at_offset_zero_equals_sma_at_window_centre() -> None:
    """For a perfectly linear series the regression line passes through the window mean."""
    values = series([float(i) for i in range(30)])
    assert pm.linreg(values, 10, 0).iloc[-1] == pytest.approx(values.iloc[-10:].mean())


def test_true_range_first_bar_uses_high_low_when_handle_na() -> None:
    high = series([11.0, 12.0])
    low = series([9.0, 8.0])
    close = series([10.0, 11.0])
    result = pm.true_range(high, low, close, handle_na=True)
    assert result.iloc[0] == pytest.approx(2.0)
    assert result.iloc[1] == pytest.approx(max(4.0, abs(12.0 - 10.0), abs(8.0 - 10.0)))


def test_session_vwap_is_cumulative_within_a_session() -> None:
    result = pm.session_vwap(series([2.0, 4.0, 6.0]), series([1.0, 3.0, 1.0]))
    assert result.iloc[0] == pytest.approx(2.0)
    assert result.iloc[1] == pytest.approx((2.0 * 1 + 4.0 * 3) / 4)
    assert result.iloc[2] == pytest.approx((2.0 * 1 + 4.0 * 3 + 6.0 * 1) / 5)


def test_session_vwap_resets_at_the_utc_day_boundary() -> None:
    index = pd.to_datetime(
        ["2024-01-01 20:00", "2024-01-01 23:00", "2024-01-02 01:00"], utc=True
    )
    result = pm.session_vwap(
        pd.Series([10.0, 20.0, 40.0], index=index),
        pd.Series([1.0, 1.0, 1.0], index=index),
    )
    assert result.iloc[1] == pytest.approx(15.0)
    assert result.iloc[2] == pytest.approx(40.0)


def test_session_vwap_without_dates_treats_the_whole_series_as_one_session() -> None:
    result = pm.session_vwap(pd.Series([2.0, 4.0]), pd.Series([1.0, 1.0]))
    assert result.iloc[1] == pytest.approx(3.0)


def test_session_vwap_honours_a_shifted_start_hour() -> None:
    # A 22:00 UTC boundary keeps 21:00 and 22:00 on opposite sides: with the
    # default UTC day they would share a session.
    index = pd.to_datetime(
        ["2024-01-01 21:00", "2024-01-01 22:00", "2024-01-01 23:00"], utc=True
    )
    source = pd.Series([10.0, 20.0, 40.0], index=index)
    volume = pd.Series([1.0, 1.0, 1.0], index=index)
    shifted = pm.session_vwap(source, volume, session_tz="UTC", session_start_hour=22)
    assert shifted.iloc[0] == pytest.approx(10.0)  # last bar before the boundary
    assert shifted.iloc[1] == pytest.approx(20.0)  # 22:00 opens the new session
    assert shifted.iloc[2] == pytest.approx((20.0 + 40.0) / 2)


def test_session_vwap_in_a_time_zone_keeps_dst_alignment() -> None:
    # Broker server day: 00:00 Europe/Athens. A UTC bar at 22:00 in winter
    # (EET, UTC+2) is already the *next* local day, so it must open a session.
    index = pd.to_datetime(
        ["2024-01-15 21:00", "2024-01-15 22:00", "2024-01-15 23:00"], utc=True
    )
    source = pd.Series([10.0, 20.0, 40.0], index=index)
    volume = pd.Series([1.0, 1.0, 1.0], index=index)
    broker = pm.session_vwap(source, volume, session_tz="Europe/Athens")
    assert broker.iloc[0] == pytest.approx(10.0)
    assert broker.iloc[1] == pytest.approx(20.0)
    assert broker.iloc[2] == pytest.approx((20.0 + 40.0) / 2)


def test_session_vwap_shifted_hour_classifies_overnight_bars() -> None:
    # Two bars either side of midnight UTC still belong to one 22:00 session.
    index = pd.to_datetime(
        ["2024-01-01 23:00", "2024-01-02 01:00"], utc=True
    )
    source = pd.Series([10.0, 30.0], index=index)
    volume = pd.Series([1.0, 1.0], index=index)
    result = pm.session_vwap(source, volume, session_start_hour=22)
    assert result.iloc[1] == pytest.approx(20.0)


def test_truncate_toward_zero_for_both_signs() -> None:
    assert pm.truncate(0.0123456789, 5) == pytest.approx(0.01234)
    assert pm.truncate(-0.0123456789, 5) == pytest.approx(-0.01234)