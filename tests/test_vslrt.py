"""Parity checks for the VSLRT port (LonesomeTheBlue, MPL-2.0)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forexpert.indicators.vslrt import compute_vslrt


def series(values: np.ndarray | list[float]) -> pd.Series:
    values = np.asarray(values, dtype="float64")
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC"))


def test_slope_of_a_perfect_ramp_is_one() -> None:
    result = compute_vslrt(series(np.arange(1.0, 101.0)), short_length=20, long_length=64)
    assert result.slope_long.iloc[-1] == pytest.approx(1.0)
    assert result.long_line.iloc[-1] == pytest.approx(64.0)
    assert result.short_line.iloc[-1] == pytest.approx(20.0)


def test_downtrend_is_red() -> None:
    result = compute_vslrt(series(np.arange(100.0, 0.0, -1.0)), short_length=20, long_length=64)
    assert bool(result.is_bull.iloc[-1]) is False
    assert result.long_line.iloc[-1] == pytest.approx(-64.0)


def test_slope_matches_numpy_polyfit_on_the_long_window() -> None:
    rng = np.random.default_rng(3)
    values = series(rng.normal(size=200).cumsum())
    result = compute_vslrt(values, short_length=20, long_length=64)
    window = values.to_numpy()[-64:]
    assert result.slope_long.iloc[-1] == pytest.approx(np.polyfit(np.arange(64), window, 1)[0])


def test_values_are_na_until_each_window_is_full() -> None:
    result = compute_vslrt(series(np.arange(1.0, 101.0)), short_length=20, long_length=64)
    assert np.isnan(result.slope_short.iloc[18])
    assert not np.isnan(result.slope_short.iloc[19])
    assert np.isnan(result.slope_long.iloc[62])
    assert not np.isnan(result.slope_long.iloc[63])


def test_volume_is_not_required() -> None:
    """The Pine uses volume only to colour the plots, never to compute the values."""
    result = compute_vslrt(series(np.arange(1.0, 101.0)))
    assert result.long_line.notna().sum() > 0