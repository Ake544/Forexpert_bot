"""Tests for the Lij_MC Trend Meter modes and for the KRO guard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forexpert.indicators.compute import compute_indicators, snapshot, validate_candles
from forexpert.indicators.kro import KroNotPortedError
from forexpert.indicators.trend_meter import compute_trend_meter
from forexpert.settings import IndicatorParams, MlVwapParams


def series(values: np.ndarray | list[float]) -> pd.Series:
    values = np.asarray(values, dtype="float64")
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC"))


def test_uptrend_is_green_under_crossover_mode() -> None:
    result = compute_trend_meter(series(np.arange(1.0, 401.0)), mode="crossover",
                                 fast_type="EMA", fast_length=5, slow_type="SMA", slow_length=150)
    assert bool(result.is_bull.iloc[-1]) is True


def test_downtrend_is_red_under_crossover_mode() -> None:
    result = compute_trend_meter(series(np.arange(400.0, 0.0, -1.0)), mode="crossover")
    assert bool(result.is_bull.iloc[-1]) is False


def test_slow_direction_mode_reads_ma_slope_not_price_position() -> None:
    up = compute_trend_meter(series(np.arange(1.0, 401.0)), mode="slow_direction", slow_length=20)
    down = compute_trend_meter(series(np.arange(400.0, 0.0, -1.0)), mode="slow_direction", slow_length=20)
    assert bool(up.is_bull.iloc[-1]) is True
    assert bool(down.is_bull.iloc[-1]) is False


def test_three_meters_mode_requires_all_three_conditions() -> None:
    rng = np.random.default_rng(5)
    result = compute_trend_meter(series(rng.normal(size=300)), mode="three_meters")
    assert set(result.is_bull.dropna().unique()) <= {True, False}
    assert result.is_bull.iloc[:13].isna().all()
    assert result.is_bull.iloc[50:].notna().all()


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_trend_meter(series([1.0, 2.0, 3.0]), mode="magic")


def test_kro_confirmation_refuses_to_guess() -> None:
    params = IndicatorParams(ml_vwap=MlVwapParams(window=5))
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [2.0, 2.0],
            "low": [0.5, 0.5],
            "close": [1.5, 1.5],
            "volume": [10.0, 10.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
    )
    frame.index.name = "time"
    from dataclasses import replace

    with pytest.raises(KroNotPortedError):
        compute_indicators(frame, replace(params, kro=replace(params.kro, enabled=True)))


def test_snapshot_maps_missing_values_to_none() -> None:
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [2.0, 2.0],
            "low": [0.5, 0.5],
            "close": [1.5, 1.5],
            "volume": [10.0, 10.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
    )
    frame.index.name = "time"
    params = IndicatorParams(ml_vwap=MlVwapParams(window=300))
    result = snapshot(compute_indicators(frame, params))
    assert result.vwap_bull is None, "unknown must stay unknown, never read as False"
    assert result.trend_bull is None
    assert result.vslrt_bull is None
    assert result.atr is None
    assert result.close == pytest.approx(1.5)


def test_candle_contract_rejects_naive_index() -> None:
    frame = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [1.0]},
        index=pd.date_range("2024-01-01", periods=1, freq="h"),
    )
    frame.index.name = "time"
    with pytest.raises(ValueError):
        validate_candles(frame)