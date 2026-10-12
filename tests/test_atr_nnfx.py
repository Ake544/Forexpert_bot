"""Parity checks for the 'NNFX ATR' port (sueun123, MPL-2.0)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forexpert.indicators.atr_nnfx import at_risk_levels, compute_atr


def ohlc(rows: list[tuple[float, float, float, float]]) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    index = pd.date_range("2024-01-01", periods=len(rows), freq="h", tz="UTC")
    data = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    return data["open"], data["high"], data["low"], data["close"]


def constant_range_bars(count: int, span: float, base: float = 10.0) -> list[tuple[float, float, float, float]]:
    mid = base + span / 2
    return [(mid, base + span, base, mid) for _ in range(count)]


def test_constant_true_range_gives_constant_atr() -> None:
    _, high, low, close = ohlc(constant_range_bars(20, 2.0))
    result = compute_atr(high, low, close, length=14, decimals=5)
    assert result.tr.iloc[0] == pytest.approx(2.0)
    assert result.atr.iloc[13] == pytest.approx(2.0)
    assert result.atr.iloc[19] == pytest.approx(2.0)


def test_atr_is_truncated_to_five_decimals_like_the_pine() -> None:
    _, high, low, close = ohlc(constant_range_bars(20, 2.000003))
    result = compute_atr(high, low, close, length=14, decimals=5)
    assert result.smoothed.iloc[-1] == pytest.approx(2.000003, abs=1e-9)
    assert result.atr.iloc[-1] == pytest.approx(2.0)


def test_first_true_range_uses_high_low() -> None:
    _, high, low, close = ohlc([(10.0, 20.0, 5.0, 19.0), (19.0, 19.5, 18.5, 19.0)])
    result = compute_atr(high, low, close, length=2)
    assert result.tr.iloc[0] == pytest.approx(15.0)


def test_atr_is_na_before_the_window_is_full() -> None:
    _, high, low, close = ohlc(constant_range_bars(5, 2.0))
    result = compute_atr(high, low, close, length=14)
    assert result.atr.iloc[:13].isna().all()
    assert result.atr.iloc[13:].notna().all()


def test_buy_and_sell_levels_are_mirrored() -> None:
    buy_sl, buy_tp = at_risk_levels(2.0, "buy", sl_multiplier=1.5, tp_multiplier=2.25)
    sell_sl, sell_tp = at_risk_levels(2.0, "sell", sl_multiplier=1.5, tp_multiplier=2.25)
    assert (buy_sl, buy_tp) == pytest.approx((-3.0, 4.5))
    assert (sell_sl, sell_tp) == pytest.approx((3.0, -4.5))


def test_unknown_smoothing_is_rejected() -> None:
    _, high, low, close = ohlc(constant_range_bars(20, 2.0))
    with pytest.raises(ValueError):
        compute_atr(high, low, close, length=14, smoothing="HMA")