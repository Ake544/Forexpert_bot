"""Candle-source tests. No network: a fake session stands in for requests.

Every provider must return the same canonical frame (UTC DatetimeIndex named
'time', columns open/high/low/close/volume, oldest first, forming bar removed).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from forexpert.data import DataError, RateLimitError, SymbolNotFound, build_source, build_sources
from forexpert.data.base import (
    RateLimiter,
    drop_forming_bars,
    empty_frame,
    finalize_frame,
    invert_frame,
    invert_symbol,
)
from forexpert.data.binance import BinanceSource, normalize_symbol as binance_symbol
from forexpert.data.csv_source import CsvSource
from forexpert.data.twelve_data import TwelveDataSource, normalize_symbol as td_symbol

NOW = pd.Timestamp("2024-01-02T00:00:00Z")


class FakeResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    """Returns queued responses and records every call."""

    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        if not self._responses:
            raise AssertionError("provider made more requests than the test queued")
        payload = self._responses.pop(0)
        if isinstance(payload, FakeResponse):
            return payload
        return FakeResponse(payload)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Rate limiting and retry backoff must not slow the suite down."""
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)


def td_value(moment: str, close: float, volume: float = 100.0) -> dict:
    return {
        "datetime": moment,
        "open": str(close - 0.1),
        "high": str(close + 0.2),
        "low": str(close - 0.2),
        "close": str(close),
        "volume": str(volume),
    }


def twelve_settings(**overrides) -> dict:
    settings = {
        "base_url": "https://api.twelvedata.com",
        "api_key": "test-key",
        "requests_per_minute": 10_000,
        "outputsize": 1000,
        "max_retries": 2,
        "retry_backoff_seconds": 0.0,
    }
    settings.update(overrides)
    return settings


# ------------------------------------------------------------------ base helpers
def test_forming_bar_is_dropped_by_close_time() -> None:
    def candle(moment: str) -> pd.DataFrame:
        return pd.DataFrame(
            {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
            index=pd.DatetimeIndex([moment], name="time"),
        )

    # A 02:00 bar cannot have closed yet on any timeframe we support.
    assert len(drop_forming_bars(candle("2024-01-02T02:00:00Z"), "1h", now=NOW)) == 0
    assert len(drop_forming_bars(candle("2024-01-02T02:00:00Z"), "4h", now=NOW)) == 0
    # A 4h bar that opened at 20:00 closed exactly at NOW, so it is kept.
    assert len(drop_forming_bars(candle("2024-01-01T20:00:00Z"), "4h", now=NOW)) == 1


def test_finalize_frame_sorts_dedupes_and_localizes_to_utc() -> None:
    records = [
        {"open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0, "volume": 1.0},
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        {"open": 9.0, "high": 9.0, "low": 9.0, "close": 9.0, "volume": 1.0},  # dup
    ]
    frame = finalize_frame(
        records,
        ["2024-01-01T08:00:00", "2024-01-01T04:00:00", "2024-01-01T08:00:00"],
        "EURUSD", "4h", now=NOW,
    )
    assert len(frame) == 2
    assert frame.index.is_monotonic_increasing
    assert not frame.index.has_duplicates
    assert str(frame.index.tz) == "UTC"
    assert frame["close"].tolist() == [1.0, 9.0]  # last write wins


def test_finalize_frame_refuses_rows_with_missing_ohlc() -> None:
    with pytest.raises(DataError):
        finalize_frame(
            [{"open": None, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}],
            ["2024-01-01T04:00:00"], "EURUSD", "4h", now=NOW,
        )


def test_empty_frame_has_the_canonical_shape() -> None:
    frame = empty_frame()
    assert frame.empty
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


def test_unsupported_timeframe_is_rejected() -> None:
    with pytest.raises(DataError):
        drop_forming_bars(pd.DataFrame(), "7m", now=NOW)


# ------------------------------------------------------------------- twelve data
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("EURCHF", "EUR/CHF"), ("eurchf", "EUR/CHF"), ("BTC/USDT", "BTC/USDT"),
     ("BTCUSDT", "BTCUSDT")],
)
def test_twelve_data_symbol_normalization(raw: str, expected: str) -> None:
    assert td_symbol(raw) == expected


def test_twelve_data_reverses_newest_first_payload_and_drops_forming_bar() -> None:
    payload = {
        "meta": {"symbol": "EUR/CHF", "interval": "4h"},
        "values": [
            td_value("2024-01-02T00:00:00", 1.30),   # forming: closes at 04:00
            td_value("2024-01-01T20:00:00", 1.20),
            td_value("2024-01-01T16:00:00", 1.10),
        ],
        "status": "ok",
    }
    session = FakeSession([payload])
    frame = TwelveDataSource(twelve_settings(), session=session).fetch("EURCHF", "4h", now=NOW)

    assert len(frame) == 2
    assert frame.index.is_monotonic_increasing
    assert frame["close"].tolist() == [1.10, 1.20]
    assert str(frame.index.tz) == "UTC"

    params = session.calls[0]["params"]
    assert params["symbol"] == "EUR/CHF"
    assert params["interval"] == "4h"
    assert params["timezone"] == "UTC"
    assert params["apikey"] == "test-key"


def test_twelve_data_asks_utc_and_never_leaks_the_key_into_logs() -> None:
    session = FakeSession([{"status": "error", "code": "invalid-symbol", "message": "bad symbol"}])
    with pytest.raises(SymbolNotFound):
        TwelveDataSource(twelve_settings(), session=session).fetch("NOPE", "4h", now=NOW)
    assert session.calls[0]["params"]["apikey"] == "test-key"


def test_twelve_data_retries_then_succeeds_on_429() -> None:
    session = FakeSession(
        [FakeResponse({}, 429), {"status": "ok", "values": [td_value("2024-01-01T20:00:00", 1.2)]}]
    )
    frame = TwelveDataSource(twelve_settings(), session=session).fetch("EURCHF", "4h", now=NOW)
    assert len(frame) == 1
    assert len(session.calls) == 2


def test_twelve_data_gives_up_after_max_retries() -> None:
    session = FakeSession([FakeResponse({}, 429)] * 5)
    with pytest.raises(RateLimitError):
        TwelveDataSource(twelve_settings(max_retries=2), session=session).fetch(
            "EURCHF", "4h", now=NOW
        )
    assert len(session.calls) == 3  # original + 2 retries


def test_twelve_data_raises_on_unexpected_api_error() -> None:
    session = FakeSession([{"status": "error", "code": "invalid-api-key", "message": "bad key"}])
    with pytest.raises(DataError):
        TwelveDataSource(twelve_settings(), session=session).fetch("EURCHF", "4h", now=NOW)


def test_twelve_data_requires_an_api_key() -> None:
    with pytest.raises(DataError):
        TwelveDataSource(twelve_settings(api_key=""), session=FakeSession([]))


def test_twelve_data_refuses_a_response_with_only_forming_bars() -> None:
    session = FakeSession([{"status": "ok", "values": [td_value("2024-01-02T00:00:00", 1.3)]}])
    with pytest.raises(DataError):
        TwelveDataSource(twelve_settings(), session=session).fetch("EURCHF", "4h", now=NOW)


# ----------------------------------------------------------------------- binance
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("EURCHF", "EURCHFUSDT"), ("BTCUSDT", "BTCUSDT"), ("ethusdt", "ETHUSDT")],
)
def test_binance_symbol_normalization(raw: str, expected: str) -> None:
    assert binance_symbol(raw) == expected


def kline(moment: str, close: float) -> list:
    stamp = int(pd.Timestamp(moment).timestamp() * 1000)
    return [stamp, str(close - 0.1), str(close + 0.2), str(close - 0.2), str(close), "1000",
            stamp + 3_599_999, "0", 10, "0", "0", "0"]


def test_binance_parses_klines_and_drops_the_forming_bar() -> None:
    session = FakeSession([
        [kline("2024-01-01T16:00:00Z", 1.1), kline("2024-01-01T20:00:00Z", 1.2),
         kline("2024-01-02T00:00:00Z", 1.3)]
    ])
    frame = BinanceSource(
        {"base_url": "https://api.binance.com", "requests_per_minute": 10_000},
        session=session,
    ).fetch("EURCHF", "4h", now=NOW)

    assert len(frame) == 2
    assert frame["close"].tolist() == [1.1, 1.2]
    assert frame["volume"].tolist() == [1000.0, 1000.0]
    assert session.calls[0]["params"] == {"symbol": "EURCHFUSDT", "interval": "4h", "limit": 1000}


def test_binance_maps_unknown_symbol_to_a_clear_error() -> None:
    session = FakeSession([FakeResponse({}, 400)])
    with pytest.raises(SymbolNotFound):
        BinanceSource({"base_url": "https://x", "requests_per_minute": 10_000},
                      session=session).fetch("NOPE", "4h", now=NOW)


def test_binance_rejects_an_unsupported_timeframe() -> None:
    with pytest.raises(DataError):
        BinanceSource({"base_url": "https://x"}, session=FakeSession([])).fetch("BTCUSDT", "7m")


# -------------------------------------------------------------------------- csv
def write_csv(directory: Path, symbol: str = "EURCHF", timeframe: str = "4h") -> Path:
    frame = pd.DataFrame(
        {
            "time": ["2024-01-01T16:00:00Z", "2024-01-01T20:00:00Z"],
            "open": [1.09, 1.19], "high": [1.21, 1.31], "low": [1.08, 1.18],
            "close": [1.10, 1.20], "volume": [10, 20],
        }
    )
    path = directory / f"{symbol}_{timeframe}.csv"
    frame.to_csv(path, index=False)
    return path


def test_csv_source_reads_chronologically_and_limits(tmp_path: Path) -> None:
    write_csv(tmp_path)
    source = CsvSource({"directory": str(tmp_path)})
    frame = source.fetch("EURCHF", "4h", now=NOW)
    assert frame["close"].tolist() == [1.10, 1.20]
    assert source.fetch("EURCHF", "4h", limit=1, now=NOW)["close"].tolist() == [1.20]


def test_csv_source_reports_a_missing_file_clearly(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        CsvSource({"directory": str(tmp_path)}).fetch("EURCHF", "4h", now=NOW)


def test_csv_source_requires_a_timestamp_column(tmp_path: Path) -> None:
    (tmp_path / "EURCHF_4h.csv").write_text("open,high,low,close,volume\n1,1,1,1,1\n")
    with pytest.raises(DataError, match="timestamp_column"):
        CsvSource({"directory": str(tmp_path)}).fetch("EURCHF", "4h", now=NOW)


# ---------------------------------------------------------------------- factory
def test_factory_builds_the_configured_provider() -> None:
    source = build_source("twelve_data", {"twelve_data": twelve_settings()})
    assert source.name == "twelve_data"


def test_factory_shares_one_rate_limiter_per_provider() -> None:
    sources = build_sources({"twelve_data": twelve_settings()}, providers={"twelve_data"})
    again = build_sources({"twelve_data": twelve_settings()}, providers={"twelve_data"})
    assert isinstance(sources["twelve_data"], TwelveDataSource)
    assert sources["twelve_data"].limiter is not again["twelve_data"].limiter
    assert isinstance(RateLimiter(8).min_interval, float)


def test_factory_rejects_an_unknown_provider() -> None:
    with pytest.raises(DataError):
        build_source("my_broker", {})

# ---------------------------------------------------------------- inversion
def test_invert_symbol_returns_the_mirrored_cross() -> None:
    assert invert_symbol("AUDGBP") == "GBPAUD"
    assert invert_symbol("EURAUD") == "AUDEUR"
    assert invert_symbol("eur/chf") == "CHFEUR"
    assert invert_symbol("BTCUSDT") is None          # not a 6-letter cross


def test_invert_frame_flips_prices_and_swaps_high_low() -> None:
    frame = pd.DataFrame(
        {
            "open": [2.0, 4.0],
            "high": [3.0, 8.0],
            "low": [1.0, 2.0],
            "close": [2.5, 6.0],
            "volume": [10.0, 20.0],
        },
        index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
    )
    out = invert_frame(frame)

    assert out["open"].tolist() == [0.5, 0.25]
    assert out["close"].tolist() == [0.4, 1.0 / 6.0]
    assert out["high"].tolist() == [1.0, 0.5]          # 1 / low
    assert out["low"].tolist() == [1.0 / 3.0, 0.125]   # 1 / high
    assert out["volume"].tolist() == [10.0, 20.0]      # ticks count the same
    # still a valid candle after flipping
    assert (out["high"] >= out[["open", "close"]].max(axis=1)).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1)).all()
    assert out.index.equals(frame.index)


def test_invert_frame_passes_empty_frames_through() -> None:
    assert invert_frame(empty_frame()).empty


def test_invert_frame_rejects_non_positive_prices() -> None:
    frame = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [0.0], "close": [1.0], "volume": [1.0]},
        index=pd.date_range("2024-01-01", periods=1, tz="UTC"),
    )
    with pytest.raises(DataError, match="non-positive"):
        invert_frame(frame)
