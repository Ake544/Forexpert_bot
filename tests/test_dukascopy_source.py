"""Dukascopy tick-source tests. Synthetic bi5 payloads, no network.

The bi5 format is the part most likely to be silently wrong, so these tests pin
down the three things that break quietly: the hour-relative offsets, the price
scale, and the UTC bar alignment.
"""
from __future__ import annotations

import lzma
import struct
import time
from pathlib import Path

import pandas as pd
import pytest

from forexpert.data import DataError, build_source
from forexpert.data.dukascopy import (
    DukascopySource,
    Tick,
    decode_bi5,
    hour_url,
    iter_hours,
    normalize_symbol,
)

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
NOW = pd.Timestamp("2024-01-02T00:00:00Z")


class RawResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code


class FakeSession:
    """Serves an hour->payload map, recording every URL that was requested."""

    def __init__(self, payloads: dict[str, bytes | None]) -> None:
        self._payloads = payloads
        self.calls: list[str] = []
        self.headers: list[dict[str, str]] = []

    def get(self, url, timeout=None, headers=None):
        self.calls.append(url)
        self.headers.append(dict(headers or {}))
        if url not in self._payloads:
            return RawResponse(b"", 404)
        payload = self._payloads[url]
        if payload is None:
            return RawResponse(b"", 404)
        return RawResponse(payload)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)


def bi5(ticks: list[tuple[int, int, int, float, float]]) -> bytes:
    """Pack (offset_ms, ask, bid, ask_size, bid_size) into a compressed hour."""
    return lzma.compress(b"".join(struct.pack(">IIIff", *t) for t in ticks))


def settings(**overrides) -> dict:
    base = {
        "base_url": BASE_URL,
        "requests_per_minute": 10_000,
        "max_retries": 1,
        "retry_backoff_seconds": 0.0,
        "max_workers": 1,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------- encoding
def test_url_uses_a_zero_based_month() -> None:
    url = hour_url(BASE_URL, "EURCHF", pd.Timestamp("2024-01-15T13:00:00Z"))
    assert url == f"{BASE_URL}/EURCHF/2024/00/15/13h_ticks.bi5"
    march = hour_url(BASE_URL, "EURCHF", pd.Timestamp("2024-03-05T00:00:00Z"))
    assert "/2024/02/05/00h_ticks.bi5" in march


@pytest.mark.parametrize("raw", ["EURCHF", "eur/chf", " EUR-CHF "])
def test_symbol_normalization(raw: str) -> None:
    assert normalize_symbol(raw) == "EURCHF"


def test_iter_hours_is_inclusive_of_both_ends() -> None:
    start = pd.Timestamp("2024-01-01T22:30:00Z")
    end = pd.Timestamp("2024-01-02T01:15:00Z")
    hours = list(iter_hours(start, end))
    assert [str(h) for h in hours] == [
        "2024-01-01 22:00:00+00:00", "2024-01-01 23:00:00+00:00",
        "2024-01-02 00:00:00+00:00", "2024-01-02 01:00:00+00:00",
    ]


def test_offsets_are_relative_to_the_containing_hour() -> None:
    hour = pd.Timestamp("2024-01-01T13:00:00Z")
    payload = bi5([(0, 1000, 999, 0.5, 0.4), (1_800_000, 1002, 1001, 0.6, 0.5)])
    ticks = decode_bi5(payload, hour)
    assert ticks[0].time == hour
    assert ticks[1].time == pd.Timestamp("2024-01-01T13:30:00Z")


def test_prices_are_scaled_by_one_thousand() -> None:
    payload = bi5([(0, 933, 932, 1.0, 1.0)])
    ticks = decode_bi5(payload, pd.Timestamp("2024-01-01T00:00:00Z"))
    assert ticks[0].ask == pytest.approx(0.933)
    assert ticks[0].bid == pytest.approx(0.932)
    assert ticks[0].mid == pytest.approx(0.9325)


def test_empty_payload_decodes_to_no_ticks() -> None:
    assert decode_bi5(b"") == []


def test_partial_record_at_the_end_of_an_hour_is_rejected() -> None:
    partial = lzma.compress(struct.pack(">IIIff", 0, 1000, 999, 1.0, 1.0)[:-4])
    with pytest.raises(DataError, match="multiple of"):
        decode_bi5(partial, pd.Timestamp("2024-01-01T00:00:00Z"))


def test_corrupt_compressed_payload_is_rejected() -> None:
    with pytest.raises(DataError, match="LZMA"):
        decode_bi5(b"not-compressed-at-all", pd.Timestamp("2024-01-01T00:00:00Z"))


# -------------------------------------------------------------------- fetching
def three_hour_payload() -> dict[str, bytes]:
    """Every hour of 2024-01-01, two ticks each, price rising one pip per hour.

    Prices are built as raw scaled integers rather than ``int(price * 1000)``,
    which would truncate the 0.001 grid and collapse bid/ask onto one another.
    """
    payload = {}
    for index in range(24):
        stamp = pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(index, unit="h")
        base = 930 + index                      # 930 -> 0.930
        payload[hour_url(BASE_URL, "EURCHF", stamp)] = bi5([
            (0, base + 1, base, 1.0, 1.0),
            (3_599_999, base + 2, base + 1, 1.0, 1.0),   # last ms of the hour
        ])
    return payload


def test_bars_are_bucketed_on_utc_boundaries_with_tick_volume() -> None:
    frame = DukascopySource(settings(), session=FakeSession(three_hour_payload())).fetch(
        "EURCHF", "4h", limit=2, now=NOW
    )

    assert list(frame.index) == [
        pd.Timestamp("2024-01-01T16:00:00Z"),
        pd.Timestamp("2024-01-01T20:00:00Z"),
    ]
    assert frame["volume"].tolist() == [8.0, 8.0]   # 2 ticks/hour x 4 hours
    # Last bar covers hours 20..23: mids run 0.9505 -> 0.9545.
    assert frame["open"].iloc[-1] == pytest.approx(0.9505)
    assert frame["close"].iloc[-1] == pytest.approx(0.9545)
    assert frame["low"].iloc[-1] == pytest.approx(frame["open"].iloc[-1])
    assert frame["high"].iloc[-1] == pytest.approx(frame["close"].iloc[-1])
    assert frame["close"].iloc[-1] > frame["close"].iloc[0]


def test_the_forming_bar_is_excluded() -> None:
    frame = DukascopySource(settings(), session=FakeSession(three_hour_payload())).fetch(
        "EURCHF", "4h", limit=10, now=NOW
    )
    assert frame.index[-1] + pd.Timedelta(4, unit="h") <= NOW
    assert frame.index[-1] == pd.Timestamp("2024-01-01T20:00:00Z")


def test_missing_hours_are_skipped_without_failing_the_bar() -> None:
    payloads = three_hour_payload()
    payloads[hour_url(BASE_URL, "EURCHF", pd.Timestamp("2024-01-01T22:00:00Z"))] = None
    frame = DukascopySource(settings(), session=FakeSession(payloads)).fetch(
        "EURCHF", "4h", limit=1, now=NOW
    )
    assert len(frame) == 1
    # The 20:00 bar loses hour 22 only; the other three hours still contribute.
    assert frame["volume"].iloc[0] == pytest.approx(6.0)


def test_requested_limit_is_respected() -> None:
    frame = DukascopySource(settings(), session=FakeSession(three_hour_payload())).fetch(
        "EURCHF", "4h", limit=1, now=NOW
    )
    assert len(frame) == 1


def test_symbol_with_no_ticks_at_all_is_reported_clearly() -> None:
    with pytest.raises(DataError, match="no ticks"):
        DukascopySource(settings(), session=FakeSession({})).fetch("EURCHF", "4h", now=NOW)


def test_unknown_timeframe_is_rejected() -> None:
    with pytest.raises(DataError):
        DukascopySource(settings(), session=FakeSession({})).fetch("EURCHF", "7m", now=NOW)


def test_daily_bars_resample_to_utc_midnight() -> None:
    frame = DukascopySource(settings(), session=FakeSession(three_hour_payload())).fetch(
        "EURCHF", "1d", limit=1, now=NOW
    )
    assert list(frame.index) == [pd.Timestamp("2024-01-01T00:00:00Z")]
    assert frame["volume"].tolist() == [48.0]   # 2 ticks x 24 hours
    assert frame["open"].iloc[0] == pytest.approx(0.9305)
    assert frame["close"].iloc[0] == pytest.approx(0.9545)


def test_volume_mode_size_sums_tick_sizes() -> None:
    frame = DukascopySource(
        settings(volume_mode="size"), session=FakeSession(three_hour_payload())
    ).fetch("EURCHF", "4h", limit=1, now=NOW)
    # Each of the 8 ticks carries bid_size = ask_size = 1.0, so (1+1)/2 = 1.
    assert frame["volume"].iloc[0] == pytest.approx(8.0)


def test_price_field_selects_bid_or_ask() -> None:
    bid_only = DukascopySource(
        settings(price_field="bid"), session=FakeSession(three_hour_payload())
    ).fetch("EURCHF", "4h", limit=1, now=NOW)
    ask_only = DukascopySource(
        settings(price_field="ask"), session=FakeSession(three_hour_payload())
    ).fetch("EURCHF", "4h", limit=1, now=NOW)
    assert ask_only["close"].iloc[0] > bid_only["close"].iloc[0]


def test_unknown_price_field_is_rejected_at_construction() -> None:
    with pytest.raises(DataError):
        DukascopySource(settings(price_field="close"), session=FakeSession({}))


def test_unknown_symbol_maps_to_a_clear_error() -> None:
    class Forbidden:
        def get(self, url, timeout=None, headers=None):
            return RawResponse(b"", 403)

    with pytest.raises(DataError, match="unknown symbol"):
        DukascopySource(settings(), session=Forbidden()).fetch("NOSUCH", "4h", now=NOW)


def test_a_browser_user_agent_is_sent_by_default() -> None:
    """The CDN 429s python-requests' default User-Agent outright."""
    session = FakeSession(three_hour_payload())
    DukascopySource(settings(), session=session).fetch("EURCHF", "4h", limit=1, now=NOW)
    assert "Mozilla/5.0" in session.headers[0]["User-Agent"]


def test_transport_errors_are_retried_then_reported() -> None:
    class Flaky:
        def __init__(self) -> None:
            self.attempts = 0

        def get(self, url, timeout=None, headers=None):
            self.attempts += 1
            raise TimeoutError("connection reset")

    flaky = Flaky()
    with pytest.raises(DataError, match="transport failure"):
        DukascopySource(settings(max_retries=2), session=flaky).fetch("EURCHF", "4h", now=NOW)
    assert flaky.attempts == 3


def test_retry_after_header_is_capped() -> None:
    class Slow:
        def __init__(self) -> None:
            self.attempts = 0

        def get(self, url, timeout=None, headers=None):
            self.attempts += 1
            response = RawResponse(b"", 503)
            response.headers = {"Retry-After": "9999"}
            return response

    with pytest.raises(DataError):
        DukascopySource(settings(max_retries=1), session=Slow()).fetch("EURCHF", "4h", now=NOW)
    assert DukascopySource._retry_after(RawResponse(b"", 503), 2.0) == 2.0


def test_local_file_root_needs_no_network(tmp_path: Path) -> None:
    payload = three_hour_payload()
    for url, blob in payload.items():
        target = tmp_path / url.split("/datafeed/", 1)[-1]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)

    frame = DukascopySource(settings(), root=tmp_path).fetch("EURCHF", "4h", limit=1, now=NOW)
    assert len(frame) == 1


def test_factory_builds_dukascopy() -> None:
    source = build_source("dukascopy", {"dukascopy": settings()})
    assert source.name == "dukascopy"
    assert isinstance(source, DukascopySource)


def test_tick_dataclass_exposes_mid() -> None:
    tick = Tick(time=NOW, bid=1.0, ask=1.2, bid_size=0.0, ask_size=0.0)
    assert tick.mid == pytest.approx(1.1)