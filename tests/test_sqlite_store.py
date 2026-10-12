"""Persistence: candle round-trip, state restore, and cross-restart dedupe."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from forexpert.store.sqlite_store import InMemoryStateStore, SqliteStore, to_epoch

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def candle_frame(count: int = 5) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=count, freq="4h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [1.1 + i * 0.001 for i in range(count)],
            "high": [1.105 + i * 0.001 for i in range(count)],
            "low": [1.095 + i * 0.001 for i in range(count)],
            "close": [1.102 + i * 0.001 for i in range(count)],
            "volume": [100.0 + i for i in range(count)],
        },
        index=index,
    )
    frame.index.name = "time"
    return frame


@pytest.fixture
def store(tmp_path):
    with SqliteStore(tmp_path / "test.sqlite") as opened:
        yield opened


def test_candles_round_trip_in_chronological_order(store: SqliteStore) -> None:
    store.upsert_candles("EURUSD", "4h", candle_frame(5))
    loaded = store.load_candles("EURUSD", "4h")
    assert len(loaded) == 5
    assert loaded.index.is_monotonic_increasing
    assert loaded.index.tz is not None
    assert loaded["close"].iloc[0] == pytest.approx(1.102)


def test_upsert_replaces_instead_of_duplicating(store: SqliteStore) -> None:
    store.upsert_candles("EURUSD", "4h", candle_frame(5))
    store.upsert_candles("EURUSD", "4h", candle_frame(5))
    assert store.candle_count("EURUSD", "4h") == 5


def test_load_limit_returns_the_newest_bars(store: SqliteStore) -> None:
    store.upsert_candles("EURUSD", "4h", candle_frame(10))
    loaded = store.load_candles("EURUSD", "4h", limit=3)
    assert len(loaded) == 3
    assert loaded.index[-1] == pd.Timestamp("2024-01-02T12:00:00Z")
    assert loaded.index[0] == pd.Timestamp("2024-01-02T04:00:00Z")
    assert loaded.index.is_monotonic_increasing


def test_symbols_and_timeframes_do_not_mix(store: SqliteStore) -> None:
    store.upsert_candles("EURUSD", "4h", candle_frame(3))
    store.upsert_candles("EURUSD", "1d", candle_frame(3))
    assert store.candle_count("EURUSD", "4h") == 3
    assert store.candle_count("EURUSD", "1d") == 3


def test_empty_store_returns_a_typed_empty_frame(store: SqliteStore) -> None:
    loaded = store.load_candles("GBPUSD", "4h")
    assert loaded.empty
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index.tz is not None


def test_state_survives_a_new_connection(store: SqliteStore, tmp_path) -> None:
    store.save_state("EURUSD:4h:buy", {"phase": "waiting_touch", "bars_since_candle2": 4})
    reopened = SqliteStore(store.path)
    try:
        assert reopened.load_state("EURUSD:4h:buy")["bars_since_candle2"] == 4
    finally:
        reopened.close()


def test_missing_state_returns_none(store: SqliteStore) -> None:
    assert store.load_state("nothing:here") is None


def test_signal_ids_are_unique_across_reconnects(store: SqliteStore, tmp_path) -> None:
    payload = {"entry": 1.1}
    store.record_signal("abc123", "EURUSD", "4h", "buy", BASE, payload)
    store.record_signal("abc123", "EURUSD", "4h", "buy", BASE, payload)
    assert store.has_signal("abc123")
    assert len(store.signals("EURUSD")) == 1


def test_processed_bars_make_ingestion_idempotent(store: SqliteStore) -> None:
    bar_time = BASE + timedelta(hours=4)
    store.mark_bar_processed("EURUSD", "4h", bar_time)
    store.mark_bar_processed("EURUSD", "4h", bar_time)
    assert store.bar_is_processed("EURUSD", "4h", bar_time)
    assert store.last_processed_time("EURUSD", "4h") == bar_time


def test_unprocessed_bars_only_returns_new_candles(store: SqliteStore) -> None:
    frame = candle_frame(4)
    store.upsert_candles("EURUSD", "4h", frame)
    store.mark_bar_processed("EURUSD", "4h", frame.index[0].to_pydatetime())
    store.mark_bar_processed("EURUSD", "4h", frame.index[1].to_pydatetime())

    pending = store.unprocessed_bars("EURUSD", "4h")
    assert pending == [
        frame.index[2].to_pydatetime(),
        frame.index[3].to_pydatetime(),
    ]
    assert store.unprocessed_bars("EURUSD", "4h", since=frame.index[2].to_pydatetime()) == [
        frame.index[3].to_pydatetime()
    ]


def test_in_memory_store_matches_the_sqlite_semantics() -> None:
    store = InMemoryStateStore()
    store.save_state("k", {"a": 1})
    store.record_signal("id1", "EURUSD", "4h", "buy", BASE, {})
    store.mark_bar_processed("EURUSD", "4h", BASE)
    assert store.load_state("k") == {"a": 1}
    assert store.has_signal("id1")
    assert store.last_processed_time("EURUSD", "4h") == BASE


def test_epoch_conversion_is_utc() -> None:
    assert to_epoch(datetime(2024, 1, 1, tzinfo=timezone.utc)) == 1_704_067_200
    naive = pd.Timestamp("2024-01-01T00:00:00")
    assert to_epoch(naive) == 1_704_067_200