"""SQLite persistence: candles, per-setup state, signals and processed bars.

Design points that matter for correctness:

  * ``processed_bars`` makes ingestion idempotent across restarts. A bar that
    already fed the state machine is never fed again, so a crash-and-restart
    cannot re-fire an alert.
  * ``signals.signal_id`` is the deterministic id from
    ``strategy.state.make_signal_id``, so duplicate suppression works across
    processes rather than only within one memory space.
  * All timestamps are stored as UTC epoch seconds.
  * WAL mode so a long-running bot does not block on readers (the backtester).
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from ..data.base import empty_frame

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol      TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    open_time   INTEGER NOT NULL,
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      REAL,
    PRIMARY KEY (symbol, timeframe, open_time)
);

CREATE TABLE IF NOT EXISTS strategy_state (
    key        TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id  TEXT PRIMARY KEY,
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    direction  TEXT NOT NULL,
    bar_time   INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    payload    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processed_bars (
    symbol    TEXT    NOT NULL,
    timeframe TEXT    NOT NULL,
    open_time INTEGER NOT NULL,
    PRIMARY KEY (symbol, timeframe, open_time)
);

CREATE INDEX IF NOT EXISTS idx_candles_lookup ON candles (symbol, timeframe, open_time);
CREATE INDEX IF NOT EXISTS idx_signals_time   ON signals (bar_time);
"""


def to_epoch(value: datetime | pd.Timestamp) -> int:
    moment = pd.Timestamp(value)
    if moment.tz is None:
        moment = moment.tz_localize("UTC")
    return int(moment.timestamp())


def from_epoch(value: int) -> datetime:
    return datetime.fromtimestamp(value, tz=timezone.utc)


class StateStore(Protocol):
    """Interface the runner/state machine depend on, so persistence stays swappable."""

    def upsert_candles(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> int: ...

    def load_state(self, key: str) -> dict[str, Any] | None: ...

    def save_state(self, key: str, payload: dict[str, Any]) -> None: ...

    def has_signal(self, signal_id: str) -> bool: ...

    def record_signal(self, signal_id: str, symbol: str, timeframe: str,
                      direction: str, bar_time: datetime, payload: dict[str, Any]) -> None: ...

    def mark_bar_processed(self, symbol: str, timeframe: str, open_time: datetime) -> None: ...

    def last_processed_time(self, symbol: str, timeframe: str) -> datetime | None: ...


class InMemoryStateStore:
    """Test double / ephemeral fallback with the same semantics as SQLite."""

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}
        self._signals: dict[str, dict[str, Any]] = {}
        self._bars: dict[tuple[str, str], int] = {}
        self._candles: dict[tuple[str, str], pd.DataFrame] = {}

    # ------------------------------------------------------------------ candles
    def upsert_candles(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        key = (symbol, timeframe)
        existing = self._candles.get(key)
        merged = pd.concat([existing, frame]) if existing is not None else frame
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.index.name = "time"
        self._candles[key] = merged
        return len(frame)

    def load_candles(
        self, symbol: str, timeframe: str, limit: int | None = None
    ) -> pd.DataFrame:
        frame = self._candles.get((symbol, timeframe))
        if frame is None:
            return empty_frame()
        return frame.iloc[-int(limit):] if limit else frame

    # -------------------------------------------------------------------- state
    def load_state(self, key: str) -> dict[str, Any] | None:
        return self._states.get(key)

    def save_state(self, key: str, payload: dict[str, Any]) -> None:
        self._states[key] = payload

    def has_signal(self, signal_id: str) -> bool:
        return signal_id in self._signals

    def record_signal(self, signal_id: str, symbol: str, timeframe: str,
                      direction: str, bar_time: datetime, payload: dict[str, Any]) -> None:
        self._signals[signal_id] = {
            "symbol": symbol, "timeframe": timeframe, "direction": direction,
            "bar_time": to_epoch(bar_time), "payload": payload,
        }

    def mark_bar_processed(self, symbol: str, timeframe: str, open_time: datetime) -> None:
        self._bars[(symbol, timeframe)] = max(
            self._bars.get((symbol, timeframe), 0), to_epoch(open_time)
        )

    def last_processed_time(self, symbol: str, timeframe: str) -> datetime | None:
        value = self._bars.get((symbol, timeframe))
        return from_epoch(value) if value is not None else None


class SqliteStore:
    """SQLite implementation of candles + :class:`StateStore`."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SqliteStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ candles
    def upsert_candles(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> int:
        """Insert or replace closed candles. Returns the number of rows written."""
        if frame.empty:
            return 0
        if "time" in frame.columns:
            times = pd.to_datetime(frame["time"], utc=True)
        else:
            times = pd.DatetimeIndex(frame.index)
        rows = [
            (
                symbol,
                timeframe,
                to_epoch(t),
                float(r.open),
                float(r.high),
                float(r.low),
                float(r.close),
                None if pd.isna(r.volume) else float(r.volume),
            )
            for t, r in zip(times, frame.itertuples())
        ]
        with self._connection:
            self._connection.executemany(
                "INSERT OR REPLACE INTO candles "
                "(symbol, timeframe, open_time, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def load_candles(
        self, symbol: str, timeframe: str, limit: int | None = None
    ) -> pd.DataFrame:
        """Load candles oldest-first as the canonical OHLCV frame.

        ``limit`` keeps the NEWEST N candles (the ones a live bot needs), then
        still returns them in chronological order.
        """
        columns = "open_time, open, high, low, close, volume"
        if limit:
            sql = (
                f"SELECT {columns} FROM ("
                f"SELECT {columns} FROM candles WHERE symbol = ? AND timeframe = ? "
                f"ORDER BY open_time DESC LIMIT {int(limit)}"
                f") ORDER BY open_time"
            )
        else:
            sql = (
                f"SELECT {columns} FROM candles "
                "WHERE symbol = ? AND timeframe = ? ORDER BY open_time"
            )
        rows = self._connection.execute(sql, (symbol, timeframe)).fetchall()
        if not rows:
            empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            empty.index = pd.DatetimeIndex([], tz="UTC", name="time")
            return empty
        frame = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
        frame.index = pd.to_datetime(frame.pop("open_time"), unit="s", utc=True)
        frame.index.name = "time"
        return frame

    def candle_count(self, symbol: str, timeframe: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS n FROM candles WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        ).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------- state
    def load_state(self, key: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT payload FROM strategy_state WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def save_state(self, key: str, payload: dict[str, Any]) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO strategy_state (key, payload, updated_at) "
                "VALUES (?, ?, ?)",
                (key, json.dumps(payload), int(time.time())),
            )

    # ----------------------------------------------------------------- signals
    def has_signal(self, signal_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM signals WHERE signal_id = ?", (signal_id,)
        ).fetchone()
        return row is not None

    def record_signal(
        self,
        signal_id: str,
        symbol: str,
        timeframe: str,
        direction: str,
        bar_time: datetime,
        payload: dict[str, Any],
    ) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO signals "
                "(signal_id, symbol, timeframe, direction, bar_time, created_at, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    signal_id, symbol, timeframe, direction,
                    to_epoch(bar_time), int(time.time()), json.dumps(payload),
                ),
            )

    def signals(self, symbol: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT payload FROM signals"
        params: tuple[Any, ...] = ()
        if symbol:
            sql += " WHERE symbol = ?"
            params = (symbol,)
        sql += " ORDER BY bar_time"
        return [json.loads(r["payload"]) for r in self._connection.execute(sql, params)]

    # ----------------------------------------------------------- processed bars
    def mark_bar_processed(self, symbol: str, timeframe: str, open_time: datetime) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO processed_bars (symbol, timeframe, open_time) "
                "VALUES (?, ?, ?)",
                (symbol, timeframe, to_epoch(open_time)),
            )

    def bar_is_processed(self, symbol: str, timeframe: str, open_time: datetime) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM processed_bars WHERE symbol = ? AND timeframe = ? AND open_time = ?",
            (symbol, timeframe, to_epoch(open_time)),
        ).fetchone()
        return row is not None

    def last_processed_time(self, symbol: str, timeframe: str) -> datetime | None:
        row = self._connection.execute(
            "SELECT MAX(open_time) AS last FROM processed_bars WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        ).fetchone()
        return from_epoch(int(row["last"])) if row and row["last"] is not None else None

    def unprocessed_bars(
        self, symbol: str, timeframe: str, since: datetime | None = None
    ) -> list[datetime]:
        """Closed candle times present in the store that the machine has not seen."""
        sql = (
            "SELECT c.open_time FROM candles c LEFT JOIN processed_bars p "
            "ON p.symbol = c.symbol AND p.timeframe = c.timeframe AND p.open_time = c.open_time "
            "WHERE c.symbol = ? AND c.timeframe = ? AND p.open_time IS NULL"
        )
        params: list[Any] = [symbol, timeframe]
        if since is not None:
            sql += " AND c.open_time > ?"
            params.append(to_epoch(since))
        sql += " ORDER BY c.open_time"
        return [from_epoch(int(row["open_time"])) for row in self._connection.execute(sql, params)]


def load_or_create_state(store: StateStore, key: str, factory: Any) -> Any:
    """Rebuild a `SetupState` from a store, or create a fresh one."""
    payload = store.load_state(key)
    if payload is None:
        return factory()
    from ..strategy.state import SetupState

    return SetupState.from_dict(payload)