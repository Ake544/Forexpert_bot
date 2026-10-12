"""MetaTrader 5 source tests — the terminal is faked, nothing touches a live account."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from forexpert.data import build_source
from forexpert.data.base import OHLCV_COLUMNS, DataError, SymbolNotFound
from forexpert.data.mt5_source import Mt5Source, available_symbols

# Real MT5 constant values, so an accidental numeric mix-up would be caught.
TF = {
    "TIMEFRAME_M1": 1,
    "TIMEFRAME_M3": 3,
    "TIMEFRAME_M5": 5,
    "TIMEFRAME_M15": 15,
    "TIMEFRAME_M30": 30,
    "TIMEFRAME_H1": 16385,
    "TIMEFRAME_H2": 16386,
    "TIMEFRAME_H4": 16388,
    "TIMEFRAME_H6": 16390,
    "TIMEFRAME_H12": 16396,
    "TIMEFRAME_D1": 16408,
    "TIMEFRAME_W1": 32769,
}

RATE_DTYPE = np.dtype(
    [
        ("time", "<i8"),
        ("open", "<f8"),
        ("high", "<f8"),
        ("low", "<f8"),
        ("close", "<f8"),
        ("tick_volume", "<i8"),
        ("real_volume", "<i8"),
        ("spread", "<i8"),
    ]
)


_NO_RATES = object()


def make_rates(
    bars: int = 6,
    start: str = "2024-01-15 08:00:00+00:00",
    step_minutes: int = 240,
    tick_volume: int = 100,
    real_volume: int = 0,
    volume_stride: int = 1,
) -> np.ndarray:
    """Chronological MT5-style bars: last element is the current forming bar."""
    rows = []
    moment = pd.Timestamp(start)
    for i in range(bars):
        rows.append(
            (
                int(moment.timestamp()),
                1.10 + i * 0.001,
                1.11 + i * 0.001,
                1.09 + i * 0.001,
                1.105 + i * 0.001,
                tick_volume + i * volume_stride,
                real_volume,
                12,
            )
        )
        moment = moment + pd.Timedelta(step_minutes, unit="m")
    return np.array(rows, dtype=RATE_DTYPE)


class FakeMt5:
    """Implements exactly the surface Mt5Source touches."""

    def __init__(
        self,
        rates: Any = _NO_RATES,
        symbols: tuple[str, ...] = ("EURCHF", "BTCUSDT"),
        init_ok: bool = True,
        last_error: tuple[int, str] = (0, "No error"),
        auto_step: bool = False,
    ) -> None:
        for name, value in TF.items():
            setattr(self, name, value)
        self.rates = make_rates() if rates is _NO_RATES else rates
        self.auto_step = auto_step
        self.symbols = symbols
        self.init_ok = init_ok
        self._last_error = last_error
        self.initialize_calls: list[str | None] = []
        self.login_calls: list[dict[str, object]] = []
        self.copy_calls: list[tuple[str, int, int, int]] = []
        self.select_calls: list[tuple[str, bool]] = []
        self.shutdown_calls = 0

    # --- session -----------------------------------------------------------
    def initialize(self, path: str | None = None) -> bool:
        self.initialize_calls.append(path)
        return self.init_ok

    def login(self, login: int, password: str = "", server: str = "") -> bool:
        self.login_calls.append({"login": login, "password": password, "server": server})
        return True

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def last_error(self) -> tuple[int, str]:
        return self._last_error

    def terminal_info(self) -> SimpleNamespace:
        return SimpleNamespace(path="C:/MT5/terminal64.exe")

    def account_info(self) -> SimpleNamespace:
        return SimpleNamespace(login=424242)

    # --- symbols -----------------------------------------------------------
    def symbol_info(self, name: str) -> SimpleNamespace | None:
        if name not in self.symbols:
            return None
        return SimpleNamespace(name=name, visible=True)

    def symbol_select(self, name: str, enable: bool) -> bool:
        self.select_calls.append((name, enable))
        return enable

    def symbols_get(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=s) for s in self.symbols]

    # --- bars --------------------------------------------------------------
    def copy_rates_from_pos(
        self, symbol: str, timeframe: int, start_pos: int, count: int
    ) -> np.ndarray | None:
        self.copy_calls.append((symbol, timeframe, start_pos, count))
        if self.rates is None:
            return None
        if self.auto_step:
            # space the bars properly for whatever timeframe was requested,
            # otherwise a "daily" fetch would be fed 4-hour timestamps
            step = 1440 if timeframe == self.TIMEFRAME_D1 else 240
            rates = make_rates(bars=max(count, 6), step_minutes=step)
        else:
            rates = self.rates
        return rates[-count:] if count else rates


# now sits inside the final bar (04:00 -> 08:00), so that bar must be dropped.
NOW = pd.Timestamp("2024-01-16 05:30:00+00:00")


def source(mt5: FakeMt5 | None = None, **settings: object) -> Mt5Source:
    merged: dict[str, object] = {"requests_per_minute": 10**9}
    merged.update(settings)
    return Mt5Source(merged, mt5=mt5 if mt5 is not None else FakeMt5())


def test_fetch_builds_canonical_frame_with_tick_volume() -> None:
    fake = FakeMt5()
    frame = source(fake).fetch("EURCHF", "4h", 6, now=NOW)

    assert list(frame.columns) == list(OHLCV_COLUMNS)
    assert frame.index.tz is not None
    assert frame.index.is_monotonic_increasing
    # the forming bar is gone, the five closed bars remain
    assert len(frame) == 5
    assert frame.index[-1] == pd.Timestamp("2024-01-16 00:00:00+00:00")
    # tick_volume is what TradingView shows for FX; real_volume is 0 here,
    # so any nonzero volume proves the tick field was used
    assert frame["volume"].iloc[0] == 100
    assert frame["volume"].iloc[-1] == 104
    assert (frame["volume"] > 0).all()
    assert frame.dtypes.eq(float).all() or all(
        str(dt).startswith("float") for dt in frame.dtypes
    )
    assert fake.copy_calls == [("EURCHF", TF["TIMEFRAME_H4"], 0, 6)]


def test_forming_bar_is_dropped_by_time_not_by_position() -> None:
    fake = FakeMt5()
    source = Mt5Source({"requests_per_minute": 10**9}, mt5=fake)
    # now after the last bar has closed -> nothing is forming, all bars survive
    frame = source.fetch("EURCHF", "4h", 6, now=pd.Timestamp("2024-01-16 09:00:00+00:00"))
    assert len(frame) == 6
    # now inside the 4th bar -> bars 4..5 are still forming
    partial = source.fetch("EURCHF", "4h", 6, now=pd.Timestamp("2024-01-15 21:00:00+00:00"))
    assert len(partial) == 3
    assert partial.index[-1] == pd.Timestamp("2024-01-15 16:00:00+00:00")


def test_real_volume_mode_reads_the_real_volume_field() -> None:
    fake = FakeMt5(rates=make_rates(tick_volume=0, real_volume=7, volume_stride=0))
    frame = source(fake, volume_mode="real_volume").fetch("EURCHF", "4h", 6, now=NOW)
    assert (frame["volume"] == 7).all()


def test_symbol_suffix_is_applied() -> None:
    fake = FakeMt5(symbols=("EURCHF.pro",))
    frame = source(fake, symbol_suffix=".pro").fetch("EURCHF", "4h", 6, now=NOW)
    assert fake.copy_calls[0][0] == "EURCHF.pro"
    assert len(frame) == 5


def test_terminal_is_initialized_once_per_session_and_shut_down() -> None:
    fake = FakeMt5(auto_step=True)
    src = source(fake)
    later = pd.Timestamp("2024-01-21 09:00:00+00:00")  # every 4h and 1d bar closed
    src.fetch("EURCHF", "4h", 6, now=later)
    src.fetch("EURCHF", "1d", 6, now=later)
    src.close()

    assert fake.initialize_calls == [None]
    assert fake.shutdown_calls == 1
    assert [call[1] for call in fake.copy_calls] == [TF["TIMEFRAME_H4"], TF["TIMEFRAME_D1"]]


def test_close_is_idempotent() -> None:
    fake = FakeMt5()
    src = source(fake)
    src.fetch("EURCHF", "4h", 6, now=NOW)
    src.close()
    src.close()
    assert fake.shutdown_calls == 1


def test_credentials_from_env_are_used_when_provided() -> None:
    fake = FakeMt5()
    source(
        fake, login=42, password="hunter2", server="Broker-Demo"
    ).fetch("EURCHF", "4h", 6, now=NOW)

    assert fake.initialize_calls == [None]
    assert fake.login_calls == [
        {"login": 42, "password": "hunter2", "server": "Broker-Demo"}
    ]


def test_missing_terminal_raises_a_guided_error() -> None:
    fake = FakeMt5(init_ok=False)
    with pytest.raises(DataError, match="initialize failed"):
        source(fake).fetch("EURCHF", "4h", 6, now=NOW)


def test_unknown_broker_symbol_raises_symbol_not_found() -> None:
    with pytest.raises(SymbolNotFound, match="symbol_suffix"):
        source().fetch("EURNZD", "4h", 6, now=NOW)


def test_inverse_pair_is_fetched_when_the_broker_only_quotes_it() -> None:
    fake = FakeMt5(symbols=("GBPAUD",))
    frame = source(fake).fetch("AUDGBP", "4h", 6, now=NOW)

    # prices are 1/GBPAUD, volume is the same ticks either way
    raw = pd.DataFrame(make_rates())
    expected_close = 1.0 / float(raw["close"].iloc[0])
    assert frame["close"].iloc[0] == pytest.approx(expected_close)
    assert frame["volume"].iloc[0] == float(raw["tick_volume"].iloc[0])
    assert fake.copy_calls[0][0] == "GBPAUD"


def test_inversion_swaps_high_and_low() -> None:
    fake = FakeMt5(symbols=("GBPAUD",))
    frame = source(fake).fetch("AUDGBP", "4h", 6, now=NOW)
    raw = pd.DataFrame(make_rates())

    assert frame["high"].iloc[0] == pytest.approx(1.0 / float(raw["low"].iloc[0]))
    assert frame["low"].iloc[0] == pytest.approx(1.0 / float(raw["high"].iloc[0]))
    # still a valid candle
    assert (frame["high"] >= frame[["open", "close"]].max(axis=1)).all()
    assert (frame["low"] <= frame[["open", "close"]].min(axis=1)).all()
    assert frame.index.is_monotonic_increasing


def test_missing_pair_with_missing_inverse_still_raises_the_original_error() -> None:
    with pytest.raises(SymbolNotFound, match="EURNZD"):
        source(FakeMt5(symbols=("EURCHF",))).fetch("EURNZD", "4h", 6, now=NOW)


def test_no_rates_raises_data_error_with_terminal_status() -> None:
    fake = FakeMt5(rates=None, last_error=(54001, "Invalid symbol"))
    with pytest.raises(DataError, match="no rates.*54001",):
        source(fake).fetch("EURCHF", "4h", 6, now=NOW)


def test_mt5_has_no_2h_but_the_error_names_what_is_supported() -> None:
    with pytest.raises(DataError, match="no timeframe '3h'"):
        source().fetch("EURCHF", "3h", 6, now=NOW)


def test_unsupported_broker_timeframe_is_rejected_for_missing_constant() -> None:
    fake = FakeMt5()
    fake.TIMEFRAME_W1 = None  # a terminal build without that constant
    with pytest.raises(DataError, match="no timeframe '1w'"):
        source(fake).fetch("EURCHF", "1w", 6, now=NOW)


def test_unknown_volume_mode_fails_fast() -> None:
    with pytest.raises(DataError, match="volume_mode"):
        Mt5Source({"volume_mode": "bid"}, mt5=FakeMt5())


def test_all_zero_volume_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeMt5(rates=make_rates(tick_volume=0, real_volume=0, volume_stride=0))
    with caplog.at_level("WARNING", logger="forexpert.data.mt5_source"):
        frame = source(fake).fetch("EURCHF", "4h", 6, now=NOW)
    assert (frame["volume"] == 0).all()
    assert "zero tick_volume" in caplog.text


def test_factory_returns_an_mt5_source() -> None:
    built = build_source("mt5", {"mt5": {"symbol_suffix": ".m", "volume_mode": "real_volume"}})
    assert isinstance(built, Mt5Source)
    assert built.symbol_for("nzdchf") == "NZDCHF.m"


def test_available_symbols_lists_the_terminal() -> None:
    fake = FakeMt5(symbols=("NZDCHF", "AUDCAD", "BTCUSDT"))
    assert available_symbols(fake) == ["AUDCAD", "BTCUSDT", "NZDCHF"]
