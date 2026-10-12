"""Journal, notifier and runner tests. No network: everything is faked."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from forexpert.journal import Journal
from forexpert.notify import (
    DiscordNotifier,
    FanoutNotifier,
    NullNotifier,
    TelegramNotifier,
    format_signal,
)
from forexpert.runner import Runner, SymbolOutcome, iter_bar_inputs, pandas_to_datetime
from forexpert.settings import (
    AppConfig,
    IndicatorParams,
    NotifierParams,
    RiskParams,
    StrategyParams,
    SymbolConfig,
)
from forexpert.store.sqlite_store import InMemoryStateStore
from forexpert.strategy.state import BarInput, Phase, TradeSignal

BASE = pd.Timestamp("2024-01-01", tz="UTC")


# ------------------------------------------------------------------- fixtures
class FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "ok") -> None:
        self.status_code = status_code
        self.text = text
        self.headers: dict[str, str] = {}


class FakeSession:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, json=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "json": json, "timeout": timeout, **kwargs})
        if not self._responses:
            raise AssertionError("provider made more requests than the test queued")
        payload = self._responses.pop(0)
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, FakeResponse):
            return payload
        return FakeResponse(**payload) if isinstance(payload, dict) else payload


class FakeSource:
    name = "fake"

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.fetches = 0

    def fetch(self, symbol, timeframe, limit, now=None):
        self.fetches += 1
        return self.frame


class RecordingNotifier:
    name = "recording"

    def __init__(self, fail: bool = False) -> None:
        self.sent: list[str] = []
        self.fail = fail

    def send(self, text: str) -> bool:
        self.sent.append(text)
        return not self.fail


def candles(count: int = 400, step: str = "4h") -> pd.DataFrame:
    index = pd.date_range(BASE, periods=count, freq=step, tz="UTC")
    close = 1.10 + pd.Series(range(count)).mul(0.0005)
    frame = pd.DataFrame(
        {
            "open": close - 0.0002,
            "high": close + 0.0004,
            "low": close - 0.0004,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )
    frame.index.name = "time"
    return frame


def make_config(
    symbols=(("EURCHF", "4h"),), alerts: dict | None = None, **runtime
) -> AppConfig:
    runtime.setdefault("max_bars_in_memory", 1000)
    runtime.setdefault("max_catchup_bars", 20)
    runtime.setdefault("wake_delay_seconds", 5)
    return AppConfig(
        runtime=runtime,
        data={},
        strategy=StrategyParams(),
        risk=RiskParams(),
        notifier=NotifierParams(),
        journal={"enabled": True, "path": "unused.jsonl"},
        alerts=alerts if alerts is not None else {"stale_data_bars": 3},
        symbols=tuple(
            SymbolConfig(symbol=s, timeframe=tf, provider="fake", profile="default",
                         params=IndicatorParams())
            for s, tf in symbols
        ),
        source_path=Path("config.yaml"),
    )


def make_runner(tmp_path: Path, **kwargs) -> tuple[Runner, RecordingNotifier, Journal]:
    notifier = RecordingNotifier()
    journal = Journal(tmp_path / "journal.jsonl")
    runner = Runner(
        make_config(alerts=kwargs.pop("alerts", None), **kwargs.pop("runtime", {})),
        store=InMemoryStateStore(),
        sources=kwargs.pop("sources", {"fake": FakeSource(candles())}),
        journal=journal,
        notifier=notifier,
        clock=kwargs.pop("clock", lambda: pd.Timestamp("2024-01-06", tz="UTC")),
        **kwargs,
    )
    return runner, notifier, journal


def sample_signal(direction: str = "buy") -> TradeSignal:
    return TradeSignal(
        signal_id="abc123def456",
        symbol="EURCHF", timeframe="4h", direction=direction,
        bar_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        entry=1.10, stop_loss=1.09, take_profit=1.12, atr=0.01,
        sl_distance=0.01, tp_distance=0.02, risk_reward=2.0,
        level=1.10, close=1.10, confirmations=("VSLRT",),
        vwap_bull=True, trend_bull=True, vslrt_bull=True,
        bars_from_candle1=2, bars_from_candle2=1,
        position_size_lots=0.5,
    )


# --------------------------------------------------------------------- journal
def test_journal_writes_one_json_object_per_line(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.jsonl")
    journal.record("startup", {"symbols": 3})
    journal.record("signal", {"signal": {"signal_id": "s1"}})
    lines = (tmp_path / "j.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert journal.events("signal")[0]["signal"]["signal_id"] == "s1"


def test_journal_is_idempotent_to_read_after_a_torn_write(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.jsonl")
    journal.record("ok", {"a": 1})
    with (tmp_path / "j.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"broken": ')
    assert len(journal.events()) == 1


def test_journal_collects_signal_ids_for_dedupe(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.jsonl")
    journal.signal({"signal_id": "one"})
    journal.signal({"signal_id": "two"})
    journal.rejection("EURCHF", "4h", "expired")
    assert journal.signal_ids() == {"one", "two"}


def test_disabled_journal_records_in_memory_without_touching_disk(tmp_path: Path) -> None:
    path = tmp_path / "j.jsonl"
    journal = Journal(path, enabled=False)
    journal.record("noop")
    assert not path.exists()


# --------------------------------------------------------------------- notifier
def test_format_signal_contains_levels_and_identity() -> None:
    text = format_signal(sample_signal())
    assert "BUY EURCHF 4h" in text
    assert "entry  : 1.1" in text
    assert "RR     : 2.00" in text
    assert "lots   : 0.5" in text
    assert "abc123def456" in text


def test_telegram_posts_to_the_bot_endpoint() -> None:
    session = FakeSession([FakeResponse(200)])
    notifier = TelegramNotifier(
        {"bot_token": "TOKEN", "chat_id": "42", "max_retries": 1}, session=session
    )
    assert notifier.send("hello") is True
    call = session.calls[0]
    assert call["url"].endswith("/botTOKEN/sendMessage")
    assert call["json"]["chat_id"] == "42"


def test_telegram_is_inert_when_credentials_are_missing() -> None:
    notifier = TelegramNotifier({"bot_token": "", "chat_id": ""}, session=FakeSession([]))
    assert notifier.send("hello") is False


def test_telegram_retries_transport_failures_then_gives_up() -> None:
    session = FakeSession([TimeoutError("boom")] * 10)
    notifier = TelegramNotifier(
        {"bot_token": "T", "chat_id": "C", "max_retries": 2,
         "retry_backoff_seconds": 0.0},
        session=session,
    )
    assert notifier.send("hello") is False
    assert len(session.calls) == 3


def test_telegram_does_not_retry_a_client_error() -> None:
    session = FakeSession([FakeResponse(403, "Forbidden")])
    notifier = TelegramNotifier(
        {"bot_token": "T", "chat_id": "C", "max_retries": 5}, session=session
    )
    assert notifier.send("hello") is False
    assert len(session.calls) == 1


def test_discord_posts_content() -> None:
    session = FakeSession([FakeResponse(204)])
    notifier = DiscordNotifier(
        {"webhook_url": "https://discord/abc", "max_retries": 1}, session=session
    )
    assert notifier.send("alert") is True
    assert session.calls[0]["json"] == {"content": "alert"}


def test_fanout_delivers_to_every_child() -> None:
    first, second = RecordingNotifier(), RecordingNotifier()
    fanout = FanoutNotifier(first, second)
    assert fanout.send("x") is True
    assert first.sent == ["x"] and second.sent == ["x"]
    assert fanout.delivered == [("recording", True), ("recording", True)]


def test_fanout_keeps_going_when_one_child_fails() -> None:
    ok, bad = RecordingNotifier(), RecordingNotifier(fail=True)
    assert FanoutNotifier(ok, bad).send("x") is True
    assert ok.sent == ["x"] and bad.sent == ["x"]


def test_null_notifier_reports_no_delivery() -> None:
    null = NullNotifier()
    assert null.send("x") is False
    assert null.sent == ["x"]


# ----------------------------------------------------------------------- runner
def test_first_run_baselines_only_the_newest_bar(tmp_path: Path) -> None:
    runner, _notifier, _journal = make_runner(tmp_path)
    outcome = runner.process_symbol(runner.config.symbols[0])
    assert outcome.ok
    assert outcome.bars_processed == 1      # never replay 400 bars on boot


def test_new_bars_are_replayed_and_then_marked_processed(tmp_path: Path) -> None:
    source = FakeSource(candles(400))
    runner, _n, _j = make_runner(tmp_path, sources={"fake": source})
    runner.process_symbol(runner.config.symbols[0])

    source.frame = candles(410)             # ten new closed candles arrive
    outcome = runner.process_symbol(runner.config.symbols[0])
    assert outcome.bars_processed == 10

    outcome = runner.process_symbol(runner.config.symbols[0])  # nothing new
    assert outcome.bars_processed == 0


def test_catch_up_is_bounded_by_max_catchup_bars(tmp_path: Path) -> None:
    source = FakeSource(candles(400))
    runner, _n, _j = make_runner(
        tmp_path, sources={"fake": source}, runtime={"max_catchup_bars": 5}
    )
    runner.process_symbol(runner.config.symbols[0])
    source.frame = candles(500)
    outcome = runner.process_symbol(runner.config.symbols[0])
    assert outcome.bars_processed == 5


def test_a_failing_symbol_does_not_stop_the_others(tmp_path: Path) -> None:
    class ExplodingSource:
        name = "bad"

        def fetch(self, *args, **kwargs):
            raise RuntimeError("provider is down")

    config = AppConfig(
        runtime={"max_bars_in_memory": 1000, "max_catchup_bars": 20},
        data={},
        strategy=StrategyParams(),
        risk=RiskParams(),
        notifier=NotifierParams(),
        journal={"enabled": True, "path": "unused.jsonl"},
        alerts={"stale_data_bars": 3},
        symbols=(
            SymbolConfig("EURCHF", "4h", "bad", "default", IndicatorParams()),
            SymbolConfig("GBPJPY", "4h", "good", "default", IndicatorParams()),
        ),
        source_path=Path("config.yaml"),
    )
    runner = Runner(
        config,
        store=InMemoryStateStore(),
        sources={"bad": ExplodingSource(), "good": FakeSource(candles())},
        journal=Journal(tmp_path / "j.jsonl"),
        notifier=RecordingNotifier(),
    )
    outcomes = {o.symbol: o for o in runner.run_once()}
    assert not outcomes["EURCHF"].ok
    assert "provider is down" in (outcomes["EURCHF"].error or "")
    assert outcomes["GBPJPY"].ok
    assert outcomes["GBPJPY"].bars_processed == 1


def test_signals_are_deduped_across_restarts(tmp_path: Path) -> None:
    store = InMemoryStateStore()
    first = RecordingNotifier()
    runner_a = Runner(
        make_config(), store=store, sources={"fake": FakeSource(candles())},
        journal=Journal(tmp_path / "a.jsonl"), notifier=first,
    )
    outcome_a = SymbolOutcome("EURCHF", "4h")
    runner_a._handle_signal(sample_signal(), outcome_a)
    assert first.sent and store.has_signal("abc123def456")

    # A brand new process restores the same store: the id must not notify again.
    second = RecordingNotifier()
    runner_b = Runner(
        make_config(), store=store, sources={"fake": FakeSource(candles())},
        journal=Journal(tmp_path / "b.jsonl"), notifier=second,
    )
    outcome_b = SymbolOutcome("EURCHF", "4h")
    runner_b._handle_signal(sample_signal(), outcome_b)
    assert second.sent == []
    assert outcome_b.signals == []


def test_handle_signal_persists_journals_and_notifies(tmp_path: Path) -> None:
    runner, notifier, journal = make_runner(tmp_path)
    outcome = SymbolOutcome("EURCHF", "4h")
    runner._handle_signal(sample_signal(), outcome)

    assert runner.store.has_signal("abc123def456")
    assert len(journal.signal_ids()) == 1
    assert len(notifier.sent) == 1
    assert len(outcome.signals) == 1

    runner._handle_signal(sample_signal(), outcome)   # same signal again
    assert len(journal.signal_ids()) == 1
    assert len(notifier.sent) == 1


def test_stale_data_is_journaled_and_notified(tmp_path: Path) -> None:
    """Bars ending 2024-01-01 with a clock at 2024-01-06 are far beyond 3 bars."""
    notifier = RecordingNotifier()
    journal = Journal(tmp_path / "j.jsonl")
    runner = Runner(
        make_config(),
        store=InMemoryStateStore(),
        sources={"fake": FakeSource(candles(20))},
        journal=journal,
        notifier=notifier,
        clock=lambda: pd.Timestamp("2024-01-10", tz="UTC"),
    )
    outcome = runner.process_symbol(runner.config.symbols[0])
    assert outcome.stale
    assert any(e["event"] == "stale_data" for e in journal.events())
    assert any(text.startswith("STALE") for text in notifier.sent)


def test_fresh_data_is_not_flagged_stale(tmp_path: Path) -> None:
    source = FakeSource(candles(400))
    runner, notifier, journal = make_runner(tmp_path, sources={"fake": source})
    outcome = runner.process_symbol(runner.config.symbols[0])
    assert not outcome.stale
    assert notifier.sent == []


def test_run_once_collects_outcomes_per_symbol(tmp_path: Path) -> None:
    config = make_config(symbols=(("EURCHF", "4h"), ("GBPJPY", "4h")))
    runner = Runner(
        config,
        store=InMemoryStateStore(),
        sources={"fake": FakeSource(candles())},
        journal=Journal(tmp_path / "j.jsonl"),
        notifier=RecordingNotifier(),
    )
    outcomes = runner.run_once()
    assert [o.symbol for o in outcomes] == ["EURCHF", "GBPJPY"]
    assert all(o.ok for o in outcomes)
    assert all(o.bars_processed == 1 for o in outcomes)


def test_seconds_until_next_close_is_bounded_and_positive(tmp_path: Path) -> None:
    runner, _n, _j = make_runner(tmp_path, clock=lambda: pd.Timestamp("2024-01-01T02:00", tz="UTC"))
    delay = runner.seconds_until_next_close()
    assert 0 < delay <= 4 * 3600 + 5


# ------------------------------------------------------------- startup/errors
def test_startup_alert_is_sent_once_per_process(tmp_path: Path) -> None:
    runner, notifier, journal = make_runner(tmp_path)
    runner.run_once()
    runner.run_once()

    started = [text for text in notifier.sent if text.startswith("Forexpert started")]
    assert len(started) == 1
    assert "providers fake" in started[0]
    assert "notifier recording" in started[0]
    assert [e["event"] for e in journal.events()].count("startup") == 1


def test_startup_alert_can_be_disabled(tmp_path: Path) -> None:
    runner, notifier, _journal = make_runner(
        tmp_path, alerts={"stale_data_bars": 3, "notify_on_startup": False}
    )
    runner.run_once()
    assert not any(text.startswith("Forexpert started") for text in notifier.sent)


def test_error_alerts_respect_the_cooldown_per_symbol(tmp_path: Path) -> None:
    clock = {"now": pd.Timestamp("2024-01-06", tz="UTC")}
    runner, notifier, journal = make_runner(
        tmp_path,
        sources={},   # no source registered for provider "fake" -> every pass fails
        clock=lambda: clock["now"],
        alerts={"stale_data_bars": 3, "error_cooldown_minutes": 30},
    )

    def errors() -> list[str]:
        return [t for t in notifier.sent if t.startswith("ERROR ")]

    runner.run_once()
    assert len(errors()) == 1
    assert "no candle source" in errors()[0]

    runner.run_once()                       # still inside the 30 minute window
    assert len(errors()) == 1

    clock["now"] = clock["now"] + pd.Timedelta(31, unit="m")
    runner.run_once()                       # window elapsed -> alert again
    assert len(errors()) == 2
    assert [e["event"] for e in journal.events()].count("error_alert") == 2


# ------------------------------------------------------------------ bar inputs
def test_iter_bar_inputs_yields_a_bar_per_row_with_no_nan_truthiness() -> None:
    from forexpert.indicators.compute import compute_indicators

    frame = candles(400)
    indicators = compute_indicators(frame, IndicatorParams())
    bars = list(iter_bar_inputs(indicators))

    assert len(bars) == len(frame)
    assert isinstance(bars[0], BarInput)
    assert bars[0].time < bars[-1].time
    # early rows cannot all be booleans: an indicator may still be warming up
    assert all(isinstance(b.close, float) for b in bars)


def test_pandas_to_datetime_localizes_naive_timestamps() -> None:
    assert pandas_to_datetime(pd.Timestamp("2024-01-01")).tzinfo is not None
    assert pandas_to_datetime(pd.Timestamp("2024-01-01", tz="UTC")).tzinfo is not None


def test_bar_time_comparisons_are_utc_aware(tmp_path: Path) -> None:
    from forexpert.indicators.compute import compute_indicators

    indicators = compute_indicators(candles(400), IndicatorParams())
    bars = list(iter_bar_inputs(indicators))
    runner, _n, _j = make_runner(tmp_path)
    machine = runner.machines["EURCHF:4h"][0]
    machine.on_bar(bars[-1])
    assert machine.state.last_bar_time == bars[-1].time
    assert machine.state.last_bar_time.tzinfo is not None
    assert machine.state.bars_seen == 1


# --------------------------------------------------------------------- looping
def test_run_forever_stops_cleanly_on_keyboard_interrupt(tmp_path, monkeypatch) -> None:
    runner, _notifier, _journal = make_runner(tmp_path)
    calls = {"n": 0}

    def fake_pass() -> list:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise KeyboardInterrupt
        return []

    monkeypatch.setattr(runner, "run_once", fake_pass)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    saved = {"n": 0}
    monkeypatch.setattr(runner, "_save_state", lambda: saved.__setitem__("n", saved["n"] + 1))

    outcomes = runner.run_forever()

    assert outcomes == []           # the last successful pass is returned
    assert calls["n"] == 2          # interrupted on the second pass
    assert saved["n"] == 1          # state saved exactly once on shutdown


def test_run_forever_honours_max_passes(tmp_path, monkeypatch) -> None:
    runner, _notifier, _journal = make_runner(tmp_path)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    outcomes = runner.run_forever(max_passes=3)
    assert len(outcomes) == 1       # one symbol in the fixture config