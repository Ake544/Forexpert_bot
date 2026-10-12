"""Notifier tests — everything offline: fake sessions, no real HTTP.

Covers the behaviour that matters when a signal fires at 3am: retries on 5xx/429,
no retry on a hard 4xx, one dead provider never blocking the other, and the
message body actually containing the numbers the trader needs.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from forexpert.notify import (
    DiscordNotifier,
    FanoutNotifier,
    HttpNotifier,
    NullNotifier,
    TelegramNotifier,
    build_notifier,
    format_signal,
)
from forexpert.strategy.state import TradeSignal


def make_signal(**overrides: object) -> TradeSignal:
    values: dict = dict(
        signal_id="abc123",
        symbol="EURCHF",
        timeframe="1d",
        direction="buy",
        bar_time=__import__("datetime").datetime(2024, 3, 1, 8, tzinfo=__import__("datetime").timezone.utc),
        entry=0.9319,
        stop_loss=0.9299,
        take_profit=0.9359,
        atr=0.002,
        sl_distance=0.002,
        tp_distance=0.004,
        risk_reward=2.0,
        level=0.9315,
        close=0.9319,
        confirmations=("vwap_bull", "trend_bull"),
        vwap_bull=True,
        trend_bull=True,
        vslrt_bull=True,
        bars_from_candle1=1,
        bars_from_candle2=1,
    )
    values.update(overrides)
    return TradeSignal(**values)  # type: ignore[arg-type]


class FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "ok") -> None:
        self.status_code = status_code
        self.text = text


class FakeSession:
    """Returns scripted responses/exceptions, one per post()."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def post(self, url: str, json: dict | None = None, timeout: float | None = None, **kwargs: object):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        item = self.script.pop(0) if self.script else FakeResponse(200)
        if isinstance(item, Exception):
            raise item
        return item


class HttpProbe(HttpNotifier):
    """Concrete subclass so the shared retry logic can be tested directly."""

    name = "probe"

    def send(self, text: str) -> bool:
        return self._post("https://example.test/hook", {"content": text})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HttpNotifier, "_sleep", staticmethod(lambda _seconds: None))


# --------------------------------------------------------------------- message
def test_format_signal_carries_every_number_a_trader_needs() -> None:
    text = format_signal(make_signal())

    assert "BUY EURCHF 1d" in text
    assert "entry  : 0.9319" in text
    assert "SL     : 0.9299" in text
    assert "TP     : 0.9359" in text
    assert "RR     : 2.00" in text
    assert "vwap_bull, trend_bull" in text
    assert "id     : abc123" in text


def test_format_signal_omits_lots_when_there_are_none() -> None:
    text = format_signal(make_signal(position_size_lots=None))
    assert "lots" not in text
    assert "lots   : 0.05" in format_signal(make_signal(position_size_lots=0.05))


def test_format_signal_appends_the_balance_note() -> None:
    assert "balance 1000" in format_signal(make_signal(), balance_note="balance 1000")


# --------------------------------------------------------------------- retries
def test_retries_a_server_error_then_succeeds() -> None:
    session = FakeSession([FakeResponse(503), FakeResponse(429), FakeResponse(200)])
    notifier = HttpProbe({"max_retries": 5}, session=session)

    assert notifier.send("x") is True
    assert len(session.calls) == 3


def test_gives_up_after_max_retries() -> None:
    session = FakeSession([FakeResponse(500)] * 3)
    notifier = HttpProbe({"max_retries": 2}, session=session)

    assert notifier.send("x") is False
    assert len(session.calls) == 3


def test_transport_errors_are_retried() -> None:
    session = FakeSession([ConnectionError("boom"), FakeResponse(200)])
    notifier = HttpProbe({"max_retries": 3}, session=session)

    assert notifier.send("x") is True
    assert len(session.calls) == 2


def test_a_client_error_is_not_retried() -> None:
    session = FakeSession([FakeResponse(404, "nope")])
    notifier = HttpProbe({"max_retries": 5}, session=session)

    assert notifier.send("x") is False
    assert len(session.calls) == 1


def test_disabled_notifier_never_touches_the_network() -> None:
    session = FakeSession([])
    notifier = HttpProbe({}, session=session, enabled=False)

    assert notifier.send("x") is False
    assert session.calls == []


# ------------------------------------------------------------------ providers
def test_discord_posts_content_and_truncates_at_the_limit() -> None:
    session = FakeSession([FakeResponse(204)])
    notifier = DiscordNotifier({"webhook_url": "https://discord.test/webhook"}, session=session)

    assert notifier.send("x" * 5000) is True
    call = session.calls[0]
    assert call["url"] == "https://discord.test/webhook"
    assert len(call["json"]["content"]) == 2000


def test_discord_without_a_url_is_disabled_and_stays_quiet() -> None:
    session = FakeSession([])
    notifier = DiscordNotifier({}, session=session)

    assert notifier.send("hello") is False
    assert session.calls == []


def test_telegram_builds_its_api_url_and_payload() -> None:
    session = FakeSession([FakeResponse(200)])
    notifier = TelegramNotifier(
        {"bot_token": "TOKEN", "chat_id": "42"}, session=session
    )

    assert notifier.send("hi") is True
    call = session.calls[0]
    assert call["url"] == "https://api.telegram.org/botTOKEN/sendMessage"
    assert call["json"]["chat_id"] == "42"
    assert call["json"]["text"] == "hi"


def test_telegram_without_credentials_is_disabled() -> None:
    session = FakeSession([])
    notifier = TelegramNotifier({}, session=session)

    assert notifier.send("hi") is False
    assert session.calls == []


def test_fanout_delivers_to_healthy_children_even_when_one_fails() -> None:
    good = FakeSession([FakeResponse(200)])
    bad = FakeSession([FakeResponse(401)])
    fanout = FanoutNotifier(
        DiscordNotifier({"webhook_url": "https://discord.test/a"}, session=good),
        TelegramNotifier({"bot_token": "T", "chat_id": "C"}, session=bad),
    )

    assert fanout.send("signal") is True
    assert fanout.delivered == [("discord", True), ("telegram", False)]


def test_fanout_returns_false_when_every_child_fails() -> None:
    failing = FakeSession([FakeResponse(500)] * 10)
    fanout = FanoutNotifier(DiscordNotifier({"webhook_url": "https://d.test"}, session=failing))
    assert fanout.send("signal") is False


# -------------------------------------------------------------------- builder
def test_no_configuration_yields_a_null_notifier() -> None:
    assert isinstance(build_notifier({}), NullNotifier)
    assert isinstance(build_notifier({"discord": {"enabled": False}}), NullNotifier)


def test_only_discord_configured_returns_that_notifier() -> None:
    built = build_notifier({"discord": {"enabled": True, "webhook_url": "https://d.test"}})
    assert isinstance(built, DiscordNotifier)


def test_both_configured_return_a_fanout() -> None:
    built = build_notifier(
        {
            "telegram": {"enabled": True, "bot_token": "T", "chat_id": "C"},
            "discord": {"enabled": True, "webhook_url": "https://d.test"},
        }
    )
    assert isinstance(built, FanoutNotifier)
    assert [child.name for child in built._notifiers] == ["telegram", "discord"]
