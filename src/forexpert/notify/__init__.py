"""Notifications: Telegram and Discord, both opt-in and both retrying.

`build_notifier(config)` returns a NullNotifier when nothing is configured, so
the runner can always call `.send()` without branching on configuration state.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Protocol

from ..strategy.state import TradeSignal

LOGGER = logging.getLogger(__name__)


class Notifier(Protocol):
    name: str

    def send(self, text: str) -> bool:
        """Deliver a message. Returns True when the provider accepted it."""
        ...


class NullNotifier:
    """Used when no provider is configured: records instead of sending."""

    name = "null"

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, text: str) -> bool:
        self.sent.append(text)
        LOGGER.debug("no notifier configured, message dropped: %s", text[:120])
        return False


def format_signal(signal: TradeSignal, balance_note: str | None = None) -> str:
    """Plain-text alert body shared by every provider."""
    lines = [
        f"{signal.direction.upper()} {signal.symbol} {signal.timeframe}",
        f"bar    : {signal.bar_time.isoformat()}",
        f"entry  : {signal.entry}",
        f"SL     : {signal.stop_loss}  (-{signal.sl_distance})",
        f"TP     : {signal.take_profit}  (+{signal.tp_distance})",
        f"RR     : {signal.risk_reward:.2f}",
        f"level  : {signal.level}",
        f"confirm: {', '.join(signal.confirmations) or 'none'}",
    ]
    if signal.position_size_lots:
        lines.append(f"lots   : {signal.position_size_lots}")
    if balance_note:
        lines.append(balance_note)
    lines.append(f"id     : {signal.signal_id}")
    return "\n".join(lines)


class HttpNotifier:
    """Shared retry/backoff behaviour for webhook-style providers."""

    name = "http"

    def __init__(
        self,
        settings: Mapping[str, Any],
        session: Any | None = None,
        enabled: bool = True,
    ) -> None:
        self._enabled = bool(enabled)
        self._max_retries = int(settings.get("max_retries", 5))
        self._backoff = float(settings.get("retry_backoff_seconds", 3.0))
        self._timeout = float(settings.get("request_timeout_seconds", 20))
        self._session = session

    @staticmethod
    def _sleep(seconds: float) -> None:
        import time

        time.sleep(seconds)

    def _post(self, url: str, payload: dict[str, Any], **kwargs: Any) -> bool:
        import requests

        if not self._enabled:
            LOGGER.debug("%s disabled, message dropped", self.name)
            return False
        session = self._session or requests.Session()
        for attempt in range(self._max_retries + 1):
            try:
                response = session.post(url, json=payload, timeout=self._timeout, **kwargs)
            except Exception as exc:  # noqa: BLE001 - any transport error is retryable
                if attempt >= self._max_retries:
                    LOGGER.error("%s transport failure after %s tries: %s",
                                 self.name, attempt + 1, exc)
                    return False
                LOGGER.warning("%s retry %s: %s", self.name, attempt + 1, exc)
                self._sleep(self._backoff * (2**attempt))
                continue
            status = getattr(response, "status_code", 500)
            if 200 <= status < 300:
                return True
            if status >= 500 or status == 429:
                if attempt >= self._max_retries:
                    LOGGER.error("%s gave up after %s attempts: HTTP %s",
                                 self.name, attempt + 1, status)
                    return False
                LOGGER.warning("%s retry %s: HTTP %s", self.name, attempt + 1, status)
                self._sleep(self._backoff * (2**attempt))
                continue
            LOGGER.error("%s rejected (HTTP %s): %s", self.name, status, getattr(response, "text", "")[:200])
            return False
        return False


class TelegramNotifier(HttpNotifier):
    name = "telegram"

    def __init__(self, settings: Mapping[str, Any], session: Any | None = None) -> None:
        token = str(settings.get("bot_token") or "").strip()
        chat_id = str(settings.get("chat_id") or "").strip()
        super().__init__(settings, session=session, enabled=bool(token and chat_id))
        self._token = token
        self._chat_id = chat_id
        self._parse_mode = settings.get("parse_mode") or "HTML"
        if not token or not chat_id:
            LOGGER.warning(
                "telegram notifier missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID; "
                "alerts will be dropped"
            )

    def send(self, text: str) -> bool:
        return self._post(
            f"https://api.telegram.org/bot{self._token}/sendMessage",
            {
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": self._parse_mode,
                "disable_web_page_preview": True,
            },
        )


class DiscordNotifier(HttpNotifier):
    name = "discord"

    def __init__(self, settings: Mapping[str, Any], session: Any | None = None) -> None:
        self._url = str(settings.get("webhook_url") or "").strip()
        super().__init__(settings, session=session, enabled=bool(self._url))
        if not self._url:
            LOGGER.warning("discord notifier missing DISCORD_WEBHOOK_URL; alerts will be dropped")

    def send(self, text: str) -> bool:
        return self._post(self._url, {"content": text[:2000]})


class FanoutNotifier:
    """Sends to every child; a failing child never blocks the others."""

    name = "fanout"

    def __init__(self, *notifiers: Notifier) -> None:
        self._notifiers = [n for n in notifiers if n is not None]
        self.delivered: list[tuple[str, bool]] = []

    def send(self, text: str) -> bool:
        results = [n.send(text) for n in self._notifiers]
        self.delivered.extend(zip((n.name for n in self._notifiers), results))
        return any(results)


def build_notifier(notifier: Mapping[str, Any], session: Any | None = None) -> Notifier:
    """Create the configured notifier. Falls back to a NullNotifier."""
    telegram = notifier.get("telegram") or {}
    discord = notifier.get("discord") or {}
    configured = []
    if telegram.get("enabled"):
        configured.append(TelegramNotifier(telegram, session=session))
    if discord.get("enabled"):
        configured.append(DiscordNotifier(discord, session=session))
    if not configured:
        return NullNotifier()
    if len(configured) == 1:
        return configured[0]
    return FanoutNotifier(*configured)
