"""Value objects for the signal state machine: phases, per-bar input, signals."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

Direction = Literal["buy", "sell"]


class Phase(str, Enum):
    """Setup lifecycle for one (symbol, timeframe, direction) triple.

    IDLE -> CANDLE1_OK -> CANDLE2_OK -> WAITING_TOUCH -> (signal) -> COOLDOWN -> IDLE
    """

    IDLE = "idle"
    CANDLE1_OK = "candle1_ok"
    CANDLE2_OK = "candle2_ok"
    WAITING_TOUCH = "waiting_touch"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class BarInput:
    """One CLOSED candle plus every indicator state the rules need.

    Tri-state booleans matter: None means "indicator not ready" and must never be
    treated as False, otherwise a warming-up VWAP would look like a valid short.
    """

    time: datetime
    open: float
    high: float
    low: float
    close: float
    level: float | None
    vwap_bull: bool | None
    trend_bull: bool | None
    vslrt_bull: bool | None
    atr: float | None
    kro_bottom_touch: bool | None = None
    kro_top_touch: bool | None = None
    vslrt_long_line: float | None = None

    @classmethod
    def from_snapshot(cls, snapshot: Any) -> "BarInput":
        return cls(
            time=snapshot.bar_time.to_pydatetime(),
            open=snapshot.open,
            high=snapshot.high,
            low=snapshot.low,
            close=snapshot.close,
            level=snapshot.level,
            vwap_bull=snapshot.vwap_bull,
            trend_bull=snapshot.trend_bull,
            vslrt_bull=snapshot.vslrt_bull,
            atr=snapshot.atr,
        )


@dataclass(frozen=True)
class TradeSignal:
    """A confirmed entry event. Purely descriptive - nothing is executed."""

    signal_id: str
    symbol: str
    timeframe: str
    direction: Direction
    bar_time: datetime
    entry: float
    stop_loss: float
    take_profit: float
    atr: float
    sl_distance: float
    tp_distance: float
    risk_reward: float
    level: float
    close: float
    confirmations: tuple[str, ...]
    vwap_bull: bool
    trend_bull: bool
    vslrt_bull: bool
    bars_from_candle1: int
    bars_from_candle2: int
    position_size_lots: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bar_time"] = self.bar_time.isoformat()
        payload["confirmations"] = list(self.confirmations)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TradeSignal":
        data = dict(payload)
        data["bar_time"] = datetime.fromisoformat(data["bar_time"])
        data["confirmations"] = tuple(data.get("confirmations") or ())
        return cls(**data)


def make_signal_id(symbol: str, timeframe: str, direction: str, bar_time: datetime) -> str:
    """Deterministic id so the same bar can never be notified twice.

    Deterministic (not random) because dedup has to survive a process restart:
    the id is recomputed from the bar itself, not from a counter in memory.
    """
    raw = f"{symbol}|{timeframe}|{direction}|{bar_time.isoformat()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class SetupState:
    """Persisted per (symbol, timeframe, direction). Must survive restarts."""

    symbol: str
    timeframe: str
    direction: Direction
    phase: Phase = Phase.IDLE
    candle1_time: datetime | None = None
    candle2_time: datetime | None = None
    bars_seen: int = 0
    candle1_seen: int = 0
    bars_since_candle2: int = 0
    bars_until_ready: int = 0
    sticky_trend_ok: bool = False
    sticky_confirmation_ok: bool = False
    last_bar_time: datetime | None = None
    last_signal_id: str | None = None
    last_signal_time: datetime | None = None
    reject_reason: str | None = None
    processed_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phase"] = self.phase.value
        for key in ("candle1_time", "candle2_time", "last_bar_time", "last_signal_time"):
            value = payload.get(key)
            payload[key] = value.isoformat() if isinstance(value, datetime) else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SetupState":
        data = dict(payload)
        data["phase"] = Phase(data.get("phase", "idle"))
        for key in ("candle1_time", "candle2_time", "last_bar_time", "last_signal_time"):
            value = data.get(key)
            data[key] = datetime.fromisoformat(value) if isinstance(value, str) else None
        data["processed_signals"] = list(data.get("processed_signals") or [])
        return cls(**data)