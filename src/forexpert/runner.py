"""The live loop: fetch → indicators → both state machines → journal → notify.

Responsibilities kept deliberately narrow so the same path is reusable by the
backtester:

  * fetch closed candles from the configured provider and upsert them
  * compute the indicator stack once, then hand **one `BarInput` per bar** to
    each machine (buy and sell run independently over the same bars)
  * dedupe signals across restarts via the store, journal every decision
  * never let one symbol's failure stop the others

First-run behaviour: with no `processed_bars` recorded yet we baseline on the
**newest bar only** instead of replaying the whole history, so booting with 1000
cached candles cannot manufacture signals for setups that formed weeks ago.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterator, Mapping

import pandas as pd

from .data import DataError, build_sources
from .indicators.compute import IndicatorFrame, compute_indicators
from .journal import Journal
from .notify import Notifier, build_notifier, format_signal
from .settings import AppConfig, SymbolConfig
from .store.sqlite_store import SqliteStore, StateStore
from .strategy.machine import StrategyMachine
from .strategy.state import BarInput, SetupState, TradeSignal

LOGGER = logging.getLogger(__name__)


def default_clock() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _opt_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


def _tri(value: Any) -> bool | None:
    """NaN/None -> None (not ready); otherwise a real bool.

    `bool(float('nan'))` is True in Python, so this must be explicit or a
    warming-up indicator would read as a confirmed bull flag.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


def iter_bar_inputs(indicators: IndicatorFrame) -> Iterator[BarInput]:
    """Yield one BarInput per closed bar, oldest first."""
    levels = indicators.level_frame()
    for moment, row in levels.iterrows():
        yield BarInput(
            time=pandas_to_datetime(moment),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            level=_opt_float(row["level"]),
            vwap_bull=_tri(row["vwap_bull"]),
            trend_bull=_tri(row["trend_bull"]),
            vslrt_bull=_tri(row["vslrt_bull"]),
            atr=_opt_float(row["atr"]),
            vslrt_long_line=_opt_float(row["vslrt_long_line"]),
        )


def pandas_to_datetime(moment: Any) -> datetime:
    value = pd.Timestamp(moment)
    if value.tz is None:
        value = value.tz_localize("UTC")
    return value.to_pydatetime()


@dataclass
class SymbolOutcome:
    """What happened to one symbol during a pass."""

    symbol: str
    timeframe: str
    signals: list[TradeSignal] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    bars_processed: int = 0
    error: str | None = None
    stale: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


class Runner:
    """Drives every configured (symbol, timeframe) through one pass."""

    def __init__(
        self,
        config: AppConfig,
        store: StateStore | None = None,
        sources: Mapping[str, Any] | None = None,
        journal: Journal | None = None,
        notifier: Notifier | None = None,
        session: Any | None = None,
        clock: Callable[[], pd.Timestamp] = default_clock,
    ) -> None:
        self.config = config
        self.clock = clock
        runtime = config.runtime or {}
        self.max_bars = int(runtime.get("max_bars_in_memory", 1000))
        self.max_catchup = int(runtime.get("max_catchup_bars", 20))
        self.wake_delay = float(runtime.get("wake_delay_seconds", 5))
        alerts = config.alerts or {}
        self._stale_bars = int(alerts.get("stale_data_bars", 3))
        self._notify_startup = bool(alerts.get("notify_on_startup", True))
        self._error_alerts_enabled = bool(alerts.get("error_alerts", True))
        self._error_cooldown = pd.Timedelta(
            float(alerts.get("error_cooldown_minutes", 30)), unit="m"
        )
        self._startup_sent = False
        self._last_error_alert: dict[str, pd.Timestamp] = {}

        self.store = store if store is not None else SqliteStore(
            runtime.get("db_path", "data/forexpert.sqlite")
        )
        self.sources = dict(sources) if sources is not None else build_sources(config.data)
        journal_cfg = config.journal or {}
        self.journal = journal if journal is not None else Journal(
            journal_cfg.get("path", "data/journal.jsonl"),
            enabled=bool(journal_cfg.get("enabled", True)),
        )
        self.notifier = notifier if notifier is not None else build_notifier(
            {"telegram": config.notifier.telegram, "discord": config.notifier.discord},
            session=session,
        )
        self.machines: dict[str, list[StrategyMachine]] = {
            cfg.key: self._build_machines(cfg) for cfg in config.enabled_symbols()
        }

    # ------------------------------------------------------------------ setup
    def _build_machines(self, symbol: SymbolConfig) -> list[StrategyMachine]:
        machines: list[StrategyMachine] = []
        for direction in ("buy", "sell"):
            key = f"{symbol.key}:{direction}"
            restored = self.store.load_state(key)
            state = None
            if restored is not None:
                try:
                    state = SetupState.from_dict(restored)
                except Exception as exc:  # noqa: BLE001 - corrupt state must not stop the run
                    LOGGER.error("%s: persisted state unusable (%s), starting fresh", key, exc)
            machines.append(
                StrategyMachine(
                    symbol=symbol.symbol,
                    timeframe=symbol.timeframe,
                    direction=direction,  # type: ignore[arg-type]
                    params=self.config.strategy,
                    risk=self.config.risk,
                    state=state,
                    atr_params=symbol.params.atr,
                )
            )
        return machines

    def _save_state(self) -> None:
        for machines in self.machines.values():
            for machine in machines:
                self.store.save_state(machine.key, machine.state.to_dict())

    # ------------------------------------------------------------------ pass
    def run_once(self) -> list[SymbolOutcome]:
        self._maybe_notify_startup()
        outcomes = []
        for symbol in self.config.enabled_symbols():
            try:
                outcomes.append(self.process_symbol(symbol))
            except Exception as exc:  # noqa: BLE001 - isolate per-symbol failures
                LOGGER.exception("pass failed for %s", symbol.key)
                outcomes.append(SymbolOutcome(symbol.symbol, symbol.timeframe, error=str(exc)))
            finally:
                self._save_state()   # persist after each symbol, not once at the end
        for outcome in outcomes:
            if outcome.error:
                self._alert_error(outcome)
        return outcomes

    def process_symbol(self, symbol: SymbolConfig) -> SymbolOutcome:
        outcome = SymbolOutcome(symbol.symbol, symbol.timeframe)
        source = self.sources.get(symbol.provider)
        if source is None:
            raise DataError(f"no candle source for provider {symbol.provider!r}")

        frame = source.fetch(symbol.symbol, symbol.timeframe, self.max_bars)
        if frame.empty:
            outcome.error = f"{symbol.symbol} {symbol.timeframe}: provider returned no candles"
            return outcome
        self.store.upsert_candles(symbol.symbol, symbol.timeframe, frame)

        outcome.stale = self._is_stale(frame, symbol.timeframe)
        if outcome.stale:
            message = (
                f"STALE {symbol.symbol} {symbol.timeframe}: "
                f"no closed candle since {frame.index[-1]}"
            )
            LOGGER.warning(message)
            self.journal.record("stale_data", {
                "symbol": symbol.symbol, "timeframe": symbol.timeframe,
                "last_bar": str(frame.index[-1]),
            })
            self.notifier.send(message)

        indicators = compute_indicators(
            frame, symbol.params, touch_line=self.config.strategy.touch_line
        )
        bars = list(iter_bar_inputs(indicators))
        fed = self._new_bars(symbol, bars)
        for bar in fed:
            self._feed(symbol, bar, outcome)
        for bar in fed:
            self.store.mark_bar_processed(symbol.symbol, symbol.timeframe, bar.time)
        outcome.bars_processed = len(fed)
        return outcome

    def _new_bars(self, symbol: SymbolConfig, bars: list[BarInput]) -> list[BarInput]:
        """Closed bars the machine has not seen yet, bounded by max_catchup_bars."""
        if not bars:
            return []
        last_processed = self.store.last_processed_time(symbol.symbol, symbol.timeframe)
        if last_processed is None:
            LOGGER.info("%s: first run, baselining on the newest bar only", symbol.key)
            return bars[-1:]
        fresh = [bar for bar in bars if pandas_to_datetime(bar.time) > last_processed]
        if len(fresh) > self.max_catchup:
            LOGGER.warning(
                "%s: %s bars pending, replaying only the newest %s",
                symbol.key, len(fresh), self.max_catchup,
            )
            fresh = fresh[-self.max_catchup:]
        return fresh

    def _feed(self, symbol: SymbolConfig, bar: BarInput, outcome: SymbolOutcome) -> None:
        for machine in self.machines[symbol.key]:
            try:
                signal = machine.on_bar(bar)
            except Exception as exc:  # noqa: BLE001 - one bad bar must not kill the pass
                LOGGER.error("%s: machine rejected bar %s: %s", machine.key, bar.time, exc)
                outcome.error = f"{machine.key}: {exc}"
                continue
            if signal is not None:
                self._handle_signal(signal, outcome)
            elif machine.state.reject_reason:
                outcome.rejections.append(f"{machine.key}:{machine.state.reject_reason}")
                self.journal.rejection(
                    symbol.symbol, symbol.timeframe, machine.state.reject_reason,
                    bar_time=bar.time.isoformat(), direction=machine.direction,
                )

    def _handle_signal(self, signal: TradeSignal, outcome: SymbolOutcome) -> None:
        if self.store.has_signal(signal.signal_id):
            LOGGER.debug("duplicate signal suppressed: %s", signal.signal_id)
            return
        self.store.record_signal(
            signal.signal_id, signal.symbol, signal.timeframe,
            signal.direction, signal.bar_time, signal.to_dict(),
        )
        self.journal.signal(signal.to_dict())
        outcome.signals.append(signal)
        delivered = self.notifier.send(format_signal(signal))
        LOGGER.info(
            "%s signal %s entry=%s sl=%s tp=%s notified=%s",
            signal.symbol, signal.signal_id, signal.entry,
            signal.stop_loss, signal.take_profit, delivered,
        )

    # ----------------------------------------------------------------- alerts
    def _maybe_notify_startup(self) -> None:
        """One 'I am alive' message per process, so a silent channel is noticed."""
        if self._startup_sent or not self._notify_startup:
            return
        self._startup_sent = True
        symbols = self.config.enabled_symbols()
        providers = sorted({symbol.provider for symbol in symbols})
        payload = {
            "symbols": len(symbols),
            "providers": providers,
            "notifier": self.notifier.name,
        }
        LOGGER.info("startup alert via %s", self.notifier.name)
        self.journal.record("startup", payload)
        self.notifier.send(
            f"Forexpert started: {len(symbols)} symbol configs | "
            f"providers {', '.join(providers)} | notifier {self.notifier.name}"
        )

    def _alert_error(self, outcome: SymbolOutcome) -> None:
        """Per-symbol failure alert, re-sent only after `error_cooldown_minutes`."""
        if not self._error_alerts_enabled:
            return
        key = f"{outcome.symbol} {outcome.timeframe}"
        now = self.clock()
        last = self._last_error_alert.get(key)
        if last is not None and (now - last) < self._error_cooldown:
            return
        self._last_error_alert[key] = now
        LOGGER.warning("ERROR %s: %s", key, outcome.error)
        self.journal.record("error_alert", {
            "symbol": outcome.symbol, "timeframe": outcome.timeframe,
            "error": outcome.error,
        })
        self.notifier.send(f"ERROR {key}: {outcome.error}")

    def _is_stale(self, frame: pd.DataFrame, timeframe: str) -> bool:
        """True when no new closed candle has arrived for `stale_data_bars` bars."""
        if self._stale_bars <= 0 or frame.empty:
            return False
        from .data.base import timeframe_duration

        try:
            duration = timeframe_duration(timeframe)
        except DataError:
            return False
        last_close = frame.index[-1] + duration
        age = self.clock() - last_close
        return age > duration * self._stale_bars

    # ------------------------------------------------------------------- loop
    def seconds_until_next_close(self) -> float:
        """Sleep until the soonest configured timeframe closes, plus wake_delay."""
        now = self.clock()
        best: float | None = None
        for symbol in self.config.enabled_symbols():
            try:
                duration = _timeframe_minutes(symbol.timeframe) * 60
            except DataError:
                continue
            elapsed = now.timestamp() % duration
            remaining = (duration - elapsed) + self.wake_delay
            best = remaining if best is None else min(best, remaining)
        return float(best if best is not None else 60.0)

    def run_forever(self, max_passes: int | None = None) -> None:
        passes = 0
        while max_passes is None or passes < max_passes:
            outcomes = self.run_once()
            passes += 1
            signals = sum(len(o.signals) for o in outcomes)
            errors = [o for o in outcomes if not o.ok]
            LOGGER.info("pass %s: %s signals, %s errors", passes, signals, len(errors))
            if max_passes is not None and passes >= max_passes:
                break
            delay = max(1.0, self.seconds_until_next_close())
            LOGGER.debug("sleeping %.1fs", delay)
            time.sleep(delay)


def _timeframe_minutes(timeframe: str) -> int:
    from .data.base import TIMEFRAME_MINUTES

    if timeframe not in TIMEFRAME_MINUTES:
        raise DataError(f"Unsupported timeframe: {timeframe!r}")
    return TIMEFRAME_MINUTES[timeframe]
