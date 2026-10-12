"""The signal state machine for one (symbol, timeframe, direction).

Per closed candle, in order:

    IDLE         a candle that closes beyond the level starts the setup
    CANDLE1_OK   waiting for the very next candle to open AND close beyond it
    CANDLE2_OK   sequence confirmed; the next bar is the earliest entry candle
    WAITING_TOUCH price must come back to the level within the expiry window
    COOLDOWN     bars are swallowed after a signal so one setup cannot re-fire

Fail-closed by design: a missing level, a missing ATR or an unknown indicator
state (`None`) blocks a signal instead of defaulting to a permissive value.
"""
from __future__ import annotations

import logging
from datetime import datetime

from ..risk.levels import TradeLevels, build_levels, position_size
from ..settings import RiskParams, StrategyParams
from .state import BarInput, Direction, Phase, SetupState, TradeSignal, make_signal_id

LOGGER = logging.getLogger(__name__)

OPPOSITE: dict[Direction, Direction] = {"buy": "sell", "sell": "buy"}


class BarOrderError(ValueError):
    """Raised when bars arrive out of order or repeated."""


class StrategyMachine:
    """Stateful, pure-with-respect-to-the-world detector for one direction."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        direction: Direction,
        params: StrategyParams,
        risk: RiskParams | None = None,
        state: SetupState | None = None,
        atr_params: object | None = None,
    ) -> None:
        if direction not in ("buy", "sell"):
            raise ValueError("direction must be 'buy' or 'sell'")
        self.symbol = symbol
        self.timeframe = timeframe
        self.direction = direction
        self.params = params
        self.risk = risk or RiskParams()
        self.state = state or SetupState(
            symbol=symbol, timeframe=timeframe, direction=direction
        )
        self._sl_multiplier, self._tp_multiplier = self._resolve_atr_multipliers(atr_params)

    @staticmethod
    def _resolve_atr_multipliers(atr_params: object | None) -> tuple[float, float]:
        sl = getattr(atr_params, "sl_multiplier", None)
        tp = getattr(atr_params, "tp_multiplier", None)
        if sl is None or tp is None:
            return 1.5, 2.25
        return float(sl), float(tp)

    # ------------------------------------------------------------------ helpers
    @property
    def key(self) -> str:
        return f"{self.symbol}:{self.timeframe}:{self.direction}"

    def _beyond(self, price: float, level: float) -> bool:
        """True when `price` is strictly on the setup side of the level."""
        return price > level if self.direction == "buy" else price < level

    def _wrong_side(self, bar: BarInput) -> bool:
        level = bar.level
        return level is not None and not self._beyond(bar.close, level)

    def _touched(self, bar: BarInput) -> bool:
        level = bar.level
        if level is None:
            return False
        if self.params.touch_definition == "close":
            return self._beyond_or_equal_close(bar, level)
        return bar.low <= level <= bar.high

    def _beyond_or_equal_close(self, bar: BarInput, level: float) -> bool:
        return bar.close <= level if self.direction == "buy" else bar.close >= level

    def _candle1_ok(self, bar: BarInput) -> bool:
        return bar.level is not None and self._beyond(bar.close, bar.level)

    def _candle2_ok(self, bar: BarInput) -> bool:
        if bar.level is None:
            return False
        return self._beyond(bar.open, bar.level) and self._beyond(bar.close, bar.level)

    def _trend_filter_ok(self, bar: BarInput) -> bool:
        return bar.vwap_bull is True and bar.trend_bull is True

    def _confirmation(self, bar: BarInput) -> tuple[str, ...]:
        fired: list[str] = []
        if bar.vslrt_bull is True:
            fired.append("VSLRT")
        if self.direction == "buy" and bar.kro_bottom_touch is True:
            fired.append("KRO")
        if self.direction == "sell" and bar.kro_top_touch is True:
            fired.append("KRO")
        return tuple(fired)

    def _reset(self, reason: str) -> None:
        self.state.phase = Phase.IDLE
        self.state.candle1_time = None
        self.state.candle2_time = None
        self.state.bars_since_candle2 = 0
        self.state.sticky_trend_ok = False
        self.state.sticky_confirmation_ok = False
        self.state.reject_reason = reason

    # --------------------------------------------------------------------- api
    def on_bar(self, bar: BarInput) -> TradeSignal | None:
        """Consume one CLOSED candle. Returns a signal at most once per bar."""
        if self.state.last_bar_time is not None and bar.time <= self.state.last_bar_time:
            raise BarOrderError(
                f"{self.key}: bars must strictly increase "
                f"(last={self.state.last_bar_time.isoformat()}, got={bar.time.isoformat()})"
            )
        self.state.last_bar_time = bar.time
        self.state.bars_seen += 1

        if self.state.phase is Phase.COOLDOWN:
            self.state.bars_until_ready = max(0, self.state.bars_until_ready - 1)
            if self.state.bars_until_ready == 0:
                self._reset("cooldown_elapsed")
            return None

        if bar.level is None:
            self.state.reject_reason = "level_unavailable"
            return None

        if self.state.phase is not Phase.IDLE or self._candle1_ok(bar):
            self._accumulate_sticky(bar)

        if self.state.phase is Phase.IDLE:
            if self._candle1_ok(bar):
                self.state.phase = Phase.CANDLE1_OK
                self.state.candle1_time = bar.time
                self.state.candle1_seen = self.state.bars_seen
                self.state.reject_reason = None
            return None

        if self.state.phase is Phase.CANDLE1_OK:
            if self._candle2_ok(bar):
                self.state.phase = Phase.CANDLE2_OK
                self.state.candle2_time = bar.time
                self.state.bars_since_candle2 = 0
                self.state.reject_reason = None
                return None
            self._reset("candle2_failed")
            if self.params.reseed_candle1_on_failure and self._candle1_ok(bar):
                self.state.phase = Phase.CANDLE1_OK
                self.state.candle1_time = bar.time
                self.state.candle1_seen = self.state.bars_seen
                self.state.reject_reason = "candle2_failed_reseeded"
            return None

        return self._evaluate_entry_bar(bar)

    def _accumulate_sticky(self, bar: BarInput) -> None:
        """Latch filter/confirmation passes across the whole setup.

        Only used when ``filters_on_entry_bar`` is off: the bar that confirms
        candle 1 counts too, because the setup starts on that candle.
        """
        if self.params.filters_on_entry_bar:
            return
        if self._trend_filter_ok(bar):
            self.state.sticky_trend_ok = True
        if self._confirmation(bar):
            self.state.sticky_confirmation_ok = True

    def _evaluate_entry_bar(self, bar: BarInput) -> TradeSignal | None:
        self.state.bars_since_candle2 += 1
        self.state.phase = Phase.WAITING_TOUCH

        if self.params.expire_on_wrong_side_close and self._wrong_side(bar):
            self._reset("closed_wrong_side")
            return None
        if self.state.bars_since_candle2 > self.params.expiry_bars:
            self._reset("expired")
            return None
        if not self._touched(bar):
            self.state.reject_reason = "awaiting_touch"
            return None

        if self.params.filters_on_entry_bar:
            filters_ok = self._trend_filter_ok(bar)
            confirmations = self._confirmation(bar)
        else:
            filters_ok = self.state.sticky_trend_ok
            confirmations = self._confirmation(bar) or (
                ("VSLRT",) if self.state.sticky_confirmation_ok else ()
            )

        if not filters_ok:
            self.state.reject_reason = "trend_filter_not_met"
            return None
        if not confirmations:
            self.state.reject_reason = "no_confirmation"
            return None

        signal = self._build_signal(bar, confirmations)
        if signal is None:
            return None

        self.state.last_signal_id = signal.signal_id
        self.state.last_signal_time = bar.time
        self.state.processed_signals.append(signal.signal_id)
        self.state.processed_signals = self.state.processed_signals[-50:]
        self.state.phase = Phase.COOLDOWN
        self.state.bars_until_ready = max(0, int(self.params.cooldown_bars))
        self.state.reject_reason = None
        LOGGER.info(
            "%s %s signal at %s entry=%s sl=%s tp=%s confirmations=%s",
            signal.symbol, signal.direction, signal.bar_time.isoformat(),
            signal.entry, signal.stop_loss, signal.take_profit, signal.confirmations,
        )
        return signal

    def _build_signal(
        self, bar: BarInput, confirmations: tuple[str, ...]
    ) -> TradeSignal | None:
        atr = bar.atr
        if atr is None or atr <= 0:
            self.state.reject_reason = "atr_unavailable"
            return None
        if bar.level is None:
            self.state.reject_reason = "level_unavailable"
            return None

        entry = bar.close if self.params.entry_price == "close" else bar.level
        levels: TradeLevels = build_levels(
            entry=entry,
            atr=atr,
            direction=self.direction,
            sl_multiplier=self._sl_multiplier,
            tp_multiplier=self._tp_multiplier,
        )
        lots = position_size(levels, self.risk) if self.risk else None
        return TradeSignal(
            signal_id=make_signal_id(self.symbol, self.timeframe, self.direction, bar.time),
            symbol=self.symbol,
            timeframe=self.timeframe,
            direction=self.direction,
            bar_time=bar.time,
            entry=levels.entry,
            stop_loss=levels.stop_loss,
            take_profit=levels.take_profit,
            atr=levels.atr,
            sl_distance=levels.sl_distance,
            tp_distance=levels.tp_distance,
            risk_reward=levels.risk_reward,
            level=bar.level,
            close=bar.close,
            confirmations=confirmations,
            vwap_bull=bool(bar.vwap_bull),
            trend_bull=bool(bar.trend_bull),
            vslrt_bull=bool(bar.vslrt_bull),
            bars_from_candle1=self.state.bars_seen - self.state.candle1_seen,
            bars_from_candle2=self.state.bars_since_candle2,
            position_size_lots=lots,
        )