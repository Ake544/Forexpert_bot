"""Backtest: the same indicators and state machines as live, then simulate fills.

Two deliberately separate phases:

1. ``collect_signals`` runs the *identical* pipeline the live runner uses —
   ``compute_indicators`` → ``iter_bar_inputs`` → buy/sell `StrategyMachine` —
   over a historical candle frame. No indicator sees a bar it shouldn't, so the
   signals are exactly what live would have produced on that history.
2. ``simulate_trade`` walks forward from each signal and applies ATR stops.

Modelling rules (kept conservative on purpose):

* **Entry** is `signal.entry` — the close of the touch candle (or the level,
  per `strategy.entry_price`). Stops are live from the **next** bar onward: the
  bar that produced the entry cannot also stop the trade out.
* **Same-bar stop+target → stop first.** Real fills in that situation are
  ambiguous, so the worst case is taken. It cannot inflate the win rate.
* Trades still running at the end of the data are reported as `open` (marked to
  market at the last close) or `timeout` (if `max_hold_bars` elapsed), never
  silently dropped — an unprofitable strategy must not look good because its
  losers never closed.
* No spread, slippage, swap or commission is modelled: results are price-level
  results, not broker statements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..indicators.compute import compute_indicators
from ..runner import iter_bar_inputs
from ..settings import AppConfig, SymbolConfig
from ..strategy.machine import StrategyMachine
from ..strategy.state import TradeSignal

DEFAULT_MAX_HOLD_BARS = 100

OUTCOMES = ("tp", "sl", "timeout", "open")


@dataclass(frozen=True)
class SimulatedTrade:
    """One signal resolved against the bars that followed it."""

    signal: TradeSignal
    entry_bar: pd.Timestamp
    exit_bar: pd.Timestamp | None
    exit_price: float | None
    outcome: str  # tp | sl | timeout | open
    bars_held: int
    r_multiple: float

    @property
    def signal_id(self) -> str:
        return self.signal.signal_id

    @property
    def direction(self) -> str:
        return self.signal.direction

    @property
    def won(self) -> bool:
        return self.outcome == "tp"

    @property
    def closed(self) -> bool:
        return self.outcome in ("tp", "sl")


@dataclass
class BacktestResult:
    """All signals for one (symbol, timeframe) plus how they resolved."""

    symbol: str
    timeframe: str
    bars: int = 0
    signals: list[TradeSignal] = field(default_factory=list)
    trades: list[SimulatedTrade] = field(default_factory=list)

    def stats(self) -> dict[str, Any]:
        closed = [t for t in self.trades if t.closed]
        wins = [t for t in closed if t.won]
        losses = [t for t in closed if not t.won]
        open_trades = [t for t in self.trades if t.outcome == "open"]
        timeouts = [t for t in self.trades if t.outcome == "timeout"]
        total_r = sum(t.r_multiple for t in closed)
        gross_win = sum(t.r_multiple for t in wins)
        gross_loss = abs(sum(t.r_multiple for t in losses))
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bars": self.bars,
            "signals": len(self.signals),
            "trades": len(self.trades),
            "closed": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "timeout": len(timeouts),
            "open": len(open_trades),
            "win_rate": (len(wins) / len(closed)) if closed else 0.0,
            "total_r": total_r,
            "avg_r": (total_r / len(closed)) if closed else 0.0,
            "profit_factor": (gross_win / gross_loss) if gross_loss else float("inf") if gross_win else 0.0,
            "avg_bars_held": (
                sum(t.bars_held for t in closed) / len(closed) if closed else 0.0
            ),
            "max_consecutive_losses": _max_consecutive_losses(closed),
        }


def _max_consecutive_losses(closed: list[SimulatedTrade]) -> int:
    worst = run = 0
    for trade in closed:
        run = run + 1 if not trade.won else 0
        worst = max(worst, run)
    return worst


def aggregate(results: list[BacktestResult]) -> dict[str, Any]:
    """Portfolio-level totals across symbols (sums the per-symbol counts)."""
    merged = BacktestResult(symbol="*", timeframe="*")
    for result in results:
        merged.signals.extend(result.signals)
        merged.trades.extend(result.trades)
        merged.bars += result.bars
    stats = merged.stats()
    stats["symbols"] = len(results)
    return stats


# --------------------------------------------------------------------- phase 1
def collect_signals(
    frame: pd.DataFrame, symbol: SymbolConfig, config: AppConfig
) -> list[TradeSignal]:
    """Run the live pipeline over history and return every signal it produced."""
    indicators = compute_indicators(
        frame, symbol.params, touch_line=config.strategy.touch_line
    )
    machines = [
        StrategyMachine(
            symbol=symbol.symbol,
            timeframe=symbol.timeframe,
            direction=direction,  # type: ignore[arg-type]
            params=config.strategy,
            risk=config.risk,
            state=None,
            atr_params=symbol.params.atr,
        )
        for direction in ("buy", "sell")
    ]
    signals: list[TradeSignal] = []
    for bar in iter_bar_inputs(indicators):
        for machine in machines:
            signal = machine.on_bar(bar)
            if signal is not None:
                signals.append(signal)
    return signals


# --------------------------------------------------------------------- phase 2
def simulate_trade(
    signal: TradeSignal,
    frame: pd.DataFrame,
    max_hold_bars: int = DEFAULT_MAX_HOLD_BARS,
) -> SimulatedTrade:
    """Walk the bars after the signal until SL/TP hits or the data runs out."""
    entry_time = pd.Timestamp(signal.bar_time)
    positions = frame.index.get_indexer([entry_time], method="pad")
    position = int(positions[0]) if len(positions) else -1
    if position < 0 or frame.empty:
        # the signal bar is not in this frame: there is nothing to simulate
        return SimulatedTrade(
            signal=signal,
            entry_bar=entry_time,
            exit_bar=None,
            exit_price=float(signal.entry),
            outcome="open",
            bars_held=0,
            r_multiple=0.0,
        )

    entry_bar = frame.index[position]
    entry = float(signal.entry)
    stop_loss = float(signal.stop_loss)
    take_profit = float(signal.take_profit)
    distance = float(signal.sl_distance) or abs(entry - stop_loss)
    is_buy = signal.direction == "buy"
    last_close = float(frame["close"].iloc[-1])

    exit_bar: pd.Timestamp | None = None
    exit_price: float | None = None
    outcome = "open"
    bars_held = 0

    start = position + 1
    stop = min(len(frame), start + int(max_hold_bars))
    for i in range(start, stop):
        row = frame.iloc[i]
        high, low = float(row["high"]), float(row["low"])
        bars_held = i - position
        if is_buy:
            hit_sl, hit_tp = low <= stop_loss, high >= take_profit
        else:
            hit_sl, hit_tp = high >= stop_loss, low <= take_profit
        if hit_sl:  # same bar as a target? the stop is assumed to fill first
            exit_bar, exit_price, outcome = frame.index[i], stop_loss, "sl"
            break
        if hit_tp:
            exit_bar, exit_price, outcome = frame.index[i], take_profit, "tp"
            break
        if i == stop - 1:
            exit_bar, exit_price = frame.index[i], float(row["close"])
            # reached the hold limit -> timeout; ran out of data -> still open
            outcome = "timeout" if stop < len(frame) else "open"
    else:
        exit_price = last_close

    if exit_price is None:
        exit_price = last_close

    signed = (exit_price - entry) if is_buy else (entry - exit_price)
    r_multiple = signed / distance if distance else 0.0
    return SimulatedTrade(
        signal=signal,
        entry_bar=pd.Timestamp(entry_bar),
        exit_bar=pd.Timestamp(exit_bar) if exit_bar is not None else None,
        exit_price=float(exit_price),
        outcome=outcome,
        bars_held=bars_held,
        r_multiple=float(r_multiple),
    )


def run_backtest(
    frame: pd.DataFrame,
    symbol: SymbolConfig,
    config: AppConfig,
    max_hold_bars: int = DEFAULT_MAX_HOLD_BARS,
) -> BacktestResult:
    """Collect signals from `frame`, then simulate each one forward."""
    signals = collect_signals(frame, symbol, config)
    trades = [simulate_trade(signal, frame, max_hold_bars) for signal in signals]
    return BacktestResult(
        symbol=symbol.symbol,
        timeframe=symbol.timeframe,
        bars=len(frame),
        signals=signals,
        trades=trades,
    )
