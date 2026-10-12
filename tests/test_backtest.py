"""Backtester tests: fill simulation, statistics, and one end-to-end replay.

`collect_signals` is exercised against a fixed-seed synthetic candle series
(a seeded generator, so the signals are reproducible without shipping fixtures).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from forexpert.backtest import (
    BacktestResult,
    SimulatedTrade,
    aggregate,
    collect_signals,
    run_backtest,
    simulate_trade,
)
from forexpert.data.base import OHLCV_COLUMNS
from forexpert.settings import SymbolConfig, load_config

ENTRY = 100.0
SL = 99.0
TP = 101.5
BARS = pd.date_range("2024-01-01", periods=8, freq="4h", tz="UTC")
BARS.name = "time"


def frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """(open, high, low, close) rows on the shared 4h index, volume 1."""
    data = [list(row) + [1.0] for row in rows]
    return pd.DataFrame(data, columns=list(OHLCV_COLUMNS), index=BARS[: len(rows)])


def make_signal(
    direction: str = "buy",
    bar_time: datetime | None = None,
    entry: float = ENTRY,
    stop_loss: float = SL,
    take_profit: float = TP,
) -> object:
    from forexpert.strategy.state import TradeSignal

    when = bar_time or BARS[0].to_pydatetime()
    return TradeSignal(
        signal_id="sig",
        symbol="TEST",
        timeframe="4h",
        direction=direction,  # type: ignore[arg-type]
        bar_time=when,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        atr=1.0,
        sl_distance=abs(entry - stop_loss),
        tp_distance=abs(take_profit - entry),
        risk_reward=abs(take_profit - entry) / abs(entry - stop_loss),
        level=ENTRY,
        close=entry,
        confirmations=(),
        vwap_bull=direction == "buy",
        trend_bull=direction == "buy",
        vslrt_bull=direction == "buy",
        bars_from_candle1=1,
        bars_from_candle2=1,
    )


# ------------------------------------------------------------------- fills
def test_take_profit_is_filled_at_the_target_on_the_next_bar() -> None:
    candles = frame([(100.0, 100.2, 99.8, 100.0), (100.0, 102.0, 99.9, 101.8)])
    trade = simulate_trade(make_signal(), candles)

    assert trade.outcome == "tp"
    assert trade.exit_price == pytest.approx(TP)
    assert trade.exit_bar == BARS[1]
    assert trade.bars_held == 1
    assert trade.r_multiple == pytest.approx(1.5)  # 1.5R target on a 1R stop
    assert trade.won and trade.closed


def test_stop_loss_costs_exactly_one_r() -> None:
    candles = frame([(100.0, 100.5, 99.5, 100.0), (99.5, 99.6, 98.0, 98.2)])
    trade = simulate_trade(make_signal(), candles)

    assert trade.outcome == "sl"
    assert trade.exit_price == pytest.approx(SL)
    assert trade.r_multiple == pytest.approx(-1.0)


def test_the_signal_bar_cannot_stop_the_trade_it_created() -> None:
    # bar 0 already trades below the stop; simulation must start on bar 1
    candles = frame(
        [
            (100.0, 100.4, 97.5, 100.0),  # signal bar, low pierces the stop
            (100.0, 100.6, 99.7, 100.2),
            (100.2, 102.5, 100.1, 102.0),
        ]
    )
    trade = simulate_trade(make_signal(), candles)

    assert trade.outcome == "tp"
    assert trade.bars_held == 2
    assert trade.exit_bar == BARS[2]


def test_same_bar_stop_and_target_assumes_the_stop_filled_first() -> None:
    candles = frame([(100.0, 100.2, 99.9, 100.0), (100.0, 105.0, 95.0, 100.0)])
    trade = simulate_trade(make_signal(), candles)

    assert trade.outcome == "sl"
    assert trade.r_multiple == pytest.approx(-1.0)


def test_sell_direction_is_mirrored() -> None:
    # a sell stops above the entry and targets below
    candles = frame([(100.0, 100.4, 99.6, 100.0), (100.0, 100.3, 98.0, 98.2)])
    trade = simulate_trade(
        make_signal(direction="sell", stop_loss=101.0, take_profit=98.5), candles
    )

    assert trade.outcome == "tp"
    assert trade.exit_price == pytest.approx(98.5)
    assert trade.r_multiple == pytest.approx(1.5)


def test_trade_running_at_the_end_of_the_data_stays_open() -> None:
    candles = frame([(100.0, 100.5, 99.5, 100.0), (100.0, 100.6, 99.6, 100.4)])
    trade = simulate_trade(make_signal(), candles)

    assert trade.outcome == "open"
    assert not trade.closed
    assert trade.exit_price == pytest.approx(100.4)  # marked at the last close
    assert trade.r_multiple == pytest.approx(0.4)    # 100.4 - 100.0 over 1.0 stop


def test_max_hold_turns_a_stuck_trade_into_a_timeout() -> None:
    candles = frame(
        [(100.0, 100.5, 99.5, 100.0)]
        + [(100.0, 100.5, 99.5, 100.0)] * 5
    )
    trade = simulate_trade(make_signal(), candles, max_hold_bars=2)

    assert trade.outcome == "timeout"
    assert trade.bars_held == 2
    assert trade.exit_bar == BARS[2]


def test_signal_bar_absent_from_the_frame_is_reported_open() -> None:
    candles = frame([(100.0, 100.5, 99.5, 100.0)])
    later = make_signal(bar_time=BARS[5].to_pydatetime())
    trade = simulate_trade(later, candles)

    assert trade.outcome == "open"
    assert trade.bars_held == 0
    assert trade.r_multiple == 0.0


# ---------------------------------------------------------------- statistics
def _fake(outcome: str, r: float) -> SimulatedTrade:
    signal = make_signal()
    return SimulatedTrade(
        signal=signal,
        entry_bar=BARS[0],
        exit_bar=BARS[1],
        exit_price=TP if outcome == "tp" else SL,
        outcome=outcome,
        bars_held=1,
        r_multiple=r,
    )


def test_stats_count_wins_losses_and_streaks() -> None:
    result = BacktestResult(
        symbol="TEST",
        timeframe="4h",
        bars=100,
        signals=[make_signal()],  # type: ignore[list-item]
        trades=[
            _fake("tp", 1.5),
            _fake("sl", -1.0),
            _fake("sl", -1.0),
            _fake("tp", 1.5),
            _fake("open", 0.4),
            _fake("timeout", -0.2),
        ],
    )
    stats = result.stats()

    assert stats["signals"] == 1
    assert stats["trades"] == 6
    assert stats["closed"] == 4
    assert stats["wins"] == 2 and stats["losses"] == 2
    assert stats["open"] == 1 and stats["timeout"] == 1
    assert stats["win_rate"] == pytest.approx(0.5)
    assert stats["total_r"] == pytest.approx(1.0)   # realized only: open/timeout excluded
    assert stats["profit_factor"] == pytest.approx(1.5)  # 3.0R won / 2.0R lost
    assert stats["max_consecutive_losses"] == 2


def test_aggregate_sums_across_symbols() -> None:
    one = BacktestResult(
        symbol="A", timeframe="4h", bars=10, trades=[_fake("tp", 1.5)]
    )
    two = BacktestResult(
        symbol="B", timeframe="4h", bars=20, trades=[_fake("sl", -1.0)]
    )
    total = aggregate([one, two])

    assert total["symbols"] == 2
    assert total["bars"] == 30
    assert total["closed"] == 2
    assert total["total_r"] == pytest.approx(0.5)


# ------------------------------------------------------------------ end to end
def synthetic_candles(seed: int = 10, count: int = 600) -> pd.DataFrame:
    """Seeded random walk — deterministic, so the signals it fires are stable."""
    rng = np.random.default_rng(seed)
    close = 1.0 + np.cumsum(rng.normal(0.0004, 0.004, count))
    opened = np.roll(close, 1)
    opened[0] = close[0]
    high = np.maximum(opened, close) + np.abs(rng.normal(0, 0.003, count))
    low = np.minimum(opened, close) - np.abs(rng.normal(0, 0.003, count))
    volume = np.maximum(1.0, rng.normal(1000, 300, count))
    index = pd.date_range("2024-01-01", periods=count, freq="4h", tz="UTC")
    index.name = "time"
    return pd.DataFrame(
        {
            "open": opened,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def symbol_config() -> SymbolConfig:
    config = load_config()
    template = config.symbols[0]
    return SymbolConfig(
        symbol="TEST",
        timeframe="4h",
        provider="csv",
        profile=template.profile,
        params=template.params,
    )


def test_collect_signals_replays_the_live_pipeline() -> None:
    config = load_config()
    signals = collect_signals(synthetic_candles(), symbol_config(), config)

    assert signals, "the seeded series must produce at least one signal"
    assert all(s.entry > 0 and s.stop_loss > 0 and s.take_profit > 0 for s in signals)
    assert len({s.signal_id for s in signals}) == len(signals)  # ids are unique
    times = [s.bar_time for s in signals]
    assert times == sorted(times)  # collected oldest first


def test_run_backtest_simulates_every_signal_it_found() -> None:
    config = load_config()
    result = run_backtest(synthetic_candles(), symbol_config(), config)

    assert len(result.trades) == len(result.signals) >= 1
    stats = result.stats()
    assert stats["closed"] + stats["open"] + stats["timeout"] == stats["trades"]
    assert -1.0 <= stats["win_rate"] <= 1.0
    for trade in result.trades:
        if trade.outcome == "sl":
            assert trade.r_multiple == pytest.approx(-1.0)
