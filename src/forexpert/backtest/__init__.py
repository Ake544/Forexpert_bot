"""Backtesting: replay history through the live pipeline, then simulate fills."""
from __future__ import annotations

from .engine import (
    DEFAULT_MAX_HOLD_BARS,
    BacktestResult,
    SimulatedTrade,
    aggregate,
    collect_signals,
    run_backtest,
    simulate_trade,
)

__all__ = [
    "DEFAULT_MAX_HOLD_BARS",
    "BacktestResult",
    "SimulatedTrade",
    "aggregate",
    "collect_signals",
    "run_backtest",
    "simulate_trade",
]
