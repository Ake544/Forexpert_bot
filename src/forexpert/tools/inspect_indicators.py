"""Smoke check for the indicator ports on synthetic candles (no network needed).

Usage:
    python -m forexpert.tools.inspect_indicators --bars 1200 --symbol EURUSD

This prints the last closed bar's indicator values so you can sanity-check the
ports before wiring up real data. It proves the pipeline runs; it does NOT prove
TradingView parity - for that use tests/parity/.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from forexpert.indicators.compute import compute_indicators, snapshot
from forexpert.settings import IndicatorParams, load_config


def synthetic_candles(bars: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=bars, freq="4h", tz="UTC")
    returns = rng.normal(scale=0.0015, size=bars)
    close = 1.0850 * np.exp(np.cumsum(returns))
    spread = np.abs(rng.normal(scale=0.0008, size=bars))
    frame = pd.DataFrame(
        {
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(200, 2000, size=bars).astype("float64"),
        },
        index=index,
    )
    frame.insert(0, "open", np.clip(close + rng.normal(scale=0.0004, size=bars),
                                    frame["low"], frame["high"]))
    frame.index.name = "time"
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=1200)
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    params: IndicatorParams = next(
        (s.params for s in config.symbols if s.symbol == args.symbol), IndicatorParams()
    )

    frame = synthetic_candles(args.bars)
    indicators = compute_indicators(frame, params, touch_line=config.strategy.touch_line)
    result = snapshot(indicators)
    levels = indicators.level_frame()

    print(f"symbol            {args.symbol}")
    print(f"bars              {len(frame)} ({frame.index[0]} -> {frame.index[-1]})")
    print(f"reference level   {result.level!r}  ({config.strategy.touch_line})")
    print(f"ml vwap bull      {result.vwap_bull}")
    print(f"trend bull        {result.trend_bull}  (mode={params.trend_meter.mode})")
    print(f"vslrt bull        {result.vslrt_bull}  (long line={result.vslrt_long_line})")
    print(f"atr               {result.atr}")
    print(f"kro enabled       {params.kro.enabled}")
    print(f"level na count    {int(levels['level'].isna().sum())}")
    print(f"green bars        {int(levels['vwap_bull'].eq(True).sum())}")


if __name__ == "__main__":
    main()