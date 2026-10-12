"""Print this repo's indicator values for bars of a candle CSV.

    python tests/parity/dump_values.py tests/parity/EURCHF1d.csv
    python tests/parity/dump_values.py tests/parity/EURCHF1d.csv --bars 2024-03-01T00:00:00+00:00

Used while collecting TradingView reference values: run this on the exported
chart data, put the table next to the TradingView data window, and only the
TradingView numbers need retyping into reference_values.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from test_parity import load_csv  # noqa: E402

from forexpert.indicators.compute import compute_indicators  # noqa: E402
from forexpert.settings import IndicatorParams  # noqa: E402


def values_for(frame: pd.DataFrame, bars: list[str] | None) -> pd.DataFrame:
    indicators = compute_indicators(frame, IndicatorParams(), touch_line="vwap")
    levels = indicators.level_frame()
    table = pd.DataFrame(
        {
            "close": indicators.candles["close"],
            "vwap": indicators.ml_vwap.vwap,
            "ml_output": indicators.ml_vwap.ml_output,
            "trend_fast": indicators.trend_meter.fast,
            "trend_slow": indicators.trend_meter.slow,
            "vslrt_long_line": levels["vslrt_long_line"],
            "atr": levels["atr"],
            "vwap_bull": levels["vwap_bull"],
            "trend_bull": levels["trend_bull"],
            "vslrt_bull": levels["vslrt_bull"],
        },
        index=levels.index,
    )
    if bars:
        wanted = [pd.Timestamp(bar) for bar in bars]
        missing = [str(b) for b in wanted if b not in table.index]
        if missing:
            raise SystemExit(
                f"not in the CSV: {', '.join(missing)}\n"
                f"first bar {table.index[0]}, last bar {table.index[-1]}"
            )
        return table.loc[wanted]
    return table.tail(3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="TradingView chart-data export")
    parser.add_argument("--bars", nargs="*", default=None, help="ISO8601 bar times")
    args = parser.parse_args(argv)

    frame = load_csv(args.csv)
    print(f"{args.csv.name}: {len(frame)} bars, {frame.index[0]} .. {frame.index[-1]}")
    table = values_for(frame, args.bars)
    print()
    print(table.to_string(float_format=lambda value: f"{value:.8f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
