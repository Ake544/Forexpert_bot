"""Rebuild `BTCUSDT-1d.csv` from Binance's public REST API.

Why BTCUSDT: TradingView serves `BTCUSDT` from Binance itself, so a chart export
and this file contain *the same candles*. That lets a free TradingView plan
(where chart-data export is paywalled) still prove indicator parity - the only
thing needed from the user is reading a few numbers off the Data window.

The full daily history is fetched on purpose: it matches what TradingView loads
for BINANCE:BTCUSDT, so both sides see the same candles. On a daily chart the
session VWAP is degenerate (one bar per session), so the anchor depth does not
change the VWAP. Re-run when you want fresh bars:

    python tests/parity/fetch_binance_csv.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

URL = "https://api.binance.com/api/v3/klines"
COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time"]


def fetch_all(interval: str = "1d") -> pd.DataFrame:
    rows: list[list] = []
    start = 0
    while True:
        response = requests.get(
            URL,
            params={"symbol": "BTCUSDT", "interval": interval, "limit": 1000, "startTime": start},
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        start = batch[-1][0] + 1
    frame = pd.DataFrame(rows).iloc[:, :7]
    frame.columns = COLUMNS
    frame["time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = frame[column].astype(float)
    return frame.set_index("time")[["open", "high", "low", "close", "volume"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", default="1d", help="Binance interval, e.g. 1d or 4h")
    args = parser.parse_args()

    output = Path(__file__).parent / f"BTCUSDT-{args.interval}.csv"
    frame = fetch_all(args.interval)
    frame.to_csv(output, float_format="%.8f")
    closed = frame.iloc[:-1]  # the newest bar is usually still forming
    print(f"{output.name}: {len(frame)} bars")
    print(f"first bar {frame.index[0]}  close {frame['close'].iloc[0]:.2f}")
    print(f"last bar  {frame.index[-1]}  close {frame['close'].iloc[-1]:.2f}")
    print(f"last closed bar {closed.index[-1]}  close {closed['close'].iloc[-1]:.2f}")
    print()
    for offset in (2, 20, 200):
        print(f"suggested closed bar (-{offset}): {closed.index[-offset]}  close {closed['close'].iloc[-offset]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
