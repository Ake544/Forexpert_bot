# TradingView parity reference values

`test_parity.py` asserts that our Pine ports reproduce TradingView. Tests skip
automatically while `cases` is empty, so the suite stays green before you have
values to compare against.

## How the values were collected (working workflow)

Chart-data export is paywalled on a free TradingView plan, and the free plan
allows only a couple of indicators per chart. So parity is proven by:

1. **A shared candle source.** TradingView serves `BTCUSDT` from Binance itself,
   so `BINANCE:BTCUSDT` on a 1D/UTC chart has the exact candles of
   `BTCUSDT-1d.csv` (regenerate it with `fetch_binance_csv.py`).
2. **One dumper script.** `parity_dumper.pine` recomputes *all four* indicators
   in a single Pine v5 script and prints a 3-row table with the values plus the
   trend/vslrt bull flags (and the bar's volume, for diagnosing data drift). The
   rows are sampled at the **fixed UTC bar-open times** of its three `targetN`
   inputs, so the output does not depend on when the script is run. Paste it into
   the Pine Editor on the `BINANCE:BTCUSDT` 4H chart, time zone UTC, set the three
   targets to the case times, and read the table.
3. **Read the table** and add a case per row.

When adding cases for an intraday timeframe, both sides must see the same
history depth (see the VWAP warning below), and prefer bars in settled history —
a bar sampled relative to "now" can sit next to a still-forming bar whose volume
TradingView has not finalized, which shows up as a VWAP (and only VWAP) mismatch.

## TradingView facts this harness encodes

* `ta.vwap(source)` with **no anchor** defaults to `timeframe.change("1D")`: a
  **daily session** anchor that resets at the start of each trading day on the
  exchange clock. On a daily chart every bar is its own session, so the VWAP
  line simply equals `close`. Our port resets on UTC days
  (`pine_math.session_vwap`) and the boundary is configurable per profile via
  `ml_vwap.session_tz` / `session_start_hour` for feeds whose day opens off
  midnight UTC (e.g. TradingView's FX day starts at 17:00 `America/New_York`).
* `Trend Meter` plots **states**, not MA values, so `trend_fast`/`trend_slow`
  cannot be read off the chart. `trend_bull` is checked instead.
* `NNFX ATR` truncates to 5 decimals, matching `pine_math.truncate`.

## Case fields

| field | meaning |
|---|---|
| `bar_time` | candle open time, ISO8601 with offset, e.g. `2026-10-11T00:00:00+00:00` |
| `csv_file` | candle file next to this README used to rebuild the bars |
| `ml_window` | the ML Window used on the chart (must match the config) |
| `vwap` | value of the VWAP line at that bar |
| `ml_output` | value of the ML Prediction line at that bar |
| `vslrt_long_line` | the VSLRT long-term line value at that bar |
| `atr` | the NNFX ATR "Main ATR" line value at that bar |
| `trend_bull` / `vslrt_bull` | boolean states at that bar (`as_bool` accepts true/false) |
| `tolerance` | relative tolerance, defaults to 1e-6 |

## Tolerance

`tolerance` defaults to 1e-6 relative, which is tight enough to catch an
incorrect EMA seed or a population-vs-sample stdev mistake. Widen it only if the
broker feed differs from TradingView's candles.
