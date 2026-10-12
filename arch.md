# Forexpert — System Architecture

Rule-based signal engine that watches a fixed set of (symbol, timeframe) pairs,
runs a TradingView-derived indicator stack over **closed candles only**, and
emits an alert — entry, stop loss, take profit, risk-reward and risk-sized lot —
when a three-candle setup completes and price touches the reference level.

The system **never places an order**. It produces a journal record and a
notification; execution is deliberately out of scope for now.

---

## 1. High-level pipeline

```
                       ┌──────────────────────────────────────────────────────┐
   provider APIs ───►  │  data/                                               │
   (Twelve Data,        │    DukascopySource / TwelveDataSource /             │
    Binance, Dukascopy, │    BinanceSource / CsvSource                        │
    local CSV)          │    • decode → canonical UTC frame                   │
                       │    • drop the forming bar                           │
                       │    • shared rate limiter + retry/backoff            │
                       └───────────────┬──────────────────────────────────────┘
                                       │  pd.DataFrame[open,high,low,close,volume]
                                       │  DatetimeIndex name='time', tz=UTC
                       ┌───────────────▼──────────────────────────────────────┐
                       │  store/sqlite_store                                 │
                       │    upsert_candles() → data/forexpert.sqlite         │
                       └───────────────┬──────────────────────────────────────┘
                                       │  replayable history
                       ┌───────────────▼──────────────────────────────────────┐
                       │  indicators/compute                                 │
                       │    ml_vwap → trend_meter → vslrt → atr → level       │
                       │    compute_indicators(frame) → IndicatorFrame        │
                       │    snapshot(frame)          → Snapshot (last bar)    │
                       └───────────────┬──────────────────────────────────────┘
                                       │  BarInput (tri-state booleans)
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
             StrategyMachine("buy")            StrategyMachine("sell")
             IDLE → CANDLE1_OK →               same states, mirrored
             CANDLE2_OK → WAITING_TOUCH
             → COOLDOWN → IDLE
                      │                                 │
                      └────────────────┬────────────────┘
                                       │  TradeSignal (deterministic signal_id)
                       ┌───────────────▼──────────────────────────────────────┐
                       │  store.record_signal()  ← cross-restart dedupe      │
                       │  journal.append()       ← data/journal.jsonl        │
                       │  notifier.send()        ← Telegram / Discord        │
                       └──────────────────────────────────────────────────────┘
```

One **runner loop** drives all symbols. Each (symbol, timeframe) owns its own
`StrategyMachine` pair and its own persisted `SetupState`.

---

## 2. Repository layout

```
Forexpert/
├── config/
│   ├── config.yaml          # every tunable; ${ENV_VAR} secret references
│   ├── .env.example         # secret template, committed
│   └── .env                 # real secrets, git-ignored
├── data/
│   ├── forexpert.sqlite     # candles + state + signals + processed bars
│   ├── journal.jsonl        # append-only signal audit log
│   └── csv/                 # offline candle dumps for backtests
├── src/forexpert/
│   ├── settings.py          # YAML + env loading, typed frozen params
│   ├── cli.py               # `forexpert run|check|backtest`
│   ├── __main__.py          # python -m forexpert
│   ├── runner.py            # the live loop (fetch→indicators→machines→notify)
│   ├── journal.py           # append-only JSONL decision log
│   ├── data/                # candle sources (provider adapters)
│   ├── indicators/          # Pine → numpy/pandas ports
│   ├── strategy/            # the state machine
│   ├── risk/                # ATR levels + position sizing
│   ├── store/               # SQLite persistence
│   ├── notify/              # Telegram / Discord / fan-out
│   └── backtest/            # signal replay + fill simulation
├── tests/
│   ├── test_*.py            # deterministic unit tests, no network
│   └── parity/              # TradingView reference-value tests
│       ├── reference_values.yaml
│       └── test_parity.py
├── Indicators.txt           # original Pine sources (the spec)
└── trading strategy.docx    # the rule document this implements
```

---

## 3. Modules

### 3.1 `settings.py` — configuration

* Every tunable lives in `config/config.yaml`. Secrets never do.
* `${TWELVE_DATA_API_KEY}`-style references are expanded recursively after
  `config/.env` is loaded into `os.environ` (`os.environ.setdefault`, so a real
  environment variable always wins over the file).
* `no python-dotenv` dependency — `load_dotenv()` is ~20 lines and ships with us.
* Profiles support **one-level** `extends:` inheritance (`default` → `ml350`,
  `tm_sma200`); an unknown key in a dataclass section is a `ConfigError`, so a
  typo in config can never be silently ignored.
* Output: `AppConfig` with typed frozen dataclasses (`StrategyParams`,
  `RiskParams`, `IndicatorParams`, `NotifierParams`) and
  `SymbolConfig` per monitored pair.

**Design rule:** no module constructs its own configuration; everything is
injected from `AppConfig`, which keeps tests independent of the real config file.

### 3.2 `data/` — candle sources

| Source | URL / file | Volume | Auth |
|---|---|---|---|
| `TwelveDataSource` | `/time_series` | **null for spot FX** | `TWELVE_DATA_API_KEY` |
| `Mt5Source` | local terminal via `copy_rates_from_pos` | broker **tick volume** | terminal session (`MT5_*`) |
| `BinanceSource` | `/api/v3/klines` | real | none (public) |
| `DukascopySource` | `datafeed…/{Y}/{MM0}/{DD}/{HH}h_ticks.bi5` | tick count | none (currently unreachable) |
| `CsvSource` | `data/csv/{SYMBOL}_{tf}.csv` | whatever is in the file | none |

All five satisfy the same `CandleSource` protocol and return one canonical frame:

* `DatetimeIndex` named `time`, **tz-aware UTC**, strictly increasing, no duplicates
* columns `open, high, low, close, volume`, all numeric
* **only closed bars** — `drop_forming_bars()` removes any bar whose
  `open + timeframe_duration > now`, so no provider-specific "is the market
  open?" logic is needed
* unsupported rows fail loudly (`DataError`) instead of being silently coerced

Cross-cutting concerns live in `base.py`:

* `RateLimiter` — thread-safe spacing limiter, one instance **shared per
  provider** so 19 symbols cannot burst a 8/min quota.
* `finalize_frame()` — sort, dedupe, drop the forming bar, coerce dtypes;
  accepts `timestamp_unit` because Binance sends epoch **milliseconds** while
  the others send ISO strings.
* Retries use exponential backoff and honour `Retry-After`.

**Provider-specific gotchas encoded in the code:**

* Twelve Data returns values **newest-first**, wants `timezone=UTC` (otherwise
  daily bars land on the wrong UTC day), reports errors as HTTP 200 with
  `{"status":"error"}`, and is included **with** the forming bar.
* Binance klines are 12-element arrays at fixed offsets; the last one is always
  the forming bar.
* Dukascopy records are LZMA-compressed `>IIIff` structs (20 bytes) whose price
  is `value / 1000` and whose offset is **milliseconds since the containing
  hour**, not an epoch. Month in the URL path is **zero-based**.
* MT5 rates are a structured NumPy array (`time,open,high,low,close,tick_volume,
  real_volume,spread`) with `time` in **epoch seconds**; `copy_rates_from_pos`
  includes the forming bar, the terminal must be initialized **once** per
  process (one shared source instance), brokers rename symbols (`symbol_suffix`),
  and the terminal has no 2h/6h/12h-free guarantee — unsupported timeframes are
  rejected rather than rounded.

#### FX volume: MetaTrader 5 (resolved and live-verified)

Spot FX has no centralized volume: Twelve Data returns `null` for every pair,
which would disable the volume-weighted ML VWAP bands on 18 of 19 symbols.
Three routes were tried:

| Option | Result |
|---|---|
| Dukascopy tick volume | unreachable — `datafeed` 503/429/timeouts, `datafeed2`/`.eu` do not resolve (0/8 requests) |
| Finnhub `forex/candle` | key valid (`/quote`, `/search`, `/forex/symbol` → 200) but **every `/candle` endpoint returns 403** — historical candles are paid-only |
| **MetaTrader 5** | **chosen and verified live** |

`Mt5Source` reads the local terminal over IPC (no API quota): `tick_volume`
per bar — the same quantity TradingView's FX charts display — with
`real_volume` selectable. Live verification against the terminal: 17 of 17
configured symbol/timeframe pairs returned 998–999 bars with **non-zero
volume on every bar**, and ML VWAP + ATR computed for all of them.

Two broker realities are handled in code:

* **Missing crosses** — MetaQuotes quotes `GBPAUD`/`EURAUD` but not the
  mirrors `AUDGBP`/`AUDEUR`. `_resolve_symbol()` falls back to the inverse
  pair and `invert_frame()` flips it (prices `1/x`, `high`/`low` swapped,
  volume unchanged), so the configured symbols stay as the strategy defines.
* **Depth** — the demo feed serves ~3000 4h bars (from 2024-10) and ~3000
  daily bars (from 2015), far beyond the 1000 bars the indicators need.

### 3.3 `indicators/` — the indicator stack

| Module | Origin | Produces |
|---|---|---|
| `ml_vwap.py` | "VWAP Bands with ML [CryptoSea]" | reference level, banded ML line, `vwap_bull` |
| `trend_meter.py` | "Trend meter by Lij_MC" | `trend_bull` in three modes: `crossover`, `slow_direction`, `three_meters` |
| `vslrt.py` | "Volume-Supported Linear Regression Trend" (LonesomeTheBlue) | `vslrt_bull`, `vslrt_long_line` |
| `atr_nnfx.py` | "NNFX ATR" (sueun123) | ATR via RMA/SMMA, used for SL/TP distance |
| `pine_math.py` | — | shared Pine helpers (`truncate`, …) |
| `kro.py` | QuantraSystems KRO | **raises** — see below |

`compute.py` is the aggregate entry point:

```python
indicators = compute_indicators(frame, params, touch_line=cfg.strategy.touch_line)
snap       = snapshot(indicators)     # values of the single most recent CLOSED bar
```

`validate_candles()` enforces the OHLCV contract at the boundary (sorted,
tz-aware, `high >= max(open,close) >= min(open,close) >= low`, no null volume).
`IndicatorFrame.level_frame()` exposes the exact column names the state machine
reads: `open high low close level vwap_bull trend_bull vslrt_bull vslrt_long_line atr`.

**KRO is deliberately unported.** The Pine source imports
`QuantraSystems/DynamicFunctions/1`, whose `waveCalculation`,
`kernelRegression` (×3) and `stdv_bands` were never supplied. Reimplementing
them from the published LuxAlgo variant would silently change the signal, so
`kro_confirmation()` raises `KroNotPortedError` and `kro.enabled: false` in
config is the only way to run. This is a **known gap**, not an oversight.

### 3.4 `strategy/` — the state machine

`machine.py` implements one `StrategyMachine` per
`(symbol, timeframe, direction)`:

```
IDLE ──close beyond level──► CANDLE1_OK ──open AND close beyond level──► CANDLE2_OK
                                                                            │
                                          ◄── price touches the level ──────┘
                                          ▼
                             WAITING_TOUCH ──signal──► COOLDOWN ──N bars──► IDLE
```

* **Candle 1**: close strictly beyond the level (the wick alone is not enough).
* **Candle 2**: must be the *immediately following* bar and must **open and
  close** beyond the level. Failure resets, with an optional `reseed`.
* **Entry window**: the earliest entry candle is the bar after candle 2; the
  setup dies after `expiry_bars` bars, or immediately when a bar closes on the
  wrong side of the level (`expire_on_wrong_side_close`).
* **Touch**: `low <= level <= high` (`touch_definition: wick`) or the close test.
* **Filters**: `filters_on_entry_bar: true` requires VWAP + Trend Meter to be
  bullish *on the touch bar*; `false` makes them sticky — latched at any point
  from candle 1 onward.
* **Confirmation**: VSLRT, or KRO once ported. None ⇒ no signal.
* **Cooldown**: `cooldown_bars` bars are swallowed after a signal so one setup
  cannot re-fire.

**Correctness invariants**

1. **Fail-closed.** A `None` indicator state means "not ready" and is never
   treated as `False`. Missing level, missing ATR or unknown filter state block
   the signal (`reject_reason` is set to a machine-readable string).
2. **Monotonic bars.** Feeding the same or an earlier bar raises `BarOrderError`.
3. **Deterministic identity.** `signal_id = sha1(symbol|tf|direction|bar_time)[:16]`
   — recomputed from the bar itself, so dedupe survives a process restart.
4. **Bar distance is counted, never derived from wall time** —
   `bars_from_candle1 = state.bars_seen - state.candle1_seen`. Inferring
   duration from timestamps would break across weekends, gaps and backfills.
5. **Tri-state booleans** (`True`/`False`/`None`) propagate from indicators to
   `BarInput`, so a warming-up VWAP can never masquerade as a valid short.

State (`SetupState`) is a plain dataclass with `to_dict()`/`from_dict()` so it
can be persisted per key `symbol:timeframe:direction` and restored on boot.

### 3.5 `risk/levels.py` — levels and sizing

```python
levels = build_levels(entry, atr, direction, sl_multiplier=1.5, tp_multiplier=2.25)
lots    = position_size(levels, risk)
```

* SL distance = `atr * sl_multiplier`, TP distance = `atr * tp_multiplier`
  (1.5R/2.25R → RR ≈ 1.5). Non-positive ATR raises instead of producing a
  zero-distance stop.
* `sl_distance`/`tp_distance` are **price** units; a lot costs
  `sl_distance * assumed_contract_size` currency.
* Sizing risks `risk_percent_per_trade` of the balance, then rounds **down** to
  `lot_step`. If the result is below `min_lot` the size is `0.0` (skip the
  trade) rather than rounding **up** to `min_lot`, which would breach the risk
  budget. Upper bound is `max_lot`.

### 3.6 `store/sqlite_store.py` — persistence

SQLite in WAL mode. Tables:

| Table | Purpose |
|---|---|
| `candles` | OHLCV history, PK `(symbol, timeframe, open_time)` — upsert, idempotent |
| `strategy_state` | serialized `SetupState` per key, survives restart |
| `signals` | PK `signal_id` → `INSERT OR IGNORE`, cross-process dedupe |
| `processed_bars` | bars already fed to the machine → ingestion idempotency |

* `unprocessed_bars(symbol, timeframe, since)` gives the restart catch-up list
  (`runtime.max_catchup_bars` bounds it).
* `InMemoryStateStore` implements the same protocol for tests and ephemeral runs.
* Timestamps are stored as **UTC epoch seconds**; `from_epoch()` always
  re-localizes to UTC.

### 3.7 Planned components

| Module | Responsibility |
|---|---|
| `runner.py` | wake schedule around candle close, load state, replay catch-up bars, feed both machines, persist, journal, notify, alert on stale data / errors |
| `journal.py` | append-only JSONL audit trail of every decision (signals *and* rejections) for later analysis |
| `notify/` | `TelegramNotifier` / `DiscordNotifier` with retry + backoff, deduped by `signal_id` |
| `backtest/` | replay a cached candle history through the same machine, report equity/results — the machine is already pure, so this is mostly plumbing |
| `tools/inspect_indicators.py` | already present: dumps indicator columns for a symbol |

---

## 4. Runtime behaviour (target design)

1. **Boot** — load config, open SQLite, rebuild each `SetupState`.
2. **Catch-up** — `last_processed_time()` → replay up to `max_catchup_bars`
   missing closed bars (bounded so a long outage cannot stall startup).
3. **Steady state** — wake `wake_delay_seconds` after each expected candle
   close; fetch → upsert → recompute → feed machines.
4. **On signal** — `record_signal()` (ignored if it already exists), append to
   the journal, notify, log.
5. **Alerts** — stale data (`alerts.stale_data_bars`), errors with a
   `error_cooldown_minutes` throttle, optional startup notice.

Failures are isolated per symbol: one symbol erroring must not stop the others.

---

## 5. Correctness principles

* **Closed bars only** — the forming bar is removed at the source boundary;
  otherwise indicators would read forward-looking data and signals would
  evaporate when the bar settled.
* **Idempotency** — candles, signals and processed bars are all keyed and
  upserted, so a crash-restart re-ingests nothing.
* **Fail closed** — unknown/unready indicator state blocks a signal; there is no
  permissive default.
* **UTC everywhere** — naive timestamps are rejected at boundaries.
* **No secrets in code or config** — `${VAR}` indirection only.
* **Determinism over convenience** — bar counts counted, signal ids hashed from
  content, lot sizes rounded down.

---

## 6. Testing strategy

| Layer | Approach |
|---|---|
| Indicators | fixed input series with hand-computed expectations |
| State machine | `conftest.make_bar()` builds deterministic candles around a 100.0 level; sequences cover valid/invalid/expiry/cooldown/sticky/confirmations |
| Risk | sizing and rounding rules, incl. the `min_lot` skip |
| Data layer | fake HTTP sessions and synthetic LZMA payloads — **no network in the suite** |
| Parity | `tests/parity/reference_values.yaml` holds values transcribed from TradingView (3 BTCUSDT 1D + 2 BTCUSDT 4H cases); `parity_dumper.pine` recomputes all four indicators on the TV chart |
| End-to-end | backtest over cached candles (planned) |

Run with `python -m pytest`.

---

## 7. Security

* `config/.env` and `data/` are git-ignored; only `.env.example` is committed.
* `settings.load_dotenv()` writes to `os.environ` with `setdefault`, so a genuine
  env var (CI, container) always wins over the file.
* URLs containing `apikey` are logged with the key stripped.
* The system is read-only with respect to broker accounts — no credential for
  execution exists.

---

## 8. Key configuration reference

| Key | Meaning |
|---|---|
| `runtime.db_path` | SQLite location |
| `runtime.max_catchup_bars` | bounded restart replay |
| `data.volume_policy` | `strict` = never compute VWAP without volume |
| `strategy.touch_definition` | `wick` \| `close` |
| `strategy.expiry_bars` | bars beyond candle 2 before the setup dies |
| `strategy.filters_on_entry_bar` | strict-on-touch vs sticky-from-candle-1 |
| `risk.risk_percent_per_trade` | risk per trade in % of balance |
| `symbols[].profile` | which indicator profile applies |

---

## 9. Status and open gaps

**Built and tested (249 unit tests passing):** config, all five candle sources,
all ported indicators, the state machine, risk levels, SQLite persistence, the
runtime (`runner.py` → fetch → indicators → both machines → store →
`journal.jsonl` → notifiers, with per-symbol error isolation, bounded catch-up,
stale-data alerting and cross-restart signal dedupe), the CLI
(`forexpert run|check|backtest`) and the backtester.

**Backtester** (`backtest/engine.py`) reuses the *live* pipeline — same
`compute_indicators`, same `iter_bar_inputs`, same buy/sell `StrategyMachine` —
then walks forward from each signal applying the ATR stops. Fills start on the
**bar after** the signal (the entry bar cannot stop out its own trade),
same-bar stop+target assumes the **stop** filled (worst case), and trades still
running at the end of the data are reported as `open`/`timeout` instead of being
dropped. No spread/slippage is modelled — results are price-level results.
First run over the full book: **44 signals, 40.9% win rate, +1.00R total**
(expectancy +0.023R/trade at a 1.5R target), reported per symbol plus a total.
After the VWAP anchor fix below the same 18-pair book reported **146 signals,
41.8% win, +6.50R** — the fix is material, not cosmetic. The 8 dead 1D pairs were
then disabled (see below), so the monitored book is **10 4H pairs: 136 signals,
40.7% win, +2.50R** (expectancy +0.019R/trade).

**CLI:**

| Command | What it does |
|---|---|
| `forexpert run` | live loop (`--once`, `--max-passes N`) |
| `forexpert check` | read-only health table for every symbol: bars, volume, VWAP, ATR |
| `forexpert backtest` | replay history (`--symbol`, `--timeframe`, `--bars`, `--max-hold`, `--json`); an explicit `--symbol` also reaches disabled pairs |

**Live-verified:** FX volume (MT5, 17/17 symbol configs, volume never zero) and
the full runner loop over the (then 18-pair) book — cold start baselines exactly
one bar per symbol, a second process immediately after replays nothing and emits
no duplicate signal.

**Parity (green):** `tests/parity/` proves the ports against real TradingView
values. Chart-data export is paywalled on a free plan and only ~2 indicators fit
per chart, so `parity_dumper.pine` recomputes all four indicators in one Pine v5
script and prints a 3-row table (1/20/200 bars back); its output is transcribed
into `reference_values.yaml`. TradingView serves `BTCUSDT` from Binance, so the
`BINANCE:BTCUSDT` 1D/UTC chart shares its candles with the bundled
`BTCUSDT-1d.csv`. Three bars across 2026 matched to 1e-6 relative on
`vslrt_long_line` and `atr` and exactly on the bull flags, which caught the VWAP
bug below.

**VWAP anchor bug (found by parity, fixed):** Pine's `ta.vwap(source)` with no
anchor defaults to `timeframe.change("1D")` — a **daily session** anchor, so on a
daily chart it equals `close` (TradingView returned exactly the close where our
port returned a 9-year cumulative average ~29k against an 84k price). Our port
had anchored at the first bar of the series instead. `pine_math.session_vwap`
now resets the running sums at each **UTC** trading day (bars without datetime
information are treated as one session). Consequence: on 1D symbols `vwap_bull`
collapses to a lagged-SMA/close test and the touch line equals close; on 4h
symbols the VWAP resets daily. Known deviation: FX feeds whose daily bar opens
off midnight UTC (most brokers) would need their own session boundary.

**1D degeneracy (consequence of the fix, DONE):** because the session VWAP on a
daily bar *is* the close, `touch_line: vwap` makes the state machine's first
candle (`close` beyond the level) impossible — verified 0 signals over 3344
daily bars, reaching `candle2_failed` only on float noise. Switching 1D to
`ml_output` revives it barely (7 signals in 9 years vs 56 on 4H), so the strategy
is intraday-only. The 8 1D pairs are now `enabled: false` in `config.yaml`; the
monitored book is 10 4H pairs. An explicit `forexpert backtest --symbol X` still
reaches a disabled pair, so a candidate can be tested before enabling it.

**Open:**

1. **Parity coverage** — 3 BTCUSDT 1D + 2 BTCUSDT 4H cases pass. The 4H cases
   are session-first bars (VWAP = close). A mid-session 4H bar (`2026-10-12
   04:00`) did **not** reproduce: TV returned vwap 83724.42 and atr 597.65276
   against our 83824.41 / 597.86526, while `close` and `ml_output` matched
   exactly, and no source/window on `BTCUSDT-4h.csv` reproduces TV's vwap. This
   points at TV's recent 4H bars or volume differing from the Binance REST
   snapshot (not a session-rule bug, since session-first bars match) — needs a
   fresh read at a mid-session bar to confirm. Also consider a broker-day session
   boundary for the FX feeds.
2. **1D viability** — re-enable the 8 1D pairs only with a different level or
   anchoring (the ML VWAP degenerates on daily bars), and the expected signal
   count is very low.
3. **KRO confirmation** — blocked on the missing `QuantraSystems` library source.
4. **Live FX feed stability** — MetaQuotes demo history can be trimmed by the
   broker; `DukascopySource` stays as a tested fallback.
