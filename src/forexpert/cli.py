"""Command line entry point: ``python -m forexpert <command>`` (or ``forexpert``).

Commands
--------
run      the live loop (fetch → indicators → machines → journal → notify)
check    fetch every configured symbol and print data health — read-only
doctor   preflight the environment (config, dependencies, paths, notifier)
backtest replay history through the same pipeline and simulate the trades
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .backtest import DEFAULT_MAX_HOLD_BARS, aggregate, run_backtest
from .data import build_sources
from .indicators.compute import compute_indicators
from .runner import Runner
from .settings import AppConfig, SymbolConfig, load_config

LOGGER = logging.getLogger(__name__)

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forexpert",
        description="Rule-based VWAP/TrendMeter/VSLRT signal engine.",
    )
    parser.add_argument(
        "-c", "--config", default=None, metavar="PATH", help="path to config.yaml"
    )
    # `forexpert` with no subcommand behaves like `forexpert run`
    parser.set_defaults(once=False, max_passes=None)
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="run the live signal loop")
    run_p.add_argument("--once", action="store_true", help="one pass, then exit")
    run_p.add_argument(
        "--max-passes", type=int, default=None, help="stop after N passes"
    )

    check_p = sub.add_parser("check", help="fetch every symbol and print data health")
    check_p.add_argument("--bars", type=int, default=None, help="bars to fetch")

    sub.add_parser("doctor", help="preflight the environment before going live")

    backtest_p = sub.add_parser(
        "backtest", help="replay history through the strategy and simulate trades"
    )
    backtest_p.add_argument(
        "--symbol", action="append", default=None, metavar="SYMBOL",
        help="only this symbol (repeatable); default is every configured symbol",
    )
    backtest_p.add_argument("--timeframe", default=None, help="only this timeframe")
    backtest_p.add_argument("--bars", type=int, default=1000, help="history to fetch")
    backtest_p.add_argument(
        "--max-hold", type=int, default=DEFAULT_MAX_HOLD_BARS,
        help="close a trade after N bars without SL/TP",
    )
    backtest_p.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def _configure_logging(config: AppConfig) -> None:
    runtime = config.runtime or {}
    level = getattr(logging, str(runtime.get("log_level", "INFO")).upper(), logging.INFO)
    fmt = logging.Formatter(str(runtime.get("log_format", DEFAULT_LOG_FORMAT)))
    root = logging.getLogger()
    root.setLevel(level)

    if not any(getattr(handler, "_forexpert_console", False) for handler in root.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        stream._forexpert_console = True  # type: ignore[attr-defined]
        root.addHandler(stream)

    log_file = runtime.get("log_file")
    if log_file:
        from logging.handlers import RotatingFileHandler

        path = Path(str(log_file)).resolve()
        if str(path) not in {
            getattr(handler, "baseFilename", None) for handler in root.handlers
        }:
            path.parent.mkdir(parents=True, exist_ok=True)
            rotating = RotatingFileHandler(
                path,
                maxBytes=int(runtime.get("log_max_bytes", 5_000_000)),
                backupCount=int(runtime.get("log_backup_count", 5)),
                encoding="utf-8",
            )
            rotating.setFormatter(fmt)
            root.addHandler(rotating)


def _select(config: AppConfig, args: argparse.Namespace) -> list[SymbolConfig]:
    """Symbols a command should act on.

    Without ``--symbol`` the monitored book is used. An explicit ``--symbol``
    searches every configured pair, disabled ones included, so a candidate can
    be backtested before it is enabled.
    """
    if args.symbol:
        wanted = {s.upper() for s in args.symbol}
        selected = [s for s in config.symbols if s.symbol.upper() in wanted]
    else:
        selected = config.enabled_symbols()
    if args.timeframe:
        selected = [s for s in selected if s.timeframe == args.timeframe]
    return selected


# ---------------------------------------------------------------------- commands
def cmd_run(config: AppConfig, args: argparse.Namespace) -> int:
    runner = Runner(config)
    if args.once:
        outcomes = runner.run_once()
    else:
        outcomes = runner.run_forever(max_passes=args.max_passes) or []
    errors = [o for o in outcomes if not o.ok]
    for outcome in outcomes:
        state = "ok" if outcome.ok else f"ERROR {outcome.error}"
        print(
            f"{outcome.symbol:8} {outcome.timeframe:4} bars={outcome.bars_processed:<4} "
            f"signals={len(outcome.signals)} stale={int(outcome.stale)} {state}"
        )
    return 1 if errors else 0


def cmd_check(config: AppConfig, args: argparse.Namespace) -> int:
    """Read-only health check: every symbol, its volume, and the key indicators."""
    sources = build_sources(config.data)
    bars = args.bars or int((config.runtime or {}).get("max_bars_in_memory", 1000))
    rows: list[list[str]] = []
    failures = 0

    for symbol in config.enabled_symbols():
        source = sources.get(symbol.provider)
        try:
            if source is None:
                raise RuntimeError(f"no candle source for provider {symbol.provider!r}")
            frame = source.fetch(symbol.symbol, symbol.timeframe, bars)
            if frame.empty:
                raise RuntimeError("provider returned no candles")
            indicators = compute_indicators(
                frame, symbol.params, touch_line=config.strategy.touch_line
            )
            vwap = float(indicators.ml_vwap.vwap.iloc[-1])
            atr = float(indicators.atr.atr.iloc[-1])
            volume_ok = bool((frame["volume"] > 0).all())
            rows.append([
                symbol.symbol,
                symbol.timeframe,
                symbol.provider,
                str(len(frame)),
                "yes" if volume_ok else "NO",
                str(frame.index[-1]),
                f"{vwap:.5f}",
                f"{atr:.5f}",
                "",
            ])
            failures += 0 if volume_ok else 1
        except Exception as exc:  # noqa: BLE001 - report, then keep going
            failures += 1
            rows.append([
                symbol.symbol, symbol.timeframe, symbol.provider,
                "-", "-", "-", "-", "-", f"{type(exc).__name__}: {exc}",
            ])

    _print_table(
        ["SYMBOL", "TF", "PROVIDER", "BARS", "VOLUME", "LAST BAR", "VWAP", "ATR", "ERROR"],
        rows,
    )
    print(f"\n{len(rows) - failures}/{len(rows)} symbols healthy")
    return 1 if failures else 0


# Provider -> the third-party module it needs at fetch time. `csv` needs none.
_PROVIDER_DEPENDENCIES = {
    "twelve_data": "requests",
    "binance": "requests",
    "dukascopy": "requests",
    "mt5": "MetaTrader5",
}


def _writable(path: Path) -> str | None:
    """Return None when `path` (a file) can be written, else the error text."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = path.parent / f".{path.name}.writecheck"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return str(exc)
    return None


def cmd_doctor(config: AppConfig, args: argparse.Namespace) -> int:
    """Preflight the environment: config, dependencies, paths and notifier.

    Read-only apart from writing a throwaway probe file next to the database and
    journal, so it is safe to run before every deployment. Exit code 1 when any
    check FAILs (warnings alone exit 0).
    """
    rows: list[list[str]] = []

    def check(name: str, problems: list[str], warn: bool = False) -> None:
        if problems:
            rows.append([name, "WARN" if warn else "FAIL", "; ".join(problems)])
        else:
            rows.append([name, "PASS", ""])

    enabled = config.enabled_symbols()
    check(
        "config",
        [] if enabled else ["no enabled symbols (all disabled in config.yaml)"],
    )

    providers = sorted({s.provider for s in enabled})
    provider_problems: list[str] = []
    for provider in providers:
        module = _PROVIDER_DEPENDENCIES.get(provider)
        if module is None:
            continue
        if importlib.util.find_spec(module) is None:
            provider_problems.append(f"{provider} needs '{module}'")
    check("dependencies", provider_problems)

    runtime = config.runtime or {}
    path_problems: list[str] = []
    db_problem = _writable(Path(str(runtime.get("db_path", "data/forexpert.sqlite"))))
    if db_problem:
        path_problems.append(f"db: {db_problem}")
    journal_cfg = config.journal or {}
    if journal_cfg.get("enabled", True):
        journal_problem = _writable(Path(str(journal_cfg.get("path", "data/journal.jsonl"))))
        if journal_problem:
            path_problems.append(f"journal: {journal_problem}")
    check("paths", path_problems)

    csv_symbols = [s for s in enabled if s.provider == "csv"]
    if csv_symbols:
        csv_dir = Path(str((config.data.get("csv") or {}).get("directory", "data/csv")))
        missing = [
            f"{s.symbol}_{s.timeframe}.csv"
            for s in csv_symbols
            if not (csv_dir / f"{s.symbol.upper()}_{s.timeframe}.csv").is_file()
        ]
        check("csv files", [f"missing in {csv_dir}: {', '.join(missing)}"] if missing else [])

    from .notify import build_notifier

    notifier = build_notifier(
        {"telegram": config.notifier.telegram, "discord": config.notifier.discord}
    )
    check(
        "notifier",
        [] if notifier.name != "null" else ["no notifier enabled; alerts are logged only"],
        warn=True,
    )

    _print_table(["CHECK", "STATUS", "DETAIL"], rows)
    failed = sum(1 for row in rows if row[1] == "FAIL")
    warned = sum(1 for row in rows if row[1] == "WARN")
    print(f"\n{len(rows) - failed - warned} passed, {warned} warning(s), {failed} failure(s)")
    return 1 if failed else 0


def cmd_backtest(config: AppConfig, args: argparse.Namespace) -> int:
    symbols = _select(config, args)
    if not symbols:
        print("no symbols matched")
        return 2

    sources = build_sources(config.data)
    results = []
    for symbol in symbols:
        source = sources.get(symbol.provider)
        try:
            if source is None:
                raise RuntimeError(f"no candle source for provider {symbol.provider!r}")
            frame = source.fetch(symbol.symbol, symbol.timeframe, args.bars)
            results.append(
                run_backtest(frame, symbol, config, max_hold_bars=args.max_hold)
            )
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not hide the rest
            LOGGER.error("backtest failed for %s: %s", symbol.key, exc)

    if not results:
        print("nothing was backtested")
        return 1

    stats = [result.stats() for result in results]
    combined = aggregate(results)
    if args.json:
        print(json.dumps({"symbols": stats, "total": combined}, indent=2))
        return 0

    _print_table(
        ["SYMBOL", "TF", "BARS", "SIGNALS", "CLOSED", "W/L", "WIN%", "TOTAL R", "AVG R", "PF", "MAX STRK L"],
        [[
            s["symbol"], s["timeframe"], str(s["bars"]), str(s["signals"]),
            str(s["closed"]), f"{s['wins']}/{s['losses']}",
            f"{s['win_rate'] * 100:.1f}", f"{s['total_r']:+.2f}",
            f"{s['avg_r']:+.2f}",
            "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}",
            str(s["max_consecutive_losses"]),
        ] for s in stats],
    )
    print(
        f"\ntotal: {combined['signals']} signals, {combined['closed']} closed, "
        f"{combined['wins']}W/{combined['losses']}L "
        f"({combined['win_rate'] * 100:.1f}%), {combined['total_r']:+.2f}R, "
        f"expectancy {combined['avg_r']:+.3f}R/trade"
    )
    return 0


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        print("(nothing to show)")
        return
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))
    ]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


COMMANDS = {"run": cmd_run, "check": cmd_check, "doctor": cmd_doctor, "backtest": cmd_backtest}


def resolve_command(args: argparse.Namespace) -> str:
    """The subcommand to run, defaulting to the live loop."""
    return str(args.command or "run")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config)
    _configure_logging(config)
    handler: Any = COMMANDS[resolve_command(args)]
    return int(handler(config, args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
