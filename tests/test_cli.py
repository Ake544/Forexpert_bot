"""CLI tests: argument handling, output formatting, and command dispatch.

Nothing here touches the network or the SQLite store — the sources and the
runner are replaced with fakes, so `forexpert check` can be tested offline.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from forexpert import cli
from forexpert.runner import SymbolOutcome

from test_backtest import synthetic_candles


class FakeSource:
    def __init__(self, frame):  # noqa: ANN001 - the frame is a test fixture
        self._frame = frame

    def fetch(self, symbol, timeframe, limit=None, now=None):  # noqa: ANN001
        return self._frame.iloc[: int(limit or len(self._frame))]


@pytest.fixture
def offline_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = synthetic_candles()
    monkeypatch.setattr(
        cli, "build_sources", lambda data: {"mt5": FakeSource(frame), "binance": FakeSource(frame)}
    )


def test_no_subcommand_means_run() -> None:
    args = cli.build_parser().parse_args([])
    assert args.command is None
    assert cli.resolve_command(args) == "run"
    assert args.once is False
    assert args.max_passes is None


def test_backtest_flags_parse() -> None:
    args = cli.build_parser().parse_args(
        ["backtest", "--symbol", "EURCHF", "--symbol", "AUDJPY",
         "--timeframe", "4h", "--bars", "500", "--max-hold", "12", "--json"]
    )
    assert args.command == "backtest"
    assert args.symbol == ["EURCHF", "AUDJPY"]
    assert args.timeframe == "4h"
    assert args.bars == 500
    assert args.max_hold == 12
    assert args.json is True


def test_check_prints_a_health_table(capsys: pytest.CaptureFixture, offline_sources) -> None:
    code = cli.main(["check", "--bars", "300"])
    out = capsys.readouterr().out

    assert code == 0
    assert "SYMBOL" in out and "VOLUME" in out and "VWAP" in out
    assert "symbols healthy" in out
    assert "  yes " in out  # volume present on every row


def test_check_fails_when_a_source_is_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "build_sources", lambda data: {})
    code = cli.main(["check", "--bars", "300"])
    out = capsys.readouterr().out

    assert code == 1
    assert "no candle source" in out


def test_backtest_json_is_machine_readable(capsys, offline_sources) -> None:
    code = cli.main(["backtest", "--symbol", "EURCHF", "--bars", "300", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert len(payload["symbols"]) == 1
    assert payload["symbols"][0]["symbol"] == "EURCHF"
    assert payload["total"]["symbols"] == 1
    assert {"signals", "closed", "win_rate", "total_r"} <= payload["total"].keys()


def test_explicit_symbol_selects_a_disabled_pair() -> None:
    """An explicit --symbol reaches disabled pairs; the default set does not."""
    config = cli.load_config()
    disabled = next(s for s in config.symbols if not s.enabled)

    chosen = cli._select(config, SimpleNamespace(symbol=[disabled.symbol], timeframe=None))
    assert any(s.symbol == disabled.symbol and not s.enabled for s in chosen)
    assert disabled not in config.enabled_symbols()


def test_backtest_with_no_matching_symbol_exits_2(capsys) -> None:
    code = cli.main(["backtest", "--symbol", "NOTAPAIR"])
    assert code == 2
    assert "no symbols matched" in capsys.readouterr().out


def test_run_once_reports_outcomes_and_exit_codes(monkeypatch, capsys) -> None:
    class FakeRunner:
        fail = False

        def __init__(self, config):  # noqa: ANN001
            self.config = config

        def run_once(self):
            outcome = SymbolOutcome("EURCHF", "4h", bars_processed=1)
            if self.fail:
                outcome.error = "provider exploded"
            return [outcome]

    monkeypatch.setattr(cli, "Runner", FakeRunner)

    assert cli.main(["run", "--once"]) == 0
    assert "EURCHF" in capsys.readouterr().out

    FakeRunner.fail = True
    assert cli.main(["run", "--once"]) == 1
    assert "ERROR provider exploded" in capsys.readouterr().out


def test_print_table_handles_empty_input(capsys) -> None:
    cli._print_table(["A", "B"], [])
    assert "nothing to show" in capsys.readouterr().out
