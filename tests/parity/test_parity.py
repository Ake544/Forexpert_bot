"""Indicator parity against TradingView reference values.

This is the test that proves the port is faithful. It is skipped until you paste
real values into `reference_values.yaml` (see the README in this folder).

Why this matters: the Pine ports were written from source, but two things can
still silently diverge from TradingView at runtime:
  1. ta.vwap is session anchored (Pine's default ``timeframe.change("1D")``), so
     on a daily chart it equals close and on intraday charts it resets each UTC
     day. An anchor mismatch produces different VWAP values with no code bug.
  2. ta.stdev is population stdev and ta.ema/ta.rma are SMA-seeded; both are
     easy to get wrong and produce plausible-looking but wrong numbers.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from forexpert.indicators.compute import compute_indicators
from forexpert.settings import IndicatorParams, MlVwapParams

PARITY_DIR = Path(__file__).parent
REFERENCE_FILE = PARITY_DIR / "reference_values.yaml"


def load_cases() -> list[dict]:
    if not REFERENCE_FILE.is_file():
        return []
    with REFERENCE_FILE.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    return list(document.get("cases") or [])


CASES = load_cases()
NEEDS_CASES = pytest.mark.skipif(
    not CASES, reason="No TradingView reference values in tests/parity/reference_values.yaml yet"
)

TIME_COLUMNS = ("time", "date", "datetime")
VOLUME_COLUMNS = ("volume", "tick_volume", "vol")
OHLC = ("open", "high", "low", "close")


def load_csv(path: Path) -> pd.DataFrame:
    """Load candles from a TradingView export or one of our own CSV dumps.

    TradingView's *Export chart data* writes epoch seconds under ``time`` and
    a capitalised ``Volume``; our dumps are ISO8601 + ``volume``. Both must land
    on the same validated frame, because parity is only meaningful against
    TradingView's own bars - they define the VWAP anchor.
    """
    raw = pd.read_csv(path)
    raw.columns = [str(column).strip() for column in raw.columns]
    found = {column.lower(): column for column in raw.columns}

    time_column = next((found[name] for name in TIME_COLUMNS if name in found), None)
    if time_column is None:
        raise AssertionError(
            f"{path.name} has no time column (columns: {list(raw.columns)}); "
            "expected a TradingView chart-data export or our CSV format"
        )
    stamp = raw[time_column]
    if pd.api.types.is_numeric_dtype(stamp):
        # TradingView exports epoch seconds; tolerate milliseconds too.
        unit = "ms" if stamp.abs().max() > 10**11 else "s"
        index = pd.to_datetime(stamp, unit=unit, utc=True)
    else:
        try:
            index = pd.to_datetime(stamp, utc=True)
        except ValueError:
            index = pd.to_datetime(stamp, utc=True, format="mixed")

    volume_column = next((found[name] for name in VOLUME_COLUMNS if name in found), None)
    if volume_column is None:
        raise AssertionError(
            f"{path.name} has no volume column; the VWAP is volume weighted, so a "
            "chart export without volume cannot prove parity"
        )

    # .to_numpy() on purpose: a dict of Series would align on their RangeIndex
    # against the new DatetimeIndex and silently produce an all-NaN frame.
    frame = pd.DataFrame(
        {
            name: pd.to_numeric(raw[found[name]], errors="coerce").to_numpy()
            for name in (*OHLC, "volume")
            if name in found
        },
        index=index,
    )
    if "volume" not in frame:
        frame["volume"] = pd.to_numeric(raw[volume_column], errors="coerce").to_numpy()
    frame.index.name = "time"
    frame = frame[~frame.index.isna()].sort_index()
    if frame.empty:
        raise AssertionError(f"{path.name} contains no readable bars")
    missing = [column for column in OHLC if column not in frame.columns]
    if missing:
        raise AssertionError(f"{path.name} is missing candle columns: {missing}")
    if not (frame["volume"] > 0).any():
        raise AssertionError(
            f"{path.name} reports no volume at all - VWAP would be undefined. "
            "Pick a feed that reports tick volume (OANDA/FXCM/FOREX.com)."
        )
    return frame[list(OHLC) + ["volume"]]


def as_bool(value: object) -> bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        return None
    return bool(value)


@NEEDS_CASES
@pytest.mark.parametrize("case", CASES, ids=lambda c: str(c.get("bar_time")))
def test_matches_tradingview(case: dict) -> None:
    csv_path = PARITY_DIR / case["csv_file"]
    assert csv_path.is_file(), f"Missing candle file for parity case: {csv_path}"
    frame = load_csv(csv_path)

    bar_time = pd.Timestamp(case["bar_time"])
    assert bar_time in frame.index, f"{bar_time} is not present in {csv_path.name}"

    base = IndicatorParams()
    params = replace(base, ml_vwap=replace(base.ml_vwap, window=int(case.get("ml_window", 300))))
    indicators = compute_indicators(frame, params, touch_line=case.get("touch_line", "vwap"))
    levels = indicators.level_frame()
    row = levels.loc[bar_time]

    tolerance = float(case.get("tolerance", 1e-6))

    checks = {
        "vwap": (row["level"], case.get("vwap")),
        "trend_fast": (indicators.trend_meter.fast.loc[bar_time], case.get("trend_fast")),
        "trend_slow": (indicators.trend_meter.slow.loc[bar_time], case.get("trend_slow")),
        "vslrt_long_line": (row["vslrt_long_line"], case.get("vslrt_long_line")),
        "atr": (row["atr"], case.get("atr")),
    }
    for name, (actual, expected) in checks.items():
        if expected is None:
            continue
        assert actual == pytest.approx(float(expected), rel=tolerance, abs=0.0), (
            f"{name} mismatch at {bar_time}: port={actual!r} tradingview={expected!r}"
        )

    # The Trend Meter plots a colour/thickness, never its MA values, so states are
    # the only comparable output for it - and the only output the strategy reads.
    states = {
        "vwap_bull": (row["vwap_bull"], case.get("vwap_bull")),
        "trend_bull": (row["trend_bull"], case.get("trend_bull")),
        "vslrt_bull": (row["vslrt_bull"], case.get("vslrt_bull")),
    }
    for name, (actual, expected) in states.items():
        if expected is None:
            continue
        assert as_bool(actual) is not None, f"{name} is still warming up at {bar_time}"
        assert as_bool(actual) is bool(expected), (
            f"{name} mismatch at {bar_time}: port={actual!r} tradingview={expected!r}"
        )

    if "ml_output" in case and case["ml_output"] is not None:
        assert indicators.ml_vwap.ml_output.loc[bar_time] == pytest.approx(
            float(case["ml_output"]), rel=tolerance, abs=0.0
        )


@NEEDS_CASES
def test_ml_window_in_case_matches_profile_default() -> None:
    """Guards against pasting values collected with a different ML Window."""
    default_window = MlVwapParams().window
    for case in CASES:
        assert int(case.get("ml_window", default_window)) == default_window, (
            f"Parity case {case.get('bar_time')} was collected with ML window "
            f"{case.get('ml_window')} but the config default is {default_window}. "
            "Change one of them, not both."
        )


# ------------------------------------------------------------------ csv loading
def write_tv_export(path: Path, rows: int = 5) -> Path:
    """A raw TradingView *Export chart data* file: epoch seconds, `Volume`."""
    start = 1709251200  # 2024-03-01T00:00:00Z
    lines = ["time,open,high,low,close,Volume"]
    for offset in range(rows):
        base = 1.08 + offset * 0.001
        lines.append(
            f"{start + offset * 86400},{base:.5f},{base + 0.002:.5f},"
            f"{base - 0.002:.5f},{base + 0.001:.5f},{1000 + offset}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_tradingview_export_loads_as_utc_ohlcv(tmp_path: Path) -> None:
    frame = load_csv(write_tv_export(tmp_path / "tv.csv"))

    assert list(frame.columns) == [*OHLC, "volume"]
    assert frame.index.name == "time"
    assert frame.index.tz is not None
    assert frame.index[0] == pd.Timestamp("2024-03-01", tz="UTC")
    assert frame["volume"].iloc[-1] == 1004


def test_our_own_iso_csv_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "ours.csv"
    path.write_text(
        "time,open,high,low,close,volume\n"
        "2024-03-02T00:00:00+00:00,1.08,1.082,1.078,1.081,500\n"
        "2024-03-01T00:00:00+00:00,1.07,1.072,1.068,1.071,400\n",  # out of order
        encoding="utf-8",
    )
    frame = load_csv(path)

    assert list(frame.index) == [
        pd.Timestamp("2024-03-01", tz="UTC"),
        pd.Timestamp("2024-03-02", tz="UTC"),
    ]


def test_export_without_volume_is_rejected_with_a_useful_reason(tmp_path: Path) -> None:
    path = tmp_path / "nolume.csv"
    path.write_text(
        "time,open,high,low,close\n1709251200,1.08,1.082,1.078,1.081\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="volume"):
        load_csv(path)


def test_all_zero_volume_is_rejected_because_vwap_needs_it(tmp_path: Path) -> None:
    path = tmp_path / "zero.csv"
    path.write_text(
        "time,open,high,low,close,Volume\n1709251200,1.08,1.082,1.078,1.081,0\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="no volume"):
        load_csv(path)