"""Shared fixtures: deterministic closed-candle inputs for the state machine."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from forexpert.strategy.state import BarInput

BASE_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)
LEVEL = 100.0


def make_bar(index: int, **overrides: Any) -> BarInput:
    """Build a closed candle at a deterministic time.

    Defaults describe a bar sitting above a 100.0 level with every indicator
    ready, so each test only has to state the part it is actually testing.
    """
    values: dict[str, Any] = {
        "time": BASE_TIME + timedelta(hours=4 * index),
        "open": 101.0,
        "high": 102.0,
        "low": 100.5,
        "close": 101.5,
        "level": LEVEL,
        "vwap_bull": True,
        "trend_bull": True,
        "vslrt_bull": True,
        "atr": 1.0,
        "kro_bottom_touch": None,
        "kro_top_touch": None,
        "vslrt_long_line": 0.42,
    }
    values.update(overrides)
    return BarInput(**values)


@pytest.fixture
def defaults() -> dict[str, Any]:
    return {"level": LEVEL, "atr": 1.0, "vwap_bull": True, "trend_bull": True, "vslrt_bull": True}