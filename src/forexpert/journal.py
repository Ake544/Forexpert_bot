"""Append-only JSONL journal of every decision the system makes.

Signals *and* rejections are recorded: a trader debugging "why no alert?" needs
the rejects as much as the hits. One JSON object per line, so the file is
streamable with `tail -f`, greppable, and never needs rewriting.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Iterator


class Journal:
    """Append-only JSONL writer/reader. Thread-safe for in-process producers."""

    def __init__(self, path: Path | str, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._lock = threading.Lock()

    def _ensure_parent(self) -> None:
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Append one event. Returns the written record so callers can log it."""
        entry = {"ts": time.time(), "event": event}
        if payload:
            entry.update(payload)
        if not self.enabled:
            return entry
        line = json.dumps(entry, sort_keys=True, default=str)
        with self._lock:
            self._ensure_parent()
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
        return entry

    def signal(self, signal_payload: dict[str, Any]) -> dict[str, Any]:
        return self.record("signal", {"signal": signal_payload})

    def rejection(self, symbol: str, timeframe: str, reason: str,
                  bar_time: str | None = None, direction: str | None = None) -> dict[str, Any]:
        return self.record("rejection", {
            "symbol": symbol, "timeframe": timeframe, "direction": direction,
            "reason": reason, "bar_time": bar_time,
        })

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn last line after a crash must not break the run

    def events(self, event: str | None = None) -> list[dict[str, Any]]:
        return [e for e in self if event is None or e.get("event") == event]

    def signal_ids(self) -> set[str]:
        """Every signal id already journaled, for restart-safe dedupe."""
        return {
            str(e["signal"]["signal_id"])
            for e in self.events("signal")
            if isinstance(e.get("signal"), dict) and e["signal"].get("signal_id")
        }
