# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Player — replays recorded sensor events with speed control.

Reads a JSON-lines recording file and yields events with timing that
respects the original inter-event delays, scaled by a speed factor.

Usage
-----
    player = Player("/tmp/patrol.jsonl")
    player.speed = 5.0  # 5x faster than real time
    for event in player.replay():
        handle(event)

    # Non-blocking iteration (no sleep, just yields)
    for event in player.events():
        print(event.event_type, event.ts)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional


@dataclass
class ReplayEvent:
    """A single event from a recording file."""

    ts: float
    event_type: str
    source: str
    data: dict[str, Any]

    @classmethod
    def from_line(cls, line: str) -> ReplayEvent:
        """Parse a JSON-lines entry into a ReplayEvent."""
        obj = json.loads(line)
        return cls(
            ts=obj["ts"],
            event_type=obj["event_type"],
            source=obj.get("source", ""),
            data=obj.get("data", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a dict."""
        return {
            "ts": self.ts,
            "event_type": self.event_type,
            "source": self.source,
            "data": self.data,
        }


class Player:
    """Replays recorded sensor data from a JSON-lines file.

    Parameters
    ----------
    path : str or Path
        Path to the .jsonl recording file.
    speed : float
        Playback speed multiplier. 1.0 = real time, 2.0 = 2x, 0.5 = half.
    skip_control : bool
        If True, skip _session_start and _session_end events during replay.
    """

    def __init__(
        self,
        path: str | Path,
        speed: float = 1.0,
        skip_control: bool = True,
    ):
        self._path = Path(path)
        self.speed = speed
        self.skip_control = skip_control
        self._events: list[ReplayEvent] = []
        self._loaded = False

    @property
    def path(self) -> Path:
        """Path to the recording file."""
        return self._path

    @property
    def event_count(self) -> int:
        """Total number of events (loaded). Excludes control events if skip_control."""
        self._ensure_loaded()
        if self.skip_control:
            return sum(
                1 for e in self._events if not e.event_type.startswith("_")
            )
        return len(self._events)

    @property
    def duration(self) -> float:
        """Duration of the recording in seconds (wall-clock, not replay)."""
        self._ensure_loaded()
        if len(self._events) < 2:
            return 0.0
        return self._events[-1].ts - self._events[0].ts

    @property
    def start_time(self) -> float:
        """Timestamp of the first event."""
        self._ensure_loaded()
        return self._events[0].ts if self._events else 0.0

    @property
    def end_time(self) -> float:
        """Timestamp of the last event."""
        self._ensure_loaded()
        return self._events[-1].ts if self._events else 0.0

    def load(self) -> int:
        """Load the recording file into memory. Returns event count.

        Raises
        ------
        FileNotFoundError
            If the recording file does not exist.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"Recording file not found: {self._path}")

        self._events = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._events.append(ReplayEvent.from_line(line))

        self._loaded = True
        return len(self._events)

    def _ensure_loaded(self) -> None:
        """Load the file if not already loaded."""
        if not self._loaded:
            self.load()

    def events(self) -> Generator[ReplayEvent, None, None]:
        """Yield all events without timing delays.

        Useful for analysis, filtering, or fast-forward processing.
        Skips control events if skip_control is True.
        """
        self._ensure_loaded()
        for event in self._events:
            if self.skip_control and event.event_type.startswith("_"):
                continue
            yield event

    def replay(self) -> Generator[ReplayEvent, None, None]:
        """Yield events with real-time delays scaled by speed.

        At speed=1.0, events are yielded at the same pace they were recorded.
        At speed=10.0, a 10-second gap becomes 1 second.

        Skips control events if skip_control is True.
        """
        self._ensure_loaded()
        if not self._events:
            return

        effective_speed = max(self.speed, 0.001)  # prevent division by zero

        prev_ts: Optional[float] = None
        for event in self._events:
            if self.skip_control and event.event_type.startswith("_"):
                continue

            if prev_ts is not None:
                delta = event.ts - prev_ts
                if delta > 0:
                    time.sleep(delta / effective_speed)

            prev_ts = event.ts
            yield event

    def slice(
        self,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        event_types: Optional[set[str]] = None,
        sources: Optional[set[str]] = None,
    ) -> list[ReplayEvent]:
        """Return a filtered subset of events.

        Parameters
        ----------
        start_ts : float, optional
            Only events at or after this timestamp.
        end_ts : float, optional
            Only events at or before this timestamp.
        event_types : set[str], optional
            Only events of these types.
        sources : set[str], optional
            Only events from these sources.

        Returns
        -------
        list[ReplayEvent]
            Matching events in chronological order.
        """
        self._ensure_loaded()
        result = []
        for event in self._events:
            if self.skip_control and event.event_type.startswith("_"):
                continue
            if start_ts is not None and event.ts < start_ts:
                continue
            if end_ts is not None and event.ts > end_ts:
                continue
            if event_types is not None and event.event_type not in event_types:
                continue
            if sources is not None and event.source not in sources:
                continue
            result.append(event)
        return result

    def sensor_types(self) -> set[str]:
        """Return the set of all event types in the recording."""
        self._ensure_loaded()
        types = set()
        for event in self._events:
            if not event.event_type.startswith("_"):
                types.add(event.event_type)
        return types

    def sources(self) -> set[str]:
        """Return the set of all sources in the recording."""
        self._ensure_loaded()
        srcs = set()
        for event in self._events:
            if event.source and not event.event_type.startswith("_"):
                srcs.add(event.source)
        return srcs
