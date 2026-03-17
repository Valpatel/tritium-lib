# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Debug data streams for all game subsystems.

Every game module can register debug streams that emit structured data
each tick. Useful for: overlay rendering, logging, conductor dashboard,
unit tests, performance profiling.
"""
from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class DebugFrame:
    """One frame of debug data from a subsystem."""
    system: str          # "steering", "physics", "city_sim", "combat_ai", etc.
    tick: int
    timestamp: float
    entries: list[dict] = field(default_factory=list)


class DebugStream:
    """Collects debug data from a game subsystem."""

    def __init__(self, system_name: str, max_history: int = 60):
        self.system = system_name
        self.enabled = False  # Off by default, zero overhead when disabled
        self.max_history = max_history
        self._frames: list[DebugFrame] = []
        self._tick = 0
        self._listeners: list[callable] = []

    def begin_frame(self) -> DebugFrame | None:
        """Start a new debug frame. Returns frame to add entries to."""
        if not self.enabled:
            return None
        self._tick += 1
        frame = DebugFrame(system=self.system, tick=self._tick, timestamp=time.time())
        return frame

    def end_frame(self, frame: DebugFrame | None) -> None:
        """Finish a frame, store it, notify listeners."""
        if frame is None:
            return
        self._frames.append(frame)
        if len(self._frames) > self.max_history:
            self._frames.pop(0)
        for listener in self._listeners:
            listener(frame)

    def on_frame(self, callback: callable) -> None:
        """Register a listener for new frames."""
        self._listeners.append(callback)

    @property
    def latest(self) -> DebugFrame | None:
        return self._frames[-1] if self._frames else None

    @property
    def history(self) -> list[DebugFrame]:
        return list(self._frames)


class DebugOverlay:
    """Collects debug streams from multiple systems into one view."""

    def __init__(self):
        self.streams: dict[str, DebugStream] = {}

    def register(self, stream: DebugStream) -> None:
        self.streams[stream.system] = stream

    def enable_all(self) -> None:
        for s in self.streams.values():
            s.enabled = True

    def disable_all(self) -> None:
        for s in self.streams.values():
            s.enabled = False

    def get_snapshot(self) -> dict[str, DebugFrame]:
        """Get latest frame from all streams -- for dashboard/overlay."""
        return {name: stream.latest for name, stream in self.streams.items() if stream.latest}

    def to_dict(self) -> dict:
        """Full export for WebSocket/API."""
        result = {}
        for name, stream in self.streams.items():
            if stream.latest:
                result[name] = {
                    "tick": stream.latest.tick,
                    "entries": stream.latest.entries,
                    "entry_count": len(stream.latest.entries),
                }
        return result
