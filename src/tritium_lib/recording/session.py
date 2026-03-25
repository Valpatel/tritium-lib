# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Session — metadata about a recording file.

Parses the header and footer of a JSON-lines recording to extract
session information without loading all events into memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Session:
    """Metadata about a sensor data recording session.

    Attributes
    ----------
    session_id : str
        Unique identifier for the recording.
    path : str
        File path of the recording.
    start_time : float
        Unix timestamp when recording started.
    end_time : float
        Unix timestamp when recording stopped (0 if still open).
    duration : float
        Duration in seconds.
    event_count : int
        Number of sensor events (excludes header/footer).
    sensor_types : list[str]
        Sorted list of event types present.
    sources : list[str]
        Sorted list of sensor/node sources present.
    metadata : dict
        User-provided metadata from the recording.
    complete : bool
        True if the recording has a proper _session_end footer.
    """

    session_id: str = ""
    path: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    event_count: int = 0
    sensor_types: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    complete: bool = False

    @classmethod
    def from_file(cls, path: str | Path) -> Session:
        """Parse session metadata from a recording file.

        Reads the first line (header) and last line (footer) to extract
        session information. Also counts events and collects sensor types
        by scanning all lines.

        Parameters
        ----------
        path : str or Path
            Path to the .jsonl recording file.

        Returns
        -------
        Session
            Populated session metadata.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is empty or has no valid header.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Recording file not found: {filepath}")

        lines: list[str] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for raw in f:
                stripped = raw.strip()
                if stripped:
                    lines.append(stripped)

        if not lines:
            raise ValueError(f"Recording file is empty: {filepath}")

        # Parse header
        first = json.loads(lines[0])
        session = cls(path=str(filepath))

        if first.get("event_type") == "_session_start":
            header_data = first.get("data", {})
            session.session_id = header_data.get("session_id", "")
            session.start_time = first.get("ts", 0.0)
            session.metadata = header_data.get("metadata", {})

        # Parse footer (if present)
        last = json.loads(lines[-1])
        if last.get("event_type") == "_session_end":
            footer_data = last.get("data", {})
            session.end_time = footer_data.get("end_time", last.get("ts", 0.0))
            session.duration = footer_data.get("duration", 0.0)
            session.complete = True
            # Use footer stats if available
            if "event_count" in footer_data:
                session.event_count = footer_data["event_count"]
            if "sensor_types" in footer_data:
                session.sensor_types = footer_data["sensor_types"]
            if "sources" in footer_data:
                session.sources = footer_data["sources"]
        else:
            # No footer — compute from content
            session.complete = False

        # If footer didn't provide stats, compute by scanning
        if not session.event_count:
            types: set[str] = set()
            srcs: set[str] = set()
            count = 0
            for line_str in lines:
                obj = json.loads(line_str)
                et = obj.get("event_type", "")
                if et.startswith("_"):
                    continue
                count += 1
                types.add(et)
                src = obj.get("source", "")
                if src:
                    srcs.add(src)
            session.event_count = count
            session.sensor_types = sorted(types)
            session.sources = sorted(srcs)

        # Compute duration from timestamps if not set
        if not session.duration and len(lines) >= 2:
            first_ts = json.loads(lines[0]).get("ts", 0.0)
            last_ts = json.loads(lines[-1]).get("ts", 0.0)
            session.duration = max(0.0, last_ts - first_ts)

        # Fill start/end from line timestamps if header was missing
        if not session.start_time:
            session.start_time = json.loads(lines[0]).get("ts", 0.0)
        if not session.end_time and len(lines) >= 2:
            session.end_time = json.loads(lines[-1]).get("ts", 0.0)

        return session

    def to_dict(self) -> dict[str, Any]:
        """Serialize session metadata to a dict."""
        return {
            "session_id": self.session_id,
            "path": self.path,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "event_count": self.event_count,
            "sensor_types": self.sensor_types,
            "sources": self.sources,
            "metadata": self.metadata,
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Session:
        """Deserialize from a dict."""
        return cls(
            session_id=d.get("session_id", ""),
            path=d.get("path", ""),
            start_time=d.get("start_time", 0.0),
            end_time=d.get("end_time", 0.0),
            duration=d.get("duration", 0.0),
            event_count=d.get("event_count", 0),
            sensor_types=d.get("sensor_types", []),
            sources=d.get("sources", []),
            metadata=d.get("metadata", {}),
            complete=d.get("complete", False),
        )

    def summary(self) -> str:
        """Human-readable one-line summary."""
        status = "complete" if self.complete else "incomplete"
        return (
            f"Session {self.session_id[:8]}... "
            f"[{status}] "
            f"{self.event_count} events, "
            f"{self.duration:.1f}s, "
            f"sensors: {', '.join(self.sensor_types) or 'none'}"
        )
