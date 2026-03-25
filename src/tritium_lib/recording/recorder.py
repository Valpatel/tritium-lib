# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Recorder — writes sensor events to a JSON-lines file.

Each line is a self-contained JSON object with:
    {
        "ts": 1711324800.123,        # Unix timestamp (float)
        "event_type": "ble_sighting", # Event category
        "source": "node_alpha",       # Sensor / node that produced it
        "data": { ... }               # Arbitrary payload
    }

The first line of every recording is a header with event_type="_session_start",
and the last line (written on stop) is "_session_end" with summary stats.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# Well-known event types for sensor data
SENSOR_EVENT_TYPES = frozenset({
    "ble_sighting",
    "wifi_probe",
    "camera_detection",
    "acoustic_event",
    "mesh_node",
    "rf_signal",
    "espnow_packet",
    # Pipeline outputs
    "fusion_result",
    "alert",
    "zone_event",
    "target_update",
    "correlation",
    "classification",
    # Control
    "_session_start",
    "_session_end",
})


@dataclass
class _RecordingState:
    """Mutable recording state, protected by lock."""
    started: bool = False
    session_id: str = ""
    start_time: float = 0.0
    event_count: int = 0
    sensor_types: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)


class Recorder:
    """Records sensor events to a JSON-lines (.jsonl) file.

    Thread-safe: multiple threads can call record() concurrently.

    Parameters
    ----------
    path : str or Path
        Output file path. Will be created (or truncated) on start().
    metadata : dict, optional
        Extra metadata to include in the session header.
    """

    def __init__(self, path: str | Path, metadata: Optional[dict[str, Any]] = None):
        self._path = Path(path)
        self._metadata = metadata or {}
        self._state = _RecordingState()
        self._lock = threading.Lock()
        self._file: Optional[Any] = None

    @property
    def path(self) -> Path:
        """Path to the recording file."""
        return self._path

    @property
    def is_recording(self) -> bool:
        """True if recording is active."""
        return self._state.started

    @property
    def event_count(self) -> int:
        """Number of events recorded so far (excludes header/footer)."""
        return self._state.event_count

    @property
    def session_id(self) -> str:
        """Unique ID for this recording session."""
        return self._state.session_id

    def start(self) -> str:
        """Begin recording. Returns the session ID.

        Creates or truncates the output file and writes the session header.

        Raises
        ------
        RuntimeError
            If recording is already active.
        """
        with self._lock:
            if self._state.started:
                raise RuntimeError("Recording already active")

            self._state = _RecordingState()
            self._state.session_id = uuid.uuid4().hex
            self._state.start_time = time.time()
            self._state.started = True

            # Ensure parent directory exists
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._path, "w", encoding="utf-8")

            header = {
                "ts": self._state.start_time,
                "event_type": "_session_start",
                "source": "",
                "data": {
                    "session_id": self._state.session_id,
                    "metadata": self._metadata,
                },
            }
            self._file.write(json.dumps(header, separators=(",", ":")) + "\n")
            self._file.flush()

            return self._state.session_id

    def record(
        self,
        event_type: str,
        source: str = "",
        data: Optional[dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a single sensor event.

        Parameters
        ----------
        event_type : str
            Category of the event (e.g., "ble_sighting", "camera_detection").
        source : str
            Sensor or node that produced the event.
        data : dict, optional
            Arbitrary payload data.
        timestamp : float, optional
            Override the timestamp. Defaults to time.time().

        Raises
        ------
        RuntimeError
            If recording is not active.
        """
        ts = timestamp if timestamp is not None else time.time()
        line = {
            "ts": ts,
            "event_type": event_type,
            "source": source,
            "data": data or {},
        }

        with self._lock:
            if not self._state.started:
                raise RuntimeError("Recording not active — call start() first")

            self._state.event_count += 1
            self._state.sensor_types.add(event_type)
            if source:
                self._state.sources.add(source)

            self._file.write(json.dumps(line, separators=(",", ":")) + "\n")
            self._file.flush()

    def stop(self) -> dict[str, Any]:
        """Stop recording. Returns session summary dict.

        Writes a _session_end footer and closes the file.

        Returns
        -------
        dict
            Session summary with session_id, duration, event_count, etc.

        Raises
        ------
        RuntimeError
            If recording is not active.
        """
        with self._lock:
            if not self._state.started:
                raise RuntimeError("Recording not active")

            end_time = time.time()
            duration = end_time - self._state.start_time

            summary = {
                "session_id": self._state.session_id,
                "start_time": self._state.start_time,
                "end_time": end_time,
                "duration": duration,
                "event_count": self._state.event_count,
                "sensor_types": sorted(self._state.sensor_types),
                "sources": sorted(self._state.sources),
                "metadata": self._metadata,
            }

            footer = {
                "ts": end_time,
                "event_type": "_session_end",
                "source": "",
                "data": summary,
            }
            self._file.write(json.dumps(footer, separators=(",", ":")) + "\n")
            self._file.flush()
            self._file.close()
            self._file = None

            self._state.started = False

            return summary

    def __enter__(self) -> Recorder:
        """Context manager: start recording on enter."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager: stop recording on exit."""
        if self._state.started:
            self.stop()

    def __del__(self) -> None:
        """Ensure file is closed on garbage collection."""
        if self._file is not None and not self._file.closed:
            try:
                if self._state.started:
                    self.stop()
                else:
                    self._file.close()
            except Exception:
                pass
