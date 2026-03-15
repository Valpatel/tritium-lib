# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SQLite-backed tactical event store for historical analysis and audit.

Persists ALL tactical events (target sightings, alerts, detections,
commands, state changes) with full metadata. Designed for:
  - Historical replay and analysis
  - Post-action review
  - Pattern detection over time
  - Audit trail for all tactical decisions

Usage
-----
    store = EventStore("/path/to/events.db")
    store.record("target_detected", severity="info", source="ble_scanner",
                 target_id="ble_AA:BB:CC:DD:EE:FF",
                 data={"rssi": -45, "manufacturer": "Apple"})
    events = store.query_time_range(start=time.time() - 3600)
    events = store.query_by_type("target_detected", limit=50)
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .base import BaseStore

# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_SCHEMA_EVENTS = """
CREATE TABLE IF NOT EXISTS tactical_events (
    event_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'info',
    source TEXT NOT NULL DEFAULT '',
    target_id TEXT NOT NULL DEFAULT '',
    operator TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    data TEXT NOT NULL DEFAULT '{}',
    position_lat REAL,
    position_lng REAL,
    site_id TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON tactical_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON tactical_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_severity ON tactical_events(severity);
CREATE INDEX IF NOT EXISTS idx_events_source ON tactical_events(source);
CREATE INDEX IF NOT EXISTS idx_events_target ON tactical_events(target_id);
CREATE INDEX IF NOT EXISTS idx_events_site ON tactical_events(site_id);
"""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class TacticalEvent:
    """A single tactical event record."""

    event_id: str = ""
    timestamp: float = 0.0
    event_type: str = ""          # target_detected, alert_raised, command_sent, state_change, etc.
    severity: str = "info"        # debug, info, warning, error, critical
    source: str = ""              # ble_scanner, yolo, meshtastic, operator, automation, etc.
    target_id: str = ""           # target this event relates to (if any)
    operator: str = ""            # operator/user who triggered (if any)
    summary: str = ""             # human-readable description
    data: dict = field(default_factory=dict)  # arbitrary structured payload
    position_lat: Optional[float] = None
    position_lng: Optional[float] = None
    site_id: str = ""

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "severity": self.severity,
            "source": self.source,
            "target_id": self.target_id,
            "operator": self.operator,
            "summary": self.summary,
            "data": self.data,
            "position_lat": self.position_lat,
            "position_lng": self.position_lng,
            "site_id": self.site_id,
        }


# ---------------------------------------------------------------------------
# Severity levels for filtering
# ---------------------------------------------------------------------------

SEVERITY_LEVELS = ("debug", "info", "warning", "error", "critical")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class EventStore(BaseStore):
    """SQLite-backed persistent tactical event store.

    Thread-safe. Supports time-range queries, type filtering,
    severity filtering, and target-based lookups.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Use ":memory:" for testing.
    max_events:
        Maximum events to retain. Oldest are pruned during cleanup.
    """

    _SCHEMAS = (_SCHEMA_EVENTS,)

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        max_events: int = 500_000,
    ) -> None:
        self._max_events = max_events
        super().__init__(db_path)

    # ------------------------------------------------------------------
    # Record events
    # ------------------------------------------------------------------

    def record(
        self,
        event_type: str,
        *,
        severity: str = "info",
        source: str = "",
        target_id: str = "",
        operator: str = "",
        summary: str = "",
        data: Optional[dict] = None,
        position_lat: Optional[float] = None,
        position_lng: Optional[float] = None,
        site_id: str = "",
        timestamp: Optional[float] = None,
        event_id: Optional[str] = None,
    ) -> str:
        """Record a tactical event.

        Returns the generated event_id.
        """
        eid = event_id or str(uuid.uuid4())
        ts = timestamp if timestamp is not None else time.time()
        data_json = json.dumps(data or {})

        with self._lock:
            self._execute(
                """INSERT INTO tactical_events
                   (event_id, timestamp, event_type, severity, source,
                    target_id, operator, summary, data,
                    position_lat, position_lng, site_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    eid, ts, event_type, severity, source,
                    target_id, operator, summary, data_json,
                    position_lat, position_lng, site_id,
                ),
            )
            self._commit()
        return eid

    def record_batch(self, events: list[TacticalEvent]) -> int:
        """Record multiple events in a single transaction.

        Returns the number of events inserted.
        """
        if not events:
            return 0

        rows = []
        for ev in events:
            eid = ev.event_id or str(uuid.uuid4())
            ts = ev.timestamp if ev.timestamp else time.time()
            rows.append((
                eid, ts, ev.event_type, ev.severity, ev.source,
                ev.target_id, ev.operator, ev.summary,
                json.dumps(ev.data or {}),
                ev.position_lat, ev.position_lng, ev.site_id,
            ))

        with self._lock:
            self._executemany(
                """INSERT INTO tactical_events
                   (event_id, timestamp, event_type, severity, source,
                    target_id, operator, summary, data,
                    position_lat, position_lng, site_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self._commit()
        return len(rows)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_event(self, event_id: str) -> Optional[TacticalEvent]:
        """Get a specific event by ID."""
        with self._lock:
            row = self._fetchone(
                "SELECT * FROM tactical_events WHERE event_id = ?",
                (event_id,),
            )
        if row is None:
            return None
        return self._row_to_event(row)

    def query_time_range(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[TacticalEvent]:
        """Query events within a time range.

        Returns events sorted by timestamp descending (most recent first).
        """
        conditions: list[str] = []
        params: list = []

        if start is not None:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            conditions.append("timestamp <= ?")
            params.append(end)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        sql = f"""SELECT * FROM tactical_events {where}
                  ORDER BY timestamp DESC LIMIT ? OFFSET ?"""
        params.extend([limit, offset])

        with self._lock:
            rows = self._fetchall(sql, tuple(params))
        return [self._row_to_event(r) for r in rows]

    def query_by_type(
        self,
        event_type: str,
        *,
        start: Optional[float] = None,
        end: Optional[float] = None,
        limit: int = 100,
    ) -> list[TacticalEvent]:
        """Query events of a specific type."""
        conditions = ["event_type = ?"]
        params: list = [event_type]

        if start is not None:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            conditions.append("timestamp <= ?")
            params.append(end)

        where = "WHERE " + " AND ".join(conditions)
        sql = f"""SELECT * FROM tactical_events {where}
                  ORDER BY timestamp DESC LIMIT ?"""
        params.append(limit)

        with self._lock:
            rows = self._fetchall(sql, tuple(params))
        return [self._row_to_event(r) for r in rows]

    def query_by_severity(
        self,
        min_severity: str = "info",
        *,
        start: Optional[float] = None,
        end: Optional[float] = None,
        limit: int = 100,
    ) -> list[TacticalEvent]:
        """Query events at or above a severity level.

        Severity ordering: debug < info < warning < error < critical.
        """
        idx = 0
        try:
            idx = SEVERITY_LEVELS.index(min_severity)
        except ValueError:
            pass

        allowed = SEVERITY_LEVELS[idx:]
        placeholders = ",".join("?" * len(allowed))
        conditions = [f"severity IN ({placeholders})"]
        params: list = list(allowed)

        if start is not None:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            conditions.append("timestamp <= ?")
            params.append(end)

        where = "WHERE " + " AND ".join(conditions)
        sql = f"""SELECT * FROM tactical_events {where}
                  ORDER BY timestamp DESC LIMIT ?"""
        params.append(limit)

        with self._lock:
            rows = self._fetchall(sql, tuple(params))
        return [self._row_to_event(r) for r in rows]

    def query_by_target(
        self,
        target_id: str,
        *,
        limit: int = 100,
    ) -> list[TacticalEvent]:
        """Get all events related to a specific target."""
        with self._lock:
            rows = self._fetchall(
                """SELECT * FROM tactical_events
                   WHERE target_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (target_id, limit),
            )
        return [self._row_to_event(r) for r in rows]

    def query_by_source(
        self,
        source: str,
        *,
        limit: int = 100,
    ) -> list[TacticalEvent]:
        """Get all events from a specific source."""
        with self._lock:
            rows = self._fetchall(
                """SELECT * FROM tactical_events
                   WHERE source = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (source, limit),
            )
        return [self._row_to_event(r) for r in rows]

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def count(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        source: Optional[str] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> int:
        """Count events matching filters."""
        conditions: list[str] = []
        params: list = []

        if event_type is not None:
            conditions.append("event_type = ?")
            params.append(event_type)
        if severity is not None:
            conditions.append("severity = ?")
            params.append(severity)
        if source is not None:
            conditions.append("source = ?")
            params.append(source)
        if start is not None:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            conditions.append("timestamp <= ?")
            params.append(end)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        with self._lock:
            row = self._fetchone(
                f"SELECT COUNT(*) FROM tactical_events {where}",
                tuple(params),
            )
        return row[0] if row else 0

    def get_stats(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> dict:
        """Get aggregate statistics about stored events."""
        time_filter = ""
        params: list = []
        if start is not None:
            time_filter += " AND timestamp >= ?"
            params.append(start)
        if end is not None:
            time_filter += " AND timestamp <= ?"
            params.append(end)

        # Remove leading " AND " if present for WHERE clause
        where = ""
        if time_filter:
            where = "WHERE " + time_filter.lstrip(" AND ")

        with self._lock:
            total = self._fetchone(
                f"SELECT COUNT(*) FROM tactical_events {where}",
                tuple(params),
            )[0]

            by_type: dict[str, int] = {}
            for row in self._fetchall(
                f"SELECT event_type, COUNT(*) FROM tactical_events {where} GROUP BY event_type ORDER BY COUNT(*) DESC LIMIT 30",
                tuple(params),
            ):
                by_type[row[0]] = row[1]

            by_severity: dict[str, int] = {}
            for row in self._fetchall(
                f"SELECT severity, COUNT(*) FROM tactical_events {where} GROUP BY severity",
                tuple(params),
            ):
                by_severity[row[0]] = row[1]

            by_source: dict[str, int] = {}
            for row in self._fetchall(
                f"SELECT source, COUNT(*) FROM tactical_events {where} GROUP BY source ORDER BY COUNT(*) DESC LIMIT 20",
                tuple(params),
            ):
                by_source[row[0]] = row[1]

            oldest = self._fetchone(
                f"SELECT MIN(timestamp) FROM tactical_events {where}",
                tuple(params),
            )[0]

            newest = self._fetchone(
                f"SELECT MAX(timestamp) FROM tactical_events {where}",
                tuple(params),
            )[0]

        return {
            "total_events": total,
            "by_type": by_type,
            "by_severity": by_severity,
            "by_source": by_source,
            "oldest_event": oldest,
            "newest_event": newest,
            "max_events": self._max_events,
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup(self, keep: Optional[int] = None) -> int:
        """Remove oldest events if over the limit.

        Returns the number of events deleted.
        """
        limit = keep if keep is not None else self._max_events

        with self._lock:
            total = self._fetchone(
                "SELECT COUNT(*) FROM tactical_events"
            )[0]

            if total <= limit:
                return 0

            to_delete = total - limit
            self._execute(
                """DELETE FROM tactical_events WHERE event_id IN (
                   SELECT event_id FROM tactical_events
                   ORDER BY timestamp ASC LIMIT ?)""",
                (to_delete,),
            )
            self._commit()
            return to_delete

    def clear(self) -> int:
        """Delete all events. Returns count deleted."""
        with self._lock:
            count = self._fetchone(
                "SELECT COUNT(*) FROM tactical_events"
            )[0]
            self._execute("DELETE FROM tactical_events")
            self._commit()
        return count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> TacticalEvent:
        """Convert a database row to a TacticalEvent."""
        try:
            data = json.loads(row["data"])
        except (json.JSONDecodeError, KeyError):
            data = {}

        return TacticalEvent(
            event_id=row["event_id"],
            timestamp=row["timestamp"],
            event_type=row["event_type"],
            severity=row["severity"],
            source=row["source"],
            target_id=row["target_id"],
            operator=row["operator"],
            summary=row["summary"],
            data=data,
            position_lat=row["position_lat"],
            position_lng=row["position_lng"],
            site_id=row["site_id"],
        )
