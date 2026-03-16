# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SQLite-backed audit log store for security compliance.

Records all system actions (who did what, when) with full detail.
Designed for security auditing, forensic review, and compliance
reporting. Every Tritium service can write audit entries.

Usage
-----
    store = AuditStore("/path/to/audit.db")
    store.log("user:admin", "login", "Authenticated via JWT")
    store.log("plugin:acoustic", "classify", "Gunshot detected", severity="high")
    entries = store.query(actor="user:admin", limit=100)
    stats = store.get_stats()
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from .base import BaseStore

# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_SCHEMA_AUDIT = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    severity TEXT NOT NULL DEFAULT 'info',
    resource TEXT NOT NULL DEFAULT '',
    resource_id TEXT NOT NULL DEFAULT '',
    ip_address TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_severity ON audit_log(severity);
"""

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AuditSeverity(str, Enum):
    """Severity level for audit entries."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """A single audit log entry."""
    id: int = 0
    timestamp: float = 0.0
    actor: str = ""       # who: "user:admin", "plugin:acoustic", "system"
    action: str = ""      # what: "login", "classify", "create_target", "delete"
    detail: str = ""      # human-readable description
    severity: str = "info"
    resource: str = ""    # affected resource type: "target", "user", "plugin"
    resource_id: str = "" # affected resource ID
    ip_address: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "detail": self.detail,
            "severity": self.severity,
            "resource": self.resource,
            "resource_id": self.resource_id,
            "ip_address": self.ip_address,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class AuditStore(BaseStore):
    """SQLite-backed audit log for security compliance.

    Thread-safe. Each instance manages its own database connection.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Use ":memory:" for testing.
    max_entries:
        Maximum entries to retain. Oldest are pruned during cleanup.
    """

    _SCHEMAS = (_SCHEMA_AUDIT,)

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        max_entries: int = 100_000,
    ) -> None:
        self._max_entries = max_entries
        super().__init__(db_path)

    def log(
        self,
        actor: str,
        action: str,
        detail: str = "",
        severity: str = "info",
        resource: str = "",
        resource_id: str = "",
        ip_address: str = "",
        metadata: Optional[dict] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """Record an audit log entry.

        Args:
            actor:       Who performed the action (e.g., "user:admin").
            action:      What was done (e.g., "login", "delete_target").
            detail:      Human-readable description.
            severity:    One of debug/info/warning/error/critical.
            resource:    Affected resource type (e.g., "target", "user").
            resource_id: Affected resource ID.
            ip_address:  Source IP address.
            metadata:    Additional structured data.
            timestamp:   Override timestamp (default: now).

        Returns:
            The inserted row ID.
        """
        ts = timestamp if timestamp is not None else time.time()
        meta_json = json.dumps(metadata or {})

        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO audit_log
                   (timestamp, actor, action, detail, severity,
                    resource, resource_id, ip_address, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ts, actor, action, detail, severity,
                 resource, resource_id, ip_address, meta_json),
            )
            self._conn.commit()
            return cursor.lastrowid

    def query(
        self,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        severity: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """Query audit log entries with optional filters.

        Returns entries sorted by timestamp descending (most recent first).
        """
        conditions = []
        params: list = []

        if actor is not None:
            conditions.append("actor = ?")
            params.append(actor)
        if action is not None:
            conditions.append("action = ?")
            params.append(action)
        if severity is not None:
            conditions.append("severity = ?")
            params.append(severity)
        if resource is not None:
            conditions.append("resource = ?")
            params.append(resource)
        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        sql = f"""SELECT * FROM audit_log {where}
                  ORDER BY timestamp DESC
                  LIMIT ? OFFSET ?"""
        params.extend([limit, offset])

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        return [self._row_to_entry(row) for row in rows]

    def get_entry(self, entry_id: int) -> Optional[AuditEntry]:
        """Get a specific audit entry by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM audit_log WHERE id = ?", (entry_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def count(
        self,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> int:
        """Count audit entries matching filters."""
        conditions = []
        params: list = []

        if actor is not None:
            conditions.append("actor = ?")
            params.append(actor)
        if action is not None:
            conditions.append("action = ?")
            params.append(action)
        if severity is not None:
            conditions.append("severity = ?")
            params.append(severity)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM audit_log {where}", params
            ).fetchone()
        return row[0]

    def get_stats(self) -> dict:
        """Get aggregate statistics about the audit log."""
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM audit_log"
            ).fetchone()[0]

            by_severity = {}
            for row in self._conn.execute(
                "SELECT severity, COUNT(*) FROM audit_log GROUP BY severity"
            ):
                by_severity[row[0]] = row[1]

            by_action = {}
            for row in self._conn.execute(
                "SELECT action, COUNT(*) FROM audit_log GROUP BY action ORDER BY COUNT(*) DESC LIMIT 20"
            ):
                by_action[row[0]] = row[1]

            actors = self._conn.execute(
                "SELECT COUNT(DISTINCT actor) FROM audit_log"
            ).fetchone()[0]

            oldest = self._conn.execute(
                "SELECT MIN(timestamp) FROM audit_log"
            ).fetchone()[0]
            newest = self._conn.execute(
                "SELECT MAX(timestamp) FROM audit_log"
            ).fetchone()[0]

        return {
            "total_entries": total,
            "by_severity": by_severity,
            "top_actions": by_action,
            "unique_actors": actors,
            "oldest_entry": oldest,
            "newest_entry": newest,
            "max_entries": self._max_entries,
        }

    def cleanup(self, keep: Optional[int] = None) -> int:
        """Remove oldest entries if over the limit.

        Args:
            keep: Number of entries to keep (default: max_entries).

        Returns:
            Number of entries deleted.
        """
        limit = keep if keep is not None else self._max_entries

        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM audit_log"
            ).fetchone()[0]

            if total <= limit:
                return 0

            to_delete = total - limit
            self._conn.execute(
                """DELETE FROM audit_log WHERE id IN (
                   SELECT id FROM audit_log ORDER BY timestamp ASC LIMIT ?)""",
                (to_delete,),
            )
            self._conn.commit()
            return to_delete

    def clear(self) -> int:
        """Delete all audit entries. Returns count deleted."""
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM audit_log"
            ).fetchone()[0]
            self._conn.execute("DELETE FROM audit_log")
            self._conn.commit()
        return count

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> AuditEntry:
        """Convert a database row to an AuditEntry."""
        try:
            metadata = json.loads(row["metadata"])
        except (json.JSONDecodeError, KeyError):
            metadata = {}

        return AuditEntry(
            id=row["id"],
            timestamp=row["timestamp"],
            actor=row["actor"],
            action=row["action"],
            detail=row["detail"],
            severity=row["severity"],
            resource=row["resource"],
            resource_id=row["resource_id"],
            ip_address=row["ip_address"],
            metadata=metadata,
        )
