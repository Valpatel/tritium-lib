# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""AuditTrail — high-level audit interface for compliance and forensics.

Wraps the low-level ``AuditStore`` with:
  - Predefined compliance action helpers (target_accessed, zone_modified, etc.)
  - Fluent ``AuditQuery`` integration
  - Automatic rotation (oldest records pruned on a configurable schedule)
  - Export for compliance reporting
  - Entry-level ``AuditEntry`` objects (immutable, UUID-keyed)

The trail is backed by SQLite with WAL journal mode for safe concurrent
reads alongside a single writer.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from .actions import AuditAction
from .entry import AuditEntry, AuditSeverity
from .query import AuditQuery

# ---------------------------------------------------------------------------
# SQL schema — extends store/audit_log with entry_id column
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_trail (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT NOT NULL UNIQUE,
    timestamp REAL NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    resource TEXT NOT NULL DEFAULT '',
    resource_id TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL DEFAULT '',
    severity TEXT NOT NULL DEFAULT 'info',
    ip_address TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_trail_timestamp ON audit_trail(timestamp);
CREATE INDEX IF NOT EXISTS idx_trail_actor ON audit_trail(actor);
CREATE INDEX IF NOT EXISTS idx_trail_action ON audit_trail(action);
CREATE INDEX IF NOT EXISTS idx_trail_severity ON audit_trail(severity);
CREATE INDEX IF NOT EXISTS idx_trail_resource ON audit_trail(resource);
CREATE INDEX IF NOT EXISTS idx_trail_entry_id ON audit_trail(entry_id);
"""


class AuditTrail:
    """Persistent audit trail for compliance and forensics.

    Parameters
    ----------
    db_path :
        Path to the SQLite database file.  Use ``":memory:"`` for testing.
    max_entries :
        Maximum entries to retain.  Oldest are pruned during ``rotate()``.
    auto_rotate_threshold :
        When total entries exceed this multiple of ``max_entries``, an
        automatic rotation is triggered on the next ``record()`` call.
        Set to 0 to disable auto-rotation.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        max_entries: int = 100_000,
        auto_rotate_threshold: float = 1.1,
    ) -> None:
        self._db_path = str(db_path)
        self._max_entries = max_entries
        self._auto_rotate_threshold = auto_rotate_threshold
        self._lock = threading.Lock()
        self._record_count = 0

        # Create parent directory if needed
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        # Cache the current count
        with self._lock:
            self._record_count = self._conn.execute(
                "SELECT COUNT(*) FROM audit_trail"
            ).fetchone()[0]

    # ------------------------------------------------------------------
    # Core recording
    # ------------------------------------------------------------------

    def record(
        self,
        actor: str,
        action: str,
        resource: str = "",
        resource_id: str = "",
        details: str = "",
        severity: str = "info",
        source_ip: str = "",
        metadata: Optional[dict] = None,
        timestamp: Optional[float] = None,
    ) -> AuditEntry:
        """Record an audit entry.

        Returns the created ``AuditEntry`` with its assigned UUID and
        database ID.
        """
        entry_id = str(uuid.uuid4())
        ts = timestamp if timestamp is not None else time.time()
        meta_json = json.dumps(metadata or {})

        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO audit_trail
                   (entry_id, timestamp, actor, action, resource, resource_id,
                    detail, severity, ip_address, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (entry_id, ts, actor, action, resource, resource_id,
                 details, severity, source_ip, meta_json),
            )
            self._conn.commit()
            db_id = cursor.lastrowid
            self._record_count += 1

        entry = AuditEntry(
            entry_id=entry_id,
            timestamp=ts,
            actor=actor,
            action=action,
            resource=resource,
            resource_id=resource_id,
            details=details,
            severity=severity,
            source_ip=source_ip,
            metadata=metadata or {},
            db_id=db_id,
        )

        # Auto-rotate if we've exceeded the threshold
        if (
            self._auto_rotate_threshold > 0
            and self._record_count > self._max_entries * self._auto_rotate_threshold
        ):
            self.rotate()

        return entry

    # ------------------------------------------------------------------
    # Typed compliance helpers
    # ------------------------------------------------------------------

    def record_target_accessed(
        self,
        actor: str,
        target_id: str,
        details: str = "",
        source_ip: str = "",
        metadata: Optional[dict] = None,
    ) -> AuditEntry:
        """Record that a target's data was accessed."""
        return self.record(
            actor=actor,
            action=AuditAction.TARGET_ACCESSED,
            resource="target",
            resource_id=target_id,
            details=details,
            severity="info",
            source_ip=source_ip,
            metadata=metadata,
        )

    def record_target_created(
        self,
        actor: str,
        target_id: str,
        details: str = "",
        source_ip: str = "",
        metadata: Optional[dict] = None,
    ) -> AuditEntry:
        """Record that a new target was created."""
        return self.record(
            actor=actor,
            action=AuditAction.TARGET_CREATED,
            resource="target",
            resource_id=target_id,
            details=details,
            severity="info",
            source_ip=source_ip,
            metadata=metadata,
        )

    def record_target_deleted(
        self,
        actor: str,
        target_id: str,
        details: str = "",
        source_ip: str = "",
        metadata: Optional[dict] = None,
    ) -> AuditEntry:
        """Record that a target was deleted."""
        return self.record(
            actor=actor,
            action=AuditAction.TARGET_DELETED,
            resource="target",
            resource_id=target_id,
            details=details,
            severity="warning",
            source_ip=source_ip,
            metadata=metadata,
        )

    def record_zone_modified(
        self,
        actor: str,
        zone_id: str,
        details: str = "",
        source_ip: str = "",
        metadata: Optional[dict] = None,
    ) -> AuditEntry:
        """Record that a zone/geofence was modified."""
        return self.record(
            actor=actor,
            action=AuditAction.ZONE_MODIFIED,
            resource="zone",
            resource_id=zone_id,
            details=details,
            severity="info",
            source_ip=source_ip,
            metadata=metadata,
        )

    def record_alert_acknowledged(
        self,
        actor: str,
        alert_id: str,
        details: str = "",
        source_ip: str = "",
        metadata: Optional[dict] = None,
    ) -> AuditEntry:
        """Record that an alert was acknowledged by an operator."""
        return self.record(
            actor=actor,
            action=AuditAction.ALERT_ACKNOWLEDGED,
            resource="alert",
            resource_id=alert_id,
            details=details,
            severity="info",
            source_ip=source_ip,
            metadata=metadata,
        )

    def record_report_generated(
        self,
        actor: str,
        report_id: str,
        details: str = "",
        source_ip: str = "",
        metadata: Optional[dict] = None,
    ) -> AuditEntry:
        """Record that a compliance/forensic report was generated."""
        return self.record(
            actor=actor,
            action=AuditAction.REPORT_GENERATED,
            resource="report",
            resource_id=report_id,
            details=details,
            severity="info",
            source_ip=source_ip,
            metadata=metadata,
        )

    def record_config_changed(
        self,
        actor: str,
        config_key: str,
        old_value: str = "",
        new_value: str = "",
        details: str = "",
        source_ip: str = "",
    ) -> AuditEntry:
        """Record a configuration change with before/after values."""
        meta = {"old_value": old_value, "new_value": new_value}
        return self.record(
            actor=actor,
            action=AuditAction.CONFIG_CHANGED,
            resource="config",
            resource_id=config_key,
            details=details or f"{config_key}: '{old_value}' -> '{new_value}'",
            severity="warning",
            source_ip=source_ip,
            metadata=meta,
        )

    def record_auth_login(
        self,
        actor: str,
        source_ip: str = "",
        details: str = "",
        metadata: Optional[dict] = None,
    ) -> AuditEntry:
        """Record a successful authentication."""
        return self.record(
            actor=actor,
            action=AuditAction.AUTH_LOGIN,
            resource="auth",
            details=details or "Login successful",
            severity="info",
            source_ip=source_ip,
            metadata=metadata,
        )

    def record_auth_failed(
        self,
        actor: str,
        source_ip: str = "",
        details: str = "",
        metadata: Optional[dict] = None,
    ) -> AuditEntry:
        """Record a failed authentication attempt."""
        return self.record(
            actor=actor,
            action=AuditAction.AUTH_FAILED,
            resource="auth",
            details=details or "Authentication failed",
            severity="warning",
            source_ip=source_ip,
            metadata=metadata,
        )

    def record_data_exported(
        self,
        actor: str,
        resource_id: str = "",
        details: str = "",
        source_ip: str = "",
        metadata: Optional[dict] = None,
    ) -> AuditEntry:
        """Record that data was exported from the system."""
        return self.record(
            actor=actor,
            action=AuditAction.DATA_EXPORTED,
            resource="data",
            resource_id=resource_id,
            details=details,
            severity="info",
            source_ip=source_ip,
            metadata=metadata,
        )

    def record_data_purged(
        self,
        actor: str,
        details: str = "",
        source_ip: str = "",
        metadata: Optional[dict] = None,
    ) -> AuditEntry:
        """Record that data was purged from the system."""
        return self.record(
            actor=actor,
            action=AuditAction.DATA_PURGED,
            resource="data",
            details=details,
            severity="critical",
            source_ip=source_ip,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def search(self, query: AuditQuery) -> list[AuditEntry]:
        """Execute an ``AuditQuery`` and return matching entries."""
        where, params = query.build_sql()
        sql = f"""SELECT * FROM audit_trail {where}
                  ORDER BY timestamp DESC
                  LIMIT ? OFFSET ?"""
        params.extend([query.max_results, query.skip])

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        return [self._row_to_entry(row) for row in rows]

    def get_by_id(self, entry_id: str) -> Optional[AuditEntry]:
        """Retrieve a specific audit entry by its UUID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM audit_trail WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def get_by_db_id(self, db_id: int) -> Optional[AuditEntry]:
        """Retrieve a specific audit entry by its database row ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM audit_trail WHERE id = ?",
                (db_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def count(self, query: Optional[AuditQuery] = None) -> int:
        """Count entries matching the query (or all entries if no query)."""
        if query is None:
            with self._lock:
                return self._conn.execute(
                    "SELECT COUNT(*) FROM audit_trail"
                ).fetchone()[0]

        where, params = query.build_sql()
        sql = f"SELECT COUNT(*) FROM audit_trail {where}"
        with self._lock:
            return self._conn.execute(sql, params).fetchone()[0]

    # ------------------------------------------------------------------
    # Export / compliance
    # ------------------------------------------------------------------

    def export(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        actions: Optional[list[str]] = None,
        limit: int = 10_000,
    ) -> list[dict]:
        """Export audit entries as a list of dictionaries for compliance.

        Parameters
        ----------
        start_time : float, optional
            Include entries at or after this epoch timestamp.
        end_time : float, optional
            Include entries at or before this epoch timestamp.
        actions : list[str], optional
            Only include these action types.
        limit : int
            Maximum number of entries to export.

        Returns
        -------
        list[dict]
            Each entry as a serializable dictionary, sorted by timestamp
            ascending (chronological order for reports).
        """
        query = AuditQuery().limit(limit)
        if start_time is not None:
            query.since(start_time)
        if end_time is not None:
            query.until(end_time)

        # For action filtering, we need a custom approach if multiple
        if actions and len(actions) == 1:
            query.by_action(actions[0])

        where, params = query.build_sql()

        # Handle multiple-action filter
        if actions and len(actions) > 1:
            placeholders = ",".join("?" for _ in actions)
            action_clause = f"action IN ({placeholders})"
            if where:
                where += f" AND {action_clause}"
            else:
                where = f"WHERE {action_clause}"
            params.extend(actions)

        sql = f"""SELECT * FROM audit_trail {where}
                  ORDER BY timestamp ASC
                  LIMIT ?"""
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        return [self._row_to_entry(row).to_dict() for row in rows]

    def get_stats(self) -> dict:
        """Aggregate statistics about the audit trail."""
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM audit_trail"
            ).fetchone()[0]

            by_severity: dict[str, int] = {}
            for row in self._conn.execute(
                "SELECT severity, COUNT(*) FROM audit_trail GROUP BY severity"
            ):
                by_severity[row[0]] = row[1]

            by_action: dict[str, int] = {}
            for row in self._conn.execute(
                "SELECT action, COUNT(*) FROM audit_trail "
                "GROUP BY action ORDER BY COUNT(*) DESC LIMIT 20"
            ):
                by_action[row[0]] = row[1]

            unique_actors = self._conn.execute(
                "SELECT COUNT(DISTINCT actor) FROM audit_trail"
            ).fetchone()[0]

            oldest = self._conn.execute(
                "SELECT MIN(timestamp) FROM audit_trail"
            ).fetchone()[0]
            newest = self._conn.execute(
                "SELECT MAX(timestamp) FROM audit_trail"
            ).fetchone()[0]

        return {
            "total_entries": total,
            "by_severity": by_severity,
            "top_actions": by_action,
            "unique_actors": unique_actors,
            "oldest_entry": oldest,
            "newest_entry": newest,
            "max_entries": self._max_entries,
        }

    # ------------------------------------------------------------------
    # Rotation / maintenance
    # ------------------------------------------------------------------

    def rotate(self, keep: Optional[int] = None) -> int:
        """Remove oldest entries exceeding the retention limit.

        Parameters
        ----------
        keep : int, optional
            Number of entries to keep.  Defaults to ``max_entries``.

        Returns
        -------
        int
            Number of entries deleted.
        """
        limit = keep if keep is not None else self._max_entries

        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM audit_trail"
            ).fetchone()[0]

            if total <= limit:
                return 0

            to_delete = total - limit
            self._conn.execute(
                """DELETE FROM audit_trail WHERE id IN (
                   SELECT id FROM audit_trail ORDER BY timestamp ASC LIMIT ?)""",
                (to_delete,),
            )
            self._conn.commit()
            self._record_count = self._conn.execute(
                "SELECT COUNT(*) FROM audit_trail"
            ).fetchone()[0]
            return to_delete

    def clear(self) -> int:
        """Delete all audit entries.  Returns the count deleted."""
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM audit_trail"
            ).fetchone()[0]
            self._conn.execute("DELETE FROM audit_trail")
            self._conn.commit()
            self._record_count = 0
        return count

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> AuditEntry:
        """Convert a database row to an ``AuditEntry``."""
        try:
            metadata = json.loads(row["metadata"])
        except (json.JSONDecodeError, KeyError):
            metadata = {}

        return AuditEntry(
            db_id=row["id"],
            entry_id=row["entry_id"],
            timestamp=row["timestamp"],
            actor=row["actor"],
            action=row["action"],
            resource=row["resource"],
            resource_id=row["resource_id"],
            details=row["detail"],
            severity=row["severity"],
            source_ip=row["ip_address"],
            metadata=metadata,
        )
