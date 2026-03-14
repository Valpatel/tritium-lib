# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SQLite-backed target store for persistent target tracking.

Tracks targets across the Tritium ecosystem with position history,
full-text search, and aggregated statistics.  Any Tritium service
(edge server, command center, standalone tool) can instantiate this
with a path to a SQLite database file.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# SQL schemas
# ---------------------------------------------------------------------------

_SCHEMA_TARGETS = """
CREATE TABLE IF NOT EXISTS targets (
    target_id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    alliance TEXT NOT NULL DEFAULT '',
    asset_type TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    position_x REAL,
    position_y REAL,
    position_confidence REAL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_targets_alliance ON targets(alliance);
CREATE INDEX IF NOT EXISTS idx_targets_source ON targets(source);
CREATE INDEX IF NOT EXISTS idx_targets_last_seen ON targets(last_seen);
"""

_SCHEMA_TARGET_HISTORY = """
CREATE TABLE IF NOT EXISTS target_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    timestamp REAL NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (target_id) REFERENCES targets(target_id)
);
CREATE INDEX IF NOT EXISTS idx_history_target ON target_history(target_id);
CREATE INDEX IF NOT EXISTS idx_history_ts ON target_history(timestamp);
"""

_SCHEMA_TARGETS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS targets_fts USING fts5(
    target_id,
    name,
    source,
    content='targets',
    content_rowid='rowid'
);
"""

# Triggers to keep FTS in sync with the targets table.
_SCHEMA_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS targets_ai AFTER INSERT ON targets BEGIN
    INSERT INTO targets_fts(rowid, target_id, name, source)
    VALUES (new.rowid, new.target_id, new.name, new.source);
END;
CREATE TRIGGER IF NOT EXISTS targets_ad AFTER DELETE ON targets BEGIN
    INSERT INTO targets_fts(targets_fts, rowid, target_id, name, source)
    VALUES ('delete', old.rowid, old.target_id, old.name, old.source);
END;
CREATE TRIGGER IF NOT EXISTS targets_au AFTER UPDATE ON targets BEGIN
    INSERT INTO targets_fts(targets_fts, rowid, target_id, name, source)
    VALUES ('delete', old.rowid, old.target_id, old.name, old.source);
    INSERT INTO targets_fts(rowid, target_id, name, source)
    VALUES (new.rowid, new.target_id, new.name, new.source);
END;
"""


class TargetStore:
    """SQLite-backed persistent target tracking.

    Shared across the Tritium ecosystem — any service (edge server,
    command center, standalone tool) can instantiate this with a path
    to get persistent target tracking with position history and search.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Open (or create) the target tracking database.

        Parameters
        ----------
        db_path:
            Path to the SQLite database file.  Use ``":memory:"`` for
            an ephemeral in-memory database (useful for testing).
        """
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(_SCHEMA_TARGETS)
        cur.executescript(_SCHEMA_TARGET_HISTORY)
        cur.executescript(_SCHEMA_TARGETS_FTS)
        cur.executescript(_SCHEMA_FTS_TRIGGERS)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @staticmethod
    def _row_to_target(row: sqlite3.Row) -> dict:
        """Convert a targets table row to a dict with parsed metadata."""
        d = dict(row)
        try:
            d["metadata"] = json.loads(d.get("metadata") or "{}")
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = {}
        return d

    # ------------------------------------------------------------------
    # Sighting recording (upsert target + append history)
    # ------------------------------------------------------------------

    def record_sighting(
        self,
        target_id: str,
        name: str = "",
        alliance: str = "",
        asset_type: str = "",
        source: str = "",
        position_x: float | None = None,
        position_y: float | None = None,
        position_confidence: float | None = None,
        metadata: dict | None = None,
        timestamp: float | None = None,
    ) -> dict:
        """Record a target sighting — upserts the target and appends history.

        Parameters
        ----------
        target_id:
            Unique identifier for the target.
        name:
            Human-readable name/label.
        alliance:
            Faction/alliance (e.g. ``"friendly"``, ``"hostile"``, ``"unknown"``).
        asset_type:
            Type of asset (e.g. ``"vehicle"``, ``"person"``, ``"drone"``).
        source:
            Source of the sighting (e.g. ``"ble"``, ``"wifi"``, ``"camera"``).
        position_x, position_y:
            Position coordinates.
        position_confidence:
            Confidence in position estimate (0.0 to 1.0).
        metadata:
            Arbitrary JSON-serialisable metadata dict.
        timestamp:
            Unix timestamp.  Defaults to ``time.time()``.

        Returns the target record as a dict.
        """
        ts = timestamp or time.time()
        meta_json = json.dumps(metadata or {})

        with self._lock:
            # Check if target exists
            existing = self._conn.execute(
                "SELECT * FROM targets WHERE target_id = ?", (target_id,)
            ).fetchone()

            if existing is None:
                # Insert new target
                self._conn.execute(
                    """INSERT OR IGNORE INTO targets
                       (target_id, name, alliance, asset_type, source,
                        first_seen, last_seen,
                        position_x, position_y, position_confidence, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        target_id, name, alliance, asset_type, source,
                        ts, ts,
                        position_x, position_y, position_confidence, meta_json,
                    ),
                )
                # If another thread inserted first, fall through to update
                if self._conn.execute(
                    "SELECT changes()"
                ).fetchone()[0] == 0:
                    existing = self._conn.execute(
                        "SELECT * FROM targets WHERE target_id = ?",
                        (target_id,),
                    ).fetchone()

            if existing is not None:
                # Update existing — only overwrite non-empty fields
                updates: list[str] = ["last_seen = ?"]
                params: list[object] = [ts]
                if name:
                    updates.append("name = ?")
                    params.append(name)
                if alliance:
                    updates.append("alliance = ?")
                    params.append(alliance)
                if asset_type:
                    updates.append("asset_type = ?")
                    params.append(asset_type)
                if source:
                    updates.append("source = ?")
                    params.append(source)
                if position_x is not None:
                    updates.append("position_x = ?")
                    params.append(position_x)
                if position_y is not None:
                    updates.append("position_y = ?")
                    params.append(position_y)
                if position_confidence is not None:
                    updates.append("position_confidence = ?")
                    params.append(position_confidence)
                if metadata:
                    # Merge metadata
                    old_meta = json.loads(existing["metadata"] or "{}")
                    old_meta.update(metadata)
                    updates.append("metadata = ?")
                    params.append(json.dumps(old_meta))
                params.append(target_id)
                self._conn.execute(
                    f"UPDATE targets SET {', '.join(updates)} WHERE target_id = ?",
                    params,
                )

            # Append position history if coordinates provided
            if position_x is not None and position_y is not None:
                self._conn.execute(
                    """INSERT INTO target_history
                       (target_id, x, y, timestamp, source)
                       VALUES (?, ?, ?, ?, ?)""",
                    (target_id, position_x, position_y, ts, source),
                )

            self._conn.commit()

            # Read the result within the lock to avoid concurrent access
            row = self._conn.execute(
                "SELECT * FROM targets WHERE target_id = ?", (target_id,)
            ).fetchone()

        return self._row_to_target(row)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_target(self, target_id: str) -> dict | None:
        """Get a single target by ID, or ``None`` if not found."""
        row = self._conn.execute(
            "SELECT * FROM targets WHERE target_id = ?", (target_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_target(row)

    def get_all_targets(
        self,
        since: float | None = None,
        source: str | None = None,
        alliance: str | None = None,
    ) -> list[dict]:
        """Get all targets, optionally filtered.

        Parameters
        ----------
        since:
            Only return targets with ``last_seen >= since`` (unix timestamp).
        source:
            Filter by source string.
        alliance:
            Filter by alliance string.
        """
        clauses: list[str] = []
        params: list[object] = []
        if since is not None:
            clauses.append("last_seen >= ?")
            params.append(since)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if alliance is not None:
            clauses.append("alliance = ?")
            params.append(alliance)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM targets {where} ORDER BY last_seen DESC", params
        ).fetchall()
        return [self._row_to_target(r) for r in rows]

    def get_history(self, target_id: str, limit: int = 100) -> list[dict]:
        """Get position history for a target.

        Returns a list of dicts with ``x``, ``y``, ``timestamp``, ``source``,
        ordered newest first.
        """
        rows = self._conn.execute(
            """SELECT id, target_id, x, y, timestamp, source
               FROM target_history
               WHERE target_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (target_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Full-text search
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[dict]:
        """Full-text search on target_id, name, and source.

        Uses SQLite FTS5.  Supports standard FTS5 query syntax
        (prefix matching with ``*``, boolean operators, etc.).
        """
        if not query or not query.strip():
            return []
        # Escape quotes in query for safety, wrap terms for prefix match
        safe_q = query.strip().replace('"', '""')
        rows = self._conn.execute(
            """SELECT t.*
               FROM targets t
               JOIN targets_fts f ON t.rowid = f.rowid
               WHERE targets_fts MATCH ?
               ORDER BY rank""",
            (safe_q,),
        ).fetchall()
        return [self._row_to_target(r) for r in rows]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Database statistics.

        Returns ``total_targets``, ``active_last_hour``,
        ``by_source`` (dict), ``by_alliance`` (dict).
        """
        total = self._conn.execute(
            "SELECT COUNT(*) AS c FROM targets"
        ).fetchone()["c"]

        hour_ago = time.time() - 3600
        active = self._conn.execute(
            "SELECT COUNT(*) AS c FROM targets WHERE last_seen >= ?",
            (hour_ago,),
        ).fetchone()["c"]

        source_rows = self._conn.execute(
            "SELECT source, COUNT(*) AS c FROM targets GROUP BY source"
        ).fetchall()
        by_source = {r["source"]: r["c"] for r in source_rows}

        alliance_rows = self._conn.execute(
            "SELECT alliance, COUNT(*) AS c FROM targets GROUP BY alliance"
        ).fetchall()
        by_alliance = {r["alliance"]: r["c"] for r in alliance_rows}

        history_count = self._conn.execute(
            "SELECT COUNT(*) AS c FROM target_history"
        ).fetchone()["c"]

        db_size = 0
        if self._db_path != ":memory:":
            try:
                db_size = os.path.getsize(self._db_path)
            except OSError:
                pass

        return {
            "total_targets": total,
            "active_last_hour": active,
            "by_source": by_source,
            "by_alliance": by_alliance,
            "history_count": history_count,
            "db_size_bytes": db_size,
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete_target(self, target_id: str) -> bool:
        """Delete a target and its history.  Returns ``True`` if it existed."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM target_history WHERE target_id = ?", (target_id,)
            )
            cur = self._conn.execute(
                "DELETE FROM targets WHERE target_id = ?", (target_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def prune_history(self, older_than: float) -> int:
        """Delete history records older than *older_than* (unix timestamp).

        Returns the number of rows deleted.
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM target_history WHERE timestamp < ?", (older_than,)
            )
            self._conn.commit()
            return cur.rowcount
