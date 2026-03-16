# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SQLite-backed dossier store for persistent target intelligence.

A dossier is an accumulated intelligence profile of a real-world entity
(person, vehicle, device, animal) built from multiple correlated signals
across sensors and time.  Any Tritium service (edge server, command center,
standalone tool) can instantiate this with a path to a SQLite database file.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid

from .base import BaseStore

# ---------------------------------------------------------------------------
# SQL schemas
# ---------------------------------------------------------------------------

_SCHEMA_DOSSIERS = """
CREATE TABLE IF NOT EXISTS dossiers (
    dossier_id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'Unknown',
    entity_type TEXT NOT NULL DEFAULT 'unknown',
    confidence REAL NOT NULL DEFAULT 0.0,
    alliance TEXT NOT NULL DEFAULT 'unknown',
    threat_level TEXT NOT NULL DEFAULT 'none',
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    identifiers TEXT NOT NULL DEFAULT '{}',
    tags TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_dossiers_entity_type ON dossiers(entity_type);
CREATE INDEX IF NOT EXISTS idx_dossiers_alliance ON dossiers(alliance);
CREATE INDEX IF NOT EXISTS idx_dossiers_threat_level ON dossiers(threat_level);
CREATE INDEX IF NOT EXISTS idx_dossiers_last_seen ON dossiers(last_seen);
"""

_SCHEMA_SIGNALS = """
CREATE TABLE IF NOT EXISTS dossier_signals (
    signal_id TEXT PRIMARY KEY,
    dossier_id TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    signal_type TEXT NOT NULL DEFAULT '',
    data TEXT NOT NULL DEFAULT '{}',
    timestamp REAL NOT NULL,
    position_x REAL,
    position_y REAL,
    confidence REAL NOT NULL DEFAULT 0.5,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id)
);
CREATE INDEX IF NOT EXISTS idx_signals_dossier ON dossier_signals(dossier_id);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON dossier_signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_source ON dossier_signals(source);
"""

_SCHEMA_ENRICHMENTS = """
CREATE TABLE IF NOT EXISTS dossier_enrichments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dossier_id TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    enrichment_type TEXT NOT NULL DEFAULT '',
    data TEXT NOT NULL DEFAULT '{}',
    timestamp REAL NOT NULL,
    FOREIGN KEY (dossier_id) REFERENCES dossiers(dossier_id)
);
CREATE INDEX IF NOT EXISTS idx_enrichments_dossier ON dossier_enrichments(dossier_id);
"""

_SCHEMA_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS dossiers_fts USING fts5(
    name,
    entity_type,
    identifiers,
    tags,
    content='dossiers',
    content_rowid='rowid'
);
"""

_SCHEMA_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS dossiers_ai AFTER INSERT ON dossiers BEGIN
    INSERT INTO dossiers_fts(rowid, name, entity_type, identifiers, tags)
    VALUES (new.rowid, new.name, new.entity_type, new.identifiers, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS dossiers_ad AFTER DELETE ON dossiers BEGIN
    INSERT INTO dossiers_fts(dossiers_fts, rowid, name, entity_type, identifiers, tags)
    VALUES ('delete', old.rowid, old.name, old.entity_type, old.identifiers, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS dossiers_au AFTER UPDATE ON dossiers BEGIN
    INSERT INTO dossiers_fts(dossiers_fts, rowid, name, entity_type, identifiers, tags)
    VALUES ('delete', old.rowid, old.name, old.entity_type, old.identifiers, old.tags);
    INSERT INTO dossiers_fts(rowid, name, entity_type, identifiers, tags)
    VALUES (new.rowid, new.name, new.entity_type, new.identifiers, new.tags);
END;
"""


class DossierStore(BaseStore):
    """SQLite-backed persistent target intelligence.

    Accumulates signals, enrichments, and metadata into dossiers that
    represent real-world entities.  Supports full-text search, identifier
    lookup, and merging of duplicate dossiers.
    """

    _SCHEMAS = (
        _SCHEMA_DOSSIERS,
        _SCHEMA_SIGNALS,
        _SCHEMA_ENRICHMENTS,
        _SCHEMA_FTS,
        _SCHEMA_FTS_TRIGGERS,
    )
    _FOREIGN_KEYS = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(value: str | None, default: object = None) -> object:
        if default is None:
            default = {}
        try:
            return json.loads(value or json.dumps(default))
        except (json.JSONDecodeError, TypeError):
            return default

    @staticmethod
    def _row_to_dossier(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["identifiers"] = DossierStore._parse_json(d.get("identifiers"), {})
        d["tags"] = DossierStore._parse_json(d.get("tags"), [])
        d["notes"] = DossierStore._parse_json(d.get("notes"), [])
        return d

    @staticmethod
    def _row_to_signal(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["data"] = DossierStore._parse_json(d.get("data"), {})
        return d

    @staticmethod
    def _row_to_enrichment(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["data"] = DossierStore._parse_json(d.get("data"), {})
        return d

    # ------------------------------------------------------------------
    # Dossier CRUD
    # ------------------------------------------------------------------

    def create_dossier(
        self,
        name: str,
        entity_type: str = "unknown",
        identifiers: dict[str, str] | None = None,
        *,
        alliance: str = "unknown",
        threat_level: str = "none",
        confidence: float = 0.0,
        tags: list[str] | None = None,
        notes: list[str] | None = None,
        timestamp: float | None = None,
    ) -> str:
        """Create a new dossier and return its dossier_id."""
        dossier_id = str(uuid.uuid4())
        ts = timestamp or time.time()
        ids_json = json.dumps(identifiers or {})
        tags_json = json.dumps(tags or [])
        notes_json = json.dumps(notes or [])

        with self._lock:
            self._conn.execute(
                """INSERT INTO dossiers
                   (dossier_id, name, entity_type, confidence, alliance,
                    threat_level, first_seen, last_seen,
                    identifiers, tags, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    dossier_id, name, entity_type, confidence, alliance,
                    threat_level, ts, ts,
                    ids_json, tags_json, notes_json,
                ),
            )
            self._conn.commit()

        return dossier_id

    def get_dossier(self, dossier_id: str) -> dict | None:
        """Get a full dossier with its signals and enrichments."""
        row = self._conn.execute(
            "SELECT * FROM dossiers WHERE dossier_id = ?", (dossier_id,)
        ).fetchone()
        if row is None:
            return None

        dossier = self._row_to_dossier(row)

        signals = self._conn.execute(
            """SELECT * FROM dossier_signals
               WHERE dossier_id = ?
               ORDER BY timestamp DESC""",
            (dossier_id,),
        ).fetchall()
        dossier["signals"] = [self._row_to_signal(s) for s in signals]

        enrichments = self._conn.execute(
            """SELECT * FROM dossier_enrichments
               WHERE dossier_id = ?
               ORDER BY timestamp DESC""",
            (dossier_id,),
        ).fetchall()
        dossier["enrichments"] = [self._row_to_enrichment(e) for e in enrichments]

        return dossier

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def add_signal(
        self,
        dossier_id: str,
        source: str,
        signal_type: str,
        data: dict | None = None,
        *,
        position_x: float | None = None,
        position_y: float | None = None,
        confidence: float = 0.5,
        timestamp: float | None = None,
    ) -> str:
        """Append a new signal to a dossier.  Returns the signal_id."""
        signal_id = str(uuid.uuid4())
        ts = timestamp or time.time()
        data_json = json.dumps(data or {})

        with self._lock:
            self._conn.execute(
                """INSERT INTO dossier_signals
                   (signal_id, dossier_id, source, signal_type, data,
                    timestamp, position_x, position_y, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal_id, dossier_id, source, signal_type, data_json,
                    ts, position_x, position_y, confidence,
                ),
            )
            # Update last_seen on the parent dossier
            self._conn.execute(
                """UPDATE dossiers SET last_seen = MAX(last_seen, ?)
                   WHERE dossier_id = ?""",
                (ts, dossier_id),
            )
            self._conn.commit()

        return signal_id

    # ------------------------------------------------------------------
    # Enrichments
    # ------------------------------------------------------------------

    def add_enrichment(
        self,
        dossier_id: str,
        provider: str,
        enrichment_type: str,
        data: dict | None = None,
        *,
        timestamp: float | None = None,
    ) -> int:
        """Add an enrichment to a dossier.  Returns the enrichment row id."""
        ts = timestamp or time.time()
        data_json = json.dumps(data or {})

        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO dossier_enrichments
                   (dossier_id, provider, enrichment_type, data, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (dossier_id, provider, enrichment_type, data_json, ts),
            )
            self._conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def find_by_identifier(
        self, id_type: str, id_value: str
    ) -> dict | None:
        """Find a dossier by identifier key/value (e.g. MAC address).

        Scans the identifiers JSON column for a matching key-value pair.
        Returns the full dossier or None.
        """
        # Use JSON extract for efficient lookup
        rows = self._conn.execute(
            """SELECT * FROM dossiers
               WHERE json_extract(identifiers, ?) = ?""",
            (f"$.{id_type}", id_value),
        ).fetchall()
        if not rows:
            return None
        # Return the first match as a full dossier
        return self.get_dossier(rows[0]["dossier_id"])

    # ------------------------------------------------------------------
    # Full-text search
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[dict]:
        """Full-text search on name, entity_type, identifiers, tags.

        Uses SQLite FTS5.  Returns dossier summaries (no signals/enrichments).
        """
        if not query or not query.strip():
            return []
        safe_q = '"' + query.strip().replace('"', '""') + '"'
        rows = self._conn.execute(
            """SELECT d.*
               FROM dossiers d
               JOIN dossiers_fts f ON d.rowid = f.rowid
               WHERE dossiers_fts MATCH ?
               ORDER BY rank""",
            (safe_q,),
        ).fetchall()
        return [self._row_to_dossier(r) for r in rows]

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def get_recent(
        self,
        limit: int = 50,
        since: float | None = None,
    ) -> list[dict]:
        """Get recently active dossiers, ordered by last_seen descending.

        Parameters
        ----------
        limit:
            Maximum number of dossiers to return.
        since:
            Only return dossiers with ``last_seen >= since``.
        """
        if since is not None:
            rows = self._conn.execute(
                """SELECT * FROM dossiers
                   WHERE last_seen >= ?
                   ORDER BY last_seen DESC
                   LIMIT ?""",
                (since, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM dossiers
                   ORDER BY last_seen DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row_to_dossier(r) for r in rows]

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update_threat_level(self, dossier_id: str, level: str) -> bool:
        """Update the threat level of a dossier.  Returns True if it existed."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE dossiers SET threat_level = ? WHERE dossier_id = ?",
                (level, dossier_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def _update_json_field(self, dossier_id: str, field: str, value: object) -> bool:
        """Update a JSON-encoded column (tags, notes, identifiers).

        Returns True if the row existed.
        """
        if field not in ("tags", "notes", "identifiers"):
            raise ValueError(f"Cannot update field: {field}")
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE dossiers SET {field} = ? WHERE dossier_id = ?",
                (json.dumps(value), dossier_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_dossiers(self, primary_id: str, secondary_id: str) -> bool:
        """Merge secondary dossier into primary.

        Moves all signals and enrichments from secondary to primary,
        merges identifiers/tags/notes, adopts the earlier first_seen,
        and deletes the secondary dossier.

        Returns True if both dossiers existed and merge succeeded.
        """
        with self._lock:
            primary = self._conn.execute(
                "SELECT * FROM dossiers WHERE dossier_id = ?", (primary_id,)
            ).fetchone()
            secondary = self._conn.execute(
                "SELECT * FROM dossiers WHERE dossier_id = ?", (secondary_id,)
            ).fetchone()

            if primary is None or secondary is None:
                return False

            p = self._row_to_dossier(primary)
            s = self._row_to_dossier(secondary)

            # Merge identifiers (secondary values don't overwrite primary)
            merged_ids = {**s["identifiers"], **p["identifiers"]}
            # Merge tags (union, deduplicated)
            merged_tags = list(dict.fromkeys(p["tags"] + s["tags"]))
            # Merge notes (concatenate)
            merged_notes = p["notes"] + s["notes"]
            # Earliest first_seen, latest last_seen
            first_seen = min(p["first_seen"], s["first_seen"])
            last_seen = max(p["last_seen"], s["last_seen"])

            # Update primary dossier
            self._conn.execute(
                """UPDATE dossiers SET
                       identifiers = ?, tags = ?, notes = ?,
                       first_seen = ?, last_seen = ?
                   WHERE dossier_id = ?""",
                (
                    json.dumps(merged_ids),
                    json.dumps(merged_tags),
                    json.dumps(merged_notes),
                    first_seen, last_seen,
                    primary_id,
                ),
            )

            # Re-parent signals
            self._conn.execute(
                """UPDATE dossier_signals SET dossier_id = ?
                   WHERE dossier_id = ?""",
                (primary_id, secondary_id),
            )

            # Re-parent enrichments
            self._conn.execute(
                """UPDATE dossier_enrichments SET dossier_id = ?
                   WHERE dossier_id = ?""",
                (primary_id, secondary_id),
            )

            # Delete secondary dossier
            self._conn.execute(
                "DELETE FROM dossiers WHERE dossier_id = ?", (secondary_id,)
            )

            self._conn.commit()
            return True

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete_dossier(self, dossier_id: str) -> bool:
        """Delete a dossier and all its signals/enrichments."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM dossier_signals WHERE dossier_id = ?",
                (dossier_id,),
            )
            self._conn.execute(
                "DELETE FROM dossier_enrichments WHERE dossier_id = ?",
                (dossier_id,),
            )
            cur = self._conn.execute(
                "DELETE FROM dossiers WHERE dossier_id = ?", (dossier_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0
