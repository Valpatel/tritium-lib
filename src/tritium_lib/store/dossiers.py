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
import os
import sqlite3
import time
import uuid
from pathlib import Path

from .base import BaseStore

# ---------------------------------------------------------------------------
# Retention defaults (QUESTIONS.md 2026-04-29 — dossiers.db hit 5.3 GB with a
# ~50 GB/90-day trajectory on the old dev box; dossier_signals is the
# high-volume table).  Override per-instance via constructor params, or
# globally via the TRITIUM_DOSSIER_SIGNAL_TTL_DAYS environment variable.
# ---------------------------------------------------------------------------

DEFAULT_SIGNAL_TTL_DAYS: float = 30.0
DEFAULT_PRUNE_BATCH_SIZE: int = 1000
DEFAULT_MAYBE_PRUNE_INTERVAL_S: float = 3600.0
ENV_SIGNAL_TTL_DAYS = "TRITIUM_DOSSIER_SIGNAL_TTL_DAYS"

#: Tag that exempts a dossier from the signal TTL, equivalent to pinned=1.
VIP_TAG = "vip"

_DAY_S = 86400.0

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
    notes TEXT NOT NULL DEFAULT '[]',
    pinned INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_dossiers_entity_type ON dossiers(entity_type);
CREATE INDEX IF NOT EXISTS idx_dossiers_alliance ON dossiers(alliance);
CREATE INDEX IF NOT EXISTS idx_dossiers_threat_level ON dossiers(threat_level);
CREATE INDEX IF NOT EXISTS idx_dossiers_last_seen ON dossiers(last_seen);
CREATE INDEX IF NOT EXISTS idx_dossiers_pinned ON dossiers(pinned);
"""

# ---------------------------------------------------------------------------
# Migration — add ``pinned`` column to existing databases that pre-date the
# Gap-fix C M-8 retention work.  SQLite ``ALTER TABLE`` with IF NOT EXISTS
# is not directly supported; we probe the column list instead.
# ---------------------------------------------------------------------------

def _migrate_pinned_column(conn) -> None:
    """Add the pinned column to legacy dossier tables if missing."""
    try:
        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(dossiers)").fetchall()
        }
        if "pinned" not in cols:
            conn.execute(
                "ALTER TABLE dossiers ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dossiers_pinned ON dossiers(pinned)"
            )
            conn.commit()
    except Exception:
        # Table may not exist yet on a fresh DB — schema creation will
        # cover it.  Don't fail the constructor.
        pass

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

    Doctrine (ontology study §2.2/§2.8): the tracker is a read model; the
    dossier store is the edits layer; they join on deterministic target IDs
    (``ble_{mac}`` etc.) so re-detected entities reattach to their history.

    Retention policy (Gap-fix C M-8, hardened per QUESTIONS.md 2026-04-29)
    ----------------------------------------------------------------------
    ``dossier_signals`` is the high-volume table (5.3 GB observed on the
    old dev box, ~50 GB/90-day trajectory).  Signal rows carry a TTL
    (default 30 days); callers should periodically invoke :meth:`prune`
    — or call :meth:`maybe_prune` from write paths, which self-limits to
    roughly once per hour — to:
      * Drop ``dossier_signals`` older than the TTL for any dossier that
        is **not** exempt.  A dossier is exempt when ``pinned = 1`` OR it
        carries the ``'vip'`` tag — VIP dossiers keep signals forever.
      * Cap signals per dossier at 10000 — drop the oldest first.  The
        cap applies to all dossiers, including pinned/VIP ones.
    Deletes run in bounded batches (default 1000 rows per transaction) so
    the writer lock is never held for multi-second stretches.

    TTL resolution order: explicit constructor param > the
    ``TRITIUM_DOSSIER_SIGNAL_TTL_DAYS`` environment variable > the
    module default ``DEFAULT_SIGNAL_TTL_DAYS`` (30).
    """

    _SCHEMAS = (
        _SCHEMA_DOSSIERS,
        _SCHEMA_SIGNALS,
        _SCHEMA_ENRICHMENTS,
        _SCHEMA_FTS,
        _SCHEMA_FTS_TRIGGERS,
    )
    _FOREIGN_KEYS = True

    def __init__(
        self,
        db_path: str | Path,
        *,
        signal_ttl_days: float | None = None,
        prune_batch_size: int | None = None,
        maybe_prune_interval_s: float | None = None,
    ) -> None:
        """Open (or create) the dossier database.

        Parameters
        ----------
        db_path:
            Path to the SQLite file, or ``":memory:"`` for ephemeral use.
        signal_ttl_days:
            Signal-row TTL in days.  ``None`` (default) reads the
            ``TRITIUM_DOSSIER_SIGNAL_TTL_DAYS`` env var, falling back to
            :data:`DEFAULT_SIGNAL_TTL_DAYS`.
        prune_batch_size:
            Maximum rows deleted per transaction inside :meth:`prune`.
            Defaults to :data:`DEFAULT_PRUNE_BATCH_SIZE`.
        maybe_prune_interval_s:
            Minimum seconds between effective :meth:`maybe_prune` runs.
            Defaults to :data:`DEFAULT_MAYBE_PRUNE_INTERVAL_S` (1 hour).
        """
        if signal_ttl_days is None:
            raw = os.environ.get(ENV_SIGNAL_TTL_DAYS)
            if raw is not None:
                try:
                    signal_ttl_days = float(raw)
                except ValueError:
                    signal_ttl_days = None  # garbage env → module default
        self.signal_ttl_days: float = float(
            signal_ttl_days if signal_ttl_days is not None
            else DEFAULT_SIGNAL_TTL_DAYS
        )
        self.prune_batch_size: int = int(
            prune_batch_size if prune_batch_size is not None
            else DEFAULT_PRUNE_BATCH_SIZE
        )
        self.maybe_prune_interval_s: float = float(
            maybe_prune_interval_s if maybe_prune_interval_s is not None
            else DEFAULT_MAYBE_PRUNE_INTERVAL_S
        )
        self._last_maybe_prune: float = 0.0
        super().__init__(db_path)

    def _create_tables(self) -> None:  # noqa: D401 — override
        """Run schema creation, then any column-level migrations."""
        super()._create_tables()
        _migrate_pinned_column(self._conn)

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
        # Coerce pinned to bool for JSON-friendliness; legacy rows missing
        # the column return None and we treat that as not pinned.
        d["pinned"] = bool(d.get("pinned") or 0)
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

    # ------------------------------------------------------------------
    # Retention (Gap-fix C M-8)
    # ------------------------------------------------------------------

    def set_pinned(self, dossier_id: str, pinned: bool) -> bool:
        """Mark a dossier as pinned (exempt from retention) or unpinned.

        Returns True if the dossier existed.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE dossiers SET pinned = ? WHERE dossier_id = ?",
                (1 if pinned else 0, dossier_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def prune(
        self,
        *,
        max_signal_age_s: float | None = None,
        max_signals_per_dossier: int = 10000,
        batch_size: int | None = None,
        vacuum: bool = False,
        now: float | None = None,
    ) -> dict:
        """Apply the retention policy to dossier_signals.

        Rules (Gap-fix C M-8 + QUESTIONS.md 2026-04-29):
          * Drop signals older than the TTL for any dossier that is not
            exempt.  Exempt = ``pinned = 1`` OR tagged ``'vip'`` —
            VIP/pinned dossiers keep all signals at any age.
          * Cap each dossier's signal count at ``max_signals_per_dossier``;
            when exceeded, drop the oldest first.  The cap applies to all
            dossiers, including pinned/VIP ones.

        Deletes run in bounded batches: each batch is its own short
        transaction and the writer lock is released between batches, so
        concurrent ``add_signal`` calls are never blocked for multi-second
        stretches even when draining millions of expired rows.  The method
        is idempotent — safe to call repeatedly; a second call on a
        drained store deletes nothing.

        Enrichments and the dossier rows themselves are never deleted —
        only signals, which are the high-volume contributor to growth
        (5.3 GB / ~50 GB-90d projection in W204 Behavioral SAT).

        Parameters
        ----------
        max_signal_age_s:
            Maximum signal age in seconds.  ``None`` (default) uses the
            configured ``signal_ttl_days``.  An explicit value (e.g. from
            SC's DossierManager) always wins over the configured TTL.
        max_signals_per_dossier:
            Hard cap on per-dossier signal count.  Defaults to 10000.
        batch_size:
            Max rows deleted per transaction.  ``None`` uses the
            configured ``prune_batch_size``.
        vacuum:
            When True, run ``VACUUM`` after pruning to return freed pages
            to the filesystem.  Expensive (rewrites the whole database
            file) — off by default; the caller opts in explicitly, e.g.
            during scheduled maintenance windows.
        now:
            Override the wall-clock anchor (used in tests).  Defaults to
            ``time.time()``.

        Returns
        -------
        dict with ``aged_out`` (count from age-based prune),
        ``capped_out`` (count from per-dossier cap), ``batches``
        (delete transactions executed), and ``vacuumed`` (bool) — useful
        for instrumentation and logging.
        """
        anchor = now if now is not None else time.time()
        ttl_s = (
            max_signal_age_s if max_signal_age_s is not None
            else self.signal_ttl_days * _DAY_S
        )
        cutoff = anchor - ttl_s
        batch = max(1, int(batch_size if batch_size is not None
                           else self.prune_batch_size))

        aged_out = 0
        capped_out = 0
        batches = 0

        # 1) Age-based prune for non-exempt dossiers, in bounded batches.
        #    Exempt = pinned flag OR 'vip' present in the tags JSON array.
        while True:
            with self._lock:
                cur = self._conn.execute(
                    """DELETE FROM dossier_signals
                       WHERE signal_id IN (
                           SELECT s.signal_id
                           FROM dossier_signals s
                           JOIN dossiers d ON d.dossier_id = s.dossier_id
                           WHERE s.timestamp < ?
                             AND d.pinned = 0
                             AND NOT EXISTS (
                                 SELECT 1 FROM json_each(d.tags)
                                 WHERE json_each.value = ?
                             )
                           LIMIT ?
                       )""",
                    (cutoff, VIP_TAG, batch),
                )
                n = cur.rowcount or 0
                self._conn.commit()
            aged_out += n
            batches += 1
            if n < batch:
                break

        # 2) Per-dossier signal-count cap (applies to pinned/VIP too).
        with self._lock:
            over_cap = self._conn.execute(
                """SELECT dossier_id, COUNT(*) AS n
                   FROM dossier_signals
                   GROUP BY dossier_id
                   HAVING n > ?""",
                (max_signals_per_dossier,),
            ).fetchall()

        for row in over_cap:
            dossier_id = row["dossier_id"]
            remaining = row["n"] - max_signals_per_dossier
            while remaining > 0:
                chunk = min(remaining, batch)
                with self._lock:
                    cur = self._conn.execute(
                        """DELETE FROM dossier_signals
                           WHERE signal_id IN (
                               SELECT signal_id FROM dossier_signals
                               WHERE dossier_id = ?
                               ORDER BY timestamp ASC
                               LIMIT ?
                           )""",
                        (dossier_id, chunk),
                    )
                    n = cur.rowcount or 0
                    self._conn.commit()
                capped_out += n
                batches += 1
                remaining -= n
                if n == 0:
                    break  # row vanished underneath us — don't spin

        vacuumed = False
        if vacuum:
            with self._lock:
                self._conn.commit()  # VACUUM refuses to run mid-transaction
                self._conn.execute("VACUUM")
            vacuumed = True

        return {
            "aged_out": aged_out,
            "capped_out": capped_out,
            "batches": batches,
            "vacuumed": vacuumed,
        }

    def maybe_prune(self, *, now: float | None = None) -> dict | None:
        """Run :meth:`prune` if enough time has passed; otherwise no-op.

        Internally rate-limited to one effective run per
        ``maybe_prune_interval_s`` (default 1 hour), so write paths can
        call it unconditionally after every ``add_signal`` burst without
        paying the prune cost on each call.

        Integration point for SC (do not wire here): SC's
        ``DossierManager`` currently schedules ``store.prune(...)`` from
        its own hourly ``_maybe_prune`` job.  Any other store writer
        (router, plugin, ingest worker) that bypasses DossierManager can
        simply call ``store.maybe_prune()`` after writing — double
        rate-limiting is harmless because the prune itself is idempotent.

        Parameters
        ----------
        now:
            Override the wall-clock anchor (used in tests).  Defaults to
            ``time.time()``.

        Returns
        -------
        The :meth:`prune` stats dict when a prune ran, or ``None`` when
        rate-limited.
        """
        anchor = now if now is not None else time.time()
        with self._lock:
            if anchor - self._last_maybe_prune < self.maybe_prune_interval_s:
                return None
            self._last_maybe_prune = anchor
        return self.prune(now=anchor)
