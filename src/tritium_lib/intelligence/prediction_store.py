# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Persistent SQLite store for RL ``PredictionRecord`` history.

Implements **B-6** from gap-fix-B: ``RLMetrics`` previously kept
predictions in an in-memory ``deque`` that reset on every server
restart, so ``/api/intelligence/rl-metrics`` was effectively empty
after every reboot.  This store persists each prediction to a small
SQLite file (default: ``data/predictions.db``) and exposes a
deque-compatible API so callers can drop it in beside (or behind)
the existing in-memory deque without behavioural surprise.

Design notes
------------
- Pure stdlib ``sqlite3`` — tritium-lib is intentionally framework-
  free (no SQLAlchemy, see lib CLAUDE.md).
- Thread-safe: each call opens a connection with
  ``check_same_thread=False`` and uses a single ``threading.Lock`` to
  serialise writes.  Read queries are cheap and occur off the hot
  path.
- Bounded growth: ``max_records`` (default 100k) prunes oldest rows.
- API parity with ``collections.deque``: ``append``, ``__iter__``,
  ``__len__``, ``clear``.  Newest-first iteration matches the way
  RLMetrics consumes ``self._predictions`` for the API export.

Usage
-----
    from tritium_lib.intelligence.prediction_store import PredictionStore
    from tritium_lib.intelligence.rl_metrics import PredictionRecord

    store = PredictionStore("data/predictions.db")
    store.append(PredictionRecord(
        timestamp=time.time(),
        predicted_class=1,
        probability=0.82,
        correct=True,
    ))

    # Read back
    records = list(store)
    print(len(store))
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .rl_metrics import PredictionRecord

log = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    predicted_class INTEGER NOT NULL,
    probability REAL NOT NULL,
    correct INTEGER,                     -- 1 / 0 / NULL
    model_name TEXT NOT NULL DEFAULT 'correlation'
);
CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_pred_model ON predictions(model_name);
"""


class PredictionStore:
    """SQLite-backed deque-compatible PredictionRecord store.

    Mirrors the API ``RLMetrics`` already uses on its in-memory deque
    (``append``, ``__iter__``, ``__len__``, ``clear``) so the in-memory
    path keeps working in tests and the persistent path activates only
    when a store is wired in.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Parent dirs are created on demand.
    max_records:
        Maximum rows retained.  When exceeded, oldest rows are pruned
        on the next ``append``.  Set to 0 for unbounded.
    model_name:
        Default model label persisted with each row when ``append`` is
        called without an explicit ``model_name``.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        max_records: int = 100_000,
        model_name: str = "correlation",
    ) -> None:
        self._db_path = Path(db_path)
        self._max_records = int(max_records)
        self._default_model = model_name
        self._lock = threading.Lock()

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema / connection
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.executescript(_SCHEMA)
                conn.commit()
        except sqlite3.Error as exc:
            log.warning("PredictionStore schema init failed at %s: %s",
                        self._db_path, exc)

    # ------------------------------------------------------------------
    # Deque-compatible API
    # ------------------------------------------------------------------

    def append(
        self,
        record: PredictionRecord,
        *,
        model_name: str | None = None,
    ) -> None:
        """Persist a single PredictionRecord.

        ``correct`` is stored as 1 / 0 / NULL so we can round-trip the
        ``None = no feedback yet`` semantics.
        """
        model = model_name or self._default_model
        correct: Optional[int]
        if record.correct is True:
            correct = 1
        elif record.correct is False:
            correct = 0
        else:
            correct = None

        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO predictions "
                    "(timestamp, predicted_class, probability, correct, model_name) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        float(record.timestamp),
                        int(record.predicted_class),
                        float(record.probability),
                        correct,
                        model,
                    ),
                )

                # Prune oldest rows if we overflow.  Bounded by primary key
                # so this stays O(deletes), not O(n).
                if self._max_records > 0:
                    cur = conn.execute(
                        "SELECT COUNT(*) AS n FROM predictions"
                    )
                    n = cur.fetchone()["n"]
                    if n > self._max_records:
                        excess = n - self._max_records
                        conn.execute(
                            "DELETE FROM predictions WHERE id IN ("
                            "SELECT id FROM predictions "
                            "ORDER BY id ASC LIMIT ?)",
                            (excess,),
                        )

                conn.commit()
        except sqlite3.Error as exc:
            log.warning("PredictionStore.append failed: %s", exc)

    def extend(
        self,
        records: Iterable[PredictionRecord],
        *,
        model_name: str | None = None,
    ) -> None:
        """Bulk-append predictions in one transaction."""
        model = model_name or self._default_model
        rows = []
        for record in records:
            if record.correct is True:
                correct: Optional[int] = 1
            elif record.correct is False:
                correct = 0
            else:
                correct = None
            rows.append((
                float(record.timestamp),
                int(record.predicted_class),
                float(record.probability),
                correct,
                model,
            ))
        if not rows:
            return

        try:
            with self._lock, self._connect() as conn:
                conn.executemany(
                    "INSERT INTO predictions "
                    "(timestamp, predicted_class, probability, correct, model_name) "
                    "VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
        except sqlite3.Error as exc:
            log.warning("PredictionStore.extend failed: %s", exc)

    def __iter__(self) -> Iterator[PredictionRecord]:
        """Iterate predictions oldest-first to match deque iteration order.

        ``RLMetrics`` consumers slice the tail of this iterable when
        they want recent items, so old-first is the contractual order.
        """
        return iter(self.fetch())

    def __len__(self) -> int:
        try:
            with self._lock, self._connect() as conn:
                cur = conn.execute("SELECT COUNT(*) AS n FROM predictions")
                return int(cur.fetchone()["n"])
        except sqlite3.Error as exc:
            log.warning("PredictionStore.__len__ failed: %s", exc)
            return 0

    def clear(self) -> None:
        """Delete all stored predictions."""
        try:
            with self._lock, self._connect() as conn:
                conn.execute("DELETE FROM predictions")
                conn.commit()
        except sqlite3.Error as exc:
            log.warning("PredictionStore.clear failed: %s", exc)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def fetch(
        self,
        limit: int | None = None,
        *,
        model_name: str | None = None,
        since: float | None = None,
        newest_first: bool = False,
    ) -> list[PredictionRecord]:
        """Return persisted predictions.

        Default order is oldest-first to mimic deque iteration.  Pass
        ``newest_first=True`` for the API/dashboard view.
        """
        clauses: list[str] = []
        params: list = []

        if model_name is not None:
            clauses.append("model_name = ?")
            params.append(model_name)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(float(since))

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        order = "DESC" if newest_first else "ASC"
        sql = f"SELECT * FROM predictions{where} ORDER BY id {order}"
        if limit is not None and limit > 0:
            sql += " LIMIT ?"
            params.append(int(limit))

        try:
            with self._lock, self._connect() as conn:
                cur = conn.execute(sql, params)
                rows = cur.fetchall()
        except sqlite3.Error as exc:
            log.warning("PredictionStore.fetch failed: %s", exc)
            return []

        out: list[PredictionRecord] = []
        for row in rows:
            correct_val = row["correct"]
            if correct_val is None:
                correct: bool | None = None
            else:
                correct = bool(correct_val)
            out.append(PredictionRecord(
                timestamp=float(row["timestamp"]),
                predicted_class=int(row["predicted_class"]),
                probability=float(row["probability"]),
                correct=correct,
            ))
        return out

    def stats(self) -> dict:
        """Return summary statistics for the store."""
        try:
            with self._lock, self._connect() as conn:
                cur = conn.execute(
                    "SELECT "
                    "COUNT(*) AS n, "
                    "MIN(timestamp) AS first_ts, "
                    "MAX(timestamp) AS last_ts, "
                    "SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS correct_n, "
                    "SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END) AS incorrect_n "
                    "FROM predictions"
                )
                row = cur.fetchone()
        except sqlite3.Error as exc:
            log.warning("PredictionStore.stats failed: %s", exc)
            return {"count": 0}

        if row is None or row["n"] == 0:
            return {"count": 0}

        return {
            "count": int(row["n"]),
            "first_timestamp": float(row["first_ts"] or 0.0),
            "last_timestamp": float(row["last_ts"] or 0.0),
            "correct": int(row["correct_n"] or 0),
            "incorrect": int(row["incorrect_n"] or 0),
            "db_path": str(self._db_path),
            "max_records": self._max_records,
            "exported_at": time.time(),
        }

    @property
    def db_path(self) -> Path:
        return self._db_path
