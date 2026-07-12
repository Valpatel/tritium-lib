# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""BaseStore — common SQLite WAL foundation for all Tritium stores.

Every persistent store in Tritium uses the same pattern:
  1. Open SQLite with WAL journal mode
  2. Row factory = sqlite3.Row
  3. Thread-safe via threading.Lock
  4. Optional PRAGMA foreign_keys
  5. close() to shut down

This base class extracts that boilerplate so subclasses only need to
define their schemas and domain methods.

Usage::

    class MyStore(BaseStore):
        _SCHEMAS = ["CREATE TABLE IF NOT EXISTS ...", ...]
        _FOREIGN_KEYS = True  # optional, default False

        def __init__(self, db_path):
            super().__init__(db_path)

        def my_method(self):
            with self._lock:
                return self._fetchall("SELECT * FROM my_table")
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Sequence


class BaseStore:
    """Thread-safe SQLite store with WAL journal mode.

    Subclass attributes:
        _SCHEMAS:      List of SQL strings to execute on init (CREATE TABLE, etc.)
        _FOREIGN_KEYS: Whether to enable foreign_keys pragma (default: False)
    """

    _SCHEMAS: Sequence[str] = ()
    _FOREIGN_KEYS: bool = False

    #: Upper bound (bytes) the -wal file is truncated to after a
    #: successful checkpoint.  Without this SQLite never shrinks the WAL
    #: file — it keeps its high-water mark forever (a long-running
    #: server once grew a 12 GB -wal next to a 1.3 GB db, 2026-07-10).
    _JOURNAL_SIZE_LIMIT: int = 64 * 1024 * 1024

    _CHECKPOINT_MODES = ("PASSIVE", "FULL", "RESTART", "TRUNCATE")

    def __init__(self, db_path: str | Path) -> None:
        """Open (or create) the SQLite database.

        Parameters
        ----------
        db_path:
            Path to the SQLite database file.  Use ``":memory:"`` for
            an ephemeral in-memory database (useful for testing).
        """
        self._db_path = str(db_path)
        self._lock = threading.Lock()

        # Create parent directory if needed
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Bound the -wal file: after any checkpoint that resets the log,
        # SQLite truncates the file down to this many bytes instead of
        # leaving it at its high-water mark.
        self._conn.execute(
            f"PRAGMA journal_size_limit={int(self._JOURNAL_SIZE_LIMIT)}"
        )
        if self._FOREIGN_KEYS:
            self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        """Execute all schema statements defined in _SCHEMAS."""
        for schema in self._SCHEMAS:
            self._conn.executescript(schema)
        self._conn.commit()

    def checkpoint(self, mode: str = "TRUNCATE") -> dict[str, int]:
        """Run a WAL checkpoint and return its result.

        Long-running services should call this periodically (e.g. from
        an hourly retention sweep): passive auto-checkpoints can be
        starved by concurrent readers, and only RESTART/TRUNCATE modes
        (or ``journal_size_limit``) ever shrink the -wal file.

        Parameters
        ----------
        mode:
            One of ``PASSIVE``, ``FULL``, ``RESTART``, ``TRUNCATE``
            (default ``TRUNCATE`` — checkpoint everything and truncate
            the -wal file).

        Returns ``{"busy": 0|1, "log": n_frames, "checkpointed": n_frames}``.
        ``busy == 1`` means a concurrent reader/writer prevented
        completion — safe to retry later.
        """
        mode = mode.upper()
        if mode not in self._CHECKPOINT_MODES:
            raise ValueError(
                f"invalid checkpoint mode {mode!r}; "
                f"expected one of {self._CHECKPOINT_MODES}"
            )
        with self._lock:
            row = self._conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        busy, log, checkpointed = (int(v) for v in row)
        return {"busy": busy, "log": log, "checkpointed": checkpointed}

    def close(self) -> None:
        """Close the database connection (checkpointing the WAL first)."""
        try:
            self.checkpoint()
        except sqlite3.Error:
            pass  # best effort — close() must never raise on cleanup
        self._conn.close()

    # ------------------------------------------------------------------
    # Convenience wrappers (all require caller to hold self._lock)
    # ------------------------------------------------------------------

    def _execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        """Execute a single SQL statement. Caller must hold self._lock."""
        return self._conn.execute(sql, params)

    def _executemany(self, sql: str, params_seq: Sequence[tuple | dict]) -> sqlite3.Cursor:
        """Execute a SQL statement with many parameter sets. Caller must hold self._lock."""
        return self._conn.executemany(sql, params_seq)

    def _fetchall(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        """Execute and return all rows. Caller must hold self._lock."""
        return self._conn.execute(sql, params).fetchall()

    def _fetchone(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        """Execute and return one row. Caller must hold self._lock."""
        return self._conn.execute(sql, params).fetchone()

    def _commit(self) -> None:
        """Commit the current transaction. Caller must hold self._lock."""
        self._conn.commit()
