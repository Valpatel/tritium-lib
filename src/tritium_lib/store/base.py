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
        if self._FOREIGN_KEYS:
            self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        """Execute all schema statements defined in _SCHEMAS."""
        for schema in self._SCHEMAS:
            self._conn.executescript(schema)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
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
