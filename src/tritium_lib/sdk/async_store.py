# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Async SQLite base store for Tritium addons.

Provides WAL-mode, dict-row, parameterized-query defaults so every
addon store starts from a safe, performant baseline.

Usage::

    from tritium_lib.sdk import AsyncBaseStore

    class MyStore(AsyncBaseStore):
        async def _create_tables(self) -> None:
            await self.execute(
                "CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT)"
            )
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

try:
    import aiosqlite
except ImportError:
    raise ImportError(
        "aiosqlite is required for AsyncBaseStore. "
        "Install it with: pip install aiosqlite"
    )

logger = logging.getLogger(__name__)


class AsyncBaseStore(ABC):
    """Abstract async SQLite store with WAL mode and dict rows."""

    def __init__(self, db_path: str | Path, addon_id: str = "") -> None:
        self.db_path: Path = Path(db_path)
        self.addon_id: str = addon_id
        self._db: aiosqlite.Connection | None = None
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Open connection, enable WAL + foreign keys, create tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._create_tables()
        await self._db.commit()
        self._initialized = True
        logger.debug("AsyncBaseStore initialized: %s", self.db_path)

    async def close(self) -> None:
        """Close the database connection safely."""
        if self._db is not None:
            try:
                await self._db.close()
            except Exception:
                logger.debug("Error closing database (may already be closed)")
            finally:
                self._db = None
                self._initialized = False

    @abstractmethod
    async def _create_tables(self) -> None:
        """Subclass must create its tables here."""
        ...

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AsyncBaseStore":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _ensure_open(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Store not initialized — call initialize() first")
        return self._db

    async def execute(self, sql: str, params: tuple = ()) -> aiosqlite.Cursor:
        """Execute a single SQL statement and commit."""
        db = self._ensure_open()
        cursor = await db.execute(sql, params)
        await db.commit()
        return cursor

    async def executemany(self, sql: str, params_seq: list[tuple]) -> None:
        """Execute a SQL statement for each parameter set and commit."""
        db = self._ensure_open()
        await db.executemany(sql, params_seq)
        await db.commit()

    async def fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        """Return one row as a dict, or None."""
        db = self._ensure_open()
        cursor = await db.execute(sql, params)
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    async def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        """Return all matching rows as a list of dicts."""
        db = self._ensure_open()
        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def count(self, table: str) -> int:
        """Return ``SELECT COUNT(*) FROM <table>``."""
        # Table name can't be parameterized; validate it.
        if not table.isidentifier():
            raise ValueError(f"Invalid table name: {table!r}")
        row = await self.fetchone(f"SELECT COUNT(*) AS cnt FROM {table}")
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def vacuum(self) -> None:
        """Run VACUUM to reclaim unused space."""
        db = self._ensure_open()
        await db.execute("VACUUM")

    async def table_exists(self, table_name: str) -> bool:
        """Check whether *table_name* exists in the database."""
        row = await self.fetchone(
            "SELECT COUNT(*) AS cnt FROM sqlite_master "
            "WHERE type='table' AND name=?",
            (table_name,),
        )
        return bool(row and row["cnt"] > 0)
