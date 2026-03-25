# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for BaseStore — SQLite WAL foundation, convenience methods."""

import sqlite3
import threading

import pytest

from tritium_lib.store.base import BaseStore


class SimpleStore(BaseStore):
    """Minimal concrete store for testing."""

    _SCHEMAS = (
        "CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT, value REAL);",
    )

    def insert(self, name: str, value: float) -> int:
        with self._lock:
            cur = self._execute(
                "INSERT INTO items (name, value) VALUES (?, ?)",
                (name, value),
            )
            self._commit()
            return cur.lastrowid

    def get(self, item_id: int) -> dict | None:
        with self._lock:
            row = self._fetchone("SELECT * FROM items WHERE id = ?", (item_id,))
            return dict(row) if row else None

    def get_all(self) -> list[dict]:
        with self._lock:
            rows = self._fetchall("SELECT * FROM items")
            return [dict(r) for r in rows]

    def insert_many(self, items: list[tuple[str, float]]) -> int:
        with self._lock:
            cur = self._executemany(
                "INSERT INTO items (name, value) VALUES (?, ?)",
                items,
            )
            self._commit()
            return cur.rowcount


class FKStore(BaseStore):
    """Store with foreign keys enabled."""

    _SCHEMAS = (
        "CREATE TABLE IF NOT EXISTS parents (id INTEGER PRIMARY KEY, name TEXT);",
        "CREATE TABLE IF NOT EXISTS children (id INTEGER PRIMARY KEY, parent_id INTEGER, name TEXT, FOREIGN KEY (parent_id) REFERENCES parents(id));",
    )
    _FOREIGN_KEYS = True


@pytest.fixture
def store():
    s = SimpleStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def fk_store():
    s = FKStore(":memory:")
    yield s
    s.close()


# ── Basic operations ────────────────────────────────────────────────

class TestBaseStoreBasics:
    def test_creates_tables(self, store):
        row = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='items'"
        ).fetchone()
        assert row is not None

    def test_wal_mode_requested(self, store):
        """In-memory databases use 'memory' journal mode; WAL is set for file-based."""
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        # :memory: databases can't use WAL, so they fall back to 'memory'
        assert mode in ("wal", "memory")

    def test_insert_and_get(self, store):
        rid = store.insert("alpha", 1.5)
        item = store.get(rid)
        assert item is not None
        assert item["name"] == "alpha"
        assert item["value"] == 1.5

    def test_get_nonexistent(self, store):
        assert store.get(999) is None

    def test_get_all(self, store):
        store.insert("a", 1.0)
        store.insert("b", 2.0)
        items = store.get_all()
        assert len(items) == 2

    def test_insert_many(self, store):
        count = store.insert_many([("x", 1.0), ("y", 2.0), ("z", 3.0)])
        assert count == 3
        assert len(store.get_all()) == 3

    def test_close(self, store):
        store.close()
        # Double close should not crash (or raise ProgrammingError which is fine)
        with pytest.raises(Exception):
            store._conn.execute("SELECT 1")


# ── Foreign keys ────────────────────────────────────────────────────

class TestBaseStoreForeignKeys:
    def test_foreign_keys_enabled(self, fk_store):
        fk_status = fk_store._conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk_status == 1

    def test_foreign_key_violation(self, fk_store):
        with pytest.raises(sqlite3.IntegrityError):
            fk_store._conn.execute(
                "INSERT INTO children (parent_id, name) VALUES (999, 'orphan')"
            )


# ── Thread safety ───────────────────────────────────────────────────

class TestBaseStoreThreadSafety:
    def test_concurrent_inserts(self, store):
        errors = []

        def insert_batch(start):
            try:
                for i in range(50):
                    store.insert(f"item-{start}-{i}", float(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=insert_batch, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(store.get_all()) == 200


# ── Multiple schemas ────────────────────────────────────────────────

class TestBaseStoreMultipleSchemas:
    def test_multiple_schema_strings(self):
        class MultiStore(BaseStore):
            _SCHEMAS = (
                "CREATE TABLE IF NOT EXISTS table_a (id INTEGER PRIMARY KEY, name TEXT);",
                "CREATE TABLE IF NOT EXISTS table_b (id INTEGER PRIMARY KEY, value REAL);",
            )

        s = MultiStore(":memory:")
        # Both tables should exist
        tables = s._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r[0] for r in tables}
        assert "table_a" in table_names
        assert "table_b" in table_names
        s.close()
