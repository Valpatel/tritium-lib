# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Tests for AsyncBaseStore."""
from __future__ import annotations

import pytest
import pytest_asyncio

from tritium_lib.sdk.async_store import AsyncBaseStore


# -- concrete test subclass ---------------------------------------------------

class ItemStore(AsyncBaseStore):
    """Minimal store used by the tests."""

    async def _create_tables(self) -> None:
        await self.execute(
            "CREATE TABLE IF NOT EXISTS items ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  name TEXT NOT NULL,"
            "  value REAL DEFAULT 0"
            ")"
        )


# -- fixtures -----------------------------------------------------------------

@pytest_asyncio.fixture
async def store(tmp_path):
    s = ItemStore(db_path=tmp_path / "test.db", addon_id="test-addon")
    await s.initialize()
    yield s
    await s.close()


# -- tests --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_initialize_creates_db(tmp_path):
    db_path = tmp_path / "sub" / "init.db"
    s = ItemStore(db_path=db_path, addon_id="init")
    await s.initialize()
    assert db_path.exists()
    assert s.is_initialized
    await s.close()


@pytest.mark.asyncio
async def test_wal_mode(store: ItemStore):
    row = await store.fetchone("PRAGMA journal_mode")
    assert row is not None
    assert row["journal_mode"] == "wal"


@pytest.mark.asyncio
async def test_execute_and_fetchall(store: ItemStore):
    await store.execute("INSERT INTO items (name, value) VALUES (?, ?)", ("alpha", 1.0))
    await store.execute("INSERT INTO items (name, value) VALUES (?, ?)", ("beta", 2.0))
    rows = await store.fetchall("SELECT name, value FROM items ORDER BY name")
    assert len(rows) == 2
    assert rows[0]["name"] == "alpha"
    assert rows[1]["value"] == 2.0


@pytest.mark.asyncio
async def test_fetchone_returns_dict(store: ItemStore):
    await store.execute("INSERT INTO items (name, value) VALUES (?, ?)", ("gamma", 3.0))
    row = await store.fetchone("SELECT * FROM items WHERE name = ?", ("gamma",))
    assert isinstance(row, dict)
    assert row["name"] == "gamma"
    assert row["value"] == 3.0


@pytest.mark.asyncio
async def test_fetchone_returns_none(store: ItemStore):
    row = await store.fetchone("SELECT * FROM items WHERE name = ?", ("nonexistent",))
    assert row is None


@pytest.mark.asyncio
async def test_count(store: ItemStore):
    assert await store.count("items") == 0
    await store.execute("INSERT INTO items (name) VALUES (?)", ("x",))
    assert await store.count("items") == 1
    await store.execute("INSERT INTO items (name) VALUES (?)", ("y",))
    assert await store.count("items") == 2


@pytest.mark.asyncio
async def test_table_exists(store: ItemStore):
    assert await store.table_exists("items") is True
    assert await store.table_exists("nonexistent_table") is False


@pytest.mark.asyncio
async def test_context_manager(tmp_path):
    db_path = tmp_path / "ctx.db"
    async with ItemStore(db_path=db_path, addon_id="ctx") as s:
        assert s.is_initialized
        await s.execute("INSERT INTO items (name) VALUES (?)", ("ctx_item",))
        assert await s.count("items") == 1
    # after exit, store should be closed
    assert not s.is_initialized


@pytest.mark.asyncio
async def test_close_and_reinitialize(tmp_path):
    db_path = tmp_path / "reopen.db"
    s = ItemStore(db_path=db_path, addon_id="reopen")
    await s.initialize()
    await s.execute("INSERT INTO items (name) VALUES (?)", ("persist",))
    await s.close()
    assert not s.is_initialized

    # re-open and verify data persisted
    await s.initialize()
    assert s.is_initialized
    rows = await s.fetchall("SELECT name FROM items")
    assert len(rows) == 1
    assert rows[0]["name"] == "persist"
    await s.close()


@pytest.mark.asyncio
async def test_vacuum(store: ItemStore):
    await store.execute("INSERT INTO items (name) VALUES (?)", ("v",))
    await store.execute("DELETE FROM items WHERE name = ?", ("v",))
    # vacuum should not raise
    await store.vacuum()


@pytest.mark.asyncio
async def test_executemany(store: ItemStore):
    data = [("a",), ("b",), ("c",)]
    await store.executemany("INSERT INTO items (name) VALUES (?)", data)
    assert await store.count("items") == 3
    rows = await store.fetchall("SELECT name FROM items ORDER BY name")
    names = [r["name"] for r in rows]
    assert names == ["a", "b", "c"]
