# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for ScreenshotStore."""

import pytest
from tritium_lib.store.screenshot_store import ScreenshotStore


@pytest.fixture
def store():
    s = ScreenshotStore(":memory:")
    yield s
    s.close()


FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


def test_save_and_list(store):
    meta = store.save(FAKE_PNG, operator="alice", description="test capture")
    assert meta["operator"] == "alice"
    assert meta["description"] == "test capture"
    assert meta["file_size"] == len(FAKE_PNG)
    assert "screenshot_id" in meta

    items = store.list_screenshots()
    assert len(items) == 1
    assert items[0]["screenshot_id"] == meta["screenshot_id"]
    assert "png_data" not in items[0]  # list should not include binary


def test_get_returns_binary(store):
    meta = store.save(FAKE_PNG, operator="bob")
    result = store.get(meta["screenshot_id"])
    assert result is not None
    assert result["png_data"] == FAKE_PNG
    assert result["operator"] == "bob"


def test_get_missing(store):
    assert store.get("nonexistent-id") is None


def test_delete(store):
    meta = store.save(FAKE_PNG)
    assert store.delete(meta["screenshot_id"]) is True
    assert store.get(meta["screenshot_id"]) is None
    assert store.delete(meta["screenshot_id"]) is False


def test_count(store):
    assert store.count() == 0
    store.save(FAKE_PNG)
    store.save(FAKE_PNG)
    assert store.count() == 2


def test_list_by_operator(store):
    store.save(FAKE_PNG, operator="alice")
    store.save(FAKE_PNG, operator="bob")
    store.save(FAKE_PNG, operator="alice")

    alice_shots = store.list_screenshots(operator="alice")
    assert len(alice_shots) == 2
    assert all(s["operator"] == "alice" for s in alice_shots)


def test_list_pagination(store):
    for i in range(5):
        store.save(FAKE_PNG, description=f"shot-{i}")

    page1 = store.list_screenshots(limit=2, offset=0)
    page2 = store.list_screenshots(limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    assert page1[0]["screenshot_id"] != page2[0]["screenshot_id"]


def test_save_with_tags(store):
    meta = store.save(FAKE_PNG, tags=["battle", "zone-alpha"])
    assert meta["tags"] == ["battle", "zone-alpha"]

    result = store.get(meta["screenshot_id"])
    assert result["tags"] == ["battle", "zone-alpha"]


def test_save_with_dimensions(store):
    meta = store.save(FAKE_PNG, width=1920, height=1080)
    assert meta["width"] == 1920
    assert meta["height"] == 1080
