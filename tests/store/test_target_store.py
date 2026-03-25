# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.store.targets — SQLite-backed target store."""

import time

import pytest

from tritium_lib.store.targets import TargetStore


@pytest.fixture
def store():
    """Create an in-memory TargetStore for testing."""
    s = TargetStore(":memory:")
    yield s
    s.close()


class TestTargetStore:
    def test_record_sighting_new(self, store):
        result = store.record_sighting(
            target_id="ble_AA:BB:CC",
            name="iPhone",
            alliance="unknown",
            asset_type="phone",
            source="ble",
            position_x=10.0,
            position_y=20.0,
        )
        assert result["target_id"] == "ble_AA:BB:CC"
        assert result["name"] == "iPhone"
        assert result["source"] == "ble"

    def test_record_sighting_update(self, store):
        store.record_sighting(
            target_id="t1", name="Initial", source="ble",
            position_x=0.0, position_y=0.0,
        )
        result = store.record_sighting(
            target_id="t1", name="Updated", source="wifi",
            position_x=5.0, position_y=5.0,
        )
        assert result["name"] == "Updated"
        assert result["source"] == "wifi"

    def test_get_target(self, store):
        store.record_sighting(target_id="t1", name="Device")
        result = store.get_target("t1")
        assert result is not None
        assert result["name"] == "Device"

    def test_get_target_not_found(self, store):
        assert store.get_target("nonexistent") is None

    def test_get_all_targets(self, store):
        store.record_sighting(target_id="t1", source="ble")
        store.record_sighting(target_id="t2", source="wifi")
        targets = store.get_all_targets()
        assert len(targets) == 2

    def test_get_all_targets_filter_source(self, store):
        store.record_sighting(target_id="t1", source="ble")
        store.record_sighting(target_id="t2", source="wifi")
        targets = store.get_all_targets(source="ble")
        assert len(targets) == 1
        assert targets[0]["target_id"] == "t1"

    def test_get_all_targets_filter_alliance(self, store):
        store.record_sighting(target_id="t1", alliance="friendly")
        store.record_sighting(target_id="t2", alliance="hostile")
        targets = store.get_all_targets(alliance="hostile")
        assert len(targets) == 1

    def test_get_all_targets_filter_since(self, store):
        now = time.time()
        store.record_sighting(target_id="old", timestamp=now - 7200)
        store.record_sighting(target_id="new", timestamp=now)
        targets = store.get_all_targets(since=now - 3600)
        assert len(targets) == 1
        assert targets[0]["target_id"] == "new"

    def test_position_history(self, store):
        store.record_sighting(target_id="t1", position_x=0.0, position_y=0.0)
        store.record_sighting(target_id="t1", position_x=5.0, position_y=5.0)
        store.record_sighting(target_id="t1", position_x=10.0, position_y=10.0)
        history = store.get_history("t1")
        assert len(history) == 3
        # Newest first
        assert history[0]["x"] == 10.0

    def test_search_by_name(self, store):
        store.record_sighting(target_id="t1", name="iPhone 15 Pro")
        store.record_sighting(target_id="t2", name="Samsung Galaxy")
        results = store.search("iPhone")
        assert len(results) == 1
        assert results[0]["target_id"] == "t1"

    def test_search_empty_query(self, store):
        store.record_sighting(target_id="t1", name="Test")
        assert store.search("") == []
        assert store.search("   ") == []

    def test_get_stats(self, store):
        store.record_sighting(target_id="t1", source="ble", alliance="friendly",
                              position_x=0.0, position_y=0.0)
        store.record_sighting(target_id="t2", source="wifi", alliance="hostile",
                              position_x=5.0, position_y=5.0)
        stats = store.get_stats()
        assert stats["total_targets"] == 2
        assert "ble" in stats["by_source"]
        assert "friendly" in stats["by_alliance"]
        assert stats["history_count"] == 2

    def test_delete_target(self, store):
        store.record_sighting(target_id="t1", position_x=0.0, position_y=0.0)
        assert store.delete_target("t1") is True
        assert store.get_target("t1") is None
        assert store.get_history("t1") == []

    def test_delete_nonexistent(self, store):
        assert store.delete_target("nothing") is False

    def test_prune_history(self, store):
        now = time.time()
        store.record_sighting(target_id="t1", position_x=0.0, position_y=0.0,
                              timestamp=now - 7200)
        store.record_sighting(target_id="t1", position_x=5.0, position_y=5.0,
                              timestamp=now)
        pruned = store.prune_history(now - 3600)
        assert pruned == 1
        history = store.get_history("t1")
        assert len(history) == 1

    def test_metadata_merge(self, store):
        store.record_sighting(target_id="t1", metadata={"rssi": -60})
        store.record_sighting(target_id="t1", metadata={"name": "phone"})
        result = store.get_target("t1")
        assert result["metadata"]["rssi"] == -60
        assert result["metadata"]["name"] == "phone"

    def test_position_confidence(self, store):
        store.record_sighting(
            target_id="t1", position_x=10.0, position_y=20.0,
            position_confidence=0.85,
        )
        result = store.get_target("t1")
        assert result["position_confidence"] == 0.85
