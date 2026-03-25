# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Extended TargetStore tests — sighting upserts, queries, FTS, stats, maintenance."""

import time

import pytest

from tritium_lib.store.targets import TargetStore


@pytest.fixture
def store():
    """Create an in-memory TargetStore for testing."""
    s = TargetStore(":memory:")
    yield s
    s.close()


# ── Sighting recording ─────────────────────────────────────────────

class TestRecordSighting:
    """Tests for record_sighting() upsert behavior."""

    def test_new_target_created(self, store):
        result = store.record_sighting(
            "ble_AA:BB:CC",
            name="Phone",
            alliance="unknown",
            asset_type="device",
            source="ble",
            position_x=10.0,
            position_y=20.0,
        )
        assert result["target_id"] == "ble_AA:BB:CC"
        assert result["name"] == "Phone"

    def test_update_existing_target(self, store):
        store.record_sighting("t1", name="First", source="ble", timestamp=1000.0)
        store.record_sighting("t1", name="Updated", source="wifi", timestamp=2000.0)
        t = store.get_target("t1")
        assert t["name"] == "Updated"
        assert t["source"] == "wifi"
        assert t["last_seen"] == pytest.approx(2000.0)

    def test_empty_fields_dont_overwrite(self, store):
        store.record_sighting("t1", name="Original", alliance="friendly", source="ble")
        store.record_sighting("t1", name="", alliance="", source="")
        t = store.get_target("t1")
        assert t["name"] == "Original"
        assert t["alliance"] == "friendly"
        assert t["source"] == "ble"

    def test_position_history_appended(self, store):
        store.record_sighting("t1", position_x=10.0, position_y=20.0, timestamp=1000.0)
        store.record_sighting("t1", position_x=15.0, position_y=25.0, timestamp=2000.0)
        history = store.get_history("t1")
        assert len(history) == 2

    def test_no_position_no_history(self, store):
        store.record_sighting("t1", name="No Position")
        history = store.get_history("t1")
        assert len(history) == 0

    def test_metadata_merged(self, store):
        store.record_sighting("t1", metadata={"rssi": -65})
        store.record_sighting("t1", metadata={"ssid": "MyWiFi"})
        t = store.get_target("t1")
        assert t["metadata"]["rssi"] == -65
        assert t["metadata"]["ssid"] == "MyWiFi"

    def test_metadata_update_overwrites_key(self, store):
        store.record_sighting("t1", metadata={"rssi": -65})
        store.record_sighting("t1", metadata={"rssi": -40})
        t = store.get_target("t1")
        assert t["metadata"]["rssi"] == -40

    def test_returns_target_dict(self, store):
        result = store.record_sighting("t1", name="Test")
        assert isinstance(result, dict)
        assert "target_id" in result
        assert "first_seen" in result
        assert "last_seen" in result
        assert "metadata" in result


# ── Queries ─────────────────────────────────────────────────────────

class TestTargetQueries:
    """Tests for query methods."""

    def test_get_target(self, store):
        store.record_sighting("t1", name="Alpha")
        t = store.get_target("t1")
        assert t is not None
        assert t["name"] == "Alpha"

    def test_get_target_nonexistent(self, store):
        assert store.get_target("fake") is None

    def test_get_all_targets(self, store):
        store.record_sighting("t1", source="ble")
        store.record_sighting("t2", source="wifi")
        all_targets = store.get_all_targets()
        assert len(all_targets) == 2

    def test_get_all_filter_source(self, store):
        store.record_sighting("t1", source="ble")
        store.record_sighting("t2", source="wifi")
        ble_only = store.get_all_targets(source="ble")
        assert len(ble_only) == 1
        assert ble_only[0]["source"] == "ble"

    def test_get_all_filter_alliance(self, store):
        store.record_sighting("t1", alliance="friendly")
        store.record_sighting("t2", alliance="hostile")
        hostile = store.get_all_targets(alliance="hostile")
        assert len(hostile) == 1

    def test_get_all_filter_since(self, store):
        store.record_sighting("t1", timestamp=1000.0)
        store.record_sighting("t2", timestamp=2000.0)
        recent = store.get_all_targets(since=1500.0)
        assert len(recent) == 1
        assert recent[0]["target_id"] == "t2"

    def test_get_history(self, store):
        store.record_sighting("t1", position_x=1.0, position_y=2.0, source="ble", timestamp=1000.0)
        store.record_sighting("t1", position_x=3.0, position_y=4.0, source="wifi", timestamp=2000.0)
        history = store.get_history("t1")
        assert len(history) == 2
        assert history[0]["timestamp"] > history[1]["timestamp"]  # newest first

    def test_get_history_limit(self, store):
        for i in range(20):
            store.record_sighting("t1", position_x=float(i), position_y=float(i), timestamp=float(1000 + i))
        history = store.get_history("t1", limit=5)
        assert len(history) == 5


# ── Full-text search ────────────────────────────────────────────────

class TestTargetSearch:
    """Tests for FTS5 search."""

    def test_search_by_name(self, store):
        store.record_sighting("t1", name="Red Truck")
        store.record_sighting("t2", name="Blue Car")
        results = store.search("Truck")
        assert len(results) == 1
        assert results[0]["name"] == "Red Truck"

    def test_search_by_target_id(self, store):
        store.record_sighting("ble_AA:BB:CC", name="Phone")
        results = store.search("ble_AA")
        assert len(results) >= 1

    def test_search_by_source(self, store):
        store.record_sighting("t1", name="A", source="camera_yolo")
        store.record_sighting("t2", name="B", source="ble")
        results = store.search("camera_yolo")
        assert len(results) == 1

    def test_search_empty_query(self, store):
        store.record_sighting("t1")
        assert store.search("") == []
        assert store.search("   ") == []

    def test_search_no_results(self, store):
        store.record_sighting("t1", name="Alpha")
        assert store.search("zzznotfound") == []


# ── Statistics ──────────────────────────────────────────────────────

class TestTargetStats:
    """Tests for get_stats()."""

    def test_empty_stats(self, store):
        stats = store.get_stats()
        assert stats["total_targets"] == 0
        assert stats["active_last_hour"] == 0
        assert stats["by_source"] == {}
        assert stats["by_alliance"] == {}
        assert stats["history_count"] == 0

    def test_populated_stats(self, store):
        store.record_sighting("t1", source="ble", alliance="friendly",
                              position_x=1, position_y=2)
        store.record_sighting("t2", source="wifi", alliance="hostile",
                              position_x=3, position_y=4)
        store.record_sighting("t3", source="ble", alliance="friendly")
        stats = store.get_stats()
        assert stats["total_targets"] == 3
        assert stats["by_source"]["ble"] == 2
        assert stats["by_source"]["wifi"] == 1
        assert stats["by_alliance"]["friendly"] == 2
        assert stats["by_alliance"]["hostile"] == 1
        assert stats["history_count"] == 2  # only t1 and t2 have positions


# ── Maintenance ─────────────────────────────────────────────────────

class TestTargetMaintenance:
    """Tests for delete and prune operations."""

    def test_delete_target(self, store):
        store.record_sighting("t1", position_x=1, position_y=2)
        assert store.delete_target("t1") is True
        assert store.get_target("t1") is None
        assert store.get_history("t1") == []

    def test_delete_nonexistent(self, store):
        assert store.delete_target("fake") is False

    def test_prune_history(self, store):
        store.record_sighting("t1", position_x=1, position_y=2, timestamp=1000.0)
        store.record_sighting("t1", position_x=3, position_y=4, timestamp=2000.0)
        store.record_sighting("t1", position_x=5, position_y=6, timestamp=3000.0)
        pruned = store.prune_history(older_than=2500.0)
        assert pruned == 2
        history = store.get_history("t1")
        assert len(history) == 1
        assert history[0]["timestamp"] == pytest.approx(3000.0)
