# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the SQLite-backed TargetStore."""

import time

import pytest

from tritium_lib.store.targets import TargetStore


@pytest.fixture
def store():
    """In-memory TargetStore for testing."""
    s = TargetStore(":memory:")
    yield s
    s.close()


# ------------------------------------------------------------------
# CRUD basics
# ------------------------------------------------------------------


class TestRecordAndGet:
    def test_record_new_target(self, store: TargetStore):
        result = store.record_sighting(
            "tgt-001", name="Alpha", alliance="friendly",
            asset_type="vehicle", source="ble",
            position_x=10.0, position_y=20.0, position_confidence=0.9,
            metadata={"color": "red"},
        )
        assert result["target_id"] == "tgt-001"
        assert result["name"] == "Alpha"
        assert result["alliance"] == "friendly"
        assert result["asset_type"] == "vehicle"
        assert result["source"] == "ble"
        assert result["position_x"] == 10.0
        assert result["position_y"] == 20.0
        assert result["position_confidence"] == 0.9
        assert result["metadata"] == {"color": "red"}

    def test_get_target(self, store: TargetStore):
        store.record_sighting("tgt-002", name="Bravo", source="wifi")
        target = store.get_target("tgt-002")
        assert target is not None
        assert target["name"] == "Bravo"

    def test_get_nonexistent_target(self, store: TargetStore):
        assert store.get_target("nope") is None

    def test_upsert_updates_existing(self, store: TargetStore):
        ts1 = time.time() - 100
        store.record_sighting("tgt-003", name="Charlie", source="ble", timestamp=ts1)
        ts2 = time.time()
        store.record_sighting(
            "tgt-003", name="Charlie Updated", alliance="hostile",
            position_x=5.0, position_y=6.0, timestamp=ts2,
        )
        target = store.get_target("tgt-003")
        assert target is not None
        assert target["name"] == "Charlie Updated"
        assert target["alliance"] == "hostile"
        assert target["first_seen"] == pytest.approx(ts1, abs=1)
        assert target["last_seen"] == pytest.approx(ts2, abs=1)

    def test_upsert_merges_metadata(self, store: TargetStore):
        store.record_sighting("tgt-meta", metadata={"a": 1})
        store.record_sighting("tgt-meta", metadata={"b": 2})
        target = store.get_target("tgt-meta")
        assert target["metadata"] == {"a": 1, "b": 2}

    def test_delete_target(self, store: TargetStore):
        store.record_sighting("tgt-del", name="Delete Me",
                              position_x=1.0, position_y=2.0)
        assert store.delete_target("tgt-del") is True
        assert store.get_target("tgt-del") is None
        # History should also be gone
        assert store.get_history("tgt-del") == []

    def test_delete_nonexistent(self, store: TargetStore):
        assert store.delete_target("nope") is False


# ------------------------------------------------------------------
# Listing / filtering
# ------------------------------------------------------------------


class TestGetAllTargets:
    def test_get_all(self, store: TargetStore):
        store.record_sighting("a", name="A", source="ble")
        store.record_sighting("b", name="B", source="wifi")
        store.record_sighting("c", name="C", source="ble")
        targets = store.get_all_targets()
        assert len(targets) == 3

    def test_filter_by_source(self, store: TargetStore):
        store.record_sighting("a", source="ble")
        store.record_sighting("b", source="wifi")
        targets = store.get_all_targets(source="ble")
        assert len(targets) == 1
        assert targets[0]["target_id"] == "a"

    def test_filter_by_alliance(self, store: TargetStore):
        store.record_sighting("a", alliance="friendly")
        store.record_sighting("b", alliance="hostile")
        targets = store.get_all_targets(alliance="hostile")
        assert len(targets) == 1
        assert targets[0]["target_id"] == "b"

    def test_filter_by_since(self, store: TargetStore):
        old_ts = time.time() - 7200
        new_ts = time.time()
        store.record_sighting("old", timestamp=old_ts)
        store.record_sighting("new", timestamp=new_ts)
        cutoff = time.time() - 3600
        targets = store.get_all_targets(since=cutoff)
        assert len(targets) == 1
        assert targets[0]["target_id"] == "new"

    def test_combined_filters(self, store: TargetStore):
        store.record_sighting("a", source="ble", alliance="friendly")
        store.record_sighting("b", source="ble", alliance="hostile")
        store.record_sighting("c", source="wifi", alliance="friendly")
        targets = store.get_all_targets(source="ble", alliance="friendly")
        assert len(targets) == 1
        assert targets[0]["target_id"] == "a"


# ------------------------------------------------------------------
# Position history
# ------------------------------------------------------------------


class TestHistory:
    def test_history_recorded(self, store: TargetStore):
        store.record_sighting("h1", position_x=1.0, position_y=2.0, source="ble")
        store.record_sighting("h1", position_x=3.0, position_y=4.0, source="wifi")
        history = store.get_history("h1")
        assert len(history) == 2
        # Newest first
        assert history[0]["x"] == 3.0
        assert history[1]["x"] == 1.0

    def test_no_history_without_position(self, store: TargetStore):
        store.record_sighting("h2", name="No Position")
        history = store.get_history("h2")
        assert history == []

    def test_history_limit(self, store: TargetStore):
        for i in range(10):
            store.record_sighting("h3", position_x=float(i), position_y=0.0)
        history = store.get_history("h3", limit=5)
        assert len(history) == 5

    def test_prune_history(self, store: TargetStore):
        old_ts = time.time() - 7200
        new_ts = time.time()
        store.record_sighting("h4", position_x=1.0, position_y=1.0, timestamp=old_ts)
        store.record_sighting("h4", position_x=2.0, position_y=2.0, timestamp=new_ts)
        cutoff = time.time() - 3600
        pruned = store.prune_history(cutoff)
        assert pruned == 1
        history = store.get_history("h4")
        assert len(history) == 1
        assert history[0]["x"] == 2.0


# ------------------------------------------------------------------
# Full-text search
# ------------------------------------------------------------------


class TestSearch:
    def test_search_by_name(self, store: TargetStore):
        store.record_sighting("s1", name="Red Falcon")
        store.record_sighting("s2", name="Blue Eagle")
        results = store.search("Falcon")
        assert len(results) == 1
        assert results[0]["target_id"] == "s1"

    def test_search_by_target_id(self, store: TargetStore):
        store.record_sighting("drone-x42", name="Drone")
        results = store.search("drone")
        assert len(results) >= 1
        ids = [r["target_id"] for r in results]
        assert "drone-x42" in ids

    def test_search_by_source(self, store: TargetStore):
        store.record_sighting("s3", name="Thing", source="camera-north")
        results = store.search("camera")
        assert len(results) >= 1

    def test_search_empty_query(self, store: TargetStore):
        store.record_sighting("s4", name="Test")
        assert store.search("") == []
        assert store.search("   ") == []

    def test_search_no_results(self, store: TargetStore):
        store.record_sighting("s5", name="Alpha")
        assert store.search("zzzznotfound") == []


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------


class TestStats:
    def test_empty_stats(self, store: TargetStore):
        stats = store.get_stats()
        assert stats["total_targets"] == 0
        assert stats["active_last_hour"] == 0
        assert stats["by_source"] == {}
        assert stats["by_alliance"] == {}
        assert stats["history_count"] == 0

    def test_stats_counts(self, store: TargetStore):
        store.record_sighting("a", source="ble", alliance="friendly",
                              position_x=1.0, position_y=2.0)
        store.record_sighting("b", source="ble", alliance="hostile",
                              position_x=3.0, position_y=4.0)
        store.record_sighting("c", source="wifi", alliance="friendly")
        stats = store.get_stats()
        assert stats["total_targets"] == 3
        assert stats["active_last_hour"] == 3
        assert stats["by_source"] == {"ble": 2, "wifi": 1}
        assert stats["by_alliance"] == {"friendly": 2, "hostile": 1}
        assert stats["history_count"] == 2  # only a and b had positions

    def test_active_last_hour_excludes_old(self, store: TargetStore):
        old_ts = time.time() - 7200  # 2 hours ago
        store.record_sighting("old", timestamp=old_ts)
        store.record_sighting("new")
        stats = store.get_stats()
        assert stats["total_targets"] == 2
        assert stats["active_last_hour"] == 1
