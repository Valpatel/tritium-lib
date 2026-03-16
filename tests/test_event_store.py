# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the EventStore — tactical event persistence."""

import time
import pytest

from tritium_lib.store.event_store import EventStore, TacticalEvent, SEVERITY_LEVELS


@pytest.fixture
def store():
    """In-memory EventStore for testing."""
    s = EventStore(":memory:")
    yield s
    s.close()


class TestEventStoreRecord:
    """Test recording events."""

    def test_record_returns_event_id(self, store):
        eid = store.record("target_detected", source="ble_scanner")
        assert eid
        assert len(eid) == 36  # UUID length

    def test_record_with_all_fields(self, store):
        eid = store.record(
            "alert_raised",
            severity="warning",
            source="automation",
            target_id="ble_AA:BB:CC:DD:EE:FF",
            operator="admin",
            summary="Unknown device detected in restricted zone",
            data={"rssi": -45, "zone": "north_perimeter"},
            position_lat=40.7128,
            position_lng=-74.0060,
            site_id="hq",
        )
        event = store.get_event(eid)
        assert event is not None
        assert event.event_type == "alert_raised"
        assert event.severity == "warning"
        assert event.source == "automation"
        assert event.target_id == "ble_AA:BB:CC:DD:EE:FF"
        assert event.operator == "admin"
        assert event.summary == "Unknown device detected in restricted zone"
        assert event.data["rssi"] == -45
        assert event.position_lat == pytest.approx(40.7128)
        assert event.position_lng == pytest.approx(-74.0060)
        assert event.site_id == "hq"

    def test_record_with_custom_timestamp(self, store):
        ts = 1700000000.0
        eid = store.record("test_event", timestamp=ts)
        event = store.get_event(eid)
        assert event.timestamp == ts

    def test_record_with_custom_event_id(self, store):
        eid = store.record("test_event", event_id="custom-id-123")
        assert eid == "custom-id-123"
        event = store.get_event("custom-id-123")
        assert event is not None

    def test_record_batch(self, store):
        events = [
            TacticalEvent(event_type="det_1", severity="info", source="ble"),
            TacticalEvent(event_type="det_2", severity="warning", source="yolo"),
            TacticalEvent(event_type="det_3", severity="error", source="mesh"),
        ]
        count = store.record_batch(events)
        assert count == 3
        assert store.count() == 3

    def test_record_batch_empty(self, store):
        count = store.record_batch([])
        assert count == 0


class TestEventStoreQueries:
    """Test query methods."""

    def _seed(self, store):
        """Seed the store with test data."""
        now = time.time()
        store.record("target_detected", severity="info", source="ble",
                     target_id="ble_1", timestamp=now - 100)
        store.record("alert_raised", severity="warning", source="automation",
                     target_id="ble_1", timestamp=now - 80)
        store.record("target_detected", severity="info", source="yolo",
                     target_id="det_person_1", timestamp=now - 60)
        store.record("command_sent", severity="info", source="operator",
                     operator="admin", timestamp=now - 40)
        store.record("state_change", severity="error", source="system",
                     timestamp=now - 20)
        store.record("target_lost", severity="critical", source="ble",
                     target_id="ble_1", timestamp=now)
        return now

    def test_query_time_range_all(self, store):
        self._seed(store)
        events = store.query_time_range()
        assert len(events) == 6

    def test_query_time_range_with_bounds(self, store):
        now = self._seed(store)
        events = store.query_time_range(start=now - 50, end=now - 10)
        assert len(events) == 2  # command_sent and state_change

    def test_query_time_range_order(self, store):
        self._seed(store)
        events = store.query_time_range()
        # Most recent first
        for i in range(len(events) - 1):
            assert events[i].timestamp >= events[i + 1].timestamp

    def test_query_by_type(self, store):
        self._seed(store)
        events = store.query_by_type("target_detected")
        assert len(events) == 2
        for e in events:
            assert e.event_type == "target_detected"

    def test_query_by_severity(self, store):
        self._seed(store)
        # warning and above
        events = store.query_by_severity("warning")
        assert len(events) == 3  # warning + error + critical
        for e in events:
            assert e.severity in ("warning", "error", "critical")

    def test_query_by_severity_critical(self, store):
        self._seed(store)
        events = store.query_by_severity("critical")
        assert len(events) == 1
        assert events[0].severity == "critical"

    def test_query_by_target(self, store):
        self._seed(store)
        events = store.query_by_target("ble_1")
        assert len(events) == 3  # detected, alert, lost

    def test_query_by_source(self, store):
        self._seed(store)
        events = store.query_by_source("ble")
        assert len(events) == 2

    def test_get_event_not_found(self, store):
        assert store.get_event("nonexistent") is None

    def test_query_with_limit(self, store):
        self._seed(store)
        events = store.query_time_range(limit=2)
        assert len(events) == 2


class TestEventStoreAggregation:
    """Test count and stats methods."""

    def test_count_all(self, store):
        store.record("a", source="x")
        store.record("b", source="y")
        store.record("c", source="x")
        assert store.count() == 3

    def test_count_filtered(self, store):
        store.record("a", source="x", severity="info")
        store.record("b", source="y", severity="warning")
        store.record("c", source="x", severity="info")
        assert store.count(source="x") == 2
        assert store.count(severity="warning") == 1
        assert store.count(event_type="a") == 1

    def test_get_stats(self, store):
        store.record("target_detected", source="ble", severity="info")
        store.record("alert_raised", source="auto", severity="warning")
        store.record("target_detected", source="ble", severity="info")

        stats = store.get_stats()
        assert stats["total_events"] == 3
        assert stats["by_type"]["target_detected"] == 2
        assert stats["by_severity"]["info"] == 2
        assert stats["by_severity"]["warning"] == 1
        assert stats["by_source"]["ble"] == 2
        assert stats["oldest_event"] is not None
        assert stats["newest_event"] is not None


class TestEventStoreMaintenance:
    """Test cleanup and clear."""

    def test_cleanup_removes_oldest(self, store):
        for i in range(10):
            store.record(f"event_{i}", timestamp=float(i))
        deleted = store.cleanup(keep=5)
        assert deleted == 5
        remaining = store.query_time_range()
        assert len(remaining) == 5
        # Should keep the 5 newest (timestamps 5-9)
        timestamps = [e.timestamp for e in remaining]
        assert min(timestamps) >= 5.0

    def test_cleanup_noop_when_under_limit(self, store):
        store.record("a")
        store.record("b")
        deleted = store.cleanup(keep=100)
        assert deleted == 0

    def test_clear(self, store):
        store.record("a")
        store.record("b")
        store.record("c")
        deleted = store.clear()
        assert deleted == 3
        assert store.count() == 0


class TestTacticalEvent:
    """Test the TacticalEvent dataclass."""

    def test_to_dict(self):
        ev = TacticalEvent(
            event_id="abc",
            timestamp=1000.0,
            event_type="target_detected",
            severity="info",
            source="ble",
            target_id="ble_1",
            data={"rssi": -50},
        )
        d = ev.to_dict()
        assert d["event_id"] == "abc"
        assert d["event_type"] == "target_detected"
        assert d["data"]["rssi"] == -50

    def test_default_values(self):
        ev = TacticalEvent()
        assert ev.event_id == ""
        assert ev.severity == "info"
        assert ev.data == {}
        assert ev.position_lat is None


class TestSeverityLevels:
    """Test severity ordering."""

    def test_severity_order(self):
        assert SEVERITY_LEVELS == ("debug", "info", "warning", "error", "critical")

    def test_severity_index(self):
        assert SEVERITY_LEVELS.index("debug") == 0
        assert SEVERITY_LEVELS.index("critical") == 4
