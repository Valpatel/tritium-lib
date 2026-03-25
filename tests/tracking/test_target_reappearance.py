# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.target_reappearance."""

import time

from tritium_lib.tracking.target_reappearance import (
    DepartureRecord,
    ReappearanceEvent,
    TargetReappearanceMonitor,
    _format_duration,
)


# --- _format_duration ---

def test_format_seconds():
    assert _format_duration(30) == "30s"


def test_format_minutes():
    assert _format_duration(150) == "2m 30s"


def test_format_exact_minutes():
    assert _format_duration(120) == "2m"


def test_format_hours():
    assert _format_duration(3660) == "1h 1m"


def test_format_exact_hours():
    assert _format_duration(7200) == "2h"


# --- DepartureRecord ---

def test_departure_record_defaults():
    r = DepartureRecord(target_id="ble_aabb")
    assert r.target_id == "ble_aabb"
    assert r.name == ""
    assert r.last_position == (0.0, 0.0)


# --- ReappearanceEvent ---

def test_reappearance_event_to_dict():
    e = ReappearanceEvent(
        target_id="ble_aabb",
        name="Phone-1",
        source="ble",
        absence_seconds=300,
        last_position=(10, 20),
        return_position=(30, 40),
    )
    d = e.to_dict()
    assert d["target_id"] == "ble_aabb"
    assert d["absence_seconds"] == 300.0
    assert d["absence_human"] == "5m"
    assert "returned after" in d["message"]
    assert d["last_position"]["x"] == 10
    assert d["return_position"]["y"] == 40


def test_reappearance_event_uses_name_in_message():
    e = ReappearanceEvent(target_id="t-1", name="MyPhone", absence_seconds=90)
    d = e.to_dict()
    assert "MyPhone" in d["message"]


def test_reappearance_event_no_name_uses_id():
    e = ReappearanceEvent(target_id="ble_aabb", name="", absence_seconds=90)
    d = e.to_dict()
    assert "ble_aabb" in d["message"]


# --- TargetReappearanceMonitor ---

def test_monitor_empty_stats():
    m = TargetReappearanceMonitor()
    stats = m.stats
    assert stats["total_departures"] == 0
    assert stats["total_reappearances"] == 0
    assert stats["currently_departed"] == 0


def test_monitor_record_departure():
    m = TargetReappearanceMonitor()
    m.record_departure("t-1", name="Phone", source="ble")
    assert m.stats["total_departures"] == 1
    assert m.stats["currently_departed"] == 1


def test_monitor_get_departed():
    m = TargetReappearanceMonitor()
    m.record_departure("t-1", name="Phone", source="ble", last_position=(5.0, 10.0))
    departed = m.get_departed()
    assert len(departed) == 1
    assert departed[0]["target_id"] == "t-1"
    assert departed[0]["name"] == "Phone"


def test_monitor_reappearance_below_threshold():
    m = TargetReappearanceMonitor(min_absence_seconds=60.0)
    m.record_departure("t-1")
    # Check immediately — absence < 60s
    event = m.check_reappearance("t-1")
    assert event is None


def test_monitor_reappearance_unknown_target():
    m = TargetReappearanceMonitor()
    event = m.check_reappearance("never-departed")
    assert event is None


def test_monitor_reappearance_above_threshold():
    m = TargetReappearanceMonitor(min_absence_seconds=0.0)
    m.record_departure("t-1", name="Phone", source="ble", last_position=(1, 2))
    # With min_absence=0, any reappearance should trigger
    time.sleep(0.01)
    event = m.check_reappearance("t-1", position=(3, 4))
    assert event is not None
    assert event.target_id == "t-1"
    assert event.absence_seconds > 0
    assert m.stats["total_reappearances"] == 1


def test_monitor_departure_removes_on_reappearance():
    m = TargetReappearanceMonitor(min_absence_seconds=0.0)
    m.record_departure("t-1")
    time.sleep(0.01)
    m.check_reappearance("t-1")
    assert m.stats["currently_departed"] == 0


def test_monitor_recent_events():
    m = TargetReappearanceMonitor(min_absence_seconds=0.0)
    m.record_departure("t-1")
    time.sleep(0.01)
    m.check_reappearance("t-1")
    events = m.get_recent_events()
    assert len(events) == 1
    assert events[0]["target_id"] == "t-1"


def test_monitor_eviction_at_max_departures():
    m = TargetReappearanceMonitor(max_tracked_departures=3)
    for i in range(5):
        m.record_departure(f"t-{i}")
    assert m.stats["currently_departed"] == 3
    assert m.stats["total_departures"] == 5


def test_monitor_event_bus_publish():
    published = []

    class FakeEventBus:
        def publish(self, topic, data=None):
            published.append((topic, data))

    m = TargetReappearanceMonitor(event_bus=FakeEventBus(), min_absence_seconds=0.0)
    m.record_departure("t-1")
    time.sleep(0.01)
    m.check_reappearance("t-1")
    assert len(published) == 1
    assert published[0][0] == "target:reappearance"


def test_monitor_inherits_departure_info():
    m = TargetReappearanceMonitor(min_absence_seconds=0.0)
    m.record_departure("t-1", name="OrigName", source="ble", asset_type="phone")
    time.sleep(0.01)
    event = m.check_reappearance("t-1")
    assert event.name == "OrigName"
    assert event.source == "ble"
    assert event.asset_type == "phone"


# --- Deepened tests ---

def test_monitor_multiple_departures_and_reappearances():
    """Multiple targets can depart and reappear independently."""
    m = TargetReappearanceMonitor(min_absence_seconds=0.0)
    m.record_departure("t-1", name="A")
    m.record_departure("t-2", name="B")
    m.record_departure("t-3", name="C")
    time.sleep(0.01)
    e1 = m.check_reappearance("t-1")
    e3 = m.check_reappearance("t-3")
    assert e1 is not None
    assert e3 is not None
    assert m.stats["currently_departed"] == 1  # only t-2 remains
    assert m.stats["total_reappearances"] == 2


def test_monitor_reappearance_overrides_departure_info():
    """If reappearance provides name/source, those override departure record."""
    m = TargetReappearanceMonitor(min_absence_seconds=0.0)
    m.record_departure("t-1", name="OldName", source="ble")
    time.sleep(0.01)
    event = m.check_reappearance("t-1", name="NewName", source="yolo")
    assert event.name == "NewName"
    assert event.source == "yolo"


def test_monitor_threshold_filtering():
    """Absence below threshold should NOT trigger, above should."""
    m = TargetReappearanceMonitor(min_absence_seconds=999.0)
    m.record_departure("t-1")
    time.sleep(0.01)
    event = m.check_reappearance("t-1")
    assert event is None
    # Target should have been removed from _departed even without event
    # Actually: it was popped and absence < threshold, so no event but target is gone
    assert m.stats["currently_departed"] == 0


def test_monitor_double_departure_overwrites():
    """Recording departure for same target twice should overwrite."""
    m = TargetReappearanceMonitor()
    m.record_departure("t-1", name="First")
    m.record_departure("t-1", name="Second")
    assert m.stats["total_departures"] == 2
    assert m.stats["currently_departed"] == 1
    departed = m.get_departed()
    assert departed[0]["name"] == "Second"


def test_monitor_recent_events_limit():
    """Recent events should be capped at max_recent (100)."""
    m = TargetReappearanceMonitor(min_absence_seconds=0.0)
    for i in range(120):
        m.record_departure(f"t-{i}")
    time.sleep(0.01)
    for i in range(120):
        m.check_reappearance(f"t-{i}")
    assert len(m._recent_events) <= 100


def test_monitor_event_bus_receives_correct_data():
    """Event bus data should match the ReappearanceEvent.to_dict() output."""
    published = []

    class FakeEventBus:
        def publish(self, topic, data=None):
            published.append((topic, data))

    m = TargetReappearanceMonitor(
        event_bus=FakeEventBus(), min_absence_seconds=0.0
    )
    m.record_departure("t-1", name="Phone", source="ble", last_position=(10, 20))
    time.sleep(0.01)
    m.check_reappearance("t-1", position=(30, 40))
    assert len(published) == 1
    data = published[0][1]
    assert data["target_id"] == "t-1"
    assert data["last_position"]["x"] == 10
    assert data["return_position"]["x"] == 30


def test_departure_record_full_fields():
    """DepartureRecord stores all fields correctly."""
    r = DepartureRecord(
        target_id="ble_aabb",
        name="TestPhone",
        source="ble",
        asset_type="phone",
        last_position=(15.0, 25.0),
    )
    assert r.name == "TestPhone"
    assert r.source == "ble"
    assert r.asset_type == "phone"
    assert r.last_position == (15.0, 25.0)
