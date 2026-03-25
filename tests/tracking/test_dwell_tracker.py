# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.dwell_tracker."""

import time
import pytest

from tritium_lib.tracking.dwell_tracker import DwellTracker
from tritium_lib.tracking.target_tracker import TargetTracker, TrackedTarget
from tritium_lib.models.dwell import DwellSeverity, DwellState, classify_dwell_severity


class FakeEventBus:
    def __init__(self):
        self.events = []

    def publish(self, topic, data=None):
        self.events.append((topic, data))


def _make_target(target_id, position=(10.0, 10.0), name="Test", alliance="unknown", asset_type="person"):
    return TrackedTarget(
        target_id=target_id,
        name=name,
        alliance=alliance,
        asset_type=asset_type,
        position=position,
        source="ble",
        last_seen=time.monotonic(),
        first_seen=time.monotonic(),
    )


class TestDwellTrackerInit:
    def test_defaults(self):
        bus = FakeEventBus()
        tracker = TargetTracker()
        dt = DwellTracker(bus, tracker)
        assert dt.active_dwells == []
        assert dt.history == []

    def test_custom_threshold(self):
        bus = FakeEventBus()
        tracker = TargetTracker()
        dt = DwellTracker(bus, tracker, threshold_s=60.0, radius_m=5.0)
        assert dt._threshold_s == 60.0
        assert dt._radius_m == 5.0


class TestCheckTarget:
    def test_first_check_sets_anchor(self):
        bus = FakeEventBus()
        tracker = TargetTracker()
        dt = DwellTracker(bus, tracker, threshold_s=10.0)
        now = time.time()
        dt._check_target("t1", 10.0, 10.0, now, "Phone", "unknown", "person")
        assert "t1" in dt._tracking
        assert dt._tracking["t1"]["anchor_x"] == 10.0

    def test_movement_resets_anchor(self):
        bus = FakeEventBus()
        tracker = TargetTracker()
        dt = DwellTracker(bus, tracker, threshold_s=10.0, radius_m=5.0)
        now = time.time()
        dt._check_target("t1", 10.0, 10.0, now, "Phone", "unknown", "person")
        dt._check_target("t1", 100.0, 100.0, now + 1, "Phone", "unknown", "person")
        assert dt._tracking["t1"]["anchor_x"] == 100.0

    def test_dwell_detected_after_threshold(self):
        bus = FakeEventBus()
        tracker = TargetTracker()
        dt = DwellTracker(bus, tracker, threshold_s=5.0, radius_m=15.0)
        now = time.time()
        dt._check_target("t1", 10.0, 10.0, now, "Phone", "unknown", "person")
        # Stay in same spot past threshold
        dt._check_target("t1", 10.1, 10.1, now + 6.0, "Phone", "unknown", "person")
        assert "t1" in dt._active_dwells
        assert len(bus.events) >= 1
        assert bus.events[0][0] == "dwell_start"

    def test_dwell_update_on_continued_dwell(self):
        bus = FakeEventBus()
        tracker = TargetTracker()
        dt = DwellTracker(bus, tracker, threshold_s=5.0, radius_m=15.0)
        now = time.time()
        dt._check_target("t1", 10.0, 10.0, now, "Phone", "unknown", "person")
        dt._check_target("t1", 10.1, 10.1, now + 6.0, "Phone", "unknown", "person")
        dt._check_target("t1", 10.2, 10.2, now + 12.0, "Phone", "unknown", "person")
        update_events = [e for e in bus.events if e[0] == "dwell_update"]
        assert len(update_events) >= 1


class TestMultipleTargets:
    def test_track_multiple(self):
        bus = FakeEventBus()
        tracker = TargetTracker()
        dt = DwellTracker(bus, tracker, threshold_s=5.0, radius_m=15.0)
        now = time.time()
        dt._check_target("t1", 10.0, 10.0, now, "A", "unknown", "person")
        dt._check_target("t2", 50.0, 50.0, now, "B", "unknown", "vehicle")
        assert len(dt._tracking) == 2

    def test_only_dwelling_targets_get_events(self):
        bus = FakeEventBus()
        tracker = TargetTracker()
        dt = DwellTracker(bus, tracker, threshold_s=5.0, radius_m=15.0)
        now = time.time()
        dt._check_target("dweller", 10.0, 10.0, now, "A", "unknown", "person")
        dt._check_target("mover", 50.0, 50.0, now, "B", "unknown", "person")
        # Dweller stays, mover moves
        dt._check_target("dweller", 10.1, 10.1, now + 6.0, "A", "unknown", "person")
        dt._check_target("mover", 200.0, 200.0, now + 6.0, "B", "unknown", "person")
        assert "dweller" in dt._active_dwells
        assert "mover" not in dt._active_dwells


class TestSeverityClassification:
    def test_normal(self):
        assert classify_dwell_severity(300) == DwellSeverity.NORMAL

    def test_extended(self):
        assert classify_dwell_severity(1800) == DwellSeverity.EXTENDED

    def test_prolonged(self):
        assert classify_dwell_severity(7200) == DwellSeverity.PROLONGED

    def test_critical(self):
        assert classify_dwell_severity(14400) == DwellSeverity.CRITICAL


class TestEndDwell:
    def test_end_dwell_moves_to_history(self):
        bus = FakeEventBus()
        tracker = TargetTracker()
        dt = DwellTracker(bus, tracker, threshold_s=5.0, radius_m=15.0)
        now = time.time()
        dt._check_target("t1", 10.0, 10.0, now, "Phone", "unknown", "person")
        dt._check_target("t1", 10.1, 10.1, now + 6.0, "Phone", "unknown", "person")
        # Move away to end dwell
        dt._check_target("t1", 100.0, 100.0, now + 10.0, "Phone", "unknown", "person")
        assert "t1" not in dt._active_dwells
        assert len(dt.history) == 1
        end_events = [e for e in bus.events if e[0] == "dwell_end"]
        assert len(end_events) == 1


class TestEviction:
    def test_gone_target_ends_dwell(self):
        bus = FakeEventBus()
        tracker = TargetTracker()
        dt = DwellTracker(bus, tracker, threshold_s=5.0, radius_m=15.0)
        now = time.time()
        # Set up a dwell
        dt._check_target("t1", 10.0, 10.0, now, "P", "unknown", "person")
        dt._check_target("t1", 10.1, 10.1, now + 6.0, "P", "unknown", "person")
        assert "t1" in dt._active_dwells
        # Simulate target disappearing: _check_all_targets with empty tracker
        dt._check_all_targets()
        assert "t1" not in dt._active_dwells


class TestGetDwellForTarget:
    def test_no_dwell(self):
        bus = FakeEventBus()
        tracker = TargetTracker()
        dt = DwellTracker(bus, tracker)
        assert dt.get_dwell_for_target("t1") is None

    def test_active_dwell(self):
        bus = FakeEventBus()
        tracker = TargetTracker()
        dt = DwellTracker(bus, tracker, threshold_s=5.0, radius_m=15.0)
        now = time.time()
        dt._check_target("t1", 10.0, 10.0, now, "P", "unknown", "person")
        dt._check_target("t1", 10.1, 10.1, now + 6.0, "P", "unknown", "person")
        dwell = dt.get_dwell_for_target("t1")
        assert dwell is not None
        assert dwell.target_id == "t1"
