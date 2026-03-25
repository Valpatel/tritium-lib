# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.vehicle_tracker."""

import time
import pytest
from tritium_lib.tracking.vehicle_tracker import (
    VehicleBehavior,
    VehicleTrackingManager,
    VEHICLE_CLASSES,
    STOPPED_SPEED_MPH,
)


# --- VehicleBehavior ---

def test_initial_state():
    vb = VehicleBehavior("car-1", "car")
    assert vb.target_id == "car-1"
    assert vb.speed_mph == 0.0
    assert vb.heading == 0.0
    assert vb.is_moving is False
    assert vb.is_parked is False
    assert len(vb.positions) == 0


def test_update_computes_speed():
    vb = VehicleBehavior("car-1")
    ts = 1000.0
    vb.update(0, 0, ts)
    vb.update(10, 0, ts + 1)  # 10 m/s = ~22 mph
    assert vb.speed_mph > 20
    assert vb.is_moving is True


def test_update_computes_heading():
    vb = VehicleBehavior("car-1")
    ts = 1000.0
    vb.update(0, 0, ts)
    vb.update(10, 0, ts + 1)  # moving east (positive x, zero y)
    # atan2(dx=10, dy=0) = 90 degrees (east)
    assert 85 < vb.heading < 95


def test_stopped_vehicle():
    vb = VehicleBehavior("car-1")
    ts = 1000.0
    vb.update(0, 0, ts)
    vb.update(0.01, 0, ts + 1)  # barely moved
    assert vb.speed_mph < STOPPED_SPEED_MPH
    assert vb.is_moving is False


def test_direction_label():
    vb = VehicleBehavior("car-1")
    ts = 1000.0
    vb.update(0, 0, ts)
    vb.update(0, 10, ts + 1)  # north
    assert vb.direction_label == "N"


def test_suspicious_score_zero_for_moving():
    vb = VehicleBehavior("car-1")
    ts = 1000.0
    vb.update(0, 0, ts)
    vb.update(50, 0, ts + 1)  # fast
    score = vb.get_suspicious_score()
    assert score == 0.0


def test_trail_max_length():
    vb = VehicleBehavior("car-1")
    for i in range(30):
        vb.update(float(i), 0, float(1000 + i))
    assert len(vb.positions) == 20  # MAX_TRAIL_LENGTH


def test_to_dict():
    vb = VehicleBehavior("car-1", "truck")
    vb.update(0, 0, 1000.0)
    vb.update(10, 0, 1001.0)
    d = vb.to_dict()
    assert d["target_id"] == "car-1"
    assert d["vehicle_class"] == "truck"
    assert "speed_mph" in d
    assert "trail" in d


def test_to_vehicle_track():
    vb = VehicleBehavior("car-1")
    vb.update(0, 0, 1000.0)
    vb.update(10, 5, 1001.0)
    track = vb.to_vehicle_track()
    assert track.target_id == "car-1"
    assert track.speed_mph > 0


def test_speed_variance():
    vb = VehicleBehavior("car-1")
    ts = 1000.0
    # Constant speed
    for i in range(10):
        vb.update(float(i * 10), 0, ts + i)
    variance = vb.speed_variance
    assert variance < 1.0  # low variance for constant speed


# --- VehicleTrackingManager ---

def test_manager_update_creates():
    mgr = VehicleTrackingManager()
    vb = mgr.update_vehicle("car-1", 0, 0, "car", 1000.0)
    assert mgr.count == 1
    assert vb.target_id == "car-1"


def test_manager_get_vehicle():
    mgr = VehicleTrackingManager()
    mgr.update_vehicle("car-1", 0, 0, "car", 1000.0)
    assert mgr.get_vehicle("car-1") is not None
    assert mgr.get_vehicle("car-999") is None


def test_manager_remove():
    mgr = VehicleTrackingManager()
    mgr.update_vehicle("car-1", 0, 0, "car", 1000.0)
    mgr.remove("car-1")
    assert mgr.count == 0


def test_manager_get_all():
    mgr = VehicleTrackingManager()
    mgr.update_vehicle("car-1", 0, 0, "car", 1000.0)
    mgr.update_vehicle("car-2", 0, 0, "truck", 1000.0)
    assert len(mgr.get_all()) == 2


def test_manager_summary():
    mgr = VehicleTrackingManager()
    ts = 1000.0
    mgr.update_vehicle("car-1", 0, 0, "car", ts)
    mgr.update_vehicle("car-1", 50, 0, "car", ts + 1)  # moving
    mgr.update_vehicle("car-2", 0, 0, "car", ts)  # stopped
    summary = mgr.get_summary()
    assert summary["total"] == 2
    assert summary["moving"] == 1
    assert summary["stopped"] == 1


def test_manager_eviction():
    mgr = VehicleTrackingManager(max_vehicles=2)
    mgr.update_vehicle("car-1", 0, 0, "car", 1000.0)
    mgr.update_vehicle("car-2", 0, 0, "car", 1001.0)
    mgr.update_vehicle("car-3", 0, 0, "car", 1002.0)  # evicts oldest
    assert mgr.count == 2
    assert mgr.get_vehicle("car-1") is None  # evicted


# --- Constants ---

def test_vehicle_classes():
    assert "car" in VEHICLE_CLASSES
    assert "truck" in VEHICLE_CLASSES
    assert "bus" in VEHICLE_CLASSES


def test_heading_change_rate_insufficient_data():
    vb = VehicleBehavior("car-1")
    assert vb.heading_change_rate == 0.0


def test_heading_change_rate_with_turns():
    vb = VehicleBehavior("car-1")
    ts = 1000.0
    # Drive north then turn east
    vb.update(0, 0, ts)
    vb.update(0, 10, ts + 1)
    vb.update(10, 10, ts + 2)
    assert vb.heading_change_rate > 0


def test_suspicious_slow_crawl():
    vb = VehicleBehavior("car-1")
    ts = 1000.0
    vb.update(0, 0, ts)
    # Move slowly: ~3 m/s = ~6.7 mph (between STOPPED and 10mph)
    vb.update(3, 0, ts + 1)
    score = vb.get_suspicious_score()
    assert score > 0


def test_suspicious_unusual_location():
    vb = VehicleBehavior("car-1")
    ts = 1000.0
    vb.update(0, 0, ts)
    vb.update(0.01, 0, ts + 40)  # stopped for 40s
    score = vb.get_suspicious_score(is_unusual_location=True)
    assert score > 0


def test_manager_get_stopped_and_parked():
    mgr = VehicleTrackingManager()
    ts = 1000.0
    # Moving vehicle
    mgr.update_vehicle("car-1", 0, 0, "car", ts)
    mgr.update_vehicle("car-1", 50, 0, "car", ts + 1)
    # Stopped vehicle
    mgr.update_vehicle("car-2", 0, 0, "car", ts)
    mgr.update_vehicle("car-2", 0.001, 0, "car", ts + 1)
    stopped = mgr.get_stopped()
    assert any(v.target_id == "car-2" for v in stopped)
    assert not any(v.target_id == "car-1" for v in stopped)
