# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for MovementAnalytics and FleetMetrics models."""

import pytest
from datetime import datetime, timezone
from tritium_lib.models.movement_analytics import (
    ActivityPeriod,
    DwellTime,
    FleetMetrics,
    MovementAnalytics,
)


class TestActivityPeriod:
    def test_to_dict(self):
        ap = ActivityPeriod(start_epoch=100.0, end_epoch=200.0, avg_speed_mps=1.5, distance_m=150.0)
        d = ap.to_dict()
        assert d["start_epoch"] == 100.0
        assert d["duration_s"] == 100.0
        assert d["avg_speed_mps"] == 1.5

    def test_roundtrip(self):
        ap = ActivityPeriod(start_epoch=50.0, end_epoch=150.0, avg_speed_mps=2.0, distance_m=200.0)
        restored = ActivityPeriod.from_dict(ap.to_dict())
        assert restored.start_epoch == ap.start_epoch
        assert restored.distance_m == ap.distance_m


class TestDwellTime:
    def test_to_dict(self):
        dw = DwellTime(zone_id="z1", zone_name="Lobby", total_seconds=300.0, entry_count=3)
        d = dw.to_dict()
        assert d["zone_name"] == "Lobby"
        assert d["entry_count"] == 3

    def test_roundtrip(self):
        dw = DwellTime(zone_id="z2", zone_name="Parking", total_seconds=600.0, entry_count=5)
        restored = DwellTime.from_dict(dw.to_dict())
        assert restored.zone_id == dw.zone_id
        assert restored.total_seconds == dw.total_seconds


class TestMovementAnalytics:
    def test_defaults(self):
        ma = MovementAnalytics(target_id="ble_abc123")
        assert ma.target_id == "ble_abc123"
        assert ma.is_stationary is True
        assert ma.avg_speed_mps == 0.0
        assert len(ma.direction_histogram) == 8
        assert ma.generated_at is not None

    def test_to_dict(self):
        ma = MovementAnalytics(
            target_id="det_person_1",
            avg_speed_mps=1.2,
            max_speed_mps=3.5,
            total_distance_m=500.0,
            current_speed_mps=1.0,
            current_heading_deg=45.0,
            is_stationary=False,
        )
        d = ma.to_dict()
        assert d["target_id"] == "det_person_1"
        assert d["avg_speed_mps"] == 1.2
        assert d["is_stationary"] is False
        assert "direction_histogram" in d

    def test_roundtrip(self):
        ma = MovementAnalytics(
            target_id="wifi_test",
            avg_speed_mps=0.5,
            max_speed_mps=2.0,
            total_distance_m=100.0,
            dwell_times=[DwellTime(zone_id="z1", zone_name="Entry", total_seconds=60.0)],
            activity_periods=[ActivityPeriod(start_epoch=10.0, end_epoch=70.0, avg_speed_mps=0.5)],
        )
        restored = MovementAnalytics.from_dict(ma.to_dict())
        assert restored.target_id == "wifi_test"
        assert len(restored.dwell_times) == 1
        assert len(restored.activity_periods) == 1
        assert restored.dwell_times[0].zone_name == "Entry"

    def test_direction_histogram_init(self):
        ma = MovementAnalytics()
        assert set(ma.direction_histogram.keys()) == {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}


class TestFleetMetrics:
    def test_defaults(self):
        fm = FleetMetrics()
        assert fm.total_targets == 0
        assert fm.moving_targets == 0

    def test_to_dict(self):
        fm = FleetMetrics(total_targets=10, moving_targets=3, avg_fleet_speed_mps=1.5)
        d = fm.to_dict()
        assert d["total_targets"] == 10
        assert d["moving_targets"] == 3

    def test_roundtrip(self):
        fm = FleetMetrics(total_targets=5, busiest_zone="Lobby", dominant_direction="N")
        restored = FleetMetrics.from_dict(fm.to_dict())
        assert restored.busiest_zone == "Lobby"
        assert restored.dominant_direction == "N"

    def test_from_analytics(self):
        analytics = [
            MovementAnalytics(
                target_id="t1",
                avg_speed_mps=1.0,
                max_speed_mps=2.0,
                total_distance_m=100.0,
                current_speed_mps=1.0,
                is_stationary=False,
                dwell_times=[DwellTime(zone_id="z1", zone_name="Lobby", total_seconds=60.0)],
                direction_histogram={"N": 0.5, "NE": 0.0, "E": 0.3, "SE": 0.0, "S": 0.2, "SW": 0.0, "W": 0.0, "NW": 0.0},
            ),
            MovementAnalytics(
                target_id="t2",
                avg_speed_mps=0.0,
                max_speed_mps=0.5,
                total_distance_m=10.0,
                current_speed_mps=0.0,
                is_stationary=True,
                dwell_times=[DwellTime(zone_id="z1", zone_name="Lobby", total_seconds=120.0)],
                direction_histogram={"N": 0.1, "NE": 0.0, "E": 0.0, "SE": 0.0, "S": 0.9, "SW": 0.0, "W": 0.0, "NW": 0.0},
            ),
        ]
        fm = FleetMetrics.from_analytics(analytics)
        assert fm.total_targets == 2
        assert fm.moving_targets == 1
        assert fm.stationary_targets == 1
        assert fm.total_fleet_distance_m == 110.0
        assert fm.busiest_zone == "Lobby"
        assert fm.avg_fleet_speed_mps == 1.0  # only 1 moving target at speed 1.0

    def test_from_analytics_empty(self):
        fm = FleetMetrics.from_analytics([])
        assert fm.total_targets == 0

    def test_import_from_init(self):
        """Verify we can import from the top-level models package."""
        from tritium_lib.models import MovementAnalytics, FleetMetrics, ActivityPeriod, DwellTime
        assert MovementAnalytics is not None
        assert FleetMetrics is not None
