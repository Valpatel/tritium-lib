# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for vehicle tracking models."""

import pytest

from tritium_lib.models.vehicle import (
    VehicleTrack,
    compute_heading,
    compute_speed_mph,
    compute_suspicious_score,
    heading_to_label,
)


class TestVehicleTrack:
    def test_create_basic(self):
        vt = VehicleTrack(target_id="det_car_1")
        assert vt.target_id == "det_car_1"
        assert vt.speed_mph == 0.0
        assert vt.suspicious_score == 0.0
        assert vt.vehicle_class == "car"

    def test_direction_label_auto(self):
        vt = VehicleTrack(target_id="det_car_1", heading=90.0)
        assert vt.direction_label == "E"

    def test_is_suspicious(self):
        vt = VehicleTrack(target_id="det_car_1", suspicious_score=0.7)
        assert vt.is_suspicious()
        assert not vt.is_suspicious(threshold=0.8)

    def test_is_moving(self):
        vt = VehicleTrack(target_id="det_car_1", speed_mph=35.0)
        assert vt.is_moving()
        vt2 = VehicleTrack(target_id="det_car_2", speed_mph=1.0)
        assert not vt2.is_moving()

    def test_to_dict(self):
        vt = VehicleTrack(
            target_id="det_truck_1",
            speed_mph=55.0,
            heading=180.0,
            vehicle_class="truck",
        )
        d = vt.to_dict()
        assert d["target_id"] == "det_truck_1"
        assert d["speed_mph"] == 55.0
        assert d["vehicle_class"] == "truck"
        assert d["direction_label"] == "S"

    def test_parked_state(self):
        vt = VehicleTrack(
            target_id="det_car_1",
            speed_mph=0.0,
            stopped_duration_s=120.0,
            is_parked=True,
        )
        assert vt.is_parked
        assert not vt.is_moving()


class TestHeadingToLabel:
    def test_north(self):
        assert heading_to_label(0) == "N"
        assert heading_to_label(360) == "N"

    def test_east(self):
        assert heading_to_label(90) == "E"

    def test_south(self):
        assert heading_to_label(180) == "S"

    def test_west(self):
        assert heading_to_label(270) == "W"

    def test_northeast(self):
        assert heading_to_label(45) == "NE"

    def test_southwest(self):
        assert heading_to_label(225) == "SW"


class TestComputeSpeedMph:
    def test_stationary(self):
        assert compute_speed_mph((0, 0), (0, 0), 1.0) == 0.0

    def test_zero_time(self):
        assert compute_speed_mph((0, 0), (10, 0), 0.0) == 0.0

    def test_10_meters_per_second(self):
        """10 m/s = ~22.37 mph."""
        speed = compute_speed_mph((0, 0), (10, 0), 1.0)
        assert 22.0 < speed < 23.0

    def test_highway_speed(self):
        """~30 m/s = ~67 mph."""
        speed = compute_speed_mph((0, 0), (30, 0), 1.0)
        assert 66.0 < speed < 68.0


class TestComputeHeading:
    def test_north(self):
        h = compute_heading((0, 0), (0, 10))
        assert h < 1 or h > 359  # ~0 degrees (north)

    def test_east(self):
        h = compute_heading((0, 0), (10, 0))
        assert 89 < h < 91  # ~90 degrees (east)

    def test_south(self):
        h = compute_heading((0, 0), (0, -10))
        assert 179 < h < 181

    def test_west(self):
        h = compute_heading((0, 0), (-10, 0))
        assert 269 < h < 271

    def test_stationary(self):
        assert compute_heading((5, 5), (5, 5)) == 0.0


class TestComputeSuspiciousScore:
    def test_normal_driving(self):
        score = compute_suspicious_score(
            speed_mph=45.0,
            stopped_duration_s=0.0,
        )
        assert score == 0.0

    def test_loitering(self):
        score = compute_suspicious_score(
            speed_mph=0.0,
            stopped_duration_s=400.0,
        )
        assert score >= 0.3

    def test_unusual_location_amplifier(self):
        normal = compute_suspicious_score(
            speed_mph=0.0,
            stopped_duration_s=60.0,
        )
        unusual = compute_suspicious_score(
            speed_mph=0.0,
            stopped_duration_s=60.0,
            is_unusual_location=True,
        )
        assert unusual > normal

    def test_slow_crawling(self):
        score = compute_suspicious_score(
            speed_mph=5.0,
            stopped_duration_s=0.0,
        )
        assert score >= 0.15

    def test_erratic_speed(self):
        score = compute_suspicious_score(
            speed_mph=30.0,
            stopped_duration_s=0.0,
            speed_variance=150.0,
        )
        assert score >= 0.15

    def test_max_clamped(self):
        score = compute_suspicious_score(
            speed_mph=5.0,
            stopped_duration_s=600.0,
            is_unusual_location=True,
            speed_variance=200.0,
            heading_change_rate=50.0,
        )
        assert score <= 1.0
