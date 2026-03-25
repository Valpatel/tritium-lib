# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.core.movement — MovementController and smooth_path."""

import math
import pytest

from tritium_lib.sim_engine.core.movement import (
    MovementController,
    smooth_path,
    _angle_diff,
)


class TestAngleDiff:
    """Test the internal angle diff helper."""

    def test_same_angle(self):
        assert _angle_diff(90.0, 90.0) == 0.0

    def test_clockwise_90(self):
        diff = _angle_diff(0.0, 90.0)
        assert abs(diff - 90.0) < 0.01

    def test_counterclockwise_90(self):
        diff = _angle_diff(90.0, 0.0)
        assert abs(diff - (-90.0)) < 0.01

    def test_wrap_around_positive(self):
        diff = _angle_diff(350.0, 10.0)
        assert abs(diff - 20.0) < 0.01

    def test_wrap_around_negative(self):
        diff = _angle_diff(10.0, 350.0)
        assert abs(diff - (-20.0)) < 0.01

    def test_180_degrees(self):
        diff = _angle_diff(0.0, 180.0)
        assert abs(abs(diff) - 180.0) < 0.01


class TestMovementControllerConstruction:
    def test_defaults(self):
        mc = MovementController()
        assert mc.max_speed == 2.0
        assert mc.turn_rate == 180.0
        assert mc.acceleration == 4.0
        assert mc.deceleration == 6.0
        assert mc.x == 0.0
        assert mc.y == 0.0
        assert mc.arrived

    def test_custom_values(self):
        mc = MovementController(max_speed=5.0, turn_rate=360.0, x=10.0, y=20.0)
        assert mc.max_speed == 5.0
        assert mc.turn_rate == 360.0
        assert mc.x == 10.0
        assert mc.y == 20.0


class TestMovementControllerSetPath:
    def test_set_path_not_arrived(self):
        mc = MovementController()
        mc.set_path([(10.0, 0.0), (20.0, 0.0)])
        assert not mc.arrived
        assert mc.remaining_waypoints == 2

    def test_set_empty_path_arrived(self):
        mc = MovementController()
        mc.set_path([])
        assert mc.arrived

    def test_set_destination(self):
        mc = MovementController()
        mc.set_destination(50.0, 50.0)
        assert not mc.arrived
        assert mc.current_waypoint == (50.0, 50.0)

    def test_stop_clears_path(self):
        mc = MovementController()
        mc.set_path([(10.0, 0.0), (20.0, 0.0)])
        mc.stop()
        assert mc.arrived
        assert mc.remaining_waypoints == 0


class TestMovementControllerTick:
    def test_tick_moves_toward_waypoint(self):
        mc = MovementController(max_speed=10.0, acceleration=100.0)
        mc.set_destination(100.0, 0.0)
        mc.tick(0.1)
        mc.tick(0.1)
        mc.tick(0.1)
        assert mc.x > 0.0

    def test_tick_arrives_at_destination(self):
        mc = MovementController(max_speed=100.0, acceleration=1000.0)
        mc.set_destination(1.0, 0.0)
        for _ in range(100):
            mc.tick(0.1)
        assert mc.arrived

    def test_tick_follows_multi_waypoint_path(self):
        mc = MovementController(max_speed=50.0, acceleration=500.0)
        mc.set_path([(10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        for _ in range(200):
            mc.tick(0.1)
        assert mc.arrived

    def test_tick_patrol_loop_never_arrives(self):
        mc = MovementController(max_speed=50.0, acceleration=500.0)
        mc.set_path([(5.0, 0.0), (5.0, 5.0), (0.0, 5.0)], loop=True)
        for _ in range(500):
            mc.tick(0.1)
        assert not mc.arrived, "Loop patrol should never mark arrived"

    def test_tick_decelerates_when_arrived(self):
        mc = MovementController(max_speed=10.0, acceleration=100.0)
        mc.speed = 5.0
        # No path — should decelerate
        mc.tick(0.1)
        assert mc.speed < 5.0

    def test_speed_never_negative(self):
        mc = MovementController(max_speed=10.0, deceleration=100.0)
        mc.speed = 1.0
        mc.tick(1.0)
        assert mc.speed >= 0.0

    def test_heading_stays_in_0_360(self):
        mc = MovementController(max_speed=10.0, acceleration=100.0, turn_rate=720.0)
        mc.set_destination(-10.0, -10.0)
        for _ in range(50):
            mc.tick(0.1)
        assert 0.0 <= mc.heading < 360.0


class TestSmoothPath:
    def test_short_path_unchanged(self):
        path = [(0.0, 0.0), (10.0, 10.0)]
        result = smooth_path(path)
        assert result == path

    def test_single_point_unchanged(self):
        result = smooth_path([(5.0, 5.0)])
        assert len(result) == 1

    def test_collinear_points_removed(self):
        # Three collinear points — middle should be removed
        path = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        result = smooth_path(path, tolerance=0.5)
        assert len(result) == 2
        assert result[0] == (0.0, 0.0)
        assert result[-1] == (10.0, 0.0)

    def test_non_collinear_points_kept(self):
        # Sharp turn — middle point should be kept
        path = [(0.0, 0.0), (5.0, 10.0), (10.0, 0.0)]
        result = smooth_path(path, tolerance=0.5)
        assert len(result) == 3

    def test_empty_path(self):
        result = smooth_path([])
        assert result == []

    def test_preserves_endpoints(self):
        path = [(0.0, 0.0), (3.0, 0.01), (6.0, 0.0), (10.0, 5.0)]
        result = smooth_path(path, tolerance=0.5)
        assert result[0] == (0.0, 0.0)
        assert result[-1] == (10.0, 5.0)
