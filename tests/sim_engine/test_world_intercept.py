# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.world.intercept — proportional navigation math."""

import math
import pytest

from tritium_lib.sim_engine.world.intercept import (
    predict_intercept,
    lead_target,
    time_to_intercept,
    target_velocity,
    _UNCATCHABLE_TIME,
)


class TestTargetVelocity:
    """Tests for heading+speed to velocity conversion."""

    def test_north(self):
        vx, vy = target_velocity(0.0, 10.0)
        assert abs(vx) < 1e-6
        assert abs(vy - 10.0) < 1e-6

    def test_east(self):
        vx, vy = target_velocity(90.0, 10.0)
        assert abs(vx - 10.0) < 1e-6
        assert abs(vy) < 1e-6

    def test_south(self):
        vx, vy = target_velocity(180.0, 5.0)
        assert abs(vx) < 1e-6
        assert abs(vy - (-5.0)) < 1e-6

    def test_west(self):
        vx, vy = target_velocity(270.0, 5.0)
        assert abs(vx - (-5.0)) < 1e-6
        assert abs(vy) < 1e-6

    def test_zero_speed(self):
        vx, vy = target_velocity(45.0, 0.0)
        assert vx == 0.0
        assert vy == 0.0

    def test_negative_speed(self):
        vx, vy = target_velocity(45.0, -1.0)
        assert vx == 0.0
        assert vy == 0.0


class TestPredictIntercept:
    """Tests for the quadratic intercept solver."""

    def test_stationary_target(self):
        """Pursuer should intercept at target's position."""
        ix, iy = predict_intercept(
            pursuer_pos=(0.0, 0.0),
            pursuer_speed=10.0,
            target_pos=(30.0, 40.0),
            target_vel=(0.0, 0.0),
        )
        assert abs(ix - 30.0) < 1e-3
        assert abs(iy - 40.0) < 1e-3

    def test_moving_target_interceptable(self):
        """Fast pursuer should intercept ahead of moving target."""
        ix, iy = predict_intercept(
            pursuer_pos=(0.0, 0.0),
            pursuer_speed=20.0,
            target_pos=(50.0, 0.0),
            target_vel=(5.0, 0.0),
        )
        # Intercept should be ahead of target's current position
        assert ix > 50.0

    def test_already_at_target(self):
        """Zero distance should return target position."""
        ix, iy = predict_intercept(
            pursuer_pos=(10.0, 20.0),
            pursuer_speed=5.0,
            target_pos=(10.0, 20.0),
            target_vel=(3.0, 0.0),
        )
        assert abs(ix - 10.0) < 1e-3
        assert abs(iy - 20.0) < 1e-3

    def test_uncatchable_falls_back(self):
        """When target is uncatchable, return current target position."""
        ix, iy = predict_intercept(
            pursuer_pos=(0.0, 0.0),
            pursuer_speed=1.0,  # very slow
            target_pos=(100.0, 0.0),
            target_vel=(50.0, 0.0),  # very fast, running away
        )
        # Should fall back to target position
        assert abs(ix - 100.0) < 1e-3
        assert abs(iy - 0.0) < 1e-3


class TestLeadTarget:
    """Tests for projectile lead targeting."""

    def test_stationary_target(self):
        """Lead point for stationary target is the target itself."""
        lx, ly = lead_target(
            shooter_pos=(0.0, 0.0),
            target_pos=(20.0, 0.0),
            target_vel=(0.0, 0.0),
            projectile_speed=100.0,
        )
        assert abs(lx - 20.0) < 1e-3
        assert abs(ly - 0.0) < 1e-3

    def test_lead_ahead(self):
        """Lead point should be ahead of a moving target."""
        lx, ly = lead_target(
            shooter_pos=(0.0, 0.0),
            target_pos=(50.0, 0.0),
            target_vel=(10.0, 0.0),
            projectile_speed=100.0,
        )
        # Lead point should be ahead of current position
        assert lx > 50.0


class TestTimeToIntercept:
    """Tests for intercept time estimation."""

    def test_stationary_target(self):
        t = time_to_intercept(
            pursuer_pos=(0.0, 0.0),
            pursuer_speed=10.0,
            target_pos=(50.0, 0.0),
            target_vel=(0.0, 0.0),
        )
        # 50m at 10m/s = 5s
        assert abs(t - 5.0) < 1e-3

    def test_uncatchable(self):
        t = time_to_intercept(
            pursuer_pos=(0.0, 0.0),
            pursuer_speed=0.0,  # can't move
            target_pos=(50.0, 0.0),
            target_vel=(10.0, 0.0),
        )
        assert t == _UNCATCHABLE_TIME

    def test_already_at_target(self):
        t = time_to_intercept(
            pursuer_pos=(5.0, 5.0),
            pursuer_speed=10.0,
            target_pos=(5.0, 5.0),
            target_vel=(1.0, 0.0),
        )
        assert t == 0.0

    def test_positive_time(self):
        t = time_to_intercept(
            pursuer_pos=(0.0, 0.0),
            pursuer_speed=20.0,
            target_pos=(30.0, 0.0),
            target_vel=(5.0, 0.0),
        )
        assert t > 0.0
        assert t < _UNCATCHABLE_TIME
