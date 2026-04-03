# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.idm -- Intelligent Driver Model."""

import math

import pytest

from tritium_lib.sim_engine.idm import (
    IDMParams,
    IDM_DEFAULTS,
    IDMStepResult,
    ROAD_SPEEDS,
    VEHICLE_IDM_PROFILES,
    idm_acceleration,
    idm_free_flow,
    idm_step,
    get_idm_for_road,
)


# ===================================================================
# IDMParams
# ===================================================================


class TestIDMParams:
    def test_defaults(self):
        p = IDMParams()
        assert p.v0 == 12.0
        assert p.a == 1.4
        assert p.b == 2.0
        assert p.s0 == 2.0
        assert p.T == 1.5
        assert p.delta == 4

    def test_custom(self):
        p = IDMParams(v0=30.0, a=2.0, b=3.0, s0=3.0, T=1.0, delta=6)
        assert p.v0 == 30.0
        assert p.a == 2.0
        assert p.delta == 6

    def test_idm_defaults_singleton(self):
        assert IDM_DEFAULTS.v0 == 12.0
        assert IDM_DEFAULTS.T == 1.5


# ===================================================================
# idm_acceleration
# ===================================================================


class TestIDMAcceleration:
    def test_stopped_no_leader(self):
        """Stopped car with no leader should accelerate positively."""
        acc = idm_acceleration(0.0, 1000.0, 0.0)
        assert acc > 0.0

    def test_at_desired_speed_no_leader(self):
        """At desired speed with large gap, acceleration should be near zero."""
        p = IDMParams(v0=12.0)
        acc = idm_acceleration(12.0, 1000.0, 12.0, p)
        assert abs(acc) < 0.5  # nearly zero

    def test_close_to_stopped_leader(self):
        """Close to a stopped leader, should brake hard."""
        acc = idm_acceleration(10.0, 3.0, 0.0)
        assert acc < -1.0  # braking

    def test_large_gap_accelerates(self):
        """Large gap with slow speed should give positive acceleration."""
        acc = idm_acceleration(5.0, 200.0, 15.0)
        assert acc > 0.0

    def test_approaching_leader_brakes(self):
        """Going faster than leader at small gap should brake."""
        acc = idm_acceleration(15.0, 10.0, 5.0)
        assert acc < 0.0

    def test_same_speed_comfortable_gap(self):
        """Same speed with comfortable gap should have mild acceleration."""
        p = IDMParams(v0=12.0, T=1.5, s0=2.0)
        # Gap = v0 * T + s0 = 12 * 1.5 + 2 = 20m -- comfortable distance
        acc = idm_acceleration(10.0, 20.0, 10.0, p)
        # Should be slightly positive (approaching desired speed, comfortable gap)
        assert acc > -1.0

    def test_braking_limit(self):
        """Acceleration should be clamped to -9 m/s^2 (1g)."""
        acc = idm_acceleration(20.0, 0.5, 0.0)
        assert acc >= -9.0

    def test_acceleration_limit(self):
        """Acceleration should not exceed max acceleration."""
        p = IDMParams(a=1.4)
        acc = idm_acceleration(0.0, 1000.0, 0.0, p)
        assert acc <= p.a

    def test_zero_v0(self):
        """v0=0 should not crash (edge case)."""
        p = IDMParams(v0=0.0)
        acc = idm_acceleration(5.0, 10.0, 5.0, p)
        assert isinstance(acc, float)

    def test_negative_gap_clamped(self):
        """Negative or very small gap should be clamped, not cause division by zero."""
        acc = idm_acceleration(5.0, 0.01, 0.0)
        assert isinstance(acc, float)
        assert not math.isnan(acc)
        assert not math.isinf(acc)


# ===================================================================
# idm_free_flow
# ===================================================================


class TestIDMFreeFlow:
    def test_stopped_accelerates(self):
        """Stopped vehicle in free flow should get max acceleration."""
        acc = idm_free_flow(0.0)
        assert acc == pytest.approx(IDM_DEFAULTS.a, abs=0.01)

    def test_at_desired_speed(self):
        """At desired speed, free-flow acceleration should be zero."""
        p = IDMParams(v0=12.0)
        acc = idm_free_flow(12.0, p)
        assert acc == pytest.approx(0.0, abs=0.01)

    def test_above_desired_speed(self):
        """Above desired speed, should decelerate."""
        p = IDMParams(v0=10.0)
        acc = idm_free_flow(15.0, p)
        assert acc < 0.0

    def test_half_speed(self):
        """At half desired speed, should have significant acceleration."""
        p = IDMParams(v0=12.0, a=1.4, delta=4)
        acc = idm_free_flow(6.0, p)
        # (1 - (6/12)^4) = (1 - 0.0625) = 0.9375
        expected = 1.4 * 0.9375
        assert acc == pytest.approx(expected, abs=0.01)

    def test_zero_v0(self):
        p = IDMParams(v0=0.0)
        acc = idm_free_flow(5.0, p)
        assert acc == 0.0


# ===================================================================
# idm_step
# ===================================================================


class TestIDMStep:
    def test_positive_acceleration(self):
        result = idm_step(5.0, 1.0, 0.1)
        assert isinstance(result, IDMStepResult)
        assert result.v == pytest.approx(5.1, abs=0.01)
        assert result.ds > 0

    def test_braking(self):
        result = idm_step(10.0, -3.0, 0.1)
        assert result.v == pytest.approx(9.7, abs=0.01)
        assert result.ds > 0

    def test_hard_braking_clamps_speed(self):
        """Speed should not go negative."""
        result = idm_step(1.0, -20.0, 1.0)
        assert result.v == 0.0

    def test_zero_speed_zero_acc(self):
        result = idm_step(0.0, 0.0, 0.1)
        assert result.v == 0.0
        assert result.ds == 0.0

    def test_distance_formula(self):
        """ds = v*dt + 0.5*a*dt^2"""
        v, a, dt = 10.0, 2.0, 0.1
        result = idm_step(v, a, dt)
        expected_ds = v * dt + 0.5 * a * dt * dt
        assert result.ds == pytest.approx(expected_ds, abs=0.001)


# ===================================================================
# ROAD_SPEEDS
# ===================================================================


class TestRoadSpeeds:
    def test_motorway_fastest(self):
        assert ROAD_SPEEDS["motorway"] >= ROAD_SPEEDS["residential"]

    def test_service_slowest(self):
        assert ROAD_SPEEDS["service"] <= ROAD_SPEEDS["residential"]

    def test_all_positive(self):
        for cls, speed in ROAD_SPEEDS.items():
            assert speed > 0, f"{cls} speed must be positive"


# ===================================================================
# VEHICLE_IDM_PROFILES
# ===================================================================


class TestVehicleProfiles:
    def test_sedan_exists(self):
        assert "sedan" in VEHICLE_IDM_PROFILES

    def test_truck_slower(self):
        assert VEHICLE_IDM_PROFILES["truck"].v0 < VEHICLE_IDM_PROFILES["sedan"].v0

    def test_motorcycle_faster(self):
        assert VEHICLE_IDM_PROFILES["motorcycle"].v0 > VEHICLE_IDM_PROFILES["sedan"].v0

    def test_all_profiles_valid(self):
        for name, params in VEHICLE_IDM_PROFILES.items():
            assert params.v0 > 0, f"{name} v0"
            assert params.a > 0, f"{name} a"
            assert params.b > 0, f"{name} b"
            assert params.s0 > 0, f"{name} s0"
            assert params.T > 0, f"{name} T"


# ===================================================================
# get_idm_for_road
# ===================================================================


class TestGetIDMForRoad:
    def test_residential(self):
        p = get_idm_for_road("residential", speed_variation=0.0)
        assert p.v0 == pytest.approx(ROAD_SPEEDS["residential"], abs=0.01)

    def test_motorway(self):
        p = get_idm_for_road("motorway", speed_variation=0.0)
        assert p.v0 == pytest.approx(ROAD_SPEEDS["motorway"], abs=0.01)

    def test_unknown_road_class_fallback(self):
        p = get_idm_for_road("alien_highway", speed_variation=0.0)
        assert p.v0 == pytest.approx(10.0, abs=0.01)  # default fallback

    def test_variation_within_range(self):
        """With variation, speed should be within +/-10% of base."""
        base = ROAD_SPEEDS["residential"]
        for _ in range(20):
            p = get_idm_for_road("residential", speed_variation=0.1)
            assert base * 0.85 <= p.v0 <= base * 1.15

    def test_preserves_base_params(self):
        base = IDMParams(a=2.0, b=3.0, s0=4.0, T=1.0)
        p = get_idm_for_road("residential", base_params=base, speed_variation=0.0)
        assert p.a == 2.0
        assert p.b == 3.0
        assert p.s0 == 4.0
        assert p.T == 1.0


# ===================================================================
# IDM behavioral properties (higher-level integration)
# ===================================================================


class TestIDMBehavior:
    """Test that IDM produces physically reasonable traffic behavior."""

    def test_equilibrium_speed(self):
        """A single car should converge toward v0 after many steps."""
        p = IDMParams(v0=12.0, a=1.4)
        v = 0.0
        dt = 0.1
        for _ in range(500):
            acc = idm_free_flow(v, p)
            result = idm_step(v, acc, dt)
            v = result.v
        assert v == pytest.approx(12.0, abs=0.5)

    def test_platoon_stabilizes(self):
        """Two cars following each other should stabilize at a safe gap."""
        p = IDMParams(v0=12.0, s0=2.0, T=1.5)
        leader_v = 10.0  # constant speed leader
        follower_v = 0.0
        gap = 50.0
        dt = 0.1

        for _ in range(1000):
            acc = idm_acceleration(follower_v, gap, leader_v, p)
            result = idm_step(follower_v, acc, dt)
            gap -= (result.ds - leader_v * dt)
            follower_v = result.v

        # Follower should match leader speed
        assert follower_v == pytest.approx(leader_v, abs=1.0)
        # Gap should be > s0 (minimum gap)
        assert gap > p.s0

    def test_emergency_stop(self):
        """Car approaching a stopped obstacle should stop before collision."""
        p = IDMParams(v0=12.0, s0=2.0, T=1.5, b=2.0)
        v = 12.0
        gap = 30.0
        dt = 0.05

        for _ in range(2000):
            acc = idm_acceleration(v, gap, 0.0, p)
            result = idm_step(v, acc, dt)
            gap -= result.ds
            v = result.v
            if v < 0.01:
                break

        # Car should stop before collision
        assert gap > 0, f"Car collided! Remaining gap: {gap:.2f}"
        assert v < 0.1, f"Car didn't stop: v={v:.2f}"
