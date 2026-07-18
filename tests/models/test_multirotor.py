# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the multirotor (quadcopter) body model — the third fleet body.

Pins the aerial seam vocabulary (ControlIntent superset / 6-DOF BodyState),
the Tritium frame (x=east, y=north, heading 0 = north increasing clockwise),
the thrust/tilt envelope, and the battery model — so a multirotor controller,
the sim, and real hardware agree.
"""
from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from tritium_lib.models.multirotor import (
    DEFAULT_HEXROTOR,
    DEFAULT_QUADCOPTER,
    G_MPS2,
    BodyState,
    ControlIntent,
    MultirotorProfile,
)


# --- Defaults: Mavic-3-class quadcopter out of the box -----------------------


def test_default_profile_is_valid():
    m = MultirotorProfile()
    assert m.profile == "multirotor"
    assert m.name == "quadcopter"
    assert m.asset_type == "drone"
    assert m.rotor_count == 4
    assert m.rotor_layout == "x"
    assert m.thrust_to_weight() > 1.0


def test_default_profile_numbers_match_contract():
    """DEFAULT_QUADCOPTER is the named canonical profile — pin its envelope."""
    m = DEFAULT_QUADCOPTER
    assert (m.mass_kg, m.max_thrust_n) == (0.9, 22.0)
    assert (m.max_climb_mps, m.max_descent_mps, m.max_speed_mps) == (5.0, 3.0, 16.0)
    assert (m.max_tilt_deg, m.max_yaw_rate_dps) == (30.0, 120.0)
    assert (m.battery_wh, m.idle_power_w, m.hover_power_w, m.max_power_w) == (
        77.0, 6.0, 115.0, 260.0,
    )


def test_default_hexrotor_is_heavier_lift():
    h = DEFAULT_HEXROTOR
    assert h.rotor_count == 6
    assert h.mass_kg > DEFAULT_QUADCOPTER.mass_kg
    assert h.thrust_to_weight() > 1.0


def test_profile_literal_rejects_other_values():
    with pytest.raises(ValidationError):
        MultirotorProfile(profile="fixedwing")


# --- Field constraints + envelope validator ----------------------------------


def test_invalid_geometry_rejected():
    with pytest.raises(ValidationError):
        MultirotorProfile(mass_kg=0.0)  # gt=0
    with pytest.raises(ValidationError):
        MultirotorProfile(rotor_count=2)  # ge=3
    with pytest.raises(ValidationError):
        MultirotorProfile(max_tilt_deg=95.0)  # le=90


def test_thrust_below_weight_rejected():
    """A multirotor that cannot out-thrust its own weight cannot hover."""
    with pytest.raises(ValidationError):
        MultirotorProfile(mass_kg=5.0)  # weight ~49 N > default 22 N thrust


def test_power_ordering_enforced():
    with pytest.raises(ValidationError):
        MultirotorProfile(hover_power_w=300.0)  # hover > max (260)
    with pytest.raises(ValidationError):
        MultirotorProfile(idle_power_w=200.0)  # idle > hover (115)


# --- ControlIntent / BodyState: the aerial seam vocabulary -------------------


def test_zero_intent_is_the_default():
    """Zero intent = the body's steady state (hover for a multirotor)."""
    intent = ControlIntent()
    assert (intent.forward, intent.turn, intent.climb) == (0.0, 0.0, 0.0)


def test_intent_bounds_enforced():
    with pytest.raises(ValidationError):
        ControlIntent(forward=2.0)
    with pytest.raises(ValidationError):
        ControlIntent(climb=-1.5)


def test_body_state_defaults_and_ground_floor():
    s = BodyState()
    assert (s.x, s.y, s.alt_m, s.heading_deg) == (0.0, 0.0, 0.0, 0.0)
    assert (s.pitch_deg, s.roll_deg, s.speed_mps, s.climb_mps) == (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValidationError):
        BodyState(alt_m=-1.0)  # ge=0 — underground is not a pose


# --- Thrust envelope helpers -------------------------------------------------


def test_thrust_to_weight_and_hover_frac_are_reciprocal():
    m = MultirotorProfile()
    assert m.thrust_to_weight() == pytest.approx(22.0 / (0.9 * G_MPS2))
    assert m.hover_thrust_frac() == pytest.approx(1.0 / m.thrust_to_weight())
    assert 0.0 < m.hover_thrust_frac() < 1.0


def test_tilt_for_speed_linear_and_clamped():
    m = MultirotorProfile()
    assert m.tilt_for_speed(0.0) == pytest.approx(0.0)
    assert m.tilt_for_speed(8.0) == pytest.approx(15.0)  # half speed -> half tilt
    assert m.tilt_for_speed(16.0) == pytest.approx(30.0)
    assert m.tilt_for_speed(99.0) == pytest.approx(30.0)  # clamped at max tilt
    assert m.tilt_for_speed(-8.0) == pytest.approx(15.0)  # magnitude only


# --- step: flight kinematics in the Tritium frame ----------------------------


def test_step_hover_holds_position():
    m = MultirotorProfile()
    s = BodyState(x=3.0, y=4.0, alt_m=20.0, heading_deg=90.0)
    for _ in range(10):
        s = m.step(s, ControlIntent(), 0.1)
    assert (s.x, s.y, s.alt_m, s.heading_deg) == (3.0, 4.0, 20.0, 90.0)
    assert (s.speed_mps, s.climb_mps, s.pitch_deg, s.roll_deg) == (0.0, 0.0, 0.0, 0.0)


def test_step_straight_flies_north_no_east_drift():
    m = MultirotorProfile()
    s = BodyState(alt_m=20.0)  # heading 0 = north
    for _ in range(10):
        s = m.step(s, ControlIntent(forward=1.0), 0.1)
    assert s.heading_deg == pytest.approx(0.0)
    assert s.y > 10.0, f"should have flown north, y={s.y}"
    assert abs(s.x) < 1e-9, f"no east drift, x={s.x}"


def test_step_turn_then_forward_moves_east():
    """Yaw clockwise ~90deg (toward east), then fly -> +x (east)."""
    m = MultirotorProfile(max_yaw_rate_dps=90.0)
    s = BodyState(alt_m=20.0)
    s = m.step(s, ControlIntent(turn=1.0), 1.0)  # yaw 90 dps for 1s
    assert s.heading_deg == pytest.approx(90.0, abs=1e-6)
    s = m.step(s, ControlIntent(forward=1.0), 1.0)  # fly east
    assert s.x > 10.0, f"should have flown east, x={s.x}"
    assert abs(s.y) < 1e-6, f"no north drift after heading east, y={s.y}"


def test_step_climb_and_descent_use_asymmetric_limits():
    m = MultirotorProfile()
    s = BodyState(alt_m=10.0)
    up = m.step(s, ControlIntent(climb=1.0), 1.0)
    assert up.alt_m == pytest.approx(15.0)  # max_climb_mps = 5
    assert up.climb_mps == pytest.approx(5.0)
    down = m.step(s, ControlIntent(climb=-1.0), 1.0)
    assert down.alt_m == pytest.approx(7.0)  # max_descent_mps = 3
    assert down.climb_mps == pytest.approx(-3.0)


def test_step_clamps_at_ground():
    m = MultirotorProfile()
    s = BodyState(alt_m=1.0)
    s = m.step(s, ControlIntent(climb=-1.0), 1.0)  # would reach -2 m
    assert s.alt_m == 0.0
    assert s.climb_mps == 0.0


def test_step_forward_flight_pitches_nose_down():
    """A multirotor tilts INTO its motion: forward -> negative pitch, and
    backward -> positive pitch; yaw is coordinated so roll stays zero."""
    m = MultirotorProfile()
    s = BodyState(alt_m=20.0)
    ahead = m.step(s, ControlIntent(forward=1.0), 0.1)
    assert ahead.pitch_deg == pytest.approx(-30.0)  # full tilt, nose down
    astern = m.step(s, ControlIntent(forward=-0.5), 0.1)
    assert astern.pitch_deg == pytest.approx(15.0)  # half tilt, nose up
    assert ahead.roll_deg == 0.0


def test_step_is_deterministic():
    m = MultirotorProfile()

    def run():
        s = BodyState(alt_m=5.0, heading_deg=30.0)
        for _ in range(20):
            s = m.step(s, ControlIntent(forward=0.8, turn=0.2, climb=0.3), 0.1)
        return tuple(round(v, 9) for v in (s.x, s.y, s.alt_m, s.heading_deg))

    assert run() == run()


# --- battery -----------------------------------------------------------------


def test_drain_landed_uses_idle_power():
    m = MultirotorProfile()
    assert m.drain_pct_per_s(None) == pytest.approx(6.0 / (77.0 * 3600.0))


def test_drain_hover_matches_hover_power():
    m = MultirotorProfile()
    assert m.drain_pct_per_s(0.0) == pytest.approx(115.0 / (77.0 * 3600.0))


def test_drain_monotonic_with_speed_and_climb():
    m = MultirotorProfile()
    speeds = [0.0, 4.0, 8.0, 12.0, m.max_speed_mps]
    drains = [m.drain_pct_per_s(v) for v in speeds]
    assert drains == sorted(drains)
    climbs = [0.0, 1.0, 3.0, m.max_climb_mps]
    drains = [m.drain_pct_per_s(0.0, c) for c in climbs]
    assert drains == sorted(drains)
    # descending costs no more than hovering
    assert m.drain_pct_per_s(0.0, -3.0) == pytest.approx(m.drain_pct_per_s(0.0))


def test_full_battery_hover_endurance_is_mavic_plausible():
    """A 77 Wh pack hovering at 115 W should last roughly 2410 s (~40 min)."""
    m = MultirotorProfile()
    endurance_s = 1.0 / m.drain_pct_per_s(0.0)
    assert 1800.0 < endurance_s < 3000.0


# --- Serialization: wire round-trip -----------------------------------------


def test_model_dump_round_trip_is_stable():
    m = MultirotorProfile()
    wire = m.model_dump()
    restored = MultirotorProfile.model_validate(wire)
    assert restored == m
    assert restored.model_dump_json() == m.model_dump_json()


def test_body_state_json_round_trip():
    s = BodyState(x=1.5, y=-2.5, alt_m=30.0, heading_deg=270.0,
                  pitch_deg=-10.0, roll_deg=5.0, speed_mps=12.0, climb_mps=2.0)
    restored = BodyState.model_validate_json(s.model_dump_json())
    assert restored == s
