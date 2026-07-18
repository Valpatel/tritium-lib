# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the fixed-wing body model — the fourth fleet body.

Pins the airspeed envelope (a wing NEVER flies below stall), the
coordinated-turn relation (heading rate = g*tan(bank)/v), the control-surface
table, and the battery model — so a fixed-wing controller, the sim, and real
hardware agree. Shares the multirotor's ControlIntent/BodyState vocabulary.
"""
from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from tritium_lib.models.fixedwing import (
    DEFAULT_FIXEDWING,
    DEFAULT_SURFACES,
    ControlSurfaceSpec,
    FixedWingProfile,
)
from tritium_lib.models.multirotor import G_MPS2, BodyState, ControlIntent


# --- Defaults: foam-trainer-class plane out of the box -----------------------


def test_default_profile_is_valid():
    w = FixedWingProfile()
    assert w.profile == "fixedwing"
    assert w.asset_type == "fixed_wing"
    assert w.stall_speed_mps < w.cruise_speed_mps <= w.max_speed_mps
    assert set(w.surfaces) == {"aileron", "elevator", "rudder"}


def test_default_profile_numbers_match_contract():
    """DEFAULT_FIXEDWING is the named canonical profile — pin its envelope."""
    w = DEFAULT_FIXEDWING
    assert (w.mass_kg, w.wing_area_m2, w.wingspan_m) == (2.0, 0.60, 1.8)
    assert (w.stall_speed_mps, w.cruise_speed_mps, w.max_speed_mps) == (9.0, 16.0, 28.0)
    assert (w.max_bank_deg, w.max_climb_mps, w.max_sink_mps) == (45.0, 5.0, 4.0)
    assert (w.battery_wh, w.idle_power_w, w.cruise_power_w, w.max_power_w) == (
        88.0, 4.0, 110.0, 350.0,
    )


def test_default_surface_numbers_match_contract():
    aileron = DEFAULT_SURFACES["aileron"]
    assert (aileron.travel_deg, aileron.rate_dps) == (25.0, 200.0)
    elevator = DEFAULT_SURFACES["elevator"]
    assert (elevator.travel_deg, elevator.rate_dps) == (20.0, 180.0)
    rudder = DEFAULT_SURFACES["rudder"]
    assert (rudder.travel_deg, rudder.rate_dps) == (25.0, 180.0)


def test_surfaces_default_is_a_copy_not_shared_state():
    """Mutating one profile's surface table must not bleed into
    DEFAULT_SURFACES or into other profiles (default_factory deep-copies)."""
    a = FixedWingProfile()
    a.surfaces["aileron"].travel_deg = 99.0
    a.surfaces["flap"] = ControlSurfaceSpec(travel_deg=40.0, rate_dps=60.0)
    assert DEFAULT_SURFACES["aileron"].travel_deg == 25.0
    assert "flap" not in DEFAULT_SURFACES
    b = FixedWingProfile()
    assert b.surfaces["aileron"].travel_deg == 25.0
    assert "flap" not in b.surfaces


def test_wing_loading():
    assert FixedWingProfile().wing_loading_kg_m2() == pytest.approx(2.0 / 0.60)


# --- Field constraints + envelope validator ----------------------------------


def test_invalid_geometry_rejected():
    with pytest.raises(ValidationError):
        FixedWingProfile(wing_area_m2=0.0)  # gt=0
    with pytest.raises(ValidationError):
        FixedWingProfile(max_bank_deg=90.0)  # lt=90 — a knife-edge is not a turn
    with pytest.raises(ValidationError):
        ControlSurfaceSpec(travel_deg=0.0, rate_dps=100.0)


def test_airspeed_ordering_enforced():
    with pytest.raises(ValidationError):
        FixedWingProfile(stall_speed_mps=20.0)  # stall >= cruise (16)
    with pytest.raises(ValidationError):
        FixedWingProfile(cruise_speed_mps=30.0)  # cruise > max (28)


def test_power_ordering_enforced():
    with pytest.raises(ValidationError):
        FixedWingProfile(cruise_power_w=400.0)  # cruise > max (350)


# --- airspeed_for_intent: the [stall, max] envelope --------------------------


def test_intent_maps_stall_cruise_max():
    w = FixedWingProfile()
    assert w.airspeed_for_intent(-1.0) == pytest.approx(w.stall_speed_mps)
    assert w.airspeed_for_intent(0.0) == pytest.approx(w.cruise_speed_mps)
    assert w.airspeed_for_intent(1.0) == pytest.approx(w.max_speed_mps)
    assert w.airspeed_for_intent(0.5) == pytest.approx(22.0)  # cruise + half of (28-16)
    assert w.airspeed_for_intent(-0.5) == pytest.approx(12.5)  # cruise - half of (16-9)


def test_intent_clamps_and_never_goes_below_stall():
    w = FixedWingProfile()
    assert w.airspeed_for_intent(-5.0) == pytest.approx(w.stall_speed_mps)
    assert w.airspeed_for_intent(5.0) == pytest.approx(w.max_speed_mps)


# --- Coordinated turn: r = v^2 / (g tan bank), omega = g tan bank / v --------


def test_turn_radius_matches_formula():
    w = FixedWingProfile()
    r = w.turn_radius_m(16.0, bank_deg=45.0)
    assert r == pytest.approx(16.0 * 16.0 / (G_MPS2 * math.tan(math.radians(45.0))))
    # steeper bank -> tighter turn; faster -> wider turn
    assert w.turn_radius_m(16.0, bank_deg=30.0) > r
    assert w.turn_radius_m(28.0, bank_deg=45.0) > r


def test_turn_rate_agrees_with_radius():
    """omega (rad/s) must equal v / r — the two helpers agree by construction."""
    w = FixedWingProfile()
    v = 20.0
    omega_rad = math.radians(w.turn_rate_dps(v, bank_deg=30.0))
    assert omega_rad == pytest.approx(v / w.turn_radius_m(v, bank_deg=30.0))


def test_turn_helpers_clamp_to_flyable_envelope():
    w = FixedWingProfile()
    # airspeed below stall is clamped up to stall; bank beyond limit clamped down
    assert w.turn_radius_m(1.0) == pytest.approx(w.turn_radius_m(w.stall_speed_mps))
    assert w.turn_rate_dps(16.0, bank_deg=80.0) == pytest.approx(
        w.turn_rate_dps(16.0, bank_deg=w.max_bank_deg)
    )


# --- step: flight kinematics in the Tritium frame ----------------------------


def test_step_zero_intent_cruises_north_no_east_drift():
    """Zero intent = steady cruise, level, wings level — a wing cannot hover."""
    w = FixedWingProfile()
    s = BodyState(alt_m=50.0)  # heading 0 = north
    for _ in range(10):
        s = w.step(s, ControlIntent(), 0.1)
    assert s.speed_mps == pytest.approx(w.cruise_speed_mps)
    assert s.y == pytest.approx(w.cruise_speed_mps)  # 1 s of cruise
    assert abs(s.x) < 1e-9, f"no east drift, x={s.x}"
    assert (s.pitch_deg, s.roll_deg, s.climb_mps) == (0.0, 0.0, 0.0)


def test_step_never_flies_below_stall():
    w = FixedWingProfile()
    s = BodyState(alt_m=50.0)
    s = w.step(s, ControlIntent(forward=-1.0), 1.0)
    assert s.speed_mps == pytest.approx(w.stall_speed_mps)
    assert s.y == pytest.approx(w.stall_speed_mps)  # still moving — no hover


def test_step_banks_into_the_turn():
    w = FixedWingProfile()
    s = BodyState(alt_m=50.0)
    s = w.step(s, ControlIntent(turn=0.5), 0.1)
    assert s.roll_deg == pytest.approx(22.5)  # half of max bank 45
    assert s.heading_deg > 0.0  # clockwise
    expected_dps = math.degrees(
        G_MPS2 * math.tan(math.radians(22.5)) / w.cruise_speed_mps
    )
    assert s.heading_deg == pytest.approx(expected_dps * 0.1)


def test_step_same_turn_intent_is_wider_at_speed():
    """The coordinated-turn relation: full throttle turns SLOWER (deg/s)
    than cruise at the same bank — unlike the yaw-in-place multirotor."""
    w = FixedWingProfile()
    s = BodyState(alt_m=50.0)
    at_cruise = w.step(s, ControlIntent(turn=1.0), 1.0).heading_deg
    at_max = w.step(s, ControlIntent(turn=1.0, forward=1.0), 1.0).heading_deg
    assert at_max < at_cruise


def test_step_climb_sets_flight_path_pitch():
    w = FixedWingProfile()
    s = BodyState(alt_m=50.0)
    s = w.step(s, ControlIntent(climb=1.0), 1.0)
    assert s.alt_m == pytest.approx(55.0)  # max_climb_mps = 5
    assert s.pitch_deg == pytest.approx(
        math.degrees(math.asin(5.0 / w.cruise_speed_mps))
    )
    down = w.step(BodyState(alt_m=50.0), ControlIntent(climb=-1.0), 1.0)
    assert down.alt_m == pytest.approx(46.0)  # max_sink_mps = 4
    assert down.pitch_deg < 0.0


def test_step_clamps_at_ground():
    w = FixedWingProfile()
    s = BodyState(alt_m=2.0)
    s = w.step(s, ControlIntent(climb=-1.0), 1.0)  # would reach -2 m
    assert s.alt_m == 0.0
    assert s.climb_mps == 0.0


def test_step_is_deterministic():
    w = FixedWingProfile()

    def run():
        s = BodyState(alt_m=30.0, heading_deg=30.0)
        for _ in range(20):
            s = w.step(s, ControlIntent(forward=0.4, turn=0.3, climb=0.2), 0.1)
        return tuple(round(v, 9) for v in (s.x, s.y, s.alt_m, s.heading_deg))

    assert run() == run()


# --- battery -----------------------------------------------------------------


def test_drain_parked_uses_idle_power():
    w = FixedWingProfile()
    assert w.drain_pct_per_s(None) == pytest.approx(4.0 / (88.0 * 3600.0))


def test_drain_cruise_matches_cruise_power():
    w = FixedWingProfile()
    assert w.drain_pct_per_s(w.cruise_speed_mps) == pytest.approx(
        110.0 / (88.0 * 3600.0)
    )
    # loitering near stall costs no more than cruise in this coarse envelope
    assert w.drain_pct_per_s(w.stall_speed_mps) == pytest.approx(
        w.drain_pct_per_s(w.cruise_speed_mps)
    )


def test_drain_monotonic_with_speed_and_climb():
    w = FixedWingProfile()
    speeds = [16.0, 20.0, 24.0, w.max_speed_mps]
    drains = [w.drain_pct_per_s(v) for v in speeds]
    assert drains == sorted(drains)
    climbs = [0.0, 2.0, w.max_climb_mps]
    drains = [w.drain_pct_per_s(w.cruise_speed_mps, c) for c in climbs]
    assert drains == sorted(drains)
    # descending costs no more than cruising
    assert w.drain_pct_per_s(w.cruise_speed_mps, -4.0) == pytest.approx(
        w.drain_pct_per_s(w.cruise_speed_mps)
    )


def test_full_battery_cruise_endurance_is_trainer_plausible():
    """An 88 Wh pack cruising at 110 W should last roughly 2880 s (~48 min)."""
    w = FixedWingProfile()
    endurance_s = 1.0 / w.drain_pct_per_s(w.cruise_speed_mps)
    assert 2400.0 < endurance_s < 3600.0


# --- Serialization: wire round-trip -----------------------------------------


def test_model_dump_round_trip_is_stable():
    w = FixedWingProfile()
    wire = w.model_dump()
    restored = FixedWingProfile.model_validate(wire)
    assert restored == w
    assert restored.model_dump_json() == w.model_dump_json()


def test_json_round_trip_preserves_surface_table():
    w = FixedWingProfile(name="albatross")
    restored = FixedWingProfile.model_validate_json(w.model_dump_json())
    assert restored.name == "albatross"
    assert restored.surfaces["elevator"].travel_deg == 20.0
