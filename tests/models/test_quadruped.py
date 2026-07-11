# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the quadruped (robot dog) profile model — gait vocabulary."""

import pytest
from pydantic import ValidationError

from tritium_lib.models.quadruped import (
    DEFAULT_GAITS,
    GaitSpec,
    QuadrupedProfile,
)


# --- Defaults: Go2-class dog out of the box --------------------------------


def test_default_profile_has_three_gaits():
    profile = QuadrupedProfile()
    assert set(profile.gaits) == {"walk", "trot", "bound"}
    assert profile.default_gait == "trot"


def test_default_identity_fields():
    profile = QuadrupedProfile()
    assert profile.profile == "quadruped"
    assert profile.name == "robot_dog"
    assert profile.asset_type == "robot_dog"
    assert profile.leg_count == 4
    assert profile.body_height_m == 0.40
    assert profile.turn_rate_dps == 120.0
    assert profile.battery_wh == 155.0
    assert profile.idle_power_w == 25.0


def test_profile_literal_rejects_other_values():
    with pytest.raises(ValidationError):
        QuadrupedProfile(profile="biped")


def test_default_gait_numbers_match_contract():
    """The robot-template example embeds these same numbers as a documented
    mirror (like turret.py mirrors the fire_control servo bounds) — pin them."""
    walk = DEFAULT_GAITS["walk"]
    assert (walk.speed_mps, walk.stride_hz, walk.power_w) == (0.7, 1.6, 65.0)
    trot = DEFAULT_GAITS["trot"]
    assert (trot.speed_mps, trot.stride_hz, trot.power_w) == (1.6, 2.6, 120.0)
    bound = DEFAULT_GAITS["bound"]
    assert (bound.speed_mps, bound.stride_hz, bound.power_w) == (3.0, 3.2, 250.0)
    assert bound.roll_amp_deg == 4.0
    assert bound.pitch_amp_deg == 6.0
    assert bound.bob_amp_m == 0.04


def test_gaits_default_is_a_copy_not_shared_state():
    """Mutating one profile's gait table must not bleed into DEFAULT_GAITS
    or into other profiles (default_factory deep-copies)."""
    a = QuadrupedProfile()
    a.gaits["walk"].speed_mps = 9.9
    a.gaits["extra"] = GaitSpec(
        speed_mps=5.0, stride_hz=4.0, roll_amp_deg=0, pitch_amp_deg=0,
        bob_amp_m=0, power_w=400.0,
    )
    assert DEFAULT_GAITS["walk"].speed_mps == 0.7
    assert "extra" not in DEFAULT_GAITS
    b = QuadrupedProfile()
    assert b.gaits["walk"].speed_mps == 0.7
    assert "extra" not in b.gaits


# --- Field constraints ------------------------------------------------------


def test_gait_spec_rejects_nonpositive_speed_and_power():
    with pytest.raises(ValidationError):
        GaitSpec(speed_mps=0.0, stride_hz=1.0, roll_amp_deg=0,
                 pitch_amp_deg=0, bob_amp_m=0, power_w=10.0)
    with pytest.raises(ValidationError):
        GaitSpec(speed_mps=1.0, stride_hz=1.0, roll_amp_deg=0,
                 pitch_amp_deg=0, bob_amp_m=0, power_w=0.0)


def test_gait_spec_amplitudes_allow_zero_but_not_negative():
    spec = GaitSpec(speed_mps=1.0, stride_hz=1.0, roll_amp_deg=0.0,
                    pitch_amp_deg=0.0, bob_amp_m=0.0, power_w=10.0)
    assert spec.roll_amp_deg == 0.0
    with pytest.raises(ValidationError):
        GaitSpec(speed_mps=1.0, stride_hz=1.0, roll_amp_deg=-0.1,
                 pitch_amp_deg=0, bob_amp_m=0, power_w=10.0)


def test_leg_count_bounds():
    assert QuadrupedProfile(leg_count=6).leg_count == 6
    with pytest.raises(ValidationError):
        QuadrupedProfile(leg_count=3)
    with pytest.raises(ValidationError):
        QuadrupedProfile(leg_count=7)


# --- gait_for_speed: slowest gait that covers the request -------------------


def test_gait_for_speed_selects_by_requested_speed():
    profile = QuadrupedProfile()
    assert profile.gait_for_speed(0.5)[0] == "walk"
    assert profile.gait_for_speed(1.0)[0] == "trot"
    assert profile.gait_for_speed(2.0)[0] == "bound"


def test_gait_for_speed_caps_at_fastest_gait():
    profile = QuadrupedProfile()
    name, spec = profile.gait_for_speed(99.0)
    assert name == "bound"
    assert spec.speed_mps == 3.0


def test_gait_for_speed_returns_spec_alongside_name():
    profile = QuadrupedProfile()
    name, spec = profile.gait_for_speed(0.5)
    assert name == "walk"
    assert spec is profile.gaits["walk"]


def test_gait_for_speed_nonpositive_request_returns_slowest():
    """"stand" (not moving) is the caller's state, not a gait — any request
    at or below zero maps to the slowest available gait."""
    profile = QuadrupedProfile()
    assert profile.gait_for_speed(0.0)[0] == "walk"
    assert profile.gait_for_speed(-1.0)[0] == "walk"


def test_gait_for_speed_robust_to_declaration_order():
    """Selection must sort by speed_mps, not trust dict insertion order."""
    profile = QuadrupedProfile(
        gaits={
            "bound": DEFAULT_GAITS["bound"].model_copy(),
            "walk": DEFAULT_GAITS["walk"].model_copy(),
            "trot": DEFAULT_GAITS["trot"].model_copy(),
        },
    )
    assert profile.gait_for_speed(0.5)[0] == "walk"
    assert profile.gait_for_speed(1.0)[0] == "trot"
    assert profile.gait_for_speed(2.0)[0] == "bound"


def test_gait_for_speed_honors_custom_override():
    """A profile that re-tunes trot to 2.0 m/s should satisfy a 2.0 request
    with trot (slowest covering gait), not escalate to bound."""
    gaits = {k: v.model_copy() for k, v in DEFAULT_GAITS.items()}
    gaits["trot"].speed_mps = 2.0
    profile = QuadrupedProfile(gaits=gaits)
    assert profile.gait_for_speed(2.0)[0] == "trot"
    assert profile.gait_for_speed(2.1)[0] == "bound"


# --- drain_pct_per_s: battery fraction per second ---------------------------


def test_drain_trot_matches_power_over_capacity():
    profile = QuadrupedProfile()
    assert profile.drain_pct_per_s("trot") == pytest.approx(
        120.0 / (155.0 * 3600.0)
    )


def test_drain_idle_uses_idle_power():
    profile = QuadrupedProfile()
    assert profile.drain_pct_per_s(None) == pytest.approx(
        25.0 / (155.0 * 3600.0)
    )


def test_full_battery_trot_endurance_is_go2_plausible():
    """A Go2-class pack at trot should last roughly 4650 s (~78 min)."""
    profile = QuadrupedProfile()
    endurance_s = 1.0 / profile.drain_pct_per_s("trot")
    assert 4000.0 < endurance_s < 5500.0


def test_drain_unknown_gait_raises_key_error():
    profile = QuadrupedProfile()
    with pytest.raises(KeyError):
        profile.drain_pct_per_s("gallop")


# --- Serialization: wire round-trip -----------------------------------------


def test_model_dump_round_trip_is_stable():
    profile = QuadrupedProfile()
    wire = profile.model_dump()
    restored = QuadrupedProfile.model_validate(wire)
    assert restored == profile
    assert restored.model_dump() == wire
    assert restored.model_dump_json() == profile.model_dump_json()


def test_json_round_trip_preserves_gait_table():
    profile = QuadrupedProfile(name="rex", default_gait="walk")
    restored = QuadrupedProfile.model_validate_json(profile.model_dump_json())
    assert restored.name == "rex"
    assert restored.default_gait == "walk"
    assert restored.gaits["bound"].power_w == 250.0


# --- Validator: default_gait must exist in the gait table -------------------


def test_default_gait_must_be_in_gaits():
    with pytest.raises(ValidationError):
        QuadrupedProfile(default_gait="gallop")


def test_default_gait_rejected_when_custom_gaits_omit_it():
    only_walk = {"walk": DEFAULT_GAITS["walk"].model_copy()}
    with pytest.raises(ValidationError):
        QuadrupedProfile(gaits=only_walk)  # default_gait "trot" missing
    profile = QuadrupedProfile(gaits=only_walk, default_gait="walk")
    assert profile.default_gait == "walk"
