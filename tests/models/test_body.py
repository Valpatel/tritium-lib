# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the neutral body seam — Track A step 2 of the multi-body plan.

Pins the body-agnostic vocabulary (ControlIntent superset / 6-DOF BodyState)
in its NEUTRAL home (models/body), the BodyController protocol + optional
capability hooks, the ground motor-twist <-> intent mapping, and — critically
— that the hoist out of models/multirotor changed NOTHING: the old import
paths still serve the very same objects and every body steps identically.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from tritium_lib.models.body import (
    G_MPS2,
    BodyController,
    BodyState,
    ControlIntent,
    SupportsBattery,
    SupportsGps,
    SupportsImu,
    SupportsTurret,
    intent_from_motors,
    motors_from_intent,
)


# --- ControlIntent / BodyState: construction + round-trip --------------------


def test_zero_intent_is_the_default():
    """Zero intent = the body's steady state (hover / cruise / stand)."""
    intent = ControlIntent()
    assert (intent.forward, intent.turn, intent.climb) == (0.0, 0.0, 0.0)


def test_intent_bounds_enforced():
    with pytest.raises(ValidationError):
        ControlIntent(forward=2.0)
    with pytest.raises(ValidationError):
        ControlIntent(turn=-1.5)
    with pytest.raises(ValidationError):
        ControlIntent(climb=1.5)


def test_intent_json_round_trip():
    intent = ControlIntent(forward=0.5, turn=-0.25, climb=1.0)
    restored = ControlIntent.model_validate_json(intent.model_dump_json())
    assert restored == intent


def test_body_state_defaults_and_ground_floor():
    s = BodyState()
    assert (s.x, s.y, s.alt_m, s.heading_deg) == (0.0, 0.0, 0.0, 0.0)
    assert (s.pitch_deg, s.roll_deg, s.speed_mps, s.climb_mps) == (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValidationError):
        BodyState(alt_m=-1.0)  # ge=0 — underground is not a pose


def test_body_state_json_round_trip():
    s = BodyState(x=1.5, y=-2.5, alt_m=30.0, heading_deg=270.0,
                  pitch_deg=-10.0, roll_deg=5.0, speed_mps=12.0, climb_mps=2.0)
    restored = BodyState.model_validate_json(s.model_dump_json())
    assert restored == s


def test_ground_state_embeds_losslessly():
    """A 2D body's (x, y, heading, speed) fills BodyState with zero altitude
    fields — the ground specialization the plan doc keeps."""
    s = BodyState(x=3.0, y=4.0, heading_deg=90.0, speed_mps=1.2)
    assert (s.alt_m, s.pitch_deg, s.roll_deg, s.climb_mps) == (0.0, 0.0, 0.0, 0.0)


# --- The hoist: old import paths serve the SAME objects ----------------------


def test_multirotor_reexports_are_identical_objects():
    """`from tritium_lib.models.multirotor import ...` (the pre-hoist path)
    must serve the very same classes/constant — not copies."""
    from tritium_lib.models import multirotor

    assert multirotor.ControlIntent is ControlIntent
    assert multirotor.BodyState is BodyState
    assert multirotor.G_MPS2 == G_MPS2


def test_models_package_exports_the_seam():
    import tritium_lib.models as models

    assert models.ControlIntent is ControlIntent
    assert models.BodyState is BodyState
    assert models.BodyController is BodyController
    for name in (
        "G_MPS2", "BodyController", "BodyState", "ControlIntent",
        "SupportsBattery", "SupportsGps", "SupportsImu", "SupportsTurret",
        "intent_from_motors", "motors_from_intent",
    ):
        assert name in models.__all__, f"{name} missing from models.__all__"


# --- BodyController protocol: structural, runtime-checkable ------------------


class _MinimalBody:
    """A body with ONLY the required seam pair — no lib inheritance."""

    def __init__(self) -> None:
        self._state = BodyState()

    def command(self, intent: ControlIntent) -> None:
        self._state = BodyState(speed_mps=intent.forward)

    def get_state(self) -> BodyState:
        return self._state


class _FullBody(_MinimalBody):
    """A body carrying every optional hook (edge HardwareInterface names)."""

    def set_turret(self, pan: float, tilt: float) -> None:
        self.aim = (pan, tilt)

    def fire_trigger(self) -> None:
        self.fired = True

    def get_battery(self) -> float:
        return 0.87

    def get_imu(self):
        return (0.0, 0.0, self._state.heading_deg)

    def get_gps(self) -> tuple[float, float, float] | None:
        return None


def test_minimal_body_satisfies_body_controller():
    body = _MinimalBody()
    assert isinstance(body, BodyController)
    body.command(ControlIntent(forward=0.5))
    assert body.get_state().speed_mps == pytest.approx(0.5)


def test_incomplete_body_fails_the_protocol():
    class _NoState:
        def command(self, intent: ControlIntent) -> None:
            pass

    assert not isinstance(_NoState(), BodyController)
    assert not isinstance(object(), BodyController)


def test_optional_hooks_are_separate_capabilities():
    """A minimal body owes NO hooks; a full body satisfies them all."""
    minimal, full = _MinimalBody(), _FullBody()
    for proto in (SupportsTurret, SupportsBattery, SupportsImu, SupportsGps):
        assert not isinstance(minimal, proto)
        assert isinstance(full, proto)
    assert isinstance(full, BodyController)


# --- Ground twist mapping: motors <-> intent ---------------------------------


def test_intent_from_motors_matches_rover_twist_contract():
    """The mapping and RoverProfile.twist agree by construction — the same
    (l+r)/2, (l-r) contract, expressed through the neutral intent."""
    from tritium_lib.models.rover import RoverProfile

    r = RoverProfile()
    for left, right in [(1.0, 1.0), (0.5, -0.5), (-1.0, 1.0), (0.3, 0.9), (2.0, -3.0)]:
        intent = intent_from_motors(left, right)
        fwd_mps, turn_dps = r.twist(left, right)
        assert intent.forward * r.max_speed_mps == pytest.approx(fwd_mps)
        assert intent.turn * r.max_turn_dps == pytest.approx(turn_dps)
        assert intent.climb == 0.0  # ground bodies have no vertical intent


def test_motors_round_trip_within_envelope():
    """motors -> intent -> motors is exact when the pair fits the envelope
    (|forward| + |turn|/2 <= 1)."""
    for left, right in [(1.0, 1.0), (0.5, 0.0), (-0.4, -0.4), (0.25, -0.25), (0.0, 0.0)]:
        intent = intent_from_motors(left, right)
        assert motors_from_intent(intent) == (
            pytest.approx(left), pytest.approx(right),
        )


def test_motors_from_intent_clamps_saturated_demands():
    left, right = motors_from_intent(ControlIntent(forward=1.0, turn=1.0))
    assert (left, right) == (1.0, 0.5)  # left clamped at full throttle
    left, right = motors_from_intent(ControlIntent(forward=-1.0, turn=-1.0))
    assert (left, right) == (-1.0, -0.5)  # left clamped at full reverse


def test_motors_from_intent_ignores_climb():
    grounded = motors_from_intent(ControlIntent(forward=0.6, turn=0.2))
    flying = motors_from_intent(ControlIntent(forward=0.6, turn=0.2, climb=1.0))
    assert grounded == flying


# --- The hoist changed no body's behavior ------------------------------------


def test_multirotor_step_unchanged_by_hoist():
    """The multirotor integrates the SAME numbers with the hoisted types and
    returns the neutral BodyState class."""
    from tritium_lib.models.multirotor import MultirotorProfile

    m = MultirotorProfile()
    s = m.step(BodyState(alt_m=10.0), ControlIntent(forward=1.0, climb=1.0), 1.0)
    assert type(s) is BodyState
    assert s.alt_m == pytest.approx(15.0)  # max_climb_mps = 5
    assert s.y == pytest.approx(16.0)  # max_speed_mps = 16, heading north
    assert s.x == pytest.approx(0.0)


def test_fixedwing_step_unchanged_by_hoist():
    """The fixed-wing consumes the hoisted intent and cruises at trim."""
    from tritium_lib.models.fixedwing import FixedWingProfile

    w = FixedWingProfile()
    s = w.step(BodyState(alt_m=50.0), ControlIntent(), 1.0)
    assert type(s) is BodyState
    assert s.speed_mps == pytest.approx(w.cruise_speed_mps)  # zero intent = cruise
    assert s.y == pytest.approx(w.cruise_speed_mps)  # flew north one second
    assert s.alt_m == pytest.approx(50.0)
