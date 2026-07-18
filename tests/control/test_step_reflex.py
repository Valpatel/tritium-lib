# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the capture-point stepping reflex.

Two things are protected here, in order of importance:

1. **The undisturbed gait is untouched.**  AttitudeStabilizer measures 100%
   upright (34/34) on live Newton; the reflex is an additive layer that must
   not move a single byte of that behavior.  A hardcoded SHA-256 pin holds
   today's trim outputs, and a layered run under the gate must reproduce the
   exact same bytes AND the exact same offsets object.
2. **The step math is closed-form honest.**  Every expectation is computed
   from the linear-inverted-pendulum formula in the test itself — no golden
   blobs — so a failure says which physics changed, not just "digest moved".
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from tritium_lib.control import (
    AttitudeStabilizer,
    LegPlacement,
    ReachLimits,
    StepReflex,
    capture_point,
    step_target,
    velocity_from_impulse,
)
from tritium_lib.control.step_reflex import (
    DEFAULT_CAPTURE_THRESHOLD_M,
    GRAVITY_MPS2,
)

# Go2-class stance layout, body frame (REP-103: +X forward, +Y left).
LEGS = (
    LegPlacement("FL", 0.19, 0.13),
    LegPlacement("FR", 0.19, -0.13),
    LegPlacement("RL", -0.19, 0.13),
    LegPlacement("RR", -0.19, -0.13),
)

WIDE = ReachLimits(max_dx=10.0, max_dy=10.0)  # never clamps
COM_H = 0.31  # Go2-class ride height used throughout


def _canonical(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


# --------------------------------------------------------------------------
# capture_point — the LIP formula itself
# --------------------------------------------------------------------------

class TestCapturePoint:
    def test_matches_closed_form(self):
        vx, vy, z, g = 0.42, 0.0, 0.36, 9.81
        cx, cy = capture_point((vx, vy), z, g)
        assert cx == pytest.approx(vx * math.sqrt(z / g), rel=1e-12)
        assert cy == 0.0

    def test_zero_velocity_is_already_captured(self):
        assert capture_point((0.0, 0.0), COM_H) == (0.0, 0.0)

    def test_offset_is_parallel_to_velocity(self):
        vx, vy = 0.3, -0.4
        cx, cy = capture_point((vx, vy), COM_H)
        # Same direction: cross product zero, dot product positive.
        assert vx * cy - vy * cx == pytest.approx(0.0, abs=1e-15)
        assert vx * cx + vy * cy > 0.0

    def test_scales_linearly_with_speed(self):
        one = capture_point((0.2, 0.1), COM_H)
        two = capture_point((0.4, 0.2), COM_H)
        assert two[0] == pytest.approx(2.0 * one[0], rel=1e-12)
        assert two[1] == pytest.approx(2.0 * one[1], rel=1e-12)

    def test_taller_body_needs_farther_step(self):
        # Time constant sqrt(z/g): a taller pendulum falls slower but its
        # capture point sits farther out for the same velocity.
        low = capture_point((0.5, 0.0), 0.2)
        high = capture_point((0.5, 0.0), 0.8)
        assert high[0] == pytest.approx(2.0 * low[0], rel=1e-12)

    @pytest.mark.parametrize("bad", [(), (1.0,), (1.0, 2.0, 3.0)])
    def test_rejects_wrong_arity(self, bad):
        with pytest.raises(ValueError, match="2 components"):
            capture_point(bad, COM_H)

    def test_rejects_nonpositive_height_and_gravity(self):
        with pytest.raises(ValueError, match="com_height"):
            capture_point((0.1, 0.0), 0.0)
        with pytest.raises(ValueError, match="g must be positive"):
            capture_point((0.1, 0.0), COM_H, 0.0)


class TestVelocityFromImpulse:
    def test_j_over_m(self):
        vx, vy = velocity_from_impulse((5.0, -2.4), 12.0)
        assert vx == pytest.approx(5.0 / 12.0, rel=1e-12)
        assert vy == pytest.approx(-0.2, rel=1e-12)

    def test_rejects_nonpositive_mass(self):
        with pytest.raises(ValueError, match="body_mass"):
            velocity_from_impulse((1.0, 0.0), 0.0)

    def test_rejects_wrong_arity(self):
        with pytest.raises(ValueError, match="2 components"):
            velocity_from_impulse((1.0, 0.0, 0.0), 12.0)


# --------------------------------------------------------------------------
# step_target — leg choice and reach clamping
# --------------------------------------------------------------------------

class TestStepTarget:
    def test_unclamped_target_is_exactly_the_capture_point(self):
        cp = (0.05, 0.30)
        _leg, target = step_target(LEGS, cp, reach_limits=WIDE)
        assert target == pytest.approx(cp, rel=1e-12)

    def test_reach_limits_clamp_to_rectangle_corner(self):
        limits = ReachLimits(max_dx=0.08, max_dy=0.06)
        cp = (5.0, 5.0)  # far outside every reach rectangle
        leg, target = step_target(LEGS, cp, reach_limits=limits)
        # FL is nearest a far ++ capture point; its corner is home + limits.
        assert leg == "FL"
        assert target[0] == pytest.approx(0.19 + 0.08, rel=1e-12)
        assert target[1] == pytest.approx(0.13 + 0.06, rel=1e-12)

    def test_left_push_selects_a_left_leg(self):
        # Push to +Y (left) with a slight forward component: FL's reach
        # rectangle is closest, so FL leaves the least residual divergence.
        leg, _ = step_target(
            LEGS, (0.05, 0.30), reach_limits=ReachLimits(0.1, 0.1),
        )
        assert leg == "FL"

    def test_right_rear_push_selects_the_right_rear_leg(self):
        leg, _ = step_target(
            LEGS, (-0.25, -0.30), reach_limits=ReachLimits(0.1, 0.1),
        )
        assert leg == "RR"

    def test_selected_leg_is_the_one_that_can_arrest(self):
        # Capture point reachable ONLY by FL: inside FL's rectangle, outside
        # everyone else's.  The arresting leg must win over merely-near legs.
        limits = ReachLimits(max_dx=0.1, max_dy=0.1)
        cp = (0.19, 0.20)
        leg, target = step_target(LEGS, cp, reach_limits=limits)
        assert leg == "FL"
        assert target == pytest.approx(cp, rel=1e-12)  # residual zero

    def test_exact_tie_resolves_to_input_order_not_hash_order(self):
        # A pure-lateral capture point sits symmetrically between FL and RL:
        # identical residuals, so the FIRST leg in input order must win, and
        # reversing the order must flip the answer.
        limits = ReachLimits(max_dx=0.1, max_dy=0.1)
        cp = (0.0, 0.35)
        leg_fwd, _ = step_target(LEGS, cp, reach_limits=limits)
        leg_rev, _ = step_target(tuple(reversed(LEGS)), cp, reach_limits=limits)
        assert leg_fwd == "FL"
        assert leg_rev == "RL"

    def test_rejects_empty_and_duplicate_layouts(self):
        with pytest.raises(ValueError, match="at least one leg"):
            step_target((), (0.1, 0.0), reach_limits=WIDE)
        dup = (LegPlacement("FL", 0.1, 0.1), LegPlacement("FL", -0.1, 0.1))
        with pytest.raises(ValueError, match="duplicate"):
            step_target(dup, (0.1, 0.0), reach_limits=WIDE)

    def test_rejects_bad_capture_point_arity(self):
        with pytest.raises(ValueError, match="2 components"):
            step_target(LEGS, (0.1, 0.0, 0.0), reach_limits=WIDE)

    def test_reach_limits_must_be_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            ReachLimits(max_dx=0.0, max_dy=0.1)


# --------------------------------------------------------------------------
# StepReflex — the gate
# --------------------------------------------------------------------------

class TestGate:
    def test_below_threshold_no_step(self):
        reflex = StepReflex(com_height_m=COM_H)
        # 0.1 m/s -> ~0.018 m capture excursion, well under the 0.05 gate.
        decision = reflex.decide((0.1, 0.0), LEGS, reach_limits=WIDE)
        assert decision.step is None
        assert not decision.stepping

    def test_zero_velocity_no_step(self):
        reflex = StepReflex(com_height_m=COM_H)
        decision = reflex.decide((0.0, 0.0), LEGS, reach_limits=WIDE)
        assert decision.step is None
        assert decision.capture_pt == (0.0, 0.0)
        assert decision.capture_distance_m == 0.0

    def test_exactly_at_threshold_stays_closed(self):
        # The gate fires strictly ABOVE threshold.  Build the threshold from
        # the identical computation path so the equality is byte-exact.
        vel = (0.0, 0.4)
        cp = capture_point(vel, COM_H)
        distance = math.hypot(cp[0], cp[1])
        reflex = StepReflex(com_height_m=COM_H, threshold_m=distance)
        assert reflex.decide(vel, LEGS, reach_limits=WIDE).step is None

    def test_pass_through_offsets_are_the_same_object(self):
        offsets = {"FL": 0.01, "FR": -0.01, "RL": 0.0, "RR": 0.0}
        reflex = StepReflex(com_height_m=COM_H)
        decision = reflex.decide(
            (0.05, 0.0), LEGS, reach_limits=WIDE, leg_height_offsets=offsets,
        )
        assert decision.leg_height_offsets is offsets  # identity, not a copy
        assert decision.step is None

    def test_lateral_push_steps_in_the_push_direction(self):
        # A +Y (leftward) shove: the step target must displace to +Y by the
        # closed-form LIP magnitude — with wide limits, exactly v*sqrt(z/g).
        vel = (0.0, 0.6)
        reflex = StepReflex(com_height_m=COM_H)
        decision = reflex.decide(vel, LEGS, reach_limits=WIDE)
        assert decision.step is not None
        expected_y = 0.6 * math.sqrt(COM_H / GRAVITY_MPS2)
        assert decision.step.foot_target[0] == pytest.approx(0.0, abs=1e-15)
        assert decision.step.foot_target[1] == pytest.approx(
            expected_y, rel=1e-12,
        )
        assert decision.step.leg in ("FL", "RL")  # a left-side leg
        assert decision.step.residual_m == pytest.approx(0.0, abs=1e-15)

    def test_measured_impulse_drives_the_same_decision(self):
        # 7.2 N*s lateral on a 12 kg body = 0.6 m/s — above the ~5 N*s
        # envelope the trim handles, so the reflex must engage.
        vel = velocity_from_impulse((0.0, 7.2), 12.0)
        assert vel == pytest.approx((0.0, 0.6), rel=1e-12)
        decision = StepReflex(com_height_m=COM_H).decide(
            vel, LEGS, reach_limits=WIDE,
        )
        assert decision.stepping

    def test_clamped_step_reports_honest_residual(self):
        limits = ReachLimits(max_dx=0.1, max_dy=0.1)
        vel = (0.0, 1.5)  # violent shove: capture point beyond every reach
        reflex = StepReflex(com_height_m=COM_H)
        decision = reflex.decide(vel, LEGS, reach_limits=limits)
        assert decision.step is not None
        cp = capture_point(vel, COM_H)
        tx, ty = decision.step.foot_target
        assert decision.step.residual_m == pytest.approx(
            math.hypot(tx - cp[0], ty - cp[1]), rel=1e-12,
        )
        assert decision.step.residual_m > 0.0

    def test_config_validation(self):
        with pytest.raises(ValueError, match="com_height_m"):
            StepReflex(com_height_m=0.0)
        with pytest.raises(ValueError, match="threshold_m"):
            StepReflex(com_height_m=COM_H, threshold_m=-0.1)
        with pytest.raises(ValueError, match="g_mps2"):
            StepReflex(com_height_m=COM_H, g_mps2=0.0)

    def test_default_threshold_sits_below_measured_inversion(self):
        # The trim-only stack inverts above ~5 N*s (~0.42 m/s on 12 kg,
        # ~0.07 m capture excursion at Go2 ride height).  The default gate
        # must open BEFORE that regime, not at it.
        excursion = math.hypot(
            *capture_point(velocity_from_impulse((5.0, 0.0), 12.0), COM_H)
        )
        assert DEFAULT_CAPTURE_THRESHOLD_M < excursion


# --------------------------------------------------------------------------
# Regression pin — the undisturbed trim is byte-identical to today
# --------------------------------------------------------------------------

def _quat_roll(deg: float) -> tuple[float, float, float, float]:
    h = math.radians(deg) / 2.0
    return (math.cos(h), math.sin(h), 0.0, 0.0)


def _quat_pitch(deg: float) -> tuple[float, float, float, float]:
    h = math.radians(deg) / 2.0
    return (math.cos(h), 0.0, math.sin(h), 0.0)


# A fixed undisturbed-walk attitude history: level, small roll, small pitch,
# a non-unit mixed-axis solver read, settling back.  dt = 50 Hz.
_PIN_QUATS = (
    (1.0, 0.0, 0.0, 0.0),
    _quat_roll(10.0),
    _quat_pitch(8.0),
    (0.9, 0.05, -0.04, 0.02),
    (0.999, 0.01, -0.01, 0.0),
)

# SHA-256 of the canonical trim trace below, captured on the code that
# measured 100% upright (34/34) on live Newton.  Recompute ONLY as part of a
# deliberate, measured retune of AttitudeStabilizer — if this moves as a side
# effect of another change, that change just altered the shipped gait.
_PIN_DIGEST = "6e5d1be2da70dbb59b79afad9dd409e76e6ebf04b636c679728e4bf8646af131"


def _trim_trace() -> list[dict]:
    stab = AttitudeStabilizer()
    rows = []
    for quat in _PIN_QUATS:
        c = stab.update(quat, 0.02)
        rows.append({
            "roll_deg": c.roll_deg,
            "pitch_deg": c.pitch_deg,
            "roll_rate_dps": c.roll_rate_dps,
            "pitch_rate_dps": c.pitch_rate_dps,
            "roll_cmd": c.roll_cmd,
            "pitch_cmd": c.pitch_cmd,
            "offsets": c.leg_height_offsets(LEGS),
        })
    return rows


class TestUndisturbedRegression:
    def test_stabilizer_output_matches_the_pinned_digest(self):
        digest = hashlib.sha256(_canonical(_trim_trace())).hexdigest()
        assert digest == _PIN_DIGEST, (
            "AttitudeStabilizer's undisturbed output moved.  That behavior "
            "measures 100% upright on live Newton; the stepping reflex was "
            "required to be additive.  Revert, or re-measure before re-pinning."
        )

    def test_layered_reflex_under_gate_changes_nothing(self):
        # Run the identical attitude history twice: bare trim, and trim with
        # the reflex layered on at sub-threshold velocity.  The trim bytes
        # must be identical, and the offsets objects the very same dicts.
        bare = _trim_trace()

        stab = AttitudeStabilizer()
        reflex = StepReflex(com_height_m=COM_H)
        layered = []
        for quat in _PIN_QUATS:
            c = stab.update(quat, 0.02)
            offsets = c.leg_height_offsets(LEGS)
            decision = reflex.decide(
                (0.02, -0.01),  # undisturbed-walk drift, far under the gate
                LEGS,
                reach_limits=WIDE,
                leg_height_offsets=offsets,
            )
            assert decision.step is None
            assert decision.leg_height_offsets is offsets
            layered.append({
                "roll_deg": c.roll_deg,
                "pitch_deg": c.pitch_deg,
                "roll_rate_dps": c.roll_rate_dps,
                "pitch_rate_dps": c.pitch_rate_dps,
                "roll_cmd": c.roll_cmd,
                "pitch_cmd": c.pitch_cmd,
                "offsets": decision.leg_height_offsets,
            })

        assert _canonical(layered) == _canonical(bare)
        assert (
            hashlib.sha256(_canonical(layered)).hexdigest() == _PIN_DIGEST
        )


# --------------------------------------------------------------------------
# Determinism — in-process repeatability (hash-seed and wall-clock
# independence are pinned in tests/test_determinism_new_surface.py, which
# rebuilds this surface in subprocesses under different PYTHONHASHSEED)
# --------------------------------------------------------------------------

def _reflex_payload() -> dict:
    reflex = StepReflex(com_height_m=COM_H)
    limits = ReachLimits(max_dx=0.1, max_dy=0.08)
    velocities = (
        (0.0, 0.0), (0.03, -0.02), (0.0, 0.6), (-0.5, 0.4), (1.5, 0.0),
    )
    return {
        "decisions": [
            reflex.decide(
                v, LEGS, reach_limits=limits,
                leg_height_offsets={"FL": 0.01, "FR": -0.01},
            ).as_dict()
            for v in velocities
        ],
        "capture": [list(capture_point(v, COM_H)) for v in velocities],
    }


class TestDeterminism:
    def test_payload_is_byte_identical_across_repeated_calls(self):
        first = _canonical(_reflex_payload())
        for _ in range(3):
            assert _canonical(_reflex_payload()) == first
