# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the capture-point stepping reflex (deviation-gated revision).

Three things are protected here, in order of importance:

1. **The measured live failure is pinned.**  The first shipped reflex gated
   on ABSOLUTE capture point (default 0.05 m) and, live on Newton
   (2026-07-17, undisturbed, matched trials), turned a 6/6-upright gait into
   0/6 — a healthy trot's capture point (0.101–0.131 m measured) crosses an
   absolute gate every stride.  The steady-walk tests below construct that
   exact regime (absolute capture point above the disproven gate) and assert
   the deviation gate stays SHUT through it.
2. **The undisturbed trim is untouched.**  AttitudeStabilizer measures 100%
   upright (34/34) on live Newton; the reflex is an additive layer that must
   not move a single byte of that behavior.  A hardcoded SHA-256 pin holds
   today's trim outputs, and a layered run under the gate — now AT walking
   speed, inside the measured failure band — must reproduce the exact same
   bytes AND the exact same offsets object.
3. **The step math is closed-form honest.**  Every expectation is computed
   from the linear-inverted-pendulum formula in the test itself — no golden
   blobs — so a failure says which physics changed, not just "digest moved".

The live re-test of the deviation gate has since RUN, and it disproved the
walking use too (live Newton 2026-07-17/18: gate open on 100.0% of walking
ticks at the legal nominal, baseline 6/6 upright vs reflex 0/5, Fisher
p = 0.0022 — full numbers and three caveats in the module docstring).  The
tests below are design-conformance pins of the math, the STANDING regime
(the only supported one), and the layering contract — plus pins that the
measured verdict stays stated in the module's own documentation, because
documentation is now the only thing between a future integrator and the
measured 0/5 walking faceplant.
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
    velocity_deviation,
    velocity_from_impulse,
)
from tritium_lib.control.step_reflex import (
    DEFAULT_DEVIATION_THRESHOLD_M,
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

# The disproven absolute gate (2026-07-17 live failure: 6/6 -> 0/6
# undisturbed).  Kept here as a literal, not an import — the constant was
# deliberately removed from the module.
DISPROVEN_ABSOLUTE_GATE_M = 0.05

# Measured 5 N*s push contribution to capture-point excursion, live Newton.
MEASURED_PUSH_CONTRIBUTION_LOW_M = 0.058
MEASURED_PUSH_CONTRIBUTION_HIGH_M = 0.087


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


class TestVelocityDeviation:
    def test_componentwise_subtraction(self):
        assert velocity_deviation((0.72, -0.05), (0.60, 0.0)) == pytest.approx(
            (0.12, -0.05), rel=1e-12,
        )

    def test_push_left_during_forward_walk_deviates_left_only(self):
        # The whole point of the redesign: a lateral shove during a forward
        # trot must appear as a PURELY lateral deviation — the forward
        # carrier subtracts out as common mode.
        dev = velocity_deviation((0.60, 0.50), (0.60, 0.0))
        assert dev[0] == pytest.approx(0.0, abs=1e-15)
        assert dev[1] == pytest.approx(0.50, rel=1e-12)

    def test_falling_behind_the_gait_deviates_backward(self):
        # Blocked/tripped: measured slower than commanded.  The deviation —
        # and therefore the recovery direction — points BACKWARD (-X).
        dev = velocity_deviation((0.20, 0.0), (0.60, 0.0))
        assert dev == pytest.approx((-0.40, 0.0), rel=1e-12)

    def test_standing_nominal_is_the_identity(self):
        # For a body commanded to stand still, deviation == measured: the
        # gate degenerates exactly to the old absolute behavior, which is
        # the one regime the absolute design was ever right in.
        assert velocity_deviation((0.3, -0.4), (0.0, 0.0)) == (0.3, -0.4)

    def test_rejects_wrong_arity_on_either_argument(self):
        with pytest.raises(ValueError, match="measured_vel_xy"):
            velocity_deviation((1.0,), (0.0, 0.0))
        with pytest.raises(ValueError, match="nominal_vel_xy"):
            velocity_deviation((1.0, 0.0), (0.0, 0.0, 0.0))


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
# StepReflex — the deviation gate
# --------------------------------------------------------------------------

# A stride's worth of measured-velocity samples for a trot commanded at
# NOMINAL_WALK: the CoM speeds up and slows down through stance and swing,
# oscillating about the commanded mean.  Chosen so every sample's ABSOLUTE
# capture point exceeds the disproven 0.05 m gate (several inside the
# measured 0.101–0.131 m band) while every DEVIATION stays under the new
# gate — the exact regime that measured 6/6 -> 0/6 live.
NOMINAL_WALK = (0.60, 0.0)
STRIDE_SAMPLES = (
    (0.45, 0.05),
    (0.60, -0.08),
    (0.74, 0.03),
    (0.52, -0.06),
    (0.68, 0.07),
)


class TestGate:
    def test_steady_walk_at_nominal_never_opens_the_gate(self):
        # THE regression this module shipped broken on.  Live Newton
        # 2026-07-17: absolute gating opened essentially every stride of an
        # undisturbed 1.2 m/s trot (capture point 0.101–0.131 m vs a 0.05 m
        # gate) and flipped 6/6 upright to 0/6.  With the nominal supplied,
        # the same walking carrier must subtract out and the gate stay SHUT.
        reflex = StepReflex(com_height_m=COM_H)
        for measured in STRIDE_SAMPLES:
            # Prove each sample IS in the old failure regime first…
            absolute = math.hypot(*capture_point(measured, COM_H))
            assert absolute > DISPROVEN_ABSOLUTE_GATE_M, (
                f"test setup broken: {measured} would not have tripped the "
                "old absolute gate, so it does not pin the failure"
            )
            # …then that the deviation gate stays closed through it.
            decision = reflex.decide(
                measured, LEGS,
                nominal_vel_xy=NOMINAL_WALK, reach_limits=WIDE,
            )
            assert decision.step is None, (
                f"gate opened on an undisturbed stride sample {measured} — "
                "this is the exact failure that measured 0/6 upright live"
            )
            assert not decision.stepping

    def test_stride_samples_span_the_measured_carrier_band(self):
        # At least one undisturbed sample must sit inside the LIVE-measured
        # 0.101–0.131 m absolute band, or the pin above is weaker than the
        # real failure.
        absolutes = [
            math.hypot(*capture_point(m, COM_H)) for m in STRIDE_SAMPLES
        ]
        assert any(0.101 <= a <= 0.131 for a in absolutes)

    def test_push_on_top_of_nominal_opens_the_gate(self):
        # The same walking carrier plus a lateral shove: deviation is the
        # shove alone, and it must fire.  0.50 m/s of deviation at Go2 ride
        # height is ~0.089 m of deviation capture point — inside the
        # measured 0.058–0.087 push band's upper edge.
        reflex = StepReflex(com_height_m=COM_H)
        decision = reflex.decide(
            (0.60, 0.50), LEGS,
            nominal_vel_xy=NOMINAL_WALK, reach_limits=WIDE,
        )
        assert decision.stepping
        expected = 0.50 * math.sqrt(COM_H / GRAVITY_MPS2)
        assert decision.deviation_distance_m == pytest.approx(
            expected, rel=1e-12,
        )

    def test_trim_ceiling_push_during_walk_fires(self):
        # 5 N*s on 12 kg — the measured inversion ceiling of the trim-only
        # stack — delivered laterally during the nominal walk.  Unopposed
        # dv = 0.4167 m/s -> ~0.074 m of deviation capture point: above the
        # default gate, so the reflex engages before the trim's ceiling.
        dv = velocity_from_impulse((0.0, 5.0), 12.0)
        measured = (NOMINAL_WALK[0] + dv[0], NOMINAL_WALK[1] + dv[1])
        decision = StepReflex(com_height_m=COM_H).decide(
            measured, LEGS, nominal_vel_xy=NOMINAL_WALK, reach_limits=WIDE,
        )
        assert decision.stepping
        assert decision.deviation_distance_m > DEFAULT_DEVIATION_THRESHOLD_M

    def test_deviation_direction_is_the_push_not_the_walk(self):
        # Lateral push during forward walk: the deviation capture point must
        # be PURELY lateral — zero forward component — even though the body
        # is moving forward the whole time.
        decision = StepReflex(com_height_m=COM_H).decide(
            (0.60, 0.50), LEGS,
            nominal_vel_xy=NOMINAL_WALK, reach_limits=WIDE,
        )
        assert decision.deviation_vel_xy == pytest.approx(
            (0.0, 0.50), abs=1e-15,
        )
        assert decision.deviation_capture_pt[0] == pytest.approx(
            0.0, abs=1e-15,
        )
        assert decision.deviation_capture_pt[1] == pytest.approx(
            0.50 * math.sqrt(COM_H / GRAVITY_MPS2), rel=1e-12,
        )

    def test_falling_behind_the_gait_is_a_backward_disturbance(self):
        # Tripped/blocked: measured well below commanded.  The deviation
        # points -X, the gate opens, and the sign is preserved end-to-end.
        decision = StepReflex(com_height_m=COM_H).decide(
            (0.20, 0.0), LEGS,
            nominal_vel_xy=NOMINAL_WALK, reach_limits=WIDE,
        )
        assert decision.deviation_vel_xy == pytest.approx(
            (-0.40, 0.0), rel=1e-12,
        )
        assert decision.deviation_capture_pt[0] < 0.0
        assert decision.stepping

    def test_step_targets_the_total_capture_point_arrest(self):
        # When the gate opens the step is an arrest-to-stand: with wide
        # limits the landing point is exactly the TOTAL-velocity capture
        # point (carrier + disturbance), the LIP point that brings the whole
        # body to rest.  (This policy is NOT live-validated — see module
        # docstring — but it must at least be the policy the code claims.)
        measured = (0.60, 0.50)
        decision = StepReflex(com_height_m=COM_H).decide(
            measured, LEGS, nominal_vel_xy=NOMINAL_WALK, reach_limits=WIDE,
        )
        assert decision.step is not None
        total_cp = capture_point(measured, COM_H)
        assert decision.step.foot_target == pytest.approx(total_cp, rel=1e-12)
        assert decision.step.residual_m == pytest.approx(0.0, abs=1e-12)
        assert decision.capture_pt == pytest.approx(total_cp, rel=1e-12)

    def test_standing_body_degenerates_to_absolute_gating(self):
        # nominal (0,0) is the honest nominal for a stand — and there the
        # deviation gate IS the absolute gate: a lateral shove steps in the
        # shove direction by the closed-form LIP magnitude.
        vel = (0.0, 0.6)
        reflex = StepReflex(com_height_m=COM_H)
        decision = reflex.decide(
            vel, LEGS, nominal_vel_xy=(0.0, 0.0), reach_limits=WIDE,
        )
        assert decision.step is not None
        expected_y = 0.6 * math.sqrt(COM_H / GRAVITY_MPS2)
        assert decision.step.foot_target[0] == pytest.approx(0.0, abs=1e-15)
        assert decision.step.foot_target[1] == pytest.approx(
            expected_y, rel=1e-12,
        )
        assert decision.step.leg in ("FL", "RL")  # a left-side leg
        assert decision.step.residual_m == pytest.approx(0.0, abs=1e-15)

    def test_zero_deviation_zero_velocity_no_step(self):
        reflex = StepReflex(com_height_m=COM_H)
        decision = reflex.decide(
            (0.0, 0.0), LEGS, nominal_vel_xy=(0.0, 0.0), reach_limits=WIDE,
        )
        assert decision.step is None
        assert decision.deviation_capture_pt == (0.0, 0.0)
        assert decision.deviation_distance_m == 0.0

    def test_exactly_at_threshold_stays_closed(self):
        # The gate fires strictly ABOVE threshold.  Build the threshold from
        # the identical computation path so the equality is byte-exact.
        measured, nominal = (0.60, 0.40), NOMINAL_WALK
        dev = velocity_deviation(measured, nominal)
        cp = capture_point(dev, COM_H)
        distance = math.hypot(cp[0], cp[1])
        reflex = StepReflex(com_height_m=COM_H, threshold_m=distance)
        decision = reflex.decide(
            measured, LEGS, nominal_vel_xy=nominal, reach_limits=WIDE,
        )
        assert decision.step is None

    def test_pass_through_offsets_are_the_same_object(self):
        offsets = {"FL": 0.01, "FR": -0.01, "RL": 0.0, "RR": 0.0}
        reflex = StepReflex(com_height_m=COM_H)
        decision = reflex.decide(
            (0.62, 0.02), LEGS,
            nominal_vel_xy=NOMINAL_WALK, reach_limits=WIDE,
            leg_height_offsets=offsets,
        )
        assert decision.leg_height_offsets is offsets  # identity, not a copy
        assert decision.step is None

    def test_clamped_step_reports_honest_residual(self):
        limits = ReachLimits(max_dx=0.1, max_dy=0.1)
        measured = (0.60, 1.50)  # violent shove during the walk
        reflex = StepReflex(com_height_m=COM_H)
        decision = reflex.decide(
            measured, LEGS, nominal_vel_xy=NOMINAL_WALK, reach_limits=limits,
        )
        assert decision.step is not None
        cp = capture_point(measured, COM_H)
        tx, ty = decision.step.foot_target
        assert decision.step.residual_m == pytest.approx(
            math.hypot(tx - cp[0], ty - cp[1]), rel=1e-12,
        )
        assert decision.step.residual_m > 0.0


class TestNominalIsRequired:
    """The API must fail LOUDLY without a nominal — never fall back to the
    absolute-velocity behavior that measured 0/6 upright undisturbed."""

    def test_old_call_shape_without_nominal_raises_type_error(self):
        # The exact call every pre-revision caller makes.  It must not run.
        reflex = StepReflex(com_height_m=COM_H)
        with pytest.raises(TypeError):
            reflex.decide((0.60, 0.0), LEGS, reach_limits=WIDE)

    def test_nominal_none_raises_with_an_explanation(self):
        reflex = StepReflex(com_height_m=COM_H)
        with pytest.raises(ValueError, match="0/6"):
            reflex.decide(
                (0.60, 0.0), LEGS, nominal_vel_xy=None, reach_limits=WIDE,
            )

    def test_nominal_wrong_arity_raises(self):
        reflex = StepReflex(com_height_m=COM_H)
        with pytest.raises(ValueError, match="nominal_vel_xy"):
            reflex.decide(
                (0.60, 0.0), LEGS, nominal_vel_xy=(0.6,), reach_limits=WIDE,
            )


class TestVerdictStaysStated:
    """The measured walking verdict must remain in the module's face.

    The math is correct and the module ships for the standing regime, so
    nothing *executable* stops a future integrator from wiring it to a
    walking body — only the documentation does.  Pin its load-bearing
    phrases so a docstring rewrite cannot quietly drop the verdict.
    """

    def test_module_docstring_states_the_walking_verdict(self):
        import tritium_lib.control.step_reflex as mod

        doc = mod.__doc__
        for phrase in (
            "cannot separate a push from the gait",  # the conclusion
            "DO NOT WIRE",                            # the instruction
            "WALKING BODY",                           # its scope
            "STANDING",                               # the surviving regime
            "contact/force",                          # the required rewire
            "100.0%",                                 # gate-open rate, legal nominal
            "0/5",                                    # live A/B reflex arm
            "6/6",                                    # live A/B baseline arm
            "p = 0.0022",                             # its significance
            "2350/2350",                              # pooled gate-open ticks
            "0.264",                                  # measured quiet floor
        ):
            assert phrase in doc, (
                f"module docstring lost the verdict phrase {phrase!r}; the "
                "2026-07-17/18 measured verdict must stay stated in full"
            )

    def test_module_docstring_states_all_three_caveats(self):
        import tritium_lib.control.step_reflex as mod

        doc = mod.__doc__
        for phrase in (
            "No visual evidence",   # caveat 1: numeric telemetry only
            "control arm",          # caveat 2: gate-shut arm never ran
            "~18%",                 # caveat 3: push signature's gait quality
        ):
            assert phrase in doc, (
                f"module docstring lost caveat phrase {phrase!r}; the verdict "
                "must not be overstated either"
            )

    def test_class_docstring_carries_the_verdict_too(self):
        # Editors surface the class docstring, not the module's — the
        # warning must live where autocomplete shows it.
        doc = StepReflex.__doc__
        assert "STANDING bodies only" in doc
        assert "do not wire this to a" in doc
        assert "0/5" in doc


class TestConfig:
    def test_config_validation(self):
        with pytest.raises(ValueError, match="com_height_m"):
            StepReflex(com_height_m=0.0)
        with pytest.raises(ValueError, match="threshold_m"):
            StepReflex(com_height_m=COM_H, threshold_m=-0.1)
        with pytest.raises(ValueError, match="g_mps2"):
            StepReflex(com_height_m=COM_H, g_mps2=0.0)

    def test_default_gate_sits_below_the_measured_push_signal(self):
        # The gate must catch the WEAKEST measured signature of a
        # trim-ceiling push (5 N*s adds 0.058–0.087 m of capture point, and
        # a push survives into deviation space at full size).
        assert (
            DEFAULT_DEVIATION_THRESHOLD_M < MEASURED_PUSH_CONTRIBUTION_LOW_M
        )
        # Cross-check against the closed form: 5 N*s / 12 kg at Go2 ride
        # height lands inside the measured band, above the gate.
        closed_form = math.hypot(
            *capture_point(velocity_from_impulse((5.0, 0.0), 12.0), COM_H)
        )
        assert (
            MEASURED_PUSH_CONTRIBUTION_LOW_M
            <= closed_form
            <= MEASURED_PUSH_CONTRIBUTION_HIGH_M
        )
        assert DEFAULT_DEVIATION_THRESHOLD_M < closed_form

    def test_default_gate_is_not_the_disproven_absolute_default(self):
        # 0.05 m ABSOLUTE was measured killing the gait.  The deviation gate
        # is a different signal, but shipping the same number would invite
        # the same copy-paste back into absolute gating; pin the separation.
        assert DEFAULT_DEVIATION_THRESHOLD_M != DISPROVEN_ABSOLUTE_GATE_M


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

# The layered pin below runs the reflex AT WALKING SPEED: each measured
# sample's absolute capture point exceeds the disproven 0.05 m gate, so the
# old code would have fired on every tick of this very trace.  The deviation
# gate must stay shut and the trim bytes must not move.
_PIN_MEASURED_VELS = (
    (0.62, -0.03),
    (0.55, 0.04),
    (0.71, 0.02),
    (0.58, -0.05),
    (0.66, 0.03),
)


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
        # the reflex layered on at WALKING speed with the nominal supplied.
        # The trim bytes must be identical, and the offsets objects the very
        # same dicts.  This is the live 2026-07-17 failure condition
        # replayed against the corrected gate.
        bare = _trim_trace()

        stab = AttitudeStabilizer()
        reflex = StepReflex(com_height_m=COM_H)
        layered = []
        for quat, measured in zip(_PIN_QUATS, _PIN_MEASURED_VELS):
            # Each sample would have tripped the disproven absolute gate.
            absolute = math.hypot(*capture_point(measured, COM_H))
            assert absolute > DISPROVEN_ABSOLUTE_GATE_M

            c = stab.update(quat, 0.02)
            offsets = c.leg_height_offsets(LEGS)
            decision = reflex.decide(
                measured,
                LEGS,
                nominal_vel_xy=NOMINAL_WALK,
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
    cases = (
        # (measured, nominal)
        ((0.0, 0.0), (0.0, 0.0)),       # standing, captured — closed
        ((0.62, -0.03), NOMINAL_WALK),  # walking at nominal — closed
        ((0.60, 0.50), NOMINAL_WALK),   # lateral shove on the walk — fires
        ((0.20, 0.0), NOMINAL_WALK),    # blocked/tripped — fires backward
        ((0.0, 0.6), (0.0, 0.0)),       # standing shove — fires
        ((-0.5, 0.4), (0.0, 0.0)),      # standing diagonal shove — fires
        ((2.1, 0.0), NOMINAL_WALK),     # violent — clamped, honest residual
    )
    return {
        "decisions": [
            reflex.decide(
                measured, LEGS,
                nominal_vel_xy=nominal, reach_limits=limits,
                leg_height_offsets={"FL": 0.01, "FR": -0.01},
            ).as_dict()
            for measured, nominal in cases
        ],
        "deviation": [
            list(velocity_deviation(measured, nominal))
            for measured, nominal in cases
        ],
        "capture": [list(capture_point(m, COM_H)) for m, _ in cases],
    }


class TestDeterminism:
    def test_payload_is_byte_identical_across_repeated_calls(self):
        first = _canonical(_reflex_payload())
        for _ in range(3):
            assert _canonical(_reflex_payload()) == first
