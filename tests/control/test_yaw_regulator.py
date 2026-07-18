# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the heading-hold yaw regulator.

Three things are protected here, in order of importance:

1. **The undisturbed gait is untouched.**  A body exactly on its commanded
   heading with no yaw rate gets a correction of EXACTLY ``0.0`` — pinned at
   the byte level — and folding that into a turn command reproduces the same
   float, bit for bit.  The regulator composes over the measured-good gait
   without moving it.
2. **The signs are right.**  A wrong-sign heading regulator is positive
   feedback that ACCELERATES the +/-15 deg/run drift it exists to null, so
   the compass-clockwise convention is checked end to end: wrap direction,
   correction direction, and the motor split that
   :func:`~tritium_lib.models.body.motors_from_intent` (and the edge tier's
   ``twist_to_motors``, same pinned contract) produces from it.
3. **The damping claim is demonstrated, not asserted.**  A lagged closed-loop
   plant is run with ``kd = 0`` and ``kd > 0`` and the overshoot ordering the
   module docstring promises is required to hold.  Simulation only — the
   module says so honestly — but the claim is at least earned against
   dynamics rather than stated.
"""

from __future__ import annotations

import struct

import pytest

from tritium_lib.control import (
    YawRegulator,
    heading_error_deg,
)
from tritium_lib.control.yaw_regulator import (
    DEFAULT_KD,
    DEFAULT_KP,
    DEFAULT_MAX_CORRECTION_DPS,
    _closed_loop_overshoot,
)
from tritium_lib.models.body import ControlIntent, motors_from_intent


def _bits(value: float) -> bytes:
    """The IEEE-754 bytes of a float — equality here is byte identity."""
    return struct.pack("<d", value)


# --------------------------------------------------------------------------
# heading_error_deg — the wrap is the entire difficulty
# --------------------------------------------------------------------------

class TestHeadingError:
    def test_error_is_commanded_minus_measured(self):
        assert heading_error_deg(10.0, 30.0) == pytest.approx(20.0)
        assert heading_error_deg(30.0, 10.0) == pytest.approx(-20.0)

    def test_wrap_across_north_goes_the_short_way(self):
        # Measured 359, commanded 1: two degrees clockwise, NOT -358.
        assert heading_error_deg(359.0, 1.0) == pytest.approx(2.0)
        assert heading_error_deg(1.0, 359.0) == pytest.approx(-2.0)

    def test_equivalent_headings_are_zero_error(self):
        assert heading_error_deg(180.0, -180.0) == 0.0
        assert heading_error_deg(0.0, 360.0) == 0.0
        assert heading_error_deg(-90.0, 270.0) == 0.0

    def test_range_is_half_open(self):
        # Exactly opposite: both ways are equally short; the contract picks
        # -180 (the closed end of [-180, 180)) deterministically.
        assert heading_error_deg(0.0, 180.0) == -180.0
        assert heading_error_deg(90.0, 270.0) == -180.0
        # And +180 is never emitted.
        for measured in (0.0, 37.0, 359.9, -720.0):
            err = heading_error_deg(measured, measured + 180.0)
            assert -180.0 <= err < 180.0

    def test_unnormalized_inputs_are_folded(self):
        assert heading_error_deg(725.0, 5.0) == 0.0
        assert heading_error_deg(-355.0, 3.0) == pytest.approx(-2.0)


# --------------------------------------------------------------------------
# Zero-error no-op — the layering contract, pinned at the byte level
# --------------------------------------------------------------------------

class TestZeroErrorNoOp:
    def test_on_heading_correction_is_byte_identical_zero(self):
        regulator = YawRegulator()
        for heading in (0.0, 90.0, 179.5, -180.0, 359.0):
            corr = regulator.correct(heading, heading)
            assert _bits(corr.correction_dps) == _bits(0.0)
            assert corr.error_deg == 0.0
            assert not corr.saturated

    def test_wrapped_equivalent_heading_is_also_a_no_op(self):
        corr = YawRegulator().correct(360.0, 0.0)
        assert _bits(corr.correction_dps) == _bits(0.0)

    def test_turn_intent_addend_is_byte_identical_zero(self):
        corr = YawRegulator().correct(45.0, 45.0)
        assert _bits(corr.turn_intent(60.0)) == _bits(0.0)
        # Folding into a gait turn command reproduces the same float.
        gait_turn = 0.37
        assert _bits(gait_turn + corr.turn_intent(60.0)) == _bits(gait_turn)

    def test_hold_returns_gait_turn_unchanged(self):
        regulator = YawRegulator()
        for gait_turn in (0.0, 0.37, -0.8, 1.0):
            held = regulator.hold(90.0, 90.0, gait_turn, 60.0)
            assert _bits(held) == _bits(gait_turn)

    def test_zero_rate_supplied_explicitly_is_still_a_no_op(self):
        corr = YawRegulator().correct(90.0, 90.0, measured_yaw_rate_dps=0.0)
        assert _bits(corr.correction_dps) == _bits(0.0)


# --------------------------------------------------------------------------
# Sign convention — wrong sign is positive feedback, so check end to end
# --------------------------------------------------------------------------

class TestSignConvention:
    def test_commanded_clockwise_of_measured_gives_positive_correction(self):
        # Facing north (0), told east (90): turn clockwise -> positive.
        corr = YawRegulator().correct(0.0, 90.0)
        assert corr.error_deg > 0.0
        assert corr.correction_dps > 0.0

    def test_commanded_counterclockwise_gives_negative_correction(self):
        # Facing east (90), told north (0): turn counter-clockwise.
        corr = YawRegulator().correct(90.0, 0.0)
        assert corr.correction_dps < 0.0

    def test_wrap_case_produces_small_positive_not_large_negative(self):
        corr = YawRegulator(kp=1.0, kd=0.0).correct(359.0, 1.0)
        assert corr.error_deg == pytest.approx(2.0)
        assert corr.correction_dps == pytest.approx(2.0)
        assert not corr.saturated

    def test_positive_correction_drives_left_faster_than_right(self):
        # The full chain: positive correction -> positive ControlIntent.turn
        # -> left = f + t/2 > right = f - t/2, the pinned compass-CLOCKWISE
        # contract shared by motors_from_intent and the edge tier's
        # twist_to_motors.
        corr = YawRegulator().correct(0.0, 45.0)
        turn = corr.turn_intent(60.0)
        assert turn > 0.0
        left, right = motors_from_intent(
            ControlIntent(forward=0.5, turn=turn)
        )
        assert left > right

    def test_negative_correction_drives_right_faster_than_left(self):
        corr = YawRegulator().correct(45.0, 0.0)
        left, right = motors_from_intent(
            ControlIntent(forward=0.5, turn=corr.turn_intent(60.0))
        )
        assert right > left

    def test_damping_opposes_the_measured_rate(self):
        # On heading but spinning clockwise: the only demand is damping,
        # and it must be counter-clockwise (negative).
        corr = YawRegulator(kp=1.0, kd=0.5).correct(
            90.0, 90.0, measured_yaw_rate_dps=20.0,
        )
        assert corr.correction_dps == pytest.approx(-10.0)

    def test_rate_toward_target_reduces_the_correction(self):
        regulator = YawRegulator(kp=1.0, kd=0.5)
        still = regulator.correct(0.0, 20.0, measured_yaw_rate_dps=0.0)
        already_turning = regulator.correct(
            0.0, 20.0, measured_yaw_rate_dps=10.0,
        )
        assert already_turning.correction_dps < still.correction_dps


# --------------------------------------------------------------------------
# Clamping — bounded authority, honest saturation report
# --------------------------------------------------------------------------

class TestClamping:
    def test_large_error_clamps_exactly_at_max(self):
        corr = YawRegulator(kp=1.5, kd=0.0, max_correction_dps=30.0).correct(
            0.0, 179.0,
        )
        assert corr.correction_dps == 30.0
        assert corr.saturated

    def test_large_negative_error_clamps_at_negative_max(self):
        corr = YawRegulator(kp=1.5, kd=0.0, max_correction_dps=30.0).correct(
            179.0, 0.0,
        )
        assert corr.correction_dps == -30.0
        assert corr.saturated

    def test_demand_inside_the_ceiling_is_not_marked_saturated(self):
        corr = YawRegulator(kp=1.0, kd=0.0, max_correction_dps=30.0).correct(
            0.0, 10.0,
        )
        assert corr.correction_dps == pytest.approx(10.0)
        assert not corr.saturated

    def test_damping_demand_is_clamped_too(self):
        corr = YawRegulator(kp=0.0, kd=5.0, max_correction_dps=15.0).correct(
            0.0, 0.0, measured_yaw_rate_dps=100.0,
        )
        assert corr.correction_dps == -15.0
        assert corr.saturated

    def test_turn_intent_normalizes_and_clamps(self):
        corr = YawRegulator(kp=10.0, kd=0.0, max_correction_dps=90.0).correct(
            0.0, 90.0,
        )
        assert corr.correction_dps == 90.0
        assert corr.turn_intent(60.0) == 1.0  # 90 dps into a 60 dps profile
        assert corr.turn_intent(180.0) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Deadbeat cap — dt bounds the toward-target rate
# --------------------------------------------------------------------------

class TestDeadbeatCap:
    def test_toward_target_rate_capped_at_error_over_dt(self):
        # kp * error = 50 dps, but 1 deg of error at dt=0.1 s can only
        # absorb 10 dps without in-tick overshoot.
        corr = YawRegulator(kp=50.0, kd=0.0, max_correction_dps=90.0).correct(
            0.0, 1.0, dt=0.1,
        )
        assert corr.correction_dps == pytest.approx(10.0)

    def test_cap_is_symmetric(self):
        corr = YawRegulator(kp=50.0, kd=0.0, max_correction_dps=90.0).correct(
            1.0, 0.0, dt=0.1,
        )
        assert corr.correction_dps == pytest.approx(-10.0)

    def test_away_pointing_damping_is_not_capped(self):
        # Zero error, body spinning: the damping demand points away from no
        # target at all and must pass the cap untouched.
        corr = YawRegulator(kp=1.0, kd=1.0, max_correction_dps=90.0).correct(
            0.0, 0.0, measured_yaw_rate_dps=-20.0, dt=0.1,
        )
        assert corr.correction_dps == pytest.approx(20.0)

    def test_no_dt_means_no_cap(self):
        corr = YawRegulator(kp=50.0, kd=0.0, max_correction_dps=90.0).correct(
            0.0, 1.0,
        )
        assert corr.correction_dps == pytest.approx(50.0)


# --------------------------------------------------------------------------
# kd damping — demonstrated against dynamics, not asserted
# --------------------------------------------------------------------------

class TestDampingReducesOvershoot:
    def test_kd_zero_overshoots_and_kd_damps_it(self):
        undamped = _closed_loop_overshoot(kp=DEFAULT_KP, kd=0.0)
        damped = _closed_loop_overshoot(kp=DEFAULT_KP, kd=DEFAULT_KD)
        # kp alone overshoots visibly against the lagged plant...
        assert undamped > 0.5
        # ...kd removes essentially all of it...
        assert damped < 0.05
        # ...and the ordering itself is the module's promise.
        assert undamped > damped

    def test_ordering_holds_for_a_hotter_proportional_gain(self):
        undamped = _closed_loop_overshoot(kp=3.0, kd=0.0)
        damped = _closed_loop_overshoot(kp=3.0, kd=0.6)
        assert undamped > damped

    def test_defaults_converge_on_the_simulated_plant(self):
        # Not just "less overshoot" — the default gains must actually
        # arrive: re-run the same plant and check the terminal error.
        regulator = YawRegulator()
        commanded, heading, rate, dt = 30.0, 0.0, 0.0, 0.02
        alpha = dt / (0.3 + dt)
        for _ in range(int(8.0 / dt)):
            corr = regulator.correct(
                heading, commanded, measured_yaw_rate_dps=rate,
            )
            rate += alpha * (corr.correction_dps - rate)
            heading += rate * dt
        assert abs(heading_error_deg(heading, commanded)) < 0.1


# --------------------------------------------------------------------------
# Statelessness and determinism
# --------------------------------------------------------------------------

class TestStateless:
    def test_repeated_calls_are_identical(self):
        regulator = YawRegulator()
        first = regulator.correct(10.0, 40.0, measured_yaw_rate_dps=5.0)
        for _ in range(5):
            again = regulator.correct(10.0, 40.0, measured_yaw_rate_dps=5.0)
            assert again == first

    def test_regulator_is_frozen(self):
        regulator = YawRegulator()
        with pytest.raises(AttributeError):
            regulator.kp = 2.0  # type: ignore[misc]

    def test_two_instances_same_config_agree(self):
        a = YawRegulator(kp=1.2, kd=0.4, max_correction_dps=25.0)
        b = YawRegulator(kp=1.2, kd=0.4, max_correction_dps=25.0)
        assert a.correct(3.0, 17.0) == b.correct(3.0, 17.0)

    def test_as_dict_round_trips_the_fields(self):
        corr = YawRegulator().correct(10.0, 40.0, measured_yaw_rate_dps=2.0)
        d = corr.as_dict()
        assert d["measured_heading_deg"] == 10.0
        assert d["commanded_heading_deg"] == 40.0
        assert d["error_deg"] == corr.error_deg
        assert d["yaw_rate_dps"] == 2.0
        assert d["correction_dps"] == corr.correction_dps
        assert d["saturated"] is corr.saturated


# --------------------------------------------------------------------------
# Validation — misconfiguration fails loudly, with a reason
# --------------------------------------------------------------------------

class TestValidation:
    def test_negative_kp_rejected(self):
        with pytest.raises(ValueError, match="positive feedback"):
            YawRegulator(kp=-0.1)

    def test_negative_kd_rejected(self):
        with pytest.raises(ValueError, match="positive feedback"):
            YawRegulator(kd=-0.1)

    def test_non_positive_ceiling_rejected(self):
        with pytest.raises(ValueError, match="max_correction_dps"):
            YawRegulator(max_correction_dps=0.0)
        with pytest.raises(ValueError, match="max_correction_dps"):
            YawRegulator(max_correction_dps=-5.0)

    def test_non_positive_dt_rejected(self):
        regulator = YawRegulator()
        with pytest.raises(ValueError, match="dt"):
            regulator.correct(0.0, 10.0, dt=0.0)
        with pytest.raises(ValueError, match="dt"):
            regulator.correct(0.0, 10.0, dt=-0.02)

    def test_non_positive_turn_rate_rejected(self):
        corr = YawRegulator().correct(0.0, 10.0)
        with pytest.raises(ValueError, match="turn_rate_dps"):
            corr.turn_intent(0.0)

    def test_defaults_are_the_documented_ones(self):
        regulator = YawRegulator()
        assert regulator.kp == DEFAULT_KP
        assert regulator.kd == DEFAULT_KD
        assert regulator.max_correction_dps == DEFAULT_MAX_CORRECTION_DPS
