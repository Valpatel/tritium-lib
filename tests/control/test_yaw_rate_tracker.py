# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the yaw-rate tracker — the actuator end of the yaw cascade.

Four things are protected here, in order of importance:

1. **The undisturbed gait is untouched.**  Zero demand with zero measured
   rate returns a ``turn`` of EXACTLY ``0.0`` — pinned at the byte level —
   and threads to a state that keeps returning it.
2. **Anti-windup, explicitly.**  The classic bug of this loop shape is an
   integral that keeps climbing while the output is pinned against its stop.
   The freeze is pinned byte-identical across a saturated hold, recovery is
   shown to begin on the FIRST desaturating tick, and a deliberately wound
   integral is run as a counterfactual to demonstrate what the guard is
   worth: the wound arm is still saturated seconds after the guarded arm has
   settled.
3. **The weak plant claim is earned, not asserted.**  The A/B this module
   exists for: on a lagged body that delivers 12% of the commanded rate,
   pure feedforward settles at 12%, a bare proportional trim at ~21%, and
   the PI tracker at 100% — all three arms asserted.  Simulation only; the
   module docstring says so honestly.
4. **The signs agree end to end.**  With gains zeroed the tracker MUST
   degenerate to exactly ``YawCorrection.turn_intent`` — the same
   normalization the edge tier's ``twist_to_motors`` performs — and a
   positive (clockwise) demand must drive the left side faster than the
   right through :func:`~tritium_lib.models.body.motors_from_intent`.
"""

from __future__ import annotations

import math
import struct

import pytest

from tritium_lib.control import (
    YawRateState,
    YawRateTracker,
    YawRegulator,
    heading_error_deg,
)
from tritium_lib.models.body import ControlIntent, motors_from_intent


def _bits(value: float) -> bytes:
    """The IEEE-754 bytes of a float — equality here is byte identity."""
    return struct.pack("<d", value)


# The body profile's claimed full-scale turn rate (deg/s) — the number
# twist_to_motors divides by — and the control tick used throughout.
TURN_RATE_DPS = 60.0
DT = 0.02


class _WeakTurnPlant:
    """A body whose turn command buys less yaw rate than the profile claims.

    ``turn`` in ``[-1, 1]`` maps to an achieved rate through a gain of
    ``full_scale_dps`` (the profile claims ``TURN_RATE_DPS``; the default
    here delivers 12% of that — the measured live deficit) and a first-order
    lag, so the test is not the trivial algebraic inverse of a pure gain.
    """

    def __init__(self, full_scale_dps: float = 7.2, tau_s: float = 0.3):
        self.full_scale_dps = full_scale_dps
        self.tau_s = tau_s
        self.rate_dps = 0.0

    def step(self, turn: float, dt: float) -> float:
        target = self.full_scale_dps * max(-1.0, min(1.0, turn))
        alpha = dt / (self.tau_s + dt)
        self.rate_dps += alpha * (target - self.rate_dps)
        return self.rate_dps


def _run_tracking(
    tracker: YawRateTracker,
    demand_dps: float,
    plant: _WeakTurnPlant,
    steps: int,
    state: YawRateState | None = None,
):
    """Thread ``steps`` ticks of tracker-vs-plant; return (rate, state)."""
    state = state or YawRateState()
    rate = plant.rate_dps
    for _ in range(steps):
        cmd = tracker.track(demand_dps, rate, DT, state=state)
        state = cmd.state
        rate = plant.step(cmd.turn, DT)
    return rate, state


# --------------------------------------------------------------------------
# 1. The byte-identical no-op
# --------------------------------------------------------------------------

class TestZeroNoOp:
    def test_zero_in_zero_out_is_byte_identical_zero(self):
        cmd = YawRateTracker(TURN_RATE_DPS).track(0.0, 0.0, DT)
        assert _bits(cmd.turn) == _bits(0.0)
        assert _bits(cmd.compensated_dps) == _bits(0.0)
        assert _bits(cmd.state.integral_deg) == _bits(0.0)
        assert cmd.error_dps == 0.0
        assert cmd.saturated is False

    def test_noop_threads_to_a_noop(self):
        tracker = YawRateTracker(TURN_RATE_DPS)
        state = None
        for _ in range(5):
            cmd = tracker.track(0.0, 0.0, DT, state=state)
            assert _bits(cmd.turn) == _bits(0.0)
            assert _bits(cmd.state.integral_deg) == _bits(0.0)
            state = cmd.state

    def test_explicit_fresh_state_equals_the_default(self):
        tracker = YawRateTracker(TURN_RATE_DPS)
        assert tracker.track(3.0, 1.0, DT, state=YawRateState()) == (
            tracker.track(3.0, 1.0, DT)
        )

    def test_folding_zero_into_a_gait_turn_is_identity(self):
        gait_turn = 0.25
        cmd = YawRateTracker(TURN_RATE_DPS).track(0.0, 0.0, DT)
        assert _bits(gait_turn + cmd.turn) == _bits(gait_turn)


# --------------------------------------------------------------------------
# 2. Sign convention — compass clockwise, end to end
# --------------------------------------------------------------------------

class TestSignConvention:
    def test_positive_demand_is_a_clockwise_turn_command(self):
        cmd = YawRateTracker(TURN_RATE_DPS).track(6.0, 0.0, DT)
        assert cmd.turn > 0.0

    def test_negative_demand_is_a_counterclockwise_turn_command(self):
        cmd = YawRateTracker(TURN_RATE_DPS).track(-6.0, 0.0, DT)
        assert cmd.turn < 0.0

    def test_positive_turn_drives_left_faster_than_right(self):
        cmd = YawRateTracker(TURN_RATE_DPS).track(10.0, 0.0, DT)
        left, right = motors_from_intent(
            ControlIntent(forward=0.3, turn=cmd.turn)
        )
        assert left > right  # clockwise: the pinned motor contract

    def test_negative_turn_drives_right_faster_than_left(self):
        cmd = YawRateTracker(TURN_RATE_DPS).track(-10.0, 0.0, DT)
        left, right = motors_from_intent(
            ControlIntent(forward=0.3, turn=cmd.turn)
        )
        assert right > left

    def test_overdelivery_pulls_the_command_below_feedforward(self):
        # Body turning FASTER than demanded: the command must come DOWN
        # relative to the open-loop feedforward, not up.
        cmd = YawRateTracker(TURN_RATE_DPS).track(5.0, 10.0, DT)
        assert cmd.turn < 5.0 / TURN_RATE_DPS

    def test_gains_zeroed_degenerates_to_turn_intent_exactly(self):
        """The seam agreement: ff-only tracker == YawCorrection.turn_intent.

        Byte identity, not approximation — both sides are the same division
        by the same full-scale number, and the cascade's sign contract rests
        on them never diverging.
        """
        ff_only = YawRateTracker(TURN_RATE_DPS, kp=0.0, ki=0.0)
        regulator = YawRegulator()
        for measured, commanded in (
            (0.0, 10.0), (10.0, 0.0), (359.0, 1.0), (0.0, 179.0), (90.0, 90.0),
        ):
            corr = regulator.correct(measured, commanded)
            cmd = ff_only.track(corr.correction_dps, 0.0, DT)
            assert _bits(cmd.turn) == _bits(corr.turn_intent(TURN_RATE_DPS))

    def test_feedforward_is_the_twist_to_motors_normalization(self):
        # The edge tier's twist_to_motors computes
        #   turn = clamp(degrees(wz) / turn_rate_dps, -1, 1)
        # (body_node.py, pinned contract: +wz -> clockwise turn intent).
        # The tracker's feedforward path must be the SAME map.
        ff_only = YawRateTracker(TURN_RATE_DPS, kp=0.0, ki=0.0)
        for wz_radps in (0.1, -0.25, 0.9, -2.0):
            demanded_dps = math.degrees(wz_radps)
            edge_turn = max(
                -1.0, min(1.0, math.degrees(wz_radps) / TURN_RATE_DPS)
            )
            cmd = ff_only.track(demanded_dps, 0.0, DT)
            assert _bits(cmd.turn) == _bits(edge_turn)


# --------------------------------------------------------------------------
# 3. Clamping
# --------------------------------------------------------------------------

class TestClamping:
    def test_hot_demand_clamps_exactly_at_max_turn(self):
        cmd = YawRateTracker(TURN_RATE_DPS).track(500.0, 0.0, DT)
        assert cmd.turn == 1.0
        assert cmd.saturated is True

    def test_clamp_is_symmetric(self):
        cmd = YawRateTracker(TURN_RATE_DPS).track(-500.0, 0.0, DT)
        assert cmd.turn == -1.0
        assert cmd.saturated is True

    def test_reduced_authority_ceiling_is_respected(self):
        tracker = YawRateTracker(TURN_RATE_DPS, max_turn=0.5)
        cmd = tracker.track(500.0, 0.0, DT)
        assert cmd.turn == 0.5

    def test_inside_the_ceiling_is_not_marked_saturated(self):
        cmd = YawRateTracker(TURN_RATE_DPS).track(3.0, 2.0, DT)
        assert abs(cmd.turn) < 1.0
        assert cmd.saturated is False

    def test_command_never_exceeds_the_ceiling_under_a_weak_plant(self):
        tracker = YawRateTracker(TURN_RATE_DPS, max_turn=0.8)
        plant, state, rate = _WeakTurnPlant(), YawRateState(), 0.0
        for _ in range(600):
            cmd = tracker.track(30.0, rate, DT, state=state)
            assert abs(cmd.turn) <= 0.8 + 1e-12
            state = cmd.state
            rate = plant.step(cmd.turn, DT)

    def test_integral_is_clamped_to_its_limit(self):
        tracker = YawRateTracker(
            TURN_RATE_DPS, kp=0.0, ki=1.0, integral_limit_deg=0.5
        )
        state = YawRateState()
        for _ in range(50):
            cmd = tracker.track(1.0, 0.0, 0.1, state=state)
            state = cmd.state
        assert state.integral_deg == pytest.approx(0.5)

    def test_default_integral_limit_is_full_authority(self):
        # max_turn * turn_rate_dps / ki: the integral that ALONE saturates
        # the actuator — anything larger is windup by definition.
        assert YawRateTracker(TURN_RATE_DPS).effective_integral_limit_deg == (
            pytest.approx(10.0)
        )
        assert YawRateTracker(
            TURN_RATE_DPS, ki=0.0
        ).effective_integral_limit_deg == 0.0
        assert YawRateTracker(
            TURN_RATE_DPS, integral_limit_deg=3.0
        ).effective_integral_limit_deg == 3.0

    def test_stale_state_is_clamped_to_the_current_config(self):
        # A state recorded under a looser config enters through the clamp:
        # the integral cannot smuggle in more authority than the config
        # allows, and the successor state is back inside the limit.
        cmd = YawRateTracker(TURN_RATE_DPS).track(
            0.0, 0.0, DT, state=YawRateState(integral_deg=50.0)
        )
        assert cmd.state.integral_deg == pytest.approx(10.0)
        assert cmd.turn == pytest.approx(1.0)  # ki * 10 == full scale


# --------------------------------------------------------------------------
# 4. Anti-windup — the classic bug, tested explicitly
# --------------------------------------------------------------------------

class TestAntiWindup:
    def test_cold_start_saturated_hold_never_accumulates(self):
        """8 s pinned against the stop from a cold start: the integral must
        stay EXACTLY zero.  Unconditional integration would have banked
        ~260 deg (bounded only by the huge limit granted here)."""
        tracker = YawRateTracker(TURN_RATE_DPS, integral_limit_deg=1000.0)
        plant, state, rate = _WeakTurnPlant(), YawRateState(), 0.0
        for _ in range(400):
            cmd = tracker.track(40.0, rate, DT, state=state)
            assert cmd.saturated is True
            assert _bits(cmd.state.integral_deg) == _bits(0.0)
            state = cmd.state
            rate = plant.step(cmd.turn, DT)

    def test_warm_integral_freezes_byte_identical_while_pinned(self):
        """A converged integral must FREEZE — not climb, not decay — for as
        long as the output is pinned into the stop by a same-sign error."""
        tracker = YawRateTracker(TURN_RATE_DPS, integral_limit_deg=1000.0)
        plant = _WeakTurnPlant()
        # Converge at an achievable demand first, banking a real integral.
        rate, state = _run_tracking(tracker, 4.0, plant, 600)
        warm = state.integral_deg
        assert warm > 1.0  # the deficit made the integral do real work
        # Jack the demand far beyond the plant ceiling: pinned from here on.
        for _ in range(200):
            cmd = tracker.track(40.0, rate, DT, state=state)
            assert cmd.saturated is True
            assert _bits(cmd.state.integral_deg) == _bits(warm)
            state = cmd.state
            rate = plant.step(cmd.turn, DT)

    def test_integral_moves_on_the_first_desaturating_tick(self):
        tracker = YawRateTracker(TURN_RATE_DPS, integral_limit_deg=1000.0)
        plant = _WeakTurnPlant()
        rate, state = _run_tracking(tracker, 4.0, plant, 600)
        warm = state.integral_deg
        # Measured rate leaps past the demand: the raw output falls off the
        # stop and the integrator must be free THAT tick, not seconds later.
        cmd = tracker.track(40.0, 50.0, DT, state=state)
        assert cmd.saturated is False
        assert cmd.state.integral_deg < warm

    def test_error_reversal_unwinds_even_while_output_saturated(self):
        # Saturated HIGH but error now negative: conditional integration
        # blocks only same-sign pushes, so recovery starts immediately —
        # while the output is still pinned.
        tracker = YawRateTracker(TURN_RATE_DPS, integral_limit_deg=1000.0)
        cmd = tracker.track(
            4.0, 7.2, DT, state=YawRateState(integral_deg=262.0)
        )
        assert cmd.saturated is True
        assert cmd.state.integral_deg < 262.0

    def test_recovery_after_a_hold_settles_like_a_cold_start(self):
        """Drop from a saturated hold to an achievable demand: the guarded
        loop settles into the 10% band and STAYS there within 4 s (measured
        3.46 s), because the hold banked nothing to unwind."""
        tracker = YawRateTracker(TURN_RATE_DPS, integral_limit_deg=1000.0)
        plant = _WeakTurnPlant()
        rate, state = _run_tracking(tracker, 40.0, plant, 400)  # 8 s pinned
        trace = []
        for _ in range(400):  # 8 s of recovery at an achievable demand
            cmd = tracker.track(4.0, rate, DT, state=state)
            state = cmd.state
            rate = plant.step(cmd.turn, DT)
            trace.append(rate)
        assert all(abs(r - 4.0) <= 0.4 for r in trace[200:]), (
            "loop did not settle within 4 s of desaturating"
        )

    def test_a_wound_integral_is_what_the_guard_prevents(self):
        """The counterfactual, run honestly: inject the ~260 deg integral
        unconditional integration would have banked, and the loop is STILL
        pinned at full turn — overshooting a demand of 4 deg/s at the plant's
        entire 7.2 deg/s ceiling — 8 s after the guarded arm settled."""
        tracker = YawRateTracker(TURN_RATE_DPS, integral_limit_deg=1000.0)
        plant = _WeakTurnPlant()
        plant.rate_dps = 7.2  # arriving out of the same saturated hold
        state, rate = YawRateState(integral_deg=262.0), 7.2
        for _ in range(400):  # the same 8 s recovery window as above
            cmd = tracker.track(4.0, rate, DT, state=state)
            state = cmd.state
            rate = plant.step(cmd.turn, DT)
        assert cmd.saturated is True
        assert cmd.turn == pytest.approx(1.0)
        assert rate > 6.0  # still slewing at the ceiling, nowhere near 4
        assert 200.0 < state.integral_deg < 262.0  # unwinding, slowly

    def test_dt_zero_holds_the_integrator(self):
        tracker = YawRateTracker(TURN_RATE_DPS)
        held = YawRateState(integral_deg=1.5)
        cmd = tracker.track(5.0, 0.0, 0.0, state=held)
        assert cmd.state.integral_deg == pytest.approx(1.5)


# --------------------------------------------------------------------------
# 5. The weak plant A/B — the reason this module exists
# --------------------------------------------------------------------------

class TestWeakPlantAB:
    """Same plant, same demand, same duration; only the law differs."""

    DEMAND = 5.0
    STEPS = 600  # 12 s at 50 Hz

    def test_feedforward_alone_settles_at_the_plant_deficit(self):
        ff_only = YawRateTracker(TURN_RATE_DPS, kp=0.0, ki=0.0)
        rate, _ = _run_tracking(
            ff_only, self.DEMAND, _WeakTurnPlant(), self.STEPS
        )
        assert rate == pytest.approx(0.12 * self.DEMAND, rel=0.02)
        assert rate < 0.2 * self.DEMAND

    def test_pi_converges_where_bare_proportional_does_not(self):
        """The load-bearing A/B: on a step demand the PI tracker reaches the
        commanded rate; a bare proportional command settles at the closed
        form ``g(1+kp)d / (T + g kp)`` — nowhere near it."""
        bare_p = YawRateTracker(TURN_RATE_DPS, kp=1.0, ki=0.0)
        p_rate, _ = _run_tracking(
            bare_p, self.DEMAND, _WeakTurnPlant(), self.STEPS
        )
        expected_p = 7.2 * 2.0 * self.DEMAND / (TURN_RATE_DPS + 7.2)
        assert p_rate == pytest.approx(expected_p, rel=0.02)  # ~1.07 deg/s
        assert p_rate < 0.5 * self.DEMAND  # bare P: NOT converged

        pi = YawRateTracker(TURN_RATE_DPS)
        pi_rate, _ = _run_tracking(
            pi, self.DEMAND, _WeakTurnPlant(), self.STEPS
        )
        assert pi_rate == pytest.approx(self.DEMAND, rel=0.05)  # PI: converged

    def test_pi_beats_bare_p_by_an_order_of_magnitude(self):
        bare_p = YawRateTracker(TURN_RATE_DPS, kp=1.0, ki=0.0)
        pi = YawRateTracker(TURN_RATE_DPS)
        p_rate, _ = _run_tracking(
            bare_p, self.DEMAND, _WeakTurnPlant(), self.STEPS
        )
        pi_rate, _ = _run_tracking(
            pi, self.DEMAND, _WeakTurnPlant(), self.STEPS
        )
        assert abs(self.DEMAND - pi_rate) < abs(self.DEMAND - p_rate) / 10.0

    def test_converges_when_the_plant_is_twice_as_strong_as_measured(self):
        """Gain scheduling would break here; integral action must not."""
        pi = YawRateTracker(TURN_RATE_DPS)
        rate, _ = _run_tracking(
            pi, self.DEMAND, _WeakTurnPlant(full_scale_dps=14.4), self.STEPS
        )
        assert rate == pytest.approx(self.DEMAND, rel=0.05)

    def test_zero_demand_does_not_creep(self):
        pi = YawRateTracker(TURN_RATE_DPS)
        rate, state = _run_tracking(pi, 0.0, _WeakTurnPlant(), self.STEPS)
        assert rate == pytest.approx(0.0, abs=1e-3)
        assert state.integral_deg == pytest.approx(0.0, abs=1e-3)

    def test_tracks_a_reversal(self):
        pi = YawRateTracker(TURN_RATE_DPS)
        plant = _WeakTurnPlant()
        _, state = _run_tracking(pi, self.DEMAND, plant, self.STEPS)
        rate, _ = _run_tracking(
            pi, -self.DEMAND, plant, self.STEPS, state=state
        )
        assert rate == pytest.approx(-self.DEMAND, rel=0.05)


# --------------------------------------------------------------------------
# 6. The cascade under YawRegulator — heading -> rate -> command
# --------------------------------------------------------------------------

def _walk_heading(
    with_tracker: bool,
    start_deg: float = 0.0,
    commanded_deg: float = 20.0,
    seconds: float = 12.0,
) -> dict[float, float]:
    """Close the full loop on a weak body; heading error at marked times.

    The ff arm is the cascade EXACTLY as it existed before this module:
    ``YawRegulator`` -> ``turn_intent`` -> motors.  The tracker arm inserts
    the rate loop at the seam the regulator's docstring names as required.
    """
    regulator = YawRegulator()
    tracker = YawRateTracker(TURN_RATE_DPS)
    plant = _WeakTurnPlant()
    heading, rate, state = start_deg, 0.0, YawRateState()
    errors: dict[float, float] = {}
    for i in range(int(seconds / DT)):
        corr = regulator.correct(
            heading, commanded_deg, measured_yaw_rate_dps=rate, dt=DT
        )
        if with_tracker:
            cmd = tracker.track(corr.correction_dps, rate, DT, state=state)
            state = cmd.state
            turn = cmd.turn
        else:
            turn = corr.turn_intent(TURN_RATE_DPS)
        rate = plant.step(turn, DT)
        heading = (heading + rate * DT) % 360.0
        t = (i + 1) * DT
        for mark in (4.0, 12.0):
            if abs(t - mark) < DT / 2.0:
                errors[mark] = heading_error_deg(heading, commanded_deg)
    return errors


class TestCascadeUnderYawRegulator:
    def test_cascade_converges_on_the_weak_plant(self):
        errors = _walk_heading(with_tracker=True)
        assert abs(errors[4.0]) < 2.0   # measured -0.89 deg
        assert abs(errors[12.0]) < 0.5  # measured -0.01 deg

    def test_feedforward_alone_is_still_short_after_twelve_seconds(self):
        # The regulator's own outer loop DOES eventually integrate a heading
        # error away — but through a 12% body it closes at an eighth of the
        # intended pace, which is exactly the deficit the regulator's
        # docstring disclaims responsibility for.
        errors = _walk_heading(with_tracker=False)
        assert errors[4.0] > 5.0        # measured 10.4 deg still to go
        assert abs(errors[12.0]) > 1.0  # measured 2.5 deg — still short

    def test_rate_loop_beats_feedforward_at_every_marked_time(self):
        with_loop = _walk_heading(with_tracker=True)
        without = _walk_heading(with_tracker=False)
        for mark in (4.0, 12.0):
            assert abs(with_loop[mark]) < abs(without[mark])

    def test_wrap_case_goes_the_short_way(self):
        # 350 -> 10 is +20 clockwise through north; a sign slip anywhere in
        # the cascade turns this into a 340 deg tour the other way.
        errors = _walk_heading(
            with_tracker=True, start_deg=350.0, commanded_deg=10.0
        )
        assert abs(errors[4.0]) < 2.0
        assert abs(errors[12.0]) < 0.5


# --------------------------------------------------------------------------
# 7. Threaded state discipline
# --------------------------------------------------------------------------

class TestThreadedState:
    def test_track_is_pure_and_replayable(self):
        tracker = YawRateTracker(TURN_RATE_DPS)
        state = YawRateState(integral_deg=1.25)
        first = tracker.track(6.0, 2.0, DT, state=state)
        second = tracker.track(6.0, 2.0, DT, state=state)
        assert first == second
        assert state.integral_deg == 1.25  # the input state was not touched

    def test_a_recorded_run_replays_byte_identical(self):
        tracker = YawRateTracker(TURN_RATE_DPS)
        script = ((6.0, 0.0), (6.0, 2.0), (40.0, 5.0), (4.0, 6.0), (0.0, 0.5))

        def run() -> list[bytes]:
            state, out = None, []
            for demanded, measured in script:
                cmd = tracker.track(demanded, measured, DT, state=state)
                state = cmd.state
                out.append(_bits(cmd.turn))
            return out

        assert run() == run()

    def test_tracker_is_frozen(self):
        with pytest.raises(AttributeError):
            YawRateTracker(TURN_RATE_DPS).kp = 2.0  # type: ignore[misc]

    def test_state_is_frozen(self):
        with pytest.raises(AttributeError):
            YawRateState().integral_deg = 5.0  # type: ignore[misc]

    def test_two_instances_with_the_same_config_agree(self):
        a = YawRateTracker(TURN_RATE_DPS)
        b = YawRateTracker(TURN_RATE_DPS)
        assert a.track(7.0, 3.0, DT) == b.track(7.0, 3.0, DT)

    def test_as_dict_reports_every_term(self):
        cmd = YawRateTracker(TURN_RATE_DPS).track(6.0, 2.0, DT)
        d = cmd.as_dict()
        assert d == {
            "demanded_dps": cmd.demanded_dps,
            "measured_dps": cmd.measured_dps,
            "error_dps": cmd.error_dps,
            "compensated_dps": cmd.compensated_dps,
            "turn": cmd.turn,
            "saturated": cmd.saturated,
            "integral_deg": cmd.state.integral_deg,
        }


# --------------------------------------------------------------------------
# 8. Validation
# --------------------------------------------------------------------------

class TestValidation:
    def test_non_positive_turn_rate_rejected(self):
        with pytest.raises(ValueError):
            YawRateTracker(0.0)
        with pytest.raises(ValueError):
            YawRateTracker(-60.0)

    def test_negative_gains_rejected(self):
        with pytest.raises(ValueError):
            YawRateTracker(TURN_RATE_DPS, kp=-0.1)
        with pytest.raises(ValueError):
            YawRateTracker(TURN_RATE_DPS, ki=-0.1)

    def test_max_turn_outside_unit_interval_rejected(self):
        with pytest.raises(ValueError):
            YawRateTracker(TURN_RATE_DPS, max_turn=0.0)
        with pytest.raises(ValueError):
            YawRateTracker(TURN_RATE_DPS, max_turn=1.5)

    def test_negative_integral_limit_rejected(self):
        with pytest.raises(ValueError):
            YawRateTracker(TURN_RATE_DPS, integral_limit_deg=-1.0)

    def test_negative_dt_rejected(self):
        with pytest.raises(ValueError):
            YawRateTracker(TURN_RATE_DPS).track(1.0, 0.0, -0.02)

    def test_defaults_are_the_documented_ones(self):
        tracker = YawRateTracker(TURN_RATE_DPS)
        assert tracker.kp == 1.0
        assert tracker.ki == 6.0
        assert tracker.max_turn == 1.0
        assert tracker.integral_limit_deg is None
