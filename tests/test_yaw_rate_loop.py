# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the yaw-rate compensator.

The load-bearing test is :class:`TestAgainstAWeakPlant` — a body that only
delivers a fraction of the yaw rate it is told to.  That is not a hypothetical:
a live Newton-stepped Go2 was measured at ~12% of commanded yaw, and no
position loop can fix it because the deficit is in the plant, not the path.
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.control import (
    YawRateCorrection,
    YawRateLoop,
    yaw_rate_from_headings,
)


class TestYawRateFromHeadings:
    """Measuring a rate from two angles — the wrap is the whole difficulty."""

    def test_plain_difference(self):
        assert yaw_rate_from_headings(0.0, 0.2, 0.1) == pytest.approx(2.0)

    def test_negative_rate_when_turning_starboard(self):
        assert yaw_rate_from_headings(0.2, 0.0, 0.1) == pytest.approx(-2.0)

    def test_wraps_the_short_way_across_pi(self):
        # 179 deg -> -179 deg is a 2 deg turn to port, not a 358 deg spin back.
        rate = yaw_rate_from_headings(math.radians(179), math.radians(-179), 1.0)
        assert rate == pytest.approx(math.radians(2), abs=1e-9)

    def test_wraps_the_short_way_the_other_direction(self):
        rate = yaw_rate_from_headings(math.radians(-179), math.radians(179), 1.0)
        assert rate == pytest.approx(math.radians(-2), abs=1e-9)

    def test_zero_dt_is_not_an_infinite_rate(self):
        # A repeated timestamp must not produce inf and poison the integrator.
        assert yaw_rate_from_headings(0.0, 0.5, 0.0) == 0.0

    def test_negative_dt_is_rejected(self):
        with pytest.raises(ValueError):
            yaw_rate_from_headings(0.0, 0.5, -0.1)


class TestConstruction:
    def test_rejects_negative_gains(self):
        with pytest.raises(ValueError):
            YawRateLoop(kp=-1.0)
        with pytest.raises(ValueError):
            YawRateLoop(ki=-1.0)

    def test_rejects_non_positive_output_limit(self):
        with pytest.raises(ValueError):
            YawRateLoop(max_output_rps=0.0)


class TestProportionalTerm:
    def test_no_error_passes_the_command_through(self):
        loop = YawRateLoop(kp=1.0, ki=0.0)
        out = loop.update(commanded_rps=0.4, measured_rps=0.4, dt_s=0.02)
        assert out.compensated_rps == pytest.approx(0.4)
        assert out.error_rps == pytest.approx(0.0)

    def test_shortfall_is_pushed_up(self):
        # Asked 0.4, got 0.1 -> error 0.3 -> command 0.4 + 1.0*0.3.
        loop = YawRateLoop(kp=1.0, ki=0.0)
        out = loop.update(commanded_rps=0.4, measured_rps=0.1, dt_s=0.02)
        assert out.compensated_rps == pytest.approx(0.7)

    def test_overshoot_is_pulled_down(self):
        loop = YawRateLoop(kp=1.0, ki=0.0)
        out = loop.update(commanded_rps=0.2, measured_rps=0.5, dt_s=0.02)
        assert out.compensated_rps == pytest.approx(-0.1)

    def test_returns_a_correction_record(self):
        loop = YawRateLoop()
        assert isinstance(loop.update(0.3, 0.1, 0.02), YawRateCorrection)


class TestIntegralTerm:
    def test_integral_accumulates_a_persistent_error(self):
        loop = YawRateLoop(kp=0.0, ki=1.0)
        loop.update(commanded_rps=0.5, measured_rps=0.0, dt_s=0.1)
        second = loop.update(commanded_rps=0.5, measured_rps=0.0, dt_s=0.1)
        # Two 0.1 s steps at 0.5 rad/s of error -> integral 0.1.
        assert second.integral == pytest.approx(0.1)
        assert second.compensated_rps == pytest.approx(0.6)

    def test_integral_is_clamped(self):
        loop = YawRateLoop(kp=0.0, ki=1.0, integral_limit=0.05)
        for _ in range(50):
            out = loop.update(commanded_rps=1.0, measured_rps=0.0, dt_s=0.1)
        assert out.integral == pytest.approx(0.05)

    def test_reset_clears_the_integral(self):
        loop = YawRateLoop(kp=0.0, ki=1.0)
        loop.update(0.5, 0.0, 0.1)
        loop.reset()
        out = loop.update(0.5, 0.5, 0.1)
        assert out.integral == pytest.approx(0.0)

    def test_zero_dt_does_not_advance_the_integral(self):
        loop = YawRateLoop(kp=0.0, ki=1.0)
        out = loop.update(0.5, 0.0, 0.0)
        assert out.integral == pytest.approx(0.0)


class TestSaturationAndAntiWindup:
    def test_output_is_clamped_to_the_limit(self):
        loop = YawRateLoop(kp=10.0, ki=0.0, max_output_rps=0.8)
        out = loop.update(commanded_rps=0.5, measured_rps=0.0, dt_s=0.02)
        assert out.compensated_rps == pytest.approx(0.8)
        assert out.saturated is True

    def test_clamp_is_symmetric(self):
        loop = YawRateLoop(kp=10.0, ki=0.0, max_output_rps=0.8)
        out = loop.update(commanded_rps=-0.5, measured_rps=0.0, dt_s=0.02)
        assert out.compensated_rps == pytest.approx(-0.8)

    def test_integral_does_not_wind_up_while_saturated(self):
        """The classic failure: a saturated output keeps integrating, and when
        the error finally reverses the loop spends seconds unwinding a term
        that could never have been delivered anyway."""
        loop = YawRateLoop(kp=1.0, ki=5.0, max_output_rps=0.5, integral_limit=100.0)
        for _ in range(100):
            out = loop.update(commanded_rps=2.0, measured_rps=0.0, dt_s=0.02)
        assert out.saturated is True
        # Conditional integration: the term must stay small, nowhere near the
        # 100*2.0*0.02 = 4.0 it would reach if it integrated unconditionally.
        assert abs(out.integral) < 0.5

    def test_integral_still_moves_when_error_unwinds_saturation(self):
        loop = YawRateLoop(kp=1.0, ki=5.0, max_output_rps=0.5)
        for _ in range(20):
            loop.update(commanded_rps=2.0, measured_rps=0.0, dt_s=0.02)
        held = loop.integral
        # Error now reverses: the integrator must be free to come back down.
        out = loop.update(commanded_rps=2.0, measured_rps=4.0, dt_s=0.02)
        assert out.integral < held


class _WeakPlant:
    """A body that delivers only ``gain`` of the yaw rate it is commanded.

    First-order lag on top, so the test is not the trivial algebraic inverse
    of a pure gain — a controller that only works against an instantaneous
    plant is not a controller.
    """

    def __init__(self, gain: float = 0.12, tau_s: float = 0.15):
        self.gain = gain
        self.tau_s = tau_s
        self.rate = 0.0

    def step(self, command: float, dt: float) -> float:
        target = self.gain * command
        alpha = dt / (self.tau_s + dt)
        self.rate += alpha * (target - self.rate)
        return self.rate


class TestAgainstAWeakPlant:
    """The reason this module exists, stated as an A/B.

    Same plant, same demand, same duration.  The only difference is whether
    the compensator is in the path.
    """

    DEMAND = 0.5
    DT = 0.02
    STEPS = 600  # 12 s

    def _run_open_loop(self, plant):
        rate = 0.0
        for _ in range(self.STEPS):
            rate = plant.step(self.DEMAND, self.DT)
        return rate

    def _run_closed_loop(self, plant, loop):
        rate = 0.0
        for _ in range(self.STEPS):
            out = loop.update(self.DEMAND, rate, self.DT)
            rate = plant.step(out.compensated_rps, self.DT)
        return rate

    def test_open_loop_leaves_the_body_at_the_plant_gain(self):
        """The control arm: no compensator, ~12% of demand, forever."""
        settled = self._run_open_loop(_WeakPlant(gain=0.12))
        assert settled == pytest.approx(0.12 * self.DEMAND, rel=0.02)
        assert settled < 0.2 * self.DEMAND

    def test_closed_loop_reaches_the_commanded_rate(self):
        settled = self._run_closed_loop(
            _WeakPlant(gain=0.12),
            YawRateLoop(kp=1.0, ki=6.0, max_output_rps=8.0),
        )
        assert settled == pytest.approx(self.DEMAND, rel=0.05)

    def test_closed_loop_beats_open_loop_by_an_order_of_magnitude(self):
        demand = self.DEMAND
        open_err = abs(demand - self._run_open_loop(_WeakPlant(gain=0.12)))
        closed_err = abs(
            demand
            - self._run_closed_loop(
                _WeakPlant(gain=0.12),
                YawRateLoop(kp=1.0, ki=6.0, max_output_rps=8.0),
            )
        )
        assert closed_err < open_err / 10.0

    def test_holds_up_when_the_plant_is_twice_as_strong_as_tuned_for(self):
        """Gain scheduling would break here; integral action must not."""
        settled = self._run_closed_loop(
            _WeakPlant(gain=0.24),
            YawRateLoop(kp=1.0, ki=6.0, max_output_rps=8.0),
        )
        assert settled == pytest.approx(self.DEMAND, rel=0.05)

    def test_does_not_demand_more_than_the_body_can_give(self):
        """A saturating limit must still be respected under a weak plant."""
        plant, loop = _WeakPlant(gain=0.12), YawRateLoop(
            kp=1.0, ki=6.0, max_output_rps=2.0
        )
        rate = 0.0
        for _ in range(self.STEPS):
            out = loop.update(self.DEMAND, rate, self.DT)
            assert abs(out.compensated_rps) <= 2.0 + 1e-9
            rate = plant.step(out.compensated_rps, self.DT)

    def test_a_zero_demand_settles_at_zero(self):
        """No creep: commanding straight ahead must not accumulate a turn."""
        plant, loop = _WeakPlant(gain=0.12), YawRateLoop(kp=1.0, ki=6.0)
        rate = 0.0
        for _ in range(self.STEPS):
            out = loop.update(0.0, rate, self.DT)
            rate = plant.step(out.compensated_rps, self.DT)
        assert rate == pytest.approx(0.0, abs=1e-3)

    def test_tracks_a_reversal(self):
        """Port then starboard — the sign must follow without a long unwind."""
        plant, loop = _WeakPlant(gain=0.12), YawRateLoop(
            kp=1.0, ki=6.0, max_output_rps=8.0
        )
        rate = 0.0
        for _ in range(self.STEPS):
            out = loop.update(0.5, rate, self.DT)
            rate = plant.step(out.compensated_rps, self.DT)
        for _ in range(self.STEPS):
            out = loop.update(-0.5, rate, self.DT)
            rate = plant.step(out.compensated_rps, self.DT)
        assert rate == pytest.approx(-0.5, rel=0.05)
