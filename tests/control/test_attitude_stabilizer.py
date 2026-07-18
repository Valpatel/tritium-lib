# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Closed-loop attitude regulation: does the correction actually oppose the tilt?

The tests that matter here are the sign tests.  A PD controller with a flipped
sign is not a weak controller, it is an *accelerator* — it drives the body over
faster than no controller at all, which is precisely the 180-degree inversion
failure this module was written to stop.  Every sign convention is therefore
pinned by a test that fails if it is reversed.
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.control.attitude_stabilizer import (
    DEFAULT_KD,
    DEFAULT_KP,
    AttitudeStabilizer,
    LegPlacement,
    roll_pitch_deg,
)


def _quat_about_axis(axis: str, deg: float) -> tuple[float, float, float, float]:
    """Unit quaternion (w, x, y, z) rotating ``deg`` about a principal axis."""
    half = math.radians(deg) / 2.0
    s, c = math.sin(half), math.cos(half)
    return {
        "x": (c, s, 0.0, 0.0),
        "y": (c, 0.0, s, 0.0),
        "z": (c, 0.0, 0.0, s),
    }[axis]


class TestRollPitchExtraction:
    def test_level_body_reads_zero(self):
        roll, pitch = roll_pitch_deg((1.0, 0.0, 0.0, 0.0))
        assert roll == pytest.approx(0.0, abs=1e-9)
        assert pitch == pytest.approx(0.0, abs=1e-9)

    def test_roll_about_x_reads_as_roll_only(self):
        roll, pitch = roll_pitch_deg(_quat_about_axis("x", 20.0))
        assert roll == pytest.approx(20.0, abs=1e-6)
        assert pitch == pytest.approx(0.0, abs=1e-6)

    def test_pitch_about_y_reads_as_pitch_only(self):
        roll, pitch = roll_pitch_deg(_quat_about_axis("y", 15.0))
        assert pitch == pytest.approx(15.0, abs=1e-6)
        assert roll == pytest.approx(0.0, abs=1e-6)

    def test_pure_yaw_is_invisible(self):
        """A walking body changes heading constantly; that is not a tilt."""
        roll, pitch = roll_pitch_deg(_quat_about_axis("z", 90.0))
        assert roll == pytest.approx(0.0, abs=1e-9)
        assert pitch == pytest.approx(0.0, abs=1e-9)

    def test_non_unit_quaternion_is_normalised(self):
        w, x, y, z = _quat_about_axis("x", 30.0)
        roll, _ = roll_pitch_deg((w * 7.0, x * 7.0, y * 7.0, z * 7.0))
        assert roll == pytest.approx(30.0, abs=1e-6)

    def test_zero_quaternion_raises_rather_than_reading_level(self):
        """A failed pose read must not be silently reported as 'upright'."""
        with pytest.raises(ValueError, match="not a rotation"):
            roll_pitch_deg((0.0, 0.0, 0.0, 0.0))

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="4 components"):
            roll_pitch_deg((1.0, 0.0, 0.0))


class TestCorrectionSigns:
    """The load-bearing tests: the correction must OPPOSE the tilt."""

    def test_level_body_needs_no_correction(self):
        s = AttitudeStabilizer(kp=1.0, kd=0.0)
        c = s.update((1.0, 0.0, 0.0, 0.0), dt=0.02)
        assert c.roll_cmd == pytest.approx(0.0, abs=1e-9)
        assert c.pitch_cmd == pytest.approx(0.0, abs=1e-9)

    def test_roll_correction_opposes_roll(self):
        s = AttitudeStabilizer(kp=1.0, kd=0.0)
        c = s.update(_quat_about_axis("x", 10.0), dt=0.02)
        assert c.roll_cmd < 0.0, "correction must push back against +roll"

    def test_pitch_correction_opposes_pitch(self):
        s = AttitudeStabilizer(kp=1.0, kd=0.0)
        c = s.update(_quat_about_axis("y", 10.0), dt=0.02)
        assert c.pitch_cmd < 0.0, "correction must push back against +pitch"

    def test_correction_scales_with_error(self):
        small = AttitudeStabilizer(kp=1.0, kd=0.0).update(
            _quat_about_axis("x", 5.0), dt=0.02)
        big = AttitudeStabilizer(kp=1.0, kd=0.0).update(
            _quat_about_axis("x", 20.0), dt=0.02)
        assert abs(big.roll_cmd) > abs(small.roll_cmd)

    def test_correction_is_clamped(self):
        s = AttitudeStabilizer(kp=100.0, kd=0.0, max_cmd=0.05)
        c = s.update(_quat_about_axis("x", 40.0), dt=0.02)
        assert abs(c.roll_cmd) == pytest.approx(0.05)

    def test_negative_gain_rejected(self):
        """A negative gain is a positive-feedback loop wearing a PD costume."""
        with pytest.raises(ValueError, match="non-negative"):
            AttitudeStabilizer(kp=-1.0, kd=0.0)


class TestDerivativeTerm:
    def test_first_sample_has_no_derivative_kick(self):
        """With no prior sample there is no rate; kd must contribute nothing."""
        s = AttitudeStabilizer(kp=0.0, kd=10.0)
        c = s.update(_quat_about_axis("x", 30.0), dt=0.02)
        assert c.roll_cmd == pytest.approx(0.0, abs=1e-9)
        assert c.roll_rate_dps == pytest.approx(0.0, abs=1e-9)

    def test_rate_is_estimated_from_successive_samples(self):
        s = AttitudeStabilizer(kp=0.0, kd=1.0)
        s.update(_quat_about_axis("x", 0.0), dt=0.1)
        c = s.update(_quat_about_axis("x", 10.0), dt=0.1)
        assert c.roll_rate_dps == pytest.approx(100.0, rel=1e-6)

    def test_damping_opposes_a_body_falling_further(self):
        """Rolling further positive must produce extra negative correction."""
        s = AttitudeStabilizer(kp=1.0, kd=1.0)
        s.update(_quat_about_axis("x", 10.0), dt=0.1)
        falling = s.update(_quat_about_axis("x", 20.0), dt=0.1)

        p_only = AttitudeStabilizer(kp=1.0, kd=0.0)
        p_only.update(_quat_about_axis("x", 10.0), dt=0.1)
        static = p_only.update(_quat_about_axis("x", 20.0), dt=0.1)

        assert falling.roll_cmd < static.roll_cmd

    def test_measured_rate_overrides_finite_difference(self):
        """A real gyro beats differencing a noisy pose read."""
        s = AttitudeStabilizer(kp=0.0, kd=1.0)
        c = s.update(_quat_about_axis("x", 5.0), dt=0.1, roll_rate_dps=42.0)
        assert c.roll_rate_dps == pytest.approx(42.0)

    def test_non_positive_dt_rejected(self):
        s = AttitudeStabilizer(kp=1.0, kd=1.0)
        with pytest.raises(ValueError, match="positive"):
            s.update((1.0, 0.0, 0.0, 0.0), dt=0.0)

    def test_reset_clears_history(self):
        s = AttitudeStabilizer(kp=0.0, kd=1.0)
        s.update(_quat_about_axis("x", 0.0), dt=0.1)
        s.reset()
        c = s.update(_quat_about_axis("x", 10.0), dt=0.1)
        assert c.roll_rate_dps == pytest.approx(0.0, abs=1e-9)


class TestLegOffsets:
    """Per-leg height offsets — the body-agnostic output stage.

    Body frame is REP-103: +X forward, +Y left, +Z up.
    """

    QUAD = (
        LegPlacement("front_left", x=0.2, y=0.1),
        LegPlacement("front_right", x=0.2, y=-0.1),
        LegPlacement("rear_left", x=-0.2, y=0.1),
        LegPlacement("rear_right", x=-0.2, y=-0.1),
    )

    def test_level_body_offsets_all_zero(self):
        s = AttitudeStabilizer(kp=1.0, kd=0.0)
        c = s.update((1.0, 0.0, 0.0, 0.0), dt=0.02)
        offsets = c.leg_height_offsets(self.QUAD)
        assert all(v == pytest.approx(0.0, abs=1e-9) for v in offsets.values())

    def test_rolling_left_side_up_shortens_the_left_legs(self):
        """+roll lifts +Y (left).  Correcting it drops the left legs."""
        s = AttitudeStabilizer(kp=1.0, kd=0.0)
        c = s.update(_quat_about_axis("x", 10.0), dt=0.02)
        o = c.leg_height_offsets(self.QUAD)
        assert o["front_left"] < 0.0 and o["rear_left"] < 0.0
        assert o["front_right"] > 0.0 and o["rear_right"] > 0.0

    def test_pitching_nose_down_raises_the_front_legs(self):
        """REP-103 +pitch is nose-DOWN; the correction extends the front."""
        s = AttitudeStabilizer(kp=1.0, kd=0.0)
        c = s.update(_quat_about_axis("y", 10.0), dt=0.02)
        o = c.leg_height_offsets(self.QUAD)
        assert o["front_left"] > 0.0 and o["front_right"] > 0.0
        assert o["rear_left"] < 0.0 and o["rear_right"] < 0.0

    def test_offsets_sum_to_zero_so_ride_height_is_unchanged(self):
        """Attitude control must not smuggle in a height change."""
        s = AttitudeStabilizer(kp=1.0, kd=0.0)
        c = s.update(_quat_about_axis("x", 12.0), dt=0.02)
        assert sum(c.leg_height_offsets(self.QUAD).values()) == pytest.approx(
            0.0, abs=1e-9)

    def test_offsets_sum_to_zero_on_an_ASYMMETRIC_layout(self):
        """The test above cannot see the mean-centring; this one can.

        On a symmetric layout the raw offsets already cancel, so removing the
        centring step entirely leaves every other test in this file green
        (confirmed by mutation).  A body whose legs are not mirrored — a
        damaged robot, a tripod, a trailer-heavy rover — is where an
        uncentred correction quietly becomes a ride-height command.
        """
        lopsided = (
            LegPlacement("a", x=0.2, y=0.30),
            LegPlacement("b", x=0.2, y=-0.05),
            LegPlacement("c", x=-0.1, y=0.30),
            LegPlacement("d", x=-0.1, y=-0.05),
        )
        s = AttitudeStabilizer(kp=1.0, kd=0.0)
        c = s.update(_quat_about_axis("x", 12.0), dt=0.02)
        offsets = c.leg_height_offsets(lopsided)
        assert sum(offsets.values()) == pytest.approx(0.0, abs=1e-9)
        # ...and the correction must still be a real one, not centred to nothing.
        assert max(abs(v) for v in offsets.values()) > 1e-3

    def test_pitch_offsets_sum_to_zero_on_an_asymmetric_layout(self):
        """Same hole, pitch axis: unequal fore/aft lever arms."""
        nose_heavy = (
            LegPlacement("front", x=0.45, y=0.1),
            LegPlacement("mid", x=0.05, y=-0.1),
            LegPlacement("rear", x=-0.10, y=0.1),
        )
        s = AttitudeStabilizer(kp=1.0, kd=0.0)
        c = s.update(_quat_about_axis("y", 12.0), dt=0.02)
        offsets = c.leg_height_offsets(nose_heavy)
        assert sum(offsets.values()) == pytest.approx(0.0, abs=1e-9)
        assert max(abs(v) for v in offsets.values()) > 1e-3

    def test_offset_magnitude_scales_with_lever_arm(self):
        """A wider stance needs less travel per leg for the same moment."""
        s = AttitudeStabilizer(kp=1.0, kd=0.0)
        c = s.update(_quat_about_axis("x", 10.0), dt=0.02)
        narrow = c.leg_height_offsets(
            (LegPlacement("l", x=0.0, y=0.1), LegPlacement("r", x=0.0, y=-0.1)))
        wide = c.leg_height_offsets(
            (LegPlacement("l", x=0.0, y=0.4), LegPlacement("r", x=0.0, y=-0.4)))
        assert abs(wide["l"]) > abs(narrow["l"])

    def test_works_for_a_six_legged_body(self):
        """Body-agnostic means the leg count is data, not code."""
        hexapod = tuple(
            LegPlacement(f"leg{i}", x=0.3 - 0.3 * (i // 2), y=0.15 * (1 if i % 2 else -1))
            for i in range(6)
        )
        s = AttitudeStabilizer(kp=1.0, kd=0.0)
        c = s.update(_quat_about_axis("y", 8.0), dt=0.02)
        assert len(c.leg_height_offsets(hexapod)) == 6

    def test_duplicate_leg_names_rejected(self):
        s = AttitudeStabilizer(kp=1.0, kd=0.0)
        c = s.update(_quat_about_axis("x", 5.0), dt=0.02)
        with pytest.raises(ValueError, match="duplicate"):
            c.leg_height_offsets(
                (LegPlacement("a", x=0.1, y=0.1), LegPlacement("a", x=-0.1, y=-0.1)))

    def test_empty_layout_rejected(self):
        s = AttitudeStabilizer(kp=1.0, kd=0.0)
        c = s.update(_quat_about_axis("x", 5.0), dt=0.02)
        with pytest.raises(ValueError, match="at least one leg"):
            c.leg_height_offsets(())


class TestConvergence:
    """The controller must actually settle a body, not just have good signs."""

    def _simulate(self, kp: float, kd: float, steps: int = 400) -> list[float]:
        """A 1-DOF inverted-pendulum-ish body: gravity destabilises, cmd restores.

        Deliberately crude — the point is not fidelity but that a *closed* loop
        converges where the same body open-loop diverges.
        """
        dt = 0.01
        angle, rate = 8.0, 0.0  # degrees, deg/s
        s = AttitudeStabilizer(kp=kp, kd=kd, max_cmd=1.0)
        history = []
        for _ in range(steps):
            quat = _quat_about_axis("x", angle)
            cmd = s.update(quat, dt=dt).roll_cmd
            # Toppling torque grows with lean; the command acts against it.
            accel = 30.0 * math.sin(math.radians(angle)) + 400.0 * cmd
            rate += accel * dt
            angle += rate * dt
            history.append(angle)
        return history

    def test_open_loop_body_falls_over(self):
        """The negative control: with zero gain this body must diverge."""
        history = self._simulate(kp=0.0, kd=0.0)
        assert abs(history[-1]) > 45.0, "unstable plant must topple with no control"

    def test_closed_loop_body_stays_upright_on_the_shipped_defaults(self):
        """Pinned to DEFAULT_KP/DEFAULT_KD so a retune must face this test."""
        history = self._simulate(kp=DEFAULT_KP, kd=DEFAULT_KD)
        assert abs(history[-1]) < 0.5, f"failed to settle: ended at {history[-1]:.2f} deg"
        assert max(abs(a) for a in history) <= 8.0 + 1e-6, (
            "overshot past the initial condition instead of decaying from it")

    def test_damping_reduces_oscillation(self):
        undamped = self._simulate(kp=DEFAULT_KP, kd=0.0)
        damped = self._simulate(kp=DEFAULT_KP, kd=DEFAULT_KD)
        assert max(abs(a) for a in damped) < max(abs(a) for a in undamped)
