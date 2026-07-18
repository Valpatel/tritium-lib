# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the Newton-native quadruped gait trajectory generator.

Pure kinematics — numpy/stdlib only, NO Isaac.  Validates the properties a
USD Newton driver depends on: exact periodicity, trot diagonal anti-phase,
swing-leg calf lift returning to the stand pose, plausible joint envelopes,
and stride frequency scaling with commanded speed per DEFAULT_GAITS.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tritium_lib.models.gait_trajectory import (
    GAIT_PHASE_OFFSETS,
    JOINT_LIMITS_RAD,
    JOINT_NAMES,
    LEG_NAMES,
    NEUTRAL_CALF_RAD,
    NEUTRAL_STAND_RAD,
    NEUTRAL_THIGH_RAD,
    QuadrupedGaitCycle,
    joint_targets_at,
)
from tritium_lib.models.quadruped import DEFAULT_GAITS, QuadrupedProfile


class TestShape:
    """The returned dict is the exact 12-joint contract the driver applies."""

    def test_twelve_named_joints(self):
        targets = joint_targets_at(0.0)
        assert set(targets) == set(JOINT_NAMES)
        assert len(targets) == 12

    def test_joint_name_scheme(self):
        assert JOINT_NAMES == (
            "FL_hip", "FL_thigh", "FL_calf",
            "FR_hip", "FR_thigh", "FR_calf",
            "RL_hip", "RL_thigh", "RL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
        )

    def test_all_angles_finite_floats(self):
        for t in np.linspace(0.0, 3.0, 50):
            for name, angle in joint_targets_at(float(t)).items():
                assert isinstance(angle, float), name
                assert math.isfinite(angle), name


class TestPeriodicity:
    """angles(t) == angles(t + period) exactly for every gait."""

    @pytest.mark.parametrize("gait", ["walk", "trot", "bound"])
    def test_time_periodicity(self, gait):
        cycle = QuadrupedGaitCycle(gait)
        period = cycle.period_s
        for t in (0.0, 0.13, 0.5 * period, 0.99 * period, 2.7):
            a = cycle.angles_at_time(t)
            b = cycle.angles_at_time(t + period)
            for name in JOINT_NAMES:
                assert a[name] == pytest.approx(b[name], abs=1e-9), (gait, t, name)

    @pytest.mark.parametrize("gait", ["walk", "trot", "bound"])
    def test_phase_wraps(self, gait):
        cycle = QuadrupedGaitCycle(gait)
        a = cycle.angles_at_phase(0.25)
        b = cycle.angles_at_phase(1.25)
        c = cycle.angles_at_phase(-0.75)
        assert a == b == c


class TestTrotDiagonals:
    """Trot: diagonal pairs (FL+RR, FR+RL) in phase; diagonals anti-phase."""

    def test_diagonal_pairs_in_phase(self):
        cycle = QuadrupedGaitCycle("trot")
        for phase in np.linspace(0.0, 1.0, 17, endpoint=False):
            angles = cycle.angles_at_phase(float(phase))
            for part in ("thigh", "calf"):
                assert angles[f"FL_{part}"] == pytest.approx(
                    angles[f"RR_{part}"], abs=1e-9
                )
                assert angles[f"FR_{part}"] == pytest.approx(
                    angles[f"RL_{part}"], abs=1e-9
                )

    def test_diagonals_anti_phase_at_half_cycle(self):
        cycle = QuadrupedGaitCycle("trot")
        for phase in np.linspace(0.0, 1.0, 17, endpoint=False):
            now = cycle.angles_at_phase(float(phase))
            half = cycle.angles_at_phase(float(phase) + 0.5)
            for part in ("thigh", "calf"):
                assert now[f"FL_{part}"] == pytest.approx(
                    half[f"FR_{part}"], abs=1e-9
                )
                assert now[f"RR_{part}"] == pytest.approx(
                    half[f"RL_{part}"], abs=1e-9
                )

    def test_diagonals_actually_differ_mid_cycle(self):
        # Anti-phase must be a real offset, not two constant signals.
        cycle = QuadrupedGaitCycle("trot")
        diffs = [
            abs(
                cycle.angles_at_phase(p)["FL_thigh"]
                - cycle.angles_at_phase(p)["FR_thigh"]
            )
            for p in np.linspace(0.0, 1.0, 33, endpoint=False)
        ]
        assert max(diffs) > 0.1  # radians — a visible step, not noise

    def test_walk_offsets_are_quarter_cycle(self):
        offsets = sorted(GAIT_PHASE_OFFSETS["walk"].values())
        assert offsets == [0.0, 0.25, 0.5, 0.75]


class TestSwingLiftsCalf:
    """In swing the calf tucks (lifts the foot) then returns to the stand."""

    @pytest.mark.parametrize("gait", ["walk", "trot"])
    def test_calf_tucks_mid_swing_and_returns(self, gait):
        cycle = QuadrupedGaitCycle(gait)
        duty = cycle.duty_factor
        # FL has offset 0.0 in every gait table, so global phase == FL phase.
        # At stance the calf holds neutral.
        mid_stance = cycle.angles_at_phase(duty / 2.0)
        assert mid_stance["FL_calf"] == pytest.approx(NEUTRAL_CALF_RAD, abs=1e-9)
        # Mid-swing the knee flexes further: calf strictly more negative.
        mid_swing = cycle.angles_at_phase(duty + (1.0 - duty) / 2.0)
        assert mid_swing["FL_calf"] < NEUTRAL_CALF_RAD - 0.1
        # Swing boundaries land exactly back on the stand calf angle.
        liftoff = cycle.angles_at_phase(duty)
        touchdown = cycle.angles_at_phase(0.0)
        assert liftoff["FL_calf"] == pytest.approx(NEUTRAL_CALF_RAD, abs=1e-9)
        assert touchdown["FL_calf"] == pytest.approx(NEUTRAL_CALF_RAD, abs=1e-9)

    def test_thigh_sweeps_through_neutral(self):
        # The thigh oscillates around the stand angle: mean over a cycle
        # stays near neutral and both signs of offset occur.
        cycle = QuadrupedGaitCycle("trot")
        thighs = np.array(
            [
                cycle.angles_at_phase(p)["FL_thigh"]
                for p in np.linspace(0.0, 1.0, 200, endpoint=False)
            ]
        )
        assert thighs.min() < NEUTRAL_THIGH_RAD < thighs.max()
        assert abs(float(thighs.mean()) - NEUTRAL_THIGH_RAD) < 0.1


class TestJointLimits:
    """Every emitted angle stays inside the plausible Go2-class envelope."""

    @pytest.mark.parametrize("gait", ["walk", "trot", "bound"])
    def test_within_limits_over_dense_cycle(self, gait):
        cycle = QuadrupedGaitCycle(gait)
        for phase in np.linspace(0.0, 1.0, 500, endpoint=False):
            angles = cycle.angles_at_phase(float(phase))
            for leg in LEG_NAMES:
                for part in ("hip", "thigh", "calf"):
                    lo, hi = JOINT_LIMITS_RAD[part]
                    a = angles[f"{leg}_{part}"]
                    assert lo <= a <= hi, (gait, leg, part, a)

    def test_within_limits_at_speed_extremes(self):
        for speed in (0.05, 5.0):
            cycle = QuadrupedGaitCycle("trot", speed=speed)
            for phase in np.linspace(0.0, 1.0, 100, endpoint=False):
                for name, a in cycle.angles_at_phase(float(phase)).items():
                    part = name.split("_")[1]
                    lo, hi = JOINT_LIMITS_RAD[part]
                    assert lo <= a <= hi, (speed, name, a)

    def test_neutral_stand_matches_newton_validated_pose(self):
        assert NEUTRAL_STAND_RAD["FL_hip"] == pytest.approx(0.0)
        assert NEUTRAL_STAND_RAD["FL_thigh"] == pytest.approx(math.radians(50.0))
        assert NEUTRAL_STAND_RAD["FL_calf"] == pytest.approx(math.radians(-100.0))
        assert set(NEUTRAL_STAND_RAD) == set(JOINT_NAMES)


class TestSpeedScaling:
    """Higher commanded speed -> higher step frequency, per DEFAULT_GAITS."""

    def test_stride_hz_scales_with_speed(self):
        spec = DEFAULT_GAITS["trot"]
        slow = QuadrupedGaitCycle("trot", speed=spec.speed_mps * 0.5)
        nominal = QuadrupedGaitCycle("trot", speed=spec.speed_mps)
        fast = QuadrupedGaitCycle("trot", speed=spec.speed_mps * 1.5)
        assert slow.stride_hz < nominal.stride_hz < fast.stride_hz
        assert nominal.stride_hz == pytest.approx(spec.stride_hz)
        assert fast.stride_hz == pytest.approx(spec.stride_hz * 1.5)

    def test_frequency_clamped_to_sane_band(self):
        spec = DEFAULT_GAITS["trot"]
        crawl = QuadrupedGaitCycle("trot", speed=1e-6)
        sprint = QuadrupedGaitCycle("trot", speed=1e6)
        assert crawl.stride_hz == pytest.approx(spec.stride_hz * 0.2)
        assert sprint.stride_hz == pytest.approx(spec.stride_hz * 2.0)

    def test_no_speed_uses_gait_nominal(self):
        for gait, spec in DEFAULT_GAITS.items():
            cycle = QuadrupedGaitCycle(gait)
            assert cycle.stride_hz == pytest.approx(spec.stride_hz)

    def test_module_level_speed_changes_period(self):
        spec = DEFAULT_GAITS["trot"]
        t = 0.1
        slow = joint_targets_at(t, gait="trot", speed=spec.speed_mps * 0.5)
        fast = joint_targets_at(t, gait="trot", speed=spec.speed_mps * 2.0)
        # Same wall-clock instant, different cadence -> different pose.
        assert any(
            abs(slow[n] - fast[n]) > 1e-6 for n in JOINT_NAMES
        )


class TestSampleCycle:
    """Full-cycle sampling for recording/replay."""

    def test_sample_count_and_phases(self):
        cycle = QuadrupedGaitCycle("trot")
        samples = cycle.sample_cycle(steps=16)
        assert len(samples) == 16
        phases = [p for p, _ in samples]
        assert phases == pytest.approx([i / 16 for i in range(16)])
        for _, targets in samples:
            assert set(targets) == set(JOINT_NAMES)

    def test_sample_matches_direct_evaluation(self):
        cycle = QuadrupedGaitCycle("walk")
        for phase, targets in cycle.sample_cycle(steps=8):
            assert targets == cycle.angles_at_phase(phase)

    def test_rejects_degenerate_step_count(self):
        with pytest.raises(ValueError):
            QuadrupedGaitCycle("trot").sample_cycle(steps=1)


class TestProfileAndErrors:
    def test_unknown_gait_raises(self):
        with pytest.raises(KeyError):
            QuadrupedGaitCycle("gallop")

    def test_custom_profile_gait_without_offsets_raises(self):
        profile = QuadrupedProfile(
            gaits={"crawl": DEFAULT_GAITS["walk"].model_copy()},
            default_gait="crawl",
        )
        with pytest.raises(KeyError):
            QuadrupedGaitCycle("crawl", profile)

    def test_custom_profile_used_for_amplitude(self):
        tall = QuadrupedProfile(body_height_m=0.8)
        short = QuadrupedProfile(body_height_m=0.3)
        # Longer legs -> smaller thigh sweep for the same stride.
        assert (
            QuadrupedGaitCycle("trot", tall).thigh_amp_rad
            <= QuadrupedGaitCycle("trot", short).thigh_amp_rad
        )
