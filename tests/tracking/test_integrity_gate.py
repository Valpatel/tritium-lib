# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for RAIM-style track integrity gating (innovation Mahalanobis).

Pins the behavior that a fixed max-speed threshold cannot achieve:
  * a legitimately fast, STEADY mover (aircraft) stays IN-gate, and
  * a teleport / GPS-spoof jump is flagged,
using the SAME parameters — because the gate tests the innovation against a
constant-velocity prediction, not a raw speed ceiling.
"""

from __future__ import annotations

from tritium_lib.tracking.integrity import (
    innovation_mahalanobis_sq,
    update_velocity_ewma,
    is_spoofed,
    spoof_score,
    CHI2_2DOF_999,
)


def _run_track(positions, dt=1.0):
    """Feed a sequence of positions; return (max_m2_after_baseline, flags)."""
    vel = (0.0, 0.0)
    prev = positions[0]
    samples = 0
    flags = []
    m2s = []
    for pos in positions[1:]:
        m2 = innovation_mahalanobis_sq(prev, pos, vel, dt)
        flag = is_spoofed(m2, samples)
        flags.append(flag)
        if samples >= 2:
            m2s.append(m2)
        # Two-point initialization: seed velocity exactly from the first delta
        # (alpha=1) so a steady mover has the right model from sample 1, then
        # EWMA-smooth thereafter. This mirrors the real tracker wiring.
        alpha = 1.0 if samples == 0 else 0.4
        vel = update_velocity_ewma(vel, prev, pos, dt, alpha=alpha)
        prev = pos
        samples += 1
    return m2s, flags


class TestSteadyFastMoverPasses:
    def test_aircraft_200mps_not_flagged(self):
        # 200 m/s due east, 1 Hz fixes — a fixed 50 m/s gate would flag EVERY step.
        positions = [(i * 200.0, 0.0) for i in range(8)]
        _, flags = _run_track(positions, dt=1.0)
        assert not any(flags), f"steady aircraft wrongly flagged: {flags}"

    def test_highway_vehicle_steady_not_flagged(self):
        positions = [(i * 35.0, 0.0) for i in range(8)]  # 126 km/h
        _, flags = _run_track(positions, dt=1.0)
        assert not any(flags)


class TestTeleportFlagged:
    def test_sudden_jump_flagged(self):
        # Slow walker, then a 2 km teleport in one second.
        positions = [(0, 0), (1.2, 0), (2.5, 0.1), (3.6, 0.0), (2003.6, 0.0)]
        _, flags = _run_track(positions, dt=1.0)
        assert flags[-1] is True, "teleport not flagged"
        assert not any(flags[:-1]), "false positives before the jump"

    def test_aircraft_then_teleport_flagged(self):
        # Steady 200 m/s, then a fix 3 km off-prediction -> spoof.
        positions = [(i * 200.0, 0.0) for i in range(6)]
        positions.append((positions[-1][0] + 200.0 + 3000.0, 0.0))
        _, flags = _run_track(positions, dt=1.0)
        assert flags[-1] is True
        assert not any(flags[:-1])


class TestBaselineWarmup:
    def test_no_flag_before_min_samples(self):
        # First fast fix on a freshly-acquired target must NOT flag (no model yet).
        assert is_spoofed(1e6, samples=0) is False
        assert is_spoofed(1e6, samples=1) is False
        assert is_spoofed(1e6, samples=2) is True


class TestHelpers:
    def test_zero_dt_safe(self):
        assert innovation_mahalanobis_sq((0, 0), (5, 5), (0, 0), 0.0) == 0.0
        assert update_velocity_ewma((1, 2), (0, 0), (9, 9), 0.0) == (1, 2)

    def test_spoof_score_monotonic(self):
        assert spoof_score(0.0) == 0.0
        assert spoof_score(CHI2_2DOF_999) == 0.0  # at threshold -> 0
        low = spoof_score(CHI2_2DOF_999 * 2)
        high = spoof_score(CHI2_2DOF_999 * 20)
        assert 0.0 < low < high <= 1.0

    def test_steady_velocity_in_gate_at_varied_dt(self):
        # Same steady velocity, different dt -> still in gate (uncertainty scales).
        for dt in (0.5, 1.0, 5.0):
            m2 = innovation_mahalanobis_sq((0, 0), (100 * dt, 0), (100.0, 0.0), dt)
            assert m2 < CHI2_2DOF_999
