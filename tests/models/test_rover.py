# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the mini-rover body model — the second fleet body.

Pins the twist contract (identical to gait_core: forward=(l+r)/2, turn=(l-r)),
the Tritium frame (x=east, y=north, heading 0 = north increasing clockwise), and
the battery model — so a rover controller, the sim, and real hardware agree.
"""
from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from tritium_lib.models.rover import RoverProfile


def test_default_profile_is_valid():
    r = RoverProfile()
    assert r.asset_type == "rover"
    assert r.wheel_count == 4
    assert r.max_speed_mps > 0 and r.max_turn_dps > 0


def test_invalid_geometry_rejected():
    with pytest.raises(ValidationError):
        RoverProfile(track_width_m=0.0)   # gt=0
    with pytest.raises(ValidationError):
        RoverProfile(max_speed_mps=-1.0)


# --- twist: the shared (left,right) -> (forward,turn) intent ----------------

def test_twist_straight_is_forward_no_turn():
    r = RoverProfile()
    fwd, turn = r.twist(1.0, 1.0)
    assert fwd == pytest.approx(r.max_speed_mps)
    assert turn == pytest.approx(0.0)


def test_twist_reverse_and_stop():
    r = RoverProfile()
    assert r.twist(-1.0, -1.0)[0] == pytest.approx(-r.max_speed_mps)
    assert r.twist(0.0, 0.0) == (0.0, 0.0)


def test_twist_spin_left_faster_turns_positive():
    """left > right -> positive turn (clockwise, heading increases)."""
    r = RoverProfile()
    fwd, turn = r.twist(1.0, -1.0)   # spin in place, left forward
    assert fwd == pytest.approx(0.0)
    assert turn == pytest.approx(r.max_turn_dps)      # full turn rate
    assert r.twist(-1.0, 1.0)[1] == pytest.approx(-r.max_turn_dps)


def test_twist_clamps_out_of_range_commands():
    r = RoverProfile()
    assert r.twist(5.0, 5.0)[0] == pytest.approx(r.max_speed_mps)  # clamped to 1


# --- step: differential-drive integration in the Tritium frame --------------

def test_step_straight_drives_north_no_east_drift():
    r = RoverProfile()
    x, y, h = 0.0, 0.0, 0.0   # heading 0 = north
    for _ in range(10):
        x, y, h = r.step(x, y, h, 1.0, 1.0, 0.1)
    assert h == pytest.approx(0.0)
    assert y > 1.0, f"should have driven north, y={y}"
    assert abs(x) < 1e-9, f"no east drift, x={x}"


def test_step_turn_then_forward_moves_east():
    """Turn clockwise ~90deg (toward east), then drive -> +x (east)."""
    r = RoverProfile(max_turn_dps=90.0)
    x, y, h = 0.0, 0.0, 0.0
    # spin left-forward for 1s at 90 dps -> heading ~90 (east)
    x, y, h = r.step(x, y, h, 1.0, -1.0, 1.0)
    assert h == pytest.approx(90.0, abs=1e-6)
    x, y, h = r.step(x, y, h, 1.0, 1.0, 1.0)   # drive east
    assert x > 1.0, f"should have driven east, x={x}"
    assert abs(y) < 1e-6, f"no north drift after heading east, y={y}"


def test_step_is_deterministic():
    r = RoverProfile()
    def run():
        s = (0.0, 0.0, 30.0)
        for _ in range(20):
            s = r.step(*s, 0.8, 0.2, 0.1)
        return tuple(round(v, 9) for v in s)
    assert run() == run()


# --- battery -----------------------------------------------------------------

def test_drain_idle_less_than_driving():
    r = RoverProfile()
    idle = r.drain_pct_per_s(0.0)
    driving = r.drain_pct_per_s(r.max_speed_mps)
    assert 0.0 < idle < driving
    # idle burns idle_power_w over the pack
    assert idle == pytest.approx(r.idle_power_w / (r.battery_wh * 3600.0))


def test_drain_monotonic_with_speed():
    r = RoverProfile()
    speeds = [0.0, 0.3, 0.6, 0.9, r.max_speed_mps]
    drains = [r.drain_pct_per_s(s) for s in speeds]
    assert drains == sorted(drains)
