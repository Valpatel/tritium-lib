# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Local planar-frame (x/y metres) TDoA multilateration entry point.

solve_tdoa_xy is the metres-in/metres-out wrapper the SC bridge uses to fuse
multi-node acoustic arrivals in the SAME operating frame as the edge node
position (nx/ny) and the TargetTracker. These tests pin that it recovers a
known source from ideal arrival times — INSIDE and OUTSIDE the sensor hull —
and that it degrades gracefully.
"""
from __future__ import annotations

import math

import pytest

from tritium_lib.models import solve_tdoa_xy
from tritium_lib.models.acoustic_tdoa import SPEED_OF_SOUND_MPS

C = SPEED_OF_SOUND_MPS


def _ideal_arrivals(sensors, source, base_ms=1_000_000.0):
    """Ideal NTP-synced arrival times (ms) for a source heard by sensors."""
    sx, sy = source
    return [
        base_ms + 1000.0 * math.hypot(px - sx, py - sy) / C
        for (px, py) in sensors
    ]


def test_recovers_source_inside_hull():
    sensors = [(0.0, 0.0), (200.0, 0.0), (0.0, 200.0), (200.0, 200.0)]
    source = (130.0, 70.0)
    arrivals = _ideal_arrivals(sensors, source)

    res = solve_tdoa_xy(sensors, arrivals, event_type="gunshot")
    assert res is not None
    err = math.hypot(res["x"] - source[0], res["y"] - source[1])
    assert err < 1.0, f"recovered {res['x']},{res['y']} err={err:.2f}m"
    assert res["residual_error_m"] < 0.5
    assert res["method"] == "tdoa_leastsq"
    assert res["event_type"] == "gunshot"


def test_recovers_source_outside_hull():
    # Source outside the sensor square — the hyperbolic solver must place it
    # OUTSIDE the convex hull where a weighted centroid (stuck between the
    # sensors) structurally cannot. Four well-spread mics give the geometry
    # the least-squares fit needs.
    sensors = [(0.0, 0.0), (200.0, 0.0), (0.0, 200.0), (200.0, 200.0)]
    source = (320.0, 100.0)  # 120 m east of the array's east edge
    arrivals = _ideal_arrivals(sensors, source)

    res = solve_tdoa_xy(sensors, arrivals, event_type="vehicle")
    assert res is not None
    err = math.hypot(res["x"] - source[0], res["y"] - source[1])
    assert err < 2.0, f"recovered {res['x']},{res['y']} err={err:.2f}m"
    # Proves it escaped the hull: solved x is east of every sensor.
    assert res["x"] > 200.0, f"solver stayed inside the hull at x={res['x']}"


def test_too_few_sensors_returns_none():
    assert solve_tdoa_xy([(0.0, 0.0), (10.0, 0.0)], [1.0, 2.0]) is None
    assert solve_tdoa_xy([], []) is None


def test_mismatched_lengths_returns_none():
    assert solve_tdoa_xy([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], [1.0, 2.0]) is None


def test_sensor_ids_and_confidence_threaded():
    sensors = [(0.0, 0.0), (200.0, 0.0), (100.0, 173.0), (100.0, 60.0)]
    source = (100.0, 60.0)
    arrivals = _ideal_arrivals(sensors, source)
    ids = ["mic-a", "mic-b", "mic-c", "mic-d"]

    res = solve_tdoa_xy(sensors, arrivals, sensor_ids=ids, sync_quality=0.9)
    assert res is not None
    assert set(res["sensors_used"]) == set(ids)
    assert 0.0 <= res["confidence"] <= 1.0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
