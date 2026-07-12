# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Correctness oracle + perf benchmark for the BuildingObstacles spatial index.

The uniform-grid spatial index added to ``BuildingObstacles`` is a pure
BROAD-PHASE accelerator: it changes WHICH buildings the narrow-phase math is
asked about, never the narrow-phase math itself. So for ANY query the indexed
answer MUST equal the brute-force answer that the consumers relied on before.

This module is the standing correctness oracle. It builds a dense, randomized
"hundreds of buildings" map and asserts:
    point_in_building(indexed)      == brute_force_point_in_building
    path_crosses_building(indexed)  == brute_force_path_crosses_building
for thousands of random points / segments. It runs identically with or without
the index (the index is optional-safe), so it stays GREEN before and after the
optimization and catches any future drift.
"""

from __future__ import annotations

import random
import time

import pytest

from tritium_lib.geo import point_in_polygon as _point_in_polygon
from tritium_lib.tracking.obstacles import (
    BuildingObstacles,
    _dist_to_polygon_edge,
    _segments_intersect,
)


# ---------------------------------------------------------------------------
# Brute-force oracles — these reproduce EXACTLY the pre-index algorithms.
# point_in_building honors clearance; path_crosses_building's edge test does
# NOT use clearance (it never did) — only the midpoint check does.
# ---------------------------------------------------------------------------

def _brute_point_in_building(obs: BuildingObstacles, x: float, y: float) -> bool:
    c = obs.clearance
    for poly in obs.polygons:
        if _point_in_polygon(x, y, poly):
            return True
        if c > 0.0 and _dist_to_polygon_edge(x, y, poly) <= c:
            return True
    return False


def _brute_path_crosses_building(
    obs: BuildingObstacles, waypoints: list[tuple[float, float]]
) -> bool:
    if len(waypoints) < 2:
        return False
    for i in range(len(waypoints) - 1):
        ax, ay = waypoints[i]
        bx, by = waypoints[i + 1]
        mx = (ax + bx) / 2
        my = (ay + by) / 2
        if _brute_point_in_building(obs, mx, my):
            return True
        for poly in obs.polygons:
            n = len(poly)
            for j in range(n):
                cx, cy = poly[j]
                dx, dy = poly[(j + 1) % n]
                if _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                    return True
    return False


# ---------------------------------------------------------------------------
# Map builders
# ---------------------------------------------------------------------------

def _random_building(rng: random.Random, cx: float, cy: float) -> list[tuple[float, float]]:
    """A small irregular convex-ish quad around (cx, cy)."""
    hw = rng.uniform(4.0, 18.0)
    hh = rng.uniform(4.0, 18.0)
    jitter = lambda v: v + rng.uniform(-2.0, 2.0)  # noqa: E731
    return [
        (jitter(cx - hw), jitter(cy - hh)),
        (jitter(cx + hw), jitter(cy - hh)),
        (jitter(cx + hw), jitter(cy + hh)),
        (jitter(cx - hw), jitter(cy + hh)),
    ]


def _dense_city(n_buildings: int, extent: float, seed: int = 1234) -> BuildingObstacles:
    """A randomized city of *n_buildings* over a square of side *extent*."""
    rng = random.Random(seed)
    data = []
    for _ in range(n_buildings):
        cx = rng.uniform(0.0, extent)
        cy = rng.uniform(0.0, extent)
        data.append({"polygon": _random_building(rng, cx, cy), "height": 8.0})
    obs = BuildingObstacles()
    obs.load_from_overture(data)
    return obs


# ---------------------------------------------------------------------------
# Correctness oracle: indexed == brute-force, point queries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("clearance", [0.0, 3.0])
def test_point_in_building_identical_to_bruteforce(clearance: float) -> None:
    obs = _dense_city(n_buildings=350, extent=1500.0, seed=7)
    obs.clearance = clearance
    rng = random.Random(99)
    mismatches = 0
    for _ in range(2000):
        x = rng.uniform(-50.0, 1550.0)
        y = rng.uniform(-50.0, 1550.0)
        indexed = obs.point_in_building(x, y)
        brute = _brute_point_in_building(obs, x, y)
        if indexed != brute:
            mismatches += 1
    assert mismatches == 0, f"{mismatches} point mismatches vs brute force"


@pytest.mark.parametrize("clearance", [0.0, 3.0])
def test_path_crosses_building_identical_to_bruteforce(clearance: float) -> None:
    obs = _dense_city(n_buildings=350, extent=1500.0, seed=11)
    obs.clearance = clearance
    rng = random.Random(2024)
    mismatches = 0
    for _ in range(2000):
        ax = rng.uniform(0.0, 1500.0)
        ay = rng.uniform(0.0, 1500.0)
        # Mixture of short hops and long cross-map segments (multi-cell spans).
        if rng.random() < 0.5:
            bx = ax + rng.uniform(-40.0, 40.0)
            by = ay + rng.uniform(-40.0, 40.0)
        else:
            bx = rng.uniform(0.0, 1500.0)
            by = rng.uniform(0.0, 1500.0)
        wp = [(ax, ay), (bx, by)]
        indexed = obs.path_crosses_building(wp)
        brute = _brute_path_crosses_building(obs, wp)
        if indexed != brute:
            mismatches += 1
    assert mismatches == 0, f"{mismatches} path mismatches vs brute force"


def test_multi_waypoint_path_identical() -> None:
    obs = _dense_city(n_buildings=300, extent=1200.0, seed=42)
    rng = random.Random(5)
    mismatches = 0
    for _ in range(500):
        n = rng.randint(2, 6)
        wp = [
            (rng.uniform(0.0, 1200.0), rng.uniform(0.0, 1200.0))
            for _ in range(n)
        ]
        if obs.path_crosses_building(wp) != _brute_path_crosses_building(obs, wp):
            mismatches += 1
    assert mismatches == 0, f"{mismatches} multi-waypoint mismatches"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_obstacles_identical() -> None:
    obs = BuildingObstacles()
    assert obs.point_in_building(0.0, 0.0) is False
    assert obs.point_in_building(0.0, 0.0) == _brute_point_in_building(obs, 0.0, 0.0)
    wp = [(0.0, 0.0), (100.0, 100.0)]
    assert obs.path_crosses_building(wp) is False
    assert obs.path_crosses_building(wp) == _brute_path_crosses_building(obs, wp)


def test_single_building_identical() -> None:
    obs = BuildingObstacles()
    obs.load_from_overture([
        {"polygon": [(40, 40), (60, 40), (60, 60), (40, 60)], "height": 8.0},
    ])
    rng = random.Random(1)
    for _ in range(500):
        x = rng.uniform(0.0, 100.0)
        y = rng.uniform(0.0, 100.0)
        assert obs.point_in_building(x, y) == _brute_point_in_building(obs, x, y)


def test_building_straddling_cell_boundary() -> None:
    """A building larger than one cell must be bucketed into every overlapping
    cell so points anywhere inside it are found."""
    obs = BuildingObstacles()
    # A long thin building spanning a wide x-range (many cells wide).
    obs.load_from_overture([
        {"polygon": [(0, 0), (500, 0), (500, 12), (0, 12)], "height": 8.0},
    ])
    rng = random.Random(3)
    for _ in range(500):
        x = rng.uniform(-20.0, 520.0)
        y = rng.uniform(-20.0, 40.0)
        assert obs.point_in_building(x, y) == _brute_point_in_building(obs, x, y)


def test_long_segment_spanning_many_cells() -> None:
    obs = _dense_city(n_buildings=400, extent=2000.0, seed=21)
    # A segment crossing the entire map diagonally — touches many grid cells.
    wp = [(-100.0, -100.0), (2100.0, 2100.0)]
    assert obs.path_crosses_building(wp) == _brute_path_crosses_building(obs, wp)
    wp2 = [(0.0, 1000.0), (2000.0, 1000.0)]
    assert obs.path_crosses_building(wp2) == _brute_path_crosses_building(obs, wp2)


def test_degenerate_zero_area_polygon() -> None:
    """A degenerate (collinear / zero-area) polygon must not crash the index and
    must still match brute force (which returns False inside it)."""
    obs = BuildingObstacles()
    obs.load_from_overture([
        {"polygon": [(10, 10), (20, 10), (30, 10), (10, 10)], "height": 8.0},  # collinear
        {"polygon": [(100, 100), (120, 100), (120, 120), (100, 120)], "height": 8.0},
    ])
    rng = random.Random(8)
    for _ in range(500):
        x = rng.uniform(0.0, 150.0)
        y = rng.uniform(0.0, 150.0)
        assert obs.point_in_building(x, y) == _brute_point_in_building(obs, x, y)
    wp = [(0.0, 10.0), (150.0, 10.0)]
    assert obs.path_crosses_building(wp) == _brute_path_crosses_building(obs, wp)


def test_index_optional_safe_without_compute() -> None:
    """If polygons are mutated WITHOUT calling _compute_aabbs, the linear
    fallback must still give correct (identical) answers."""
    obs = BuildingObstacles()
    # Directly poke polygons; do NOT call _compute_aabbs.
    obs.polygons = [[(40, 40), (60, 40), (60, 60), (40, 60)]]
    obs._heights = [8.0]
    obs._aabbs = []  # ensure no index/aabbs
    assert obs.point_in_building(50.0, 50.0) is True
    assert obs.point_in_building(0.0, 0.0) is False
    wp = [(0.0, 50.0), (100.0, 50.0)]
    assert obs.path_crosses_building(wp) is True


# ---------------------------------------------------------------------------
# Perf benchmark (informational — not a hard gate). Run with -s to see numbers:
#   pytest tests/tracking/test_obstacles_spatial_index.py -k benchmark -s
# ---------------------------------------------------------------------------

def test_benchmark_us_per_call(capsys) -> None:
    obs = _dense_city(n_buildings=400, extent=2000.0, seed=55)
    rng = random.Random(123)

    pts = [
        (rng.uniform(0.0, 2000.0), rng.uniform(0.0, 2000.0))
        for _ in range(3000)
    ]
    segs = []
    for _ in range(3000):
        ax = rng.uniform(0.0, 2000.0)
        ay = rng.uniform(0.0, 2000.0)
        bx = ax + rng.uniform(-60.0, 60.0)
        by = ay + rng.uniform(-60.0, 60.0)
        segs.append([(ax, ay), (bx, by)])

    t0 = time.perf_counter()
    for x, y in pts:
        obs.point_in_building(x, y)
    t1 = time.perf_counter()
    point_us = (t1 - t0) / len(pts) * 1e6

    t0 = time.perf_counter()
    for wp in segs:
        obs.path_crosses_building(wp)
    t1 = time.perf_counter()
    path_us = (t1 - t0) / len(segs) * 1e6

    with capsys.disabled():
        print(
            f"\n[BENCH n=400 buildings] "
            f"point_in_building={point_us:.2f} us/call  "
            f"path_crosses_building={path_us:.2f} us/call"
        )
    # Sanity: indexed path query should be well under the ~400us census figure.
    assert path_us < 400.0
