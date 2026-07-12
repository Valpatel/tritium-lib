# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Clearance-aware A* — unit-radius standoff around obstacles (UX Loop 3).

A wide unit (an APC) needs more wall clearance than a person.  ``plan_route``
takes a ``clearance_m`` keyword: cells whose distance to the nearest lethal
cell is below it are treated as blocked for snapping, traversal AND smoothing.
``clearance_m == 0.0`` is byte-for-byte the historical planner (golden-replay
safe); a clearance request that cannot be met falls back once to clearance 0
so it NEVER makes a routable dispatch fail (``clearance_relaxed`` records it).
"""

import math

import pytest

from tritium_lib.planning.astar import _supercover_cells, plan_route
from tritium_lib.planning.costmap import Costmap, CostmapBuilder


def _polygon_fc(ring):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {},
             "geometry": {"type": "Polygon", "coordinates": [ring]}},
        ],
    }


def _min_clearance_along(cm, path):
    """Min obstacle clearance (m) over every supercover cell of the path."""
    m = math.inf
    for a, b in zip(path, path[1:]):
        for c, r in _supercover_cells(cm, a, b, include_corner_cells=True):
            m = min(m, cm.clearance_m(c, r))
    return m


def _euclidean_len(path):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(path, path[1:]))


# ---------------------------------------------------------------------------
# Geometry fixtures
# ---------------------------------------------------------------------------

def _around_costmap():
    """40x40 @ 1m: a wall (cols 18-21) with a narrow gap (rows 19-20) and OPEN
    corridors above (rows 32+) and below (rows 0-7) so an around route exists.

    The direct line y=20 threads the gap; the tightest gap cell sits only 1 m
    from the wall, so a >= 2 m standoff cannot pass through it.
    """
    b = CostmapBuilder((0, 0, 40, 40), resolution=1.0)
    b.add_obstacles(_polygon_fc([[18, 8], [22, 8], [22, 19], [18, 19], [18, 8]]))
    b.add_obstacles(_polygon_fc([[18, 21], [22, 21], [22, 32], [18, 32], [18, 21]]))
    return b.build()


def _sealed_gap_costmap():
    """40x30 @ 1m: a FULL-height wall (cols 18-21) whose only opening is a
    narrow gap (rows 14-15) — no around route exists, so a >= 2 m standoff can
    only reach the far side by relaxing back to clearance 0.
    """
    b = CostmapBuilder((0, 0, 40, 30), resolution=1.0)
    b.add_obstacles(_polygon_fc([[18, 0], [22, 0], [22, 14], [18, 14], [18, 0]]))
    b.add_obstacles(_polygon_fc([[18, 16], [22, 16], [22, 30], [18, 30], [18, 16]]))
    return b.build()


# ---------------------------------------------------------------------------
# Clearance changes the route
# ---------------------------------------------------------------------------

class TestClearanceStandoff:
    def test_zero_clearance_threads_gap_high_clearance_detours(self):
        cm = _around_costmap()
        start, goal = (2, 20), (38, 20)

        narrow = plan_route(cm, start, goal, clearance_m=0.0)
        wide = plan_route(cm, start, goal, clearance_m=2.0)

        assert narrow.success and wide.success
        # Neither was forced to relax — both are genuine routes.
        assert narrow.clearance_relaxed is False
        assert wide.clearance_relaxed is False

        # The person threads the sub-2 m gap; the APC keeps its standoff.
        assert _min_clearance_along(cm, narrow.path) < 2.0
        assert _min_clearance_along(cm, wide.path) >= 2.0 - 0.5  # half-cell tol

        # Standoff forces a strictly longer detour around the wall.
        assert _euclidean_len(wide.path) > _euclidean_len(narrow.path)

    def test_clearance_blocks_the_gap_cells(self):
        """Every gap cell is within 1 m of the wall -> blocked at clearance 2."""
        cm = _around_costmap()
        # Gap rows 19,20 in the wall x-band both sit 1 m from lethal.
        assert cm.clearance_m(20, 19) == pytest.approx(1.0)
        assert cm.clearance_m(20, 20) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Graceful fallback: a clearance request never fails a routable dispatch
# ---------------------------------------------------------------------------

class TestClearanceFallback:
    def test_only_route_through_narrow_gap_relaxes(self):
        cm = _sealed_gap_costmap()
        start, goal = (2, 15), (38, 15)

        # Clearance 0 routes through the gap normally.
        base = plan_route(cm, start, goal, clearance_m=0.0)
        assert base.success and base.clearance_relaxed is False

        # Clearance 2 cannot honor the standoff (gap too tight, no way around)
        # -> falls back to a clearance-0 route rather than failing.
        wide = plan_route(cm, start, goal, clearance_m=2.0)
        assert wide.success
        assert wide.clearance_relaxed is True
        assert wide.reason == "ok"

    def test_relax_flag_false_when_standoff_met(self):
        cm = _around_costmap()
        wide = plan_route(cm, (2, 20), (38, 20), clearance_m=2.0)
        assert wide.success and wide.clearance_relaxed is False


# ---------------------------------------------------------------------------
# Golden: clearance 0 is exactly today's behavior
# ---------------------------------------------------------------------------

class TestClearanceZeroGolden:
    def _wall_costmap(self):
        b = CostmapBuilder((0, 0, 200, 200), resolution=10.0)
        b.add_obstacles(_polygon_fc(
            [[95, 20], [105, 20], [105, 180], [95, 180], [95, 20]]))
        return b.build()

    def test_zero_kwarg_matches_no_kwarg(self):
        cm = self._wall_costmap()
        default = plan_route(cm, (30, 100), (170, 100))
        explicit = plan_route(cm, (30, 100), (170, 100), clearance_m=0.0)
        assert explicit.path == default.path
        assert explicit.cost == default.cost
        assert explicit.expansions == default.expansions
        assert explicit.clearance_relaxed is False

    def test_zero_clearance_never_computes_field(self):
        """A clearance-0 plan must not even build the distance field."""
        cm = self._wall_costmap()
        plan_route(cm, (30, 100), (170, 100), clearance_m=0.0)
        assert cm._clearance_cache is None


# ---------------------------------------------------------------------------
# The distance field is computed once and cached on the instance
# ---------------------------------------------------------------------------

class TestClearanceCaching:
    def test_field_built_once_across_two_plans(self, monkeypatch):
        cm = _around_costmap()
        calls = {"n": 0}
        original = Costmap._build_clearance_field

        def counting(self):
            calls["n"] += 1
            return original(self)

        monkeypatch.setattr(Costmap, "_build_clearance_field", counting)

        r1 = plan_route(cm, (2, 20), (38, 20), clearance_m=2.0)
        r2 = plan_route(cm, (2, 20), (38, 20), clearance_m=2.0)
        assert r1.success and r2.success
        # Two full plans (each calling clearance_m many times) -> ONE build.
        assert calls["n"] == 1

    def test_no_lethal_field_is_infinite(self):
        cm = CostmapBuilder((0, 0, 20, 20), resolution=1.0).build()
        assert cm.clearance_m(5, 5) == math.inf
        # Out-of-bounds is treated as obstacle (0 clearance).
        assert cm.clearance_m(-1, 5) == 0.0
