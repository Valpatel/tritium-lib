# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.planning.astar — the open baseline A* planner."""

import math

import pytest

from tritium_lib.planning.astar import _supercover_cells, plan_route, segment_clear
from tritium_lib.planning.costmap import CostmapBuilder, CostmapWeights

LETHAL = float("inf")


def _polygon_fc(ring, props=None):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": props or {},
             "geometry": {"type": "Polygon", "coordinates": [ring]}},
        ],
    }


def _line_fc(coords, props=None):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": props or {},
             "geometry": {"type": "LineString", "coordinates": coords}},
        ],
    }


def _path_touches_lethal(cm, path):
    """True if any segment of the path touches a lethal cell."""
    for a, b in zip(path, path[1:]):
        for c, r in _supercover_cells(cm, a, b, include_corner_cells=True):
            if cm.is_lethal(c, r):
                return True
    return False


def _euclidean_len(path):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


# ---------------------------------------------------------------------------
# Straight route
# ---------------------------------------------------------------------------

class TestStraightRoute:
    def test_empty_map_endpoints_exact(self):
        cm = CostmapBuilder((0, 0, 200, 200), resolution=10.0).build()
        res = plan_route(cm, (5, 5), (195, 5))
        assert res.success
        assert res.reason == "ok"
        assert res.path[0] == (5, 5)
        assert res.path[-1] == (195, 5)

    def test_empty_map_cost_matches_distance(self):
        cm = CostmapBuilder((0, 0, 200, 200), resolution=10.0).build()
        res = plan_route(cm, (5, 5), (195, 5))
        # Unit-cost grid: route cost equals geometric distance in meters.
        assert res.cost == pytest.approx(190.0)

    def test_diagonal_route(self):
        cm = CostmapBuilder((0, 0, 200, 200), resolution=10.0).build()
        res = plan_route(cm, (5, 5), (195, 195))
        assert res.success
        assert res.path[0] == (5, 5)
        assert res.path[-1] == (195, 195)
        # A diagonal shortcut smooths to a near-straight line.
        assert len(res.path) <= 4


# ---------------------------------------------------------------------------
# Obstacle avoidance
# ---------------------------------------------------------------------------

class TestObstacleAvoidance:
    def _wall_costmap(self):
        # Vertical wall spanning most of the map height with gaps top+bottom.
        b = CostmapBuilder((0, 0, 200, 200), resolution=10.0)
        b.add_obstacles(_polygon_fc([[95, 20], [105, 20], [105, 180], [95, 180], [95, 20]]))
        return b.build()

    def test_routes_around_wall(self):
        cm = self._wall_costmap()
        res = plan_route(cm, (30, 100), (170, 100))
        assert res.success
        assert res.path[0] == (30, 100)
        assert res.path[-1] == (170, 100)

    def test_wall_route_never_lethal(self):
        cm = self._wall_costmap()
        res = plan_route(cm, (30, 100), (170, 100))
        assert not _path_touches_lethal(cm, res.path)

    def test_wall_route_unsmoothed_never_lethal(self):
        cm = self._wall_costmap()
        res = plan_route(cm, (30, 100), (170, 100), smooth=False)
        assert res.success
        assert not _path_touches_lethal(cm, res.path)

    def test_no_path_when_goal_enclosed(self):
        # Ring the goal cell with lethal so it cannot be entered.
        cm = CostmapBuilder((0, 0, 90, 90), resolution=10.0).build()
        gc, gr = 4, 4
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if dc == 0 and dr == 0:
                    continue
                cm.grid[gr + dr][gc + dc] = LETHAL
        res = plan_route(cm, (5, 5), (45, 45))
        assert not res.success
        assert res.reason == "no_path"


# ---------------------------------------------------------------------------
# Road preference
# ---------------------------------------------------------------------------

class TestRoadPreference:
    def test_prefers_longer_road_over_direct(self):
        # Staple-shaped road: up, across, down.  Direct base route is shorter
        # geometrically but more expensive than the discounted detour.
        road = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"width_m": 8.0},
                 "geometry": {"type": "LineString",
                              "coordinates": [[5, 5], [5, 25], [105, 25], [105, 5]]}},
            ],
        }
        b = CostmapBuilder((0, 0, 110, 50), resolution=10.0)
        b.add_roads(road)
        cm = b.build()
        road_cells = {
            (c, r)
            for c in range(cm.width)
            for r in range(cm.height)
            if cm.cost_at(c, r) < 1.0
        }
        res = plan_route(cm, (5, 5), (105, 5))
        assert res.success
        # The path deflects up onto the road corridor (row 2, y≈25).
        assert max(p[1] for p in res.path) >= 20.0
        # Most traversed cells are road cells.
        traversed = []
        for a, b2 in zip(res.path, res.path[1:]):
            traversed.extend(_supercover_cells(cm, a, b2, include_corner_cells=False))
        road_frac = sum(1 for c in traversed if c in road_cells) / len(traversed)
        assert road_frac > 0.6


# ---------------------------------------------------------------------------
# Slope avoidance
# ---------------------------------------------------------------------------

class TestSlopeAvoidance:
    def _cone_costmap(self):
        from tritium_lib.planning.layers import LocalElevationGrid

        def cone(x, y):
            # Constant-slope cone -> a lethal disk of steep terrain.
            return max(0.0, 60.0 - 1.0 * math.hypot(x - 100, y - 100))

        dem = LocalElevationGrid.from_callable((-30, -30, 230, 230), 10.0, cone)
        b = CostmapBuilder((0, 0, 200, 200), resolution=10.0)
        b.add_dem(dem)
        return b.build()

    def test_deflects_around_hill(self):
        cm = self._cone_costmap()
        # The direct line through the center is lethal (steep flank).
        assert cm.is_lethal(*cm.world_to_grid(70, 100))
        res = plan_route(cm, (5, 100), (195, 100))
        assert res.success
        assert not _path_touches_lethal(cm, res.path)
        # The route bulges well away from the center line.
        assert max(abs(p[1] - 100) for p in res.path) > 40.0


# ---------------------------------------------------------------------------
# Snapping
# ---------------------------------------------------------------------------

class TestSnapping:
    def test_goal_in_building_snaps(self):
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        # A single lethal cell at the goal, free cells all around.
        b.add_obstacles(_polygon_fc([[50, 50], [60, 50], [60, 60], [50, 60], [50, 50]]))
        cm = b.build()
        goal = (55, 55)  # inside the lethal cell
        assert cm.is_lethal(*cm.world_to_grid(*goal))
        res = plan_route(cm, (5, 5), goal)
        assert res.success
        # Contract: on success the last point is the exact requested goal.
        assert res.path[-1] == goal

    def test_start_in_building_snaps(self):
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        b.add_obstacles(_polygon_fc([[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]))
        cm = b.build()
        start = (15, 15)
        assert cm.is_lethal(*cm.world_to_grid(*start))
        res = plan_route(cm, start, (95, 95))
        assert res.success
        assert res.path[0] == start

    def test_snap_failure_goal_blocked(self):
        b = CostmapBuilder((0, 0, 200, 200), resolution=10.0)
        # Large building: nearest free cell is beyond the default snap radius.
        b.add_obstacles(_polygon_fc([[70, 70], [140, 70], [140, 140], [70, 140], [70, 70]]))
        cm = b.build()
        res = plan_route(cm, (5, 5), (105, 105))
        assert not res.success
        assert res.reason == "goal_blocked"

    def test_snap_radius_zero_blocks(self):
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        b.add_obstacles(_polygon_fc([[50, 50], [60, 50], [60, 60], [50, 60], [50, 50]]))
        cm = b.build()
        res = plan_route(cm, (5, 5), (55, 55), snap_radius_m=0.0)
        assert not res.success
        assert res.reason == "goal_blocked"

    def test_out_of_bounds_reasons(self):
        cm = CostmapBuilder((0, 0, 100, 100), resolution=10.0).build()
        assert plan_route(cm, (-5, 5), (50, 50)).reason == "start_out_of_bounds"
        assert plan_route(cm, (5, 5), (500, 50)).reason == "goal_out_of_bounds"


# ---------------------------------------------------------------------------
# Corner-cutting forbidden
# ---------------------------------------------------------------------------

class TestCornerCutting:
    def test_pinch_blocks_diagonal(self):
        # 2x2 grid with a lethal pinch at (1,0) and (0,1): the only way to
        # (1,1) would be a diagonal squeeze through the corner -> forbidden.
        cm = CostmapBuilder((0, 0, 20, 20), resolution=10.0).build()
        cm.grid[0][1] = LETHAL  # cell (1,0)
        cm.grid[1][0] = LETHAL  # cell (0,1)
        res = plan_route(cm, (5, 5), (15, 15))
        assert not res.success
        assert res.reason == "no_path"

    def test_open_diagonal_allowed(self):
        # Same geometry without the pinch: the diagonal move is legal.
        cm = CostmapBuilder((0, 0, 20, 20), resolution=10.0).build()
        res = plan_route(cm, (5, 5), (15, 15))
        assert res.success

    def test_smoothing_never_squeezes_pinch(self):
        # A wider map with a lethal blob between free start and goal cells.
        # A tempting diagonal shortcut would clip the pinch corner; smoothing
        # must reject it.
        cm = CostmapBuilder((0, 0, 50, 50), resolution=10.0).build()
        cm.grid[1][1] = LETHAL  # cell (1,1)
        cm.grid[2][1] = LETHAL  # cell (1,2)
        cm.grid[1][2] = LETHAL  # cell (2,1)
        res = plan_route(cm, (5, 5), (45, 45))  # cells (0,0) -> (4,4), both free
        assert res.success
        assert not _path_touches_lethal(cm, res.path)


# ---------------------------------------------------------------------------
# Determinism / limits / smoothing invariants
# ---------------------------------------------------------------------------

class TestDeterminismAndLimits:
    def test_deterministic(self):
        b = CostmapBuilder((0, 0, 200, 200), resolution=10.0)
        b.add_obstacles(_polygon_fc([[95, 20], [105, 20], [105, 180], [95, 180], [95, 20]]))
        cm = b.build()
        r1 = plan_route(cm, (30, 100), (170, 100))
        r2 = plan_route(cm, (30, 100), (170, 100))
        assert r1.path == r2.path
        assert r1.cost == r2.cost
        assert r1.expansions == r2.expansions

    def test_max_expansions_cap(self):
        cm = CostmapBuilder((0, 0, 500, 500), resolution=10.0).build()
        res = plan_route(cm, (5, 5), (495, 495), max_expansions=5)
        assert not res.success
        assert res.reason == "max_expansions"

    def test_smoothing_never_introduces_lethal(self):
        b = CostmapBuilder((0, 0, 200, 200), resolution=10.0)
        b.add_obstacles(_polygon_fc([[95, 20], [105, 20], [105, 180], [95, 180], [95, 20]]))
        cm = b.build()
        res = plan_route(cm, (30, 100), (170, 100), smooth=True)
        assert not _path_touches_lethal(cm, res.path)

    def test_smoothing_never_lengthens(self):
        b = CostmapBuilder((0, 0, 200, 200), resolution=10.0)
        b.add_obstacles(_polygon_fc([[95, 20], [105, 20], [105, 180], [95, 180], [95, 20]]))
        cm = b.build()
        raw = plan_route(cm, (30, 100), (170, 100), smooth=False)
        sm = plan_route(cm, (30, 100), (170, 100), smooth=True)
        assert len(sm.path) <= len(raw.path)
        assert _euclidean_len(sm.path) <= _euclidean_len(raw.path) + 1e-6
        # Reported optimal grid cost is unchanged by smoothing.
        assert sm.cost == pytest.approx(raw.cost)

    def test_smoothing_preserves_road_preference(self):
        road = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"width_m": 8.0},
                 "geometry": {"type": "LineString",
                              "coordinates": [[5, 5], [5, 25], [105, 25], [105, 5]]}},
            ],
        }
        b = CostmapBuilder((0, 0, 110, 50), resolution=10.0)
        b.add_roads(road)
        cm = b.build()
        sm = plan_route(cm, (5, 5), (105, 5), smooth=True)
        # Smoothing must not straight-line across the expensive base cells.
        assert max(p[1] for p in sm.path) >= 20.0


class TestSegmentClear:
    """Public LOS gate — corner-safe supercover, clearance-aware."""

    def test_open_line_is_clear(self):
        cm = CostmapBuilder((0, 0, 200, 200), resolution=10.0).build()
        assert segment_clear(cm, (20, 20), (180, 180)) is True

    def test_zero_length_open_cell_is_clear(self):
        cm = CostmapBuilder((0, 0, 200, 200), resolution=10.0).build()
        assert segment_clear(cm, (55, 55), (55, 55)) is True

    def test_segment_crossing_lethal_block_is_blocked(self):
        # A vertical wall at x in [95, 105] spanning the whole map.
        b = CostmapBuilder((0, 0, 200, 200), resolution=10.0)
        b.add_obstacles(_polygon_fc([[95, 20], [105, 20], [105, 180], [95, 180], [95, 20]]))
        cm = b.build()
        # A horizontal segment through the wall must be blocked...
        assert segment_clear(cm, (30, 100), (170, 100)) is False
        # ...while a segment entirely on one side is clear.
        assert segment_clear(cm, (10, 30), (80, 170)) is True

    def test_endpoint_out_of_bounds_is_blocked(self):
        cm = CostmapBuilder((0, 0, 100, 100), resolution=10.0).build()
        assert segment_clear(cm, (50, 50), (500, 50)) is False
        assert segment_clear(cm, (-50, 50), (50, 50)) is False

    def test_zero_length_lethal_cell_is_blocked(self):
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        b.add_obstacles(_polygon_fc([[40, 40], [60, 40], [60, 60], [40, 60], [40, 40]]))
        cm = b.build()
        assert segment_clear(cm, (50, 50), (50, 50)) is False

    def test_diagonal_corner_pinch_is_blocked(self):
        # Two lethal cells sharing only a corner at (10, 10): cell (1,0) and
        # cell (0,1).  The OPEN cells (0,0) and (1,1) form the anti-diagonal.
        # A diagonal threaded from open (0,0) to open (1,1) passes exactly
        # through the shared corner and must be reported blocked (corner-safe
        # supercover), never allowed to "slip between" the two obstacles.
        b = CostmapBuilder((0, 0, 30, 30), resolution=10.0)
        b.add_obstacles(_polygon_fc([[10, 0], [20, 0], [20, 10], [10, 10], [10, 0]]))
        b.add_obstacles(_polygon_fc([[0, 10], [10, 10], [10, 20], [0, 20], [0, 10]]))
        cm = b.build()
        # Endpoints sit in the OPEN anti-diagonal cells (0,0) and (1,1).
        assert cm.is_lethal(*cm.world_to_grid(5, 5)) is False
        assert cm.is_lethal(*cm.world_to_grid(15, 15)) is False
        assert segment_clear(cm, (5, 5), (15, 15)) is False

    def test_clearance_rejects_sub_standoff_corridor(self):
        # A gap one cell wide between two walls: passable at clearance 0,
        # blocked once a standoff wider than the gap half-width is required.
        b = CostmapBuilder((0, 0, 90, 90), resolution=10.0)
        b.add_obstacles(_polygon_fc([[0, 30], [40, 30], [40, 40], [0, 40], [0, 30]]))
        b.add_obstacles(_polygon_fc([[50, 30], [90, 30], [90, 40], [50, 40], [50, 30]]))
        cm = b.build()
        thread = ((45, 10), (45, 80))  # straight up the one-cell gap at x=45
        assert segment_clear(cm, *thread, clearance_m=0.0) is True
        # A 12 m standoff exceeds the ~5 m clearance in the gap -> blocked.
        assert segment_clear(cm, *thread, clearance_m=12.0) is False
