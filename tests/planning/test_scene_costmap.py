# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for projecting 3-D scene obstacles onto a 2-D planning costmap."""
from __future__ import annotations

import math

import pytest

from tritium_lib.planning import plan_route
from tritium_lib.planning.scene_costmap import (
    SceneObstacle,
    costmap_from_scene,
    footprint_polygon,
    scene_bounds,
)


def _lethal_at(costmap, x, y):
    """Is world point ``(x, y)`` lethal?  Out-of-bounds counts as lethal."""
    cell = costmap.world_to_grid(x, y)
    if cell is None:
        return True
    return costmap.is_lethal(*cell)


def _box(path, cx, cy, cz, hx, hy, hz, yaw_deg=0.0):
    return SceneObstacle(
        prim_path=path,
        center=(cx, cy, cz),
        half_extents=(hx, hy, hz),
        yaw_deg=yaw_deg,
    )


class TestSceneObstacle:
    def test_z_span_is_center_plus_minus_half_extent(self):
        obs = _box("/World/Box", 0, 0, 2.0, 1, 1, 0.5)
        assert obs.z_min == pytest.approx(1.5)
        assert obs.z_max == pytest.approx(2.5)

    def test_rejects_negative_half_extents(self):
        with pytest.raises(ValueError):
            _box("/World/Bad", 0, 0, 0, -1, 1, 1)

    def test_intersects_band_true_when_overlapping(self):
        obs = _box("/World/Box", 0, 0, 1.0, 1, 1, 1.0)  # z 0..2
        assert obs.intersects_band(0.1, 0.5) is True

    def test_intersects_band_false_when_entirely_below(self):
        slab = _box("/World/Ground", 0, 0, -0.5, 50, 50, 0.5)  # z -1..0
        assert slab.intersects_band(0.1, 0.5) is False

    def test_intersects_band_false_when_entirely_above(self):
        gantry = _box("/World/Gantry", 0, 0, 5.0, 10, 10, 0.5)  # z 4.5..5.5
        assert gantry.intersects_band(0.1, 0.5) is False


class TestFootprintPolygon:
    def test_axis_aligned_box_gives_four_corners(self):
        ring = footprint_polygon(_box("/W/B", 10, 20, 0, 2, 3, 1))
        assert len(ring) == 5, "ring should be closed"
        assert ring[0] == ring[-1]
        xs = {round(p[0], 6) for p in ring}
        ys = {round(p[1], 6) for p in ring}
        assert xs == {8.0, 12.0}
        assert ys == {17.0, 23.0}

    def test_yaw_rotates_the_footprint(self):
        ring = footprint_polygon(_box("/W/B", 0, 0, 0, 2, 1, 1, yaw_deg=90.0))
        xs = sorted({round(p[0], 6) for p in ring})
        ys = sorted({round(p[1], 6) for p in ring})
        # A 90 deg yaw swaps the extents: x half 2 -> 1, y half 1 -> 2.
        assert xs == [-1.0, 1.0]
        assert ys == [-2.0, 2.0]

    def test_45_degree_yaw_preserves_area(self):
        ring = footprint_polygon(_box("/W/B", 0, 0, 0, 2, 1, 1, yaw_deg=45.0))
        area = 0.0
        for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
            area += x0 * y1 - x1 * y0
        assert abs(area / 2.0) == pytest.approx(4.0 * 2.0)


class TestSceneBounds:
    def test_covers_all_obstacles_with_padding(self):
        obstacles = [
            _box("/W/A", 0, 0, 0, 1, 1, 1),
            _box("/W/B", 20, 10, 0, 2, 2, 1),
        ]
        min_x, min_y, max_x, max_y = scene_bounds(obstacles, padding_m=5.0)
        assert min_x == pytest.approx(-6.0)
        assert min_y == pytest.approx(-6.0)
        assert max_x == pytest.approx(27.0)
        assert max_y == pytest.approx(17.0)

    def test_includes_extra_points(self):
        obstacles = [_box("/W/A", 0, 0, 0, 1, 1, 1)]
        bounds = scene_bounds(obstacles, padding_m=1.0, include=[(50.0, -30.0)])
        assert bounds[2] >= 50.0
        assert bounds[1] <= -30.0

    def test_empty_scene_raises_without_include(self):
        with pytest.raises(ValueError):
            scene_bounds([], padding_m=1.0)


class TestCostmapFromScene:
    def test_obstacle_cell_is_lethal_and_free_cell_is_not(self):
        obstacles = [_box("/W/Pillar", 0, 0, 1.0, 2, 2, 1.0)]
        cm = costmap_from_scene(
            obstacles, bounds=(-20, -20, 20, 20), resolution=1.0,
            body_band=(0.1, 0.5),
        )
        assert _lethal_at(cm, 0.0, 0.0) is True
        assert _lethal_at(cm, 15.0, 15.0) is False

    def test_ground_slab_is_not_stamped_lethal(self):
        """The Newton ground slab is a box; projecting it would block the world."""
        slab = _box("/World/GroundSlab", 0, 0, -0.5, 50, 50, 0.5)
        cm = costmap_from_scene(
            [slab], bounds=(-20, -20, 20, 20), resolution=1.0,
            body_band=(0.1, 0.5),
        )
        assert _lethal_at(cm, 0.0, 0.0) is False
        assert _lethal_at(cm, 10.0, -10.0) is False

    def test_overhead_gantry_is_not_stamped_lethal(self):
        gantry = _box("/World/Gantry", 0, 0, 5.0, 10, 10, 0.5)
        cm = costmap_from_scene(
            [gantry], bounds=(-20, -20, 20, 20), resolution=1.0,
            body_band=(0.1, 0.5),
        )
        assert _lethal_at(cm, 0.0, 0.0) is False

    def test_ignore_prims_excludes_by_exact_path(self):
        obstacles = [_box("/World/Go2/base", 0, 0, 0.3, 0.4, 0.2, 0.15)]
        cm = costmap_from_scene(
            obstacles, bounds=(-10, -10, 10, 10), resolution=0.5,
            body_band=(0.1, 0.5), ignore_prims=["/World/Go2/base"],
        )
        assert _lethal_at(cm, 0.0, 0.0) is False

    def test_ignore_prims_excludes_descendants_of_the_robot(self):
        obstacles = [_box("/World/Go2/base/collision", 0, 0, 0.3, 0.4, 0.2, 0.15)]
        cm = costmap_from_scene(
            obstacles, bounds=(-10, -10, 10, 10), resolution=0.5,
            body_band=(0.1, 0.5), ignore_prims=["/World/Go2"],
        )
        assert _lethal_at(cm, 0.0, 0.0) is False, "robot must not be its own obstacle"

    def test_bounds_derived_from_scene_when_omitted(self):
        obstacles = [_box("/W/A", 0, 0, 1.0, 1, 1, 1.0)]
        cm = costmap_from_scene(
            obstacles, resolution=1.0, body_band=(0.1, 0.5), padding_m=10.0,
        )
        assert cm.origin_x == pytest.approx(-11.0)
        assert cm.origin_y == pytest.approx(-11.0)
        assert _lethal_at(cm, 0.0, 0.0) is True


class TestRoutesAroundSceneObstacles:
    """The payoff: the existing A* must detour around a projected 3-D wall."""

    def _wall_scene(self):
        # A wall spanning x in [-2, 2], y in [-10, 10], with gaps past |y|>10.
        return [_box("/World/Wall", 0.0, 0.0, 1.0, 2.0, 10.0, 1.0)]

    def test_route_detours_around_the_wall(self):
        cm = costmap_from_scene(
            self._wall_scene(), bounds=(-30, -30, 30, 30), resolution=1.0,
            body_band=(0.1, 0.5),
        )
        res = plan_route(cm, (-20.0, 0.0), (20.0, 0.0))
        assert res.success is True
        # Every waypoint must be outside the wall footprint.
        for x, y in res.path:
            assert not (abs(x) <= 2.0 and abs(y) <= 10.0), f"path enters wall at {x},{y}"
        # A detour is strictly longer than the 40 m straight shot.
        length = sum(
            math.dist(a, b) for a, b in zip(res.path, res.path[1:])
        )
        assert length > 40.0

    def test_straight_line_would_have_been_blocked(self):
        """Control: prove the detour was necessary, not incidental."""
        from tritium_lib.planning import segment_clear

        cm = costmap_from_scene(
            self._wall_scene(), bounds=(-30, -30, 30, 30), resolution=1.0,
            body_band=(0.1, 0.5),
        )
        assert segment_clear(cm, (-20.0, 0.0), (20.0, 0.0)) is False

    def test_clearance_keeps_the_route_off_the_wall_face(self):
        cm = costmap_from_scene(
            self._wall_scene(), bounds=(-30, -30, 30, 30), resolution=1.0,
            body_band=(0.1, 0.5),
        )
        res = plan_route(cm, (-20.0, 0.0), (20.0, 0.0), clearance_m=2.0)
        assert res.success is True
        for x, y in res.path:
            assert not (abs(x) <= 2.0 and abs(y) <= 10.0)

    def test_enclosed_goal_reports_no_path(self):
        """A goal walled in on all four sides must fail honestly, not fake a route."""
        pen = [
            _box("/World/N", 0.0, 6.0, 1.0, 6.0, 1.0, 1.0),
            _box("/World/S", 0.0, -6.0, 1.0, 6.0, 1.0, 1.0),
            _box("/World/E", 6.0, 0.0, 1.0, 1.0, 6.0, 1.0),
            _box("/World/W", -6.0, 0.0, 1.0, 1.0, 6.0, 1.0),
        ]
        cm = costmap_from_scene(
            pen, bounds=(-30, -30, 30, 30), resolution=1.0, body_band=(0.1, 0.5),
        )
        res = plan_route(cm, (-20.0, -20.0), (0.0, 0.0))
        assert res.success is False
        assert res.reason in {"no_path", "goal_blocked"}
