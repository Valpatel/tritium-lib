# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.obstacles — building collision detection."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from tritium_lib.tracking.obstacles import (
    BuildingObstacles,
    _latlng_to_local,
    _segments_intersect,
)


# ---------------------------------------------------------------------------
# Helper: a simple square building polygon
# ---------------------------------------------------------------------------

def _square_poly(cx: float, cy: float, half: float) -> list[tuple[float, float]]:
    """Return a square polygon centered at (cx, cy) with given half-width."""
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]


# ---------------------------------------------------------------------------
# 1. Test _latlng_to_local conversion
# ---------------------------------------------------------------------------

class TestLatlngToLocal:
    def test_origin_returns_zero(self) -> None:
        x, y = _latlng_to_local(40.0, -74.0, 40.0, -74.0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_north_offset(self) -> None:
        """Moving 1 degree north should be ~111,320 m."""
        x, y = _latlng_to_local(41.0, -74.0, 40.0, -74.0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(111_320.0, rel=1e-3)

    def test_east_offset(self) -> None:
        """Moving 1 degree east at 40N should be ~111320*cos(40) ~ 85,267 m."""
        x, y = _latlng_to_local(40.0, -73.0, 40.0, -74.0)
        expected = 111_320.0 * math.cos(math.radians(40.0))
        assert x == pytest.approx(expected, rel=1e-3)
        assert y == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Test _segments_intersect
# ---------------------------------------------------------------------------

class TestSegmentsIntersect:
    def test_crossing_segments(self) -> None:
        assert _segments_intersect(0, 0, 10, 10, 0, 10, 10, 0) is True

    def test_non_crossing_parallel(self) -> None:
        assert _segments_intersect(0, 0, 10, 0, 0, 5, 10, 5) is False

    def test_non_crossing_disjoint(self) -> None:
        assert _segments_intersect(0, 0, 1, 1, 5, 5, 6, 6) is False

    def test_t_intersection(self) -> None:
        """Perpendicular segments that cross in the middle."""
        assert _segments_intersect(0, 5, 10, 5, 5, 0, 5, 10) is True


# ---------------------------------------------------------------------------
# 3. Test BuildingObstacles creation (empty)
# ---------------------------------------------------------------------------

class TestBuildingObstaclesInit:
    def test_empty_on_creation(self) -> None:
        obs = BuildingObstacles()
        assert obs.polygons == []
        assert obs._heights == []
        assert obs._aabbs == []

    def test_no_building_hit_when_empty(self) -> None:
        obs = BuildingObstacles()
        assert obs.point_in_building(0.0, 0.0) is False

    def test_building_height_none_when_empty(self) -> None:
        obs = BuildingObstacles()
        assert obs.building_height_at(0.0, 0.0) is None


# ---------------------------------------------------------------------------
# 4. Test load_from_overture
# ---------------------------------------------------------------------------

class TestLoadFromOverture:
    def test_loads_polygons(self) -> None:
        obs = BuildingObstacles()
        data = [
            {"polygon": [(0, 0), (10, 0), (10, 10), (0, 10)], "height": 12.0},
            {"polygon": [(20, 20), (30, 20), (30, 30), (20, 30)], "height": 5.0},
        ]
        obs.load_from_overture(data)
        assert len(obs.polygons) == 2
        assert len(obs._heights) == 2
        assert obs._heights[0] == 12.0
        assert obs._heights[1] == 5.0

    def test_skips_degenerate_polygons(self) -> None:
        obs = BuildingObstacles()
        data = [
            {"polygon": [(0, 0), (1, 1)], "height": 8.0},  # Only 2 points
            {"polygon": [(0, 0), (10, 0), (10, 10), (0, 10)], "height": 8.0},
        ]
        obs.load_from_overture(data)
        assert len(obs.polygons) == 1

    def test_default_height(self) -> None:
        obs = BuildingObstacles()
        data = [{"polygon": [(0, 0), (10, 0), (10, 10), (0, 10)]}]
        obs.load_from_overture(data)
        assert obs._heights[0] == 8.0

    def test_aabbs_computed(self) -> None:
        obs = BuildingObstacles()
        data = [{"polygon": [(5, 10), (15, 10), (15, 20), (5, 20)]}]
        obs.load_from_overture(data)
        assert len(obs._aabbs) == 1
        min_x, min_y, max_x, max_y = obs._aabbs[0]
        assert min_x == pytest.approx(5.0)
        assert min_y == pytest.approx(10.0)
        assert max_x == pytest.approx(15.0)
        assert max_y == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 5. Test point_in_building
# ---------------------------------------------------------------------------

class TestPointInBuilding:
    def _make_obs(self) -> BuildingObstacles:
        obs = BuildingObstacles()
        obs.load_from_overture([
            {"polygon": _square_poly(50, 50, 10), "height": 10.0},
        ])
        return obs

    def test_inside(self) -> None:
        obs = self._make_obs()
        assert obs.point_in_building(50.0, 50.0) is True

    def test_outside(self) -> None:
        obs = self._make_obs()
        assert obs.point_in_building(100.0, 100.0) is False

    def test_just_outside_aabb(self) -> None:
        obs = self._make_obs()
        # Building spans 40..60 in x and y
        assert obs.point_in_building(39.0, 50.0) is False
        assert obs.point_in_building(61.0, 50.0) is False


# ---------------------------------------------------------------------------
# 6. Test building_height_at
# ---------------------------------------------------------------------------

class TestBuildingHeightAt:
    def test_returns_height_inside(self) -> None:
        obs = BuildingObstacles()
        obs.load_from_overture([
            {"polygon": _square_poly(0, 0, 10), "height": 15.0},
        ])
        assert obs.building_height_at(0.0, 0.0) == 15.0

    def test_returns_none_outside(self) -> None:
        obs = BuildingObstacles()
        obs.load_from_overture([
            {"polygon": _square_poly(0, 0, 10), "height": 15.0},
        ])
        assert obs.building_height_at(100.0, 100.0) is None


# ---------------------------------------------------------------------------
# 7. Test path_crosses_building
# ---------------------------------------------------------------------------

class TestPathCrossesBuilding:
    def _make_obs(self) -> BuildingObstacles:
        obs = BuildingObstacles()
        obs.load_from_overture([
            {"polygon": _square_poly(50, 50, 10), "height": 10.0},
        ])
        return obs

    def test_path_through_building(self) -> None:
        obs = self._make_obs()
        waypoints = [(0.0, 50.0), (100.0, 50.0)]
        assert obs.path_crosses_building(waypoints) is True

    def test_path_around_building(self) -> None:
        obs = self._make_obs()
        waypoints = [(0.0, 0.0), (100.0, 0.0)]
        assert obs.path_crosses_building(waypoints) is False

    def test_path_too_short(self) -> None:
        obs = self._make_obs()
        assert obs.path_crosses_building([(50.0, 50.0)]) is False

    def test_empty_path(self) -> None:
        obs = self._make_obs()
        assert obs.path_crosses_building([]) is False


# ---------------------------------------------------------------------------
# 8. Test to_dicts export
# ---------------------------------------------------------------------------

class TestToDicts:
    def test_round_trip(self) -> None:
        obs = BuildingObstacles()
        data = [
            {"polygon": [(0, 0), (10, 0), (10, 10), (0, 10)], "height": 12.0},
        ]
        obs.load_from_overture(data)
        exported = obs.to_dicts()
        assert len(exported) == 1
        assert exported[0]["height"] == 12.0
        assert len(exported[0]["polygon"]) == 4

    def test_default_height_fallback(self) -> None:
        obs = BuildingObstacles()
        obs.polygons = [[(0, 0), (1, 0), (1, 1), (0, 1)]]
        obs._heights = []  # No heights stored
        exported = obs.to_dicts(default_height=20.0)
        assert exported[0]["height"] == 20.0


# ---------------------------------------------------------------------------
# 9. Test _build_polygons from OSM elements
# ---------------------------------------------------------------------------

class TestBuildPolygons:
    def test_parses_osm_elements(self) -> None:
        obs = BuildingObstacles()
        elements = [
            {
                "type": "way",
                "geometry": [
                    {"lat": 40.0, "lon": -74.0},
                    {"lat": 40.0001, "lon": -74.0},
                    {"lat": 40.0001, "lon": -73.9999},
                    {"lat": 40.0, "lon": -73.9999},
                ],
                "tags": {"building": "yes", "height": "20"},
            }
        ]
        obs._build_polygons(elements, 40.0, -74.0)
        assert len(obs.polygons) == 1
        assert obs._heights[0] == pytest.approx(20.0)

    def test_skips_non_way(self) -> None:
        obs = BuildingObstacles()
        elements = [
            {"type": "node", "geometry": [{"lat": 40, "lon": -74}], "tags": {}},
        ]
        obs._build_polygons(elements, 40.0, -74.0)
        assert len(obs.polygons) == 0

    def test_height_from_levels(self) -> None:
        obs = BuildingObstacles()
        elements = [
            {
                "type": "way",
                "geometry": [
                    {"lat": 40.0, "lon": -74.0},
                    {"lat": 40.001, "lon": -74.0},
                    {"lat": 40.001, "lon": -73.999},
                ],
                "tags": {"building": "yes", "building:levels": "5"},
            }
        ]
        obs._build_polygons(elements, 40.0, -74.0)
        assert obs._heights[0] == pytest.approx(15.0)  # 5 * 3.0

    def test_default_height_when_no_tags(self) -> None:
        obs = BuildingObstacles()
        elements = [
            {
                "type": "way",
                "geometry": [
                    {"lat": 40.0, "lon": -74.0},
                    {"lat": 40.001, "lon": -74.0},
                    {"lat": 40.001, "lon": -73.999},
                ],
                "tags": {"building": "yes"},
            }
        ]
        obs._build_polygons(elements, 40.0, -74.0)
        assert obs._heights[0] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# 10. Test cache path determinism
# ---------------------------------------------------------------------------

class TestCachePath:
    def test_deterministic(self) -> None:
        p1 = BuildingObstacles._cache_path(40.0, -74.0, 300.0, "/tmp/cache")
        p2 = BuildingObstacles._cache_path(40.0, -74.0, 300.0, "/tmp/cache")
        assert p1 == p2

    def test_different_params_different_path(self) -> None:
        p1 = BuildingObstacles._cache_path(40.0, -74.0, 300.0, "/tmp/cache")
        p2 = BuildingObstacles._cache_path(41.0, -74.0, 300.0, "/tmp/cache")
        assert p1 != p2


# ---------------------------------------------------------------------------
# 11. Test _compute_aabbs with empty polygon
# ---------------------------------------------------------------------------

class TestComputeAABBs:
    def test_empty_polygon_gets_zero_aabb(self) -> None:
        obs = BuildingObstacles()
        obs.polygons = [[]]
        obs._compute_aabbs()
        assert obs._aabbs == [(0.0, 0.0, 0.0, 0.0)]

    def test_normal_polygon(self) -> None:
        obs = BuildingObstacles()
        obs.polygons = [[(1, 2), (5, 2), (5, 8), (1, 8)]]
        obs._compute_aabbs()
        assert obs._aabbs[0] == (1.0, 2.0, 5.0, 8.0)


# ---------------------------------------------------------------------------
# 12. Test multiple buildings
# ---------------------------------------------------------------------------

class TestMultipleBuildings:
    def test_point_in_second_building(self) -> None:
        obs = BuildingObstacles()
        obs.load_from_overture([
            {"polygon": _square_poly(0, 0, 5), "height": 10.0},
            {"polygon": _square_poly(100, 100, 5), "height": 20.0},
        ])
        # Not in first building
        assert obs.point_in_building(0.0, 0.0) is True
        # In second building
        assert obs.point_in_building(100.0, 100.0) is True
        # In neither
        assert obs.point_in_building(50.0, 50.0) is False

    def test_height_from_correct_building(self) -> None:
        obs = BuildingObstacles()
        obs.load_from_overture([
            {"polygon": _square_poly(0, 0, 5), "height": 10.0},
            {"polygon": _square_poly(100, 100, 5), "height": 20.0},
        ])
        assert obs.building_height_at(0.0, 0.0) == 10.0
        assert obs.building_height_at(100.0, 100.0) == 20.0
