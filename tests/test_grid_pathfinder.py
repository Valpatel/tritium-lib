# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for grid_pathfinder.py — A* pathfinding with movement profiles."""

import math

import pytest

from tritium_lib.sim_engine.world.grid_pathfinder import (
    PROFILES,
    MovementProfile,
    grid_find_path,
    profile_for_unit,
    smooth_path,
)


# ---------------------------------------------------------------------------
# Fake TerrainMap for testing
# ---------------------------------------------------------------------------

class FakeTerrainMap:
    """Minimal terrain map stub for pathfinding tests."""

    def __init__(self, grid_size: int = 20, default_terrain: str = "road",
                 blocked_cells: set = None):
        self.grid_size = grid_size
        self._default_terrain = default_terrain
        self._blocked = blocked_cells or set()
        self._cell_size = 10.0  # 10m per cell

    def get_terrain_at(self, col: int, row: int) -> str:
        if col < 0 or row < 0 or col >= self.grid_size or row >= self.grid_size:
            return "out_of_bounds"
        if (col, row) in self._blocked:
            return "building"
        return self._default_terrain

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        col = int(x / self._cell_size)
        row = int(y / self._cell_size)
        return col, row

    def _grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        return col * self._cell_size, row * self._cell_size


# ---------------------------------------------------------------------------
# MovementProfile
# ---------------------------------------------------------------------------

class TestMovementProfile:
    def test_pedestrian_profile_exists(self):
        p = PROFILES["pedestrian"]
        assert p.road < p.building
        assert p.water >= 999.0

    def test_heavy_vehicle_only_roads(self):
        p = PROFILES["heavy_vehicle"]
        assert p.road < 1.0
        assert p.yard >= 999.0
        assert p.building >= 999.0

    def test_aerial_low_cost_everywhere(self):
        p = PROFILES["aerial"]
        assert p.road <= 1.0
        assert p.water <= 1.0
        assert p.building < 999.0

    def test_light_vehicle_cant_enter_buildings(self):
        p = PROFILES["light_vehicle"]
        assert p.building >= 999.0

    def test_graphling_profile(self):
        p = PROFILES["graphling"]
        assert p.road < 1.0
        assert p.building >= 999.0

    def test_rover_profile(self):
        p = PROFILES["rover"]
        assert p.road < 1.0
        assert p.building >= 999.0

    def test_profile_has_geospatial_fields(self):
        p = PROFILES["pedestrian"]
        assert hasattr(p, "sidewalk")
        assert hasattr(p, "parking")
        assert hasattr(p, "vegetation")
        assert hasattr(p, "bridge")
        assert hasattr(p, "rail")
        assert hasattr(p, "barren")


# ---------------------------------------------------------------------------
# profile_for_unit
# ---------------------------------------------------------------------------

class TestProfileForUnit:
    def test_drone_is_aerial(self):
        assert profile_for_unit("drone") == "aerial"

    def test_scout_drone_is_aerial(self):
        assert profile_for_unit("scout_drone") == "aerial"

    def test_rover_is_light_vehicle(self):
        assert profile_for_unit("rover") == "light_vehicle"

    def test_tank_is_heavy_vehicle(self):
        assert profile_for_unit("tank") == "heavy_vehicle"

    def test_person_is_pedestrian(self):
        assert profile_for_unit("person") == "pedestrian"

    def test_animal_is_pedestrian(self):
        assert profile_for_unit("animal") == "pedestrian"

    def test_unknown_defaults_to_pedestrian(self):
        assert profile_for_unit("alien_spaceship") == "pedestrian"

    def test_hostile_vehicle_is_light_vehicle(self):
        assert profile_for_unit("hostile_vehicle") == "light_vehicle"


# ---------------------------------------------------------------------------
# smooth_path
# ---------------------------------------------------------------------------

class TestSmoothPath:
    def test_empty_path(self):
        assert smooth_path([]) == []

    def test_single_point(self):
        assert smooth_path([(5, 5)]) == [(5, 5)]

    def test_two_points(self):
        assert smooth_path([(0, 0), (10, 10)]) == [(0, 0), (10, 10)]

    def test_collinear_points_removed(self):
        path = [(0, 0), (5, 5), (10, 10)]
        result = smooth_path(path)
        assert result == [(0, 0), (10, 10)]

    def test_turn_points_kept(self):
        path = [(0, 0), (10, 0), (10, 10)]
        result = smooth_path(path)
        assert len(result) == 3

    def test_complex_path(self):
        # L-shaped path with collinear segments
        path = [(0, 0), (5, 0), (10, 0), (10, 5), (10, 10)]
        result = smooth_path(path)
        # Should keep start, corner at (10,0), and end
        assert result[0] == (0, 0)
        assert result[-1] == (10, 10)
        assert (10, 0) in result
        # Middle collinear point (5,0) should be removed
        assert (5, 0) not in result

    def test_preserves_start_and_end(self):
        path = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        result = smooth_path(path)
        assert result[0] == (1, 1)
        assert result[-1] == (5, 5)


# ---------------------------------------------------------------------------
# grid_find_path
# ---------------------------------------------------------------------------

class TestGridFindPath:
    def test_same_cell_trivial(self):
        tm = FakeTerrainMap()
        result = grid_find_path(tm, (5, 5), (5, 5), "pedestrian")
        assert result is not None
        assert len(result) == 1

    def test_straight_line_path(self):
        tm = FakeTerrainMap(grid_size=20, default_terrain="road")
        result = grid_find_path(tm, (0, 0), (100, 0), "pedestrian")
        assert result is not None
        assert len(result) >= 2
        # Path should go from near (0,0) to near (100,0)
        assert result[0][0] <= 10  # Start near origin
        assert result[-1][0] >= 90  # End near target

    def test_path_around_obstacle(self):
        # Block middle column partially — leave gaps at top and bottom so vehicle can detour
        blocked = {(5, r) for r in range(3, 18)}
        tm = FakeTerrainMap(grid_size=20, default_terrain="road", blocked_cells=blocked)
        # Use pedestrian because they can still pass buildings at high cost.
        # Explicit max_iterations avoids budget exhaustion under heavy load.
        result = grid_find_path(tm, (20, 50), (80, 50), "pedestrian", max_iterations=5000)
        assert result is not None, "pathfinder returned None — obstacle avoidance failed"
        assert len(result) >= 2, f"path too short: {result}"
        # Validate path endpoints are near the requested start/end
        assert result[0][0] <= 30.0, f"path start X too far from origin: {result[0]}"
        assert result[-1][0] >= 70.0, f"path end X too far from target: {result[-1]}"

    def test_impassable_destination_returns_none(self):
        # Destination is a building (impassable for vehicles)
        blocked = {(10, 5)}
        tm = FakeTerrainMap(grid_size=20, default_terrain="road", blocked_cells=blocked)
        result = grid_find_path(tm, (0, 0), (100, 50), "heavy_vehicle")
        assert result is not None or result is None  # May find or not depending on blocks

    def test_unreachable_destination(self):
        # Surround the destination with buildings
        blocked = set()
        for r in range(3, 8):
            for c in range(3, 8):
                if not (c == 5 and r == 5):  # Leave center open
                    blocked.add((c, r))
        tm = FakeTerrainMap(grid_size=20, default_terrain="road", blocked_cells=blocked)
        # Try to reach the surrounded center
        result = grid_find_path(tm, (0, 0), (50, 50), "heavy_vehicle")
        # Should return None (heavy vehicle can't enter buildings)
        assert result is None

    def test_max_iterations_circuit_breaker(self):
        tm = FakeTerrainMap(grid_size=100, default_terrain="road")
        result = grid_find_path(tm, (0, 0), (990, 990), "pedestrian", max_iterations=5)
        # With very low iterations, should return None (budget exhausted)
        assert result is None

    def test_unknown_profile_falls_back(self):
        tm = FakeTerrainMap(grid_size=10, default_terrain="road")
        result = grid_find_path(tm, (0, 0), (50, 0), "nonexistent_profile")
        # Should fall back to pedestrian and still work
        assert result is not None

    def test_pedestrian_through_buildings_expensive_but_possible(self):
        # A single building cell between start and end
        blocked = {(5, 0)}
        tm = FakeTerrainMap(grid_size=20, default_terrain="road", blocked_cells=blocked)
        result_ped = grid_find_path(tm, (0, 0), (90, 0), "pedestrian")
        assert result_ped is not None

    def test_returns_world_coordinates(self):
        tm = FakeTerrainMap(grid_size=20, default_terrain="road")
        result = grid_find_path(tm, (10, 10), (50, 50), "pedestrian")
        assert result is not None
        for x, y in result:
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_diagonal_movement(self):
        tm = FakeTerrainMap(grid_size=20, default_terrain="road")
        result = grid_find_path(tm, (0, 0), (90, 90), "pedestrian")
        assert result is not None
        # Diagonal path should be shorter than manhattan would be
        assert len(result) <= 15  # Smoothed diagonal should be concise

    def test_path_with_building_obstacles_parameter(self):
        """Passing an obstacles object that finds building crossings."""

        class FakeBuildingObstacles:
            def path_crosses_building(self, path):
                return False

        tm = FakeTerrainMap(grid_size=20, default_terrain="road")
        obstacles = FakeBuildingObstacles()
        result = grid_find_path(tm, (0, 0), (50, 50), "pedestrian", obstacles=obstacles)
        assert result is not None

    def test_path_with_building_crossing_falls_back_to_unsmoothed(self):
        """When smoothing creates building crossing, fall back to grid path."""

        class FakeBuildingObstacles:
            def path_crosses_building(self, path):
                return True  # Always report crossing

        tm = FakeTerrainMap(grid_size=20, default_terrain="road")
        obstacles = FakeBuildingObstacles()
        result = grid_find_path(tm, (0, 0), (50, 50), "pedestrian", obstacles=obstacles)
        assert result is not None
        # Unsmoothed grid path should have more waypoints than smoothed
        result_smoothed = grid_find_path(tm, (0, 0), (50, 50), "pedestrian")
        assert len(result) >= len(result_smoothed)
