# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.terrain — HeightMap, LineOfSight, CoverMap, MovementCost."""

import math
import pytest

from tritium_lib.sim_engine.terrain import (
    HeightMap,
    LineOfSight,
    CoverMap,
    MovementCost,
    _bresenham,
    _fbm,
    _value_noise_2d,
)


# ===========================================================================
# HeightMap construction
# ===========================================================================

class TestHeightMapConstruction:
    def test_flat_terrain_all_zero(self):
        hm = HeightMap(10, 10)
        for y in range(10):
            for x in range(10):
                assert hm.get_elevation(x, y) == 0.0

    def test_custom_cell_size(self):
        hm = HeightMap(5, 5, cell_size=2.0)
        assert hm.cell_size == 2.0
        assert hm.width == 5
        assert hm.height == 5

    def test_invalid_dimensions_raise(self):
        with pytest.raises(ValueError):
            HeightMap(0, 10)
        with pytest.raises(ValueError):
            HeightMap(10, -1)

    def test_invalid_cell_size_raises(self):
        with pytest.raises(ValueError):
            HeightMap(10, 10, cell_size=0.0)
        with pytest.raises(ValueError):
            HeightMap(10, 10, cell_size=-1.0)

    def test_set_get_elevation(self):
        hm = HeightMap(5, 5)
        hm.set_elevation(2, 3, 7.5)
        assert hm.get_elevation(2, 3) == 7.5

    def test_set_elevation_out_of_bounds_raises(self):
        hm = HeightMap(5, 5)
        with pytest.raises(IndexError):
            hm.set_elevation(5, 0, 1.0)
        with pytest.raises(IndexError):
            hm.set_elevation(0, 5, 1.0)
        with pytest.raises(IndexError):
            hm.set_elevation(-1, 0, 1.0)

    def test_get_elevation_out_of_bounds_returns_zero(self):
        hm = HeightMap(5, 5)
        hm.set_elevation(0, 0, 10.0)
        assert hm.get_elevation(-1, 0) == 0.0
        assert hm.get_elevation(5, 0) == 0.0
        assert hm.get_elevation(0, 5) == 0.0

    def test_from_array(self):
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        hm = HeightMap.from_array(data, cell_size=0.5)
        assert hm.width == 3
        assert hm.height == 2
        assert hm.get_elevation(0, 0) == 1.0
        assert hm.get_elevation(2, 1) == 6.0

    def test_from_array_empty_raises(self):
        with pytest.raises(ValueError):
            HeightMap.from_array([])
        with pytest.raises(ValueError):
            HeightMap.from_array([[]])

    def test_from_array_ragged_raises(self):
        with pytest.raises(ValueError):
            HeightMap.from_array([[1, 2], [3]])


# ===========================================================================
# Procedural generation
# ===========================================================================

class TestProceduralGeneration:
    def test_from_noise_produces_valid_heightmap(self):
        hm = HeightMap.from_noise(20, 20, seed=42)
        # Should have non-zero values somewhere
        has_nonzero = any(
            hm.get_elevation(x, y) != 0.0
            for y in range(20)
            for x in range(20)
        )
        assert has_nonzero

    def test_from_noise_deterministic(self):
        hm1 = HeightMap.from_noise(10, 10, seed=123)
        hm2 = HeightMap.from_noise(10, 10, seed=123)
        for y in range(10):
            for x in range(10):
                assert hm1.get_elevation(x, y) == hm2.get_elevation(x, y)

    def test_from_noise_different_seeds_differ(self):
        hm1 = HeightMap.from_noise(10, 10, seed=1)
        hm2 = HeightMap.from_noise(10, 10, seed=999)
        differs = any(
            hm1.get_elevation(x, y) != hm2.get_elevation(x, y)
            for y in range(10)
            for x in range(10)
        )
        assert differs

    def test_from_noise_values_in_range(self):
        hm = HeightMap.from_noise(20, 20, seed=42, amplitude=10.0)
        for y in range(20):
            for x in range(20):
                e = hm.get_elevation(x, y)
                assert 0.0 <= e <= 10.0 + 0.01  # small epsilon

    def test_value_noise_deterministic(self):
        a = _value_noise_2d(1.5, 2.3, 42)
        b = _value_noise_2d(1.5, 2.3, 42)
        assert a == b

    def test_fbm_in_01(self):
        for x in range(10):
            for y in range(10):
                v = _fbm(x * 0.1, y * 0.1, 4, 0)
                assert 0.0 <= v <= 1.0 + 1e-9


# ===========================================================================
# World / cell conversion
# ===========================================================================

class TestCoordinateConversion:
    def test_world_to_cell(self):
        hm = HeightMap(10, 10, cell_size=2.0)
        assert hm.world_to_cell((3.0, 5.0)) == (1, 2)
        assert hm.world_to_cell((0.0, 0.0)) == (0, 0)

    def test_cell_to_world_center(self):
        hm = HeightMap(10, 10, cell_size=2.0)
        wx, wy = hm.cell_to_world(0, 0)
        assert wx == pytest.approx(1.0)
        assert wy == pytest.approx(1.0)

    def test_get_elevation_world(self):
        hm = HeightMap(10, 10, cell_size=1.0)
        hm.set_elevation(3, 4, 5.0)
        assert hm.get_elevation_world((3.5, 4.5)) == 5.0


# ===========================================================================
# Slope and normals
# ===========================================================================

class TestSlopeAndNormals:
    def test_flat_terrain_zero_slope(self):
        hm = HeightMap(10, 10)
        assert hm.slope_at(5, 5) == 0.0

    def test_sloped_terrain_nonzero(self):
        hm = HeightMap(10, 10)
        for x in range(10):
            for y in range(10):
                hm.set_elevation(x, y, float(x) * 2.0)
        slope = hm.slope_at(5, 5)
        assert slope > 0.0

    def test_slope_out_of_bounds(self):
        hm = HeightMap(5, 5)
        assert hm.slope_at(-1, 0) == 0.0
        assert hm.slope_at(5, 0) == 0.0

    def test_flat_terrain_normal_is_up(self):
        hm = HeightMap(10, 10)
        nx, ny, nz = hm.normal_at(5, 5)
        assert nx == pytest.approx(0.0)
        assert ny == pytest.approx(0.0)
        assert nz == pytest.approx(1.0)

    def test_sloped_normal_tilted(self):
        hm = HeightMap(10, 10)
        for x in range(10):
            for y in range(10):
                hm.set_elevation(x, y, float(x) * 5.0)
        nx, ny, nz = hm.normal_at(5, 5)
        # Normal should point away from slope (negative x component)
        assert nx < 0
        assert nz > 0
        # Should be unit length
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        assert length == pytest.approx(1.0)

    def test_normal_out_of_bounds(self):
        hm = HeightMap(5, 5)
        assert hm.normal_at(-1, 0) == (0.0, 0.0, 1.0)


# ===========================================================================
# Bresenham
# ===========================================================================

class TestBresenham:
    def test_horizontal_line(self):
        cells = _bresenham(0, 0, 5, 0)
        assert cells[0] == (0, 0)
        assert cells[-1] == (5, 0)
        assert len(cells) == 6

    def test_same_point(self):
        cells = _bresenham(3, 3, 3, 3)
        assert cells == [(3, 3)]

    def test_diagonal(self):
        cells = _bresenham(0, 0, 3, 3)
        assert (0, 0) in cells
        assert (3, 3) in cells


# ===========================================================================
# Line of Sight
# ===========================================================================

class TestLineOfSight:
    def test_flat_terrain_always_visible(self):
        hm = HeightMap(20, 20)
        los = LineOfSight(hm)
        assert los.can_see((1.0, 1.0), (18.0, 18.0))

    def test_hill_blocks_los(self):
        hm = HeightMap(20, 20)
        # Create a ridge in the middle
        for y in range(20):
            hm.set_elevation(10, y, 50.0)
        los = LineOfSight(hm, observer_height=1.8)
        # Observer at x=2, target at x=18 — ridge at x=10 should block
        assert not los.can_see((2.5, 10.5), (18.5, 10.5))

    def test_tall_observer_sees_over_hill(self):
        hm = HeightMap(20, 20)
        for y in range(20):
            hm.set_elevation(10, y, 5.0)
        los = LineOfSight(hm)
        # With very tall height, should see over a 5m hill
        assert los.can_see((2.5, 10.5), (18.5, 10.5), from_height=100.0, to_height=100.0)

    def test_adjacent_cells_always_visible(self):
        hm = HeightMap(10, 10)
        hm.set_elevation(5, 5, 100.0)  # doesn't matter for adjacent
        los = LineOfSight(hm)
        assert los.can_see((0.5, 0.5), (1.5, 0.5))

    def test_same_position_visible(self):
        hm = HeightMap(10, 10)
        los = LineOfSight(hm)
        assert los.can_see((5.5, 5.5), (5.5, 5.5))

    def test_visibility_map_flat_terrain(self):
        hm = HeightMap(20, 20)
        los = LineOfSight(hm)
        vis = los.visibility_map((10.5, 10.5), 5.0)
        # On flat terrain, should see many cells within radius
        assert len(vis) > 10
        # Origin cell should be visible
        assert (10, 10) in vis

    def test_visibility_map_zero_radius(self):
        hm = HeightMap(10, 10)
        los = LineOfSight(hm)
        vis = los.visibility_map((5.5, 5.5), 0.0)
        # Should be empty or just the origin cell
        assert len(vis) <= 1

    def test_visibility_map_ridge_blocks(self):
        hm = HeightMap(20, 20)
        for y in range(20):
            hm.set_elevation(10, y, 50.0)
        los = LineOfSight(hm, observer_height=1.8)
        vis = los.visibility_map((5.5, 10.5), 15.0)
        # Cells beyond the ridge (x > 10) should not be visible
        beyond_ridge = [c for c in vis if c[0] > 10]
        assert len(beyond_ridge) == 0

    def test_find_defilade_returns_hidden_positions(self):
        hm = HeightMap(20, 20)
        # Create a hill that provides cover
        for y in range(8, 13):
            hm.set_elevation(10, y, 30.0)
        los = LineOfSight(hm, observer_height=1.8)
        # Threat at (0.5, 10.5), find positions near (15.5, 10.5) hidden from threat
        defilade = los.find_defilade((15.5, 10.5), (0.5, 10.5), 5.0)
        # Should find some hidden positions behind the hill
        assert len(defilade) > 0

    def test_find_defilade_sorted_by_distance(self):
        hm = HeightMap(20, 20)
        for y in range(20):
            hm.set_elevation(10, y, 30.0)
        los = LineOfSight(hm, observer_height=1.8)
        defilade = los.find_defilade((15.5, 10.5), (0.5, 10.5), 8.0)
        if len(defilade) >= 2:
            for i in range(len(defilade) - 1):
                d0 = math.hypot(defilade[i][0] - 15.5, defilade[i][1] - 10.5)
                d1 = math.hypot(defilade[i + 1][0] - 15.5, defilade[i + 1][1] - 10.5)
                assert d0 <= d1 + 1e-9


# ===========================================================================
# CoverMap
# ===========================================================================

class TestCoverMap:
    def test_no_obstacles_no_cover(self):
        hm = HeightMap(10, 10)
        cm = CoverMap(hm, obstacles=[])
        cover = cm.cover_value((5.0, 5.0), (1.0, 0.0))
        assert cover == 0.0

    def test_obstacle_provides_cover(self):
        hm = HeightMap(20, 20)
        # Obstacle at (5, 5) with radius 2, pos at (3, 5), threat from east
        cm = CoverMap(hm, obstacles=[((5.0, 5.0), 2.0)])
        cover = cm.cover_value((3.0, 5.0), (1.0, 0.0))
        assert cover > 0.0

    def test_obstacle_behind_no_cover(self):
        hm = HeightMap(20, 20)
        # Obstacle behind us relative to threat direction
        cm = CoverMap(hm, obstacles=[((1.0, 5.0), 2.0)])
        cover = cm.cover_value((5.0, 5.0), (1.0, 0.0))  # threat from east, obstacle to west
        assert cover == 0.0

    def test_terrain_ridge_provides_cover(self):
        hm = HeightMap(20, 20)
        # Ridge in threat direction
        for y in range(20):
            hm.set_elevation(7, y, 10.0)
        cm = CoverMap(hm, obstacles=[])
        cover = cm.cover_value((5.5, 10.5), (1.0, 0.0))
        assert cover > 0.0

    def test_cover_clamped_to_one(self):
        hm = HeightMap(20, 20)
        # Many obstacles = cover should not exceed 1.0
        obstacles = [((5.0 + i * 0.5, 5.0), 3.0) for i in range(10)]
        cm = CoverMap(hm, obstacles=obstacles)
        cover = cm.cover_value((3.0, 5.0), (1.0, 0.0))
        assert cover <= 1.0

    def test_zero_threat_dir_returns_zero(self):
        hm = HeightMap(10, 10)
        cm = CoverMap(hm, obstacles=[((5.0, 5.0), 2.0)])
        cover = cm.cover_value((3.0, 5.0), (0.0, 0.0))
        assert cover == 0.0

    def test_generate_cover_grid(self):
        hm = HeightMap(10, 10, cell_size=1.0)
        cm = CoverMap(hm, obstacles=[((5.0, 5.0), 2.0)])
        grid = cm.generate_cover_grid(cell_size=2.0)
        assert len(grid) > 0
        # Each entry has 8 direction values
        for key, vals in grid.items():
            assert len(vals) == 8
            for v in vals:
                assert 0.0 <= v <= 1.0


# ===========================================================================
# MovementCost
# ===========================================================================

class TestMovementCost:
    def test_flat_terrain_cost_proportional_to_distance(self):
        hm = HeightMap(20, 20)
        mc = MovementCost(hm)
        cost = mc.cost((0.5, 0.5), (10.5, 0.5))
        expected_dist = 10.0
        # On flat terrain cost ~ base_cost * distance
        assert cost == pytest.approx(expected_dist, rel=0.1)

    def test_uphill_costs_more(self):
        hm = HeightMap(20, 20)
        for x in range(20):
            for y in range(20):
                hm.set_elevation(x, y, float(x) * 5.0)
        mc = MovementCost(hm)
        uphill_cost = mc.cost((2.5, 10.5), (8.5, 10.5))
        flat_hm = HeightMap(20, 20)
        flat_mc = MovementCost(flat_hm)
        flat_cost = flat_mc.cost((2.5, 10.5), (8.5, 10.5))
        assert uphill_cost > flat_cost

    def test_same_position_zero_cost(self):
        hm = HeightMap(10, 10)
        mc = MovementCost(hm)
        assert mc.cost((5.0, 5.0), (5.0, 5.0)) == 0.0

    def test_speed_modifier_flat_is_one(self):
        hm = HeightMap(10, 10)
        mc = MovementCost(hm)
        mod = mc.max_speed_modifier((5.5, 5.5))
        assert mod == pytest.approx(1.0)

    def test_speed_modifier_steep_less_than_one(self):
        hm = HeightMap(10, 10)
        for x in range(10):
            for y in range(10):
                hm.set_elevation(x, y, float(x) * 20.0)
        mc = MovementCost(hm)
        mod = mc.max_speed_modifier((5.5, 5.5))
        assert 0.0 <= mod < 1.0

    def test_speed_modifier_in_range(self):
        hm = HeightMap.from_noise(20, 20, seed=42, amplitude=50.0)
        mc = MovementCost(hm)
        for y in range(20):
            for x in range(20):
                pos = hm.cell_to_world(x, y)
                mod = mc.max_speed_modifier(pos)
                assert 0.0 <= mod <= 1.0

    def test_custom_base_cost(self):
        hm = HeightMap(10, 10)
        mc = MovementCost(hm, base_cost=3.0)
        cost = mc.cost((0.5, 0.5), (5.5, 0.5))
        # Should be roughly 3x the default
        mc_default = MovementCost(hm, base_cost=1.0)
        cost_default = mc_default.cost((0.5, 0.5), (5.5, 0.5))
        assert cost == pytest.approx(cost_default * 3.0, rel=0.01)
