# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.world.terrain_map — grid-based terrain map."""

import math
import pytest

from tritium_lib.sim_engine.world.terrain_map import (
    TerrainMap,
    TerrainCell,
    _bresenham,
    _point_in_polygon,
    _TERRAIN_PROPERTIES,
)


class TestTerrainCell:
    def test_defaults(self):
        c = TerrainCell(x=0.0, y=0.0, terrain_type="open")
        assert c.movement_cost == 1.0
        assert c.cover_value == 0.0
        assert c.visibility == 1.0
        assert c.elevation == 0.0


class TestTerrainMap:
    def test_default_is_open(self):
        tm = TerrainMap(100.0)
        cell = tm.get_cell(0.0, 0.0)
        assert cell.terrain_type == "open"
        assert cell.movement_cost == 1.0

    def test_set_and_get_cell(self):
        tm = TerrainMap(100.0)
        tm.set_cell(10.0, 20.0, "road")
        cell = tm.get_cell(10.0, 20.0)
        assert cell.terrain_type == "road"
        assert cell.movement_cost == 0.7

    def test_set_building(self):
        tm = TerrainMap(100.0)
        tm.set_cell(50.0, 50.0, "building")
        cell = tm.get_cell(50.0, 50.0)
        assert cell.terrain_type == "building"
        assert cell.movement_cost == float("inf")
        assert cell.cover_value == 0.5
        assert cell.visibility == 0.0

    def test_set_terrain_alias(self):
        tm = TerrainMap(100.0)
        tm.set_terrain(30.0, 30.0, "water")
        assert tm.get_terrain_type(30.0, 30.0) == "water"

    def test_get_speed_multiplier_road(self):
        tm = TerrainMap(100.0)
        tm.set_cell(0.0, 0.0, "road")
        assert tm.get_speed_multiplier(0.0, 0.0) == 1.2

    def test_get_speed_multiplier_building(self):
        tm = TerrainMap(100.0)
        tm.set_cell(0.0, 0.0, "building")
        assert tm.get_speed_multiplier(0.0, 0.0) == 0.0

    def test_get_speed_multiplier_open(self):
        tm = TerrainMap(100.0)
        assert tm.get_speed_multiplier(0.0, 0.0) == 1.0

    def test_get_movement_cost(self):
        tm = TerrainMap(100.0)
        tm.set_cell(0.0, 0.0, "yard")
        assert tm.get_movement_cost(0.0, 0.0) == 1.0

    def test_get_cover_value(self):
        tm = TerrainMap(100.0)
        tm.set_cell(0.0, 0.0, "yard")
        assert tm.get_cover_value(0.0, 0.0) == pytest.approx(0.1)

    def test_get_visibility(self):
        tm = TerrainMap(100.0)
        tm.set_cell(0.0, 0.0, "yard")
        assert tm.get_visibility(0.0, 0.0) == pytest.approx(0.8)

    def test_get_cost_by_grid(self):
        tm = TerrainMap(100.0, resolution=5.0)
        # Default open cell
        assert tm.get_cost(5, 5) == 1.0
        # Out of bounds
        assert tm.get_cost(-1, 0) == float("inf")

    def test_get_terrain_at_by_grid(self):
        tm = TerrainMap(100.0, resolution=5.0)
        assert tm.get_terrain_at(5, 5) == "open"
        assert tm.get_terrain_at(-1, 0) == "out_of_bounds"

    def test_reset(self):
        tm = TerrainMap(100.0)
        tm.set_cell(0.0, 0.0, "road")
        tm.reset()
        assert tm.get_terrain_type(0.0, 0.0) == "open"

    def test_grid_size(self):
        tm = TerrainMap(50.0, resolution=5.0)
        # 2 * 50 / 5 + 1 = 21
        assert tm.grid_size == 21

    def test_bounds(self):
        tm = TerrainMap(200.0)
        assert tm.bounds == 200.0

    def test_resolution(self):
        tm = TerrainMap(100.0, resolution=2.0)
        assert tm.resolution == 2.0


class TestTerrainSpeed:
    def test_drone_ignores_terrain(self):
        tm = TerrainMap(100.0)
        tm.set_cell(0.0, 0.0, "building")
        assert tm.get_speed_modifier(0.0, 0.0, "drone") == 1.0

    def test_person_yard_penalty(self):
        tm = TerrainMap(100.0)
        tm.set_cell(0.0, 0.0, "yard")
        modifier = tm.get_speed_modifier(0.0, 0.0, "person")
        # 1.0/1.0 * 0.9 = 0.9
        assert abs(modifier - 0.9) < 1e-6

    def test_road_speed_boost(self):
        tm = TerrainMap(100.0)
        tm.set_cell(0.0, 0.0, "road")
        modifier = tm.get_speed_modifier(0.0, 0.0, "rover")
        # 1.0 / 0.7 ~ 1.43
        assert modifier > 1.0

    def test_building_blocks_ground(self):
        tm = TerrainMap(100.0)
        tm.set_cell(0.0, 0.0, "building")
        modifier = tm.get_speed_modifier(0.0, 0.0, "rover")
        assert modifier == 0.0

    def test_custom_flying_checker(self):
        def checker(asset_type):
            return asset_type == "custom_flyer"
        tm = TerrainMap(100.0, is_flying_checker=checker)
        tm.set_cell(0.0, 0.0, "building")
        assert tm.get_speed_modifier(0.0, 0.0, "custom_flyer") == 1.0
        assert tm.get_speed_modifier(0.0, 0.0, "rover") == 0.0


class TestLineOfSight:
    def test_clear_los(self):
        tm = TerrainMap(100.0)
        assert tm.line_of_sight((0.0, 0.0), (50.0, 50.0)) is True

    def test_blocked_by_building(self):
        tm = TerrainMap(100.0, resolution=5.0)
        # Place a building in the middle
        tm.set_cell(25.0, 0.0, "building")
        # LOS from left to right through the building
        assert tm.line_of_sight((0.0, 0.0), (50.0, 0.0)) is False

    def test_same_point(self):
        tm = TerrainMap(100.0)
        assert tm.line_of_sight((10.0, 10.0), (10.0, 10.0)) is True


class TestTerrainLayout:
    def test_load_buildings(self):
        tm = TerrainMap(100.0, resolution=5.0)
        buildings = [{
            "footprint": [(10, 10), (30, 10), (30, 30), (10, 30)],
            "position": (20, 20),
        }]
        tm.load_buildings(buildings)
        # Center should be building
        assert tm.get_terrain_type(20.0, 20.0) == "building"
        # Outside should be open
        assert tm.get_terrain_type(0.0, 0.0) == "open"

    def test_load_roads(self):
        tm = TerrainMap(100.0, resolution=5.0)
        roads = [{
            "start": (-50.0, 0.0),
            "end": (50.0, 0.0),
            "width": 10.0,
        }]
        tm.load_roads(roads)
        # Along the road should be road terrain
        assert tm.get_terrain_type(0.0, 0.0) == "road"

    def test_load_from_layout(self):
        tm = TerrainMap(100.0, resolution=5.0)
        layout = {
            "objects": [
                {
                    "type": "building",
                    "position": {"x": 20, "z": 20},
                    "properties": {
                        "footprint": [[10, 10], [30, 10], [30, 30], [10, 30]],
                    },
                },
            ],
        }
        tm.load_from_layout(layout)
        assert tm.get_terrain_type(20.0, 20.0) == "building"

    def test_find_terrain_of_type(self):
        tm = TerrainMap(100.0, resolution=5.0)
        tm.set_cell(10.0, 10.0, "road")
        tm.set_cell(20.0, 20.0, "road")
        tm.set_cell(80.0, 80.0, "road")
        results = tm.find_terrain_of_type("road")
        assert len(results) == 3

    def test_find_terrain_near(self):
        tm = TerrainMap(100.0, resolution=5.0)
        tm.set_cell(10.0, 10.0, "road")
        tm.set_cell(80.0, 80.0, "road")
        results = tm.find_terrain_of_type("road", near=(0.0, 0.0), radius=20.0)
        assert len(results) == 1


class TestTelemetry:
    def test_to_telemetry(self):
        tm = TerrainMap(50.0, resolution=5.0)
        tm.set_cell(10.0, 10.0, "road")
        telem = tm.to_telemetry()
        assert telem["bounds"] == 50.0
        assert telem["resolution"] == 5.0
        assert len(telem["cells"]) == 1
        assert telem["cells"][0]["terrain_type"] == "road"


class TestBresenham:
    def test_horizontal(self):
        cells = _bresenham(0, 0, 5, 0)
        assert len(cells) == 6
        assert cells[0] == (0, 0)
        assert cells[-1] == (5, 0)

    def test_vertical(self):
        cells = _bresenham(0, 0, 0, 5)
        assert len(cells) == 6

    def test_diagonal(self):
        cells = _bresenham(0, 0, 3, 3)
        assert (0, 0) in cells
        assert (3, 3) in cells

    def test_single_point(self):
        cells = _bresenham(2, 3, 2, 3)
        assert cells == [(2, 3)]


class TestPointInPolygon:
    def test_inside_square(self):
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert _point_in_polygon(5.0, 5.0, poly) is True

    def test_outside_square(self):
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert _point_in_polygon(15.0, 5.0, poly) is False

    def test_triangle(self):
        poly = [(0, 0), (10, 0), (5, 10)]
        assert _point_in_polygon(5.0, 3.0, poly) is True
        assert _point_in_polygon(0.0, 10.0, poly) is False
