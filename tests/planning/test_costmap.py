# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.planning.costmap — layer-driven costmap generation."""

import math

import pytest

from tritium_lib.planning.costmap import (
    Costmap,
    CostmapBuilder,
    CostmapWeights,
    costmap_from_terrain_map,
)
from tritium_lib.planning.layers import LocalElevationGrid

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


# ---------------------------------------------------------------------------
# Coordinate round-trips
# ---------------------------------------------------------------------------

class TestCoordinates:
    def test_grid_dimensions(self):
        b = CostmapBuilder((0, 0, 100, 60), resolution=10.0)
        cm = b.build()
        assert cm.width == 10
        assert cm.height == 6
        assert cm.origin_x == 0
        assert cm.origin_y == 0

    def test_grid_to_world_is_cell_center(self):
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        cm = b.build()
        assert cm.grid_to_world(0, 0) == (5.0, 5.0)
        assert cm.grid_to_world(9, 9) == (95.0, 95.0)

    def test_world_grid_roundtrip(self):
        b = CostmapBuilder((-50, -50, 50, 50), resolution=5.0)
        cm = b.build()
        for col in range(0, cm.width, 3):
            for row in range(0, cm.height, 3):
                x, y = cm.grid_to_world(col, row)
                assert cm.world_to_grid(x, y) == (col, row)

    def test_world_to_grid_out_of_bounds(self):
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        cm = b.build()
        assert cm.world_to_grid(-1, 50) is None
        assert cm.world_to_grid(50, -1) is None
        assert cm.world_to_grid(100.1, 50) is None

    def test_bounds(self):
        cm = CostmapBuilder((10, 20, 110, 220), resolution=10.0).build()
        assert cm.bounds() == (10, 20, 110, 220)


# ---------------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------------

class TestObstacles:
    def test_single_center_cell_lethal(self):
        # 3x3 grid; polygon covers only the center cell center (15, 15).
        b = CostmapBuilder((0, 0, 30, 30), resolution=10.0)
        b.add_obstacles(_polygon_fc([[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]))
        cm = b.build()
        lethal = {
            (c, r)
            for c in range(cm.width)
            for r in range(cm.height)
            if cm.is_lethal(c, r)
        }
        assert lethal == {(1, 1)}

    def test_larger_polygon_block(self):
        # Polygon covering a 2x2 block of cell centers.
        b = CostmapBuilder((0, 0, 40, 40), resolution=10.0)
        b.add_obstacles(_polygon_fc([[10, 10], [30, 10], [30, 30], [10, 30], [10, 10]]))
        cm = b.build()
        lethal = {
            (c, r)
            for c in range(cm.width)
            for r in range(cm.height)
            if cm.is_lethal(c, r)
        }
        # Cell centers inside [10,30]^2 are (15,15),(25,15),(15,25),(25,25).
        assert lethal == {(1, 1), (2, 1), (1, 2), (2, 2)}

    def test_kind_tag_recorded(self):
        b = CostmapBuilder((0, 0, 30, 30), resolution=10.0)
        b.add_obstacles(
            _polygon_fc([[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]),
            kind="water",
        )
        # Introspection: the internal tag store records the kind.
        assert b._obstacle_cells[(1, 1)] == "water"

    def test_multipolygon_obstacle(self):
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {},
                 "geometry": {
                     "type": "MultiPolygon",
                     "coordinates": [
                         [[[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]],
                         [[[40, 40], [50, 40], [50, 50], [40, 50], [40, 40]]],
                     ],
                 }},
            ],
        }
        b = CostmapBuilder((0, 0, 60, 60), resolution=10.0)
        b.add_obstacles(fc)
        cm = b.build()
        assert cm.is_lethal(1, 1)
        assert cm.is_lethal(4, 4)
        assert not cm.is_lethal(0, 0)


# ---------------------------------------------------------------------------
# Roads
# ---------------------------------------------------------------------------

class TestRoads:
    def test_road_corridor_default_width(self):
        # Horizontal road at y=25; default width 8 -> half 4 -> rows 4 and 5.
        b = CostmapBuilder((0, 0, 50, 50), resolution=5.0)
        b.add_roads(_line_fc([[0, 25], [50, 25]]))
        cm = b.build()
        road = {
            (c, r)
            for c in range(cm.width)
            for r in range(cm.height)
            if cm.cost_at(c, r) < 1.0
        }
        expected = {(c, r) for c in range(cm.width) for r in (4, 5)}
        assert road == expected

    def test_road_cost_is_discounted(self):
        b = CostmapBuilder((0, 0, 50, 50), resolution=5.0)
        b.add_roads(_line_fc([[0, 25], [50, 25]]))
        cm = b.build()
        col, row = cm.world_to_grid(25, 27.5)
        assert cm.cost_at(col, row) == pytest.approx(
            CostmapWeights().base_cost * CostmapWeights().road_discount
        )

    def test_road_width_override_widens_corridor(self):
        # width_m=16 -> half 8 -> rows 3..6.
        b = CostmapBuilder((0, 0, 50, 50), resolution=5.0)
        b.add_roads(_line_fc([[0, 25], [50, 25]], props={"width_m": 16.0}))
        cm = b.build()
        road_rows = {
            r
            for c in range(cm.width)
            for r in range(cm.height)
            if cm.cost_at(c, r) < 1.0
        }
        assert road_rows == {3, 4, 5, 6}

    def test_polygon_road_marks_covered_cells(self):
        fc = _polygon_fc([[10, 10], [30, 10], [30, 30], [10, 30], [10, 10]])
        b = CostmapBuilder((0, 0, 40, 40), resolution=10.0)
        b.add_roads(fc)
        cm = b.build()
        assert cm.cost_at(1, 1) < 1.0
        assert cm.cost_at(2, 2) < 1.0
        assert cm.cost_at(0, 0) == pytest.approx(1.0)

    def test_multilinestring_road(self):
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"width_m": 8.0},
                 "geometry": {
                     "type": "MultiLineString",
                     "coordinates": [
                         [[0, 25], [50, 25]],
                         [[25, 0], [25, 50]],
                     ],
                 }},
            ],
        }
        b = CostmapBuilder((0, 0, 50, 50), resolution=5.0)
        b.add_roads(fc)
        cm = b.build()
        # Both the horizontal and vertical corridors are discounted.
        assert cm.cost_at(*cm.world_to_grid(25, 27.5)) < 1.0
        assert cm.cost_at(*cm.world_to_grid(27.5, 25)) < 1.0


# ---------------------------------------------------------------------------
# DEM slope
# ---------------------------------------------------------------------------

class TestDemSlope:
    def test_gentle_ramp_adds_slope_cost(self):
        # slope 0.1 everywhere -> cost = 1 + 5*0.1 = 1.5, none lethal.
        dem = LocalElevationGrid.from_callable((-30, -30, 130, 130), 10.0, lambda x, y: 0.1 * x)
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        b.add_dem(dem)
        cm = b.build()
        for c in range(cm.width):
            for r in range(cm.height):
                assert not cm.is_lethal(c, r)
                assert cm.cost_at(c, r) == pytest.approx(1.5)

    def test_steep_region_lethal(self):
        # slope 1.0 > max_slope 0.7 everywhere -> all lethal.
        dem = LocalElevationGrid.from_callable((-30, -30, 130, 130), 10.0, lambda x, y: 1.0 * x)
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        b.add_dem(dem)
        cm = b.build()
        assert all(
            cm.is_lethal(c, r)
            for c in range(cm.width)
            for r in range(cm.height)
        )

    def test_localized_steep_band(self):
        # Flat everywhere except a steep cone flank around the map center.
        def cone(x, y):
            return max(0.0, 60.0 - 1.0 * math.hypot(x - 100, y - 100))

        dem = LocalElevationGrid.from_callable((-30, -30, 230, 230), 10.0, cone)
        b = CostmapBuilder((0, 0, 200, 200), resolution=10.0)
        b.add_dem(dem)
        cm = b.build()
        # A corner cell far from the cone is flat: cheap and traversable.
        assert not cm.is_lethal(0, 0)
        assert cm.cost_at(0, 0) == pytest.approx(1.0)
        # A flank cell near the center is lethal (steep).
        assert cm.is_lethal(*cm.world_to_grid(70, 100))


# ---------------------------------------------------------------------------
# Combined build order-independence
# ---------------------------------------------------------------------------

class TestBuildOrderIndependence:
    def _layers(self):
        obs = _polygon_fc([[40, 40], [60, 40], [60, 60], [40, 60], [40, 40]])
        road = _line_fc([[0, 30], [100, 30]], props={"width_m": 10.0})
        dem = LocalElevationGrid.from_callable(
            (-30, -30, 130, 130), 10.0, lambda x, y: 0.05 * x + 0.05 * y
        )
        return obs, road, dem

    def test_order_independent(self):
        obs, road, dem = self._layers()

        b1 = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        b1.add_obstacles(obs)
        b1.add_roads(road)
        b1.add_dem(dem)
        cm1 = b1.build()

        b2 = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        b2.add_dem(dem)
        b2.add_roads(road)
        b2.add_obstacles(obs)
        cm2 = b2.build()

        assert cm1.grid == cm2.grid

    def test_obstacle_beats_road_on_overlap(self):
        # A cell that is both road and obstacle ends up lethal.
        obs = _polygon_fc([[40, 20], [60, 20], [60, 40], [40, 40], [40, 20]])
        road = _line_fc([[0, 30], [100, 30]], props={"width_m": 10.0})
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        b.add_roads(road)
        b.add_obstacles(obs)
        cm = b.build()
        # (50, 30) is on the road line and inside the obstacle -> lethal.
        assert cm.is_lethal(*cm.world_to_grid(50, 30))


# ---------------------------------------------------------------------------
# Inflation
# ---------------------------------------------------------------------------

class TestInflation:
    def test_inflation_ring(self):
        weights = CostmapWeights(obstacle_inflation_m=10.0, inflation_cost=3.0)
        b = CostmapBuilder((0, 0, 50, 50), resolution=10.0, weights=weights)
        # Obstacle covers center cell (2, 2) center (25, 25).
        b.add_obstacles(_polygon_fc([[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]]))
        cm = b.build()
        assert cm.is_lethal(2, 2)
        # Orthogonal neighbors are exactly 10 m away -> inflated to 3.0.
        for c, r in [(1, 2), (3, 2), (2, 1), (2, 3)]:
            assert cm.cost_at(c, r) == pytest.approx(3.0)
        # Diagonal neighbors are ~14.1 m away -> not inflated.
        for c, r in [(1, 1), (3, 3), (1, 3), (3, 1)]:
            assert cm.cost_at(c, r) == pytest.approx(1.0)

    def test_no_inflation_by_default(self):
        b = CostmapBuilder((0, 0, 50, 50), resolution=10.0)
        b.add_obstacles(_polygon_fc([[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]]))
        cm = b.build()
        assert cm.cost_at(1, 2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# min_traversable_cost
# ---------------------------------------------------------------------------

class TestMinTraversableCost:
    def test_min_with_roads(self):
        b = CostmapBuilder((0, 0, 50, 50), resolution=5.0)
        b.add_roads(_line_fc([[0, 25], [50, 25]]))
        cm = b.build()
        assert cm.min_traversable_cost() == pytest.approx(0.5)

    def test_all_lethal_returns_epsilon(self):
        cm = Costmap(
            origin_x=0, origin_y=0, resolution=5.0, width=2, height=2,
            grid=[[LETHAL, LETHAL], [LETHAL, LETHAL]],
        )
        assert cm.min_traversable_cost() == pytest.approx(1e-6)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:
    def test_shape_and_lethal_encoding(self):
        b = CostmapBuilder((0, 0, 30, 30), resolution=10.0)
        b.add_obstacles(_polygon_fc([[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]))
        cm = b.build()
        tel = cm.to_telemetry()
        assert set(tel.keys()) == {"grid", "cell_size", "bounds", "max_cost"}
        assert tel["cell_size"] == 10.0
        assert tel["bounds"] == [0, 0, 30, 30]
        assert len(tel["grid"]) == 3
        assert len(tel["grid"][0]) == 3
        # Lethal encoded as -1.0.
        assert tel["grid"][1][1] == -1.0
        # Non-lethal open cells carry their cost.
        assert tel["grid"][0][0] == pytest.approx(1.0)
        assert tel["max_cost"] == pytest.approx(1.0)

    def test_downsample_path(self):
        # 10x10 = 100 cells; cap at 25 -> stride 2 -> 5x5.
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        b.add_obstacles(_polygon_fc([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]))
        cm = b.build()
        tel = cm.to_telemetry(max_cells=25)
        assert len(tel["grid"]) == 5
        assert len(tel["grid"][0]) == 5
        assert tel["cell_size"] == 20.0
        # The block containing the lethal corner cell (0,0) wins as lethal.
        assert tel["grid"][0][0] == -1.0

    def test_downsample_block_takes_max_cost(self):
        weights = CostmapWeights(road_discount=0.5)
        b = CostmapBuilder((0, 0, 40, 40), resolution=10.0, weights=weights)
        # One road cell in the SW block; the block max should be base 1.0.
        b.add_roads(_line_fc([[5, 5], [5, 6]], props={"width_m": 4.0}))
        cm = b.build()
        tel = cm.to_telemetry(max_cells=4)  # 4x4 -> 2x2
        assert len(tel["grid"]) == 2
        # SW block holds a road (0.5) and open (1.0) cells -> max 1.0.
        assert tel["grid"][0][0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TerrainMap adapter
# ---------------------------------------------------------------------------

class FakeTerrainMap:
    """Duck-typed stand-in for the sim TerrainMap."""

    def __init__(self, grid_size=10, resolution=10.0, default="open", cells=None):
        self.grid_size = grid_size
        self.resolution = resolution
        self._default = default
        self._cells = cells or {}

    def get_terrain_at(self, col, row):
        if col < 0 or row < 0 or col >= self.grid_size or row >= self.grid_size:
            return "out_of_bounds"
        return self._cells.get((col, row), self._default)

    def _grid_to_world(self, col, row):
        # SW-anchored like TerrainMap (half-extent = grid_size*res/2).
        half = self.grid_size * self.resolution / 2.0
        return (
            col * self.resolution + self.resolution * 0.5 - half,
            row * self.resolution + self.resolution * 0.5 - half,
        )


class TestTerrainMapAdapter:
    def test_terrain_mapping(self):
        ftm = FakeTerrainMap(
            cells={(5, 5): "building", (6, 6): "road", (7, 7): "water", (2, 2): "yard"}
        )
        cm = costmap_from_terrain_map(ftm)
        assert cm.width == 10
        assert cm.height == 10
        assert cm.is_lethal(5, 5)      # building
        assert cm.is_lethal(7, 7)      # water
        assert cm.cost_at(6, 6) == pytest.approx(0.5)   # road
        assert cm.cost_at(2, 2) == pytest.approx(1.0)   # yard -> base
        assert cm.cost_at(0, 0) == pytest.approx(1.0)   # open -> base

    def test_origin_derived_from_cell_center(self):
        ftm = FakeTerrainMap(grid_size=10, resolution=10.0)
        cm = costmap_from_terrain_map(ftm)
        # half-extent 50 -> SW corner at (-50, -50).
        assert cm.origin_x == pytest.approx(-50.0)
        assert cm.origin_y == pytest.approx(-50.0)
        # A costmap cell center matches the terrain cell center.
        assert cm.grid_to_world(0, 0) == pytest.approx(ftm._grid_to_world(0, 0))

    def test_custom_weights(self):
        ftm = FakeTerrainMap(cells={(3, 3): "road"})
        cm = costmap_from_terrain_map(
            ftm, weights=CostmapWeights(base_cost=2.0, road_discount=0.25)
        )
        assert cm.cost_at(3, 3) == pytest.approx(0.5)   # 2.0 * 0.25
        assert cm.cost_at(0, 0) == pytest.approx(2.0)
