# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.planning.builder_from_terrain_map.

``builder_from_terrain_map`` seeds a :class:`CostmapBuilder` from a sim
``TerrainMap`` and returns the *builder* (not a finished costmap) so the sc
engine can enrich the terrain baseline with GIS layers — DEM slope, TIGER
roads, FEMA flood, NOAA weather zones — before calling ``build()``.

Covers:
    - parity: ``costmap_from_terrain_map`` == ``builder_from_terrain_map().build()``
      cell-for-cell (the finished convenience path is now the builder path).
    - enrichment: seed terrain, then layer TIGER roads / FEMA flood / NOAA
      zones via ``add_gis_features`` and assert the resulting cell costs.
    - dem: attach a DEM ramp and assert sloped cells cost more than flat and
      steep cells go lethal.
    - frame: the built costmap's origin/resolution/width/height match
      ``costmap_from_terrain_map``, including a non-integer resolution where a
      naive ``ceil`` span could drift by a cell.

All deterministic — no network, no sc imports.
"""

import math

import pytest

from tritium_lib.planning import (
    builder_from_terrain_map,
    costmap_from_terrain_map,
)
from tritium_lib.planning.costmap import CostmapBuilder, CostmapWeights
from tritium_lib.planning.layers import LocalElevationGrid


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------

class FakeTerrainMap:
    """Duck-typed stand-in for the sim TerrainMap.

    The SW corner of the map sits at the local origin ``(0, 0)``: cell
    ``(0, 0)``'s center is ``(res/2, res/2)`` so
    ``builder_from_terrain_map`` derives an origin of ``(0, 0)`` and the grid
    spans ``[0, grid_size*res]`` on each axis.  This keeps enrichment-layer
    coordinates in natural positive meters.
    """

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
        r = self.resolution
        return (col * r + r * 0.5, row * r + r * 0.5)


class SwAnchoredTerrainMap(FakeTerrainMap):
    """Like the real TerrainMap: the map is *centered* on the local origin.

    Cell ``(0, 0)``'s center is the SW-most cell center at
    ``(res/2 - half, res/2 - half)`` where ``half = grid_size*res/2``, so the
    derived origin is ``(-half, -half)``.  Used to prove the origin
    derivation, not just the origin-at-zero convenience convention.
    """

    def _grid_to_world(self, col, row):
        r = self.resolution
        half = self.grid_size * r / 2.0
        return (col * r + r * 0.5 - half, row * r + r * 0.5 - half)


def _mixed_cells():
    """A spread of every terrain kind the mapping table cares about."""
    return {
        (5, 5): "building",
        (6, 6): "road",
        (7, 7): "water",
        (2, 2): "yard",
        (0, 0): "road",
        (9, 9): "building",
        (3, 8): "water",
        (8, 3): "road",
    }


def _fc(features):
    return {"type": "FeatureCollection", "features": features}


def _tiger_line(coords, kind, width_m=None):
    props = {"source": "tiger", "kind": kind, "name": ""}
    if width_m is not None:
        props["width_m"] = width_m
    return {
        "type": "Feature", "properties": props,
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def _fema_poly(ring, sfha, kind="AE"):
    return {
        "type": "Feature",
        "properties": {"source": "fema", "kind": kind, "sfha": sfha},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _noaa_poly(ring, severity, event="Extreme Wind Warning"):
    return {
        "type": "Feature",
        "properties": {"source": "noaa", "kind": event, "severity": severity},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _square(cx, cy, half):
    return [
        [cx - half, cy - half], [cx + half, cy - half],
        [cx + half, cy + half], [cx - half, cy + half],
        [cx - half, cy - half],
    ]


# ---------------------------------------------------------------------------
# Parity — the finished convenience path IS the builder path
# ---------------------------------------------------------------------------

class TestParity:
    def test_returns_a_builder(self):
        b = builder_from_terrain_map(FakeTerrainMap())
        assert isinstance(b, CostmapBuilder)

    def test_parity_default_weights_cell_for_cell(self):
        ftm = FakeTerrainMap(cells=_mixed_cells())
        cm_direct = costmap_from_terrain_map(ftm)
        cm_builder = builder_from_terrain_map(ftm).build()
        assert cm_builder.grid == cm_direct.grid

    def test_parity_sw_anchored_map(self):
        ftm = SwAnchoredTerrainMap(cells=_mixed_cells())
        cm_direct = costmap_from_terrain_map(ftm)
        cm_builder = builder_from_terrain_map(ftm).build()
        assert cm_builder.grid == cm_direct.grid
        assert cm_builder.origin_x == pytest.approx(cm_direct.origin_x)
        assert cm_builder.origin_y == pytest.approx(cm_direct.origin_y)

    def test_parity_custom_weights_cell_for_cell(self):
        ftm = FakeTerrainMap(cells=_mixed_cells())
        w = CostmapWeights(base_cost=2.0, road_discount=0.25)
        cm_direct = costmap_from_terrain_map(ftm, weights=w)
        cm_builder = builder_from_terrain_map(ftm, weights=w).build()
        assert cm_builder.grid == cm_direct.grid

    def test_terrain_seeding_recorded_on_builder(self):
        ftm = FakeTerrainMap(
            cells={(5, 5): "building", (7, 7): "water", (6, 6): "road"}
        )
        b = builder_from_terrain_map(ftm)
        # Obstacle cells keyed (row, col) with the terrain kind as the tag.
        assert b._obstacle_cells[(5, 5)] == "building"
        assert b._obstacle_cells[(7, 7)] == "water"
        # Road cells keyed (row, col).
        assert (6, 6) in b._road_cells
        # Open cells are untouched by either accumulator.
        assert (2, 2) not in b._obstacle_cells
        assert (2, 2) not in b._road_cells


# ---------------------------------------------------------------------------
# Enrichment — layer GIS features on the terrain baseline before build()
# ---------------------------------------------------------------------------

class TestEnrichment:
    def test_gis_layers_on_terrain_baseline(self):
        # All-open 10x10 map spanning local [0,100]^2 at resolution 10.
        ftm = FakeTerrainMap(grid_size=10, resolution=10.0, default="open")
        builder = builder_from_terrain_map(ftm)

        fc = _fc([
            # (a) TIGER local street (S1400 -> 8 m wide, half 4 m) at y=25.
            #     Only the cell row centered on y=25 (row 2) is within 4 m.
            _tiger_line([[0, 25], [100, 25]], kind="S1400"),
            # (b) FEMA SFHA flood square over cell (7, 7) center (75, 75).
            _fema_poly(_square(75, 75, 8), sfha=True),
            # (c) NOAA Extreme weather zone over cell (2, 7) center (25, 75).
            _noaa_poly(_square(25, 75, 8), severity="Extreme"),
        ])
        summary = builder.add_gis_features(fc)
        cm = builder.build()

        assert summary == {"roads": 1, "flood": 1, "zones": 1, "ignored": 0}

        # (a) road-discount cells along the TIGER line: base * road_discount.
        w = CostmapWeights()
        road_cost = w.base_cost * w.road_discount
        col, row = cm.world_to_grid(50, 25)
        assert row == 2
        assert cm.cost_at(col, row) == pytest.approx(road_cost)
        # The whole line row is discounted, neighbouring rows are not.
        assert all(cm.cost_at(c, 2) == pytest.approx(road_cost)
                   for c in range(cm.width))
        assert cm.cost_at(*cm.world_to_grid(50, 15)) == pytest.approx(w.base_cost)

        # (b) FEMA SFHA -> lethal.
        assert cm.is_lethal(*cm.world_to_grid(75, 75))

        # (c) NOAA Extreme -> soft-cost x2 over the open baseline.
        assert cm.cost_at(*cm.world_to_grid(25, 75)) == pytest.approx(
            w.base_cost * 2.0
        )

        # An untouched open cell keeps the base cost.
        assert cm.cost_at(*cm.world_to_grid(45, 45)) == pytest.approx(w.base_cost)

    def test_terrain_obstacle_survives_road_enrichment(self):
        # A building terrain cell that a later road line also crosses stays
        # lethal — obstacle always beats road at build().
        ftm = FakeTerrainMap(
            grid_size=10, resolution=10.0, cells={(5, 2): "building"}
        )
        builder = builder_from_terrain_map(ftm)
        # Road line at y=25 crosses cell col 5 row 2 (center 55, 25).
        builder.add_gis_features(_fc([_tiger_line([[0, 25], [100, 25]], kind="S1400")]))
        cm = builder.build()
        assert cm.is_lethal(*cm.world_to_grid(55, 25))


# ---------------------------------------------------------------------------
# DEM enrichment — slope cost + lethal steep
# ---------------------------------------------------------------------------

class TestDemEnrichment:
    def _flat_open_cost(self):
        ftm = FakeTerrainMap(grid_size=10, resolution=10.0, default="open")
        cm = builder_from_terrain_map(ftm).build()
        return cm.cost_at(5, 5)  # a plain open cell, no DEM

    def test_gentle_slope_costs_more_than_flat(self):
        flat = self._flat_open_cost()
        ftm = FakeTerrainMap(grid_size=10, resolution=10.0, default="open")
        builder = builder_from_terrain_map(ftm)
        # Uniform slope 0.1 (rise/run) everywhere over the [0,100] extent.
        dem = LocalElevationGrid.from_callable(
            (-30, -30, 130, 130), 10.0, lambda x, y: 0.1 * x
        )
        builder.add_dem(dem)
        cm = builder.build()
        w = CostmapWeights()
        expected = w.base_cost + w.slope_weight * 0.1  # 1.0 + 5*0.1 = 1.5
        for c in range(cm.width):
            for r in range(cm.height):
                assert not cm.is_lethal(c, r)
                assert cm.cost_at(c, r) == pytest.approx(expected)
                assert cm.cost_at(c, r) > flat

    def test_steep_slope_goes_lethal(self):
        ftm = FakeTerrainMap(grid_size=10, resolution=10.0, default="open")
        builder = builder_from_terrain_map(ftm)
        # Slope 1.0 > max_slope 0.7 everywhere -> every cell lethal.
        dem = LocalElevationGrid.from_callable(
            (-30, -30, 130, 130), 10.0, lambda x, y: 1.0 * x
        )
        builder.add_dem(dem)
        cm = builder.build()
        assert all(
            cm.is_lethal(c, r)
            for c in range(cm.width)
            for r in range(cm.height)
        )

    def test_road_discount_and_slope_combine(self):
        # A terrain road cell on a gentle slope: base+slope, then discounted.
        ftm = FakeTerrainMap(
            grid_size=10, resolution=10.0, cells={(4, 4): "road"}
        )
        builder = builder_from_terrain_map(ftm)
        dem = LocalElevationGrid.from_callable(
            (-30, -30, 130, 130), 10.0, lambda x, y: 0.1 * x
        )
        builder.add_dem(dem)
        cm = builder.build()
        w = CostmapWeights()
        base = w.base_cost + w.slope_weight * 0.1  # 1.5
        assert cm.cost_at(4, 4) == pytest.approx(base * w.road_discount)


# ---------------------------------------------------------------------------
# Frame — origin / resolution / dimensions match the finished path
# ---------------------------------------------------------------------------

class TestFrame:
    def test_frame_matches_costmap_from_terrain_map(self):
        ftm = SwAnchoredTerrainMap(grid_size=10, resolution=10.0)
        cm_direct = costmap_from_terrain_map(ftm)
        cm_builder = builder_from_terrain_map(ftm).build()
        assert cm_builder.origin_x == pytest.approx(cm_direct.origin_x)
        assert cm_builder.origin_y == pytest.approx(cm_direct.origin_y)
        assert cm_builder.resolution == pytest.approx(cm_direct.resolution)
        assert cm_builder.width == cm_direct.width
        assert cm_builder.height == cm_direct.height
        # Sanity vs the known SW-anchored geometry: half-extent 50.
        assert cm_builder.origin_x == pytest.approx(-50.0)
        assert cm_builder.origin_y == pytest.approx(-50.0)
        assert cm_builder.grid_to_world(0, 0) == pytest.approx(
            ftm._grid_to_world(0, 0)
        )

    def test_dimensions_exact_for_non_integer_resolution(self):
        # A non-integer resolution where ceil(grid_size*res / res) could drift
        # to grid_size+1 — the builder pins width/height to grid_size.
        for res in (7.3, 3.333333, 12.5, 0.7):
            ftm = FakeTerrainMap(grid_size=13, resolution=res)
            b = builder_from_terrain_map(ftm)
            assert b.width == 13, f"width drift at res={res}"
            assert b.height == 13, f"height drift at res={res}"
            cm = b.build()
            assert cm.width == 13
            assert cm.height == 13
            # The built grid actually has grid_size rows/cols.
            assert len(cm.grid) == 13
            assert all(len(rowvals) == 13 for rowvals in cm.grid)

    def test_custom_grid_size(self):
        ftm = FakeTerrainMap(grid_size=6, resolution=5.0)
        cm = builder_from_terrain_map(ftm).build()
        assert cm.width == 6
        assert cm.height == 6
        assert cm.resolution == pytest.approx(5.0)
