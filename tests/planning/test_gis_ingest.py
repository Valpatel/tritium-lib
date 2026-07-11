# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for GIS-lane reconciliation in tritium_lib.planning.

Covers:
    - :func:`local_grid_from_gis` — the north-first WGS-84 wire DEM adapter
      (row flip, bilinear values, NoData, resolution default, dict/object
      inputs, degenerate errors, ``to_local`` respected).
    - :meth:`CostmapBuilder.add_gis_features` — source/kind/sfha routing.
    - :meth:`CostmapBuilder.add_cost_zones` — soft-cost multiplier stacking.
    - the A* smoothing window cap on long paths.

Fixtures here are written to mimic the DOCUMENTED GIS shapes — they do not
import the GIS lane's own fixture files (that lane is not present in this
worktree).
"""

import math
import types

import pytest

from tritium_lib import geo
from tritium_lib.planning.costmap import (
    MTFCC_WIDTHS_M,
    CostmapBuilder,
    CostmapWeights,
)
from tritium_lib.planning.layers import (
    LocalElevationGrid,
    local_grid_from_gis,
    wgs84_to_local,
)
from tritium_lib.planning.astar import _SMOOTH_WINDOW, plan_route
from tritium_lib.planning.astar import _supercover_cells


# ---------------------------------------------------------------------------
# A fake linear projector so the geo singleton is not needed.
#
# 1 degree -> 100 meters on each axis; equirectangular and axis-separable,
# exactly the affine shape local_grid_from_gis documents.
# ---------------------------------------------------------------------------

def _fake_to_local(lng, lat):
    return (lng * 100.0, lat * 100.0)


def _plane_grid_dict():
    """A 4x4 north-first WGS-84 wire DEM encoding the plane E = 0.1x + 0.2y.

    bbox: west=0 east=3 south=0 north=3 (degrees) -> local [0,300]x[0,300] m
    under ``_fake_to_local``.  Source ``values`` are flat row-major with
    ``row 0`` = NORTH.  Because elevation increases NORTHWARD, the north rows
    hold the largest values — the row flip is provable by checking that after
    adaptation the SOUTH output row is the smallest.
    """
    # Source node (src_row, src_col): local x = col*100, local y = 300 - row*100.
    # E = 0.1*x + 0.2*y  =>  10*col + 60 - 20*row.
    values = []
    for src_row in range(4):
        for src_col in range(4):
            values.append(10.0 * src_col + 60.0 - 20.0 * src_row)
    return {
        "west": 0.0, "south": 0.0, "east": 3.0, "north": 3.0,
        "ncols": 4, "nrows": 4,
        "values": values,
        "source": "usgs", "resolution_m": None, "fixture": "plane",
    }


# ---------------------------------------------------------------------------
# local_grid_from_gis — the row-flip + bilinear adapter
# ---------------------------------------------------------------------------

class TestLocalGridFromGis:
    def test_row_flip_north_is_higher(self):
        """THE critical test: source row 0 = north (highest) must land on the
        NORTH (max-y) edge of the local grid, not the south edge."""
        grid = local_grid_from_gis(
            _plane_grid_dict(), to_local=_fake_to_local, resolution=100.0
        )
        assert isinstance(grid, LocalElevationGrid)
        # North edge (large y) is higher than the south edge (small y).
        north = grid.elevation_at(150.0, 300.0)
        south = grid.elevation_at(150.0, 0.0)
        assert north is not None and south is not None
        assert north > south
        # Plane E = 0.1x + 0.2y: south edge y=0 -> 0.1*150 = 15; north y=300
        # -> 15 + 60 = 75.
        assert south == pytest.approx(15.0, abs=1e-6)
        assert north == pytest.approx(75.0, abs=1e-6)

    def test_slope_points_uphill_northward(self):
        grid = local_grid_from_gis(
            _plane_grid_dict(), to_local=_fake_to_local, resolution=100.0
        )
        # Elevation rises 0.2/m in +y and 0.1/m in +x -> |grad| = sqrt(0.05).
        assert grid.slope_at(150.0, 150.0) == pytest.approx(
            math.hypot(0.1, 0.2), abs=1e-6
        )
        # Direction proof: a step north is higher than a step south.
        assert grid.elevation_at(150.0, 200.0) > grid.elevation_at(150.0, 100.0)

    def test_bilinear_values_exact_on_plane(self):
        # Output resolution 150 places node x=150 between source cols
        # (ix_frac=1.5) so the value is a genuine bilinear blend, not a copy.
        grid = local_grid_from_gis(
            _plane_grid_dict(), to_local=_fake_to_local, resolution=150.0
        )
        # Bilinear of a plane is exact everywhere it samples.
        assert grid.elevation_at(150.0, 150.0) == pytest.approx(45.0, abs=1e-6)
        assert grid.elevation_at(100.0, 200.0) == pytest.approx(50.0, abs=1e-6)
        assert grid.elevation_at(0.0, 0.0) == pytest.approx(0.0, abs=1e-6)
        assert grid.elevation_at(300.0, 300.0) == pytest.approx(90.0, abs=1e-6)

    def test_object_input_matches_dict_input(self):
        d = _plane_grid_dict()
        obj = types.SimpleNamespace(**d)
        g_dict = local_grid_from_gis(d, to_local=_fake_to_local, resolution=100.0)
        g_obj = local_grid_from_gis(obj, to_local=_fake_to_local, resolution=100.0)
        assert g_dict.data == g_obj.data
        assert g_obj.origin_x == pytest.approx(0.0)
        assert g_obj.origin_y == pytest.approx(0.0)

    def test_resolution_default_is_mean_cell_spacing(self):
        # 4x4 over [0,300]x[0,300] -> spacing 300/3 = 100 on each axis -> 100.
        grid = local_grid_from_gis(_plane_grid_dict(), to_local=_fake_to_local)
        assert grid.resolution == pytest.approx(100.0)

    def test_nodata_nearest_of_four(self):
        # Flat grid of 5.0 except one None in the interior.  Adaptation at a
        # node whose 4 source neighbours include the None must fall back to a
        # neighbour value (5.0), never crash or emit None.
        d = {
            "west": 0.0, "south": 0.0, "east": 3.0, "north": 3.0,
            "ncols": 4, "nrows": 4,
            "values": [5.0] * 16, "source": "usgs",
        }
        d["values"][1 * 4 + 1] = None  # source (row1, col1)
        grid = local_grid_from_gis(d, to_local=_fake_to_local, resolution=50.0)
        # No None leaked into the output.
        for row in grid.data:
            for v in row:
                assert v is not None
                assert v == pytest.approx(5.0)

    def test_nodata_all_four_none_uses_fill(self):
        # A 2x2 block of None surrounded by 7.0; a query landing entirely in
        # the hole falls back to nodata_fill.
        vals = [7.0] * 16
        for rc in ((1, 1), (1, 2), (2, 1), (2, 2)):
            vals[rc[0] * 4 + rc[1]] = None
        d = {
            "west": 0.0, "south": 0.0, "east": 3.0, "north": 3.0,
            "ncols": 4, "nrows": 4, "values": vals, "source": "usgs",
        }
        grid = local_grid_from_gis(
            d, to_local=_fake_to_local, resolution=100.0, nodata_fill=-1.0
        )
        # Output node (col=1, row=2) -> world (100, 200) maps to source index
        # (ix_frac=1, iy_frac=1): its four neighbours are exactly the None
        # block (source rows/cols 1-2) -> all-None -> fill.
        assert grid.data[2][1] == pytest.approx(-1.0)
        assert grid.elevation_at(100.0, 200.0) == pytest.approx(-1.0)

    def test_nodata_fill_defaults_to_mean(self):
        vals = [7.0] * 16
        for rc in ((1, 1), (1, 2), (2, 1), (2, 2)):
            vals[rc[0] * 4 + rc[1]] = None
        d = {
            "west": 0.0, "south": 0.0, "east": 3.0, "north": 3.0,
            "ncols": 4, "nrows": 4, "values": vals, "source": "usgs",
        }
        grid = local_grid_from_gis(d, to_local=_fake_to_local, resolution=100.0)
        # All non-None values are 7.0 -> mean 7.0 fill.
        assert grid.elevation_at(150.0, 150.0) == pytest.approx(7.0)

    def test_degenerate_ncols_raises(self):
        d = {
            "west": 0.0, "south": 0.0, "east": 3.0, "north": 3.0,
            "ncols": 1, "nrows": 4, "values": [1.0, 2.0, 3.0, 4.0],
        }
        with pytest.raises(ValueError):
            local_grid_from_gis(d, to_local=_fake_to_local)

    def test_degenerate_all_none_raises(self):
        d = {
            "west": 0.0, "south": 0.0, "east": 3.0, "north": 3.0,
            "ncols": 2, "nrows": 2, "values": [None, None, None, None],
        }
        with pytest.raises(ValueError):
            local_grid_from_gis(d, to_local=_fake_to_local)

    def test_values_length_mismatch_raises(self):
        d = {
            "west": 0.0, "south": 0.0, "east": 3.0, "north": 3.0,
            "ncols": 3, "nrows": 3, "values": [1.0, 2.0, 3.0],
        }
        with pytest.raises(ValueError):
            local_grid_from_gis(d, to_local=_fake_to_local)

    def test_missing_fields_raises(self):
        with pytest.raises(ValueError):
            local_grid_from_gis({"ncols": 4, "nrows": 4}, to_local=_fake_to_local)

    def test_to_local_respected_wgs84_singleton(self):
        # Uses the real geo singleton (like tests/geo does): init + reset.
        geo.reset()
        try:
            geo.init_reference(lat=0.0, lng=0.0)
            d = _plane_grid_dict()
            grid = local_grid_from_gis(d, to_local=wgs84_to_local())
            # north edge higher than south edge is projector-independent.
            assert grid.data[-1][0] > grid.data[0][0]
            # Local extent spans a few hundred meters (0.03 deg ~ few km? no —
            # bbox is 0..3 deg -> ~333 km; just assert positive, sane extent).
            assert grid.origin_x == pytest.approx(0.0, abs=1.0)
            assert grid.resolution > 0.0
        finally:
            geo.reset()

    def test_default_to_local_needs_geo_reference(self):
        geo.reset()
        try:
            with pytest.raises(RuntimeError):
                local_grid_from_gis(_plane_grid_dict())
        finally:
            geo.reset()


# ---------------------------------------------------------------------------
# Fixtures for CostmapBuilder ingestion
# ---------------------------------------------------------------------------

def _fc(features):
    return {"type": "FeatureCollection", "features": features}


def _tiger_line(coords, kind, name="", width_m=None):
    props = {"source": "tiger", "kind": kind, "name": name}
    if width_m is not None:
        props["width_m"] = width_m
    return {
        "type": "Feature", "properties": props,
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def _fema_poly(ring, sfha, kind="AE", subtype=""):
    return {
        "type": "Feature",
        "properties": {"source": "fema", "kind": kind, "sfha": sfha,
                       "subtype": subtype},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _noaa_poly(ring, severity, event="Red Flag Warning"):
    return {
        "type": "Feature",
        "properties": {"source": "noaa", "kind": event, "severity": severity,
                       "headline": "test", "expires": "2026-07-11T00:00:00Z"},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _square(cx, cy, half):
    return [
        [cx - half, cy - half], [cx + half, cy - half],
        [cx + half, cy + half], [cx - half, cy + half],
        [cx - half, cy - half],
    ]


# ---------------------------------------------------------------------------
# add_gis_features — source/kind/sfha routing
# ---------------------------------------------------------------------------

class TestAddGisFeatures:
    def test_fema_sfha_true_lethal_false_not(self):
        # Two flood polygons: one SFHA (lethal), one zone X (traversable).
        fc = _fc([
            _fema_poly(_square(25, 25, 8), sfha=True, kind="AE"),
            _fema_poly(_square(75, 75, 8), sfha=False, kind="X"),
        ])
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        summary = b.add_gis_features(fc)
        cm = b.build()
        # SFHA cell lethal.
        assert cm.is_lethal(*cm.world_to_grid(25, 25))
        assert b._obstacle_cells[(2, 2)] == "flood"
        # Zone-X cell traversable (base cost, not lethal).
        assert not cm.is_lethal(*cm.world_to_grid(75, 75))
        assert cm.cost_at(*cm.world_to_grid(75, 75)) == pytest.approx(1.0)
        assert summary["flood"] == 1
        assert summary["ignored"] == 1  # the sfha=False feature

    def test_tiger_width_from_mtfcc_table(self):
        # A wide primary (S1100 -> 18 m) vs a narrow alley (S1730 -> 4 m):
        # the primary corridor covers strictly more cells.
        primary = _fc([_tiger_line([[0, 25], [50, 25]], kind="S1100")])
        alley = _fc([_tiger_line([[0, 25], [50, 25]], kind="S1730")])

        b1 = CostmapBuilder((0, 0, 50, 50), resolution=5.0)
        b1.add_gis_features(primary)
        cm1 = b1.build()
        b2 = CostmapBuilder((0, 0, 50, 50), resolution=5.0)
        b2.add_gis_features(alley)
        cm2 = b2.build()

        n1 = sum(
            1 for c in range(cm1.width) for r in range(cm1.height)
            if cm1.cost_at(c, r) < 1.0
        )
        n2 = sum(
            1 for c in range(cm2.width) for r in range(cm2.height)
            if cm2.cost_at(c, r) < 1.0
        )
        assert n1 > n2
        # Sanity: the table values are what we expect.
        assert MTFCC_WIDTHS_M["S1100"] == 18.0
        assert MTFCC_WIDTHS_M["S1730"] == 4.0

    def test_tiger_width_m_override_wins(self):
        # width_m on the feature overrides the MTFCC table entirely.
        fc = _fc([_tiger_line([[0, 25], [50, 25]], kind="S1730", width_m=20.0)])
        b = CostmapBuilder((0, 0, 50, 50), resolution=5.0)
        b.add_gis_features(fc)
        cm = b.build()
        rows = {
            r for c in range(cm.width) for r in range(cm.height)
            if cm.cost_at(c, r) < 1.0
        }
        # 20 m wide -> half 10 -> rows 3..6 around y=25.
        assert rows == {3, 4, 5, 6}

    def test_tiger_unknown_mtfcc_uses_default_width(self):
        weights = CostmapWeights(road_width_m=8.0)
        fc = _fc([_tiger_line([[0, 25], [50, 25]], kind="S9999")])
        b = CostmapBuilder((0, 0, 50, 50), resolution=5.0, weights=weights)
        b.add_gis_features(fc)
        cm = b.build()
        rows = {
            r for c in range(cm.width) for r in range(cm.height)
            if cm.cost_at(c, r) < 1.0
        }
        # Default 8 m -> half 4 -> rows 4 and 5.
        assert rows == {4, 5}

    def test_noaa_severe_multiplies_minor_ignored(self):
        fc = _fc([
            _noaa_poly(_square(25, 25, 8), severity="Severe"),
            _noaa_poly(_square(75, 75, 8), severity="Minor"),
        ])
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        summary = b.add_gis_features(fc)
        cm = b.build()
        # Severe cell -> base 1.0 * 3.0 = 3.0 (deterrent-strength detour).
        assert cm.cost_at(*cm.world_to_grid(25, 25)) == pytest.approx(3.0)
        # Minor cell untouched -> base 1.0.
        assert cm.cost_at(*cm.world_to_grid(75, 75)) == pytest.approx(1.0)
        assert summary["zones"] == 1
        assert summary["ignored"] == 1

    def test_noaa_severity_multiplier_mapping(self):
        # The storm-lane contract: Severe -> x3.0, Extreme -> x6.0, and any
        # sub-warning severity (Moderate) is traversable and IGNORED (no zone).
        b = CostmapBuilder((0, 0, 150, 150), resolution=10.0)
        summary = b.add_gis_features(_fc([
            _noaa_poly(_square(25, 25, 8), severity="Severe"),
            _noaa_poly(_square(75, 75, 8), severity="Extreme"),
            _noaa_poly(_square(125, 125, 8), severity="Moderate"),
        ]))
        cm = b.build()
        assert cm.cost_at(*cm.world_to_grid(25, 25)) == pytest.approx(3.0)
        assert cm.cost_at(*cm.world_to_grid(75, 75)) == pytest.approx(6.0)
        # Moderate -> untouched baseline, not a zone.
        assert cm.cost_at(*cm.world_to_grid(125, 125)) == pytest.approx(1.0)
        assert summary["zones"] == 2
        assert summary["ignored"] == 1
        # Extreme must deter strictly harder than Severe.
        assert cm.cost_at(*cm.world_to_grid(75, 75)) > cm.cost_at(
            *cm.world_to_grid(25, 25)
        )

    def test_summary_counts_and_unknown_ignored(self):
        fc = _fc([
            _tiger_line([[0, 5], [50, 5]], kind="S1400"),
            _fema_poly(_square(25, 25, 8), sfha=True),
            _noaa_poly(_square(75, 75, 8), severity="Extreme"),
            {"type": "Feature", "properties": {"source": "mystery"},
             "geometry": {"type": "Polygon", "coordinates": [_square(90, 90, 3)]}},
        ])
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        summary = b.add_gis_features(fc)
        assert summary == {"roads": 1, "flood": 1, "zones": 1, "ignored": 1}

    def test_zone_over_road_ordering(self):
        # A road cell that also falls inside a severe weather zone: the final
        # cost must be base * road_discount * zone_multiplier.
        weights = CostmapWeights(base_cost=1.0, road_discount=0.5)
        fc = _fc([
            _tiger_line([[0, 25], [50, 25]], kind="S1100"),  # wide road at y=25
            _noaa_poly(_square(25, 25, 20), severity="Severe"),  # zone over it
        ])
        b = CostmapBuilder((0, 0, 50, 50), resolution=5.0, weights=weights)
        b.add_gis_features(fc)
        cm = b.build()
        col, row = cm.world_to_grid(25, 25)
        # Road (0.5) inside a severe zone (x3) -> 1.0 * 0.5 * 3.0 = 1.5.
        assert cm.cost_at(col, row) == pytest.approx(1.5)
        # And it's cheaper than an off-road severe-zone cell (1.0 * 3.0 = 3.0).
        assert cm.cost_at(col, row) < cm.cost_at(*cm.world_to_grid(5, 5))


# ---------------------------------------------------------------------------
# add_cost_zones — soft-cost multiplier stacking
# ---------------------------------------------------------------------------

class TestAddCostZones:
    def test_multiplier_raises_cost(self):
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        b.add_cost_zones(_fc([{
            "type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [_square(50, 50, 8)]},
        }]), multiplier=3.0)
        cm = b.build()
        assert cm.cost_at(*cm.world_to_grid(50, 50)) == pytest.approx(3.0)

    def test_max_not_compound_stacking(self):
        # Two overlapping zones (x2 and x3) over the same cell -> keep MAX (3),
        # never the product (6).
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        ring = _square(50, 50, 8)
        b.add_cost_zones(_fc([{
            "type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }]), multiplier=2.0)
        b.add_cost_zones(_fc([{
            "type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }]), multiplier=3.0)
        cm = b.build()
        assert cm.cost_at(*cm.world_to_grid(50, 50)) == pytest.approx(3.0)

    def test_zones_never_lethal(self):
        # Even a huge multiplier keeps the cell finite/traversable.
        b = CostmapBuilder((0, 0, 100, 100), resolution=10.0)
        b.add_cost_zones(_fc([{
            "type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [_square(50, 50, 40)]},
        }]), multiplier=1e6)
        cm = b.build()
        for c in range(cm.width):
            for r in range(cm.height):
                assert not cm.is_lethal(c, r)


# ---------------------------------------------------------------------------
# A* smoothing window cap
# ---------------------------------------------------------------------------

class TestSmoothingWindow:
    def _path_touches_lethal(self, cm, path):
        for a, b in zip(path, path[1:]):
            for c, r in _supercover_cells(cm, a, b, include_corner_cells=True):
                if cm.is_lethal(c, r):
                    return True
        return False

    def test_long_path_still_valid_and_bounded(self):
        # A big open map yields a very long unsmoothed grid path (>100 wp).
        # Smoothing must still return a valid non-lethal path with the window
        # cap in place.
        cm = CostmapBuilder((0, 0, 1500, 60), resolution=10.0).build()
        raw = plan_route(cm, (5, 30), (1495, 30), smooth=False)
        assert raw.success
        assert len(raw.path) > 100  # long enough to exercise the window
        sm = plan_route(cm, (5, 30), (1495, 30), smooth=True)
        assert sm.success
        assert not self._path_touches_lethal(cm, sm.path)
        # Endpoints preserved exactly.
        assert sm.path[0] == (5, 30)
        assert sm.path[-1] == (1495, 30)
        # Smoothing never lengthens.
        assert len(sm.path) <= len(raw.path)

    def test_window_constant_present(self):
        # Guard against accidental removal of the perf cap.
        assert _SMOOTH_WINDOW == 40
