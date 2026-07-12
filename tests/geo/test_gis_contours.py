# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geo.gis.contours — marching-squares iso-contours.

Pure computation, no IO / no network.  Synthetic grids exercise the edge cases
(flat, saddle-free gradients, a central peak, a NoData hole, out-of-range
levels) and the packaged 16x16 real-terrain DEM fixture exercises the whole
thing end-to-end.
"""

import json
import math
from collections import defaultdict
from importlib import resources

import pytest

from tritium_lib.geo.gis import auto_levels, contour_lines
from tritium_lib.geo.gis.models import ElevationGrid

AO_BBOX = (-121.912, 37.704, -121.880, 37.728)


def _grid(ncols, nrows, fn, west=0.0, south=0.0, east=None, north=None):
    """Build an ElevationGrid whose value at (ix, iy) is ``fn(ix, iy)``."""
    east = float(ncols - 1) if east is None else east
    north = float(nrows - 1) if north is None else north
    values = [fn(ix, iy) for iy in range(nrows) for ix in range(ncols)]
    return ElevationGrid(
        west=west, south=south, east=east, north=north,
        ncols=ncols, nrows=nrows, values=values,
    )


# ---------------------------------------------------------------------------
# auto_levels
# ---------------------------------------------------------------------------
class TestAutoLevels:
    @pytest.mark.unit
    def test_levels_strictly_inside_min_max(self):
        g = _grid(5, 5, lambda ix, iy: float(ix + iy))  # min 0, max 8
        levels = auto_levels(g, 8)
        assert len(levels) == 8
        mn, mx = g.min_max()
        assert all(mn < lv < mx for lv in levels)
        # Monotonic increasing.
        assert levels == sorted(levels)

    @pytest.mark.unit
    def test_flat_grid_has_no_levels(self):
        g = _grid(4, 4, lambda ix, iy: 5.0)
        assert auto_levels(g) == []

    @pytest.mark.unit
    def test_single_distinct_value_has_no_levels(self):
        g = ElevationGrid(west=0, south=0, east=1, north=1, ncols=2, nrows=2,
                          values=[3.0, 3.0, None, 3.0])
        assert auto_levels(g) == []

    @pytest.mark.unit
    def test_n_controls_count(self):
        g = _grid(3, 3, lambda ix, iy: float(ix))
        assert len(auto_levels(g, 3)) == 3
        assert len(auto_levels(g, 12)) == 12
        assert auto_levels(g, 0) == []


# ---------------------------------------------------------------------------
# Degenerate / boundary behaviour
# ---------------------------------------------------------------------------
class TestContourBasics:
    @pytest.mark.unit
    def test_flat_grid_yields_no_features(self):
        g = _grid(5, 5, lambda ix, iy: 42.0)
        fc = contour_lines(g, [42.0, 10.0, 100.0])
        assert fc == {"type": "FeatureCollection", "features": []}

    @pytest.mark.unit
    def test_levels_outside_range_yield_nothing(self):
        g = _grid(5, 5, lambda ix, iy: float(ix))  # 0..4
        assert contour_lines(g, [10.0])["features"] == []
        assert contour_lines(g, [-5.0])["features"] == []

    @pytest.mark.unit
    def test_empty_levels_and_tiny_grid(self):
        g = _grid(5, 5, lambda ix, iy: float(ix))
        assert contour_lines(g, [])["features"] == []
        tiny = ElevationGrid(west=0, south=0, east=0, north=0, ncols=1, nrows=1,
                            values=[1.0])
        assert contour_lines(tiny, [1.0])["features"] == []


# ---------------------------------------------------------------------------
# Gradients — straight contours at known positions
# ---------------------------------------------------------------------------
class TestGradients:
    @pytest.mark.unit
    def test_west_east_gradient_is_vertical_line(self):
        # value increases eastward; a mid level -> one vertical line at that lon.
        g = _grid(5, 5, lambda ix, iy: float(ix))  # bbox 0..4 in both axes
        fc = contour_lines(g, [2.0])
        assert len(fc["features"]) == 1
        coords = fc["features"][0]["geometry"]["coordinates"]
        assert len(coords) == 5  # one point per row
        assert all(lon == pytest.approx(2.0) for lon, _lat in coords)
        # Spans full north-south extent.
        lats = sorted(lat for _lon, lat in coords)
        assert lats[0] == pytest.approx(0.0)
        assert lats[-1] == pytest.approx(4.0)

    @pytest.mark.unit
    def test_south_north_gradient_is_horizontal_line(self):
        # value increases with row index (i.e. southward, since row 0 = north).
        g = _grid(5, 5, lambda ix, iy: float(iy))
        fc = contour_lines(g, [2.0])
        assert len(fc["features"]) == 1
        coords = fc["features"][0]["geometry"]["coordinates"]
        # Constant latitude at the iy=2 row.  cell_lat(2) on a 0..4 north span.
        assert all(lat == pytest.approx(g.cell_lat(2)) for _lon, lat in coords)

    @pytest.mark.unit
    def test_properties_contract(self):
        g = _grid(5, 5, lambda ix, iy: float(ix))
        props = contour_lines(g, [2.0])["features"][0]["properties"]
        assert props == {
            "source": "usgs", "kind": "contour",
            "elevation_m": 2.0, "level_index": 0,
        }

    @pytest.mark.unit
    def test_level_index_tracks_input_order(self):
        g = _grid(6, 6, lambda ix, iy: float(ix))
        fc = contour_lines(g, [1.0, 3.0])
        idx = {f["properties"]["level_index"] for f in fc["features"]}
        assert idx == {0, 1}
        for f in fc["features"]:
            expect = 1.0 if f["properties"]["level_index"] == 0 else 3.0
            assert f["properties"]["elevation_m"] == expect


# ---------------------------------------------------------------------------
# Central peak — closed rings
# ---------------------------------------------------------------------------
class TestCentralPeak:
    @pytest.mark.unit
    def test_peak_gives_feature_per_level_and_a_closed_ring(self):
        n = 11
        cx = cy = (n - 1) / 2.0
        g = _grid(n, n, lambda ix, iy: max(0.0, 20.0 - 3.0 * math.hypot(ix - cx, iy - cy)))
        levels = auto_levels(g, 6)
        fc = contour_lines(g, levels)

        per_level = defaultdict(lambda: [0, 0])  # [features, closed rings]
        for f in fc["features"]:
            i = f["properties"]["level_index"]
            co = f["geometry"]["coordinates"]
            per_level[i][0] += 1
            if co[0] == co[-1]:
                per_level[i][1] += 1

        # Every interior level produces at least one feature.
        assert all(per_level[i][0] >= 1 for i in range(len(levels)))
        # And the inner (fully enclosed) levels close into rings.
        assert sum(pl[1] for pl in per_level.values()) >= 1


# ---------------------------------------------------------------------------
# NoData holes
# ---------------------------------------------------------------------------
class TestNoData:
    @pytest.mark.unit
    def test_shared_center_hole_skips_all_cells(self):
        # In a 3x3 grid the centre node is a corner of ALL four cells, so a
        # NoData there means every cell touches the hole -> no features.
        vals = [float(ix + iy) for iy in range(3) for ix in range(3)]
        vals[1 * 3 + 1] = None  # centre node
        g = ElevationGrid(west=0, south=0, east=2, north=2, ncols=3, nrows=3,
                         values=vals)
        assert contour_lines(g, [2.0])["features"] == []

    @pytest.mark.unit
    def test_hole_is_not_crossed(self):
        # 5x5 diagonal gradient with a NoData at the centre node (2,2).  The
        # four cells touching it must be skipped: no contour point may fall
        # strictly inside that 2x2-cell hole region.
        vals = [float(ix + iy) for iy in range(5) for ix in range(5)]
        vals[2 * 5 + 2] = None
        g = ElevationGrid(west=0, south=0, east=4, north=4, ncols=5, nrows=5,
                         values=vals)
        fc = contour_lines(g, [4.0])
        assert fc["features"]  # contours still route around the hole
        # Hole region spans cells (1,1)..(2,2): lon in (cx-1, cx+1), lat around.
        lon_lo, lon_hi = g.cell_lon(1), g.cell_lon(3)
        lat_lo, lat_hi = g.cell_lat(3), g.cell_lat(1)  # row grows south
        for f in fc["features"]:
            for lon, lat in f["geometry"]["coordinates"]:
                strictly_inside = (
                    lon_lo < lon < lon_hi and lat_lo < lat < lat_hi
                )
                assert not strictly_inside, (lon, lat)


# ---------------------------------------------------------------------------
# Real packaged DEM fixture (16x16 Dublin, CA terrain, 101-191 m)
# ---------------------------------------------------------------------------
class TestRealFixture:
    @staticmethod
    def _load_grid():
        raw = json.loads(
            resources.files("tritium_lib.geo.gis.fixtures")
            .joinpath("usgs_dem_ao.json")
            .read_text(encoding="utf-8")
        )
        return ElevationGrid.from_dict(raw)

    @pytest.mark.unit
    def test_fixture_produces_contours(self):
        g = self._load_grid()
        assert g.ncols == 16 and g.nrows == 16
        levels = auto_levels(g, 8)
        assert len(levels) == 8
        fc = contour_lines(g, levels)

        producing = {
            f["properties"]["level_index"] for f in fc["features"]
        }
        assert len(producing) >= 4  # at least 4 levels each yield >=1 feature

    @pytest.mark.unit
    def test_fixture_coords_inside_ao_and_props_valid(self):
        g = self._load_grid()
        levels = auto_levels(g, 8)
        fc = contour_lines(g, levels)
        w, s, e, n = AO_BBOX
        for f in fc["features"]:
            p = f["properties"]
            assert p["source"] == "usgs"
            assert p["kind"] == "contour"
            assert 0 <= p["level_index"] < len(levels)
            assert p["elevation_m"] == round(p["elevation_m"], 1)
            for lon, lat in f["geometry"]["coordinates"]:
                assert w - 1e-9 <= lon <= e + 1e-9
                assert s - 1e-9 <= lat <= n + 1e-9
