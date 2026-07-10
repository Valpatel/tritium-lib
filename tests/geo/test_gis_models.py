# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geo.gis.models — GeoBBox + ElevationGrid.

Round-trip, cell geometry, and slope sanity are exercised against the packaged
demo fixture (real USGS 3DEP samples over the Dublin, CA AO) plus small
synthetic grids with analytically-known answers.  No network.
"""

import json
import math
from importlib import resources

import pytest

from tritium_lib.geo.gis.models import ElevationGrid, GeoBBox

# AO bbox used throughout the GIS lane.
AO = ("-121.912,37.704,-121.880,37.728")


def _load_fixture_grid():
    text = resources.files("tritium_lib.geo.gis.fixtures").joinpath(
        "usgs_dem_ao.json"
    ).read_text(encoding="utf-8")
    return json.loads(text)


class TestGeoBBox:
    @pytest.mark.unit
    def test_from_string_roundtrip(self):
        box = GeoBBox.from_string(AO)
        assert box.west == pytest.approx(-121.912)
        assert box.south == pytest.approx(37.704)
        assert box.east == pytest.approx(-121.880)
        assert box.north == pytest.approx(37.728)
        assert box.to_string() == "-121.912,37.704,-121.88,37.728"

    @pytest.mark.unit
    def test_from_string_wrong_count_raises(self):
        with pytest.raises(ValueError):
            GeoBBox.from_string("1,2,3")
        with pytest.raises(ValueError):
            GeoBBox.from_string("1,2,3,4,5")

    @pytest.mark.unit
    def test_from_string_non_numeric_raises(self):
        with pytest.raises(ValueError):
            GeoBBox.from_string("a,b,c,d")
        with pytest.raises(ValueError):
            GeoBBox.from_string("-121.9,37.7,east,37.7")

    @pytest.mark.unit
    def test_from_string_none_raises(self):
        with pytest.raises(ValueError):
            GeoBBox.from_string(None)

    @pytest.mark.unit
    def test_center_is_lon_lat(self):
        box = GeoBBox(west=-2.0, south=10.0, east=2.0, north=30.0)
        lon, lat = box.center()
        assert lon == pytest.approx(0.0)
        assert lat == pytest.approx(20.0)

    @pytest.mark.unit
    def test_contains(self):
        box = GeoBBox.from_string(AO)
        assert box.contains(-121.896, 37.716)  # centre
        assert not box.contains(-120.0, 37.716)  # east of box
        assert not box.contains(-121.896, 40.0)  # north of box
        # Edges are inclusive.
        assert box.contains(box.west, box.south)


class TestElevationGridRoundTrip:
    @pytest.mark.unit
    def test_fixture_roundtrip_exact(self):
        data = _load_fixture_grid()
        grid = ElevationGrid.from_dict(data)
        # from_dict tolerates the extra "fixture" marker; to_dict omits it.
        expected = {k: v for k, v in data.items() if k != "fixture"}
        assert grid.to_dict() == expected

    @pytest.mark.unit
    def test_from_to_from_stable(self):
        data = _load_fixture_grid()
        g1 = ElevationGrid.from_dict(data)
        g2 = ElevationGrid.from_dict(g1.to_dict())
        assert g1.to_dict() == g2.to_dict()

    @pytest.mark.unit
    def test_fixture_dimensions(self):
        grid = ElevationGrid.from_dict(_load_fixture_grid())
        assert grid.ncols == 16
        assert grid.nrows == 16
        assert len(grid.values) == 256
        assert grid.source == "usgs-fixture"


class TestElevationGridCells:
    def _fixture(self):
        return ElevationGrid.from_dict(_load_fixture_grid())

    @pytest.mark.unit
    def test_cell_lon_edges(self):
        g = self._fixture()
        assert g.cell_lon(0) == pytest.approx(g.west)
        assert g.cell_lon(g.ncols - 1) == pytest.approx(g.east)

    @pytest.mark.unit
    def test_cell_lat_row0_is_north(self):
        g = self._fixture()
        # Row 0 = NORTH edge — the load-bearing convention.
        assert g.cell_lat(0) == pytest.approx(g.north)
        assert g.cell_lat(g.nrows - 1) == pytest.approx(g.south)
        # Latitude decreases as the row index grows.
        assert g.cell_lat(1) < g.cell_lat(0)

    @pytest.mark.unit
    def test_value_at_matches_raw_order(self):
        g = self._fixture()
        assert g.value_at(0, 0) == g.values[0]
        assert g.value_at(g.ncols - 1, g.nrows - 1) == g.values[-1]

    @pytest.mark.unit
    def test_value_at_out_of_range(self):
        g = self._fixture()
        with pytest.raises(IndexError):
            g.value_at(g.ncols, 0)

    @pytest.mark.unit
    def test_single_column_row_guards(self):
        g = ElevationGrid(west=-1, south=2, east=3, north=4, ncols=1, nrows=1,
                          values=[42.0])
        assert g.cell_lon(0) == -1
        assert g.cell_lat(0) == 4

    @pytest.mark.unit
    def test_min_max(self):
        g = self._fixture()
        mn, mx = g.min_max()
        assert 101.0 <= mn <= 102.0
        assert 190.0 <= mx <= 192.0

    @pytest.mark.unit
    def test_min_max_all_none(self):
        g = ElevationGrid(west=0, south=0, east=1, north=1, ncols=2, nrows=2,
                          values=[None, None, None, None])
        assert g.min_max() == (None, None)


class TestElevationGridSlope:
    @pytest.mark.unit
    def test_fixture_slopes_plausible(self):
        g = ElevationGrid.from_dict(_load_fixture_grid())
        slopes = g.slope_deg()
        assert len(slopes) == 256
        present = [s for s in slopes if s is not None]
        # Real terrain (elev 101-191 m over ~2.8 km) — all fully-sampled cells
        # have a defined slope, every one gentle.
        assert len(present) == 256
        for s in present:
            assert 0.0 <= s <= 45.0

    @pytest.mark.unit
    def test_flat_grid_zero_slope(self):
        g = ElevationGrid(west=0, south=0, east=0.01, north=0.01, ncols=4,
                          nrows=4, values=[100.0] * 16)
        for s in g.slope_deg():
            assert s == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.unit
    def test_east_ramp_known_slope(self):
        # Elevation rises 1 m per column, flat north-south.  Slope should be a
        # constant atan(dz / dx_metres).
        ncols = nrows = 5
        vals = []
        for _iy in range(nrows):
            for ix in range(ncols):
                vals.append(float(ix))  # +1 m per column east
        # 0.01 deg lon span at ~37.7 deg lat.
        from tritium_lib.geo import METERS_PER_DEG_LAT
        west, east = -121.9, -121.89
        lat = 37.716
        g = ElevationGrid(west=west, south=37.71, east=east, north=37.72,
                          ncols=ncols, nrows=nrows, values=vals)
        dlon = (east - west) / (ncols - 1)
        dx_m = dlon * METERS_PER_DEG_LAT * math.cos(math.radians(g.cell_lat(2)))
        expected = math.degrees(math.atan(1.0 / dx_m))
        slopes = g.slope_deg()
        # Interior cell of the middle row.
        interior = slopes[2 * ncols + 2]
        assert interior == pytest.approx(expected, rel=1e-3)

    @pytest.mark.unit
    def test_nodata_neighbor_gives_none(self):
        # Centre cell has a NoData east neighbour -> its slope is undefined.
        vals = [10.0, 11.0, 12.0,
                13.0, 14.0, None,
                16.0, 17.0, 18.0]
        g = ElevationGrid(west=0, south=0, east=0.02, north=0.02, ncols=3,
                          nrows=3, values=vals)
        slopes = g.slope_deg()
        assert slopes[4] is None  # centre, touches the None east neighbour
        assert slopes[5] is None  # the NoData cell itself

    @pytest.mark.unit
    def test_degenerate_grid_all_none_slopes(self):
        g = ElevationGrid(west=0, south=0, east=1, north=1, ncols=1, nrows=1,
                          values=[5.0])
        assert g.slope_deg() == [None]
