# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.planning.layers — DEM grids + GeoJSON iteration."""

import math

import pytest

from tritium_lib import geo
from tritium_lib.planning.layers import (
    LINE_TYPES,
    POLYGON_TYPES,
    LocalElevationGrid,
    iter_features,
    iter_lines,
    iter_polygons,
    wgs84_to_local,
)


# ---------------------------------------------------------------------------
# LocalElevationGrid
# ---------------------------------------------------------------------------

class TestLocalElevationGrid:
    def test_from_callable_dimensions(self):
        dem = LocalElevationGrid.from_callable((0, 0, 100, 100), 10.0, lambda x, y: 0.0)
        # 0..100 inclusive at 10m spacing -> 11 nodes per axis.
        assert dem.width == 11
        assert dem.height == 11
        assert dem.origin_x == 0
        assert dem.origin_y == 0
        assert dem.resolution == 10.0

    def test_elevation_at_nodes_exact(self):
        dem = LocalElevationGrid.from_callable((0, 0, 100, 100), 10.0, lambda x, y: 0.2 * x)
        # At a node the interpolation is exact.
        assert dem.elevation_at(0, 0) == pytest.approx(0.0)
        assert dem.elevation_at(10, 50) == pytest.approx(2.0)
        assert dem.elevation_at(100, 0) == pytest.approx(20.0)

    def test_elevation_at_bilinear_interior(self):
        # Bilinear of a plane returns the exact plane value anywhere.
        dem = LocalElevationGrid.from_callable(
            (0, 0, 40, 40), 10.0, lambda x, y: 3.0 * x + 5.0 * y
        )
        assert dem.elevation_at(12.5, 7.5) == pytest.approx(3.0 * 12.5 + 5.0 * 7.5)
        assert dem.elevation_at(23.3, 11.1) == pytest.approx(3.0 * 23.3 + 5.0 * 11.1)

    def test_elevation_out_of_bounds_none(self):
        dem = LocalElevationGrid.from_callable((0, 0, 40, 40), 10.0, lambda x, y: 1.0)
        assert dem.elevation_at(-5, 10) is None
        assert dem.elevation_at(10, -5) is None
        assert dem.elevation_at(45, 10) is None
        assert dem.elevation_at(10, 45) is None

    def test_slope_on_ramp_known_gradient(self):
        # Elevation rises 0.3 per meter in x, flat in y -> slope magnitude 0.3.
        dem = LocalElevationGrid.from_callable((0, 0, 100, 100), 5.0, lambda x, y: 0.3 * x)
        assert dem.slope_at(50, 50) == pytest.approx(0.3)
        assert dem.slope_at(25, 75) == pytest.approx(0.3)

    def test_slope_diagonal_ramp(self):
        # Gradient (0.3, 0.4) -> magnitude 0.5.
        dem = LocalElevationGrid.from_callable(
            (0, 0, 100, 100), 5.0, lambda x, y: 0.3 * x + 0.4 * y
        )
        assert dem.slope_at(50, 50) == pytest.approx(0.5)

    def test_slope_flat_is_zero(self):
        dem = LocalElevationGrid.from_callable((0, 0, 100, 100), 5.0, lambda x, y: 42.0)
        assert dem.slope_at(50, 50) == pytest.approx(0.0)

    def test_slope_out_of_bounds_zero(self):
        dem = LocalElevationGrid.from_callable((0, 0, 40, 40), 5.0, lambda x, y: 0.3 * x)
        # A sample step off the grid -> slope defaults to 0.0.
        assert dem.slope_at(-100, 20) == 0.0

    def test_empty_grid(self):
        dem = LocalElevationGrid(origin_x=0, origin_y=0, resolution=5.0, data=[])
        assert dem.width == 0
        assert dem.height == 0
        assert dem.elevation_at(0, 0) is None


# ---------------------------------------------------------------------------
# GeoJSON iteration — local meters (no projection)
# ---------------------------------------------------------------------------

def _polygon_fc(ring):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": "p"},
             "geometry": {"type": "Polygon", "coordinates": [ring]}},
        ],
    }


class TestIterFeatures:
    def test_polygon_local(self):
        ring = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
        feats = list(iter_features(_polygon_fc(ring)))
        assert len(feats) == 1
        gtype, seqs, props = feats[0]
        assert gtype == "Polygon"
        assert gtype in POLYGON_TYPES
        assert len(seqs) == 1
        assert seqs[0][0] == (0.0, 0.0)
        assert seqs[0][1] == (10.0, 0.0)
        assert props["name"] == "p"

    def test_polygon_holes_ignored(self):
        exterior = [[0, 0], [30, 0], [30, 30], [0, 30], [0, 0]]
        hole = [[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {},
                 "geometry": {"type": "Polygon", "coordinates": [exterior, hole]}},
            ],
        }
        _gtype, seqs, _props = next(iter(iter_features(fc)))
        # Only the exterior ring is yielded.
        assert len(seqs) == 1
        assert len(seqs[0]) == len(exterior)

    def test_multipolygon(self):
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {},
                 "geometry": {
                     "type": "MultiPolygon",
                     "coordinates": [
                         [[[0, 0], [5, 0], [5, 5], [0, 0]]],
                         [[[10, 10], [15, 10], [15, 15], [10, 10]]],
                     ],
                 }},
            ],
        }
        gtype, seqs, _props = next(iter(iter_features(fc)))
        assert gtype == "MultiPolygon"
        assert len(seqs) == 2  # one exterior ring per polygon
        assert seqs[0][0] == (0.0, 0.0)
        assert seqs[1][0] == (10.0, 10.0)

    def test_linestring(self):
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"width_m": 6.0},
                 "geometry": {"type": "LineString",
                              "coordinates": [[0, 0], [100, 0]]}},
            ],
        }
        gtype, seqs, props = next(iter(iter_features(fc)))
        assert gtype in LINE_TYPES
        assert len(seqs) == 1
        assert seqs[0] == [(0.0, 0.0), (100.0, 0.0)]
        assert props["width_m"] == 6.0

    def test_multilinestring(self):
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {},
                 "geometry": {
                     "type": "MultiLineString",
                     "coordinates": [
                         [[0, 0], [10, 0]],
                         [[0, 10], [10, 10]],
                     ],
                 }},
            ],
        }
        gtype, seqs, _props = next(iter(iter_features(fc)))
        assert gtype == "MultiLineString"
        assert len(seqs) == 2
        assert seqs[0] == [(0.0, 0.0), (10.0, 0.0)]
        assert seqs[1] == [(0.0, 10.0), (10.0, 10.0)]

    def test_unsupported_geometry_skipped(self):
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {},
                 "geometry": {"type": "Point", "coordinates": [1, 2]}},
                {"type": "Feature", "properties": {},
                 "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}},
            ],
        }
        feats = list(iter_features(fc))
        assert len(feats) == 1
        assert feats[0][0] == "LineString"

    def test_iter_polygons_and_lines_filters(self):
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {},
                 "geometry": {"type": "Polygon",
                              "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}},
                {"type": "Feature", "properties": {},
                 "geometry": {"type": "LineString", "coordinates": [[0, 0], [5, 5]]}},
            ],
        }
        polys = list(iter_polygons(fc))
        lines = list(iter_lines(fc))
        assert len(polys) == 1
        assert len(lines) == 1

    def test_empty_and_malformed(self):
        assert list(iter_features({})) == []
        assert list(iter_features({"type": "FeatureCollection", "features": []})) == []
        assert list(iter_features("not a dict")) == []
        # Feature with no geometry is skipped.
        fc = {"type": "FeatureCollection", "features": [{"type": "Feature"}]}
        assert list(iter_features(fc)) == []


# ---------------------------------------------------------------------------
# GeoJSON iteration — WGS-84 via to_local projection
# ---------------------------------------------------------------------------

class TestToLocalProjection:
    def setup_method(self):
        geo.reset()

    def teardown_method(self):
        geo.reset()

    def test_wgs84_to_local_uninitialised_raises(self):
        geo.reset()
        with pytest.raises(RuntimeError):
            wgs84_to_local()

    def test_wgs84_to_local_projects(self):
        geo.init_reference(lat=30.0, lng=-97.0)
        to_local = wgs84_to_local()
        # At the reference point local coords are ~(0, 0).
        x, y = to_local(-97.0, 30.0)
        assert abs(x) < 1e-6
        assert abs(y) < 1e-6
        # One degree of latitude north -> +Y positive, large.
        x2, y2 = to_local(-97.0, 31.0)
        assert y2 > 100_000.0
        assert abs(x2) < 1e-3

    def test_iter_features_with_to_local(self):
        geo.init_reference(lat=30.0, lng=-97.0)
        to_local = wgs84_to_local()
        # GeoJSON coordinates are [lng, lat].
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {},
                 "geometry": {"type": "LineString",
                              "coordinates": [[-97.0, 30.0], [-97.0, 30.001]]}},
            ],
        }
        _gtype, seqs, _props = next(iter(iter_features(fc, to_local=to_local)))
        p0, p1 = seqs[0]
        assert abs(p0[0]) < 1e-3 and abs(p0[1]) < 1e-3
        # 0.001 deg lat north -> ~111 m in +Y.
        assert p1[1] == pytest.approx(0.001 * geo.METERS_PER_DEG_LAT, rel=1e-6)

    def test_multipolygon_with_to_local(self):
        geo.init_reference(lat=0.0, lng=0.0)
        to_local = wgs84_to_local()
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {},
                 "geometry": {
                     "type": "MultiPolygon",
                     "coordinates": [
                         [[[0.0, 0.0], [0.001, 0.0], [0.001, 0.001], [0.0, 0.0]]],
                     ],
                 }},
            ],
        }
        _gtype, seqs, _props = next(iter(iter_features(fc, to_local=to_local)))
        assert len(seqs) == 1
        # First vertex projects to near-origin.
        assert math.hypot(*seqs[0][0]) < 1e-3
