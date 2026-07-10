# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geo.gis.fetchers — parse funcs + degradation chain.

NETWORK IS NEVER TOUCHED: ``urllib.request.urlopen`` is monkeypatched in every
test that exercises ``fetch``.  The ``parse_*`` staticmethods are fed small
excerpts of the real captured government payloads (same shapes as the packaged
fixtures were generated from).
"""

import json

import pytest

from tritium_lib.geo.gis import (
    USGS_HILLSHADE_TILE_URL,
    FemaFloodFetcher,
    GISCache,
    NoaaAlertsFetcher,
    TigerRoadsFetcher,
    UsgsElevationFetcher,
)
from tritium_lib.geo.gis.models import ElevationGrid, GeoBBox

BOX = GeoBBox.from_string("-121.912,37.704,-121.880,37.728")


# ---------------------------------------------------------------------------
# Fake HTTP transport
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _ok(payload):
    body = json.dumps(payload).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        return _FakeResp(body)

    return fake_urlopen


def _boom(req, timeout=None):
    raise OSError("network disabled for test")


def _patch(monkeypatch, fn):
    monkeypatch.setattr("urllib.request.urlopen", fn)


# ---------------------------------------------------------------------------
# Real captured-payload excerpts (same shape as the raw government responses)
# ---------------------------------------------------------------------------
USGS_RAW = {
    "samples": [
        {"location": {"x": -121.912, "y": 37.728}, "locationId": 0,
         "value": "117.770233154", "resolution": 1},
        {"location": {"x": -121.880, "y": 37.728}, "locationId": 1,
         "value": "150.376556396", "resolution": 1},
        {"location": {"x": -121.912, "y": 37.704}, "locationId": 2,
         "value": "NoData", "resolution": 1},
        {"location": {"x": -121.880, "y": 37.704}, "locationId": 3,
         "value": "101.169273376", "resolution": 1},
    ]
}

TIGER_RAW = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature",
         "geometry": {"type": "LineString",
                      "coordinates": [[-121.888, 37.711], [-121.887, 37.711]]},
         "properties": {"BASENAME": "Summer Glen", "NAME": "Summer Glen Dr",
                        "MTFCC": "S1400"}},
        {"type": "Feature",
         "geometry": {"type": "LineString",
                      "coordinates": [[-121.90, 37.72], [-121.89, 37.72]]},
         "properties": {"BASENAME": "", "NAME": "", "MTFCC": "S1630"}},
        # Geometry-less row must be dropped.
        {"type": "Feature", "geometry": None,
         "properties": {"BASENAME": "Ghost", "NAME": "Ghost Rd",
                        "MTFCC": "S1400"}},
    ],
}

FEMA_RAW = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature",
         "geometry": {"type": "Polygon",
                      "coordinates": [[[-121.9, 37.71], [-121.89, 37.71],
                                       [-121.89, 37.72], [-121.9, 37.71]]]},
         "properties": {"FLD_ZONE": "AE", "ZONE_SUBTY": None, "SFHA_TF": "T"}},
        {"type": "Feature",
         "geometry": {"type": "Polygon",
                      "coordinates": [[[-121.9, 37.70], [-121.89, 37.70],
                                       [-121.89, 37.71], [-121.9, 37.70]]]},
         "properties": {"FLD_ZONE": "X",
                        "ZONE_SUBTY": "AREA OF MINIMAL FLOOD HAZARD",
                        "SFHA_TF": "F"}},
    ],
}

# NWS area=CA capture: zone/UGC-based alerts have geometry=null (the common
# case).  A point response can carry a real polygon.
NOAA_RAW_NO_GEOM = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "geometry": None,
         "properties": {"event": "Heat Advisory", "severity": "Moderate",
                        "headline": "Heat Advisory until 8 PM",
                        "expires": "2026-07-10T20:00:00-07:00"}},
        {"type": "Feature", "geometry": None,
         "properties": {"event": "Red Flag Warning", "severity": "Severe"}},
    ],
}

NOAA_RAW_WITH_GEOM = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature",
         "geometry": {"type": "Polygon",
                      "coordinates": [[[-121.9, 37.71], [-121.89, 37.71],
                                       [-121.89, 37.72], [-121.9, 37.71]]]},
         "properties": {"event": "Extreme Heat Warning", "severity": "Extreme",
                        "headline": "Dangerous heat", "expires": "2026-07-11T20:00:00-07:00"}},
        {"type": "Feature", "geometry": None,
         "properties": {"event": "Air Quality Alert", "severity": "Unknown"}},
    ],
}


# ---------------------------------------------------------------------------
# Pure parse tests (no IO)
# ---------------------------------------------------------------------------
class TestParseGrid:
    @pytest.mark.unit
    def test_parse_grid_shape_and_nodata(self):
        grid = UsgsElevationFetcher.parse_grid(USGS_RAW, BOX, 2, 2)
        assert isinstance(grid, ElevationGrid)
        assert grid.ncols == 2 and grid.nrows == 2
        assert grid.resolution_m == 1
        assert grid.value_at(0, 0) == pytest.approx(117.770233154)
        assert grid.value_at(1, 0) == pytest.approx(150.376556396)
        assert grid.value_at(0, 1) is None          # "NoData" -> None
        assert grid.value_at(1, 1) == pytest.approx(101.169273376)

    @pytest.mark.unit
    def test_parse_grid_row0_is_north(self):
        grid = UsgsElevationFetcher.parse_grid(USGS_RAW, BOX, 2, 2)
        # locationId 0 sits on the north edge of the box.
        assert grid.cell_lat(0) == pytest.approx(BOX.north)

    @pytest.mark.unit
    def test_parse_grid_empty(self):
        grid = UsgsElevationFetcher.parse_grid({}, BOX, 2, 2)
        assert grid.values == [None, None, None, None]


class TestParseRoads:
    @pytest.mark.unit
    def test_normalizes_and_drops_geometryless(self):
        fc = TigerRoadsFetcher.parse_roads(TIGER_RAW)
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 2  # ghost (no geometry) dropped
        props = fc["features"][0]["properties"]
        assert props["source"] == "tiger"
        assert props["kind"] == "S1400"
        assert props["name"] == "Summer Glen Dr"
        # Empty street name is preserved as "".
        assert fc["features"][1]["properties"]["name"] == ""
        assert fc["features"][1]["properties"]["kind"] == "S1630"


class TestParseFlood:
    @pytest.mark.unit
    def test_sfha_and_subtype(self):
        fc = FemaFloodFetcher.parse_flood(FEMA_RAW)
        assert len(fc["features"]) == 2
        p0 = fc["features"][0]["properties"]
        assert p0["source"] == "fema"
        assert p0["kind"] == "AE"
        assert p0["sfha"] is True
        assert p0["subtype"] == ""      # None -> ""
        p1 = fc["features"][1]["properties"]
        assert p1["kind"] == "X"
        assert p1["sfha"] is False
        assert p1["subtype"] == "AREA OF MINIMAL FLOOD HAZARD"


class TestParseAlerts:
    @pytest.mark.unit
    def test_drops_geometryless(self):
        fc = NoaaAlertsFetcher.parse_alerts(NOAA_RAW_WITH_GEOM)
        assert len(fc["features"]) == 1  # only the polygon-bearing one
        p = fc["features"][0]["properties"]
        assert p["source"] == "noaa"
        assert p["kind"] == "Extreme Heat Warning"
        assert p["severity"] == "Extreme"
        assert p["headline"] == "Dangerous heat"
        assert p["expires"].startswith("2026-07-11")

    @pytest.mark.unit
    def test_all_geometryless_yields_empty_but_valid(self):
        fc = NoaaAlertsFetcher.parse_alerts(NOAA_RAW_NO_GEOM)
        assert fc == {"type": "FeatureCollection", "features": []}

    @pytest.mark.unit
    def test_empty_input_is_valid(self):
        assert NoaaAlertsFetcher.parse_alerts({}) == {
            "type": "FeatureCollection", "features": []
        }


# ---------------------------------------------------------------------------
# Degradation chain (vector fetchers) — monkeypatched transport
# ---------------------------------------------------------------------------
class TestVectorDegradation:
    @pytest.mark.unit
    def test_live_success_parses_and_caches(self, monkeypatch, tmp_path):
        _patch(monkeypatch, _ok(TIGER_RAW))
        cache = GISCache(tmp_path)
        f = TigerRoadsFetcher(cache=cache)
        fc = f.fetch(BOX)
        assert len(fc["features"]) == 2
        # Cached under the computed key for a later offline hit.
        assert cache.get(cache.key("tiger", BOX)) == fc

    @pytest.mark.unit
    def test_live_fail_cache_hit(self, monkeypatch, tmp_path):
        cache = GISCache(tmp_path)
        f = TigerRoadsFetcher(cache=cache)
        cached = {"type": "FeatureCollection",
                  "features": [{"type": "Feature", "geometry": {},
                                "properties": {"source": "tiger",
                                               "kind": "CACHED"}}]}
        cache.put(cache.key("tiger", BOX), cached)
        _patch(monkeypatch, _boom)
        assert f.fetch(BOX) == cached

    @pytest.mark.unit
    def test_live_and_cache_fail_uses_fixture(self, monkeypatch, tmp_path):
        _patch(monkeypatch, _boom)
        f = FemaFloodFetcher(cache=GISCache(tmp_path))  # empty cache
        fc = f.fetch(BOX)
        assert fc.get("fixture") is True
        assert len(fc["features"]) >= 1
        assert fc["features"][0]["properties"]["source"] == "fema"

    @pytest.mark.unit
    def test_no_cache_falls_through_to_fixture(self, monkeypatch):
        _patch(monkeypatch, _boom)
        f = NoaaAlertsFetcher(cache=None)
        fc = f.fetch(BOX)
        assert fc.get("fixture") is True
        assert all(feat["properties"]["source"] == "noaa"
                   for feat in fc["features"])

    @pytest.mark.unit
    def test_all_fail_returns_empty_collection(self, monkeypatch):
        _patch(monkeypatch, _boom)
        f = TigerRoadsFetcher(cache=None)
        f.FIXTURE_NAME = "does_not_exist.json"  # sabotage the last fallback
        assert f.fetch(BOX) == {"type": "FeatureCollection", "features": []}

    @pytest.mark.unit
    def test_live_empty_is_cached_not_treated_as_failure(self, monkeypatch, tmp_path):
        # A genuine empty NWS response must be returned + cached, NOT trigger
        # the fixture fallback.
        _patch(monkeypatch, _ok({"type": "FeatureCollection", "features": []}))
        cache = GISCache(tmp_path)
        f = NoaaAlertsFetcher(cache=cache)
        fc = f.fetch(BOX)
        assert fc == {"type": "FeatureCollection", "features": []}
        assert "fixture" not in fc
        assert cache.get(cache.key("noaa", BOX)) == fc


# ---------------------------------------------------------------------------
# Degradation chain (USGS elevation grid)
# ---------------------------------------------------------------------------
class TestGridDegradation:
    @pytest.mark.unit
    def test_live_success_parses_and_caches(self, monkeypatch, tmp_path):
        _patch(monkeypatch, _ok(USGS_RAW))
        cache = GISCache(tmp_path)
        f = UsgsElevationFetcher(cache=cache)
        grid = f.fetch_grid(BOX, ncols=2, nrows=2)
        assert grid.value_at(0, 0) == pytest.approx(117.770233154)
        # Cached as to_dict() and reconstructable.
        cached = cache.get(cache.key("usgs", BOX, ncols=2, nrows=2))
        assert cached is not None
        assert ElevationGrid.from_dict(cached).ncols == 2

    @pytest.mark.unit
    def test_live_fail_cache_hit(self, monkeypatch, tmp_path):
        cache = GISCache(tmp_path)
        f = UsgsElevationFetcher(cache=cache)
        good = UsgsElevationFetcher.parse_grid(USGS_RAW, BOX, 2, 2)
        cache.put(cache.key("usgs", BOX, ncols=2, nrows=2), good.to_dict())
        _patch(monkeypatch, _boom)
        grid = f.fetch_grid(BOX, ncols=2, nrows=2)
        assert grid.value_at(1, 1) == pytest.approx(101.169273376)

    @pytest.mark.unit
    def test_live_and_cache_fail_uses_fixture(self, monkeypatch, tmp_path):
        _patch(monkeypatch, _boom)
        f = UsgsElevationFetcher(cache=GISCache(tmp_path))
        grid = f.fetch_grid(BOX)  # default 16x16 == fixture dims
        assert grid.source == "usgs-fixture"
        assert grid.ncols == 16 and grid.nrows == 16
        mn, mx = grid.min_max()
        assert 100.0 < mn < mx < 200.0

    @pytest.mark.unit
    def test_all_fail_returns_empty_grid(self, monkeypatch):
        _patch(monkeypatch, _boom)
        f = UsgsElevationFetcher(cache=None)
        f.FIXTURE_NAME = "does_not_exist.json"
        grid = f.fetch_grid(BOX, ncols=4, nrows=4)
        assert grid.source == "usgs-empty"
        assert grid.ncols == 4 and grid.nrows == 4
        assert all(v is None for v in grid.values)
        assert grid.min_max() == (None, None)


class TestConstants:
    @pytest.mark.unit
    def test_hillshade_url_template(self):
        assert USGS_HILLSHADE_TILE_URL.startswith("https://basemap.nationalmap.gov")
        assert USGS_HILLSHADE_TILE_URL.endswith("/tile/{z}/{y}/{x}")

    @pytest.mark.unit
    def test_noaa_uses_point_query(self):
        f = NoaaAlertsFetcher()
        url = f._build_url(BOX)
        lon, lat = BOX.center()
        assert "point=" in url
        assert f"{lat}" in url and f"{lon}" in url
