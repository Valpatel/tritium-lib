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
    OverpassBuildingsFetcher,
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
        # Multi-AO contract: FIXTURE_NAMES (not the legacy single FIXTURE_NAME)
        # is the resolution list once a fetcher registers real packs, so sabotage
        # it directly.  With no loadable pack the chain hits the bare empty FC.
        f.FIXTURE_NAMES = ("does_not_exist.json",)
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
        # Sabotage the multi-AO resolution list (see vector twin above).
        f.FIXTURE_NAMES = ("does_not_exist.json",)
        grid = f.fetch_grid(BOX, ncols=4, nrows=4)
        assert grid.source == "usgs-empty"
        assert grid.ncols == 4 and grid.nrows == 4
        assert all(v is None for v in grid.values)
        assert grid.min_max() == (None, None)


# ---------------------------------------------------------------------------
# Multi-AO fixture resolution (fake packs via monkeypatched _load_fixture)
# ---------------------------------------------------------------------------
def _pt_feature(lon, lat, kind):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {"source": "tiger", "kind": kind},
    }


def _pack(*features):
    return {"type": "FeatureCollection", "fixture": True, "features": list(features)}


class TestMultiFixtureResolution:
    @pytest.mark.unit
    def test_first_nonempty_pack_wins(self, monkeypatch):
        # Two packs registered in order: pack A only has a far-away feature
        # (clips to empty over BOX), pack B has a BOX-local one -> B wins.
        import tritium_lib.geo.gis.fetchers as fmod

        packs = {
            "a.json": _pack(_pt_feature(10.0, 10.0, "A")),        # far away
            "b.json": _pack(_pt_feature(-121.9, 37.71, "B")),     # inside BOX
        }
        monkeypatch.setattr(fmod, "_load_fixture", lambda n: packs.get(n))
        _patch(monkeypatch, _boom)  # force offline
        f = TigerRoadsFetcher(cache=None)
        f.FIXTURE_NAMES = ("a.json", "b.json")
        fc = f.fetch(BOX)
        assert fc.get("fixture") is True
        assert [x["properties"]["kind"] for x in fc["features"]] == ["B"]

    @pytest.mark.unit
    def test_all_packs_clip_empty_yields_fixture_marked_empty(self, monkeypatch):
        # Every pack clips to empty (bbox outside all AOs) -> a fixture-marked
        # empty FC (the pre-multi-AO far-bbox contract is preserved).
        import tritium_lib.geo.gis.fetchers as fmod

        far = _pack(_pt_feature(10.0, 10.0, "FAR"))
        monkeypatch.setattr(
            fmod, "_load_fixture", lambda n: far if n in ("a.json", "b.json") else None
        )
        _patch(monkeypatch, _boom)
        f = TigerRoadsFetcher(cache=None)
        f.FIXTURE_NAMES = ("a.json", "b.json")
        fc = f.fetch(BOX)
        assert fc.get("fixture") is True
        assert fc["features"] == []

    @pytest.mark.unit
    def test_legacy_fixture_name_still_resolves(self, monkeypatch):
        # Back-compat: a fetcher that sets ONLY the legacy single FIXTURE_NAME
        # (FIXTURE_NAMES empty) still resolves via (FIXTURE_NAME,).
        import tritium_lib.geo.gis.fetchers as fmod

        pack = _pack(_pt_feature(-121.9, 37.71, "LEGACY"))
        monkeypatch.setattr(
            fmod, "_load_fixture", lambda n: pack if n == "legacy.json" else None
        )
        _patch(monkeypatch, _boom)
        f = TigerRoadsFetcher(cache=None)
        f.FIXTURE_NAMES = ()             # multi-AO list not configured
        f.FIXTURE_NAME = "legacy.json"   # legacy single fixture
        fc = f.fetch(BOX)
        assert [x["properties"]["kind"] for x in fc["features"]] == ["LEGACY"]


# ---------------------------------------------------------------------------
# DEM fixture gating — the intersect-or-NoData contract change
# ---------------------------------------------------------------------------
class TestGridFixtureGating:
    @pytest.mark.unit
    def test_offline_bbox_intersecting_dublin_returns_dublin(self, monkeypatch):
        _patch(monkeypatch, _boom)
        f = UsgsElevationFetcher(cache=None)  # packaged Dublin DEM
        grid = f.fetch_grid(BOX)              # BOX == Dublin AO
        assert grid.source == "usgs-fixture"
        mn, mx = grid.min_max()
        assert 100.0 < mn < mx < 200.0        # real Dublin terrain

    @pytest.mark.unit
    def test_offline_bbox_outside_all_aos_returns_nodata(self, monkeypatch):
        # ⚠️ CONTRACT CHANGE: an offline bbox outside EVERY packaged AO no longer
        # silently gets the Dublin DEM — it gets an all-NoData "usgs-empty" grid.
        _patch(monkeypatch, _boom)
        kansas = GeoBBox.from_string("-98.00,38.00,-97.96,38.04")
        f = UsgsElevationFetcher(cache=None)
        grid = f.fetch_grid(kansas, ncols=8, nrows=8)
        assert grid.source == "usgs-empty"
        assert grid.ncols == 8 and grid.nrows == 8
        assert all(v is None for v in grid.values)
        assert grid.min_max() == (None, None)

    @pytest.mark.unit
    def test_fixture_bbox_prefers_marker_else_corner_fields(self):
        # Explicit top-level "bbox" marker wins when present ...
        assert UsgsElevationFetcher._fixture_bbox(
            {"bbox": [1.0, 2.0, 3.0, 4.0],
             "west": 9, "south": 9, "east": 9, "north": 9}
        ) == (1.0, 2.0, 3.0, 4.0)
        # ... else the grid's own corner fields (the Dublin DEM has no "bbox").
        assert UsgsElevationFetcher._fixture_bbox(
            {"west": -121.912, "south": 37.704, "east": -121.88, "north": 37.728}
        ) == (-121.912, 37.704, -121.88, 37.728)
        assert UsgsElevationFetcher._fixture_bbox({}) is None


# ---------------------------------------------------------------------------
# Boulder, CO pack — proves the offline system is NOT Dublin-hardcoded
# ---------------------------------------------------------------------------
BOULDER = GeoBBox.from_string("-105.30,39.98,-105.26,40.02")
_BOULDER_BBOX = [-105.3, 39.98, -105.26, 40.02]


def _positions(coords):
    """Yield ``(lon, lat)`` for every position in a coordinate tree."""
    if isinstance(coords, (list, tuple)):
        if (len(coords) >= 2 and isinstance(coords[0], (int, float))
                and isinstance(coords[1], (int, float))):
            yield (float(coords[0]), float(coords[1]))
        else:
            for child in coords:
                yield from _positions(child)


def _all_positions(fc):
    for feat in fc["features"]:
        yield from _positions(feat["geometry"]["coordinates"])


class TestBoulderPack:
    @pytest.mark.unit
    def test_boulder_fixtures_load_via_resources(self):
        from importlib import resources

        for name in ("tiger_roads_boulder.json", "fema_flood_boulder.json",
                     "osm_buildings_boulder.json", "usgs_dem_boulder.json"):
            res = resources.files("tritium_lib.geo.gis.fixtures").joinpath(name)
            data = json.loads(res.read_text(encoding="utf-8"))
            assert data.get("fixture") is True
            assert data.get("bbox") == _BOULDER_BBOX

    @pytest.mark.unit
    def test_offline_roads_are_boulder_not_dublin(self, monkeypatch):
        # Live + cache disabled -> packaged fixture.  The Boulder pack must win
        # for a Boulder bbox, and every coordinate must sit near Boulder (lon
        # ~-105) — never Dublin, CA (lon ~-121.9).
        _patch(monkeypatch, _boom)
        fc = TigerRoadsFetcher(cache=None).fetch(BOULDER)
        assert fc.get("fixture") is True
        assert len(fc["features"]) >= 1
        assert all(x["properties"]["source"] == "tiger" for x in fc["features"])
        for lon, lat in _all_positions(fc):
            assert -106.0 <= lon <= -105.0     # Boulder longitude band
            assert 39.0 <= lat <= 41.0

    @pytest.mark.unit
    def test_offline_buildings_are_boulder_not_dublin(self, monkeypatch):
        _patch(monkeypatch, _boom)
        fc = OverpassBuildingsFetcher(cache=None).fetch(BOULDER)
        assert fc.get("fixture") is True
        assert len(fc["features"]) >= 1
        for lon, _lat in _all_positions(fc):
            assert -106.0 <= lon <= -105.0

    @pytest.mark.unit
    def test_offline_dem_is_boulder_terrain(self, monkeypatch):
        # Boulder relief (mountains-to-plains) is ~1600-2400 m; Dublin is
        # ~100-200 m — an unambiguous discriminator.
        _patch(monkeypatch, _boom)
        grid = UsgsElevationFetcher(cache=None).fetch_grid(BOULDER, ncols=32, nrows=32)
        assert grid.source == "usgs-fixture"
        assert grid.west == pytest.approx(-105.30)
        assert grid.north == pytest.approx(40.02)
        mn, mx = grid.min_max()
        assert mn > 1000.0 and mx > 1000.0     # NOT Dublin terrain
        assert mx < 4000.0

    @pytest.mark.unit
    def test_dublin_bbox_still_gets_dublin(self, monkeypatch):
        # The Dublin pack is first in FIXTURE_NAMES; a Dublin bbox still resolves
        # to Dublin data (Boulder is registered but does not intersect).
        _patch(monkeypatch, _boom)
        fc = TigerRoadsFetcher(cache=None).fetch(BOX)
        assert fc.get("fixture") is True
        for lon, _lat in _all_positions(fc):
            assert -122.5 <= lon <= -121.5     # Dublin longitude band
        grid = UsgsElevationFetcher(cache=None).fetch_grid(BOX)
        assert grid.min_max()[0] < 200.0       # Dublin terrain


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
