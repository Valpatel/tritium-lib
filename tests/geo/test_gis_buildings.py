# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for OverpassBuildingsFetcher + filter_features_bbox (bbox clipping).

NETWORK IS NEVER TOUCHED: ``urllib.request.urlopen`` is monkeypatched wherever
``fetch`` is exercised.  ``parse_buildings`` is fed small excerpts shaped like a
real Overpass ``out geom`` response.
"""

import json

import pytest

from tritium_lib.geo.gis import (
    GISCache,
    OverpassBuildingsFetcher,
    filter_features_bbox,
)
from tritium_lib.geo.gis.models import GeoBBox

AO = GeoBBox.from_string("-121.912,37.704,-121.880,37.728")


# ---------------------------------------------------------------------------
# Fake HTTP transport (mirrors test_gis_fetchers.py)
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, body):
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


def _way(coords, tags=None):
    return {
        "type": "way",
        "id": 1,
        "geometry": [{"lon": lon, "lat": lat} for lon, lat in coords],
        "tags": tags or {},
    }


OVERPASS_RAW = {
    "version": 0.6,
    "elements": [
        _way(
            [(-121.888, 37.711), (-121.887, 37.711), (-121.887, 37.712),
             (-121.888, 37.712)],
            {"building": "house", "name": "Casa", "building:levels": "2"},
        ),
        _way(
            [(-121.890, 37.713), (-121.889, 37.713), (-121.889, 37.714)],
            {"building": "yes"},
        ),
        _way(
            [(-121.895, 37.715), (-121.894, 37.715), (-121.894, 37.716),
             (-121.895, 37.716)],
            {"building": "commercial", "height": "12 m"},
        ),
        # Fewer than 3 points -> dropped.
        _way([(-121.900, 37.720), (-121.899, 37.720)], {"building": "yes"}),
        # Non-way element -> ignored.
        {"type": "node", "id": 9, "lon": -121.9, "lat": 37.7},
    ],
}


# ---------------------------------------------------------------------------
# parse_buildings
# ---------------------------------------------------------------------------
class TestParseBuildings:
    @pytest.mark.unit
    def test_shapes_and_props(self):
        fc = OverpassBuildingsFetcher.parse_buildings(OVERPASS_RAW)
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 3  # 2-pt way + node dropped

        first = fc["features"][0]
        assert first["geometry"]["type"] == "Polygon"
        ring = first["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1]  # ring closed
        p = first["properties"]
        assert p["source"] == "osm"
        assert p["kind"] == "house"
        assert p["name"] == "Casa"
        # building:levels 2 -> 2*3 + 1 = 7 m -> levels max(1, int(7/3)) = 2.
        assert p["height_m"] == pytest.approx(7.0)
        assert p["levels"] == 2

    @pytest.mark.unit
    def test_height_from_tag_and_default(self):
        fc = OverpassBuildingsFetcher.parse_buildings(OVERPASS_RAW)
        by_kind = {f["properties"]["kind"]: f["properties"] for f in fc["features"]}
        # Explicit "12 m" height -> 12.0, levels int(12/3) = 4.
        assert by_kind["commercial"]["height_m"] == pytest.approx(12.0)
        assert by_kind["commercial"]["levels"] == 4
        # Untyped building with no height tags -> default 8.0, levels 2.
        assert by_kind["yes"]["kind"] == "yes"
        assert by_kind["yes"]["height_m"] == pytest.approx(8.0)
        assert by_kind["yes"]["levels"] == 2
        assert by_kind["yes"]["name"] == ""

    @pytest.mark.unit
    def test_empty_input_valid(self):
        assert OverpassBuildingsFetcher.parse_buildings({}) == {
            "type": "FeatureCollection", "features": []
        }


# ---------------------------------------------------------------------------
# Overpass request body ordering (south, west, north, east)
# ---------------------------------------------------------------------------
class TestOverpassBody:
    @pytest.mark.unit
    def test_body_uses_swne_order(self):
        body = OverpassBuildingsFetcher()._build_body(AO).decode("utf-8")
        # urlencoded; the bbox appears as s,w,n,e in that order.
        assert "37.704%2C-121.912%2C37.728%2C-121.88" in body
        assert "out%20geom" in body or "out+geom" in body


# ---------------------------------------------------------------------------
# Degradation chain + fixture clipping
# ---------------------------------------------------------------------------
class TestBuildingsDegradation:
    @pytest.mark.unit
    def test_live_success_parses_and_caches(self, monkeypatch, tmp_path):
        _patch(monkeypatch, _ok(OVERPASS_RAW))
        cache = GISCache(tmp_path)
        f = OverpassBuildingsFetcher(cache=cache)
        fc = f.fetch(AO)
        assert len(fc["features"]) == 3
        assert fc["features"][0]["properties"]["source"] == "osm"
        assert cache.get(cache.key("osm", AO)) == fc

    @pytest.mark.unit
    def test_offline_ao_uses_fixture(self, monkeypatch):
        _patch(monkeypatch, _boom)
        fc = OverpassBuildingsFetcher(cache=None).fetch(AO)
        assert fc.get("fixture") is True
        assert len(fc["features"]) >= 100  # trimmed to <=120, AO all inside
        assert all(x["properties"]["source"] == "osm" for x in fc["features"])
        # Real footprints carry the normalized property set.
        keys = set(fc["features"][0]["properties"])
        assert {"source", "kind", "name", "height_m", "levels"} <= keys

    @pytest.mark.unit
    def test_offline_far_bbox_is_empty(self, monkeypatch):
        # The fixture covers the demo AO; a distant window must clip to empty
        # rather than return the whole AO worth of buildings.
        _patch(monkeypatch, _boom)
        far = GeoBBox.from_string("10.0,10.0,10.1,10.1")
        fc = OverpassBuildingsFetcher(cache=None).fetch(far)
        assert fc.get("fixture") is True
        assert fc["features"] == []


# ---------------------------------------------------------------------------
# filter_features_bbox (pure)
# ---------------------------------------------------------------------------
def _pt(lon, lat):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {}}


def _poly(ring):
    return {"type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {}}


class TestFilterFeaturesBbox:
    @pytest.mark.unit
    def test_inside_kept_outside_dropped(self):
        fc = {"type": "FeatureCollection", "features": [
            _pt(-121.9, 37.71),   # inside AO
            _pt(0.0, 0.0),        # far outside
        ]}
        out = filter_features_bbox(fc, AO)
        assert len(out["features"]) == 1
        assert out["features"][0]["geometry"]["coordinates"] == [-121.9, 37.71]

    @pytest.mark.unit
    def test_straddling_feature_kept(self):
        straddle = {"type": "Feature", "geometry": {
            "type": "LineString",
            "coordinates": [[-121.9, 37.71], [-100.0, 20.0]]}, "properties": {}}
        out = filter_features_bbox(
            {"type": "FeatureCollection", "features": [straddle]}, AO)
        assert len(out["features"]) == 1

    @pytest.mark.unit
    def test_polygon_enclosing_bbox_kept(self):
        # A big polygon whose bbox fully contains the AO intersects it.
        big = _poly([[-130, 30], [-110, 30], [-110, 45], [-130, 45], [-130, 30]])
        out = filter_features_bbox(
            {"type": "FeatureCollection", "features": [big]}, AO)
        assert len(out["features"]) == 1

    @pytest.mark.unit
    def test_no_coord_features_dropped(self):
        fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": None, "properties": {}},
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []},
             "properties": {}},
        ]}
        assert filter_features_bbox(fc, AO)["features"] == []

    @pytest.mark.unit
    def test_empty_and_fixture_marker_preserved(self):
        empty = filter_features_bbox(
            {"type": "FeatureCollection", "features": []}, AO)
        assert empty == {"type": "FeatureCollection", "features": []}
        marked = filter_features_bbox(
            {"type": "FeatureCollection", "fixture": True,
             "features": [_pt(0.0, 0.0)]}, AO)
        assert marked.get("fixture") is True
        assert marked["features"] == []
