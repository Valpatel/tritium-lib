# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the NHD hydrography fetcher (NhdHydrographyFetcher).

NETWORK IS NEVER TOUCHED: ``urllib.request.urlopen`` is monkeypatched in every
test that exercises the live path. Because the NHD fetcher queries TWO layers
(flowline #3 + waterbody #9) per live fetch, the fake transport dispatches on
the request URL so each layer gets its own payload. The ``parse_*`` funcs are
fed small excerpts matching the real NHDPlus_HR GeoJSON shape; the packaged
demo-AO fixtures are asserted to load and normalize.
"""

import json

from tritium_lib.geo.gis import GISCache, NhdHydrographyFetcher
from tritium_lib.geo.gis.models import GeoBBox

DUBLIN = GeoBBox.from_string("-121.912,37.704,-121.880,37.728")
BOULDER = GeoBBox.from_string("-105.3,39.98,-105.26,40.02")
# A window far from every packaged AO (mid-Atlantic ocean).
FAR = GeoBBox.from_string("-30.0,0.0,-29.9,0.1")


# ---------------------------------------------------------------------------
# Fake HTTP transport (dispatch by layer number in the URL)
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


def _dispatch(flow_payload, wb_payload):
    """Return a fake urlopen that serves flowline vs waterbody by URL."""
    flow_body = json.dumps(flow_payload).encode("utf-8")
    wb_body = json.dumps(wb_payload).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        url = getattr(req, "full_url", "") or ""
        if "/9/query" in url:
            return _FakeResp(wb_body)
        return _FakeResp(flow_body)

    return fake_urlopen


def _boom(req, timeout=None):
    raise OSError("network disabled for test")


def _patch(monkeypatch, fn):
    monkeypatch.setattr("urllib.request.urlopen", fn)


# Line geometry near Dublin so it survives fixture bbox clipping tests.
_LINE = {"type": "LineString", "coordinates": [[-121.90, 37.71], [-121.89, 37.72]]}
_POLY = {
    "type": "Polygon",
    "coordinates": [[[-121.90, 37.71], [-121.89, 37.71], [-121.89, 37.72],
                     [-121.90, 37.72], [-121.90, 37.71]]],
}

FLOW_RAW = {
    "features": [
        {"geometry": _LINE,
         "properties": {"GNIS_NAME": "Alamo Creek", "FType": 460, "FCode": 46006}},
        {"geometry": _LINE,
         "properties": {"GNIS_NAME": "", "FType": 336, "FCode": 33600}},
        # Lowercased field names — must still parse (case-insensitive).
        {"geometry": _LINE,
         "properties": {"gnis_name": "Tassajara", "ftype": 558}},
        # Unknown FType -> "stream"; geometryless dropped.
        {"geometry": _LINE, "properties": {"FType": 99999}},
        {"geometry": None, "properties": {"FType": 460}},
    ]
}
WB_RAW = {
    "features": [
        {"geometry": _POLY,
         "properties": {"GNIS_NAME": "Shadow Cliffs", "FType": 390, "AreaSqKm": 0.42}},
        {"geometry": None, "properties": {"FType": 390}},
    ]
}


class TestParseFlowlines:
    def test_ftype_kind_mapping_and_names(self):
        feats = NhdHydrographyFetcher.parse_flowlines(FLOW_RAW)
        # 4 kept (1 geometryless dropped).
        assert len(feats) == 4
        kinds = [f["properties"]["kind"] for f in feats]
        assert kinds == ["river", "canal", "artificial", "stream"]
        assert feats[0]["properties"]["name"] == "Alamo Creek"
        assert feats[0]["properties"]["source"] == "nhd"
        assert feats[0]["properties"]["ftype"] == 460

    def test_case_insensitive_fields(self):
        feats = NhdHydrographyFetcher.parse_flowlines(FLOW_RAW)
        # The lowercased-key feature parsed its name + ftype.
        assert feats[2]["properties"]["name"] == "Tassajara"
        assert feats[2]["properties"]["ftype"] == 558

    def test_empty_input(self):
        assert NhdHydrographyFetcher.parse_flowlines({}) == []


class TestParseWaterbodies:
    def test_waterbody_normalization(self):
        feats = NhdHydrographyFetcher.parse_waterbodies(WB_RAW)
        assert len(feats) == 1  # geometryless dropped
        props = feats[0]["properties"]
        assert props["kind"] == "waterbody"
        assert props["source"] == "nhd"
        assert props["name"] == "Shadow Cliffs"
        assert props["area_sqkm"] == 0.42

    def test_empty_input(self):
        assert NhdHydrographyFetcher.parse_waterbodies({}) == []


class TestDegradation:
    def test_live_success_merges_layers_and_caches(self, monkeypatch, tmp_path):
        _patch(monkeypatch, _dispatch(FLOW_RAW, WB_RAW))
        cache = GISCache(tmp_path)
        f = NhdHydrographyFetcher(cache=cache)
        fc = f.fetch(DUBLIN)
        assert fc["type"] == "FeatureCollection"
        # 4 flowlines + 1 waterbody merged into one collection.
        kinds = sorted(x["properties"]["kind"] for x in fc["features"])
        assert kinds == ["artificial", "canal", "river", "stream", "waterbody"]
        # Cached — a subsequent offline call returns the same merged set.
        _patch(monkeypatch, _boom)
        again = f.fetch(DUBLIN)
        assert len(again["features"]) == len(fc["features"])

    def test_live_fail_falls_through_to_fixture(self, monkeypatch):
        _patch(monkeypatch, _boom)
        f = NhdHydrographyFetcher(cache=None)  # no cache -> straight to fixture
        fc = f.fetch(DUBLIN)
        assert fc.get("features"), "packaged Dublin NHD fixture should be non-empty"
        assert all(x["properties"]["source"] == "nhd" for x in fc["features"])

    def test_boulder_fixture_offline(self, monkeypatch):
        _patch(monkeypatch, _boom)
        f = NhdHydrographyFetcher(cache=None)
        fc = f.fetch(BOULDER)
        assert fc.get("features"), "packaged Boulder NHD fixture should be non-empty"

    def test_far_bbox_offline_is_empty_but_valid(self, monkeypatch):
        _patch(monkeypatch, _boom)
        f = NhdHydrographyFetcher(cache=None)
        fc = f.fetch(FAR)
        assert fc["type"] == "FeatureCollection"
        assert fc["features"] == []


class TestPackagedFixtures:
    def test_dublin_pack_has_streams_and_waterbodies(self):
        from importlib import resources

        raw = resources.files("tritium_lib.geo.gis.fixtures").joinpath(
            "nhd_hydro_ao.json"
        ).read_text(encoding="utf-8")
        data = json.loads(raw)
        assert data.get("fixture") is True
        kinds = {x["properties"]["kind"] for x in data["features"]}
        # A real hydro NETWORK: at least flowlines + waterbodies present.
        assert "waterbody" in kinds
        assert kinds & {"river", "stream", "canal", "artificial", "connector"}
