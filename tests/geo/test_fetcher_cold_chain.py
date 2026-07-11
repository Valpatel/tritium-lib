# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Pin the COLD-path fetch contract behind the chokepoints instant answer.

Phase-A DEFECT #2: the chokepoints derived layer answers COLD queries
instantly (cache / containing-entry clip / packaged fixture — never a live
round trip) and upgrades later via a background warm.  That only works if the
``fetchers.py`` / ``cache.py`` contract holds exactly:

    * ``fetch(bbox, *, allow_live=True, prefer_cache_s=None)`` — kwargs pinned.
    * ``allow_live=False`` NEVER touches the network (cache -> containing-entry
      clip -> fixture only).
    * ``prefer_cache_s`` short-circuits on a fresh cache entry; a stale entry
      goes live first.
    * ``GISCache.find_containing`` serves the smallest cached CONTAINING window
      (params-bearing / unparseable / corrupt entries skipped, best-effort).
    * error-as-200 guard: an upstream ``{"error": ...}`` 200 body degrades to
      fixture and writes NO cache entry (the cache-poisoning bug).
    * the default live-first chain is unchanged.

NETWORK IS NEVER TOUCHED: ``urllib.request.urlopen`` is monkeypatched in every
test that exercises ``fetch`` — with a *recording* fake wherever the assertion
is "no live attempt happened", because the chain's broad degrade-on-Exception
would otherwise swallow a plain AssertionError from a hit.
"""

import json
import os
import time

import pytest

from tritium_lib.geo.gis import (
    GISCache,
    NhdHydrographyFetcher,
    TigerRoadsFetcher,
)
from tritium_lib.geo.gis.fetchers import _raise_on_error_payload
from tritium_lib.geo.gis.models import GeoBBox

DUBLIN = GeoBBox.from_string("-121.912,37.704,-121.880,37.728")
#: The Boulder demo AO — the packaged boulder fixture packs cover this window.
BOULDER = GeoBBox.from_string("-105.30,39.98,-105.26,40.02")
#: A window strictly inside BOULDER (exercises the containing-entry clip).
BOULDER_NARROW = GeoBBox.from_string("-105.29,39.99,-105.27,40.01")
#: A window strictly containing BOULDER (a cached AO warm).
BOULDER_WIDE = GeoBBox.from_string("-105.32,39.96,-105.24,40.04")

#: The exact upstream error-as-200 body from the defect (ArcGIS throttling).
ERROR_BODY = {"error": {"code": 500, "message": "throttled"}}


# ---------------------------------------------------------------------------
# Fake HTTP transport (recording variants — see module docstring)
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


def _deny(calls: list):
    """A urlopen that RECORDS the hit then raises.

    Recording matters: the degradation chain catches *every* Exception on the
    live stage, so a raise alone cannot prove "never touched the network" —
    the post-fetch ``calls == []`` assertion can.
    """

    def fake_urlopen(req, timeout=None):
        calls.append(getattr(req, "full_url", "") or "")
        raise AssertionError("network hit")

    return fake_urlopen


def _counting_ok(payload, calls: list):
    """A urlopen serving a VALID body while recording every hit."""
    body = json.dumps(payload).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        calls.append(getattr(req, "full_url", "") or "")
        return _FakeResp(body)

    return fake_urlopen


def _dispatch(flow_payload, wb_payload):
    """Fake urlopen for the two-layer NHD live fetch (flowline vs waterbody)."""
    flow_body = json.dumps(flow_payload).encode("utf-8")
    wb_body = json.dumps(wb_payload).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        url = getattr(req, "full_url", "") or ""
        if "/9/query" in url:
            return _FakeResp(wb_body)
        return _FakeResp(flow_body)

    return fake_urlopen


def _patch(monkeypatch, fn):
    monkeypatch.setattr("urllib.request.urlopen", fn)


# ---------------------------------------------------------------------------
# Small GeoJSON builders (coordinates inside/outside BOULDER_NARROW)
# ---------------------------------------------------------------------------
def _pt(lon, lat, source, name):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {"source": source, "kind": "test", "name": name},
    }


def _fc(*features):
    return {"type": "FeatureCollection", "features": list(features)}


#: A cached AO-warm payload over BOULDER_WIDE: one feature inside the narrow
#: window, one inside the wide window only (must be clipped away).
def _wide_payload(source):
    return _fc(
        _pt(-105.280, 40.005, source, "inside-narrow"),
        _pt(-105.312, 40.030, source, "wide-only"),
    )


TIGER_LIVE_RAW = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature",
         "geometry": {"type": "LineString",
                      "coordinates": [[-105.28, 40.00], [-105.27, 40.01]]},
         "properties": {"BASENAME": "Canyon", "NAME": "Canyon Blvd",
                        "MTFCC": "S1400"}},
    ],
}

NHD_FLOW_RAW = {
    "features": [
        {"geometry": {"type": "LineString",
                      "coordinates": [[-105.29, 40.00], [-105.27, 40.00]]},
         "properties": {"GNIS_NAME": "Boulder Creek", "FType": 460}},
    ]
}
NHD_WB_RAW = {
    "features": [
        {"geometry": {"type": "Polygon",
                      "coordinates": [[[-105.29, 40.00], [-105.28, 40.00],
                                       [-105.28, 40.01], [-105.29, 40.00]]]},
         "properties": {"GNIS_NAME": "Wonderland Lake", "FType": 390,
                        "AreaSqKm": 0.1}},
    ]
}


# ---------------------------------------------------------------------------
# GISCache.find_containing — the containing-entry stage's lookup
# ---------------------------------------------------------------------------
class TestFindContaining:
    @pytest.mark.unit
    def test_smallest_containing_entry_wins(self, tmp_path):
        c = GISCache(tmp_path)
        huge = GeoBBox.from_string("-106.0,39.0,-104.0,41.0")
        c.put(c.key("tiger", huge), {"who": "huge"})
        c.put(c.key("tiger", BOULDER_WIDE), {"who": "modest"})
        got = c.find_containing("tiger", BOULDER_NARROW)
        assert got == {"who": "modest"}  # least over-fetch for the clip

    @pytest.mark.unit
    def test_non_containing_entries_ignored(self, tmp_path):
        c = GISCache(tmp_path)
        # Overlaps BOULDER_NARROW but does not CONTAIN it (east edge short).
        partial = GeoBBox.from_string("-105.32,39.96,-105.28,40.04")
        c.put(c.key("tiger", partial), {"who": "partial"})
        # Disjoint window far away.
        far = GeoBBox.from_string("-121.92,37.70,-121.88,37.73")
        c.put(c.key("tiger", far), {"who": "far"})
        assert c.find_containing("tiger", BOULDER_NARROW) is None

    @pytest.mark.unit
    def test_other_source_entries_ignored(self, tmp_path):
        c = GISCache(tmp_path)
        c.put(c.key("nhd", BOULDER_WIDE), {"who": "nhd"})
        assert c.find_containing("tiger", BOULDER_NARROW) is None
        assert c.find_containing("nhd", BOULDER_NARROW) == {"who": "nhd"}

    @pytest.mark.unit
    def test_params_bearing_and_unparseable_keys_skipped(self, tmp_path):
        c = GISCache(tmp_path)
        c.cache_dir.mkdir(parents=True, exist_ok=True)
        # A grid-style key with extra params (6 filename parts) — even though
        # its bbox contains the query, the key shape disqualifies it.
        params_key = c.key("nlcd", BOULDER_WIDE, ncols=32, nrows=32)
        assert "ncols-32" in params_key and "nrows-32" in params_key
        c.put(params_key, {"who": "params"})
        assert c.find_containing("nlcd", BOULDER_NARROW) is None
        # A foreign 4-part name whose parts are not floats.
        (c.cache_dir / "tiger_a_b_c_d.json").write_text(
            json.dumps({"who": "unparseable"}), encoding="utf-8"
        )
        assert c.find_containing("tiger", BOULDER_NARROW) is None

    @pytest.mark.unit
    def test_empty_and_missing_dir_return_none(self, tmp_path):
        assert GISCache(tmp_path).find_containing("tiger", BOULDER_NARROW) is None
        missing = GISCache(tmp_path / "does" / "not" / "exist")
        assert missing.find_containing("tiger", BOULDER_NARROW) is None

    @pytest.mark.unit
    def test_corrupt_best_entry_returns_none_not_raise(self, tmp_path):
        # The winning (smallest containing) entry is chosen from the FILENAME;
        # if its body is corrupt the read fails best-effort -> None, no raise.
        c = GISCache(tmp_path)
        c.cache_dir.mkdir(parents=True, exist_ok=True)
        key = c.key("tiger", BOULDER_WIDE)
        (c.cache_dir / f"{key}.json").write_text(
            "{ not valid json ]", encoding="utf-8"
        )
        assert c.find_containing("tiger", BOULDER_NARROW) is None


# ---------------------------------------------------------------------------
# fetch(allow_live=False) — the COLD path never touches the network
# ---------------------------------------------------------------------------
class TestAllowLiveFalse:
    @pytest.mark.unit
    def test_exact_cache_hit_served(self, monkeypatch, tmp_path):
        calls = []
        _patch(monkeypatch, _deny(calls))
        cache = GISCache(tmp_path)
        f = TigerRoadsFetcher(cache=cache)
        cached = _fc(_pt(-105.28, 40.0, "tiger", "cached"))
        cache.put(cache.key("tiger", BOULDER), cached)
        assert f.fetch(BOULDER, allow_live=False) == cached
        assert calls == []  # zero network hits

    @pytest.mark.unit
    def test_containing_entry_clipped_for_narrower_bbox(self, monkeypatch, tmp_path):
        calls = []
        _patch(monkeypatch, _deny(calls))
        cache = GISCache(tmp_path)
        f = TigerRoadsFetcher(cache=cache)
        # A previously warmed WIDER window; the narrow query has no exact key.
        cache.put(cache.key("tiger", BOULDER_WIDE), _wide_payload("tiger"))
        fc = f.fetch(BOULDER_NARROW, allow_live=False)
        names = [x["properties"]["name"] for x in fc["features"]]
        assert names == ["inside-narrow"]      # wide-only feature clipped away
        assert "fixture" not in fc             # cached live data, not a fixture
        assert calls == []

    @pytest.mark.unit
    def test_fixture_fallback_when_no_cache(self, monkeypatch, tmp_path):
        calls = []
        _patch(monkeypatch, _deny(calls))
        f = TigerRoadsFetcher(cache=GISCache(tmp_path))  # empty cache
        fc = f.fetch(BOULDER, allow_live=False)
        assert fc.get("fixture") is True
        assert len(fc["features"]) >= 1        # Boulder pack is non-empty
        assert all(x["properties"]["source"] == "tiger" for x in fc["features"])
        assert calls == []

    @pytest.mark.unit
    def test_nhd_same_contract(self, monkeypatch, tmp_path):
        # The NHD fetcher overrides fetch() (two-layer live step) but must keep
        # the identical cold contract: cache -> containing clip -> fixture,
        # zero network hits.
        calls = []
        _patch(monkeypatch, _deny(calls))
        cache = GISCache(tmp_path)
        f = NhdHydrographyFetcher(cache=cache)
        # (a) fixture fallback on an empty cache.
        fc = f.fetch(BOULDER, allow_live=False)
        assert fc.get("fixture") is True
        assert len(fc["features"]) >= 1
        assert all(x["properties"]["source"] == "nhd" for x in fc["features"])
        # (b) exact cache hit.
        cached = _fc(_pt(-105.28, 40.0, "nhd", "cached"))
        cache.put(cache.key("nhd", BOULDER), cached)
        assert f.fetch(BOULDER, allow_live=False) == cached
        # (c) containing-entry clip for a narrower window — its own cache dir,
        # otherwise (b)'s exact-BOULDER entry (also containing, and smaller
        # than BOULDER_WIDE) would legitimately win the smallest-containing pick.
        clip_cache = GISCache(tmp_path / "clip")
        clip_cache.put(clip_cache.key("nhd", BOULDER_WIDE), _wide_payload("nhd"))
        f_clip = NhdHydrographyFetcher(cache=clip_cache)
        fc = f_clip.fetch(BOULDER_NARROW, allow_live=False)
        assert [x["properties"]["name"] for x in fc["features"]] == ["inside-narrow"]
        assert calls == []                     # across all three stages


# ---------------------------------------------------------------------------
# prefer_cache_s — fresh entry short-circuits; stale goes live first
# ---------------------------------------------------------------------------
class TestPreferCacheS:
    @pytest.mark.unit
    def test_fresh_entry_short_circuits_live(self, monkeypatch, tmp_path):
        calls = []
        # A VALID live answer stands by — if the short-circuit failed, fetch
        # would return the parsed live data and the call count would betray it.
        _patch(monkeypatch, _counting_ok(TIGER_LIVE_RAW, calls))
        cache = GISCache(tmp_path)
        f = TigerRoadsFetcher(cache=cache)
        cached = _fc(_pt(-105.28, 40.0, "tiger", "fresh-cache"))
        cache.put(cache.key("tiger", BOULDER), cached)
        assert f.fetch(BOULDER, prefer_cache_s=3600) == cached
        assert calls == []                     # urlopen never called

    @pytest.mark.unit
    def test_stale_entry_goes_live_first(self, monkeypatch, tmp_path):
        calls = []
        _patch(monkeypatch, _counting_ok(TIGER_LIVE_RAW, calls))
        cache = GISCache(tmp_path)
        f = TigerRoadsFetcher(cache=cache)
        key = cache.key("tiger", BOULDER)
        cache.put(key, _fc(_pt(-105.28, 40.0, "tiger", "stale-cache")))
        # Age the entry past the freshness window.
        old = time.time() - 7200
        os.utime(cache._path(key), (old, old))
        fc = f.fetch(BOULDER, prefer_cache_s=3600)
        assert len(calls) >= 1                 # live attempted
        assert [x["properties"]["name"] for x in fc["features"]] == ["Canyon Blvd"]
        # And the fresh live answer replaced the stale entry.
        assert cache.get(key) == fc


# ---------------------------------------------------------------------------
# Error-as-200 guard — degrade to fixture, NEVER cache the lie
# ---------------------------------------------------------------------------
class TestErrorAs200:
    @pytest.mark.unit
    def test_raise_on_error_payload_is_pure(self):
        with pytest.raises(RuntimeError, match="throttled"):
            _raise_on_error_payload(ERROR_BODY, "tiger")
        # Valid / empty / non-dict bodies pass through untouched.
        _raise_on_error_payload({"type": "FeatureCollection", "features": []}, "tiger")
        _raise_on_error_payload({}, "tiger")
        _raise_on_error_payload(None, "tiger")

    @pytest.mark.unit
    def test_tiger_error_body_degrades_and_caches_nothing(self, monkeypatch, tmp_path):
        _patch(monkeypatch, _ok(ERROR_BODY))
        cache = GISCache(tmp_path)
        f = TigerRoadsFetcher(cache=cache)
        fc = f.fetch(BOULDER)
        assert fc.get("fixture") is True       # degraded, not an empty lie
        assert len(fc["features"]) >= 1
        # THE poisoning bug: no cache entry may be written for the error body.
        assert cache.get(cache.key("tiger", BOULDER)) is None
        assert list(tmp_path.glob("*.json")) == []

    @pytest.mark.unit
    def test_nhd_error_body_degrades_and_caches_nothing(self, monkeypatch, tmp_path):
        _patch(monkeypatch, _dispatch(ERROR_BODY, ERROR_BODY))
        cache = GISCache(tmp_path)
        f = NhdHydrographyFetcher(cache=cache)
        fc = f.fetch(BOULDER)
        assert fc.get("fixture") is True
        assert cache.get(cache.key("nhd", BOULDER)) is None
        assert list(tmp_path.glob("*.json")) == []

    @pytest.mark.unit
    def test_nhd_second_subquery_error_degrades_whole_fetch(self, monkeypatch, tmp_path):
        # Flowline (#3) succeeds, waterbody (#9) returns the error body — the
        # WHOLE fetch must degrade; a partial merge must never be cached.
        _patch(monkeypatch, _dispatch(NHD_FLOW_RAW, ERROR_BODY))
        cache = GISCache(tmp_path)
        f = NhdHydrographyFetcher(cache=cache)
        fc = f.fetch(BOULDER)
        assert fc.get("fixture") is True       # not the half-merged live set
        assert cache.get(cache.key("nhd", BOULDER)) is None
        assert list(tmp_path.glob("*.json")) == []


# ---------------------------------------------------------------------------
# Live-first contract preserved (default kwargs)
# ---------------------------------------------------------------------------
class TestLiveFirstPreserved:
    @pytest.mark.unit
    def test_default_fetch_parses_caches_returns_live(self, monkeypatch, tmp_path):
        _patch(monkeypatch, _ok(TIGER_LIVE_RAW))
        cache = GISCache(tmp_path)
        f = TigerRoadsFetcher(cache=cache)
        fc = f.fetch(BOULDER)                  # default allow_live/prefer_cache_s
        assert [x["properties"]["name"] for x in fc["features"]] == ["Canyon Blvd"]
        assert fc["features"][0]["properties"]["kind"] == "S1400"
        assert "fixture" not in fc
        assert cache.get(cache.key("tiger", BOULDER)) == fc

    @pytest.mark.unit
    def test_nhd_default_fetch_merges_and_caches(self, monkeypatch, tmp_path):
        _patch(monkeypatch, _dispatch(NHD_FLOW_RAW, NHD_WB_RAW))
        cache = GISCache(tmp_path)
        f = NhdHydrographyFetcher(cache=cache)
        fc = f.fetch(BOULDER)
        kinds = sorted(x["properties"]["kind"] for x in fc["features"])
        assert kinds == ["river", "waterbody"]
        assert cache.get(cache.key("nhd", BOULDER)) == fc
