# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geo.gis.cache — GISCache disk JSON cache.

Covers put/get round-trip, 4dp bbox key coalescing, TTL expiry, corrupt-file
tolerance, and best-effort IO (never raises).  No network.
"""

import json
import os
import time

import pytest

from tritium_lib.geo.gis.cache import GISCache
from tritium_lib.geo.gis.models import GeoBBox

BOX = GeoBBox.from_string("-121.912,37.704,-121.880,37.728")


class TestKey:
    @pytest.mark.unit
    def test_key_is_filename_safe(self):
        c = GISCache("/tmp/does-not-matter")
        key = c.key("usgs", BOX, ncols=16, nrows=16)
        assert all(ch.isalnum() or ch in "._-" for ch in key)
        assert "usgs" in key
        assert "ncols-16" in key
        assert "nrows-16" in key

    @pytest.mark.unit
    def test_key_rounds_bbox_to_4dp(self):
        c = GISCache("/tmp/x")
        b1 = GeoBBox(west=-121.91201, south=37.70399, east=-121.88, north=37.728)
        b2 = GeoBBox(west=-121.91204, south=37.70401, east=-121.88, north=37.728)
        assert c.key("tiger", b1) == c.key("tiger", b2)

    @pytest.mark.unit
    def test_key_accepts_tuple(self):
        c = GISCache("/tmp/x")
        assert c.key("fema", BOX) == c.key(
            "fema", (BOX.west, BOX.south, BOX.east, BOX.north)
        )

    @pytest.mark.unit
    def test_key_params_order_independent(self):
        c = GISCache("/tmp/x")
        k1 = c.key("usgs", BOX, ncols=16, nrows=8)
        k2 = c.key("usgs", BOX, nrows=8, ncols=16)
        assert k1 == k2


class TestPutGet:
    @pytest.mark.unit
    def test_put_get_roundtrip(self, tmp_path):
        c = GISCache(tmp_path)
        key = c.key("tiger", BOX)
        payload = {"type": "FeatureCollection", "features": [{"a": 1}]}
        assert c.put(key, payload) is True
        assert c.get(key) == payload

    @pytest.mark.unit
    def test_get_miss_returns_none(self, tmp_path):
        c = GISCache(tmp_path)
        assert c.get(c.key("tiger", BOX)) is None

    @pytest.mark.unit
    def test_ttl_fresh_hit(self, tmp_path):
        c = GISCache(tmp_path)
        key = c.key("noaa", BOX)
        c.put(key, {"ok": True})
        assert c.get(key, max_age_s=3600) == {"ok": True}

    @pytest.mark.unit
    def test_ttl_expired_miss(self, tmp_path):
        c = GISCache(tmp_path)
        key = c.key("noaa", BOX)
        c.put(key, {"ok": True})
        # Backdate the file well beyond the max age.
        path = c._path(key)
        old = time.time() - 10_000
        os.utime(path, (old, old))
        assert c.get(key, max_age_s=60) is None
        # But unlimited age still returns it.
        assert c.get(key, max_age_s=None) == {"ok": True}

    @pytest.mark.unit
    def test_corrupt_file_returns_none(self, tmp_path):
        c = GISCache(tmp_path)
        key = c.key("fema", BOX)
        c.cache_dir.mkdir(parents=True, exist_ok=True)
        c._path(key).write_text("{ this is not valid json ]", encoding="utf-8")
        assert c.get(key) is None  # tolerated, no raise

    @pytest.mark.unit
    def test_put_non_serializable_returns_false(self, tmp_path):
        c = GISCache(tmp_path)
        key = c.key("usgs", BOX)
        assert c.put(key, {"bad": object()}) is False
        # And no partial/corrupt file is left that would later mislead a reader.
        assert c.get(key) is None

    @pytest.mark.unit
    def test_put_creates_directory(self, tmp_path):
        nested = tmp_path / "deep" / "gis_cache"
        c = GISCache(nested)
        assert not nested.exists()
        assert c.put(c.key("tiger", BOX), {"x": 1}) is True
        assert nested.is_dir()


class TestDefaultDir:
    @pytest.mark.unit
    def test_env_var_default(self, monkeypatch):
        monkeypatch.setenv("TRITIUM_GIS_CACHE", "/somewhere/custom")
        c = GISCache()
        assert str(c.cache_dir) == "/somewhere/custom"

    @pytest.mark.unit
    def test_fallback_default_dir(self, monkeypatch):
        monkeypatch.delenv("TRITIUM_GIS_CACHE", raising=False)
        c = GISCache()
        assert str(c.cache_dir) == "data/gis_cache"
