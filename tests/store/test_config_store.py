# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for ConfigStore — persistent key-value system configuration."""

import pytest
from tritium_lib.store.config_store import ConfigStore


@pytest.fixture
def store():
    s = ConfigStore(":memory:")
    yield s
    s.close()


class TestConfigStoreBasic:
    def test_set_and_get(self, store):
        store.set("map", "center_lat", "39.7392")
        assert store.get("map", "center_lat") == "39.7392"

    def test_get_default(self, store):
        assert store.get("map", "nonexistent") is None
        assert store.get("map", "nonexistent", default="0") == "0"

    def test_overwrite(self, store):
        store.set("map", "zoom", "10")
        store.set("map", "zoom", "15")
        assert store.get("map", "zoom") == "15"

    def test_different_namespaces(self, store):
        store.set("map", "key", "map_val")
        store.set("demo", "key", "demo_val")
        assert store.get("map", "key") == "map_val"
        assert store.get("demo", "key") == "demo_val"

    def test_delete(self, store):
        store.set("ns", "k", "v")
        assert store.delete("ns", "k") is True
        assert store.get("ns", "k") is None
        assert store.delete("ns", "k") is False

    def test_clear_namespace(self, store):
        store.set("ns", "a", "1")
        store.set("ns", "b", "2")
        store.set("other", "c", "3")
        deleted = store.clear_namespace("ns")
        assert deleted == 2
        assert store.get("ns", "a") is None
        assert store.get("other", "c") == "3"

    def test_get_namespace(self, store):
        store.set("map", "lat", "39")
        store.set("map", "lng", "-104")
        store.set("other", "x", "y")
        ns = store.get_namespace("map")
        assert ns == {"lat": "39", "lng": "-104"}

    def test_list_namespaces(self, store):
        store.set("alpha", "k", "v")
        store.set("beta", "k", "v")
        store.set("alpha", "k2", "v2")
        ns = store.list_namespaces()
        assert ns == ["alpha", "beta"]

    def test_count(self, store):
        assert store.count() == 0
        store.set("a", "k1", "v")
        store.set("a", "k2", "v")
        store.set("b", "k1", "v")
        assert store.count() == 3
        assert store.count("a") == 2
        assert store.count("b") == 1

    def test_set_many(self, store):
        store.set_many("batch", {"x": "1", "y": "2", "z": "3"})
        assert store.count("batch") == 3
        assert store.get("batch", "y") == "2"


class TestConfigStoreJson:
    def test_set_get_json_dict(self, store):
        store.set_json("map", "center", {"lat": 39.7, "lng": -104.9})
        result = store.get_json("map", "center")
        assert result == {"lat": 39.7, "lng": -104.9}

    def test_set_get_json_list(self, store):
        store.set_json("demo", "targets", [1, 2, 3])
        assert store.get_json("demo", "targets") == [1, 2, 3]

    def test_set_get_json_bool(self, store):
        store.set_json("notify", "enabled", True)
        assert store.get_json("notify", "enabled") is True

    def test_get_json_default(self, store):
        assert store.get_json("ns", "missing") is None
        assert store.get_json("ns", "missing", default=42) == 42

    def test_get_json_invalid(self, store):
        store.set("ns", "bad", "not-json{")
        assert store.get_json("ns", "bad", default="fallback") == "fallback"


class TestConfigStoreImport:
    def test_importable_from_package(self):
        from tritium_lib.store import ConfigStore as CS
        assert CS is ConfigStore
