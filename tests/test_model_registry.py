# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for ModelRegistry — versioned ML model storage."""
import pytest

from tritium_lib.intelligence.model_registry import ModelRegistry


@pytest.fixture
def registry():
    r = ModelRegistry(":memory:")
    yield r
    r.close()


class TestSaveModel:
    def test_save_and_load(self, registry):
        data = b"model-bytes-here"
        result = registry.save_model("correlation", "1.0.0", data, {"accuracy": 0.95})
        assert result["name"] == "correlation"
        assert result["version"] == "1.0.0"
        assert result["size_bytes"] == len(data)
        assert result["id"] is not None

        loaded = registry.load_model("correlation", "1.0.0")
        assert loaded is not None
        assert loaded["data"] == data
        assert loaded["metadata"]["accuracy"] == 0.95

    def test_save_replaces_same_version(self, registry):
        registry.save_model("test", "1.0.0", b"old", {"v": 1})
        registry.save_model("test", "1.0.0", b"new", {"v": 2})

        loaded = registry.load_model("test", "1.0.0")
        assert loaded["data"] == b"new"
        assert loaded["metadata"]["v"] == 2

    def test_save_empty_name_raises(self, registry):
        with pytest.raises(ValueError, match="name"):
            registry.save_model("", "1.0.0", b"data")

    def test_save_empty_version_raises(self, registry):
        with pytest.raises(ValueError, match="version"):
            registry.save_model("test", "", b"data")

    def test_save_empty_data_raises(self, registry):
        with pytest.raises(ValueError, match="data"):
            registry.save_model("test", "1.0.0", b"")

    def test_save_no_metadata(self, registry):
        registry.save_model("test", "1.0.0", b"data")
        loaded = registry.load_model("test", "1.0.0")
        assert loaded["metadata"] == {}


class TestLoadModel:
    def test_load_nonexistent(self, registry):
        assert registry.load_model("nope", "1.0.0") is None

    def test_load_latest_when_no_version(self, registry):
        registry.save_model("test", "1.0.0", b"old")
        registry.save_model("test", "2.0.0", b"new")
        loaded = registry.load_model("test")
        assert loaded["version"] == "2.0.0"
        assert loaded["data"] == b"new"


class TestGetLatest:
    def test_get_latest(self, registry):
        registry.save_model("test", "1.0.0", b"v1")
        registry.save_model("test", "2.0.0", b"v2")
        latest = registry.get_latest("test")
        assert latest["version"] == "2.0.0"

    def test_get_latest_nonexistent(self, registry):
        assert registry.get_latest("nope") is None


class TestListModels:
    def test_list_empty(self, registry):
        assert registry.list_models() == []

    def test_list_all(self, registry):
        registry.save_model("a", "1.0.0", b"data")
        registry.save_model("b", "1.0.0", b"data")
        models = registry.list_models()
        assert len(models) == 2
        # No data blob in listings
        for m in models:
            assert "data" not in m

    def test_list_by_name(self, registry):
        registry.save_model("a", "1.0.0", b"data")
        registry.save_model("a", "2.0.0", b"data")
        registry.save_model("b", "1.0.0", b"data")
        models = registry.list_models(name="a")
        assert len(models) == 2
        assert all(m["name"] == "a" for m in models)

    def test_list_with_limit(self, registry):
        for i in range(5):
            registry.save_model("test", f"{i}.0.0", b"data")
        models = registry.list_models(limit=3)
        assert len(models) == 3


class TestDeleteModel:
    def test_delete_existing(self, registry):
        registry.save_model("test", "1.0.0", b"data")
        assert registry.delete_model("test", "1.0.0") is True
        assert registry.load_model("test", "1.0.0") is None

    def test_delete_nonexistent(self, registry):
        assert registry.delete_model("nope", "1.0.0") is False


class TestGetStats:
    def test_empty_stats(self, registry):
        stats = registry.get_stats()
        assert stats["total_models"] == 0
        assert stats["unique_names"] == 0
        assert stats["total_size_bytes"] == 0

    def test_stats_after_saves(self, registry):
        registry.save_model("a", "1.0.0", b"12345")
        registry.save_model("a", "2.0.0", b"67890")
        registry.save_model("b", "1.0.0", b"abc")
        stats = registry.get_stats()
        assert stats["total_models"] == 3
        assert stats["unique_names"] == 2
        assert stats["total_size_bytes"] == 13
