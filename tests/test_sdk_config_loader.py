# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for SDK AddonConfig — schema defaults, overrides, access patterns."""

import pytest

from tritium_lib.sdk.config_loader import AddonConfig


class TestAddonConfigDefaults:
    """Tests for schema-driven default values."""

    def test_empty_config(self):
        cfg = AddonConfig()
        assert cfg.to_dict() == {}

    def test_schema_with_dict_defaults(self):
        schema = {
            "host": {"default": "localhost"},
            "port": {"default": 8080},
            "debug": {"default": False},
        }
        cfg = AddonConfig(config_schema=schema)
        assert cfg.get("host") == "localhost"
        assert cfg.get("port") == 8080
        assert cfg.get("debug") is False

    def test_schema_with_plain_values(self):
        schema = {
            "name": "default_name",
            "count": 10,
        }
        cfg = AddonConfig(config_schema=schema)
        assert cfg.get("name") == "default_name"
        assert cfg.get("count") == 10

    def test_schema_mixed_dict_and_plain(self):
        schema = {
            "host": {"default": "0.0.0.0"},
            "enabled": True,
        }
        cfg = AddonConfig(config_schema=schema)
        assert cfg.get("host") == "0.0.0.0"
        assert cfg.get("enabled") is True


class TestAddonConfigOverrides:
    """Tests for runtime override behavior."""

    def test_overrides_replace_defaults(self):
        schema = {"host": {"default": "localhost"}, "port": {"default": 8080}}
        cfg = AddonConfig(config_schema=schema, overrides={"port": 9090})
        assert cfg.get("host") == "localhost"
        assert cfg.get("port") == 9090

    def test_overrides_add_new_keys(self):
        cfg = AddonConfig(config_schema={}, overrides={"custom_key": "value"})
        assert cfg.get("custom_key") == "value"

    def test_overrides_without_schema(self):
        cfg = AddonConfig(overrides={"a": 1, "b": 2})
        assert cfg.get("a") == 1
        assert cfg.get("b") == 2


class TestAddonConfigAccess:
    """Tests for various access patterns."""

    def test_get_missing_key_returns_default(self):
        cfg = AddonConfig()
        assert cfg.get("missing") is None
        assert cfg.get("missing", 42) == 42

    def test_attribute_access(self):
        cfg = AddonConfig(overrides={"name": "test"})
        assert cfg.name == "test"

    def test_attribute_access_missing_returns_none(self):
        cfg = AddonConfig()
        assert cfg.nonexistent is None

    def test_private_attribute_raises(self):
        cfg = AddonConfig()
        with pytest.raises(AttributeError):
            _ = cfg._private

    def test_to_dict_returns_copy(self):
        cfg = AddonConfig(overrides={"a": 1})
        d = cfg.to_dict()
        assert d == {"a": 1}
        d["a"] = 999
        assert cfg.get("a") == 1  # Original unchanged

    def test_repr(self):
        cfg = AddonConfig(overrides={"x": 1})
        r = repr(cfg)
        assert "AddonConfig" in r
        assert "x" in r


class TestAddonConfigEdgeCases:
    """Edge case tests."""

    def test_none_schema_and_overrides(self):
        cfg = AddonConfig(config_schema=None, overrides=None)
        assert cfg.to_dict() == {}

    def test_schema_dict_without_default_key(self):
        schema = {"setting": {"type": "string"}}  # no 'default' key
        cfg = AddonConfig(config_schema=schema)
        assert cfg.get("setting") is None

    def test_multiple_types_in_schema(self):
        schema = {
            "str_val": {"default": "hello"},
            "int_val": {"default": 42},
            "float_val": {"default": 3.14},
            "bool_val": {"default": True},
            "list_val": {"default": [1, 2, 3]},
            "dict_val": {"default": {"nested": True}},
        }
        cfg = AddonConfig(config_schema=schema)
        assert cfg.get("str_val") == "hello"
        assert cfg.get("int_val") == 42
        assert cfg.get("float_val") == pytest.approx(3.14)
        assert cfg.get("bool_val") is True
        assert cfg.get("list_val") == [1, 2, 3]
        assert cfg.get("dict_val") == {"nested": True}
