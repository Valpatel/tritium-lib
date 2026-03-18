# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Tests for AddonGeoLayer dataclass and AddonBase.get_geojson_layers()."""

import pytest

from tritium_lib.sdk import AddonBase, AddonGeoLayer


class TestAddonGeoLayer:
    """AddonGeoLayer creation and serialization."""

    def test_create_with_defaults(self):
        layer = AddonGeoLayer(
            layer_id="test-layer",
            addon_id="test-addon",
            label="Test Layer",
            category="TEST",
            color="#ff0000",
            geojson_endpoint="/api/addons/test/geojson",
        )
        assert layer.layer_id == "test-layer"
        assert layer.addon_id == "test-addon"
        assert layer.label == "Test Layer"
        assert layer.category == "TEST"
        assert layer.color == "#ff0000"
        assert layer.geojson_endpoint == "/api/addons/test/geojson"
        assert layer.refresh_interval == 5
        assert layer.visible_by_default is False

    def test_create_with_custom_values(self):
        layer = AddonGeoLayer(
            layer_id="adsb-aircraft",
            addon_id="hackrf",
            label="ADS-B Aircraft",
            category="SDR",
            color="#ffaa00",
            geojson_endpoint="/api/addons/hackrf/geojson/adsb",
            refresh_interval=3,
            visible_by_default=True,
        )
        assert layer.refresh_interval == 3
        assert layer.visible_by_default is True

    def test_to_dict(self):
        layer = AddonGeoLayer(
            layer_id="mesh-nodes",
            addon_id="meshtastic",
            label="Mesh Nodes",
            category="MESH",
            color="#00d4aa",
            geojson_endpoint="/api/addons/meshtastic/geojson/nodes",
            refresh_interval=5,
            visible_by_default=True,
        )
        d = layer.to_dict()
        assert isinstance(d, dict)
        assert d["layer_id"] == "mesh-nodes"
        assert d["addon_id"] == "meshtastic"
        assert d["label"] == "Mesh Nodes"
        assert d["category"] == "MESH"
        assert d["color"] == "#00d4aa"
        assert d["geojson_endpoint"] == "/api/addons/meshtastic/geojson/nodes"
        assert d["refresh_interval"] == 5
        assert d["visible_by_default"] is True

    def test_to_dict_all_keys_present(self):
        layer = AddonGeoLayer(
            layer_id="x", addon_id="y", label="Z",
            category="C", color="#000", geojson_endpoint="/e",
        )
        d = layer.to_dict()
        expected_keys = {
            "layer_id", "addon_id", "label", "category",
            "color", "geojson_endpoint", "refresh_interval",
            "visible_by_default",
        }
        assert set(d.keys()) == expected_keys


class TestAddonBaseGeoJsonLayers:
    """AddonBase.get_geojson_layers() default behavior."""

    def test_default_returns_empty_list(self):
        addon = AddonBase()
        result = addon.get_geojson_layers()
        assert result == []
        assert isinstance(result, list)

    def test_subclass_can_override(self):
        class MyAddon(AddonBase):
            def get_geojson_layers(self):
                return [
                    AddonGeoLayer(
                        layer_id="my-layer",
                        addon_id="my-addon",
                        label="My Layer",
                        category="CUSTOM",
                        color="#123456",
                        geojson_endpoint="/api/addons/my/geojson",
                    ),
                ]

        addon = MyAddon()
        layers = addon.get_geojson_layers()
        assert len(layers) == 1
        assert layers[0].layer_id == "my-layer"
        assert isinstance(layers[0], AddonGeoLayer)


class TestAddonGeoLayerImport:
    """Verify AddonGeoLayer is importable from the SDK package."""

    def test_import_from_sdk(self):
        from tritium_lib.sdk import AddonGeoLayer as AGL
        assert AGL is AddonGeoLayer
