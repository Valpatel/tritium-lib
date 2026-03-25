# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for AddonGeoLayer dataclass."""

import pytest

from tritium_lib.sdk.geo_layer import AddonGeoLayer


class TestAddonGeoLayer:
    """Tests for AddonGeoLayer."""

    def test_required_fields(self):
        layer = AddonGeoLayer(
            layer_id="adsb-aircraft",
            addon_id="adsb",
            label="Aircraft",
            category="SDR",
            color="#ffaa00",
            geojson_endpoint="/api/adsb/aircraft.geojson",
        )
        assert layer.layer_id == "adsb-aircraft"
        assert layer.addon_id == "adsb"
        assert layer.label == "Aircraft"
        assert layer.category == "SDR"
        assert layer.color == "#ffaa00"
        assert layer.geojson_endpoint == "/api/adsb/aircraft.geojson"

    def test_default_refresh_interval(self):
        layer = AddonGeoLayer(
            layer_id="x", addon_id="y", label="L",
            category="C", color="#000", geojson_endpoint="/api/x",
        )
        assert layer.refresh_interval == 5

    def test_default_visible_by_default(self):
        layer = AddonGeoLayer(
            layer_id="x", addon_id="y", label="L",
            category="C", color="#000", geojson_endpoint="/api/x",
        )
        assert layer.visible_by_default is False

    def test_custom_refresh_and_visible(self):
        layer = AddonGeoLayer(
            layer_id="x", addon_id="y", label="L",
            category="C", color="#000", geojson_endpoint="/api/x",
            refresh_interval=30,
            visible_by_default=True,
        )
        assert layer.refresh_interval == 30
        assert layer.visible_by_default is True

    def test_to_dict(self):
        layer = AddonGeoLayer(
            layer_id="mesh-nodes",
            addon_id="meshtastic",
            label="Mesh Nodes",
            category="MESH",
            color="#05ffa1",
            geojson_endpoint="/api/mesh/nodes.geojson",
            refresh_interval=10,
            visible_by_default=True,
        )
        d = layer.to_dict()
        assert d == {
            "layer_id": "mesh-nodes",
            "addon_id": "meshtastic",
            "label": "Mesh Nodes",
            "category": "MESH",
            "color": "#05ffa1",
            "geojson_endpoint": "/api/mesh/nodes.geojson",
            "refresh_interval": 10,
            "visible_by_default": True,
        }

    def test_to_dict_contains_all_fields(self):
        layer = AddonGeoLayer(
            layer_id="x", addon_id="y", label="L",
            category="C", color="#000", geojson_endpoint="/api/x",
        )
        d = layer.to_dict()
        expected_keys = {
            "layer_id", "addon_id", "label", "category",
            "color", "geojson_endpoint", "refresh_interval",
            "visible_by_default",
        }
        assert set(d.keys()) == expected_keys
