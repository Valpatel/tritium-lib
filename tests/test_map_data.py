# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.map_data — tactical map data module."""

from __future__ import annotations

import pytest
from dataclasses import dataclass

from tritium_lib.map_data import (
    MapBounds,
    MapLayer,
    MapMarker,
    MapOverlay,
    TacticalMapData,
    LayerType,
    OverlayType,
    to_geojson,
    to_maplibre_style,
    targets_to_layer,
    zones_to_layer,
    heatmap_to_overlay,
    heatmap_to_point_layer,
    routes_to_layer,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight stubs that mimic real tracker/engine objects
# ---------------------------------------------------------------------------

def _geo_converter(x: float, y: float) -> dict:
    """Stub geo converter: treat x as lng offset, y as lat offset from origin."""
    return {"lat": 40.0 + y * 0.00001, "lng": -74.0 + x * 0.00001, "alt": 0.0}


@dataclass
class _FakeTarget:
    target_id: str = "t1"
    name: str = "Alpha"
    alliance: str = "friendly"
    asset_type: str = "rover"
    position: tuple = (10.0, 20.0)
    heading: float = 90.0
    speed: float = 1.5
    battery: float = 0.8
    source: str = "simulation"
    status: str = "active"
    position_confidence: float = 0.9
    signal_count: int = 5
    classification: str = "rover"
    threat_score: float = 0.0

    @property
    def effective_confidence(self) -> float:
        return self.position_confidence


@dataclass
class _FakeZone:
    zone_id: str = "z1"
    name: str = "Alpha Zone"
    polygon: list = None
    zone_type: str = "restricted"
    enabled: bool = True
    alert_on_enter: bool = True
    alert_on_exit: bool = False

    def __post_init__(self):
        if self.polygon is None:
            self.polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


@dataclass
class _FakeRoute:
    route_id: str = "r1"
    name: str = "Perimeter"
    waypoints: list = None
    loop: bool = True
    speed: float = 2.0

    def __post_init__(self):
        if self.waypoints is None:
            self.waypoints = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


# ===================================================================
# MapBounds
# ===================================================================

class TestMapBounds:
    def test_default_bounds(self):
        b = MapBounds()
        assert b.south == 0.0
        assert b.north == 0.0

    def test_center(self):
        b = MapBounds(south=10.0, west=20.0, north=30.0, east=40.0)
        assert b.center == (20.0, 30.0)

    def test_is_valid(self):
        b = MapBounds(south=10.0, west=-74.0, north=20.0, east=-73.0, _initialized=True)
        assert b.is_valid

    def test_is_valid_uninitialized(self):
        b = MapBounds()
        assert not b.is_valid

    def test_is_invalid_inverted(self):
        b = MapBounds(south=20.0, west=-74.0, north=10.0, east=-73.0, _initialized=True)
        assert not b.is_valid

    def test_contains(self):
        b = MapBounds(south=10.0, west=-74.0, north=20.0, east=-73.0, _initialized=True)
        assert b.contains(15.0, -73.5)
        assert not b.contains(5.0, -73.5)

    def test_expand_to(self):
        b = MapBounds(south=10.0, west=-74.0, north=20.0, east=-73.0, _initialized=True)
        b.expand_to(5.0, -75.0)
        assert b.south == 5.0
        assert b.west == -75.0

    def test_expand_to_from_uninitialized(self):
        b = MapBounds()
        b.expand_to(40.0, -74.0)
        assert b.south == 40.0
        assert b.north == 40.0
        assert b.west == -74.0
        assert b.east == -74.0
        assert b._initialized

    def test_to_bbox(self):
        b = MapBounds(south=10.0, west=20.0, north=30.0, east=40.0)
        assert b.to_bbox() == [20.0, 10.0, 40.0, 30.0]

    def test_from_points(self):
        pts = [(10.0, 20.0), (30.0, 40.0), (5.0, 15.0)]
        b = MapBounds.from_points(pts)
        assert b.south == 5.0
        assert b.north == 30.0
        assert b.west == 15.0
        assert b.east == 40.0

    def test_from_points_empty(self):
        b = MapBounds.from_points([])
        assert b.south == 0.0

    def test_to_dict(self):
        b = MapBounds(south=1.0, west=2.0, north=3.0, east=4.0)
        d = b.to_dict()
        assert d == {"south": 1.0, "west": 2.0, "north": 3.0, "east": 4.0}


# ===================================================================
# MapMarker
# ===================================================================

class TestMapMarker:
    def test_default_marker(self):
        m = MapMarker(marker_id="m1", lat=40.0, lng=-74.0)
        assert m.icon == "default"
        assert m.color == "#00f0ff"

    def test_to_geojson_feature(self):
        m = MapMarker(
            marker_id="m1",
            lat=40.0,
            lng=-74.0,
            icon="tank",
            label="Alpha",
            tooltip="Alpha tank",
            color="#ff2a6d",
            heading=180.0,
            properties={"speed": 5.0},
        )
        f = m.to_geojson_feature()
        assert f["type"] == "Feature"
        assert f["geometry"]["type"] == "Point"
        assert f["geometry"]["coordinates"] == [-74.0, 40.0]
        assert f["properties"]["marker_id"] == "m1"
        assert f["properties"]["icon"] == "tank"
        assert f["properties"]["label"] == "Alpha"
        assert f["properties"]["speed"] == 5.0
        assert f["properties"]["heading"] == 180.0

    def test_geojson_feature_coordinate_order(self):
        """GeoJSON spec: coordinates are [longitude, latitude]."""
        m = MapMarker(lat=41.5, lng=-73.2)
        f = m.to_geojson_feature()
        coords = f["geometry"]["coordinates"]
        assert coords[0] == -73.2  # longitude first
        assert coords[1] == 41.5   # latitude second


# ===================================================================
# MapOverlay
# ===================================================================

class TestMapOverlay:
    def test_default_overlay(self):
        o = MapOverlay(overlay_id="o1", name="Test")
        assert o.overlay_type == OverlayType.HEATMAP
        assert o.opacity == 0.6
        assert len(o.color_stops) == 5

    def test_to_dict(self):
        o = MapOverlay(
            overlay_id="o1",
            name="Activity",
            grid=[[1.0, 2.0], [3.0, 4.0]],
            resolution=2,
            max_value=4.0,
        )
        d = o.to_dict()
        assert d["overlay_id"] == "o1"
        assert d["overlay_type"] == "heatmap"
        assert d["grid"] == [[1.0, 2.0], [3.0, 4.0]]
        assert d["max_value"] == 4.0


# ===================================================================
# MapLayer
# ===================================================================

class TestMapLayer:
    def test_empty_layer(self):
        layer = MapLayer(layer_id="test", name="Test")
        assert layer.feature_count == 0

    def test_add_marker(self):
        layer = MapLayer(layer_id="t", name="T", layer_type=LayerType.MARKERS)
        m = MapMarker(marker_id="m1", lat=40.0, lng=-74.0, label="A")
        layer.add_marker(m)
        assert layer.feature_count == 1
        assert layer.features[0]["properties"]["label"] == "A"

    def test_add_polygon(self):
        layer = MapLayer(layer_id="p", name="P", layer_type=LayerType.POLYGONS)
        ring = [(-74.0, 40.0), (-73.0, 40.0), (-73.0, 41.0), (-74.0, 41.0), (-74.0, 40.0)]
        layer.add_polygon("poly1", [ring], {"name": "Zone A"})
        assert layer.feature_count == 1
        f = layer.features[0]
        assert f["geometry"]["type"] == "Polygon"
        assert f["properties"]["polygon_id"] == "poly1"
        assert f["properties"]["name"] == "Zone A"

    def test_add_line(self):
        layer = MapLayer(layer_id="l", name="L", layer_type=LayerType.LINES)
        coords = [(-74.0, 40.0), (-73.0, 41.0)]
        layer.add_line("line1", coords, {"route": "alpha"})
        assert layer.feature_count == 1
        f = layer.features[0]
        assert f["geometry"]["type"] == "LineString"
        assert f["properties"]["line_id"] == "line1"

    def test_bounds_expand_on_add(self):
        layer = MapLayer(layer_id="t", name="T")
        m1 = MapMarker(lat=40.0, lng=-74.0)
        m2 = MapMarker(lat=41.0, lng=-73.0)
        layer.add_marker(m1)
        layer.add_marker(m2)
        assert layer.bounds.south == 40.0
        assert layer.bounds.north == 41.0
        assert layer.bounds.west == -74.0
        assert layer.bounds.east == -73.0

    def test_to_geojson_method(self):
        layer = MapLayer(layer_id="t", name="Test")
        layer.add_marker(MapMarker(lat=40.0, lng=-74.0, label="X"))
        fc = layer.to_geojson()
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 1

    def test_to_dict_metadata(self):
        layer = MapLayer(layer_id="abc", name="ABC", color="#ff0000")
        d = layer.to_dict()
        assert d["layer_id"] == "abc"
        assert d["name"] == "ABC"
        assert d["color"] == "#ff0000"
        assert d["feature_count"] == 0


# ===================================================================
# to_geojson (module-level)
# ===================================================================

class TestToGeoJSON:
    def test_basic_export(self):
        layer = MapLayer(layer_id="g", name="G")
        layer.add_marker(MapMarker(lat=40.0, lng=-74.0, label="P"))
        fc = to_geojson(layer)
        assert fc["type"] == "FeatureCollection"
        assert fc["properties"]["layer_id"] == "g"
        assert len(fc["features"]) == 1

    def test_bbox_present_when_valid(self):
        layer = MapLayer(layer_id="g", name="G")
        layer.add_marker(MapMarker(lat=40.0, lng=-74.0))
        layer.add_marker(MapMarker(lat=41.0, lng=-73.0))
        fc = to_geojson(layer)
        assert "bbox" in fc
        assert fc["bbox"] == [-74.0, 40.0, -73.0, 41.0]

    def test_empty_layer_geojson(self):
        layer = MapLayer(layer_id="e", name="E")
        fc = to_geojson(layer)
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 0


# ===================================================================
# to_maplibre_style
# ===================================================================

class TestToMaplibreStyle:
    def test_basic_style(self):
        layer = MapLayer(layer_id="t", name="T", layer_type=LayerType.MARKERS)
        layer.add_marker(MapMarker(lat=40.0, lng=-74.0))
        style = to_maplibre_style([layer])
        assert style["version"] == 8
        assert "tritium-t" in style["sources"]
        # Should have circle + label layers
        layer_ids = [sl["id"] for sl in style["layers"]]
        assert "t-circles" in layer_ids
        assert "t-labels" in layer_ids

    def test_polygon_style_layers(self):
        layer = MapLayer(layer_id="z", name="Z", layer_type=LayerType.POLYGONS)
        ring = [(-74.0, 40.0), (-73.0, 40.0), (-73.0, 41.0), (-74.0, 41.0), (-74.0, 40.0)]
        layer.add_polygon("p1", [ring])
        style = to_maplibre_style([layer])
        layer_ids = [sl["id"] for sl in style["layers"]]
        assert "z-fill" in layer_ids
        assert "z-outline" in layer_ids

    def test_line_style_layer(self):
        layer = MapLayer(layer_id="r", name="R", layer_type=LayerType.LINES)
        layer.add_line("l1", [(-74.0, 40.0), (-73.0, 41.0)])
        style = to_maplibre_style([layer])
        layer_ids = [sl["id"] for sl in style["layers"]]
        assert "r-line" in layer_ids

    def test_heatmap_style_layer(self):
        layer = MapLayer(layer_id="h", name="H", layer_type=LayerType.HEATMAP)
        layer.features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-74.0, 40.0]},
            "properties": {"weight": 0.5},
        })
        style = to_maplibre_style([layer])
        layer_ids = [sl["id"] for sl in style["layers"]]
        assert "h-heat" in layer_ids

    def test_visibility_hidden(self):
        layer = MapLayer(layer_id="v", name="V", visible=False, layer_type=LayerType.MARKERS)
        style = to_maplibre_style([layer])
        for sl in style["layers"]:
            assert sl["layout"]["visibility"] == "none"

    def test_base_style_preserved(self):
        base = {
            "version": 8,
            "name": "custom",
            "sources": {"existing": {"type": "vector"}},
            "layers": [{"id": "bg", "type": "background"}],
        }
        layer = MapLayer(layer_id="t", name="T", layer_type=LayerType.MARKERS)
        style = to_maplibre_style([layer], base_style=base)
        assert "existing" in style["sources"]
        assert "tritium-t" in style["sources"]
        ids = [sl["id"] for sl in style["layers"]]
        assert "bg" in ids
        assert "t-circles" in ids

    def test_multiple_layers(self):
        l1 = MapLayer(layer_id="a", name="A", layer_type=LayerType.MARKERS)
        l2 = MapLayer(layer_id="b", name="B", layer_type=LayerType.LINES)
        l2.add_line("ln1", [(-74.0, 40.0), (-73.0, 41.0)])
        style = to_maplibre_style([l1, l2])
        assert "tritium-a" in style["sources"]
        assert "tritium-b" in style["sources"]


# ===================================================================
# Integration adapters
# ===================================================================

class TestTargetsToLayer:
    def test_basic_conversion(self):
        targets = [_FakeTarget()]
        layer = targets_to_layer(targets, geo_converter=_geo_converter)
        assert layer.layer_id == "targets"
        assert layer.layer_type == LayerType.MARKERS
        assert layer.feature_count == 1
        f = layer.features[0]
        assert f["properties"]["target_id"] == "t1"
        assert f["properties"]["alliance"] == "friendly"

    def test_alliance_colors(self):
        friendly = _FakeTarget(target_id="f1", alliance="friendly")
        hostile = _FakeTarget(target_id="h1", alliance="hostile")
        unknown = _FakeTarget(target_id="u1", alliance="unknown")
        layer = targets_to_layer(
            [friendly, hostile, unknown], geo_converter=_geo_converter,
        )
        colors = {
            f["properties"]["target_id"]: f["properties"]["color"]
            for f in layer.features
        }
        assert colors["f1"] == "#05ffa1"
        assert colors["h1"] == "#ff2a6d"
        assert colors["u1"] == "#00f0ff"

    def test_geo_conversion_applied(self):
        t = _FakeTarget(position=(100.0, 200.0))
        layer = targets_to_layer([t], geo_converter=_geo_converter)
        f = layer.features[0]
        coords = f["geometry"]["coordinates"]
        # lng = -74.0 + 100 * 0.00001 = -73.999
        assert abs(coords[0] - (-73.999)) < 1e-6
        # lat = 40.0 + 200 * 0.00001 = 40.002
        assert abs(coords[1] - 40.002) < 1e-6

    def test_empty_targets(self):
        layer = targets_to_layer([], geo_converter=_geo_converter)
        assert layer.feature_count == 0

    def test_multiple_targets(self):
        targets = [
            _FakeTarget(target_id=f"t{i}", name=f"T{i}")
            for i in range(5)
        ]
        layer = targets_to_layer(targets, geo_converter=_geo_converter)
        assert layer.feature_count == 5


class TestZonesToLayer:
    def test_basic_conversion(self):
        zones = [_FakeZone()]
        layer = zones_to_layer(zones, geo_converter=_geo_converter)
        assert layer.layer_type == LayerType.POLYGONS
        assert layer.feature_count == 1
        f = layer.features[0]
        assert f["geometry"]["type"] == "Polygon"
        assert f["properties"]["zone_id"] == "z1"
        assert f["properties"]["zone_type"] == "restricted"

    def test_zone_type_colors(self):
        z_restricted = _FakeZone(zone_id="r", zone_type="restricted")
        z_monitored = _FakeZone(zone_id="m", zone_type="monitored")
        z_safe = _FakeZone(zone_id="s", zone_type="safe")
        layer = zones_to_layer(
            [z_restricted, z_monitored, z_safe],
            geo_converter=_geo_converter,
        )
        colors = {
            f["properties"]["zone_id"]: f["properties"]["color"]
            for f in layer.features
        }
        assert colors["r"] == "#ff2a6d"
        assert colors["m"] == "#fcee0a"
        assert colors["s"] == "#05ffa1"

    def test_polygon_ring_closed(self):
        """Zone polygon ring must be closed (first == last point)."""
        zone = _FakeZone(polygon=[(0, 0), (10, 0), (10, 10), (0, 10)])
        layer = zones_to_layer([zone], geo_converter=_geo_converter)
        ring = layer.features[0]["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1]

    def test_empty_zones(self):
        layer = zones_to_layer([], geo_converter=_geo_converter)
        assert layer.feature_count == 0


class TestHeatmapToOverlay:
    def _make_heatmap_data(self) -> dict:
        return {
            "grid": [[1.0, 2.0], [3.0, 4.0]],
            "bounds": {"min_x": 0.0, "max_x": 100.0, "min_y": 0.0, "max_y": 100.0},
            "resolution": 2,
            "max_value": 4.0,
            "event_count": 10,
            "layer": "all",
        }

    def test_basic_conversion(self):
        data = self._make_heatmap_data()
        overlay = heatmap_to_overlay(data, geo_converter=_geo_converter)
        assert overlay.overlay_id == "activity"
        assert overlay.overlay_type == OverlayType.HEATMAP
        assert overlay.max_value == 4.0
        assert overlay.grid == [[1.0, 2.0], [3.0, 4.0]]

    def test_bounds_converted(self):
        data = self._make_heatmap_data()
        overlay = heatmap_to_overlay(data, geo_converter=_geo_converter)
        assert overlay.bounds.south != 0.0 or overlay.bounds.west != 0.0

    def test_properties_preserved(self):
        data = self._make_heatmap_data()
        overlay = heatmap_to_overlay(data, geo_converter=_geo_converter)
        assert overlay.properties["event_count"] == 10
        assert overlay.properties["layer"] == "all"


class TestHeatmapToPointLayer:
    def _make_heatmap_data(self) -> dict:
        return {
            "grid": [[0.0, 5.0], [3.0, 0.0]],
            "bounds": {"min_x": 0.0, "max_x": 100.0, "min_y": 0.0, "max_y": 100.0},
            "resolution": 2,
            "max_value": 5.0,
            "event_count": 2,
        }

    def test_basic_conversion(self):
        data = self._make_heatmap_data()
        layer = heatmap_to_point_layer(data, geo_converter=_geo_converter)
        assert layer.layer_type == LayerType.HEATMAP
        # Only cells with value > 0 become points
        assert layer.feature_count == 2

    def test_weight_normalized(self):
        data = self._make_heatmap_data()
        layer = heatmap_to_point_layer(data, geo_converter=_geo_converter)
        weights = sorted(
            f["properties"]["weight"] for f in layer.features
        )
        assert weights[0] == pytest.approx(3.0 / 5.0, abs=1e-6)
        assert weights[1] == pytest.approx(5.0 / 5.0, abs=1e-6)

    def test_empty_grid(self):
        data = {
            "grid": [],
            "bounds": {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0},
            "resolution": 0,
            "max_value": 0,
        }
        layer = heatmap_to_point_layer(data, geo_converter=_geo_converter)
        assert layer.feature_count == 0


class TestRoutesToLayer:
    def test_basic_conversion(self):
        routes = [_FakeRoute()]
        layer = routes_to_layer(routes, geo_converter=_geo_converter)
        assert layer.layer_type == LayerType.LINES
        assert layer.feature_count == 1
        f = layer.features[0]
        assert f["geometry"]["type"] == "LineString"
        assert f["properties"]["route_id"] == "r1"

    def test_looping_route_closed(self):
        route = _FakeRoute(loop=True)
        layer = routes_to_layer([route], geo_converter=_geo_converter)
        coords = layer.features[0]["geometry"]["coordinates"]
        assert coords[0] == coords[-1]

    def test_non_looping_route_open(self):
        route = _FakeRoute(loop=False)
        layer = routes_to_layer([route], geo_converter=_geo_converter)
        coords = layer.features[0]["geometry"]["coordinates"]
        # 4 waypoints, non-looping → 4 coordinates (no closing)
        assert len(coords) == 4

    def test_empty_routes(self):
        layer = routes_to_layer([], geo_converter=_geo_converter)
        assert layer.feature_count == 0


# ===================================================================
# TacticalMapData
# ===================================================================

class TestTacticalMapData:
    def test_add_and_list_layers(self):
        tmd = TacticalMapData()
        l1 = MapLayer(layer_id="a", name="A")
        l2 = MapLayer(layer_id="b", name="B")
        tmd.add_layer(l1)
        tmd.add_layer(l2)
        assert len(tmd.list_layers()) == 2

    def test_get_layer(self):
        tmd = TacticalMapData()
        tmd.add_layer(MapLayer(layer_id="x", name="X"))
        assert tmd.get_layer("x") is not None
        assert tmd.get_layer("missing") is None

    def test_remove_layer(self):
        tmd = TacticalMapData()
        tmd.add_layer(MapLayer(layer_id="x", name="X"))
        assert tmd.remove_layer("x")
        assert not tmd.remove_layer("x")
        assert len(tmd.list_layers()) == 0

    def test_replace_layer(self):
        tmd = TacticalMapData()
        tmd.add_layer(MapLayer(layer_id="x", name="Old"))
        tmd.add_layer(MapLayer(layer_id="x", name="New"))
        assert len(tmd.list_layers()) == 1
        assert tmd.get_layer("x").name == "New"

    def test_add_and_list_overlays(self):
        tmd = TacticalMapData()
        tmd.add_overlay(MapOverlay(overlay_id="o1", name="O1"))
        assert len(tmd.list_overlays()) == 1

    def test_remove_overlay(self):
        tmd = TacticalMapData()
        tmd.add_overlay(MapOverlay(overlay_id="o1", name="O1"))
        assert tmd.remove_overlay("o1")
        assert not tmd.remove_overlay("o1")

    def test_to_geojson_all_layers(self):
        tmd = TacticalMapData()
        l1 = MapLayer(layer_id="a", name="A")
        l1.add_marker(MapMarker(lat=40.0, lng=-74.0))
        l2 = MapLayer(layer_id="b", name="B")
        l2.add_marker(MapMarker(lat=41.0, lng=-73.0))
        tmd.add_layer(l1)
        tmd.add_layer(l2)
        fc = tmd.to_geojson()
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 2

    def test_to_geojson_single_layer(self):
        tmd = TacticalMapData()
        l1 = MapLayer(layer_id="a", name="A")
        l1.add_marker(MapMarker(lat=40.0, lng=-74.0))
        tmd.add_layer(l1)
        fc = tmd.to_geojson(layer_id="a")
        assert len(fc["features"]) == 1

    def test_to_geojson_missing_layer(self):
        tmd = TacticalMapData()
        fc = tmd.to_geojson(layer_id="nope")
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 0

    def test_to_maplibre_style(self):
        tmd = TacticalMapData()
        tmd.add_layer(MapLayer(layer_id="t", name="T", layer_type=LayerType.MARKERS))
        style = tmd.to_maplibre_style()
        assert style["version"] == 8
        assert "tritium-t" in style["sources"]

    def test_summary(self):
        tmd = TacticalMapData()
        l = MapLayer(layer_id="x", name="X")
        l.add_marker(MapMarker(lat=40.0, lng=-74.0))
        tmd.add_layer(l)
        tmd.add_overlay(MapOverlay(overlay_id="o", name="O"))
        s = tmd.summary()
        assert s["layer_count"] == 1
        assert s["overlay_count"] == 1
        assert s["total_features"] == 1

    def test_combined_bounds(self):
        tmd = TacticalMapData()
        l1 = MapLayer(layer_id="a", name="A")
        l1.add_marker(MapMarker(lat=40.0, lng=-74.0))
        l2 = MapLayer(layer_id="b", name="B")
        l2.add_marker(MapMarker(lat=42.0, lng=-72.0))
        tmd.add_layer(l1)
        tmd.add_layer(l2)
        b = tmd.bounds
        assert b.south == 40.0
        assert b.north == 42.0
        assert b.west == -74.0
        assert b.east == -72.0

    def test_full_pipeline(self):
        """End-to-end: targets + zones + routes -> GeoJSON + style."""
        targets = [
            _FakeTarget(target_id="f1", alliance="friendly", position=(10, 20)),
            _FakeTarget(target_id="h1", alliance="hostile", position=(50, 60)),
        ]
        zones = [_FakeZone()]
        routes = [_FakeRoute()]

        tmd = TacticalMapData()
        tmd.add_layer(targets_to_layer(targets, geo_converter=_geo_converter))
        tmd.add_layer(zones_to_layer(zones, geo_converter=_geo_converter))
        tmd.add_layer(routes_to_layer(routes, geo_converter=_geo_converter))

        # GeoJSON export
        fc = tmd.to_geojson()
        assert fc["type"] == "FeatureCollection"
        # 2 targets + 1 zone + 1 route = 4 features
        assert len(fc["features"]) == 4

        # MapLibre style export
        style = tmd.to_maplibre_style()
        assert len(style["sources"]) == 3
        assert len(style["layers"]) > 0

        # Summary
        s = tmd.summary()
        assert s["layer_count"] == 3
        assert s["total_features"] == 4
