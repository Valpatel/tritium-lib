# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.map_data — tactical map data module.

Pure data module for managing map layers, markers, and overlays that any
map frontend can consume.  Converts live tracking state (targets, geofence
zones, heatmap activity, patrol routes) into standard GeoJSON
FeatureCollections and MapLibre GL JS compatible style objects.

This module owns **data**, not rendering.  It produces dicts/JSON that
MapLibre, Leaflet, Cesium, ATAK, or any other consumer can ingest.

Key classes:

    MapLayer       Named collection of GeoJSON features (points, lines, polygons)
    MapMarker      Positioned marker with icon, label, tooltip, and properties
    MapOverlay     Coverage / heatmap / density overlay definition
    MapBounds      Geographic bounding box
    TacticalMapData  Aggregator that accepts tracker/engine state and builds layers

Key functions:

    to_geojson()           Export a layer as a GeoJSON FeatureCollection dict
    to_maplibre_style()    Export layers as a MapLibre GL JS style dict

Integration adapters:

    targets_to_layer()     TargetTracker targets -> MapLayer of markers
    zones_to_layer()       GeofenceEngine zones  -> MapLayer of polygons
    heatmap_to_overlay()   HeatmapEngine grid    -> MapOverlay
    routes_to_layer()      PatrolManager routes   -> MapLayer of lines
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LayerType(str, Enum):
    """Classification of what a map layer contains."""
    MARKERS = "markers"       # point features (targets, sensors, POIs)
    POLYGONS = "polygons"     # area features (zones, buildings, regions)
    LINES = "lines"           # line features (routes, trails, roads)
    HEATMAP = "heatmap"       # density / activity overlay
    MIXED = "mixed"           # heterogeneous feature types


class OverlayType(str, Enum):
    """Type of spatial overlay."""
    HEATMAP = "heatmap"
    COVERAGE = "coverage"
    DENSITY = "density"


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------

@dataclass
class MapBounds:
    """Geographic bounding box in WGS84 coordinates.

    Attributes:
        south: Southern latitude boundary (min lat).
        west: Western longitude boundary (min lng).
        north: Northern latitude boundary (max lat).
        east: Eastern longitude boundary (max lng).
    """

    south: float = 0.0
    west: float = 0.0
    north: float = 0.0
    east: float = 0.0
    _initialized: bool = field(default=False, repr=False)

    @property
    def center(self) -> tuple[float, float]:
        """Return (lat, lng) center of the bounds."""
        return ((self.south + self.north) / 2.0,
                (self.west + self.east) / 2.0)

    @property
    def is_valid(self) -> bool:
        """Check that bounds are non-degenerate."""
        return (self._initialized
                and self.south <= self.north
                and self.west <= self.east
                and -90.0 <= self.south <= 90.0
                and -90.0 <= self.north <= 90.0
                and -180.0 <= self.west <= 180.0
                and -180.0 <= self.east <= 180.0)

    def contains(self, lat: float, lng: float) -> bool:
        """Check if a point falls within these bounds."""
        return (self.south <= lat <= self.north
                and self.west <= lng <= self.east)

    def expand_to(self, lat: float, lng: float) -> None:
        """Expand bounds to include the given point."""
        if not self._initialized:
            self.south = lat
            self.north = lat
            self.west = lng
            self.east = lng
            self._initialized = True
        else:
            self.south = min(self.south, lat)
            self.north = max(self.north, lat)
            self.west = min(self.west, lng)
            self.east = max(self.east, lng)

    def to_dict(self) -> dict:
        return {
            "south": self.south,
            "west": self.west,
            "north": self.north,
            "east": self.east,
        }

    def to_bbox(self) -> list[float]:
        """Return [west, south, east, north] for GeoJSON/MapLibre."""
        return [self.west, self.south, self.east, self.north]

    @classmethod
    def from_points(cls, points: list[tuple[float, float]]) -> MapBounds:
        """Create bounds that enclose all (lat, lng) points."""
        if not points:
            return cls()
        lats = [p[0] for p in points]
        lngs = [p[1] for p in points]
        return cls(
            south=min(lats),
            west=min(lngs),
            north=max(lats),
            east=max(lngs),
            _initialized=True,
        )


@dataclass
class MapMarker:
    """A positioned marker on the tactical map.

    Attributes:
        marker_id: Unique identifier for this marker.
        lat: Latitude in decimal degrees (WGS84).
        lng: Longitude in decimal degrees (WGS84).
        icon: Icon identifier (e.g. "hostile", "friendly", "sensor").
        label: Short display label.
        tooltip: Longer hover text.
        color: Hex color string (e.g. "#ff2a6d").
        heading: Heading in degrees (0-360, 0=North).
        properties: Arbitrary metadata attached to this marker.
    """

    marker_id: str = ""
    lat: float = 0.0
    lng: float = 0.0
    icon: str = "default"
    label: str = ""
    tooltip: str = ""
    color: str = "#00f0ff"
    heading: float = 0.0
    properties: dict = field(default_factory=dict)

    def to_geojson_feature(self) -> dict:
        """Convert to a GeoJSON Feature dict."""
        props = dict(self.properties)
        props.update({
            "marker_id": self.marker_id,
            "icon": self.icon,
            "label": self.label,
            "tooltip": self.tooltip,
            "color": self.color,
            "heading": self.heading,
        })
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [self.lng, self.lat],
            },
            "properties": props,
        }


@dataclass
class MapOverlay:
    """A coverage / heatmap / density overlay for the tactical map.

    Attributes:
        overlay_id: Unique identifier.
        name: Display name.
        overlay_type: Type of overlay (heatmap, coverage, density).
        bounds: Geographic bounds of the overlay.
        grid: 2D intensity grid (row-major, [row][col]).
        resolution: Grid resolution (cells per side).
        max_value: Maximum cell value for normalization.
        color_stops: Color ramp as list of [value, color] pairs.
        opacity: Default opacity (0.0 - 1.0).
        properties: Arbitrary metadata.
    """

    overlay_id: str = ""
    name: str = ""
    overlay_type: OverlayType = OverlayType.HEATMAP
    bounds: MapBounds = field(default_factory=MapBounds)
    grid: list[list[float]] = field(default_factory=list)
    resolution: int = 50
    max_value: float = 0.0
    color_stops: list[list] = field(default_factory=lambda: [
        [0.0, "#000000"],
        [0.25, "#00f0ff"],
        [0.5, "#05ffa1"],
        [0.75, "#fcee0a"],
        [1.0, "#ff2a6d"],
    ])
    opacity: float = 0.6
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "overlay_id": self.overlay_id,
            "name": self.name,
            "overlay_type": self.overlay_type.value,
            "bounds": self.bounds.to_dict(),
            "grid": self.grid,
            "resolution": self.resolution,
            "max_value": self.max_value,
            "color_stops": self.color_stops,
            "opacity": self.opacity,
            "properties": self.properties,
        }


@dataclass
class MapLayer:
    """A named collection of GeoJSON features (points, lines, polygons).

    Layers are the primary unit of tactical map data.  Each layer has a
    type, color, visibility flag, and a list of GeoJSON Feature dicts.

    Attributes:
        layer_id: Unique identifier.
        name: Display name.
        layer_type: What kind of features this layer contains.
        color: Default hex color for layer rendering.
        visible: Whether the layer is visible by default.
        features: List of GeoJSON Feature dicts.
        bounds: Geographic bounds enclosing all features.
        properties: Arbitrary metadata.
    """

    layer_id: str = ""
    name: str = ""
    layer_type: LayerType = LayerType.MARKERS
    color: str = "#00f0ff"
    visible: bool = True
    features: list[dict] = field(default_factory=list)
    bounds: MapBounds = field(default_factory=MapBounds)
    properties: dict = field(default_factory=dict)

    @property
    def feature_count(self) -> int:
        return len(self.features)

    def add_marker(self, marker: MapMarker) -> None:
        """Add a MapMarker as a GeoJSON point feature."""
        self.features.append(marker.to_geojson_feature())
        self.bounds.expand_to(marker.lat, marker.lng)

    def add_polygon(
        self,
        polygon_id: str,
        coordinates: list[list[tuple[float, float]]],
        properties: dict | None = None,
    ) -> None:
        """Add a polygon feature.

        Args:
            polygon_id: Unique ID for this polygon.
            coordinates: GeoJSON polygon coordinates — list of rings,
                each ring is a list of (lng, lat) tuples.  The first ring
                is the exterior; subsequent rings are holes.
            properties: Feature properties dict.
        """
        props = dict(properties or {})
        props["polygon_id"] = polygon_id
        self.features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[c[0], c[1]] for c in ring] for ring in coordinates
                ],
            },
            "properties": props,
        })
        for ring in coordinates:
            for lng, lat in ring:
                self.bounds.expand_to(lat, lng)

    def add_line(
        self,
        line_id: str,
        coordinates: list[tuple[float, float]],
        properties: dict | None = None,
    ) -> None:
        """Add a LineString feature.

        Args:
            line_id: Unique ID for this line.
            coordinates: List of (lng, lat) tuples forming the line.
            properties: Feature properties dict.
        """
        props = dict(properties or {})
        props["line_id"] = line_id
        self.features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[c[0], c[1]] for c in coordinates],
            },
            "properties": props,
        })
        for lng, lat in coordinates:
            self.bounds.expand_to(lat, lng)

    def to_geojson(self) -> dict:
        """Export as a GeoJSON FeatureCollection dict."""
        return _to_geojson(self)

    def to_dict(self) -> dict:
        """Serialize layer metadata (without full feature list)."""
        return {
            "layer_id": self.layer_id,
            "name": self.name,
            "layer_type": self.layer_type.value,
            "color": self.color,
            "visible": self.visible,
            "feature_count": self.feature_count,
            "bounds": self.bounds.to_dict(),
            "properties": self.properties,
        }


# ---------------------------------------------------------------------------
# GeoJSON export
# ---------------------------------------------------------------------------

def to_geojson(layer: MapLayer) -> dict:
    """Export a MapLayer as a GeoJSON FeatureCollection dict.

    The returned dict conforms to RFC 7946 and can be consumed directly
    by MapLibre GL JS, Leaflet, QGIS, ATAK, or any GeoJSON reader.
    """
    return _to_geojson(layer)


def _to_geojson(layer: MapLayer) -> dict:
    """Internal GeoJSON builder."""
    fc: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": list(layer.features),
    }
    if layer.bounds.is_valid:
        fc["bbox"] = layer.bounds.to_bbox()
    # Attach layer metadata as a top-level property (non-standard but
    # widely supported by tactical map consumers).
    fc["properties"] = {
        "layer_id": layer.layer_id,
        "name": layer.name,
        "layer_type": layer.layer_type.value,
        "color": layer.color,
        "feature_count": layer.feature_count,
    }
    return fc


# ---------------------------------------------------------------------------
# MapLibre style export
# ---------------------------------------------------------------------------

# Cyberpunk palette
_ALLIANCE_COLORS = {
    "friendly": "#05ffa1",
    "hostile": "#ff2a6d",
    "unknown": "#00f0ff",
    "neutral": "#fcee0a",
}

_ZONE_TYPE_COLORS = {
    "restricted": "#ff2a6d",
    "monitored": "#fcee0a",
    "safe": "#05ffa1",
}


def to_maplibre_style(
    layers: list[MapLayer],
    overlays: list[MapOverlay] | None = None,
    base_style: dict | None = None,
) -> dict:
    """Export layers and overlays as a MapLibre GL JS compatible style dict.

    This produces a style object with sources and layers that MapLibre
    can render directly.  If *base_style* is provided it is used as the
    starting template; otherwise a minimal style skeleton is created.

    Args:
        layers: List of MapLayer objects to include.
        overlays: Optional list of MapOverlay objects.
        base_style: Optional base MapLibre style dict to extend.

    Returns:
        A MapLibre GL JS style dict with sources and layers.
    """
    style = dict(base_style) if base_style else {
        "version": 8,
        "name": "tritium-tactical",
        "sources": {},
        "layers": [],
    }

    # Ensure mutable copies of sources and layers
    sources = dict(style.get("sources", {}))
    style_layers = list(style.get("layers", []))

    for layer in layers:
        source_id = f"tritium-{layer.layer_id}"
        sources[source_id] = {
            "type": "geojson",
            "data": _to_geojson(layer),
        }

        if layer.layer_type == LayerType.MARKERS:
            style_layers.append({
                "id": f"{layer.layer_id}-circles",
                "type": "circle",
                "source": source_id,
                "paint": {
                    "circle-radius": 6,
                    "circle-color": ["coalesce",
                                     ["get", "color"],
                                     layer.color],
                    "circle-stroke-width": 1,
                    "circle-stroke-color": "#ffffff",
                    "circle-opacity": 0.9,
                },
                "layout": {
                    "visibility": "visible" if layer.visible else "none",
                },
            })
            style_layers.append({
                "id": f"{layer.layer_id}-labels",
                "type": "symbol",
                "source": source_id,
                "layout": {
                    "text-field": ["get", "label"],
                    "text-size": 11,
                    "text-offset": [0, 1.2],
                    "text-anchor": "top",
                    "visibility": "visible" if layer.visible else "none",
                },
                "paint": {
                    "text-color": layer.color,
                    "text-halo-color": "#000000",
                    "text-halo-width": 1,
                },
            })

        elif layer.layer_type == LayerType.POLYGONS:
            style_layers.append({
                "id": f"{layer.layer_id}-fill",
                "type": "fill",
                "source": source_id,
                "paint": {
                    "fill-color": ["coalesce",
                                   ["get", "color"],
                                   layer.color],
                    "fill-opacity": 0.2,
                },
                "layout": {
                    "visibility": "visible" if layer.visible else "none",
                },
            })
            style_layers.append({
                "id": f"{layer.layer_id}-outline",
                "type": "line",
                "source": source_id,
                "paint": {
                    "line-color": ["coalesce",
                                   ["get", "color"],
                                   layer.color],
                    "line-width": 2,
                },
                "layout": {
                    "visibility": "visible" if layer.visible else "none",
                },
            })

        elif layer.layer_type == LayerType.LINES:
            style_layers.append({
                "id": f"{layer.layer_id}-line",
                "type": "line",
                "source": source_id,
                "paint": {
                    "line-color": ["coalesce",
                                   ["get", "color"],
                                   layer.color],
                    "line-width": 3,
                    "line-opacity": 0.8,
                },
                "layout": {
                    "line-cap": "round",
                    "line-join": "round",
                    "visibility": "visible" if layer.visible else "none",
                },
            })

        elif layer.layer_type == LayerType.HEATMAP:
            style_layers.append({
                "id": f"{layer.layer_id}-heat",
                "type": "heatmap",
                "source": source_id,
                "paint": {
                    "heatmap-weight": ["coalesce",
                                       ["get", "weight"],
                                       1],
                    "heatmap-intensity": 1,
                    "heatmap-radius": 20,
                    "heatmap-opacity": 0.7,
                    "heatmap-color": [
                        "interpolate", ["linear"], ["heatmap-density"],
                        0, "rgba(0,0,0,0)",
                        0.25, "#00f0ff",
                        0.5, "#05ffa1",
                        0.75, "#fcee0a",
                        1.0, "#ff2a6d",
                    ],
                },
                "layout": {
                    "visibility": "visible" if layer.visible else "none",
                },
            })

        else:
            # Mixed — just add a circle layer as default
            style_layers.append({
                "id": f"{layer.layer_id}-default",
                "type": "circle",
                "source": source_id,
                "paint": {
                    "circle-radius": 5,
                    "circle-color": layer.color,
                },
                "layout": {
                    "visibility": "visible" if layer.visible else "none",
                },
            })

    style["sources"] = sources
    style["layers"] = style_layers
    return style


# ---------------------------------------------------------------------------
# Integration adapters — convert tracking state to map layers
# ---------------------------------------------------------------------------

def _alliance_color(alliance: str) -> str:
    """Map alliance string to cyberpunk color."""
    return _ALLIANCE_COLORS.get(alliance, "#00f0ff")


def _zone_color(zone_type: str) -> str:
    """Map zone type to cyberpunk color."""
    return _ZONE_TYPE_COLORS.get(zone_type, "#fcee0a")


def targets_to_layer(
    targets: list,
    layer_id: str = "targets",
    name: str = "Tracked Targets",
    geo_converter: Any = None,
) -> MapLayer:
    """Convert a list of TrackedTarget objects into a MapLayer of markers.

    Args:
        targets: List of TrackedTarget dataclass instances (from TargetTracker).
        layer_id: Layer identifier.
        name: Display name.
        geo_converter: Optional callable(x, y) -> {"lat", "lng", "alt"}.
            If None, tries tritium_lib.geo.local_to_latlng; falls back to
            (0, 0) if geo is not initialized.

    Returns:
        MapLayer with one point feature per target.
    """
    if geo_converter is None:
        try:
            from tritium_lib.geo import local_to_latlng
            geo_converter = lambda x, y: local_to_latlng(x, y)
        except Exception:
            geo_converter = lambda x, y: {"lat": 0.0, "lng": 0.0, "alt": 0.0}

    layer = MapLayer(
        layer_id=layer_id,
        name=name,
        layer_type=LayerType.MARKERS,
        color="#00f0ff",
    )

    for t in targets:
        try:
            geo = geo_converter(t.position[0], t.position[1])
            lat = geo["lat"]
            lng = geo["lng"]
        except Exception:
            lat, lng = 0.0, 0.0

        marker = MapMarker(
            marker_id=t.target_id,
            lat=lat,
            lng=lng,
            icon=t.asset_type,
            label=t.name,
            tooltip=f"{t.name} ({t.asset_type}) [{t.alliance}]",
            color=_alliance_color(t.alliance),
            heading=t.heading,
            properties={
                "target_id": t.target_id,
                "alliance": t.alliance,
                "asset_type": t.asset_type,
                "source": t.source,
                "status": t.status,
                "speed": t.speed,
                "battery": t.battery,
                "confidence": getattr(t, "effective_confidence",
                                      t.position_confidence),
                "signal_count": t.signal_count,
                "classification": t.classification,
                "threat_score": t.threat_score,
            },
        )
        layer.add_marker(marker)

    return layer


def zones_to_layer(
    zones: list,
    layer_id: str = "zones",
    name: str = "Geofence Zones",
    geo_converter: Any = None,
) -> MapLayer:
    """Convert a list of GeoZone objects into a MapLayer of polygons.

    Args:
        zones: List of GeoZone dataclass instances (from GeofenceEngine).
        layer_id: Layer identifier.
        name: Display name.
        geo_converter: Optional callable(x, y) -> {"lat", "lng", "alt"}.
            If None, tries tritium_lib.geo.local_to_latlng.

    Returns:
        MapLayer with one polygon feature per zone.
    """
    if geo_converter is None:
        try:
            from tritium_lib.geo import local_to_latlng
            geo_converter = lambda x, y: local_to_latlng(x, y)
        except Exception:
            geo_converter = lambda x, y: {"lat": 0.0, "lng": 0.0, "alt": 0.0}

    layer = MapLayer(
        layer_id=layer_id,
        name=name,
        layer_type=LayerType.POLYGONS,
        color="#fcee0a",
    )

    for zone in zones:
        # Convert local-meter polygon vertices to (lng, lat) for GeoJSON
        ring: list[tuple[float, float]] = []
        for vx, vy in zone.polygon:
            try:
                geo = geo_converter(vx, vy)
                ring.append((geo["lng"], geo["lat"]))
            except Exception:
                ring.append((0.0, 0.0))

        # Close the ring if not already closed
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])

        color = _zone_color(zone.zone_type)
        layer.add_polygon(
            polygon_id=zone.zone_id,
            coordinates=[ring],
            properties={
                "zone_id": zone.zone_id,
                "name": zone.name,
                "zone_type": zone.zone_type,
                "color": color,
                "enabled": zone.enabled,
                "alert_on_enter": zone.alert_on_enter,
                "alert_on_exit": zone.alert_on_exit,
            },
        )

    return layer


def heatmap_to_overlay(
    heatmap_data: dict,
    overlay_id: str = "activity",
    name: str = "Activity Heatmap",
    geo_converter: Any = None,
) -> MapOverlay:
    """Convert a HeatmapEngine.get_heatmap() result into a MapOverlay.

    Args:
        heatmap_data: Dict from HeatmapEngine.get_heatmap() with keys:
            grid, bounds, resolution, max_value, event_count.
        overlay_id: Overlay identifier.
        name: Display name.
        geo_converter: Optional callable(x, y) -> {"lat", "lng", "alt"}.
            If None, bounds are left at (0,0,0,0).

    Returns:
        MapOverlay with the heatmap grid and geographic bounds.
    """
    raw_bounds = heatmap_data.get("bounds", {})
    min_x = raw_bounds.get("min_x", 0.0)
    min_y = raw_bounds.get("min_y", 0.0)
    max_x = raw_bounds.get("max_x", 0.0)
    max_y = raw_bounds.get("max_y", 0.0)

    if geo_converter is None:
        try:
            from tritium_lib.geo import local_to_latlng
            geo_converter = lambda x, y: local_to_latlng(x, y)
        except Exception:
            geo_converter = lambda x, y: {"lat": 0.0, "lng": 0.0, "alt": 0.0}

    try:
        sw = geo_converter(min_x, min_y)
        ne = geo_converter(max_x, max_y)
        bounds = MapBounds(
            south=sw["lat"], west=sw["lng"],
            north=ne["lat"], east=ne["lng"],
        )
    except Exception:
        bounds = MapBounds()

    return MapOverlay(
        overlay_id=overlay_id,
        name=name,
        overlay_type=OverlayType.HEATMAP,
        bounds=bounds,
        grid=heatmap_data.get("grid", []),
        resolution=heatmap_data.get("resolution", 50),
        max_value=heatmap_data.get("max_value", 0.0),
        properties={
            "event_count": heatmap_data.get("event_count", 0),
            "layer": heatmap_data.get("layer", "all"),
        },
    )


def heatmap_to_point_layer(
    heatmap_data: dict,
    layer_id: str = "heatmap-points",
    name: str = "Activity Points",
    geo_converter: Any = None,
) -> MapLayer:
    """Convert a HeatmapEngine.get_heatmap() grid into a MapLayer of weighted points.

    MapLibre's native heatmap layer works best with point features that
    have a ``weight`` property.  This function converts the grid cells
    into point features so MapLibre can render them as a heatmap layer.

    Args:
        heatmap_data: Dict from HeatmapEngine.get_heatmap().
        layer_id: Layer identifier.
        name: Display name.
        geo_converter: Optional callable(x, y) -> {"lat", "lng", "alt"}.

    Returns:
        MapLayer (type HEATMAP) with weighted point features.
    """
    if geo_converter is None:
        try:
            from tritium_lib.geo import local_to_latlng
            geo_converter = lambda x, y: local_to_latlng(x, y)
        except Exception:
            geo_converter = lambda x, y: {"lat": 0.0, "lng": 0.0, "alt": 0.0}

    layer = MapLayer(
        layer_id=layer_id,
        name=name,
        layer_type=LayerType.HEATMAP,
        color="#ff2a6d",
    )

    grid = heatmap_data.get("grid", [])
    raw_bounds = heatmap_data.get("bounds", {})
    min_x = raw_bounds.get("min_x", 0.0)
    max_x = raw_bounds.get("max_x", 0.0)
    min_y = raw_bounds.get("min_y", 0.0)
    max_y = raw_bounds.get("max_y", 0.0)
    max_value = heatmap_data.get("max_value", 1.0)
    if max_value <= 0:
        max_value = 1.0

    resolution = len(grid)
    if resolution == 0:
        return layer

    range_x = max_x - min_x
    range_y = max_y - min_y
    if range_x <= 0 or range_y <= 0:
        return layer

    cols = len(grid[0]) if grid else 0

    for row_idx, row in enumerate(grid):
        for col_idx, value in enumerate(row):
            if value <= 0:
                continue
            # Center of cell in local coords
            cell_x = min_x + (col_idx + 0.5) / cols * range_x
            cell_y = min_y + (row_idx + 0.5) / resolution * range_y
            try:
                geo = geo_converter(cell_x, cell_y)
                lat = geo["lat"]
                lng = geo["lng"]
            except Exception:
                continue

            weight = value / max_value
            layer.features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lng, lat],
                },
                "properties": {
                    "weight": weight,
                    "raw_value": value,
                },
            })
            layer.bounds.expand_to(lat, lng)

    return layer


def routes_to_layer(
    routes: list,
    layer_id: str = "routes",
    name: str = "Patrol Routes",
    geo_converter: Any = None,
) -> MapLayer:
    """Convert a list of PatrolRoute objects into a MapLayer of lines.

    Args:
        routes: List of PatrolRoute dataclass instances (from PatrolManager).
        layer_id: Layer identifier.
        name: Display name.
        geo_converter: Optional callable(x, y) -> {"lat", "lng", "alt"}.

    Returns:
        MapLayer with one LineString feature per route.
    """
    if geo_converter is None:
        try:
            from tritium_lib.geo import local_to_latlng
            geo_converter = lambda x, y: local_to_latlng(x, y)
        except Exception:
            geo_converter = lambda x, y: {"lat": 0.0, "lng": 0.0, "alt": 0.0}

    layer = MapLayer(
        layer_id=layer_id,
        name=name,
        layer_type=LayerType.LINES,
        color="#05ffa1",
    )

    for route in routes:
        coords: list[tuple[float, float]] = []
        for wx, wy in route.waypoints:
            try:
                geo = geo_converter(wx, wy)
                coords.append((geo["lng"], geo["lat"]))
            except Exception:
                coords.append((0.0, 0.0))

        # If looping, close the route visually
        if route.loop and coords and coords[0] != coords[-1]:
            coords.append(coords[0])

        layer.add_line(
            line_id=route.route_id,
            coordinates=coords,
            properties={
                "route_id": route.route_id,
                "name": route.name,
                "loop": route.loop,
                "speed": route.speed,
                "color": "#05ffa1",
            },
        )

    return layer


# ---------------------------------------------------------------------------
# TacticalMapData — aggregator
# ---------------------------------------------------------------------------

class TacticalMapData:
    """Aggregator that holds all map layers, markers, and overlays.

    Collects layers from multiple sources and exports them together
    as GeoJSON or MapLibre style.

    Example::

        tmd = TacticalMapData()
        tmd.add_layer(targets_to_layer(tracker.get_all()))
        tmd.add_layer(zones_to_layer(geofence.list_zones()))
        tmd.add_overlay(heatmap_to_overlay(heatmap.get_heatmap()))
        tmd.add_layer(routes_to_layer(patrol_mgr.list_routes()))
        style = tmd.to_maplibre_style()
    """

    def __init__(self) -> None:
        self._layers: dict[str, MapLayer] = {}
        self._overlays: dict[str, MapOverlay] = {}

    def add_layer(self, layer: MapLayer) -> None:
        """Add or replace a map layer."""
        self._layers[layer.layer_id] = layer

    def remove_layer(self, layer_id: str) -> bool:
        """Remove a layer by ID. Returns True if found."""
        return self._layers.pop(layer_id, None) is not None

    def get_layer(self, layer_id: str) -> MapLayer | None:
        """Get a layer by ID."""
        return self._layers.get(layer_id)

    def list_layers(self) -> list[MapLayer]:
        """Return all layers."""
        return list(self._layers.values())

    def add_overlay(self, overlay: MapOverlay) -> None:
        """Add or replace an overlay."""
        self._overlays[overlay.overlay_id] = overlay

    def remove_overlay(self, overlay_id: str) -> bool:
        """Remove an overlay by ID. Returns True if found."""
        return self._overlays.pop(overlay_id, None) is not None

    def get_overlay(self, overlay_id: str) -> MapOverlay | None:
        """Get an overlay by ID."""
        return self._overlays.get(overlay_id)

    def list_overlays(self) -> list[MapOverlay]:
        """Return all overlays."""
        return list(self._overlays.values())

    @property
    def bounds(self) -> MapBounds:
        """Compute combined bounds of all layers."""
        all_points: list[tuple[float, float]] = []
        for layer in self._layers.values():
            if layer.bounds.is_valid:
                all_points.append((layer.bounds.south, layer.bounds.west))
                all_points.append((layer.bounds.north, layer.bounds.east))
        for overlay in self._overlays.values():
            if overlay.bounds.is_valid:
                all_points.append((overlay.bounds.south, overlay.bounds.west))
                all_points.append((overlay.bounds.north, overlay.bounds.east))
        return MapBounds.from_points(all_points) if all_points else MapBounds()

    def to_geojson(self, layer_id: str | None = None) -> dict:
        """Export as GeoJSON FeatureCollection(s).

        If *layer_id* is given, exports only that layer.  Otherwise
        exports all layers merged into one FeatureCollection.
        """
        if layer_id is not None:
            layer = self._layers.get(layer_id)
            if layer is None:
                return {"type": "FeatureCollection", "features": []}
            return _to_geojson(layer)

        # Merge all layers
        all_features: list[dict] = []
        for layer in self._layers.values():
            all_features.extend(layer.features)

        fc: dict[str, Any] = {
            "type": "FeatureCollection",
            "features": all_features,
        }
        bounds = self.bounds
        if bounds.is_valid:
            fc["bbox"] = bounds.to_bbox()
        return fc

    def to_maplibre_style(self, base_style: dict | None = None) -> dict:
        """Export all layers and overlays as a MapLibre style dict."""
        return to_maplibre_style(
            layers=list(self._layers.values()),
            overlays=list(self._overlays.values()),
            base_style=base_style,
        )

    def summary(self) -> dict:
        """Return a summary dict of all layers and overlays."""
        return {
            "layer_count": len(self._layers),
            "overlay_count": len(self._overlays),
            "total_features": sum(
                l.feature_count for l in self._layers.values()
            ),
            "layers": [l.to_dict() for l in self._layers.values()],
            "overlays": [o.to_dict() for o in self._overlays.values()],
            "bounds": self.bounds.to_dict(),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Core classes
    "MapLayer",
    "MapMarker",
    "MapOverlay",
    "MapBounds",
    "TacticalMapData",
    # Enums
    "LayerType",
    "OverlayType",
    # Export functions
    "to_geojson",
    "to_maplibre_style",
    # Integration adapters
    "targets_to_layer",
    "zones_to_layer",
    "heatmap_to_overlay",
    "heatmap_to_point_layer",
    "routes_to_layer",
]
