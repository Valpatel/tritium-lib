# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""GeoJSON layer protocol for addon map integration.

Addons declare GeoJSON layers via AddonGeoLayer dataclasses.
Each layer points to an API endpoint that returns a GeoJSON
FeatureCollection, enabling the tactical map to consume
addon-provided spatial data with configurable refresh intervals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AddonGeoLayer:
    """A GeoJSON layer contributed by an addon to the tactical map.

    Attributes:
        layer_id: Unique layer identifier (e.g., "hackrf-adsb").
        addon_id: ID of the addon providing this layer.
        label: Human-readable display label.
        category: Layer category for grouping (e.g., "SDR", "MESH").
        color: Hex color for the layer swatch (e.g., "#ffaa00").
        geojson_endpoint: API path returning a GeoJSON FeatureCollection.
        refresh_interval: Seconds between automatic refreshes (default 5).
        visible_by_default: Whether the layer is shown on initial load.
    """

    layer_id: str
    addon_id: str
    label: str
    category: str
    color: str
    geojson_endpoint: str
    refresh_interval: int = 5
    visible_by_default: bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON responses."""
        return {
            "layer_id": self.layer_id,
            "addon_id": self.addon_id,
            "label": self.label,
            "category": self.category,
            "color": self.color,
            "geojson_endpoint": self.geojson_endpoint,
            "refresh_interval": self.refresh_interval,
            "visible_by_default": self.visible_by_default,
        }
