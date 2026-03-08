# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""GIS models — tile coordinates, map layers, regions, and offline packages.

Supports OSM slippy map tile coordinates, MBTiles metadata, and offline
map package manifests for field-disconnected operation.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TileCoord(BaseModel):
    """OSM slippy map tile coordinate (x, y, zoom).

    See: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    """
    x: int
    y: int
    zoom: int

    @property
    def quadkey(self) -> str:
        """Convert to Bing Maps quadkey string."""
        digits = []
        for i in range(self.zoom, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if self.x & mask:
                digit += 1
            if self.y & mask:
                digit += 2
            digits.append(str(digit))
        return "".join(digits)

    @property
    def url_path(self) -> str:
        """OSM-style URL path segment: {zoom}/{x}/{y}."""
        return f"{self.zoom}/{self.x}/{self.y}"


class MapLayerType(str, Enum):
    """Types of map layers."""
    RASTER = "raster"
    VECTOR = "vector"
    ELEVATION = "elevation"
    OVERLAY = "overlay"


class MapLayer(BaseModel):
    """A map tile layer — raster, vector, or overlay."""
    id: str
    name: str
    layer_type: MapLayerType = MapLayerType.RASTER
    url_template: str = ""  # e.g. "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    min_zoom: int = 0
    max_zoom: int = 19
    attribution: str = ""
    tile_size: int = 256
    format: str = "png"  # png, jpg, pbf, webp


class MapRegion(BaseModel):
    """A geographic bounding box defining a map region.

    Bounds are (south, west, north, east) in WGS84 degrees.
    """
    id: str
    name: str
    south: float
    west: float
    north: float
    east: float
    description: str = ""

    @property
    def center_lat(self) -> float:
        return (self.south + self.north) / 2.0

    @property
    def center_lon(self) -> float:
        return (self.west + self.east) / 2.0

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (south, west, north, east) tuple."""
        return (self.south, self.west, self.north, self.east)


class TilePackage(BaseModel):
    """An offline map tile package manifest.

    Represents an MBTiles or directory-based tile cache for offline use.
    """
    id: str
    name: str
    region: MapRegion
    layers: list[str] = Field(default_factory=list)  # layer IDs
    min_zoom: int = 0
    max_zoom: int = 14
    tile_count: int = 0
    size_bytes: int = 0
    format: str = "mbtiles"  # mbtiles, directory, tar
    sha256: str = ""
    created_at: Optional[datetime] = None
    # MBTiles metadata fields
    mbtiles_type: str = "baselayer"  # baselayer, overlay
    mbtiles_version: str = "1.3"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> TileCoord:
    """Convert WGS84 lat/lon to OSM slippy map tile coordinate.

    Uses the standard Mercator projection formula from the OSM wiki.
    """
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    # Clamp to valid range
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return TileCoord(x=x, y=y, zoom=zoom)


def tile_to_lat_lon(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Convert OSM tile coordinate to WGS84 lat/lon of the tile's NW corner."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return (lat, lon)


def tiles_in_bounds(
    bounds: tuple[float, float, float, float],
    zoom: int,
) -> list[TileCoord]:
    """Return all tile coordinates that intersect a bounding box at a given zoom.

    Args:
        bounds: (south, west, north, east) in WGS84 degrees.
        zoom: Zoom level.

    Returns:
        List of TileCoord covering the bounding box.
    """
    south, west, north, east = bounds
    # Get corner tiles
    nw = lat_lon_to_tile(north, west, zoom)
    se = lat_lon_to_tile(south, east, zoom)

    tiles = []
    for tx in range(nw.x, se.x + 1):
        for ty in range(nw.y, se.y + 1):
            tiles.append(TileCoord(x=tx, y=ty, zoom=zoom))
    return tiles
