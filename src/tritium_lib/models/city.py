# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Canonical city data models for 3D city simulation.

Defines the schema for city data exchanged between the geospatial
pipeline (/api/geo/city-data) and the frontend 3D renderer (map3d.js).
All coordinates are in local meters relative to a center lat/lng.

Schema version is embedded in cache keys to invalidate stale data
when the model evolves.

MQTT topics (future):
    tritium/{site}/city/buildings — building footprint updates
    tritium/{site}/city/traffic  — traffic state updates
"""

from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Bump this when the schema changes to invalidate caches
CITY_DATA_SCHEMA_VERSION = 1

# --- Building height defaults by OSM building type ---
BUILDING_TYPE_HEIGHTS: dict[str, float] = {
    "apartments": 15.0,
    "residential": 8.0,
    "house": 7.0,
    "detached": 7.0,
    "terrace": 8.0,
    "commercial": 12.0,
    "retail": 5.0,
    "industrial": 8.0,
    "warehouse": 7.0,
    "office": 18.0,
    "hotel": 20.0,
    "hospital": 15.0,
    "school": 10.0,
    "university": 12.0,
    "church": 15.0,
    "cathedral": 25.0,
    "mosque": 12.0,
    "synagogue": 10.0,
    "public": 10.0,
    "civic": 12.0,
    "government": 15.0,
    "garage": 3.0,
    "garages": 3.0,
    "parking": 9.0,
    "shed": 3.0,
    "roof": 4.0,
    "hut": 3.0,
    "cabin": 4.0,
    "farm": 6.0,
    "barn": 7.0,
    "service": 4.0,
    "kiosk": 3.0,
    "supermarket": 6.0,
    "train_station": 10.0,
    "prison": 10.0,
    "temple": 12.0,
    "shrine": 6.0,
    "chapel": 10.0,
    "dormitory": 12.0,
    "semidetached_house": 8.0,
    "manufacture": 8.0,
    "kindergarten": 6.0,
    "fire_station": 10.0,
    "yes": 8.0,
}

# --- Building categories for material selection ---
BUILDING_CATEGORIES: dict[str, str] = {
    "apartments": "residential",
    "residential": "residential",
    "house": "residential",
    "detached": "residential",
    "terrace": "residential",
    "semidetached_house": "residential",
    "dormitory": "residential",
    "farm": "residential",
    "cabin": "residential",
    "hut": "residential",
    "commercial": "commercial",
    "retail": "commercial",
    "supermarket": "commercial",
    "kiosk": "commercial",
    "office": "commercial",
    "hotel": "commercial",
    "industrial": "industrial",
    "warehouse": "industrial",
    "manufacture": "industrial",
    "hospital": "civic",
    "school": "civic",
    "university": "civic",
    "kindergarten": "civic",
    "public": "civic",
    "civic": "civic",
    "government": "civic",
    "fire_station": "civic",
    "train_station": "civic",
    "prison": "civic",
    "church": "religious",
    "cathedral": "religious",
    "chapel": "religious",
    "mosque": "religious",
    "synagogue": "religious",
    "temple": "religious",
    "shrine": "religious",
    "garage": "utility",
    "garages": "utility",
    "parking": "utility",
    "shed": "utility",
    "roof": "utility",
    "service": "utility",
}

# --- Road widths by OSM highway type ---
ROAD_WIDTHS: dict[str, float] = {
    "motorway": 14.0,
    "trunk": 12.0,
    "primary": 10.0,
    "secondary": 8.0,
    "tertiary": 7.0,
    "residential": 6.0,
    "service": 4.0,
    "unclassified": 6.0,
    "living_street": 5.0,
    "pedestrian": 4.0,
    "footway": 2.0,
    "cycleway": 2.0,
    "path": 1.5,
    "track": 3.0,
    "steps": 1.5,
    "motorway_link": 6.0,
    "trunk_link": 5.0,
    "primary_link": 5.0,
    "secondary_link": 4.5,
    "tertiary_link": 4.0,
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class CityBuilding(BaseModel):
    """A building footprint with height and classification."""

    id: int
    polygon: list[list[float]]  # [[x, z], ...] in local meters
    height: float = Field(ge=0.5, le=1000.0, default=8.0)
    type: str = "yes"
    category: str = "residential"
    name: str = ""
    levels: Optional[int] = None
    roof_shape: str = ""
    colour: str = ""

    @field_validator("polygon")
    @classmethod
    def polygon_must_be_valid(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) < 3:
            raise ValueError(f"Polygon must have >= 3 points, got {len(v)}")
        for pt in v:
            if len(pt) < 2:
                raise ValueError(f"Point must have >= 2 coordinates, got {len(pt)}")
            if any(math.isnan(c) or math.isinf(c) for c in pt):
                raise ValueError(f"Point contains NaN or Inf: {pt}")
        return v


class CityRoad(BaseModel):
    """A road segment with type and width."""

    id: int
    points: list[list[float]]  # [[x, z], ...] in local meters
    road_class: str = Field(alias="class", default="residential")
    name: str = ""
    width: float = Field(ge=0.5, le=100.0, default=6.0)
    lanes: int = Field(ge=1, le=20, default=2)
    surface: str = "asphalt"
    oneway: bool = False
    bridge: bool = False
    tunnel: bool = False
    maxspeed: str = ""

    model_config = {"populate_by_name": True}

    @field_validator("points")
    @classmethod
    def points_must_be_valid(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) < 2:
            raise ValueError(f"Road must have >= 2 points, got {len(v)}")
        for pt in v:
            if len(pt) < 2:
                raise ValueError(f"Point must have >= 2 coordinates")
            if any(math.isnan(c) or math.isinf(c) for c in pt):
                raise ValueError(f"Point contains NaN or Inf: {pt}")
        return v


class CityTree(BaseModel):
    """A tree with position and species."""

    pos: list[float]  # [x, z] in local meters
    species: str = ""
    height: float = Field(ge=0.5, le=100.0, default=6.0)
    leaf_type: str = "broadleaved"

    @field_validator("pos")
    @classmethod
    def pos_must_be_valid(cls, v: list[float]) -> list[float]:
        if len(v) < 2:
            raise ValueError("Position must have >= 2 coordinates")
        if any(math.isnan(c) or math.isinf(c) for c in v):
            raise ValueError(f"Position contains NaN or Inf: {v}")
        return v


class CityBarrier(BaseModel):
    """A barrier (fence, wall, hedge) segment."""

    id: int
    points: list[list[float]]
    type: str = "fence"
    height: float = Field(ge=0.1, le=50.0, default=1.5)


class CityWater(BaseModel):
    """A water feature (pond, stream, canal)."""

    id: int
    polygon: Optional[list[list[float]]] = None  # closed polygon for lakes
    points: Optional[list[list[float]]] = None  # open line for streams
    type: str = "water"
    name: str = ""


class CityEntrance(BaseModel):
    """A building entrance/door position."""

    pos: list[float]  # [x, z]
    type: str = "yes"
    wheelchair: str = ""
    name: str = ""


class CityPOI(BaseModel):
    """A point of interest (amenity)."""

    pos: list[float]  # [x, z]
    type: str = "amenity"
    name: str = ""
    cuisine: str = ""


class CityLanduse(BaseModel):
    """A land use zone polygon."""

    id: int
    polygon: list[list[float]]
    type: str = ""
    name: str = ""


class CityDataStats(BaseModel):
    """Statistics about city data contents."""

    buildings: int = 0
    roads: int = 0
    trees: int = 0
    landuse: int = 0
    barriers: int = 0
    water: int = 0
    entrances: int = 0
    pois: int = 0


class CityData(BaseModel):
    """Complete city data for an area — the canonical exchange format.

    All coordinates in local meters relative to center lat/lng.
    Used by /api/geo/city-data and consumed by map3d.js renderers.
    """

    center: dict[str, float]  # {"lat": ..., "lng": ...}
    radius: float = 300.0
    schema_version: int = CITY_DATA_SCHEMA_VERSION
    buildings: list[CityBuilding] = Field(default_factory=list)
    roads: list[CityRoad] = Field(default_factory=list)
    trees: list[CityTree] = Field(default_factory=list)
    landuse: list[CityLanduse] = Field(default_factory=list)
    barriers: list[CityBarrier] = Field(default_factory=list)
    water: list[CityWater] = Field(default_factory=list)
    entrances: list[CityEntrance] = Field(default_factory=list)
    pois: list[CityPOI] = Field(default_factory=list)
    stats: CityDataStats = Field(default_factory=CityDataStats)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def estimate_building_height(tags: dict[str, str]) -> float:
    """Estimate building height from OSM tags.

    Priority: explicit height > building:levels > type default > 8m.
    """
    height_str = tags.get("height")
    if height_str:
        try:
            return float(height_str.replace("m", "").strip())
        except (ValueError, TypeError):
            pass

    levels_str = tags.get("building:levels")
    if levels_str:
        try:
            return float(levels_str) * 3.0 + 1.0
        except (ValueError, TypeError):
            pass

    btype = tags.get("building", "yes").lower()
    return BUILDING_TYPE_HEIGHTS.get(btype, 8.0)


def classify_building(tags: dict[str, str]) -> str:
    """Classify building into category for material selection."""
    btype = tags.get("building", "yes").lower()
    return BUILDING_CATEGORIES.get(btype, "residential")


def get_road_width(highway_type: str, tags: Optional[dict[str, str]] = None) -> float:
    """Get road width from highway type and optional width tag."""
    if tags and tags.get("width"):
        try:
            return float(tags["width"].replace("m", "").strip())
        except (ValueError, TypeError):
            pass
    return ROAD_WIDTHS.get(highway_type, 6.0)
