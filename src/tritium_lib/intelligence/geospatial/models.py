# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Pydantic data models for geospatial segmentation pipeline.

All models are importable without heavy dependencies — they use only
pydantic and stdlib types.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from tritium_lib.models.gis import TileBounds
from tritium_lib.models.terrain import TerrainType


class AreaOfOperations(BaseModel):
    """Defines a geographic region to process.

    An AO is the unit of work for the segmentation pipeline — download
    tiles for this bbox, segment them, classify terrain, cache results.
    """

    id: str  # e.g. "downtown_austin"
    name: str
    bounds: TileBounds
    zoom: int = 17  # tile zoom level (17 ≈ 1.2m/px)
    crs: str = "EPSG:4326"


class SegmentationConfig(BaseModel):
    """Controls segmentation behavior."""

    model_name: str = "sam2-tiny"  # sam2-tiny | sam2-large | sam3
    device: str = "auto"  # auto | cuda | cpu | mps
    batch_size: int = 4
    min_area_m2: float = 10.0
    max_area_m2: float = 100_000.0
    simplify_tolerance: float = 1.0  # Douglas-Peucker meters
    text_prompts: list[str] = Field(default_factory=lambda: [
        "building", "road", "water", "vegetation",
        "parking lot", "sidewalk", "bridge",
    ])
    # Color heuristic classifier thresholds
    use_color_heuristic: bool = True
    # LLM-assisted classification via llama-server
    llm_classify: bool = False
    llm_endpoint: str = "http://127.0.0.1:8081"


class SegmentedRegion(BaseModel):
    """A single classified polygon from segmentation.

    Represents one terrain feature — a building footprint, a road
    surface, a lake, a patch of vegetation, etc.
    """

    geometry_wkt: str  # WKT polygon
    terrain_type: TerrainType
    confidence: float = 0.0  # 0.0 - 1.0
    area_m2: float = 0.0
    centroid_lat: float = 0.0
    centroid_lon: float = 0.0
    properties: dict[str, Any] = Field(default_factory=dict)


class TerrainLayerMetadata(BaseModel):
    """Metadata for a cached terrain layer."""

    ao_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=__import__('datetime').timezone.utc))
    segment_count: int = 0
    cache_path: str = ""
    model_used: str = "color_heuristic"
    processing_time_s: float = 0.0
    source_imagery: str = "satellite"  # satellite, planet, drone, camera
    source_date: Optional[datetime] = None
    bounds: Optional[TileBounds] = None


class MovementCostTable(BaseModel):
    """Movement cost multipliers per terrain type for a unit class.

    Cost 0.7-1.0 = preferred, 1.0-3.0 = traversable but slow,
    >10 = practically impassable, inf = blocked.
    """

    road: float = 0.7
    sidewalk: float = 0.7
    building: float = float("inf")
    water: float = float("inf")
    vegetation_light: float = 1.5
    vegetation_dense: float = 3.0
    parking: float = 1.0
    bridge: float = 0.7
    barren: float = 1.2
    rail: float = 5.0
    open_: float = 1.0


# Pre-built cost tables per unit class
PEDESTRIAN_COSTS = MovementCostTable(
    road=0.8, sidewalk=0.7, building=float("inf"), water=float("inf"),
    vegetation_light=1.5, vegetation_dense=3.0, parking=1.0,
    bridge=0.8, barren=1.2, rail=5.0, open_=1.0,
)

LIGHT_VEHICLE_COSTS = MovementCostTable(
    road=0.7, sidewalk=float("inf"), building=float("inf"),
    water=float("inf"), vegetation_light=3.0, vegetation_dense=float("inf"),
    parking=0.8, bridge=0.7, barren=1.5, rail=float("inf"), open_=2.0,
)

HEAVY_VEHICLE_COSTS = MovementCostTable(
    road=0.7, sidewalk=float("inf"), building=float("inf"),
    water=float("inf"), vegetation_light=float("inf"),
    vegetation_dense=float("inf"), parking=1.0, bridge=0.8,
    barren=2.0, rail=float("inf"), open_=3.0,
)

DRONE_COSTS = MovementCostTable(
    road=1.0, sidewalk=1.0, building=5.0, water=1.0,
    vegetation_light=1.0, vegetation_dense=1.0, parking=1.0,
    bridge=1.0, barren=1.0, rail=1.0, open_=1.0,
)

ROVER_COSTS = MovementCostTable(
    road=0.8, sidewalk=2.0, building=float("inf"), water=float("inf"),
    vegetation_light=1.5, vegetation_dense=float("inf"), parking=1.0,
    bridge=0.8, barren=1.2, rail=float("inf"), open_=1.0,
)
