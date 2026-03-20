# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Geospatial segmentation — satellite/aerial imagery to classified terrain polygons.

Transforms imagery into semantic terrain layers (buildings, roads, water,
vegetation, sidewalks, parking) for pathfinding, NPC AI, commander
intelligence, and GIS fusion.

Process once, use everywhere: segment → classify → cache → query.
"""

from tritium_lib.intelligence.geospatial.models import (
    AreaOfOperations,
    MovementCostTable,
    PEDESTRIAN_COSTS,
    LIGHT_VEHICLE_COSTS,
    HEAVY_VEHICLE_COSTS,
    DRONE_COSTS,
    ROVER_COSTS,
    SegmentationConfig,
    SegmentedRegion,
    TerrainLayerMetadata,
)
from tritium_lib.intelligence.geospatial._deps import (
    HAS_NUMPY,
    HAS_PILLOW,
    HAS_RASTERIO,
    HAS_SHAPELY,
    HAS_TORCH,
    HAS_SAM,
    HAS_CV2,
)

__all__ = [
    # Models (always available)
    "AreaOfOperations",
    "MovementCostTable",
    "PEDESTRIAN_COSTS",
    "LIGHT_VEHICLE_COSTS",
    "HEAVY_VEHICLE_COSTS",
    "DRONE_COSTS",
    "ROVER_COSTS",
    "SegmentationConfig",
    "SegmentedRegion",
    "TerrainLayerMetadata",
    # Dependency flags
    "HAS_NUMPY",
    "HAS_PILLOW",
    "HAS_RASTERIO",
    "HAS_SHAPELY",
    "HAS_TORCH",
    "HAS_SAM",
    "HAS_CV2",
    # Pipeline modules (import directly):
    #   from tritium_lib.intelligence.geospatial.tile_downloader import TileDownloader
    #   from tritium_lib.intelligence.geospatial.segmentation import SegmentationEngine
    #   from tritium_lib.intelligence.geospatial.terrain_classifier import TerrainClassifier
    #   from tritium_lib.intelligence.geospatial.vector_converter import VectorConverter
    #   from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
    #   from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph
    #   from tritium_lib.intelligence.geospatial.providers import ProviderRegistry
    #   from tritium_lib.intelligence.geospatial.change_detector import ChangeDetector
]
