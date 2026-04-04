# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""World sub-package — world integrator, pathfinding, vision, sensors, cover,
terrain map, hazards, intercept math, LOD, pursuit, unit comms.

Contains the World/WorldBuilder/WORLD_PRESETS from the original world.py
plus modules moved from tritium-sc/src/engine/simulation/ during Phase 4
of sim engine unification.
"""

# Re-export original world.py contents (World, WorldBuilder, WORLD_PRESETS)
from ._world import World, WorldBuilder, WorldConfig, WORLD_PRESETS  # noqa: F401

# Modules moved from SC (Phase 4)
from .pathfinding import plan_path
from .grid_pathfinder import (
    grid_find_path,
    smooth_path,
    profile_for_unit,
    MovementProfile,
    PROFILES,
)
from .vision import VisionSystem, VisibilityState, SightingReport
from .sensors import SensorSimulator, SensorDevice
from .cover import CoverSystem, CoverObject

# Wave 196: additional SC modules migrated
from .intercept import (
    predict_intercept,
    lead_target,
    time_to_intercept,
    target_velocity,
)
from .comms import Signal, UnitComms, Message
from .hazards import Hazard, HazardManager
from .pursuit import PursuitSystem
from .lod import LODSystem, LODTier, ViewportState
from .terrain_map import TerrainMap, TerrainCell
from .procedural_city import generate_demo_city

__all__ = [
    # Original world.py
    "World",
    "WorldBuilder",
    "WorldConfig",
    "WORLD_PRESETS",
    # Pathfinding
    "plan_path",
    "grid_find_path",
    "smooth_path",
    "profile_for_unit",
    "MovementProfile",
    "PROFILES",
    # Vision
    "VisionSystem",
    "VisibilityState",
    "SightingReport",
    # Sensors
    "SensorSimulator",
    "SensorDevice",
    # Cover
    "CoverSystem",
    "CoverObject",
    # Intercept math
    "predict_intercept",
    "lead_target",
    "time_to_intercept",
    "target_velocity",
    # Unit comms
    "Signal",
    "UnitComms",
    "Message",
    # Hazards
    "Hazard",
    "HazardManager",
    # Pursuit
    "PursuitSystem",
    # LOD
    "LODSystem",
    "LODTier",
    "ViewportState",
    # Terrain map
    "TerrainMap",
    "TerrainCell",
    # Procedural city
    "generate_demo_city",
]

# Note: SidewalkGraph is available via:
#   from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph
# TerrainLayer is available via:
#   from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
