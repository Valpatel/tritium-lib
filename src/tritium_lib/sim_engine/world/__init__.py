# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""World sub-package — world integrator, pathfinding, vision, sensors, cover.

Contains the World/WorldBuilder/WORLD_PRESETS from the original world.py
plus pathfinding, vision, sensors, and cover modules moved from
tritium-sc/src/engine/simulation/ during Phase 4 of sim engine unification.
"""

# Re-export original world.py contents (World, WorldBuilder, WORLD_PRESETS)
from ._world import World, WorldBuilder, WorldConfig, WORLD_PRESETS  # noqa: F401

# New modules moved from SC
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
]
