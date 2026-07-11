# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""A* pathfinder — plan_path() routes units based on type and available data.

Routing priority:
    1. Street graph A* (when loaded) — road-aware routing via OSM nodes
    2. Grid A* on TerrainMap (when loaded) — per-unit-type terrain avoidance
    3. Direct fallback (start -> end) — only when neither is available

Routing rules by unit type:
    - Rover/Tank/APC: snap to nearest road node, A* on street graph, road waypoints
    - Drone/Scout drone: straight line (ignores roads and buildings)
    - Hostile person: A* on roads for approach, then direct for last 30m
    - Turret (all types): no path (stationary)
    - Unknown: grid A* fallback, then direct

Grid A* ensures ground vehicles never drive through buildings and heavy
vehicles stay on roads, even when no StreetGraph is loaded.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass  # BuildingObstacles, StreetGraph, TerrainMap accepted via duck typing

# Unit types that are stationary (no path needed)
_STATIONARY_TYPES = {"turret", "heavy_turret", "missile_turret"}

# Unit types that fly (ignore roads and buildings)
_FLYING_TYPES = {"drone", "scout_drone"}

# Unit types that follow roads
_ROAD_TYPES = {"rover", "tank", "apc", "vehicle"}

# Unit types that prefer sidewalks (pedestrian navigation).
# A quadruped walks where people walk, not where cars drive.
_PEDESTRIAN_TYPES = {"person", "infantry", "civilian", "animal", "robot_dog"}

# Public aliases (2026-07-10): consumers (sc engine.route_path) need the same
# unit-type taxonomy to decide network-graph vs costmap planner precedence.
# Frozen so a consumer cannot mutate the dispatch tables.
PEDESTRIAN_TYPES = frozenset(_PEDESTRIAN_TYPES)
ROAD_TYPES = frozenset(_ROAD_TYPES)
FLYING_TYPES = frozenset(_FLYING_TYPES)
STATIONARY_TYPES = frozenset(_STATIONARY_TYPES)

# Per-unit-type standoff radius (meters) fed to the costmap A* planner as its
# ``clearance_m``: a wide, heavy unit keeps more distance from walls/obstacles
# than a person.  A tank/APC needs a wider berth than a rover/light vehicle.
# Pedestrians and unknown types stay 0.0 — PINNED: this protects pedestrian
# sidewalk routing and the riot golden-replay determinism (a non-zero clearance
# for peds would perturb their paths).  Consumers read this via
# :func:`clearance_for_unit_type`.
UNIT_CLEARANCE_M: dict[str, float] = {
    "rover": 1.0,
    "vehicle": 1.0,
    "tank": 2.0,
    "apc": 2.0,
}


def clearance_for_unit_type(unit_type: str) -> float:
    """Return the costmap standoff radius (meters) for ``unit_type``.

    Looks up :data:`UNIT_CLEARANCE_M`; pedestrians and any unrecognised type
    get ``0.0`` (no extra standoff — historical planner behavior).
    """
    return UNIT_CLEARANCE_M.get(unit_type, 0.0)

# Distance threshold for hostile direct approach (meters)
_HOSTILE_DIRECT_RANGE = 30.0


def plan_path(
    start: tuple[float, float],
    end: tuple[float, float],
    unit_type: str,
    street_graph: Optional[StreetGraph] = None,
    obstacles: Optional[BuildingObstacles] = None,
    alliance: str = "friendly",
    terrain_map: Optional[TerrainMap] = None,
    sidewalk_graph: Optional[SidewalkGraph] = None,
) -> Optional[list[tuple[float, float]]]:
    """Plan a path from start to end based on unit type and available data.

    Args:
        start: (x, y) in local meters
        end: (x, y) in local meters
        unit_type: asset_type from SimulationTarget
        street_graph: loaded StreetGraph (or None if unavailable)
        obstacles: loaded BuildingObstacles (or None if unavailable)
        alliance: "friendly", "hostile", or "neutral"
        terrain_map: loaded TerrainMap for grid A* fallback (or None)
        sidewalk_graph: loaded SidewalkGraph for pedestrian navigation (or None)

    Returns:
        List of (x, y) waypoints, or None for stationary units.
    """
    # Stationary units don't move
    if unit_type in _STATIONARY_TYPES:
        return None

    # Flying units go in a straight line
    if unit_type in _FLYING_TYPES:
        return [start, end]

    # Graphlings: always use grid A* with building avoidance (never street graph)
    if unit_type == "graphling":
        return _grid_fallback(start, end, unit_type, alliance, terrain_map, obstacles)

    # Pedestrians: try sidewalk graph first for terrain-aware routing
    if unit_type in _PEDESTRIAN_TYPES and sidewalk_graph is not None:
        path = _sidewalk_path(start, end, sidewalk_graph)
        if path is not None and len(path) > 2:
            # A curb-side sidewalk graph derived from roads is purely
            # geometric, so a run can clip a building that sits beside a
            # road. Only hand back a sidewalk route that stays clear of
            # buildings; otherwise fall through to the building-aware grid
            # A* so the pedestrian still REACHES its destination instead of
            # stalling against a wall on the swept per-tick collision check.
            if obstacles is None or not obstacles.path_crosses_building(path):
                return path
        # Sidewalk graph didn't help (no coverage or building-crossing) —
        # fall through to other methods (grid A* avoids buildings).

    # Hostile persons: road approach then direct last 30m
    if alliance == "hostile" and unit_type == "person":
        path = _hostile_path(start, end, street_graph)
        if path is not None and len(path) > 2:
            return path
        # Street graph didn't help — try grid A*
        return _grid_fallback(start, end, unit_type, alliance, terrain_map, obstacles)

    # Road-following ground units
    if unit_type in _ROAD_TYPES:
        path = _road_path(start, end, street_graph)
        if path is not None and len(path) > 2:
            return path
        # Street graph didn't help — try grid A*
        return _grid_fallback(start, end, unit_type, alliance, terrain_map, obstacles)

    # Unknown or other unit types: grid A* then direct fallback
    return _grid_fallback(start, end, unit_type, alliance, terrain_map, obstacles)


def _sidewalk_path(
    start: tuple[float, float],
    end: tuple[float, float],
    sidewalk_graph: SidewalkGraph,
) -> Optional[list[tuple[float, float]]]:
    """Route pedestrians along sidewalk graph for terrain-aware navigation."""
    try:
        path = sidewalk_graph.find_path(start, end)
        if path is not None and len(path) >= 2:
            return path
    except Exception:
        pass
    return None


def _grid_fallback(
    start: tuple[float, float],
    end: tuple[float, float],
    unit_type: str,
    alliance: str,
    terrain_map: Optional[TerrainMap],
    obstacles: Optional[BuildingObstacles] = None,
) -> list[tuple[float, float]]:
    """Try grid A* on terrain map, fall back to direct path.

    When *obstacles* is provided it is forwarded to ``grid_find_path()``
    so that the post-smoothing validation can reject paths whose smoothed
    segments cut through buildings.
    """
    if terrain_map is not None:
        try:
            from tritium_lib.sim_engine.world.grid_pathfinder import grid_find_path, profile_for_unit
            # Try the unit's own profile, then a permissive ground profile.
            # A vehicle on road-less terrain (no street graph) would
            # otherwise get no grid route and fall through to a straight
            # line THROUGH buildings — better to route it off-road around
            # them. The pedestrian profile treats open/yard as cheap, so it
            # finds a building-avoiding ground route whenever one exists.
            profiles = [profile_for_unit(unit_type, alliance)]
            if "pedestrian" not in profiles:
                profiles.append("pedestrian")
            for profile_name in profiles:
                path = grid_find_path(
                    terrain_map, start, end, profile_name,
                    obstacles=obstacles,
                )
                if path is not None and len(path) >= 2:
                    # Never hand back a route that crosses a building.
                    if obstacles is None or not obstacles.path_crosses_building(path):
                        return path
        except Exception:
            pass
    return [start, end]


def _road_path(
    start: tuple[float, float],
    end: tuple[float, float],
    street_graph: Optional[StreetGraph],
) -> list[tuple[float, float]]:
    """Route along roads via street graph. Fallback to direct if unavailable."""
    if street_graph is None or street_graph.graph is None:
        return [start, end]

    path = street_graph.shortest_path(start, end)
    if path is None or len(path) == 0:
        return [start, end]

    return path


def _hostile_path(
    start: tuple[float, float],
    end: tuple[float, float],
    street_graph: Optional[StreetGraph],
) -> list[tuple[float, float]]:
    """Hostile approach: follow roads to get close, then cut through for last 30m.

    If the total distance is < 30m, just go direct.
    If no street graph, go direct.
    """
    total_dist = math.hypot(end[0] - start[0], end[1] - start[1])

    # Short distance — just go direct
    if total_dist <= _HOSTILE_DIRECT_RANGE:
        return [start, end]

    if street_graph is None or street_graph.graph is None:
        return [start, end]

    # Find a road waypoint about 30m from the objective
    # Use the vector from end to start, normalized, to find the "peel off" point
    dx = start[0] - end[0]
    dy = start[1] - end[1]
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        return [start, end]

    # Point 30m from objective along the approach direction
    peel_off = (
        end[0] + (dx / dist) * _HOSTILE_DIRECT_RANGE,
        end[1] + (dy / dist) * _HOSTILE_DIRECT_RANGE,
    )

    # A* from start to peel_off on roads
    road_path = street_graph.shortest_path(start, peel_off)
    if road_path is None or len(road_path) == 0:
        return [start, end]

    # Append the objective as the final direct waypoint
    road_path.append(end)
    return road_path
