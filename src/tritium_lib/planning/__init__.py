# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Layer-driven costmap generation + the OPEN baseline A* global planner.

This package is the open-source baseline for fleet route planning (UX Loop 3
— dispatch a robot and route it around obstacles).  It has two halves:

- :mod:`tritium_lib.planning.layers` + :mod:`tritium_lib.planning.costmap`
  build a continuous float-cost grid from GIS layers: an elevation DEM
  (slope cost), obstacle polygons (buildings/water/flood -> lethal), and
  road lines (discounted corridors).

- :mod:`tritium_lib.planning.astar` plans a route over that costmap with a
  deterministic 8-connected A* (octile heuristic, no diagonal corner-cutting,
  cost-aware shortcut smoothing).

This is the OPEN baseline global planner.  An advanced flow-field planner
lives privately elsewhere and is intentionally neither referenced nor
reimplemented here.  Pure stdlib — no third-party dependencies.
"""

from __future__ import annotations

from .astar import RouteResult, plan_route
from .costmap import (
    Costmap,
    CostmapBuilder,
    CostmapWeights,
    costmap_from_terrain_map,
)
from .layers import ElevationGrid, wgs84_to_local

__all__ = [
    "Costmap",
    "CostmapWeights",
    "CostmapBuilder",
    "ElevationGrid",
    "costmap_from_terrain_map",
    "plan_route",
    "RouteResult",
    "wgs84_to_local",
]
