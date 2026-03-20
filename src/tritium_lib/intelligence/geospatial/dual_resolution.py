# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Dual-resolution terrain-aware pathfinding.

Current problem: the TerrainMap uses 5m cells. A sidewalk is 1.5m wide
and invisible at 5m resolution. This module provides dual-resolution
pathfinding:

    Coarse grid (5m): strategic pathfinding over long distances
    Fine grid (1m):   tactical movement for the last 100m near destination

The fine grid is only populated for a small area around the active
region, keeping memory usage reasonable while enabling centimeter-level
terrain awareness where it matters (sidewalks, lane boundaries, curbs).

The dual-res pathfinder uses the existing A* implementation but provides
a finer TerrainMap-compatible grid for nearby navigation.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Optional

from tritium_lib.intelligence.geospatial._deps import HAS_NUMPY
from tritium_lib.models.terrain import TerrainType

logger = logging.getLogger(__name__)

# Fine grid parameters
FINE_RESOLUTION = 1.0  # meters per cell
FINE_RADIUS = 100.0    # meters — fine grid loaded for this radius
COARSE_RESOLUTION = 5.0  # meters per cell


@dataclass
class FineGridCell:
    """A single cell in the fine-resolution terrain grid."""
    terrain_type: str
    cost: float  # movement cost multiplier
    walkable: bool


class FineTerrainGrid:
    """Fine-resolution (1m) terrain grid for a local area.

    Provides the same interface as TerrainMap (duck typing) so it
    can be used with the existing grid_find_path() function.

    The fine grid is populated from a TerrainLayer's segmented regions,
    giving centimeter-level terrain awareness for sidewalks, lane
    boundaries, building edges, and curbs.
    """

    def __init__(
        self,
        center_x: float,
        center_y: float,
        radius: float = FINE_RADIUS,
        resolution: float = FINE_RESOLUTION,
    ) -> None:
        self._center_x = center_x
        self._center_y = center_y
        self._radius = radius
        self._resolution = resolution
        self._half_size = radius

        # Grid dimensions
        self._grid_size = int(2 * radius / resolution) + 1
        # Default to "open" terrain
        self._grid: list[list[str]] = [
            ["open"] * self._grid_size for _ in range(self._grid_size)
        ]

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def resolution(self) -> float:
        return self._resolution

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to grid cell indices."""
        col = int((x - self._center_x + self._half_size) / self._resolution)
        row = int((y - self._center_y + self._half_size) / self._resolution)
        return (col, row)

    def _grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        """Convert grid cell indices to world coordinates (cell center)."""
        x = self._center_x - self._half_size + (col + 0.5) * self._resolution
        y = self._center_y - self._half_size + (row + 0.5) * self._resolution
        return (x, y)

    def get_terrain_at(self, col: int, row: int) -> str:
        """Get terrain type string at grid cell."""
        if col < 0 or row < 0 or col >= self._grid_size or row >= self._grid_size:
            return "out_of_bounds"
        return self._grid[row][col]

    def set_cell(self, col: int, row: int, terrain_type: str) -> None:
        """Set terrain type at a grid cell."""
        if 0 <= col < self._grid_size and 0 <= row < self._grid_size:
            self._grid[row][col] = terrain_type

    def populate_from_terrain_layer(
        self,
        terrain_layer: Any,
        geo_to_local: Optional[Any] = None,
    ) -> int:
        """Populate the fine grid from a TerrainLayer.

        For each segmented region in the terrain layer, sets cells
        within the region's footprint to the appropriate terrain type.

        Args:
            terrain_layer: TerrainLayer with classified regions
            geo_to_local: optional converter with to_local(lat, lon) -> (x, y)
                If None, uses centroid coordinates directly as local coords.

        Returns number of cells set.
        """
        if not hasattr(terrain_layer, 'regions'):
            return 0

        # Map TerrainType to grid terrain strings
        type_to_str = {
            TerrainType.BUILDING: "building",
            TerrainType.ROAD: "road",
            TerrainType.WATER: "water",
            TerrainType.VEGETATION: "yard",
            TerrainType.PARKING: "parking",
            TerrainType.SIDEWALK: "sidewalk",
            TerrainType.BRIDGE: "bridge",
            TerrainType.BARREN: "barren",
            TerrainType.RAIL: "rail",
        }

        count = 0
        for region in terrain_layer.regions:
            terrain_str = type_to_str.get(region.terrain_type)
            if terrain_str is None:
                continue

            # Get local coordinates
            if geo_to_local and hasattr(geo_to_local, 'to_local'):
                x, y = geo_to_local.to_local(region.centroid_lat, region.centroid_lon)
            else:
                x = region.centroid_lon
                y = region.centroid_lat

            # Check if within fine grid radius
            dx = x - self._center_x
            dy = y - self._center_y
            if dx * dx + dy * dy > self._radius * self._radius:
                continue

            # Estimate cell coverage from area
            radius_cells = max(1, int(math.sqrt(region.area_m2) / self._resolution / 2))

            col, row = self._world_to_grid(x, y)
            for dc in range(-radius_cells, radius_cells + 1):
                for dr in range(-radius_cells, radius_cells + 1):
                    self.set_cell(col + dc, row + dr, terrain_str)
                    count += 1

        logger.debug(
            "Fine grid populated: %d cells set (%.0fm radius at %.1fm resolution)",
            count, self._radius, self._resolution,
        )
        return count

    def populate_from_coarse_map(self, coarse_map: Any) -> int:
        """Populate from a coarse TerrainMap by upsampling.

        Each coarse cell (5m) fills multiple fine cells (1m).
        """
        if not hasattr(coarse_map, 'get_terrain_at'):
            return 0

        count = 0
        scale = int(COARSE_RESOLUTION / self._resolution)

        for row in range(self._grid_size):
            for col in range(self._grid_size):
                wx, wy = self._grid_to_world(col, row)
                if hasattr(coarse_map, '_world_to_grid'):
                    ccol, crow = coarse_map._world_to_grid(wx, wy)
                    terrain = coarse_map.get_terrain_at(ccol, crow)
                    if terrain != "out_of_bounds":
                        self._grid[row][col] = terrain
                        count += 1

        return count


class DualResolutionPathfinder:
    """Pathfinder that uses coarse grid for distance, fine grid for precision.

    Strategy:
    1. Use existing A* on coarse TerrainMap for the full path
    2. For the last `refine_distance` meters, re-pathfind on a fine grid
    3. Splice the refined path into the coarse path

    This gives strategic-level routing (avoid lakes, follow roads) with
    tactical-level precision (stay on sidewalks, avoid building edges).
    """

    def __init__(
        self,
        refine_distance: float = 100.0,
        fine_resolution: float = FINE_RESOLUTION,
    ) -> None:
        self.refine_distance = refine_distance
        self.fine_resolution = fine_resolution

    def find_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        profile_name: str,
        coarse_map: Any = None,
        terrain_layer: Any = None,
        max_iterations: int = 5000,
    ) -> Optional[list[tuple[float, float]]]:
        """Find a path using dual-resolution strategy.

        Args:
            start: (x, y) world coordinates
            end: (x, y) world coordinates
            profile_name: movement profile name
            coarse_map: TerrainMap for strategic routing (5m grid)
            terrain_layer: TerrainLayer for fine-grid population
            max_iterations: A* circuit breaker

        Returns:
            List of (x, y) waypoints, or None if no path found.
        """
        from tritium_lib.sim_engine.world.grid_pathfinder import grid_find_path

        # Step 1: Coarse A* for full path
        coarse_path = None
        if coarse_map is not None:
            coarse_path = grid_find_path(
                coarse_map, start, end, profile_name,
                max_iterations=max_iterations,
            )

        if coarse_path is None:
            # No coarse map or no path found — try fine grid only
            if terrain_layer is not None:
                return self._fine_grid_path(start, end, profile_name, terrain_layer)
            return [start, end]

        # Step 2: Check if path is short enough to refine entirely
        total_dist = self._path_length(coarse_path)
        if total_dist <= self.refine_distance and terrain_layer is not None:
            # Short path — refine all of it
            fine_path = self._fine_grid_path(start, end, profile_name, terrain_layer)
            if fine_path and len(fine_path) > 2:
                return fine_path
            return coarse_path

        # Step 3: Find the refinement point (where we switch to fine grid)
        refine_point = self._find_refine_point(coarse_path)
        if refine_point is None or terrain_layer is None:
            return coarse_path

        # Step 4: Re-pathfind the last segment on fine grid
        fine_path = self._fine_grid_path(
            refine_point, end, profile_name, terrain_layer,
        )

        if fine_path and len(fine_path) > 2:
            # Splice: coarse path up to refine point + fine path from there
            return self._splice_paths(coarse_path, fine_path, refine_point)

        return coarse_path

    def _fine_grid_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        profile_name: str,
        terrain_layer: Any,
    ) -> Optional[list[tuple[float, float]]]:
        """Pathfind on a fine-resolution grid around the destination."""
        from tritium_lib.sim_engine.world.grid_pathfinder import grid_find_path

        # Create fine grid centered on midpoint
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        dist = math.hypot(end[0] - start[0], end[1] - start[1])
        radius = max(dist / 2 + 20, 50.0)  # at least 50m, plus 20m margin

        fine_grid = FineTerrainGrid(
            center_x=mid_x,
            center_y=mid_y,
            radius=min(radius, FINE_RADIUS),
            resolution=self.fine_resolution,
        )
        fine_grid.populate_from_terrain_layer(terrain_layer)

        return grid_find_path(
            fine_grid, start, end, profile_name,
            max_iterations=10000,  # more iterations for fine grid
        )

    def _find_refine_point(
        self,
        path: list[tuple[float, float]],
    ) -> Optional[tuple[float, float]]:
        """Find the point on the path where we should switch to fine grid.

        This is approximately `refine_distance` meters from the end.
        """
        if len(path) < 2:
            return None

        # Walk backward from end until we've covered refine_distance
        remaining = self.refine_distance
        for i in range(len(path) - 1, 0, -1):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            seg_len = math.hypot(dx, dy)
            remaining -= seg_len
            if remaining <= 0:
                return path[i]

        return path[0]

    def _splice_paths(
        self,
        coarse: list[tuple[float, float]],
        fine: list[tuple[float, float]],
        splice_point: tuple[float, float],
    ) -> list[tuple[float, float]]:
        """Splice fine path into coarse path at the splice point."""
        # Find the closest point in coarse path to splice_point
        best_idx = 0
        best_dist = float("inf")
        for i, pt in enumerate(coarse):
            d = (pt[0] - splice_point[0]) ** 2 + (pt[1] - splice_point[1]) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i

        # Take coarse path up to splice point, then fine path
        return coarse[:best_idx + 1] + fine[1:]  # skip fine[0] to avoid duplicate

    @staticmethod
    def _path_length(path: list[tuple[float, float]]) -> float:
        """Total Euclidean length of a path."""
        total = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            total += math.hypot(dx, dy)
        return total
