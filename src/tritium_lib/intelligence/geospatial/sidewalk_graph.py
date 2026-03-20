# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SidewalkGraph — pedestrian navigation network from terrain segmentation.

Builds an A* navigable graph from sidewalk, crosswalk, and plaza polygons
extracted by the geospatial segmentation pipeline. Pedestrians pathfind on
this graph instead of walking randomly through open terrain.

Navigation stack:
    Destination (building entry, park bench, bus stop)
        │
        ▼
    SidewalkGraph.find_path(start, end)
        │   Falls back to direct path if no sidewalk data
        ▼
    Waypoint list along sidewalk centerlines
        │
        ▼
    MovementController follows waypoints
"""

from __future__ import annotations

import heapq
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

from tritium_lib.models.terrain import TerrainType

logger = logging.getLogger(__name__)


@dataclass
class SidewalkNode:
    """A node in the sidewalk navigation graph."""
    id: int
    x: float  # lon or local x
    y: float  # lat or local y
    terrain_type: TerrainType = TerrainType.SIDEWALK
    # Connected edges: list of (neighbor_id, cost)
    edges: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class SidewalkEdge:
    """An edge connecting two sidewalk nodes."""
    from_id: int
    to_id: int
    distance: float
    cost: float  # distance * terrain_cost_multiplier
    terrain_type: TerrainType = TerrainType.SIDEWALK


# Terrain cost multipliers for pedestrian sidewalk navigation
_SIDEWALK_COSTS = {
    TerrainType.SIDEWALK: 1.0,
    TerrainType.BRIDGE: 1.1,
    TerrainType.PARKING: 1.3,
    TerrainType.ROAD: 3.0,      # jaywalking — penalized but possible
    TerrainType.VEGETATION: 2.0,
    TerrainType.BARREN: 1.5,
    TerrainType.BUILDING: 100.0,  # practically impassable
    TerrainType.WATER: 1000.0,    # impassable
    TerrainType.RAIL: 5.0,
}


class SidewalkGraph:
    """Pedestrian navigation graph built from terrain segmentation.

    The graph is built by sampling sidewalk polygon centerlines and
    connecting adjacent samples. Road crossings get higher costs to
    encourage staying on sidewalks.
    """

    def __init__(self) -> None:
        self._nodes: dict[int, SidewalkNode] = {}
        self._next_id: int = 0
        self._spatial_index: dict[tuple[int, int], list[int]] = {}
        self._grid_resolution: float = 10.0  # 10m cells for local-meter coords
        self._coords_are_geo: bool = False  # set True for lat/lon coordinates

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(n.edges) for n in self._nodes.values()) // 2

    def build_from_terrain_layer(self, terrain_layer: object) -> int:
        """Build sidewalk graph from a TerrainLayer.

        Extracts sidewalk, road, bridge, parking polygons and creates
        navigation nodes at their centroids. Connects nearby nodes
        with edges weighted by terrain type.

        Returns number of nodes created.
        """
        if not hasattr(terrain_layer, 'regions'):
            return 0

        walkable_types = {
            TerrainType.SIDEWALK,
            TerrainType.BRIDGE,
            TerrainType.PARKING,
            TerrainType.ROAD,
            TerrainType.BARREN,
        }

        # Detect coordinate system: if centroids are in lat/lon range, convert to local meters
        sample_regions = [r for r in terrain_layer.regions if r.terrain_type in walkable_types]
        use_geo = False
        center_lon, center_lat = 0.0, 0.0
        if sample_regions:
            avg_lon = sum(r.centroid_lon for r in sample_regions) / len(sample_regions)
            avg_lat = sum(r.centroid_lat for r in sample_regions) / len(sample_regions)
            if -180 <= avg_lon <= 180 and -90 <= avg_lat <= 90 and abs(avg_lon) > 1:
                use_geo = True
                center_lon = avg_lon
                center_lat = avg_lat
                # Output will be in meters, use meter-scale grid
                self._grid_resolution = 10.0
            else:
                # Already in local coords
                self._grid_resolution = 10.0

        for region in terrain_layer.regions:
            if region.terrain_type not in walkable_types:
                continue

            if use_geo:
                # Convert lat/lon to local meters from center
                x = (region.centroid_lon - center_lon) * 111320 * math.cos(math.radians(center_lat))
                y = (region.centroid_lat - center_lat) * 111320
            else:
                x = region.centroid_lon
                y = region.centroid_lat

            self.add_node(x, y, region.terrain_type)

            # Sample additional nodes along polygon boundary
            if hasattr(terrain_layer, '_wkt_to_coords'):
                coords = terrain_layer._wkt_to_coords(region.geometry_wkt)
                if len(coords) >= 3:
                    if use_geo:
                        # Convert polygon coords to local meters
                        local_coords = [
                            ((lon - center_lon) * 111320 * math.cos(math.radians(center_lat)),
                             (lat - center_lat) * 111320)
                            for lon, lat in coords
                        ]
                        self._sample_polygon_edges(local_coords, region.terrain_type, sample_interval=30.0)
                    else:
                        self._sample_polygon_edges(coords, region.terrain_type)

        # Connect nearby nodes — adaptive distance based on data density
        if self.node_count > 0:
            avg_spacing = self._estimate_avg_spacing()
            # Connect at 3× average spacing to ensure connectivity
            # Clamp based on coordinate scale (meters vs degrees)
            if use_geo or avg_spacing > 1.0:
                # Local meters: connect within 50-200m
                connect_dist = max(avg_spacing * 3, 50.0)
                connect_dist = min(connect_dist, 200.0)
            else:
                # Geo degrees: connect within ~55m to ~550m
                connect_dist = max(avg_spacing * 3, 0.0005)
                connect_dist = min(connect_dist, 0.005)
            self._connect_nearby_nodes(max_distance=connect_dist)

        logger.info(
            "Built sidewalk graph: %d nodes, %d edges",
            self.node_count, self.edge_count,
        )
        return self.node_count

    def _sample_polygon_edges(
        self,
        coords: list[tuple[float, float]],
        terrain_type: TerrainType,
        sample_interval: float = 0.0003,  # ~33m between samples
    ) -> None:
        """Add nodes along polygon edges at regular intervals."""
        for i in range(len(coords) - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            edge_len = math.sqrt(dx * dx + dy * dy)
            if edge_len < sample_interval:
                continue
            num_samples = max(1, int(edge_len / sample_interval))
            for s in range(1, num_samples):
                t = s / num_samples
                self.add_node(
                    x0 + dx * t,
                    y0 + dy * t,
                    terrain_type,
                )

    def _estimate_avg_spacing(self) -> float:
        """Estimate average distance between nearby nodes."""
        if self.node_count < 2:
            return 0.001

        # Sample a few nodes and measure distance to nearest neighbor
        import random
        sample_ids = random.sample(
            list(self._nodes.keys()),
            min(20, self.node_count),
        )
        total_dist = 0.0
        count = 0
        for nid in sample_ids:
            node = self._nodes[nid]
            nearest = self._nearest_node(node.x, node.y)
            if nearest is not None and nearest != nid:
                other = self._nodes[nearest]
                d = math.sqrt((node.x - other.x) ** 2 + (node.y - other.y) ** 2)
                if d > 0:
                    total_dist += d
                    count += 1
        return total_dist / max(count, 1)

    def add_node(
        self,
        x: float,
        y: float,
        terrain_type: TerrainType = TerrainType.SIDEWALK,
    ) -> int:
        """Add a navigation node. Returns its ID."""
        node_id = self._next_id
        self._next_id += 1
        node = SidewalkNode(id=node_id, x=x, y=y, terrain_type=terrain_type)
        self._nodes[node_id] = node

        # Update spatial index
        cell = self._to_cell(x, y)
        if cell not in self._spatial_index:
            self._spatial_index[cell] = []
        self._spatial_index[cell].append(node_id)

        return node_id

    def add_edge(
        self,
        from_id: int,
        to_id: int,
        terrain_type: Optional[TerrainType] = None,
    ) -> None:
        """Add a bidirectional edge between two nodes."""
        if from_id not in self._nodes or to_id not in self._nodes:
            return

        n1 = self._nodes[from_id]
        n2 = self._nodes[to_id]
        distance = math.hypot(n2.x - n1.x, n2.y - n1.y)

        # Use the terrain type with higher cost (conservative)
        t = terrain_type or n2.terrain_type
        cost_mult = _SIDEWALK_COSTS.get(t, 2.0)

        # Road crossings: if connecting sidewalk to sidewalk through road
        if n1.terrain_type == TerrainType.SIDEWALK and n2.terrain_type == TerrainType.ROAD:
            cost_mult = _SIDEWALK_COSTS[TerrainType.ROAD]
        elif n1.terrain_type == TerrainType.ROAD and n2.terrain_type == TerrainType.SIDEWALK:
            cost_mult = _SIDEWALK_COSTS[TerrainType.ROAD]

        cost = distance * cost_mult

        n1.edges.append((to_id, cost))
        n2.edges.append((from_id, cost))

    def find_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        max_iterations: int = 5000,
    ) -> Optional[list[tuple[float, float]]]:
        """Find a path from start to end using A* on the sidewalk graph.

        Args:
            start: (x, y) or (lon, lat) start position
            end: (x, y) or (lon, lat) end position
            max_iterations: circuit breaker

        Returns:
            List of (x, y) waypoints, or None if no path found.
            Falls back to [start, end] if graph is empty.
        """
        if not self._nodes:
            return [start, end]

        # Find nearest nodes to start and end, preferring well-connected nodes
        start_id = self._nearest_connected_node(start[0], start[1])
        end_id = self._nearest_connected_node(end[0], end[1])

        if start_id is None or end_id is None:
            return [start, end]

        if start_id == end_id:
            node = self._nodes[start_id]
            return [start, (node.x, node.y), end]

        # A*
        g_score: dict[int, float] = {start_id: 0.0}
        came_from: dict[int, int] = {}

        end_node = self._nodes[end_id]
        counter = 0
        open_set: list[tuple[float, int, int]] = []

        h = self._heuristic(self._nodes[start_id], end_node)
        heapq.heappush(open_set, (h, counter, start_id))
        counter += 1

        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            f, _, current_id = heapq.heappop(open_set)

            if current_id == end_id:
                # Reconstruct path
                path = self._reconstruct(came_from, end_id)
                waypoints = [start]
                for nid in path:
                    n = self._nodes[nid]
                    waypoints.append((n.x, n.y))
                waypoints.append(end)
                return waypoints

            current_g = g_score.get(current_id, float("inf"))
            node = self._nodes[current_id]

            for neighbor_id, edge_cost in node.edges:
                tentative_g = current_g + edge_cost

                if tentative_g < g_score.get(neighbor_id, float("inf")):
                    g_score[neighbor_id] = tentative_g
                    came_from[neighbor_id] = current_id
                    h = self._heuristic(self._nodes[neighbor_id], end_node)
                    heapq.heappush(open_set, (tentative_g + h, counter, neighbor_id))
                    counter += 1

        # No path found — fall back to direct
        logger.debug("SidewalkGraph: no path found in %d iterations", iterations)
        return [start, end]

    def _nearest_connected_node(self, x: float, y: float, min_edges: int = 2) -> Optional[int]:
        """Find the nearest node in the main connected component.

        Builds and caches the main component on first call, then only
        returns nodes that are reachable from the largest component.
        Falls back to any node if the main component isn't near.
        """
        # Build main component cache on first call
        if not hasattr(self, '_main_component') or self._main_component is None:
            self._main_component = self._find_main_component()

        cell = self._to_cell(x, y)
        best_id = None
        best_dist = float("inf")
        fallback_id = None
        fallback_dist = float("inf")

        for dx in range(-10, 11):
            for dy in range(-10, 11):
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                for nid in self._spatial_index.get(neighbor_cell, []):
                    n = self._nodes[nid]
                    d = (n.x - x) ** 2 + (n.y - y) ** 2
                    if nid in self._main_component and d < best_dist:
                        best_dist = d
                        best_id = nid
                    if d < fallback_dist:
                        fallback_dist = d
                        fallback_id = nid

        return best_id if best_id is not None else fallback_id

    def _find_main_component(self) -> set[int]:
        """Find the largest connected component via BFS."""
        visited: set[int] = set()
        largest: set[int] = set()

        for start_nid in self._nodes:
            if start_nid in visited:
                continue
            component: set[int] = set()
            queue = [start_nid]
            while queue:
                nid = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                component.add(nid)
                for neighbor_id, _ in self._nodes[nid].edges:
                    if neighbor_id not in visited:
                        queue.append(neighbor_id)
            if len(component) > len(largest):
                largest = component

        return largest

    def _nearest_node(self, x: float, y: float) -> Optional[int]:
        """Find the nearest node to a point."""
        cell = self._to_cell(x, y)
        best_id = None
        best_dist = float("inf")

        # Search this cell and wider neighborhood (±5 cells = ±50m at 10m resolution)
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                for nid in self._spatial_index.get(neighbor_cell, []):
                    n = self._nodes[nid]
                    d = (n.x - x) ** 2 + (n.y - y) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_id = nid

        return best_id

    def _connect_nearby_nodes(self, max_distance: float = 0.002) -> None:
        """Connect nodes that are within max_distance of each other."""
        max_d2 = max_distance * max_distance

        for nid, node in self._nodes.items():
            cell = self._to_cell(node.x, node.y)
            cells_to_check = max(1, int(max_distance / self._grid_resolution))

            for dx in range(-cells_to_check, cells_to_check + 1):
                for dy in range(-cells_to_check, cells_to_check + 1):
                    neighbor_cell = (cell[0] + dx, cell[1] + dy)
                    for other_id in self._spatial_index.get(neighbor_cell, []):
                        if other_id <= nid:
                            continue  # avoid duplicates and self
                        other = self._nodes[other_id]
                        d2 = (node.x - other.x) ** 2 + (node.y - other.y) ** 2
                        if d2 < max_d2:
                            self.add_edge(nid, other_id)

    @staticmethod
    def _heuristic(a: SidewalkNode, b: SidewalkNode) -> float:
        """Euclidean distance heuristic for A*."""
        return math.hypot(b.x - a.x, b.y - a.y)

    @staticmethod
    def _reconstruct(came_from: dict[int, int], end_id: int) -> list[int]:
        """Walk came_from links backward."""
        path = [end_id]
        current = end_id
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Convert coordinates to spatial index cell."""
        return (
            int(x / self._grid_resolution),
            int(y / self._grid_resolution),
        )

    def get_walkable_area(
        self,
        center: tuple[float, float],
        radius: float,
    ) -> list[tuple[float, float]]:
        """Get all walkable node positions within radius of center."""
        r2 = radius * radius
        result = []
        for node in self._nodes.values():
            d2 = (node.x - center[0]) ** 2 + (node.y - center[1]) ** 2
            if d2 < r2:
                result.append((node.x, node.y))
        return result
