# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Road network pathfinding and pedestrian navigation.

Handles WHERE to go — the steering module handles HOW to move.

Two navigation systems:
    - RoadNetwork: A* pathfinding on a road graph for vehicles.
    - WalkableArea: Grid-based A* for pedestrians with obstacle avoidance.

Coordinate convention:
    Vec2 = tuple[float, float] in local meters.
    +X = East, +Y = North, same as tritium_lib.geo.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from tritium_lib.sim_engine.ai.steering import Vec2


def _distance(a: Vec2, b: Vec2) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _node_key(x: float, y: float, precision: float = 0.5) -> tuple[float, float]:
    """Round coordinates to merge nearby points into the same node."""
    inv = 1.0 / precision
    return (round(x * inv) / inv, round(y * inv) / inv)


# ---------------------------------------------------------------------------
# Road network — vehicle routing
# ---------------------------------------------------------------------------


@dataclass
class _RoadNode:
    """Internal graph node for the road network."""

    pos: Vec2
    # Adjacency: neighbor node key -> (distance, speed_limit)
    neighbors: dict[tuple[float, float], tuple[float, float]] = field(
        default_factory=dict
    )


class RoadNetwork:
    """Lightweight road graph for vehicle routing.

    Builds a graph from road segments (add_road calls), then
    routes vehicles along roads using A*. No external dependencies
    (no NetworkX) — pure Python with heapq.

    Usage:
        net = RoadNetwork()
        net.add_road((0, 0), (100, 0), speed_limit=13.4)
        net.add_road((100, 0), (100, 100))
        path = net.find_path((0, 0), (100, 100))
    """

    def __init__(self) -> None:
        # key -> _RoadNode
        self._nodes: dict[tuple[float, float], _RoadNode] = {}

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def nodes(self) -> list[Vec2]:
        """Return all node positions."""
        return [n.pos for n in self._nodes.values()]

    def add_road(
        self,
        start: Vec2,
        end: Vec2,
        speed_limit: float = 13.4,
        one_way: bool = False,
    ) -> None:
        """Add a road segment between two points.

        Intermediate nodes are created at both endpoints. If a node
        already exists near either endpoint (within 0.5m), the
        existing node is reused (intersection merging).

        Args:
            start: Road start point (x, y) in local meters.
            end: Road end point (x, y) in local meters.
            speed_limit: Max speed in m/s (default 13.4 = ~30 mph).
            one_way: If True, only start -> end is traversable.
        """
        sk = _node_key(start[0], start[1])
        ek = _node_key(end[0], end[1])

        if sk == ek:
            return  # Degenerate road

        # Create or reuse nodes
        if sk not in self._nodes:
            self._nodes[sk] = _RoadNode(pos=start)
        if ek not in self._nodes:
            self._nodes[ek] = _RoadNode(pos=end)

        dist = _distance(start, end)
        if dist < 0.1:
            return

        # Forward edge
        self._nodes[sk].neighbors[ek] = (dist, speed_limit)

        # Reverse edge (unless one-way)
        if not one_way:
            self._nodes[ek].neighbors[sk] = (dist, speed_limit)

    def find_path(self, start: Vec2, end: Vec2) -> list[Vec2]:
        """Find the shortest path between two points using A*.

        Start and end are snapped to the nearest road nodes.
        Returns a list of (x, y) waypoints along roads, or an
        empty list if no path exists.
        """
        if not self._nodes:
            return []

        start_key = self._nearest_key(start)
        end_key = self._nearest_key(end)

        if start_key is None or end_key is None:
            return []

        if start_key == end_key:
            return [self._nodes[start_key].pos]

        # A* search
        # Priority queue: (f_score, counter, node_key)
        counter = 0
        open_set: list[tuple[float, int, tuple[float, float]]] = []
        heapq.heappush(open_set, (0.0, counter, start_key))

        came_from: dict[tuple[float, float], tuple[float, float]] = {}
        g_score: dict[tuple[float, float], float] = {start_key: 0.0}

        end_pos = self._nodes[end_key].pos

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end_key:
                # Reconstruct path
                path: list[Vec2] = []
                node = current
                while node in came_from:
                    path.append(self._nodes[node].pos)
                    node = came_from[node]
                path.append(self._nodes[start_key].pos)
                path.reverse()
                return path

            current_node = self._nodes.get(current)
            if current_node is None:
                continue

            for neighbor_key, (dist, _speed) in current_node.neighbors.items():
                tentative_g = g_score.get(current, float("inf")) + dist

                if tentative_g < g_score.get(neighbor_key, float("inf")):
                    came_from[neighbor_key] = current
                    g_score[neighbor_key] = tentative_g
                    neighbor_node = self._nodes.get(neighbor_key)
                    if neighbor_node is None:
                        continue
                    h = _distance(neighbor_node.pos, end_pos)
                    f = tentative_g + h
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor_key))

        return []  # No path found

    def nearest_road_point(self, position: Vec2) -> Optional[Vec2]:
        """Snap a position to the nearest road node.

        Returns the position of the closest node, or None if the
        network is empty.
        """
        key = self._nearest_key(position)
        if key is None:
            return None
        return self._nodes[key].pos

    def random_destination(self) -> Optional[Vec2]:
        """Pick a random road node as a destination.

        Returns None if the network is empty.
        """
        if not self._nodes:
            return None
        key = random.choice(list(self._nodes.keys()))
        return self._nodes[key].pos

    def _nearest_key(
        self, position: Vec2
    ) -> Optional[tuple[float, float]]:
        """Find the nearest node key to a given position."""
        if not self._nodes:
            return None

        best_key: Optional[tuple[float, float]] = None
        best_dist = float("inf")
        for key, node in self._nodes.items():
            d = _distance(position, node.pos)
            if d < best_dist:
                best_dist = d
                best_key = key
        return best_key


# ---------------------------------------------------------------------------
# Walkable area — pedestrian navigation
# ---------------------------------------------------------------------------


def _point_in_polygon(px: float, py: float, polygon: list[Vec2]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def _segments_intersect(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> bool:
    """Check if line segment AB intersects line segment CD."""

    def cross(
        ox: float, oy: float, px: float, py: float, qx: float, qy: float
    ) -> float:
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)

    d1 = cross(cx, cy, dx, dy, ax, ay)
    d2 = cross(cx, cy, dx, dy, bx, by)
    d3 = cross(ax, ay, bx, by, cx, cy)
    d4 = cross(ax, ay, bx, by, dx, dy)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True
    return False


def _line_intersects_polygon(
    ax: float, ay: float, bx: float, by: float, polygon: list[Vec2]
) -> bool:
    """Check if line segment AB intersects any edge of a polygon."""
    n = len(polygon)
    for i in range(n):
        cx, cy = polygon[i]
        dx, dy = polygon[(i + 1) % n]
        if _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
            return True
    return False


class WalkableArea:
    """Navigation for pedestrians — avoids buildings, follows open ground.

    Uses a visibility graph approach: from any point, you can walk
    straight to any other point as long as the line doesn't cross
    an obstacle polygon. When direct paths are blocked, routes
    through obstacle vertices (with a small offset for clearance).

    Falls back to grid-based A* if the visibility graph doesn't
    find a path (complex obstacle layouts).

    Usage:
        area = WalkableArea(bounds=((0, 0), (200, 200)))
        area.add_obstacle([(50, 50), (80, 50), (80, 80), (50, 80)])
        path = area.find_path((10, 10), (150, 150))
    """

    def __init__(self, bounds: tuple[Vec2, Vec2]) -> None:
        """Initialize with a rectangular bounding area.

        Args:
            bounds: ((min_x, min_y), (max_x, max_y)) defining the
                    walkable region in local meters.
        """
        self.bounds = bounds
        self.obstacles: list[list[Vec2]] = []

    def add_obstacle(self, polygon: list[Vec2]) -> None:
        """Add a building footprint or other obstacle polygon.

        Pedestrians will route around these polygons.
        """
        if len(polygon) >= 3:
            self.obstacles.append(polygon)

    def is_walkable(self, point: Vec2) -> bool:
        """Check if a point is inside the bounds and not inside any obstacle."""
        x, y = point
        (min_x, min_y), (max_x, max_y) = self.bounds
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False
        for obs in self.obstacles:
            if _point_in_polygon(x, y, obs):
                return False
        return True

    def _line_clear(self, a: Vec2, b: Vec2) -> bool:
        """Check if a straight line between a and b is unobstructed."""
        for obs in self.obstacles:
            if _line_intersects_polygon(a[0], a[1], b[0], b[1], obs):
                return False
        return True

    def _obstacle_vertices(self, clearance: float = 1.0) -> list[Vec2]:
        """Get all obstacle vertices offset outward for clearance.

        Pushes each vertex slightly away from the polygon centroid
        so paths don't clip building corners.
        """
        vertices: list[Vec2] = []
        for obs in self.obstacles:
            if len(obs) < 3:
                continue
            # Centroid
            cx = sum(p[0] for p in obs) / len(obs)
            cy = sum(p[1] for p in obs) / len(obs)
            for vx, vy in obs:
                dx = vx - cx
                dy = vy - cy
                d = math.hypot(dx, dy)
                if d < 0.01:
                    continue
                # Push outward by clearance
                ox = vx + (dx / d) * clearance
                oy = vy + (dy / d) * clearance
                pt = (ox, oy)
                if self.is_walkable(pt):
                    vertices.append(pt)
        return vertices

    def find_path(self, start: Vec2, end: Vec2) -> list[Vec2]:
        """Find a walkable path from start to end, avoiding obstacles.

        Uses a visibility graph through obstacle corner vertices.
        Returns a list of waypoints, or empty list if no path found.
        """
        if not self.is_walkable(start) or not self.is_walkable(end):
            return []

        # Direct path?
        if self._line_clear(start, end):
            return [start, end]

        # Build visibility graph with obstacle vertices
        waypoint_candidates = self._obstacle_vertices(clearance=1.5)
        all_points = [start] + waypoint_candidates + [end]

        # Build adjacency via visibility
        n = len(all_points)
        adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                if self._line_clear(all_points[i], all_points[j]):
                    d = _distance(all_points[i], all_points[j])
                    adj[i].append((j, d))
                    adj[j].append((i, d))

        # A* on visibility graph
        start_idx = 0
        end_idx = n - 1
        end_pos = all_points[end_idx]

        counter = 0
        open_set: list[tuple[float, int, int]] = []
        heapq.heappush(open_set, (0.0, counter, start_idx))
        g_score: dict[int, float] = {start_idx: 0.0}
        came_from: dict[int, int] = {}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end_idx:
                path: list[Vec2] = []
                node = current
                while node in came_from:
                    path.append(all_points[node])
                    node = came_from[node]
                path.append(all_points[start_idx])
                path.reverse()
                return path

            for neighbor, dist in adj.get(current, []):
                tentative_g = g_score.get(current, float("inf")) + dist
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = _distance(all_points[neighbor], end_pos)
                    counter += 1
                    heapq.heappush(open_set, (tentative_g + h, counter, neighbor))

        return []  # No path found

    def random_point(self) -> Vec2:
        """Pick a random walkable point within bounds.

        Tries random samples; returns bounds center as last resort.
        """
        (min_x, min_y), (max_x, max_y) = self.bounds
        for _ in range(100):
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            if self.is_walkable((x, y)):
                return (x, y)
        # Fallback: center of bounds
        return ((min_x + max_x) / 2, (min_y + max_y) / 2)


# ---------------------------------------------------------------------------
# Route planning utilities
# ---------------------------------------------------------------------------


def plan_patrol_route(
    network: RoadNetwork,
    waypoints: list[Vec2],
    loop: bool = True,
) -> list[Vec2]:
    """Plan a patrol route visiting all waypoints in order.

    Connects consecutive waypoints via A* on the road network.
    If loop=True, connects the last waypoint back to the first.

    Returns the full sequence of road waypoints, or empty list
    if any segment is unroutable.
    """
    if len(waypoints) < 2:
        return list(waypoints)

    full_path: list[Vec2] = []
    pairs = list(zip(waypoints, waypoints[1:]))
    if loop:
        pairs.append((waypoints[-1], waypoints[0]))

    for i, (a, b) in enumerate(pairs):
        segment = network.find_path(a, b)
        if not segment:
            return []  # Unroutable
        if i == 0:
            full_path.extend(segment)
        else:
            # Skip the first point of subsequent segments (it's the
            # last point of the previous segment)
            full_path.extend(segment[1:])

    return full_path


def plan_random_walk(
    area: WalkableArea,
    start: Vec2,
    num_stops: int = 5,
) -> list[Vec2]:
    """Plan a random pedestrian walk with multiple stops.

    Generates random walkable destinations and connects them
    with obstacle-avoiding paths.

    Returns the full path (concatenated segments), or just [start]
    if no walkable destinations are found.
    """
    full_path: list[Vec2] = [start]
    current = start

    for _ in range(num_stops):
        dest = area.random_point()
        segment = area.find_path(current, dest)
        if segment and len(segment) >= 2:
            full_path.extend(segment[1:])  # Skip duplicate start point
            current = segment[-1]

    return full_path
