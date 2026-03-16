# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for pathfinding module: RoadNetwork, WalkableArea, patrol routing."""

from __future__ import annotations

import math

import pytest

from tritium_lib.game_ai.pathfinding import (
    RoadNetwork,
    WalkableArea,
    plan_patrol_route,
    plan_random_walk,
)


# ---------------------------------------------------------------------------
# RoadNetwork tests
# ---------------------------------------------------------------------------


class TestRoadNetworkAddQuery:
    """Test road network construction and basic queries."""

    def test_empty_network(self):
        net = RoadNetwork()
        assert net.node_count == 0
        assert net.nodes == []
        assert net.nearest_road_point((0, 0)) is None
        assert net.random_destination() is None

    def test_add_single_road(self):
        net = RoadNetwork()
        net.add_road((0, 0), (100, 0))
        assert net.node_count == 2

    def test_add_road_creates_bidirectional(self):
        net = RoadNetwork()
        net.add_road((0, 0), (100, 0))
        # Can route both directions
        assert len(net.find_path((0, 0), (100, 0))) > 0
        assert len(net.find_path((100, 0), (0, 0))) > 0

    def test_add_one_way_road(self):
        net = RoadNetwork()
        net.add_road((0, 0), (100, 0), one_way=True)
        # Forward works
        assert len(net.find_path((0, 0), (100, 0))) > 0
        # Reverse should fail (no other connections)
        assert len(net.find_path((100, 0), (0, 0))) == 0

    def test_degenerate_road_ignored(self):
        net = RoadNetwork()
        net.add_road((5, 5), (5, 5))  # Same point
        assert net.node_count == 0

    def test_intersection_merging(self):
        """Roads sharing an endpoint should merge at the intersection."""
        net = RoadNetwork()
        net.add_road((0, 0), (50, 0))
        net.add_road((50, 0), (50, 50))
        # The node at (50, 0) should be shared
        assert net.node_count == 3

    def test_nearest_road_point(self):
        net = RoadNetwork()
        net.add_road((0, 0), (100, 0))
        net.add_road((100, 0), (100, 100))
        # Point near (100, 0) should snap to it
        snapped = net.nearest_road_point((98, 2))
        assert snapped is not None
        assert abs(snapped[0] - 100) < 1.0
        assert abs(snapped[1] - 0) < 1.0

    def test_random_destination_returns_node(self):
        net = RoadNetwork()
        net.add_road((0, 0), (100, 0))
        net.add_road((100, 0), (100, 100))
        dest = net.random_destination()
        assert dest is not None
        # Should be one of the actual nodes
        assert any(
            abs(dest[0] - n[0]) < 1 and abs(dest[1] - n[1]) < 1
            for n in net.nodes
        )


class TestRoadNetworkAStar:
    """Test A* shortest path on road networks."""

    def test_direct_path(self):
        net = RoadNetwork()
        net.add_road((0, 0), (100, 0))
        path = net.find_path((0, 0), (100, 0))
        assert len(path) == 2

    def test_two_segment_path(self):
        net = RoadNetwork()
        net.add_road((0, 0), (100, 0))
        net.add_road((100, 0), (100, 100))
        path = net.find_path((0, 0), (100, 100))
        assert len(path) == 3

    def test_shortest_of_two_routes(self):
        """A* should pick the shorter of two available routes."""
        net = RoadNetwork()
        # Short route: (0,0) -> (10,0) -> (10,10)
        net.add_road((0, 0), (10, 0))
        net.add_road((10, 0), (10, 10))
        # Long route: (0,0) -> (0,100) -> (10,100) -> (10,10)
        net.add_road((0, 0), (0, 100))
        net.add_road((0, 100), (10, 100))
        net.add_road((10, 100), (10, 10))

        path = net.find_path((0, 0), (10, 10))
        assert len(path) > 0

        # Total distance of path should be close to the short route (~20m)
        total = sum(
            math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            for i in range(len(path) - 1)
        )
        assert total < 25  # Short route is ~20m, long is ~200m

    def test_no_path_disconnected(self):
        net = RoadNetwork()
        net.add_road((0, 0), (10, 0))
        net.add_road((100, 100), (110, 100))  # Disconnected
        path = net.find_path((0, 0), (110, 100))
        assert path == []

    def test_same_start_end(self):
        net = RoadNetwork()
        net.add_road((0, 0), (100, 0))
        path = net.find_path((0, 0), (0, 0))
        assert len(path) == 1


class TestNearestRoadPointSnap:
    """Test that nearest_road_point snaps correctly to road nodes."""

    def test_snap_to_closest_node(self):
        net = RoadNetwork()
        net.add_road((0, 0), (100, 0))
        net.add_road((100, 0), (200, 0))

        # Close to first node
        p = net.nearest_road_point((2, 1))
        assert p is not None
        assert abs(p[0]) < 1 and abs(p[1]) < 1

        # Close to middle node
        p = net.nearest_road_point((99, 3))
        assert p is not None
        assert abs(p[0] - 100) < 1 and abs(p[1]) < 1

        # Close to last node
        p = net.nearest_road_point((198, -2))
        assert p is not None
        assert abs(p[0] - 200) < 1 and abs(p[1]) < 1

    def test_snap_from_far_away(self):
        net = RoadNetwork()
        net.add_road((0, 0), (10, 0))
        p = net.nearest_road_point((1000, 1000))
        assert p is not None
        # Should still snap to one of the two nodes
        d0 = math.hypot(p[0], p[1])
        d1 = math.hypot(p[0] - 10, p[1])
        assert min(d0, d1) < 1


# ---------------------------------------------------------------------------
# WalkableArea tests
# ---------------------------------------------------------------------------


class TestWalkableArea:
    """Test pedestrian pathfinding with obstacle avoidance."""

    def test_direct_path_no_obstacles(self):
        area = WalkableArea(bounds=((0, 0), (100, 100)))
        path = area.find_path((10, 10), (90, 90))
        assert len(path) == 2
        assert path[0] == (10, 10)
        assert path[1] == (90, 90)

    def test_avoid_obstacle(self):
        area = WalkableArea(bounds=((0, 0), (100, 100)))
        # Place a building right in the middle blocking direct path
        area.add_obstacle([(40, 40), (60, 40), (60, 60), (40, 60)])

        path = area.find_path((10, 50), (90, 50))
        # Path should exist but route around the building
        assert len(path) > 2

        # Verify no point in path is inside the obstacle
        for px, py in path:
            assert not (40 < px < 60 and 40 < py < 60), \
                f"Path point ({px}, {py}) is inside the obstacle"

    def test_unreachable_start(self):
        area = WalkableArea(bounds=((0, 0), (100, 100)))
        area.add_obstacle([(40, 40), (60, 40), (60, 60), (40, 60)])
        # Start inside obstacle
        path = area.find_path((50, 50), (90, 90))
        assert path == []

    def test_outside_bounds(self):
        area = WalkableArea(bounds=((0, 0), (100, 100)))
        path = area.find_path((-10, -10), (50, 50))
        assert path == []

    def test_random_point_is_walkable(self):
        area = WalkableArea(bounds=((0, 0), (100, 100)))
        area.add_obstacle([(40, 40), (60, 40), (60, 60), (40, 60)])
        for _ in range(20):
            pt = area.random_point()
            assert area.is_walkable(pt)

    def test_is_walkable(self):
        area = WalkableArea(bounds=((0, 0), (100, 100)))
        area.add_obstacle([(40, 40), (60, 40), (60, 60), (40, 60)])
        assert area.is_walkable((10, 10))
        assert not area.is_walkable((50, 50))
        assert not area.is_walkable((-5, 50))


# ---------------------------------------------------------------------------
# Patrol routing tests
# ---------------------------------------------------------------------------


class TestPatrolRoute:
    """Test patrol route planning on road networks."""

    def test_patrol_visits_all_waypoints(self):
        net = RoadNetwork()
        # Build a simple grid
        net.add_road((0, 0), (50, 0))
        net.add_road((50, 0), (100, 0))
        net.add_road((100, 0), (100, 50))
        net.add_road((100, 50), (100, 100))
        net.add_road((0, 0), (0, 50))
        net.add_road((0, 50), (0, 100))
        net.add_road((0, 100), (50, 100))
        net.add_road((50, 100), (100, 100))

        waypoints = [(0, 0), (100, 0), (100, 100), (0, 100)]
        route = plan_patrol_route(net, waypoints, loop=True)
        assert len(route) > 0

        # Verify each waypoint appears in the route (within tolerance)
        for wp in waypoints:
            found = any(
                abs(p[0] - wp[0]) < 2 and abs(p[1] - wp[1]) < 2
                for p in route
            )
            assert found, f"Waypoint {wp} not found in patrol route"

    def test_patrol_no_loop(self):
        net = RoadNetwork()
        net.add_road((0, 0), (50, 0))
        net.add_road((50, 0), (100, 0))

        waypoints = [(0, 0), (50, 0), (100, 0)]
        route = plan_patrol_route(net, waypoints, loop=False)
        assert len(route) > 0
        # First point should be near (0,0), last near (100,0)
        assert abs(route[0][0]) < 2
        assert abs(route[-1][0] - 100) < 2

    def test_patrol_unroutable(self):
        net = RoadNetwork()
        net.add_road((0, 0), (10, 0))
        net.add_road((100, 100), (110, 100))
        # Disconnected waypoints
        route = plan_patrol_route(net, [(0, 0), (110, 100)])
        assert route == []

    def test_single_waypoint(self):
        net = RoadNetwork()
        net.add_road((0, 0), (100, 0))
        route = plan_patrol_route(net, [(50, 0)])
        assert route == [(50, 0)]


class TestRandomWalk:
    """Test random walk planning."""

    def test_random_walk_returns_path(self):
        area = WalkableArea(bounds=((0, 0), (200, 200)))
        path = plan_random_walk(area, (100, 100), num_stops=3)
        assert len(path) >= 1
        assert path[0] == (100, 100)

    def test_random_walk_with_obstacles(self):
        area = WalkableArea(bounds=((0, 0), (200, 200)))
        area.add_obstacle([(80, 80), (120, 80), (120, 120), (80, 120)])
        path = plan_random_walk(area, (10, 10), num_stops=3)
        assert len(path) >= 1
