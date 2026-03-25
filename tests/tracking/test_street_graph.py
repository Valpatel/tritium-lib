# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for street graph extraction from OpenStreetMap Overpass API.

Covers: graph construction, A* pathfinding, OSM parsing, edge weights,
caching, offline fallback, polyline export, coordinate conversion,
intersection detection, and degenerate-input handling.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import pytest

from tritium_lib.tracking.street_graph import (
    StreetGraph,
    _distance,
    _latlng_to_local,
    _node_key,
)


# ---------------------------------------------------------------------------
# Fixtures / mock data
# ---------------------------------------------------------------------------

# Realistic Overpass response for a small area with roads.
# Three roads forming an intersection network:
#   Oak Street: runs north-south at lng -122.4195
#   Elm Avenue: runs east-west at lat 37.7752
#   Main Road: runs east-west at lat 37.7754
# Intersection at (37.7752, -122.4195) connects Oak and Elm.
# Intersection at (37.7754, -122.4195) connects Oak and Main.
_MOCK_OVERPASS_ROADS = [
    {
        "type": "way",
        "id": 100001,
        "tags": {"highway": "residential", "name": "Oak Street"},
        "geometry": [
            {"lat": 37.7750, "lon": -122.4195},
            {"lat": 37.7752, "lon": -122.4195},
            {"lat": 37.7754, "lon": -122.4195},
        ],
    },
    {
        "type": "way",
        "id": 100002,
        "tags": {"highway": "residential", "name": "Elm Avenue"},
        "geometry": [
            {"lat": 37.7752, "lon": -122.4195},
            {"lat": 37.7752, "lon": -122.4190},
            {"lat": 37.7752, "lon": -122.4185},
        ],
    },
    {
        "type": "way",
        "id": 100003,
        "tags": {"highway": "tertiary", "name": "Main Road"},
        "geometry": [
            {"lat": 37.7754, "lon": -122.4195},
            {"lat": 37.7754, "lon": -122.4190},
            {"lat": 37.7754, "lon": -122.4185},
        ],
    },
]

# Reference point matching typical config
REF_LAT = 37.7749
REF_LNG = -122.4194


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    return str(tmp_path / "street_cache")


def _make_loaded_graph(temp_cache_dir: str) -> StreetGraph:
    """Helper: create and load a StreetGraph with mock data."""
    with patch("tritium_lib.tracking.street_graph._fetch_roads") as mock_fetch:
        mock_fetch.return_value = _MOCK_OVERPASS_ROADS
        sg = StreetGraph()
        sg.load(REF_LAT, REF_LNG, radius_m=300, cache_dir=temp_cache_dir)
    return sg


def _make_graph_from_elements() -> StreetGraph:
    """Helper: create a StreetGraph using load_from_elements."""
    sg = StreetGraph()
    sg.load_from_elements(_MOCK_OVERPASS_ROADS, REF_LAT, REF_LNG)
    return sg


# ---------------------------------------------------------------------------
# 1. Graph construction
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGraphConstruction:
    """Graph loads correctly from mock OSM data."""

    def test_graph_has_nodes(self, temp_cache_dir):
        """After loading, the graph should contain nodes."""
        sg = _make_loaded_graph(temp_cache_dir)
        assert sg.graph is not None
        assert sg.node_count > 0, "Graph should have at least one node"

    def test_graph_has_edges(self, temp_cache_dir):
        """After loading, the graph should contain edges."""
        sg = _make_loaded_graph(temp_cache_dir)
        assert sg.graph is not None
        assert sg.edge_count > 0, "Graph should have at least one edge"

    def test_node_count_matches_unique_points(self):
        """Node count should equal the number of unique rounded coordinates."""
        sg = _make_graph_from_elements()
        # 3 roads: Oak has 3 pts, Elm has 3 pts (1 shared with Oak), Main has 3 pts (1 shared with Oak)
        # Unique points: 3 + 2 + 2 = 7
        assert sg.node_count == 7

    def test_load_from_elements_works(self):
        """load_from_elements should produce the same graph structure as load()."""
        sg = _make_graph_from_elements()
        assert sg.graph is not None
        assert sg.node_count > 0
        assert sg.edge_count > 0


# ---------------------------------------------------------------------------
# 2. Edge weights
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEdgeWeights:
    """Edge weights represent distances in meters."""

    def test_all_edges_have_positive_weights(self):
        sg = _make_graph_from_elements()
        for u, v, data in sg.graph.edges(data=True):
            assert "weight" in data, f"Edge ({u}, {v}) missing weight"
            assert data["weight"] > 0, f"Edge ({u}, {v}) weight should be positive"

    def test_edge_weights_are_reasonable(self):
        """Road segments in a 300m radius should be < 600m."""
        sg = _make_graph_from_elements()
        for u, v, data in sg.graph.edges(data=True):
            assert data["weight"] < 600, f"Edge ({u}, {v}) weight unreasonably large: {data['weight']}"

    def test_edge_weight_approximate_distance(self):
        """Edge weight should roughly match haversine distance between endpoints."""
        sg = _make_graph_from_elements()
        for u, v, data in sg.graph.edges(data=True):
            p1 = sg._node_positions[u]
            p2 = sg._node_positions[v]
            euclidean = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            # Weight should be the euclidean distance
            assert abs(data["weight"] - euclidean) < 0.5, (
                f"Edge ({u},{v}) weight {data['weight']:.1f} vs euclidean {euclidean:.1f}"
            )


# ---------------------------------------------------------------------------
# 3. Intersection detection
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIntersectionDetection:
    """Roads that share a lat/lng point create connected graph nodes."""

    def test_graph_is_connected(self):
        """All three roads share intersections, so graph should be connected."""
        sg = _make_graph_from_elements()
        assert nx.is_connected(sg.graph), "Graph should be connected via shared intersections"

    def test_shared_point_merges_nodes(self):
        """Two roads sharing the same coordinate should share the same node."""
        sg = _make_graph_from_elements()
        # Oak and Elm share (37.7752, -122.4195). The graph should have
        # a node where degree >= 3 (Oak-south, Oak-north, Elm-east).
        max_degree = max(dict(sg.graph.degree()).values())
        assert max_degree >= 3, "Intersection node should connect at least 3 edges"


# ---------------------------------------------------------------------------
# 4. A* pathfinding
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAStarPathfinding:
    """Shortest path via A* returns valid waypoints."""

    def test_path_between_distant_points(self):
        """A path between two distant points should have multiple waypoints."""
        sg = _make_graph_from_elements()
        start = (-8.9, 11.0)   # near Oak Street south
        end = (70.0, 55.0)     # near Main Road east
        path = sg.shortest_path(start, end)
        assert path is not None, "Should find a path"
        assert len(path) >= 2, f"Path should have >= 2 waypoints, got {len(path)}"

    def test_path_returns_none_when_no_graph(self):
        """If graph is not loaded, shortest_path returns None."""
        sg = StreetGraph()
        assert sg.shortest_path((0, 0), (10, 10)) is None

    def test_same_start_and_end(self):
        """When start and end snap to the same node, path has one waypoint."""
        sg = _make_graph_from_elements()
        # Both points near Oak Street at 37.7750
        path = sg.shortest_path((-8.9, 11.0), (-8.8, 11.1))
        assert path is not None
        assert len(path) >= 1

    def test_path_waypoints_are_on_graph(self):
        """All path waypoints should be valid node positions."""
        sg = _make_graph_from_elements()
        start = (-8.9, 11.0)
        end = (70.0, 55.0)
        path = sg.shortest_path(start, end)
        if path is None:
            pytest.skip("No path found")
        for wp in path:
            _, dist = sg.nearest_node(wp[0], wp[1])
            assert dist < 1.0, f"Waypoint ({wp[0]:.1f}, {wp[1]:.1f}) not on graph: {dist:.1f}m"


# ---------------------------------------------------------------------------
# 5. Nearest node
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNearestNode:
    """nearest_node returns correct results."""

    def test_nearest_node_on_road(self):
        """A point near a known road should have a close nearest node."""
        sg = _make_graph_from_elements()
        node_id, dist = sg.nearest_node(-8.9, 11.0)
        assert node_id is not None
        assert dist < 20.0, f"Expected < 20m, got {dist:.1f}m"

    def test_nearest_node_empty_graph(self):
        """If graph not loaded, returns (None, inf)."""
        sg = StreetGraph()
        node_id, dist = sg.nearest_node(0.0, 0.0)
        assert node_id is None
        assert dist == float("inf")

    def test_nearest_node_exact_position(self):
        """A point exactly at a node position should have distance ~0."""
        sg = _make_graph_from_elements()
        # Get an actual node position
        some_nid = list(sg._node_positions.keys())[0]
        pos = sg._node_positions[some_nid]
        found_id, dist = sg.nearest_node(pos[0], pos[1])
        assert found_id == some_nid
        assert dist < 0.01


# ---------------------------------------------------------------------------
# 6. OSM parsing edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOSMParsing:
    """Handles various OSM element shapes."""

    def test_skips_non_way_elements(self):
        """Non-way elements (nodes, relations) should be ignored."""
        elements = [
            {"type": "node", "id": 1, "lat": 37.775, "lon": -122.419},
            {
                "type": "way",
                "id": 2,
                "tags": {"highway": "residential"},
                "geometry": [
                    {"lat": 37.775, "lon": -122.419},
                    {"lat": 37.776, "lon": -122.419},
                ],
            },
        ]
        sg = StreetGraph()
        sg.load_from_elements(elements, REF_LAT, REF_LNG)
        assert sg.graph is not None
        assert sg.edge_count == 1  # Only the way should produce an edge

    def test_skips_single_point_ways(self):
        """Ways with only one geometry point should be skipped."""
        elements = [
            {
                "type": "way",
                "id": 1,
                "tags": {"highway": "residential"},
                "geometry": [{"lat": 37.775, "lon": -122.419}],
            },
        ]
        sg = StreetGraph()
        sg.load_from_elements(elements, REF_LAT, REF_LNG)
        assert sg.graph is None  # No valid edges

    def test_road_class_preserved(self):
        """Edge data should preserve the highway class from OSM tags."""
        sg = _make_graph_from_elements()
        classes = set()
        for _, _, data in sg.graph.edges(data=True):
            classes.add(data.get("road_class"))
        assert "residential" in classes
        assert "tertiary" in classes

    def test_missing_tags_defaults_to_residential(self):
        """Ways without tags should default to residential road class."""
        elements = [
            {
                "type": "way",
                "id": 1,
                "geometry": [
                    {"lat": 37.775, "lon": -122.419},
                    {"lat": 37.776, "lon": -122.419},
                ],
            },
        ]
        sg = StreetGraph()
        sg.load_from_elements(elements, REF_LAT, REF_LNG)
        assert sg.graph is not None
        for _, _, data in sg.graph.edges(data=True):
            assert data["road_class"] == "residential"


# ---------------------------------------------------------------------------
# 7. Caching
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCaching:
    """Cache saves and loads correctly."""

    def test_cache_saves_and_loads(self, temp_cache_dir):
        """After first load, cache file exists. Second load skips API."""
        call_count = 0

        def counting_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _MOCK_OVERPASS_ROADS

        with patch("tritium_lib.tracking.street_graph._fetch_roads", side_effect=counting_fetch):
            sg1 = StreetGraph()
            sg1.load(REF_LAT, REF_LNG, radius_m=300, cache_dir=temp_cache_dir)
            assert call_count == 1, "First load should call API"

            sg2 = StreetGraph()
            sg2.load(REF_LAT, REF_LNG, radius_m=300, cache_dir=temp_cache_dir)
            assert call_count == 1, "Second load should use cache"
            assert sg2.graph is not None
            assert sg2.node_count > 0

    def test_expired_cache_refetches(self, temp_cache_dir):
        """Expired cache (>24h) should trigger a re-fetch."""
        call_count = 0

        def counting_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _MOCK_OVERPASS_ROADS

        with patch("tritium_lib.tracking.street_graph._fetch_roads", side_effect=counting_fetch):
            sg = StreetGraph()
            sg.load(REF_LAT, REF_LNG, radius_m=300, cache_dir=temp_cache_dir)
            assert call_count == 1

            # Age the cache file
            cache_dir = Path(temp_cache_dir)
            for f in cache_dir.glob("*.pkl"):
                old_time = time.time() - 25 * 3600
                os.utime(f, (old_time, old_time))

            sg2 = StreetGraph()
            sg2.load(REF_LAT, REF_LNG, radius_m=300, cache_dir=temp_cache_dir)
            assert call_count == 2, "Expired cache should trigger re-fetch"


# ---------------------------------------------------------------------------
# 8. Offline fallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOfflineFallback:
    """Graceful degradation when API is unreachable."""

    def test_api_failure_returns_none(self, temp_cache_dir):
        with patch("tritium_lib.tracking.street_graph._fetch_roads", side_effect=ConnectionError("offline")):
            sg = StreetGraph()
            sg.load(REF_LAT, REF_LNG, radius_m=300, cache_dir=temp_cache_dir)
        assert sg.graph is None
        assert sg.nearest_node(0, 0) == (None, float("inf"))
        assert sg.shortest_path((0, 0), (10, 10)) is None

    def test_empty_response_returns_none(self, temp_cache_dir):
        with patch("tritium_lib.tracking.street_graph._fetch_roads", return_value=[]):
            sg = StreetGraph()
            sg.load(REF_LAT, REF_LNG, radius_m=300, cache_dir=temp_cache_dir)
        assert sg.graph is None


# ---------------------------------------------------------------------------
# 9. Polyline export
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPolylineExport:
    """to_polylines returns renderable line segments."""

    def test_polylines_not_empty(self):
        sg = _make_graph_from_elements()
        polylines = sg.to_polylines()
        assert len(polylines) > 0

    def test_polyline_structure(self):
        sg = _make_graph_from_elements()
        polylines = sg.to_polylines()
        for pl in polylines:
            assert "points" in pl
            assert "class" in pl
            assert len(pl["points"]) == 2
            for pt in pl["points"]:
                assert len(pt) == 2
                assert isinstance(pt[0], float)
                assert isinstance(pt[1], float)

    def test_polylines_empty_when_no_graph(self):
        sg = StreetGraph()
        assert sg.to_polylines() == []


# ---------------------------------------------------------------------------
# 10. Coordinate conversion helpers
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCoordinateHelpers:
    """Unit-level tests for helper functions."""

    def test_latlng_to_local_origin(self):
        """Reference point should map to (0, 0)."""
        x, y = _latlng_to_local(REF_LAT, REF_LNG, REF_LAT, REF_LNG)
        assert abs(x) < 0.01
        assert abs(y) < 0.01

    def test_latlng_to_local_north(self):
        """Moving north should increase y."""
        x, y = _latlng_to_local(REF_LAT + 0.001, REF_LNG, REF_LAT, REF_LNG)
        assert y > 100  # ~111m per 0.001 degree

    def test_latlng_to_local_east(self):
        """Moving east should increase x."""
        x, y = _latlng_to_local(REF_LAT, REF_LNG + 0.001, REF_LAT, REF_LNG)
        assert x > 50  # Less than 111m due to cos(lat)

    def test_distance_zero(self):
        assert _distance((0, 0), (0, 0)) == 0.0

    def test_distance_known(self):
        assert abs(_distance((0, 0), (3, 4)) - 5.0) < 0.001

    def test_node_key_rounds(self):
        assert _node_key(1.04, 2.06) == (1.0, 2.1)
        # 1.05 as float is slightly > 1.05, so round(1.05, 1) = 1.1
        assert _node_key(1.04, 2.04) == (1.0, 2.0)
        # Verify precision is 0.1m
        assert _node_key(1.149, 2.0) == (1.1, 2.0)
        assert _node_key(1.151, 2.0) == (1.2, 2.0)


# ---------------------------------------------------------------------------
# 11. Properties
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProperties:
    """node_count, edge_count, node_positions properties."""

    def test_node_count_zero_when_empty(self):
        sg = StreetGraph()
        assert sg.node_count == 0

    def test_edge_count_zero_when_empty(self):
        sg = StreetGraph()
        assert sg.edge_count == 0

    def test_node_positions_returns_copy(self):
        sg = _make_graph_from_elements()
        pos = sg.node_positions
        assert len(pos) == sg.node_count
        # Modifying the returned dict should not affect the internal state
        pos[9999] = (0.0, 0.0)
        assert 9999 not in sg._node_positions
