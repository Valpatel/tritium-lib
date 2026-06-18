# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SidewalkGraph.build_from_street_graph — derive curb-side sidewalks from roads.

The geospatial pipeline builds sidewalks from segmented imagery; the common
synthetic / offline sim has only a road StreetGraph.  This builder lays
pedestrian sidewalk runs offset from the road centreline so peds prefer the
curb lane instead of beelining across car lanes — no imagery required.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph
from tritium_lib.models.terrain import TerrainType


class _SG:
    """Minimal StreetGraph-shaped object: .graph (networkx, x/y node attrs)
    + ._node_positions, edges tagged with road_class."""

    def __init__(self) -> None:
        self.graph = nx.Graph()
        self._node_positions: dict[int, tuple[float, float]] = {}


def _grid(span: float = 200.0, n: int = 5, road_class: str = "residential") -> _SG:
    sg = _SG()
    step = (2 * span) / (n - 1)
    ids: dict[tuple[int, int], int] = {}
    nid = 0
    for r in range(n):
        for c in range(n):
            x = -span + c * step
            y = -span + r * step
            sg.graph.add_node(nid, x=x, y=y)
            sg._node_positions[nid] = (x, y)
            ids[(r, c)] = nid
            nid += 1

    def add(a: int, b: int) -> None:
        ax, ay = sg._node_positions[a]
        bx, by = sg._node_positions[b]
        sg.graph.add_edge(a, b, weight=math.hypot(ax - bx, ay - by),
                          road_class=road_class)

    for r in range(n):
        for c in range(n - 1):
            add(ids[(r, c)], ids[(r, c + 1)])
    for c in range(n):
        for r in range(n - 1):
            add(ids[(r, c)], ids[(r + 1, c)])
    return sg, ids


def _road_edges(sg: _SG):
    return [(sg._node_positions[a], sg._node_positions[b])
            for a, b in sg.graph.edges()]


def _dist_seg(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class TestBuildFromStreetGraph:

    def test_builds_connected_graph(self):
        sg, _ids = _grid()
        sw = SidewalkGraph()
        created = sw.build_from_street_graph(sg, offset=4.0, sample_interval=12.0)
        assert created > 0
        assert sw.node_count == created
        assert sw.edge_count > 0

    def test_find_path_returns_multi_waypoint_route(self):
        sg, _ids = _grid()
        sw = SidewalkGraph()
        sw.build_from_street_graph(sg)
        path = sw.find_path((-200.0, -200.0), (200.0, 200.0))
        assert path is not None
        assert len(path) > 2  # genuine routed path, not a bare [start, end]
        assert path[0] == (-200.0, -200.0)
        assert path[-1] == (200.0, 200.0)

    def test_interior_waypoints_offset_from_car_centreline(self):
        """Sidewalk nodes should sit on the curb, ~offset from a centreline."""
        sg, _ids = _grid()
        roads = _road_edges(sg)
        sw = SidewalkGraph()
        offset = 4.0
        sw.build_from_street_graph(sg, offset=offset, sample_interval=12.0)
        # Sample many interior route waypoints; most should be off the lane
        # (> 3m from any centreline), unlike a centreline route.
        import random
        rng = random.Random(3)
        off_lane = total = 0
        for _ in range(40):
            s = (rng.uniform(-180, 180), rng.uniform(-180, 180))
            e = (rng.uniform(-180, 180), rng.uniform(-180, 180))
            path = sw.find_path(s, e)
            for wp in path[1:-1]:  # interior nodes are graph nodes (on curbs)
                total += 1
                dmin = min(_dist_seg(wp[0], wp[1], a[0], a[1], b[0], b[1])
                           for (a, b) in roads)
                if dmin > 3.0:
                    off_lane += 1
        assert total > 0
        assert off_lane / total > 0.7, off_lane / total

    def test_pedestrian_class_edges_on_centreline(self):
        """footway/path edges already ARE walkways -> laid on the centreline."""
        sg, _ids = _grid(road_class="footway")
        roads = _road_edges(sg)
        sw = SidewalkGraph()
        sw.build_from_street_graph(sg, offset=4.0)
        # All sidewalk nodes should lie ON a footway centreline (offset 0).
        on_line = 0
        nodes = list(sw._nodes.values())
        for node in nodes:
            dmin = min(_dist_seg(node.x, node.y, a[0], a[1], b[0], b[1])
                       for (a, b) in roads)
            if dmin < 0.5:
                on_line += 1
        assert on_line == len(nodes), (on_line, len(nodes))

    def test_nodes_are_sidewalk_terrain(self):
        sg, _ids = _grid()
        sw = SidewalkGraph()
        sw.build_from_street_graph(sg)
        assert all(n.terrain_type == TerrainType.SIDEWALK
                   for n in sw._nodes.values())

    def test_none_or_empty_graph_returns_zero(self):
        assert SidewalkGraph().build_from_street_graph(None) == 0
        assert SidewalkGraph().build_from_street_graph(object()) == 0
        empty = _SG()
        assert SidewalkGraph().build_from_street_graph(empty) == 0

    def test_uses_node_xy_attrs_without_node_positions(self):
        """Falls back to networkx node x/y attrs when _node_positions absent."""
        sg, _ids = _grid()
        sg._node_positions = {}  # force the attr path
        sw = SidewalkGraph()
        created = sw.build_from_street_graph(sg)
        assert created > 0
