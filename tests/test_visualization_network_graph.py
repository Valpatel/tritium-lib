# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.visualization.network_graph — nodes + edges."""

import math

import pytest

from tritium_lib.visualization.network_graph import (
    GraphEdge,
    GraphNode,
    NetworkGraph,
)


class TestGraphNode:
    def test_basic(self):
        n = GraphNode(node_id="n1", label="Node 1", group="ble")
        assert n.node_id == "n1"
        assert n.label == "Node 1"
        assert n.group == "ble"

    def test_to_dict(self):
        n = GraphNode(node_id="n1", label="N", group="wifi", x=10.0, y=20.0)
        d = n.to_dict()
        assert d["id"] == "n1"
        assert d["label"] == "N"
        assert d["group"] == "wifi"
        assert d["x"] == 10.0

    def test_to_dict_default_label(self):
        n = GraphNode(node_id="n1")
        d = n.to_dict()
        assert d["label"] == "n1"


class TestGraphEdge:
    def test_basic(self):
        e = GraphEdge(source="a", target="b", label="related", weight=0.8)
        assert e.source == "a"
        assert e.target == "b"
        assert e.weight == 0.8

    def test_to_dict(self):
        e = GraphEdge(source="a", target="b", label="co_located")
        d = e.to_dict()
        assert d["source"] == "a"
        assert d["target"] == "b"
        assert d["label"] == "co_located"


class TestNetworkGraph:
    def test_empty(self):
        g = NetworkGraph()
        assert g.node_count == 0
        assert g.edge_count == 0
        assert len(g) == 0
        assert bool(g) is False

    def test_add_node(self):
        g = NetworkGraph()
        node = g.add_node("n1", label="Node 1", group="ble")
        assert node.node_id == "n1"
        assert g.node_count == 1
        assert bool(g) is True

    def test_add_edge(self):
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        edge = g.add_edge("a", "b", label="detected_with", weight=0.9)
        assert edge.source == "a"
        assert g.edge_count == 1

    def test_add_edge_missing_source(self):
        g = NetworkGraph()
        g.add_node("b")
        with pytest.raises(ValueError, match="Source"):
            g.add_edge("a", "b")

    def test_add_edge_missing_target(self):
        g = NetworkGraph()
        g.add_node("a")
        with pytest.raises(ValueError, match="Target"):
            g.add_edge("a", "b")

    def test_remove_node(self):
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.remove_node("a")
        assert g.node_count == 1
        assert g.edge_count == 0

    def test_remove_edge(self):
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.remove_edge("a", "b")
        assert g.edge_count == 0
        assert g.node_count == 2

    def test_clear(self):
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.clear()
        assert g.node_count == 0
        assert g.edge_count == 0

    def test_get_node(self):
        g = NetworkGraph()
        g.add_node("a", label="Alpha")
        n = g.get_node("a")
        assert n is not None
        assert n.label == "Alpha"
        assert g.get_node("missing") is None

    def test_neighbors_undirected(self):
        g = NetworkGraph(directed=False)
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b")
        g.add_edge("c", "a")
        nbrs = g.neighbors("a")
        assert "b" in nbrs
        assert "c" in nbrs

    def test_neighbors_directed(self):
        g = NetworkGraph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        assert g.neighbors("a") == ["b"]
        assert g.neighbors("b") == []

    def test_degree(self):
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        assert g.degree("a") == 2
        assert g.degree("b") == 1

    def test_groups(self):
        g = NetworkGraph()
        g.add_node("a", group="ble")
        g.add_node("b", group="wifi")
        g.add_node("c", group="ble")
        assert g.groups == ["ble", "wifi"]

    def test_circular_layout(self):
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.circular_layout(radius=100.0, cx=0.0, cy=0.0)
        nodes = g.nodes
        for n in nodes:
            dist = math.sqrt(n.x ** 2 + n.y ** 2)
            assert abs(dist - 100.0) < 0.01

    def test_circular_layout_empty(self):
        g = NetworkGraph()
        g.circular_layout()  # should not crash

    def test_to_dict(self):
        g = NetworkGraph(title="Test Graph", directed=True)
        g.add_node("a", label="A")
        g.add_node("b", label="B")
        g.add_edge("a", "b", label="link")
        d = g.to_dict()
        assert d["title"] == "Test Graph"
        assert d["directed"] is True
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1

    def test_from_dict(self):
        data = {
            "title": "Restored",
            "directed": False,
            "nodes": [
                {"id": "a", "label": "A", "group": "ble"},
                {"id": "b", "label": "B", "group": "wifi"},
            ],
            "edges": [
                {"source": "a", "target": "b", "label": "related", "weight": 0.5},
            ],
        }
        g = NetworkGraph.from_dict(data)
        assert g.title == "Restored"
        assert g.node_count == 2
        assert g.edge_count == 1

    def test_from_dict_missing_node(self):
        data = {
            "nodes": [{"id": "a"}],
            "edges": [{"source": "a", "target": "missing"}],
        }
        g = NetworkGraph.from_dict(data)
        assert g.node_count == 1
        assert g.edge_count == 0  # skipped

    def test_roundtrip(self):
        g = NetworkGraph(title="RT")
        g.add_node("a", group="ble")
        g.add_node("b", group="wifi")
        g.add_edge("a", "b", label="co_located", weight=0.75)
        restored = NetworkGraph.from_dict(g.to_dict())
        assert restored.title == "RT"
        assert restored.node_count == 2
        assert restored.edge_count == 1

    def test_to_vega_lite(self):
        g = NetworkGraph(title="VL Test")
        g.add_node("a", group="ble")
        g.add_node("b", group="wifi")
        g.add_edge("a", "b")
        spec = g.to_vega_lite()
        assert spec["title"] == "VL Test"
        assert len(spec["layer"]) == 3  # edges, nodes, labels

    def test_to_vega_lite_json(self):
        g = NetworkGraph()
        g.add_node("a")
        j = g.to_vega_lite_json()
        assert '"title"' in j

    def test_to_svg_empty(self):
        g = NetworkGraph()
        svg = g.to_svg()
        assert "No nodes" in svg

    def test_to_svg(self):
        g = NetworkGraph(title="SVG Graph")
        g.add_node("a", group="ble")
        g.add_node("b", group="wifi")
        g.add_edge("a", "b", label="link")
        svg = g.to_svg()
        assert "<svg" in svg
        assert "SVG Graph" in svg
        assert "<circle" in svg
        assert "<line" in svg

    def test_nodes_property(self):
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        assert len(g.nodes) == 2

    def test_edges_property(self):
        g = NetworkGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        assert len(g.edges) == 1
