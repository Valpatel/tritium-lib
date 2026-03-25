# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.network — network topology discovery and analysis."""

import pytest

from tritium_lib.network import (
    NetworkNode,
    NetworkLink,
    NetworkTopology,
    TopologyDiscovery,
    PathAnalysis,
    NodeType,
    LinkType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(nid: str, ntype: str = "edge_device", **kw) -> NetworkNode:
    return NetworkNode(node_id=nid, node_type=ntype, **kw)


def _link(src: str, tgt: str, ltype: str = "mqtt", **kw) -> NetworkLink:
    return NetworkLink(source_id=src, target_id=tgt, link_type=ltype, **kw)


def _simple_topology() -> NetworkTopology:
    """Create a small test topology:  A -- B -- C -- D  (linear chain)."""
    topo = NetworkTopology()
    for nid in ["A", "B", "C", "D"]:
        topo.add_node(_node(nid))
    topo.add_link(_link("A", "B"))
    topo.add_link(_link("B", "C"))
    topo.add_link(_link("C", "D"))
    return topo


def _star_topology() -> NetworkTopology:
    """Hub-and-spoke: broker in center, 4 edge devices around it."""
    topo = NetworkTopology()
    topo.add_node(_node("broker", "mqtt_broker", ip="192.168.1.1"))
    for i in range(1, 5):
        topo.add_node(_node(f"edge-{i}", "edge_device", ip=f"192.168.1.{10+i}"))
    for i in range(1, 5):
        topo.add_link(_link(f"edge-{i}", "broker"))
    return topo


# ---------------------------------------------------------------------------
# NetworkNode tests
# ---------------------------------------------------------------------------

class TestNetworkNode:
    def test_create_minimal(self):
        n = NetworkNode(node_id="n1")
        assert n.node_id == "n1"
        assert n.node_type == "edge_device"
        assert n.name == "n1"  # defaults to node_id
        assert n.online is True
        assert n.capabilities == []
        assert n.metadata == {}

    def test_create_full(self):
        n = NetworkNode(
            node_id="cam-01",
            node_type="camera",
            ip="10.0.0.5",
            name="Front Door Camera",
            capabilities=["rtsp", "onvif"],
            online=False,
            metadata={"firmware": "2.1.0"},
        )
        assert n.name == "Front Door Camera"
        assert n.node_type == "camera"
        assert n.ip == "10.0.0.5"
        assert n.online is False
        assert "rtsp" in n.capabilities
        assert n.metadata["firmware"] == "2.1.0"

    def test_name_defaults_to_id(self):
        n = NetworkNode(node_id="sensor-42")
        assert n.name == "sensor-42"

    def test_name_override(self):
        n = NetworkNode(node_id="x", name="Custom Name")
        assert n.name == "Custom Name"


# ---------------------------------------------------------------------------
# NetworkLink tests
# ---------------------------------------------------------------------------

class TestNetworkLink:
    def test_create_minimal(self):
        lnk = NetworkLink(source_id="a", target_id="b")
        assert lnk.link_type == "mqtt"
        assert lnk.active is True
        assert lnk.bandwidth_mbps is None
        assert lnk.latency_ms is None
        assert lnk.packet_loss_pct == 0.0

    def test_weight_default(self):
        """With no latency, weight defaults to 1.0."""
        lnk = NetworkLink(source_id="a", target_id="b")
        assert lnk.weight == 1.0

    def test_weight_with_latency(self):
        lnk = NetworkLink(source_id="a", target_id="b", latency_ms=10.0)
        assert lnk.weight == 10.0

    def test_weight_with_loss(self):
        lnk = NetworkLink(source_id="a", target_id="b", latency_ms=10.0,
                          packet_loss_pct=50.0)
        assert lnk.weight == pytest.approx(15.0)

    def test_metadata(self):
        lnk = NetworkLink(source_id="a", target_id="b",
                          metadata={"vlan": "100"})
        assert lnk.metadata["vlan"] == "100"


# ---------------------------------------------------------------------------
# NetworkTopology tests
# ---------------------------------------------------------------------------

class TestNetworkTopology:
    def test_add_and_get_node(self):
        topo = NetworkTopology()
        topo.add_node(_node("n1"))
        assert topo.get_node("n1") is not None
        assert topo.get_node("n1").node_id == "n1"

    def test_get_missing_node(self):
        topo = NetworkTopology()
        assert topo.get_node("missing") is None

    def test_add_link_validates_endpoints(self):
        topo = NetworkTopology()
        topo.add_node(_node("a"))
        with pytest.raises(KeyError, match="Target node"):
            topo.add_link(_link("a", "missing"))
        with pytest.raises(KeyError, match="Source node"):
            topo.add_link(_link("missing", "a"))

    def test_node_ids_sorted(self):
        topo = NetworkTopology()
        topo.add_node(_node("c"))
        topo.add_node(_node("a"))
        topo.add_node(_node("b"))
        assert topo.node_ids == ["a", "b", "c"]

    def test_neighbors(self):
        topo = _simple_topology()
        assert topo.neighbors("A") == ["B"]
        assert topo.neighbors("B") == ["A", "C"]
        assert topo.neighbors("D") == ["C"]

    def test_neighbors_ignores_inactive(self):
        topo = NetworkTopology()
        topo.add_node(_node("a"))
        topo.add_node(_node("b"))
        topo.add_node(_node("c"))
        topo.add_link(_link("a", "b"))
        topo.add_link(_link("a", "c", active=False))
        assert topo.neighbors("a") == ["b"]

    def test_remove_node_cascades_links(self):
        topo = _simple_topology()
        removed = topo.remove_node("B")
        assert removed is True
        assert topo.get_node("B") is None
        assert len(topo.links) == 1  # only C--D remains
        assert topo.neighbors("A") == []

    def test_remove_missing_node(self):
        topo = NetworkTopology()
        assert topo.remove_node("ghost") is False

    def test_remove_link(self):
        topo = _simple_topology()
        count = topo.remove_link("A", "B")
        assert count == 1
        assert topo.neighbors("A") == []

    def test_remove_link_bidirectional(self):
        """remove_link should work regardless of direction."""
        topo = NetworkTopology()
        topo.add_node(_node("x"))
        topo.add_node(_node("y"))
        topo.add_link(_link("x", "y"))
        # Remove in the opposite direction
        count = topo.remove_link("y", "x")
        assert count == 1

    def test_links_for_node(self):
        topo = _star_topology()
        broker_links = topo.links_for("broker")
        assert len(broker_links) == 4

    def test_nodes_by_type(self):
        topo = _star_topology()
        brokers = topo.nodes_by_type("mqtt_broker")
        edges = topo.nodes_by_type("edge_device")
        assert len(brokers) == 1
        assert len(edges) == 4

    def test_connected_components_single(self):
        topo = _simple_topology()
        comps = topo.connected_components()
        assert len(comps) == 1
        assert comps[0] == ["A", "B", "C", "D"]

    def test_connected_components_multiple(self):
        topo = NetworkTopology()
        for nid in ["a", "b", "c", "d"]:
            topo.add_node(_node(nid))
        topo.add_link(_link("a", "b"))
        topo.add_link(_link("c", "d"))
        comps = topo.connected_components()
        assert len(comps) == 2
        assert ["a", "b"] in comps
        assert ["c", "d"] in comps

    def test_is_connected(self):
        topo = _simple_topology()
        assert topo.is_connected() is True

    def test_is_not_connected(self):
        topo = NetworkTopology()
        topo.add_node(_node("a"))
        topo.add_node(_node("b"))
        assert topo.is_connected() is False

    def test_empty_topology(self):
        topo = NetworkTopology()
        assert topo.node_ids == []
        assert topo.links == []
        assert topo.connected_components() == []
        assert topo.summary()["total_nodes"] == 0

    def test_summary(self):
        topo = _star_topology()
        s = topo.summary()
        assert s["total_nodes"] == 5
        assert s["total_links"] == 4
        assert s["active_links"] == 4
        assert s["is_connected"] is True
        assert s["connected_components"] == 1
        assert s["node_types"]["mqtt_broker"] == 1
        assert s["node_types"]["edge_device"] == 4


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestTopologySerialization:
    def test_to_dict_roundtrip(self):
        topo = _star_topology()
        data = topo.to_dict()
        assert len(data["nodes"]) == 5
        assert len(data["links"]) == 4

        restored = NetworkTopology.from_dict(data)
        assert sorted(restored.node_ids) == sorted(topo.node_ids)
        assert len(restored.links) == len(topo.links)

    def test_to_graphviz_format(self):
        topo = _star_topology()
        dot = topo.to_graphviz()
        assert dot.startswith('graph "Tritium Network Topology" {')
        assert "broker" in dot
        assert "edge-1" in dot
        assert "mqtt" in dot  # link label
        assert dot.endswith("}")

    def test_to_graphviz_custom_title(self):
        topo = NetworkTopology()
        topo.add_node(_node("x"))
        dot = topo.to_graphviz(title="Test Net")
        assert 'graph "Test Net" {' in dot

    def test_to_graphviz_inactive_link_dashed(self):
        topo = NetworkTopology()
        topo.add_node(_node("a"))
        topo.add_node(_node("b"))
        topo.add_link(_link("a", "b", active=False))
        dot = topo.to_graphviz()
        assert "style=dashed" in dot

    def test_to_graphviz_offline_node_dashed(self):
        topo = NetworkTopology()
        topo.add_node(_node("off", online=False))
        dot = topo.to_graphviz()
        assert "dashed,filled" in dot


# ---------------------------------------------------------------------------
# TopologyDiscovery tests
# ---------------------------------------------------------------------------

class TestTopologyDiscovery:
    def test_build_basic(self):
        disc = TopologyDiscovery()
        disc.add_node(_node("a"))
        disc.add_node(_node("b"))
        disc.add_link(_link("a", "b"))
        topo = disc.build()
        assert len(topo.node_ids) == 2
        assert len(topo.links) == 1

    def test_auto_discover_links(self):
        disc = TopologyDiscovery()
        disc.add_node(_node("e1", capabilities=["wifi", "ble"]))
        disc.add_node(_node("e2", capabilities=["wifi"]))
        disc.add_node(_node("e3", capabilities=["ble"]))
        count = disc.auto_discover_links()
        # e1-e2: wifi, e1-e3: ble = 2 links
        assert count == 2
        topo = disc.build()
        assert len(topo.links) == 2

    def test_auto_discover_no_shared_caps(self):
        disc = TopologyDiscovery()
        disc.add_node(_node("e1", capabilities=["wifi"]))
        disc.add_node(_node("e2", capabilities=["ble"]))
        count = disc.auto_discover_links()
        assert count == 0

    def test_discover_from_broker(self):
        disc = TopologyDiscovery()
        disc.add_node(_node("broker", "mqtt_broker"))
        disc.add_node(_node("d1"))
        disc.add_node(_node("d2"))
        disc.add_node(_node("d3"))
        count = disc.discover_from_broker("broker", ["d1", "d2", "d3"])
        assert count == 3
        topo = disc.build()
        assert len(topo.links) == 3
        assert topo.neighbors("broker") == ["d1", "d2", "d3"]

    def test_discover_from_broker_missing_broker(self):
        disc = TopologyDiscovery()
        disc.add_node(_node("d1"))
        count = disc.discover_from_broker("no-broker", ["d1"])
        assert count == 0

    def test_discover_from_broker_ignores_unknown_devices(self):
        disc = TopologyDiscovery()
        disc.add_node(_node("broker", "mqtt_broker"))
        disc.add_node(_node("d1"))
        count = disc.discover_from_broker("broker", ["d1", "unknown-device"])
        assert count == 1

    def test_build_drops_orphan_links(self):
        disc = TopologyDiscovery()
        disc.add_node(_node("a"))
        disc.add_link(_link("a", "ghost"))  # ghost node not added
        topo = disc.build()
        assert len(topo.links) == 0  # link dropped because ghost doesn't exist


# ---------------------------------------------------------------------------
# PathAnalysis tests
# ---------------------------------------------------------------------------

class TestPathAnalysisShortestPath:
    def test_direct_neighbor(self):
        topo = _simple_topology()
        pa = PathAnalysis(topo)
        path = pa.shortest_path("A", "B")
        assert path == ["A", "B"]

    def test_multi_hop(self):
        topo = _simple_topology()
        pa = PathAnalysis(topo)
        path = pa.shortest_path("A", "D")
        assert path == ["A", "B", "C", "D"]

    def test_self_path(self):
        topo = _simple_topology()
        pa = PathAnalysis(topo)
        assert pa.shortest_path("A", "A") == ["A"]

    def test_unreachable(self):
        topo = NetworkTopology()
        topo.add_node(_node("a"))
        topo.add_node(_node("b"))
        pa = PathAnalysis(topo)
        assert pa.shortest_path("a", "b") is None

    def test_unknown_source(self):
        topo = _simple_topology()
        pa = PathAnalysis(topo)
        assert pa.shortest_path("ZZZ", "A") is None


class TestPathAnalysisWeighted:
    def test_weighted_path(self):
        topo = NetworkTopology()
        for nid in ["a", "b", "c"]:
            topo.add_node(_node(nid))
        # Direct a->c is slow (latency 100)
        topo.add_link(NetworkLink(source_id="a", target_id="c",
                                  link_type="wifi", latency_ms=100.0))
        # Via b is faster (5 + 5 = 10)
        topo.add_link(NetworkLink(source_id="a", target_id="b",
                                  link_type="ethernet", latency_ms=5.0))
        topo.add_link(NetworkLink(source_id="b", target_id="c",
                                  link_type="ethernet", latency_ms=5.0))
        pa = PathAnalysis(topo)
        result = pa.weighted_shortest_path("a", "c")
        assert result is not None
        path, cost = result
        assert path == ["a", "b", "c"]
        assert cost == pytest.approx(10.0)

    def test_weighted_self(self):
        topo = _simple_topology()
        pa = PathAnalysis(topo)
        result = pa.weighted_shortest_path("A", "A")
        assert result == (["A"], 0.0)

    def test_weighted_unreachable(self):
        topo = NetworkTopology()
        topo.add_node(_node("a"))
        topo.add_node(_node("b"))
        pa = PathAnalysis(topo)
        assert pa.weighted_shortest_path("a", "b") is None

    def test_weighted_unknown_node(self):
        topo = _simple_topology()
        pa = PathAnalysis(topo)
        assert pa.weighted_shortest_path("ZZZ", "A") is None


class TestPathAnalysisAllPaths:
    def test_all_paths_linear(self):
        topo = _simple_topology()
        pa = PathAnalysis(topo)
        paths = pa.all_paths("A", "D")
        assert len(paths) == 1
        assert paths[0] == ["A", "B", "C", "D"]

    def test_all_paths_with_cycle(self):
        topo = NetworkTopology()
        for nid in ["a", "b", "c"]:
            topo.add_node(_node(nid))
        topo.add_link(_link("a", "b"))
        topo.add_link(_link("b", "c"))
        topo.add_link(_link("a", "c"))
        pa = PathAnalysis(topo)
        paths = pa.all_paths("a", "c")
        assert len(paths) == 2
        assert ["a", "c"] in paths
        assert ["a", "b", "c"] in paths

    def test_all_paths_self(self):
        topo = _simple_topology()
        pa = PathAnalysis(topo)
        assert pa.all_paths("A", "A") == [["A"]]

    def test_all_paths_unreachable(self):
        topo = NetworkTopology()
        topo.add_node(_node("a"))
        topo.add_node(_node("b"))
        pa = PathAnalysis(topo)
        assert pa.all_paths("a", "b") == []


class TestPathAnalysisBottlenecks:
    def test_linear_chain_bottlenecks(self):
        """In A--B--C--D, B and C are articulation points."""
        topo = _simple_topology()
        pa = PathAnalysis(topo)
        bottlenecks = pa.find_bottlenecks()
        assert "B" in bottlenecks
        assert "C" in bottlenecks
        assert "A" not in bottlenecks
        assert "D" not in bottlenecks

    def test_star_bottleneck(self):
        """In a star, the hub is the only bottleneck."""
        topo = _star_topology()
        pa = PathAnalysis(topo)
        bottlenecks = pa.find_bottlenecks()
        assert bottlenecks == ["broker"]

    def test_complete_graph_no_bottlenecks(self):
        """A fully connected graph has no articulation points."""
        topo = NetworkTopology()
        for nid in ["a", "b", "c"]:
            topo.add_node(_node(nid))
        topo.add_link(_link("a", "b"))
        topo.add_link(_link("b", "c"))
        topo.add_link(_link("a", "c"))
        pa = PathAnalysis(topo)
        assert pa.find_bottlenecks() == []

    def test_empty_no_bottlenecks(self):
        topo = NetworkTopology()
        pa = PathAnalysis(topo)
        assert pa.find_bottlenecks() == []


class TestPathAnalysisBridges:
    def test_linear_bridges(self):
        """All links in a linear chain are bridges."""
        topo = _simple_topology()
        pa = PathAnalysis(topo)
        bridges = pa.find_bridges()
        assert len(bridges) == 3

    def test_triangle_no_bridges(self):
        topo = NetworkTopology()
        for nid in ["a", "b", "c"]:
            topo.add_node(_node(nid))
        topo.add_link(_link("a", "b"))
        topo.add_link(_link("b", "c"))
        topo.add_link(_link("a", "c"))
        pa = PathAnalysis(topo)
        assert pa.find_bridges() == []


class TestPathAnalysisMisc:
    def test_hop_count(self):
        topo = _simple_topology()
        pa = PathAnalysis(topo)
        assert pa.hop_count("A", "D") == 3
        assert pa.hop_count("A", "B") == 1
        assert pa.hop_count("A", "A") == 0

    def test_hop_count_unreachable(self):
        topo = NetworkTopology()
        topo.add_node(_node("a"))
        topo.add_node(_node("b"))
        pa = PathAnalysis(topo)
        assert pa.hop_count("a", "b") is None

    def test_node_centrality(self):
        topo = _star_topology()  # broker has degree 4 in 5-node graph
        pa = PathAnalysis(topo)
        c = pa.node_centrality()
        assert c["broker"] == pytest.approx(1.0)  # 4/(5-1)
        assert c["edge-1"] == pytest.approx(0.25)  # 1/(5-1)

    def test_node_centrality_single(self):
        topo = NetworkTopology()
        topo.add_node(_node("solo"))
        pa = PathAnalysis(topo)
        c = pa.node_centrality()
        assert c["solo"] == 0.0


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_node_type_values(self):
        assert NodeType.EDGE_DEVICE.value == "edge_device"
        assert NodeType.MQTT_BROKER.value == "mqtt_broker"
        assert NodeType.SENSOR.value == "sensor"

    def test_link_type_values(self):
        assert LinkType.MQTT.value == "mqtt"
        assert LinkType.WIFI.value == "wifi"
        assert LinkType.LORA.value == "lora"
        assert LinkType.ETHERNET.value == "ethernet"
