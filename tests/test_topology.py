"""Tests for tritium_lib.models.topology."""

from datetime import datetime, timezone

from tritium_lib.models.topology import (
    NetworkLink,
    FleetTopology,
    ConnectivityReport,
    build_topology,
    analyze_connectivity,
)


def _link(src: str, tgt: str, transport: str = "wifi", active: bool = True) -> NetworkLink:
    """Helper to create a link with minimal boilerplate."""
    return NetworkLink(source_id=src, target_id=tgt, transport=transport, active=active)


# -- NetworkLink tests --------------------------------------------------------

class TestNetworkLink:
    def test_create_minimal(self):
        link = NetworkLink(source_id="a", target_id="b", transport="wifi")
        assert link.source_id == "a"
        assert link.target_id == "b"
        assert link.transport == "wifi"
        assert link.active is True
        assert link.rssi is None
        assert link.latency_ms is None
        assert link.bandwidth_kbps is None

    def test_create_full(self):
        ts = datetime(2026, 3, 1, tzinfo=timezone.utc)
        link = NetworkLink(
            source_id="node-001",
            target_id="node-002",
            transport="espnow",
            rssi=-55,
            latency_ms=12.5,
            bandwidth_kbps=250.0,
            last_seen=ts,
            active=False,
        )
        assert link.rssi == -55
        assert link.latency_ms == 12.5
        assert link.bandwidth_kbps == 250.0
        assert link.last_seen == ts
        assert link.active is False

    def test_json_roundtrip(self):
        link = NetworkLink(source_id="a", target_id="b", transport="ble", rssi=-70)
        link2 = NetworkLink.model_validate_json(link.model_dump_json())
        assert link2.source_id == "a"
        assert link2.rssi == -70
        assert link2.transport == "ble"


# -- FleetTopology tests ------------------------------------------------------

class TestFleetTopologyNeighbors:
    def test_neighbors_simple(self):
        topo = FleetTopology(
            nodes=["a", "b", "c"],
            links=[_link("a", "b"), _link("b", "c")],
        )
        assert topo.neighbors("a") == ["b"]
        assert topo.neighbors("b") == ["a", "c"]
        assert topo.neighbors("c") == ["b"]

    def test_neighbors_unknown_node(self):
        topo = FleetTopology(nodes=["a"], links=[])
        assert topo.neighbors("z") == []

    def test_neighbors_ignores_inactive(self):
        topo = FleetTopology(
            nodes=["a", "b", "c"],
            links=[_link("a", "b"), _link("a", "c", active=False)],
        )
        assert topo.neighbors("a") == ["b"]


class TestFleetTopologyReachable:
    def test_reachable_direct(self):
        topo = FleetTopology(nodes=["a", "b"], links=[_link("a", "b")])
        assert topo.reachable("a", "b") is True
        assert topo.reachable("b", "a") is True

    def test_reachable_multi_hop(self):
        topo = FleetTopology(
            nodes=["a", "b", "c", "d"],
            links=[_link("a", "b"), _link("b", "c"), _link("c", "d")],
        )
        assert topo.reachable("a", "d") is True

    def test_not_reachable_isolated(self):
        topo = FleetTopology(
            nodes=["a", "b", "c"],
            links=[_link("a", "b")],
        )
        assert topo.reachable("a", "c") is False

    def test_reachable_self(self):
        topo = FleetTopology(nodes=["a"], links=[])
        assert topo.reachable("a", "a") is True

    def test_reachable_unknown_source(self):
        topo = FleetTopology(nodes=["a"], links=[])
        assert topo.reachable("z", "a") is False

    def test_reachable_inactive_link_blocks(self):
        topo = FleetTopology(
            nodes=["a", "b"],
            links=[_link("a", "b", active=False)],
        )
        assert topo.reachable("a", "b") is False


class TestFleetTopologyComponents:
    def test_single_component(self):
        topo = FleetTopology(
            nodes=["a", "b", "c"],
            links=[_link("a", "b"), _link("b", "c")],
        )
        components = topo.connected_components()
        assert len(components) == 1
        assert components[0] == ["a", "b", "c"]

    def test_two_components(self):
        topo = FleetTopology(
            nodes=["a", "b", "c", "d"],
            links=[_link("a", "b"), _link("c", "d")],
        )
        components = topo.connected_components()
        assert len(components) == 2
        assert ["a", "b"] in components
        assert ["c", "d"] in components

    def test_all_isolated(self):
        topo = FleetTopology(nodes=["a", "b", "c"], links=[])
        components = topo.connected_components()
        assert len(components) == 3
        assert all(len(c) == 1 for c in components)

    def test_empty_graph(self):
        topo = FleetTopology()
        assert topo.connected_components() == []


class TestFleetTopologyAveragePathLength:
    def test_linear_three(self):
        # a--b--c: paths are a-b=1, a-c=2, b-a=1, b-c=1, c-a=2, c-b=1
        # total = 8, pairs = 6, avg = 8/6 = 1.333...
        topo = FleetTopology(
            nodes=["a", "b", "c"],
            links=[_link("a", "b"), _link("b", "c")],
        )
        avg = topo.average_path_length()
        assert abs(avg - 8 / 6) < 0.001

    def test_complete_three(self):
        # a--b, a--c, b--c: all paths are length 1, avg = 1.0
        topo = FleetTopology(
            nodes=["a", "b", "c"],
            links=[_link("a", "b"), _link("a", "c"), _link("b", "c")],
        )
        assert topo.average_path_length() == 1.0

    def test_disconnected(self):
        # a--b, c isolated: only 2 reachable pairs (a-b, b-a), avg = 1.0
        topo = FleetTopology(
            nodes=["a", "b", "c"],
            links=[_link("a", "b")],
        )
        assert topo.average_path_length() == 1.0

    def test_single_node(self):
        topo = FleetTopology(nodes=["a"], links=[])
        assert topo.average_path_length() == 0.0

    def test_empty(self):
        topo = FleetTopology()
        assert topo.average_path_length() == 0.0


# -- build_topology tests ----------------------------------------------------

class TestBuildTopology:
    def test_extracts_nodes(self):
        links = [_link("a", "b"), _link("b", "c"), _link("c", "a")]
        topo = build_topology(links)
        assert topo.nodes == ["a", "b", "c"]
        assert len(topo.links) == 3

    def test_empty_links(self):
        topo = build_topology([])
        assert topo.nodes == []
        assert topo.links == []

    def test_duplicate_nodes_deduped(self):
        links = [_link("x", "y"), _link("x", "y", transport="ble")]
        topo = build_topology(links)
        assert topo.nodes == ["x", "y"]
        assert len(topo.links) == 2


# -- analyze_connectivity tests -----------------------------------------------

class TestAnalyzeConnectivity:
    def test_fully_connected(self):
        topo = FleetTopology(
            nodes=["a", "b", "c"],
            links=[_link("a", "b"), _link("b", "c"), _link("a", "c")],
        )
        report = analyze_connectivity(topo)
        assert report.total_nodes == 3
        assert report.connected_nodes == 3
        assert report.isolated_nodes == 0
        assert report.num_components == 1
        assert report.avg_links_per_node == 2.0
        assert report.transports_used == ["wifi"]

    def test_with_isolated_node(self):
        topo = FleetTopology(
            nodes=["a", "b", "c"],
            links=[_link("a", "b")],
        )
        report = analyze_connectivity(topo)
        assert report.total_nodes == 3
        assert report.connected_nodes == 2
        assert report.isolated_nodes == 1
        assert report.num_components == 2

    def test_multiple_transports(self):
        topo = FleetTopology(
            nodes=["a", "b", "c"],
            links=[
                _link("a", "b", transport="wifi"),
                _link("b", "c", transport="lora"),
                _link("a", "c", transport="espnow"),
            ],
        )
        report = analyze_connectivity(topo)
        assert report.transports_used == ["espnow", "lora", "wifi"]

    def test_inactive_links_excluded(self):
        topo = FleetTopology(
            nodes=["a", "b"],
            links=[_link("a", "b", active=False)],
        )
        report = analyze_connectivity(topo)
        assert report.isolated_nodes == 2
        assert report.avg_links_per_node == 0.0
        assert report.transports_used == []

    def test_empty_topology(self):
        report = analyze_connectivity(FleetTopology())
        assert report.total_nodes == 0
        assert report.num_components == 0
        assert report.avg_links_per_node == 0.0

    def test_connectivity_report_json_roundtrip(self):
        report = ConnectivityReport(
            total_nodes=5,
            connected_nodes=4,
            isolated_nodes=1,
            num_components=2,
            avg_links_per_node=1.6,
            transports_used=["wifi", "ble"],
        )
        report2 = ConnectivityReport.model_validate_json(report.model_dump_json())
        assert report2.total_nodes == 5
        assert report2.transports_used == ["wifi", "ble"]
