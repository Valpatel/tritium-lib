"""Tests for tritium_lib.models.mesh."""

from datetime import datetime, timezone

from tritium_lib.models.mesh import (
    MeshNode,
    MeshRoute,
    MeshEdge,
    MeshTopology,
    MeshMessage,
    MeshMessageStatus,
)


class TestMeshNode:
    def test_create(self):
        node = MeshNode(
            node_id="node-001",
            mac="20:6E:F1:9A:12:00",
            neighbors=["node-002", "node-003"],
            hop_count=1,
            rssi_map={"node-002": -55, "node-003": -72},
        )
        assert node.node_id == "node-001"
        assert node.neighbor_count == 2
        assert node.hop_count == 1

    def test_best_neighbor(self):
        node = MeshNode(
            node_id="node-001",
            rssi_map={"node-002": -55, "node-003": -72, "node-004": -40},
        )
        assert node.best_neighbor() == "node-004"  # strongest RSSI

    def test_best_neighbor_empty(self):
        node = MeshNode(node_id="node-001")
        assert node.best_neighbor() is None

    def test_defaults(self):
        node = MeshNode(node_id="n1")
        assert node.neighbors == []
        assert node.rssi_map == {}
        assert node.hop_count == 0
        assert node.neighbor_count == 0

    def test_json_roundtrip(self):
        node = MeshNode(
            node_id="n1",
            mac="AA:BB:CC:DD:EE:FF",
            neighbors=["n2"],
            rssi_map={"n2": -60},
        )
        node2 = MeshNode.model_validate_json(node.model_dump_json())
        assert node2.node_id == "n1"
        assert node2.rssi_map["n2"] == -60


class TestMeshRoute:
    def test_create(self):
        route = MeshRoute(
            source="node-001",
            destination="node-005",
            hops=["node-002", "node-003", "node-004"],
            quality_score=0.85,
        )
        assert route.hop_count == 3
        assert route.total_hops == 5
        assert route.quality_score == 0.85

    def test_direct_route(self):
        route = MeshRoute(
            source="node-001",
            destination="node-002",
            hops=[],
            quality_score=1.0,
        )
        assert route.hop_count == 0
        assert route.total_hops == 2

    def test_json_roundtrip(self):
        route = MeshRoute(source="a", destination="b", hops=["c"], quality_score=0.5)
        route2 = MeshRoute.model_validate_json(route.model_dump_json())
        assert route2.source == "a"
        assert route2.hop_count == 1


class TestMeshTopology:
    def test_create(self):
        nodes = [
            MeshNode(node_id="n1", neighbors=["n2"]),
            MeshNode(node_id="n2", neighbors=["n1", "n3"]),
            MeshNode(node_id="n3", neighbors=["n2"]),
        ]
        edges = [
            MeshEdge(node_a="n1", node_b="n2", rssi=-50),
            MeshEdge(node_a="n2", node_b="n3", rssi=-65),
        ]
        topo = MeshTopology(nodes=nodes, edges=edges, partitions=1)
        assert topo.node_count == 3
        assert topo.edge_count == 2
        assert topo.partitions == 1

    def test_node_by_id(self):
        nodes = [
            MeshNode(node_id="n1"),
            MeshNode(node_id="n2"),
        ]
        topo = MeshTopology(nodes=nodes)
        assert topo.node_by_id("n1").node_id == "n1"
        assert topo.node_by_id("n2").node_id == "n2"
        assert topo.node_by_id("n99") is None

    def test_empty_topology(self):
        topo = MeshTopology()
        assert topo.node_count == 0
        assert topo.edge_count == 0

    def test_json_roundtrip(self):
        topo = MeshTopology(
            nodes=[MeshNode(node_id="n1")],
            edges=[MeshEdge(node_a="n1", node_b="n2")],
            partitions=2,
        )
        topo2 = MeshTopology.model_validate_json(topo.model_dump_json())
        assert topo2.node_count == 1
        assert topo2.edge_count == 1
        assert topo2.partitions == 2


class TestMeshMessage:
    def test_create(self):
        msg = MeshMessage(
            message_id="msg-001",
            source="node-001",
            destination="node-005",
            payload=b"hello mesh",
            ttl=10,
            hop_count=3,
        )
        assert msg.payload_size == 10
        assert msg.remaining_ttl == 7
        assert msg.status == MeshMessageStatus.PENDING

    def test_broadcast(self):
        msg = MeshMessage(
            source="node-001",
            destination="broadcast",
            payload=b"\x01",
            ttl=5,
        )
        assert msg.destination == "broadcast"
        assert msg.remaining_ttl == 5

    def test_ttl_exhausted(self):
        msg = MeshMessage(
            source="n1",
            destination="n2",
            ttl=3,
            hop_count=5,
        )
        assert msg.remaining_ttl == 0

    def test_status_transitions(self):
        msg = MeshMessage(
            source="n1",
            destination="n2",
            status=MeshMessageStatus.DELIVERED,
        )
        assert msg.status == MeshMessageStatus.DELIVERED

    def test_json_roundtrip(self):
        msg = MeshMessage(
            message_id="m1",
            source="n1",
            destination="n2",
            ttl=8,
            hop_count=2,
        )
        msg2 = MeshMessage.model_validate_json(msg.model_dump_json())
        assert msg2.message_id == "m1"
        assert msg2.remaining_ttl == 6
