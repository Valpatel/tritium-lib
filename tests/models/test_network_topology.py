# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for NetworkNode, PeerQuality, and enhanced topology models."""

import pytest

from tritium_lib.models.topology import (
    ConnectivityReport,
    FleetTopology,
    NetworkLink,
    NetworkNode,
    NodeRole,
    PeerQuality,
    analyze_connectivity,
    build_fleet_topology_from_mesh,
    build_topology,
)


class TestNetworkNode:
    """Tests for the NetworkNode model."""

    def test_create_default(self):
        node = NetworkNode(node_id="esp32-001")
        assert node.node_id == "esp32-001"
        assert node.role == NodeRole.RELAY
        assert node.online is True
        assert node.peer_count == 0
        assert node.transports == []

    def test_create_gateway(self):
        node = NetworkNode(
            node_id="gateway-01",
            name="Main Gateway",
            role=NodeRole.GATEWAY,
            ip="192.168.1.100",
            mac="AA:BB:CC:DD:EE:FF",
            battery_pct=85,
            lat=30.267,
            lng=-97.743,
            peer_count=5,
            avg_peer_rssi=-45.2,
            transports=["wifi", "espnow"],
        )
        assert node.role == NodeRole.GATEWAY
        assert node.peer_count == 5
        assert node.avg_peer_rssi == -45.2
        assert "wifi" in node.transports

    def test_node_roles(self):
        assert NodeRole.GATEWAY.value == "gateway"
        assert NodeRole.RELAY.value == "relay"
        assert NodeRole.LEAF.value == "leaf"
        assert NodeRole.SENSOR.value == "sensor"


class TestPeerQuality:
    """Tests for the PeerQuality model."""

    def test_create(self):
        pq = PeerQuality(
            peer_mac="AA:BB:CC:DD:EE:FF",
            rssi_current=-55,
            rssi_avg=-50.0,
            rssi_min=-65,
            rssi_max=-35,
            packet_loss_pct=2.5,
            avg_latency_ms=15.3,
            tx_count=100,
            rx_count=97,
            tx_fail=3,
        )
        assert pq.peer_mac == "AA:BB:CC:DD:EE:FF"
        assert pq.packet_loss_pct == 2.5

    def test_quality_score_good(self):
        pq = PeerQuality(peer_mac="test", rssi_avg=-40.0, packet_loss_pct=0.0)
        score = pq.quality_score
        assert 70 <= score <= 100

    def test_quality_score_poor(self):
        pq = PeerQuality(peer_mac="test", rssi_avg=-85.0, packet_loss_pct=20.0)
        score = pq.quality_score
        assert score == 0  # very weak signal + high loss

    def test_quality_score_medium(self):
        pq = PeerQuality(peer_mac="test", rssi_avg=-60.0, packet_loss_pct=5.0)
        score = pq.quality_score
        assert 30 <= score <= 60


class TestNetworkLinkEnhancements:
    """Tests for the enhanced NetworkLink fields."""

    def test_link_with_quality(self):
        link = NetworkLink(
            source_id="node-a",
            target_id="node-b",
            transport="espnow",
            rssi=-50,
            packet_loss_pct=3.2,
            quality_score=78,
        )
        assert link.packet_loss_pct == 3.2
        assert link.quality_score == 78

    def test_link_defaults(self):
        link = NetworkLink(
            source_id="a",
            target_id="b",
            transport="wifi",
        )
        assert link.packet_loss_pct == 0.0
        assert link.quality_score == 0


class TestFleetTopologyWithNodes:
    """Tests for FleetTopology with NetworkNode metadata."""

    def test_topology_with_network_nodes(self):
        nodes = [
            NetworkNode(node_id="gw-1", role=NodeRole.GATEWAY, lat=30.0, lng=-97.0),
            NetworkNode(node_id="relay-1", role=NodeRole.RELAY, lat=30.001, lng=-97.001),
            NetworkNode(node_id="leaf-1", role=NodeRole.LEAF, lat=30.002, lng=-97.002),
        ]
        links = [
            NetworkLink(source_id="gw-1", target_id="relay-1", transport="espnow", rssi=-45),
            NetworkLink(source_id="relay-1", target_id="leaf-1", transport="espnow", rssi=-60),
        ]
        topo = FleetTopology(
            nodes=["gw-1", "relay-1", "leaf-1"],
            links=links,
            network_nodes=nodes,
        )
        assert len(topo.network_nodes) == 3
        assert topo.network_nodes[0].role == NodeRole.GATEWAY
        assert topo.reachable("gw-1", "leaf-1")

    def test_build_topology_still_works(self):
        links = [
            NetworkLink(source_id="a", target_id="b", transport="wifi"),
        ]
        topo = build_topology(links)
        assert set(topo.nodes) == {"a", "b"}
        assert topo.network_nodes == []  # not populated by build_topology

    def test_analyze_connectivity_with_quality(self):
        links = [
            NetworkLink(source_id="a", target_id="b", transport="espnow",
                        quality_score=80, packet_loss_pct=1.0),
            NetworkLink(source_id="b", target_id="c", transport="wifi",
                        quality_score=60, packet_loss_pct=5.0),
        ]
        topo = build_topology(links)
        report = analyze_connectivity(topo)
        assert report.total_nodes == 3
        assert report.connected_nodes == 3
        assert report.isolated_nodes == 0
        assert "espnow" in report.transports_used
        assert "wifi" in report.transports_used
