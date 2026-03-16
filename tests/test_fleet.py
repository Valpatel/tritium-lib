# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for fleet management models."""

from datetime import datetime

from tritium_lib.models.fleet import (
    FleetNode,
    FleetStatus,
    NodeEvent,
    NodeStatus,
    fleet_health_score,
)


class TestFleetNode:
    def test_create(self):
        node = FleetNode(
            device_id="esp32-001",
            mac="AA:BB:CC:DD:EE:FF",
            ip="10.0.0.10",
            firmware_version="1.2.0",
            uptime_s=3600,
            wifi_rssi=-45,
            free_heap=200_000,
            psram_free=4_000_000,
            partition="ota_0",
            status=NodeStatus.ONLINE,
            capabilities=["camera", "audio", "ble"],
            ble_device_count=5,
        )
        assert node.device_id == "esp32-001"
        assert node.status == NodeStatus.ONLINE
        assert "camera" in node.capabilities

    def test_status_values(self):
        assert NodeStatus.ONLINE.value == "online"
        assert NodeStatus.STALE.value == "stale"
        assert NodeStatus.OFFLINE.value == "offline"

    def test_status_transitions(self):
        node = FleetNode(device_id="esp32-001", status=NodeStatus.ONLINE)
        assert node.status == NodeStatus.ONLINE
        node.status = NodeStatus.STALE
        assert node.status == NodeStatus.STALE
        node.status = NodeStatus.OFFLINE
        assert node.status == NodeStatus.OFFLINE

    def test_json_roundtrip(self):
        node = FleetNode(
            device_id="esp32-001",
            status=NodeStatus.ONLINE,
            wifi_rssi=-50,
            free_heap=150_000,
        )
        json_str = node.model_dump_json()
        node2 = FleetNode.model_validate_json(json_str)
        assert node2.device_id == node.device_id
        assert node2.status == NodeStatus.ONLINE
        assert node2.wifi_rssi == -50


class TestNodeEvent:
    def test_create(self):
        evt = NodeEvent(
            node_id="esp32-001",
            event_type="online",
            message="Node came online after OTA",
        )
        assert evt.event_type == "online"
        assert evt.node_id == "esp32-001"

    def test_json_roundtrip(self):
        evt = NodeEvent(node_id="n1", event_type="error", message="watchdog reset")
        evt2 = NodeEvent.model_validate_json(evt.model_dump_json())
        assert evt2.event_type == "error"


class TestFleetHealthScore:
    def _make_node(
        self,
        device_id: str,
        status: NodeStatus = NodeStatus.ONLINE,
        rssi: int = -50,
        heap: int = 200_000,
    ) -> FleetNode:
        return FleetNode(
            device_id=device_id,
            status=status,
            wifi_rssi=rssi,
            free_heap=heap,
        )

    def test_empty_fleet(self):
        fleet = FleetStatus(total_nodes=0)
        assert fleet_health_score(fleet) == 0.0

    def test_all_online_good_signal(self):
        nodes = [
            self._make_node("n1", NodeStatus.ONLINE, rssi=-40, heap=250_000),
            self._make_node("n2", NodeStatus.ONLINE, rssi=-40, heap=250_000),
        ]
        fleet = FleetStatus(
            nodes=nodes, total_nodes=2, online_count=2,
        )
        score = fleet_health_score(fleet)
        assert score > 0.8

    def test_all_offline(self):
        nodes = [
            self._make_node("n1", NodeStatus.OFFLINE),
            self._make_node("n2", NodeStatus.OFFLINE),
        ]
        fleet = FleetStatus(
            nodes=nodes, total_nodes=2, online_count=0,
        )
        score = fleet_health_score(fleet)
        assert score == 0.0

    def test_partial_fleet(self):
        nodes = [
            self._make_node("n1", NodeStatus.ONLINE, rssi=-50, heap=150_000),
            self._make_node("n2", NodeStatus.OFFLINE),
        ]
        fleet = FleetStatus(
            nodes=nodes, total_nodes=2, online_count=1,
        )
        score = fleet_health_score(fleet)
        assert 0.2 < score < 0.8

    def test_score_bounded(self):
        """Score should always be in [0, 1]."""
        nodes = [self._make_node("n1", NodeStatus.ONLINE, rssi=-30, heap=300_000)]
        fleet = FleetStatus(nodes=nodes, total_nodes=1, online_count=1)
        score = fleet_health_score(fleet)
        assert 0.0 <= score <= 1.0
