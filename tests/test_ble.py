# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for BLE presence models."""

from datetime import datetime

from tritium_lib.models.ble import (
    BleDevice,
    BlePresence,
    BlePresenceMap,
    BleSighting,
    set_node_positions,
    triangulate_position,
)


class TestBleDevice:
    def test_create(self):
        d = BleDevice(mac="AA:BB:CC:DD:EE:FF", rssi=-55, name="Test Tag")
        assert d.mac == "AA:BB:CC:DD:EE:FF"
        assert d.rssi == -55
        assert d.name == "Test Tag"
        assert d.seen_count == 1
        assert d.is_known is False

    def test_json_roundtrip(self):
        d = BleDevice(mac="AA:BB:CC:DD:EE:FF", rssi=-55, name="Tag")
        json_str = d.model_dump_json()
        d2 = BleDevice.model_validate_json(json_str)
        assert d2.mac == d.mac
        assert d2.rssi == d.rssi


class TestBleSighting:
    def test_create(self):
        device = BleDevice(mac="AA:BB:CC:DD:EE:FF", rssi=-60)
        s = BleSighting(device=device, node_id="node-1", node_ip="10.0.0.1")
        assert s.node_id == "node-1"
        assert s.device.rssi == -60

    def test_json_roundtrip(self):
        device = BleDevice(mac="AA:BB:CC:DD:EE:FF", rssi=-60)
        s = BleSighting(device=device, node_id="node-1")
        s2 = BleSighting.model_validate_json(s.model_dump_json())
        assert s2.device.mac == "AA:BB:CC:DD:EE:FF"


class TestBlePresenceMap:
    def test_aggregation(self):
        mac = "AA:BB:CC:DD:EE:FF"
        device = BleDevice(mac=mac, rssi=-50)
        s1 = BleSighting(device=device, node_id="node-1")
        s2 = BleSighting(
            device=BleDevice(mac=mac, rssi=-70), node_id="node-2"
        )
        presence = BlePresence(
            mac=mac,
            name="Tag",
            sightings=[s1, s2],
            strongest_rssi=-50,
            node_count=2,
        )
        pm = BlePresenceMap(
            devices={mac: presence},
            total_devices=1,
            total_nodes=2,
        )
        assert pm.total_devices == 1
        assert pm.total_nodes == 2
        assert mac in pm.devices
        assert pm.devices[mac].strongest_rssi == -50

    def test_json_roundtrip(self):
        pm = BlePresenceMap(total_devices=0, total_nodes=0)
        pm2 = BlePresenceMap.model_validate_json(pm.model_dump_json())
        assert pm2.total_devices == 0


class TestTriangulation:
    def _make_sighting(self, node_id: str, rssi: int) -> BleSighting:
        return BleSighting(
            device=BleDevice(mac="AA:BB:CC:DD:EE:FF", rssi=rssi),
            node_id=node_id,
        )

    def test_insufficient_sightings(self):
        """Need 3+ sightings from positioned nodes."""
        set_node_positions({"n1": (0, 0), "n2": (10, 0)})
        sightings = [self._make_sighting("n1", -50), self._make_sighting("n2", -60)]
        assert triangulate_position(sightings) is None

    def test_no_positions(self):
        set_node_positions({})
        sightings = [
            self._make_sighting("n1", -50),
            self._make_sighting("n2", -60),
            self._make_sighting("n3", -70),
        ]
        assert triangulate_position(sightings) is None

    def test_three_nodes(self):
        """With 3 positioned nodes, should return a position."""
        set_node_positions({
            "n1": (0.0, 0.0),
            "n2": (10.0, 0.0),
            "n3": (5.0, 10.0),
        })
        sightings = [
            self._make_sighting("n1", -40),  # strongest — closest to n1
            self._make_sighting("n2", -70),
            self._make_sighting("n3", -70),
        ]
        result = triangulate_position(sightings)
        assert result is not None
        x, y = result
        # Should be closer to n1 (0,0) since it has strongest RSSI
        assert x < 5.0
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_equal_rssi_gives_centroid(self):
        """Equal RSSI from all nodes should approximate the centroid."""
        set_node_positions({
            "n1": (0.0, 0.0),
            "n2": (10.0, 0.0),
            "n3": (5.0, 10.0),
        })
        sightings = [
            self._make_sighting("n1", -60),
            self._make_sighting("n2", -60),
            self._make_sighting("n3", -60),
        ]
        result = triangulate_position(sightings)
        assert result is not None
        x, y = result
        # Centroid of (0,0), (10,0), (5,10) = (5, 3.33)
        assert abs(x - 5.0) < 0.1
        assert abs(y - 3.33) < 0.1
