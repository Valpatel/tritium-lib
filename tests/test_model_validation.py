# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for Pydantic model validation, rejection of invalid data,
and round-trip JSON serialization across key models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tritium_lib.models.ble import BleDevice, BlePresence, BleSighting
from tritium_lib.models.camera import CameraDetection, CameraPosition
from tritium_lib.models.device import Device, DeviceHeartbeat
from tritium_lib.models.sensor import SensorReading
from tritium_lib.models.wifi import WiFiFingerprint, WiFiNetwork, WiFiProbeRequest


# ── MAC Address Validation ──────────────────────────────────────────


class TestMACAddressValidation:
    """MAC address fields must match AA:BB:CC:DD:EE:FF format."""

    def test_ble_device_valid_mac(self):
        d = BleDevice(mac="AA:BB:CC:DD:EE:FF", rssi=-50)
        assert d.mac == "AA:BB:CC:DD:EE:FF"

    def test_ble_device_lowercase_normalized(self):
        d = BleDevice(mac="aa:bb:cc:dd:ee:ff", rssi=-50)
        assert d.mac == "AA:BB:CC:DD:EE:FF"

    def test_ble_device_invalid_mac_rejected(self):
        with pytest.raises(ValidationError, match="Invalid MAC"):
            BleDevice(mac="not-a-mac", rssi=-50)

    def test_ble_device_short_mac_rejected(self):
        with pytest.raises(ValidationError, match="Invalid MAC"):
            BleDevice(mac="AA:BB:CC", rssi=-50)

    def test_ble_device_mac_no_colons_rejected(self):
        with pytest.raises(ValidationError, match="Invalid MAC"):
            BleDevice(mac="AABBCCDDEEFF", rssi=-50)

    def test_ble_device_mac_with_dashes_rejected(self):
        with pytest.raises(ValidationError, match="Invalid MAC"):
            BleDevice(mac="AA-BB-CC-DD-EE-FF", rssi=-50)

    def test_device_valid_mac(self):
        d = Device(device_id="esp32-001", mac="20:6E:F1:9A:12:00")
        assert d.mac == "20:6E:F1:9A:12:00"

    def test_device_empty_mac_allowed(self):
        d = Device(device_id="esp32-001", mac="")
        assert d.mac == ""

    def test_device_invalid_mac_rejected(self):
        with pytest.raises(ValidationError, match="Invalid MAC"):
            Device(device_id="esp32-001", mac="garbage")

    def test_device_mac_normalized_uppercase(self):
        d = Device(device_id="esp32-001", mac="aa:bb:cc:dd:ee:ff")
        assert d.mac == "AA:BB:CC:DD:EE:FF"

    def test_wifi_probe_invalid_mac_rejected(self):
        with pytest.raises(ValidationError, match="Invalid MAC"):
            WiFiProbeRequest(mac="bad-mac")

    def test_wifi_probe_mac_normalized(self):
        p = WiFiProbeRequest(mac="de:ad:be:ef:00:01")
        assert p.mac == "DE:AD:BE:EF:00:01"

    def test_wifi_network_invalid_bssid_rejected(self):
        with pytest.raises(ValidationError, match="Invalid BSSID"):
            WiFiNetwork(bssid="not-a-bssid")

    def test_wifi_network_bssid_normalized(self):
        n = WiFiNetwork(bssid="aa:bb:cc:dd:ee:ff")
        assert n.bssid == "AA:BB:CC:DD:EE:FF"

    def test_wifi_fingerprint_invalid_mac_rejected(self):
        with pytest.raises(ValidationError, match="Invalid MAC"):
            WiFiFingerprint(mac="xyz")

    def test_ble_presence_invalid_mac_rejected(self):
        with pytest.raises(ValidationError, match="Invalid MAC"):
            BlePresence(mac="not-valid")


# ── Coordinate Validation ───────────────────────────────────────────


class TestCoordinateValidation:
    """Lat/lng must be in valid ranges."""

    def test_camera_position_valid(self):
        pos = CameraPosition(lat=37.7749, lng=-122.4194)
        assert pos.lat == 37.7749
        assert pos.lng == -122.4194

    def test_camera_position_lat_too_high(self):
        with pytest.raises(ValidationError):
            CameraPosition(lat=91.0, lng=0.0)

    def test_camera_position_lat_too_low(self):
        with pytest.raises(ValidationError):
            CameraPosition(lat=-91.0, lng=0.0)

    def test_camera_position_lng_too_high(self):
        with pytest.raises(ValidationError):
            CameraPosition(lat=0.0, lng=181.0)

    def test_camera_position_lng_too_low(self):
        with pytest.raises(ValidationError):
            CameraPosition(lat=0.0, lng=-181.0)

    def test_camera_position_boundary_values(self):
        pos = CameraPosition(lat=90.0, lng=180.0)
        assert pos.lat == 90.0
        assert pos.lng == 180.0

        pos2 = CameraPosition(lat=-90.0, lng=-180.0)
        assert pos2.lat == -90.0
        assert pos2.lng == -180.0

    def test_camera_position_null_allowed(self):
        pos = CameraPosition()
        assert pos.lat is None
        assert pos.lng is None


# ── RSSI Validation ─────────────────────────────────────────────────


class TestRSSIValidation:
    """RSSI values must be negative or zero and within -127..0."""

    def test_ble_device_valid_rssi(self):
        d = BleDevice(mac="AA:BB:CC:DD:EE:FF", rssi=-65)
        assert d.rssi == -65

    def test_ble_device_rssi_too_high(self):
        with pytest.raises(ValidationError):
            BleDevice(mac="AA:BB:CC:DD:EE:FF", rssi=1)

    def test_ble_device_rssi_too_low(self):
        with pytest.raises(ValidationError):
            BleDevice(mac="AA:BB:CC:DD:EE:FF", rssi=-128)

    def test_ble_device_rssi_boundary(self):
        d_zero = BleDevice(mac="AA:BB:CC:DD:EE:FF", rssi=0)
        assert d_zero.rssi == 0
        d_min = BleDevice(mac="AA:BB:CC:DD:EE:FF", rssi=-127)
        assert d_min.rssi == -127

    def test_wifi_probe_rssi_valid(self):
        p = WiFiProbeRequest(mac="AA:BB:CC:DD:EE:FF", rssi=-45)
        assert p.rssi == -45

    def test_wifi_probe_rssi_too_high(self):
        with pytest.raises(ValidationError):
            WiFiProbeRequest(mac="AA:BB:CC:DD:EE:FF", rssi=5)

    def test_wifi_network_rssi_valid(self):
        n = WiFiNetwork(bssid="AA:BB:CC:DD:EE:FF", rssi=-30)
        assert n.rssi == -30

    def test_wifi_network_rssi_too_high(self):
        with pytest.raises(ValidationError):
            WiFiNetwork(bssid="AA:BB:CC:DD:EE:FF", rssi=10)

    def test_ble_presence_rssi_valid(self):
        p = BlePresence(mac="AA:BB:CC:DD:EE:FF", strongest_rssi=-40)
        assert p.strongest_rssi == -40

    def test_ble_presence_rssi_too_high(self):
        with pytest.raises(ValidationError):
            BlePresence(mac="AA:BB:CC:DD:EE:FF", strongest_rssi=1)


# ── Alert / Severity Validation ─────────────────────────────────────


class TestSeverityValidation:
    """Confidence and quality scores must be in 0.0..1.0 range."""

    def test_sensor_quality_valid(self):
        r = SensorReading(
            device_id="dev-1", sensor_type="temp", value=22.0, quality=0.95
        )
        assert r.quality == 0.95

    def test_sensor_quality_too_high(self):
        with pytest.raises(ValidationError):
            SensorReading(
                device_id="dev-1", sensor_type="temp", value=22.0, quality=1.5
            )

    def test_sensor_quality_too_low(self):
        with pytest.raises(ValidationError):
            SensorReading(
                device_id="dev-1", sensor_type="temp", value=22.0, quality=-0.1
            )

    def test_sensor_quality_boundaries(self):
        r_zero = SensorReading(
            device_id="dev-1", sensor_type="temp", value=22.0, quality=0.0
        )
        assert r_zero.quality == 0.0
        r_one = SensorReading(
            device_id="dev-1", sensor_type="temp", value=22.0, quality=1.0
        )
        assert r_one.quality == 1.0

    def test_camera_detection_confidence_valid(self):
        d = CameraDetection(source_id="cam-01", confidence=0.85)
        assert d.confidence == 0.85

    def test_camera_detection_confidence_too_high(self):
        with pytest.raises(ValidationError):
            CameraDetection(source_id="cam-01", confidence=1.1)

    def test_camera_detection_confidence_too_low(self):
        with pytest.raises(ValidationError):
            CameraDetection(source_id="cam-01", confidence=-0.5)


# ── Device Status Validation ────────────────────────────────────────


class TestDeviceStatusValidation:
    """Device status must be one of the allowed values."""

    def test_valid_statuses(self):
        for status in ("online", "offline", "updating", "error"):
            d = Device(device_id="esp32-001", status=status)
            assert d.status == status

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError, match="Invalid status"):
            Device(device_id="esp32-001", status="rebooting")

    def test_invalid_status_empty(self):
        with pytest.raises(ValidationError, match="Invalid status"):
            Device(device_id="esp32-001", status="")


# ── IP Address Validation ───────────────────────────────────────────


class TestIPAddressValidation:
    """IP addresses must be valid IPv4 when provided."""

    def test_heartbeat_valid_ip(self):
        hb = DeviceHeartbeat(device_id="dev-1", ip_address="192.168.1.100")
        assert hb.ip_address == "192.168.1.100"

    def test_heartbeat_invalid_ip_rejected(self):
        with pytest.raises(ValidationError, match="Invalid IPv4"):
            DeviceHeartbeat(device_id="dev-1", ip_address="not.an.ip")

    def test_heartbeat_ip_too_few_octets(self):
        with pytest.raises(ValidationError, match="Invalid IPv4"):
            DeviceHeartbeat(device_id="dev-1", ip_address="192.168.1")

    def test_heartbeat_ip_octet_out_of_range(self):
        with pytest.raises(ValidationError, match="Invalid IPv4"):
            DeviceHeartbeat(device_id="dev-1", ip_address="192.168.1.256")

    def test_heartbeat_null_ip_allowed(self):
        hb = DeviceHeartbeat(device_id="dev-1", ip_address=None)
        assert hb.ip_address is None

    def test_device_valid_ip(self):
        d = Device(device_id="dev-1", ip_address="10.0.0.1")
        assert d.ip_address == "10.0.0.1"

    def test_device_invalid_ip_rejected(self):
        with pytest.raises(ValidationError, match="Invalid IPv4"):
            Device(device_id="dev-1", ip_address="abc.def.ghi.jkl")


# ── Heartbeat Field Constraints ─────────────────────────────────────


class TestHeartbeatConstraints:
    """Heartbeat numeric fields must be non-negative where applicable."""

    def test_uptime_nonnegative(self):
        with pytest.raises(ValidationError):
            DeviceHeartbeat(device_id="dev-1", uptime_s=-1)

    def test_free_heap_nonnegative(self):
        with pytest.raises(ValidationError):
            DeviceHeartbeat(device_id="dev-1", free_heap=-100)

    def test_boot_count_nonnegative(self):
        with pytest.raises(ValidationError):
            DeviceHeartbeat(device_id="dev-1", boot_count=-1)

    def test_mesh_peers_nonnegative(self):
        with pytest.raises(ValidationError):
            DeviceHeartbeat(device_id="dev-1", mesh_peers=-1)

    def test_wifi_rssi_range(self):
        with pytest.raises(ValidationError):
            DeviceHeartbeat(device_id="dev-1", wifi_rssi=5)

    def test_device_id_required_nonempty(self):
        with pytest.raises(ValidationError):
            DeviceHeartbeat(device_id="")


# ── WiFi Channel Validation ─────────────────────────────────────────


class TestWiFiChannelValidation:
    """WiFi channels must be non-negative."""

    def test_valid_channel(self):
        p = WiFiProbeRequest(mac="AA:BB:CC:DD:EE:FF", channel=36)
        assert p.channel == 36

    def test_negative_channel_rejected(self):
        with pytest.raises(ValidationError):
            WiFiProbeRequest(mac="AA:BB:CC:DD:EE:FF", channel=-1)

    def test_channel_too_high_rejected(self):
        with pytest.raises(ValidationError):
            WiFiProbeRequest(mac="AA:BB:CC:DD:EE:FF", channel=200)


# ── Round-Trip JSON Serialization ───────────────────────────────────


class TestJSONRoundTrip:
    """Verify model -> JSON -> model roundtrip preserves all fields."""

    def test_ble_device_roundtrip(self):
        original = BleDevice(
            mac="AA:BB:CC:DD:EE:FF",
            rssi=-65,
            name="iPhone",
            seen_count=5,
            is_known=True,
        )
        json_str = original.model_dump_json()
        restored = BleDevice.model_validate_json(json_str)
        assert restored.mac == original.mac
        assert restored.rssi == original.rssi
        assert restored.name == original.name
        assert restored.seen_count == original.seen_count
        assert restored.is_known == original.is_known

    def test_ble_sighting_roundtrip(self):
        device = BleDevice(mac="AA:BB:CC:DD:EE:FF", rssi=-60)
        original = BleSighting(device=device, node_id="node-1", node_ip="10.0.0.1")
        json_str = original.model_dump_json()
        restored = BleSighting.model_validate_json(json_str)
        assert restored.device.mac == original.device.mac
        assert restored.node_id == original.node_id
        assert restored.node_ip == original.node_ip

    def test_device_roundtrip(self):
        ts = datetime(2026, 3, 7, 12, 0, 0, tzinfo=timezone.utc)
        original = Device(
            device_id="esp32-001",
            device_name="Lab Sensor",
            mac="20:6E:F1:9A:12:00",
            board="touch-lcd-43c-box",
            status="online",
            capabilities=["camera", "ble"],
            tags=["lab", "floor-1"],
            last_seen=ts,
        )
        json_str = original.model_dump_json()
        restored = Device.model_validate_json(json_str)
        assert restored.device_id == original.device_id
        assert restored.mac == original.mac
        assert restored.status == original.status
        assert restored.capabilities == original.capabilities
        assert restored.tags == original.tags
        assert restored.last_seen == original.last_seen

    def test_heartbeat_roundtrip(self):
        original = DeviceHeartbeat(
            device_id="esp32-001",
            firmware_version="1.2.3",
            board="touch-lcd-43c-box",
            uptime_s=3600,
            free_heap=180000,
            wifi_rssi=-55,
            ip_address="192.168.86.42",
            capabilities=["camera", "imu"],
        )
        json_str = original.model_dump_json()
        restored = DeviceHeartbeat.model_validate_json(json_str)
        assert restored.device_id == original.device_id
        assert restored.uptime_s == original.uptime_s
        assert restored.wifi_rssi == original.wifi_rssi
        assert restored.ip_address == original.ip_address

    def test_sensor_reading_roundtrip(self):
        ts = datetime(2026, 3, 7, 12, 0, 0, tzinfo=timezone.utc)
        original = SensorReading(
            device_id="esp32-001",
            sensor_type="temperature",
            value=23.5,
            unit="celsius",
            quality=0.95,
            timestamp=ts,
        )
        json_str = original.model_dump_json()
        restored = SensorReading.model_validate_json(json_str)
        assert restored.device_id == original.device_id
        assert restored.sensor_type == original.sensor_type
        assert restored.value == original.value
        assert restored.unit == original.unit
        assert restored.quality == original.quality
        assert restored.timestamp == original.timestamp

    def test_wifi_probe_roundtrip(self):
        original = WiFiProbeRequest(
            mac="DE:AD:BE:EF:00:01",
            ssid_probed="HomeNet",
            rssi=-60,
            channel=11,
            observer_id="edge-002",
        )
        json_str = original.model_dump_json()
        restored = WiFiProbeRequest.model_validate_json(json_str)
        assert restored.mac == original.mac
        assert restored.ssid_probed == original.ssid_probed
        assert restored.rssi == original.rssi
        assert restored.channel == original.channel

    def test_wifi_network_roundtrip(self):
        original = WiFiNetwork(
            bssid="00:11:22:33:44:55",
            ssid="CorpNet-5G",
            rssi=-30,
            channel=36,
            auth_type="wpa2-enterprise",
        )
        json_str = original.model_dump_json()
        restored = WiFiNetwork.model_validate_json(json_str)
        assert restored.bssid == original.bssid
        assert restored.ssid == original.ssid
        assert restored.rssi == original.rssi

    def test_camera_detection_roundtrip(self):
        original = CameraDetection(
            source_id="cam-front-01",
            class_name="person",
            confidence=0.92,
        )
        json_str = original.model_dump_json()
        restored = CameraDetection.model_validate_json(json_str)
        assert restored.source_id == original.source_id
        assert restored.class_name == original.class_name
        assert restored.confidence == original.confidence


# ── to_summary() Methods ────────────────────────────────────────────


class TestToSummary:
    """Verify to_summary() returns human-readable strings."""

    def test_ble_device_summary(self):
        d = BleDevice(mac="AA:BB:CC:DD:EE:FF", rssi=-65, name="iPhone")
        s = d.to_summary()
        assert "AA:BB:CC:DD:EE:FF" in s
        assert "iPhone" in s
        assert "-65" in s

    def test_ble_device_summary_no_name(self):
        d = BleDevice(mac="AA:BB:CC:DD:EE:FF", rssi=-80)
        s = d.to_summary()
        assert "AA:BB:CC:DD:EE:FF" in s
        assert "()" not in s  # no empty parens

    def test_ble_sighting_summary(self):
        device = BleDevice(mac="AA:BB:CC:DD:EE:FF", rssi=-60)
        sighting = BleSighting(device=device, node_id="node-1")
        s = sighting.to_summary()
        assert "AA:BB:CC:DD:EE:FF" in s
        assert "node-1" in s

    def test_ble_presence_summary(self):
        p = BlePresence(
            mac="AA:BB:CC:DD:EE:FF",
            name="Sensor",
            strongest_rssi=-40,
            node_count=3,
        )
        s = p.to_summary()
        assert "AA:BB:CC:DD:EE:FF" in s
        assert "Sensor" in s
        assert "-40" in s

    def test_device_summary(self):
        d = Device(
            device_id="esp32-001",
            device_name="Front Door",
            board="touch-lcd-43c-box",
            status="online",
            mac="AA:BB:CC:DD:EE:FF",
        )
        s = d.to_summary()
        assert "Front Door" in s
        assert "online" in s
        assert "AA:BB:CC:DD:EE:FF" in s

    def test_device_summary_no_name(self):
        d = Device(device_id="esp32-001", board="touch-lcd-43c-box")
        s = d.to_summary()
        assert "esp32-001" in s

    def test_heartbeat_summary(self):
        hb = DeviceHeartbeat(
            device_id="esp32-001",
            board="touch-lcd-43c-box",
            firmware_version="1.2.3",
            uptime_s=3600,
            free_heap=180000,
            wifi_rssi=-55,
            capabilities=["camera", "imu"],
        )
        s = hb.to_summary()
        assert "esp32-001" in s
        assert "1.2.3" in s
        assert "-55" in s
        assert "camera" in s

    def test_sensor_reading_summary(self):
        r = SensorReading(
            device_id="esp32-001",
            sensor_type="temperature",
            value=23.5,
            unit="celsius",
            quality=0.95,
        )
        s = r.to_summary()
        assert "esp32-001" in s
        assert "temperature" in s
        assert "23.5" in s
        assert "celsius" in s

    def test_sensor_reading_summary_truncation(self):
        r = SensorReading(
            device_id="dev-1",
            sensor_type="imu",
            value={"ax": 0.1, "ay": 0.2, "az": 9.8, "gx": 1.0, "gy": 2.0, "gz": 3.0},
        )
        s = r.to_summary()
        assert "..." in s  # long dict values get truncated

    def test_wifi_probe_summary(self):
        p = WiFiProbeRequest(
            mac="AA:BB:CC:DD:EE:FF", ssid_probed="MyNetwork", rssi=-65, channel=6
        )
        s = p.to_summary()
        assert "AA:BB:CC:DD:EE:FF" in s
        assert "MyNetwork" in s

    def test_wifi_probe_summary_broadcast(self):
        p = WiFiProbeRequest(mac="AA:BB:CC:DD:EE:FF")
        s = p.to_summary()
        assert "(broadcast)" in s

    def test_wifi_network_summary(self):
        n = WiFiNetwork(
            bssid="00:11:22:33:44:55",
            ssid="CorpNet",
            rssi=-45,
            channel=36,
            auth_type="wpa2",
        )
        s = n.to_summary()
        assert "00:11:22:33:44:55" in s
        assert "CorpNet" in s

    def test_wifi_fingerprint_summary(self):
        fp = WiFiFingerprint(
            mac="AA:BB:CC:DD:EE:FF",
            probed_ssids=["Net1", "Net2", "Net3", "Net4", "Net5"],
            device_type_hint="laptop",
            probe_count=42,
        )
        s = fp.to_summary()
        assert "AA:BB:CC:DD:EE:FF" in s
        assert "laptop" in s
        assert "+2 more" in s  # shows 3 then truncates

    def test_camera_detection_summary(self):
        d = CameraDetection(
            source_id="cam-01", class_name="person", confidence=0.92
        )
        s = d.to_summary()
        assert "cam-01" in s
        assert "person" in s
        assert "0.92" in s


# ── ConfigDict / JSON Schema Extras ─────────────────────────────────


class TestJSONSchemaExtras:
    """Verify that models expose json_schema_extra with examples."""

    def test_device_schema_has_examples(self):
        schema = Device.model_json_schema()
        assert "examples" in schema

    def test_heartbeat_schema_has_examples(self):
        schema = DeviceHeartbeat.model_json_schema()
        assert "examples" in schema

    def test_sensor_schema_has_examples(self):
        schema = SensorReading.model_json_schema()
        assert "examples" in schema

    def test_ble_device_schema_has_examples(self):
        schema = BleDevice.model_json_schema()
        assert "examples" in schema

    def test_camera_detection_schema_has_examples(self):
        schema = CameraDetection.model_json_schema()
        assert "examples" in schema

    def test_wifi_probe_schema_has_examples(self):
        schema = WiFiProbeRequest.model_json_schema()
        assert "examples" in schema
