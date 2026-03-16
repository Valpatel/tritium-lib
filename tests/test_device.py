# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for device models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tritium_lib.models.device import (
    Device,
    DeviceCapabilities,
    DeviceGroup,
    DeviceHeartbeat,
)


def _utc(hour=12, minute=0, second=0):
    return datetime(2026, 3, 7, hour, minute, second, tzinfo=timezone.utc)


class TestDeviceCapabilities:
    def test_defaults_all_false(self):
        caps = DeviceCapabilities()
        assert caps.camera is False
        assert caps.audio is False
        assert caps.imu is False
        assert caps.display is False
        assert caps.touch is False
        assert caps.rtc is False
        assert caps.power is False
        assert caps.mesh is False
        assert caps.lora is False
        assert caps.gps is False
        assert caps.temperature is False
        assert caps.humidity is False
        assert caps.custom == {}

    def test_from_list_known(self):
        caps = DeviceCapabilities.from_list(["camera", "display", "imu"])
        assert caps.camera is True
        assert caps.display is True
        assert caps.imu is True
        assert caps.audio is False

    def test_from_list_custom(self):
        caps = DeviceCapabilities.from_list(["camera", "lidar", "radar"])
        assert caps.camera is True
        assert caps.custom == {"lidar": True, "radar": True}

    def test_from_list_empty(self):
        caps = DeviceCapabilities.from_list([])
        assert caps.to_list() == []

    def test_to_list(self):
        caps = DeviceCapabilities(camera=True, display=True, imu=True)
        result = caps.to_list()
        assert result == ["camera", "display", "imu"]

    def test_to_list_with_custom(self):
        caps = DeviceCapabilities(camera=True, custom={"lidar": True, "sonar": False})
        result = caps.to_list()
        assert "camera" in result
        assert "lidar" in result
        assert "sonar" not in result

    def test_roundtrip_from_list_to_list(self):
        original = ["audio", "camera", "display", "touch"]
        caps = DeviceCapabilities.from_list(original)
        result = caps.to_list()
        assert result == sorted(original)

    def test_serialization(self):
        caps = DeviceCapabilities(camera=True, lora=True)
        d = caps.model_dump()
        assert d["camera"] is True
        assert d["lora"] is True
        assert d["audio"] is False

    def test_json_roundtrip(self):
        caps = DeviceCapabilities.from_list(["camera", "gps", "custom_sensor"])
        json_str = caps.model_dump_json()
        caps2 = DeviceCapabilities.model_validate_json(json_str)
        assert caps2.camera == caps.camera
        assert caps2.gps == caps.gps
        assert caps2.custom == caps.custom


class TestDeviceHeartbeat:
    def test_create_minimal(self):
        hb = DeviceHeartbeat(device_id="dev-1")
        assert hb.device_id == "dev-1"
        assert hb.firmware_version == "unknown"
        assert hb.board == "unknown"
        assert hb.family == "esp32"
        assert hb.capabilities == []
        assert hb.command_acks == []

    def test_create_full(self):
        hb = DeviceHeartbeat(
            device_id="dev-2",
            device_token="tok-abc",
            firmware_version="1.2.3",
            firmware_hash="abcdef1234",
            board="touch-lcd-349",
            family="esp32",
            uptime_s=3600,
            free_heap=200000,
            wifi_rssi=-55,
            ip_address="192.168.86.42",
            boot_count=5,
            reported_config={"interval": 30},
            capabilities=["camera", "display"],
            ota_status="idle",
            ota_result=None,
            command_acks=[{"id": "cmd-1", "status": "ok"}],
            mesh_peers=3,
            timestamp=1709820000,
        )
        assert hb.uptime_s == 3600
        assert hb.wifi_rssi == -55
        assert len(hb.capabilities) == 2
        assert len(hb.command_acks) == 1

    def test_serialization(self):
        hb = DeviceHeartbeat(
            device_id="dev-3",
            firmware_version="2.0.0",
            capabilities=["imu", "rtc"],
        )
        d = hb.model_dump()
        assert d["device_id"] == "dev-3"
        assert d["capabilities"] == ["imu", "rtc"]

    def test_json_roundtrip(self):
        hb = DeviceHeartbeat(
            device_id="dev-4",
            uptime_s=120,
            free_heap=150000,
        )
        json_str = hb.model_dump_json()
        hb2 = DeviceHeartbeat.model_validate_json(json_str)
        assert hb2.device_id == hb.device_id
        assert hb2.uptime_s == hb.uptime_s

    def test_none_optional_fields(self):
        hb = DeviceHeartbeat(device_id="dev-5")
        assert hb.device_token is None
        assert hb.firmware_hash is None
        assert hb.uptime_s is None
        assert hb.free_heap is None
        assert hb.wifi_rssi is None
        assert hb.ip_address is None
        assert hb.boot_count is None
        assert hb.reported_config is None
        assert hb.ota_status is None
        assert hb.ota_result is None
        assert hb.mesh_peers is None
        assert hb.timestamp is None


class TestDevice:
    def test_create_minimal(self):
        dev = Device(device_id="dev-1")
        assert dev.device_id == "dev-1"
        assert dev.device_name == ""
        assert dev.mac == ""
        assert dev.board == "unknown"
        assert dev.status == "offline"
        assert dev.capabilities == []
        assert dev.tags == []

    def test_create_full(self):
        dev = Device(
            device_id="dev-2",
            device_name="Kitchen Sensor",
            mac="1C:DB:D4:9C:CD:68",
            board="touch-amoled-241b",
            family="esp32",
            firmware_version="1.0.0",
            firmware_hash="abc123",
            ip_address="192.168.86.42",
            capabilities=["camera", "display", "imu"],
            status="online",
            last_seen=_utc(),
            registered_at=_utc(8),
            tags=["kitchen", "floor-1"],
            notes="Near the fridge",
        )
        assert dev.device_name == "Kitchen Sensor"
        assert dev.status == "online"
        assert len(dev.capabilities) == 3
        assert len(dev.tags) == 2

    def test_serialization(self):
        dev = Device(
            device_id="dev-3",
            board="touch-lcd-35bc",
            status="online",
            last_seen=_utc(),
        )
        d = dev.model_dump()
        assert d["device_id"] == "dev-3"
        assert d["board"] == "touch-lcd-35bc"
        assert d["status"] == "online"

    def test_json_roundtrip(self):
        dev = Device(
            device_id="dev-4",
            mac="20:6E:F1:9A:24:E8",
            capabilities=["display", "touch"],
            tags=["lab"],
            last_seen=_utc(),
        )
        json_str = dev.model_dump_json()
        dev2 = Device.model_validate_json(json_str)
        assert dev2.device_id == dev.device_id
        assert dev2.mac == dev.mac
        assert dev2.capabilities == dev.capabilities

    def test_empty_collections(self):
        dev = Device(device_id="dev-5")
        assert dev.capabilities == []
        assert dev.tags == []

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            Device()  # missing device_id


class TestDeviceGroup:
    def test_create_minimal(self):
        grp = DeviceGroup(id="grp-1", name="Lab Devices")
        assert grp.id == "grp-1"
        assert grp.name == "Lab Devices"
        assert grp.devices == []
        assert grp.config == {}

    def test_create_full(self):
        grp = DeviceGroup(
            id="grp-2",
            name="Floor 1",
            devices=["dev-a", "dev-b", "dev-c"],
            config={"heartbeat_interval": 60, "ota_channel": "stable"},
            created_at=_utc(8),
            updated_at=_utc(10),
        )
        assert len(grp.devices) == 3
        assert grp.config["heartbeat_interval"] == 60

    def test_serialization(self):
        grp = DeviceGroup(
            id="grp-3",
            name="Test Group",
            devices=["dev-x"],
            config={"key": "value"},
        )
        d = grp.model_dump()
        assert d["name"] == "Test Group"
        assert d["devices"] == ["dev-x"]

    def test_json_roundtrip(self):
        grp = DeviceGroup(
            id="grp-4",
            name="Outdoor",
            devices=["dev-1", "dev-2"],
            created_at=_utc(),
        )
        json_str = grp.model_dump_json()
        grp2 = DeviceGroup.model_validate_json(json_str)
        assert grp2.id == grp.id
        assert grp2.devices == grp.devices

    def test_empty_group(self):
        grp = DeviceGroup(id="grp-5", name="Empty")
        assert grp.devices == []
        assert grp.config == {}
        assert grp.created_at is None
        assert grp.updated_at is None

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            DeviceGroup(id="grp-6")  # missing name
