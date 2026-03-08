"""Tests for tritium_lib.models."""

from datetime import datetime, timezone

from tritium_lib.models import (
    Device,
    DeviceGroup,
    DeviceHeartbeat,
    DeviceCapabilities,
    Command,
    CommandType,
    CommandStatus,
    FirmwareMeta,
    OTAJob,
    OTAStatus,
    SensorReading,
)


class TestDevice:
    def test_create_minimal(self):
        d = Device(device_id="esp32-001")
        assert d.device_id == "esp32-001"
        assert d.status == "offline"
        assert d.mac == ""
        assert d.capabilities == []

    def test_create_full(self):
        d = Device(
            device_id="esp32-001",
            mac="20:6E:F1:9A:12:00",
            board="touch-lcd-35bc",
            firmware_version="1.2.3",
            status="online",
            capabilities=["camera", "imu", "audio"],
            last_seen=datetime(2026, 3, 7, tzinfo=timezone.utc),
        )
        assert d.mac == "20:6E:F1:9A:12:00"
        assert d.board == "touch-lcd-35bc"
        assert "camera" in d.capabilities

    def test_json_roundtrip(self):
        d = Device(device_id="esp32-001", mac="AA:BB:CC:DD:EE:FF")
        j = d.model_dump_json()
        d2 = Device.model_validate_json(j)
        assert d2.device_id == d.device_id
        assert d2.mac == d.mac


class TestDeviceGroup:
    def test_create(self):
        g = DeviceGroup(id="grp-1", name="Lab Sensors", devices=["d1", "d2"])
        assert g.name == "Lab Sensors"
        assert len(g.devices) == 2

    def test_config(self):
        g = DeviceGroup(
            id="grp-1",
            name="Outdoor",
            config={"update_interval": 30, "sleep_enabled": True},
        )
        assert g.config["update_interval"] == 30


class TestDeviceHeartbeat:
    def test_create(self):
        hb = DeviceHeartbeat(
            device_id="esp32-001",
            firmware_version="1.0.0",
            board="touch-lcd-35bc",
            uptime_s=3600,
            free_heap=180000,
            wifi_rssi=-55,
            capabilities=["camera", "imu"],
        )
        assert hb.device_id == "esp32-001"
        assert hb.uptime_s == 3600
        assert hb.wifi_rssi == -55


class TestDeviceCapabilities:
    def test_from_list(self):
        caps = DeviceCapabilities.from_list(["camera", "imu", "custom_sensor"])
        assert caps.camera is True
        assert caps.imu is True
        assert caps.custom["custom_sensor"] is True

    def test_to_list(self):
        caps = DeviceCapabilities(camera=True, audio=True)
        lst = caps.to_list()
        assert "camera" in lst
        assert "audio" in lst


class TestCommand:
    def test_create(self):
        cmd = Command(
            id="cmd-1",
            device_id="esp32-001",
            type=CommandType.REBOOT,
        )
        assert cmd.status == CommandStatus.PENDING
        assert cmd.type == CommandType.REBOOT


class TestFirmwareMeta:
    def test_create(self):
        fw = FirmwareMeta(id="fw-1", version="1.0.0", board="touch-lcd-35bc")
        assert fw.board == "touch-lcd-35bc"


class TestOTAJob:
    def test_create(self):
        job = OTAJob(
            id="ota-1",
            firmware_url="https://example.com/firmware.bin",
            target_devices=["esp32-001", "esp32-002"],
        )
        assert job.status == OTAStatus.PENDING
        assert len(job.target_devices) == 2
        assert job.completed_at is None

    def test_status_transitions(self):
        job = OTAJob(
            id="ota-1",
            firmware_url="https://example.com/firmware.bin",
            status=OTAStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
        )
        assert job.status == OTAStatus.COMPLETED
        assert job.completed_at is not None


class TestSensorReading:
    def test_scalar(self):
        r = SensorReading(
            device_id="esp32-001",
            sensor_type="temperature",
            value=23.5,
            unit="celsius",
        )
        assert r.value == 23.5
        assert r.unit == "celsius"

    def test_dict_value(self):
        r = SensorReading(
            device_id="esp32-001",
            sensor_type="imu",
            value={"ax": 0.1, "ay": 0.0, "az": 9.8},
            unit="m/s2",
        )
        assert isinstance(r.value, dict)
