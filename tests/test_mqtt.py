"""Tests for tritium_lib.mqtt."""

from tritium_lib.mqtt import (
    TritiumTopics,
    ParsedTopic,
    parse_topic,
    device_heartbeat,
    device_sensors,
    device_commands,
    device_ota_status,
    fleet_broadcast,
)


class TestDeviceTopics:
    def test_heartbeat(self):
        assert device_heartbeat("esp32-001") == "tritium/devices/esp32-001/heartbeat"

    def test_sensors(self):
        assert (
            device_sensors("esp32-001", "temperature")
            == "tritium/devices/esp32-001/sensors/temperature"
        )

    def test_commands(self):
        assert device_commands("esp32-001") == "tritium/devices/esp32-001/commands"

    def test_ota_status(self):
        assert device_ota_status("esp32-001") == "tritium/devices/esp32-001/ota/status"

    def test_fleet_broadcast(self):
        assert fleet_broadcast() == "tritium/fleet/broadcast"


class TestParseTopic:
    def test_heartbeat(self):
        result = parse_topic("tritium/devices/esp32-001/heartbeat")
        assert result is not None
        assert result.device_id == "esp32-001"
        assert result.message_type == "heartbeat"
        assert result.sensor_type is None

    def test_sensor(self):
        result = parse_topic("tritium/devices/esp32-001/sensors/temperature")
        assert result is not None
        assert result.device_id == "esp32-001"
        assert result.message_type == "sensors/temperature"
        assert result.sensor_type == "temperature"

    def test_ota_status(self):
        result = parse_topic("tritium/devices/esp32-001/ota/status")
        assert result is not None
        assert result.device_id == "esp32-001"
        assert result.message_type == "ota/status"

    def test_commands(self):
        result = parse_topic("tritium/devices/my-device-42/commands")
        assert result is not None
        assert result.device_id == "my-device-42"
        assert result.message_type == "commands"

    def test_invalid_topic(self):
        assert parse_topic("some/random/topic") is None

    def test_partial_match(self):
        assert parse_topic("tritium/devices/") is None

    def test_no_message_type(self):
        # Must have at least device_id and message_type
        assert parse_topic("tritium/devices/esp32-001") is None


class TestTritiumTopics:
    def test_site_scoped(self):
        t = TritiumTopics(site_id="lab")
        assert t.edge_heartbeat("d1") == "tritium/lab/edge/d1/heartbeat"
        assert t.sensor("d1", "temp") == "tritium/lab/sensors/d1/temp"

    def test_wildcards(self):
        t = TritiumTopics(site_id="home")
        assert t.all_edge() == "tritium/home/edge/+/#"
        assert t.all_sensors() == "tritium/home/sensors/+/#"

    def test_default_site(self):
        t = TritiumTopics()
        assert t.edge_heartbeat("d1").startswith("tritium/home/")
