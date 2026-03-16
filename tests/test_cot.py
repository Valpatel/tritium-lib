"""Tests for tritium_lib.cot."""

from tritium_lib.cot import device_to_cot, sensor_to_cot, parse_cot


class TestDeviceToCot:
    def test_basic(self):
        xml = device_to_cot("esp32-001", lat=37.7749, lng=-122.4194)
        assert "tritium-edge-esp32-001" in xml
        assert "37.7749" in xml
        assert "-122.4194" in xml

    def test_camera_type(self):
        xml = device_to_cot(
            "esp32-001", lat=0, lng=0, capabilities=["camera", "imu"]
        )
        parsed = parse_cot(xml)
        assert parsed["type"] == "a-f-G-E-S-C"

    def test_hostile_alliance(self):
        xml = device_to_cot("esp32-001", lat=0, lng=0, alliance="hostile")
        parsed = parse_cot(xml)
        assert parsed["type"].startswith("a-h-")

    def test_callsign(self):
        xml = device_to_cot("esp32-001", lat=0, lng=0, callsign="Alpha-1")
        parsed = parse_cot(xml)
        assert parsed["callsign"] == "Alpha-1"

    def test_extra_fields(self):
        xml = device_to_cot(
            "esp32-001", lat=0, lng=0, extra={"firmware": "1.2.3"}
        )
        assert "firmware" in xml
        assert "1.2.3" in xml


class TestSensorToCot:
    def test_temperature(self):
        xml = sensor_to_cot(
            "esp32-001", "temperature", 23.5, lat=37.0, lng=-122.0, unit="C"
        )
        assert "tritium-sensor-esp32-001-temperature" in xml
        assert "temperature: 23.5 C" in xml

    def test_no_unit(self):
        xml = sensor_to_cot("esp32-001", "humidity", 65.0, lat=0, lng=0)
        assert "humidity: 65.0" in xml


class TestParseCot:
    def test_roundtrip(self):
        xml = device_to_cot(
            "esp32-001",
            lat=37.7749,
            lng=-122.4194,
            alt=50.0,
            capabilities=["camera", "imu"],
        )
        parsed = parse_cot(xml)
        assert parsed is not None
        assert parsed["device_id"] == "esp32-001"
        assert abs(parsed["lat"] - 37.7749) < 0.001
        assert abs(parsed["lng"] - (-122.4194)) < 0.001
        assert "camera" in parsed["capabilities"]

    def test_invalid_xml(self):
        assert parse_cot("not xml at all") is None

    def test_non_event_xml(self):
        assert parse_cot("<root/>") is None
