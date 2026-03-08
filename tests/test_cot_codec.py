# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for CoT XML codec."""

import xml.etree.ElementTree as ET

from tritium_lib.cot import device_to_cot, parse_cot, sensor_to_cot


class TestDeviceToCot:
    def test_basic_output(self):
        xml_str = device_to_cot("dev-001", lat=38.8977, lng=-77.0365)
        assert "<?xml" in xml_str
        root = ET.fromstring(xml_str)
        assert root.tag == "event"
        assert root.get("uid") == "tritium-edge-dev-001"

    def test_friendly_sensor_type(self):
        xml_str = device_to_cot("dev-001", lat=0, lng=0)
        root = ET.fromstring(xml_str)
        assert root.get("type") == "a-f-G-E-S"

    def test_camera_type(self):
        xml_str = device_to_cot("cam-01", lat=0, lng=0, capabilities=["camera", "display"])
        root = ET.fromstring(xml_str)
        assert root.get("type") == "a-f-G-E-S-C"

    def test_hostile_alliance(self):
        xml_str = device_to_cot("dev-001", lat=0, lng=0, alliance="hostile")
        root = ET.fromstring(xml_str)
        assert root.get("type").startswith("a-h-")

    def test_neutral_alliance(self):
        xml_str = device_to_cot("dev-001", lat=0, lng=0, alliance="neutral")
        root = ET.fromstring(xml_str)
        assert root.get("type").startswith("a-n-")

    def test_point_coordinates(self):
        xml_str = device_to_cot("dev-001", lat=38.8977, lng=-77.0365, alt=50.0)
        root = ET.fromstring(xml_str)
        point = root.find("point")
        assert point is not None
        assert float(point.get("lat")) == 38.8977
        assert float(point.get("lon")) == -77.0365
        assert float(point.get("hae")) == 50.0

    def test_callsign(self):
        xml_str = device_to_cot("dev-001", lat=0, lng=0, callsign="Kitchen Node")
        root = ET.fromstring(xml_str)
        contact = root.find("detail/contact")
        assert contact.get("callsign") == "Kitchen Node"

    def test_default_callsign_is_device_id(self):
        xml_str = device_to_cot("dev-001", lat=0, lng=0)
        root = ET.fromstring(xml_str)
        contact = root.find("detail/contact")
        assert contact.get("callsign") == "dev-001"

    def test_team_color(self):
        xml_str = device_to_cot("dev-001", lat=0, lng=0, alliance="friendly")
        root = ET.fromstring(xml_str)
        group = root.find("detail/__group")
        assert group.get("name") == "Cyan"

    def test_capabilities_in_detail(self):
        xml_str = device_to_cot("dev-001", lat=0, lng=0, capabilities=["camera", "imu"])
        root = ET.fromstring(xml_str)
        edge = root.find("detail/tritium_edge")
        assert edge is not None
        assert edge.get("capabilities") == "camera,imu"

    def test_extra_fields(self):
        xml_str = device_to_cot("dev-001", lat=0, lng=0, extra={"firmware": "1.0.0"})
        root = ET.fromstring(xml_str)
        edge = root.find("detail/tritium_edge")
        assert edge.get("firmware") == "1.0.0"

    def test_relay_role(self):
        xml_str = device_to_cot("dev-001", lat=0, lng=0, capabilities=["lora"])
        root = ET.fromstring(xml_str)
        edge = root.find("detail/tritium_edge")
        assert edge.get("role") == "relay"

    def test_display_role(self):
        xml_str = device_to_cot("dev-001", lat=0, lng=0, capabilities=["display"])
        root = ET.fromstring(xml_str)
        assert root.get("type") == "a-f-G-U-C"


class TestSensorToCot:
    def test_basic_output(self):
        xml_str = sensor_to_cot("dev-001", "temperature", 23.5, lat=0, lng=0)
        root = ET.fromstring(xml_str)
        assert root.get("uid") == "tritium-sensor-dev-001-temperature"
        assert root.get("how") == "m-r"

    def test_remarks(self):
        xml_str = sensor_to_cot("dev-001", "temperature", 23.5, lat=0, lng=0, unit="celsius")
        root = ET.fromstring(xml_str)
        remarks = root.find("detail/remarks")
        assert remarks is not None
        assert "temperature: 23.5 celsius" in remarks.text

    def test_remarks_no_unit(self):
        xml_str = sensor_to_cot("dev-001", "humidity", 55.0, lat=0, lng=0)
        root = ET.fromstring(xml_str)
        remarks = root.find("detail/remarks")
        assert remarks.text == "humidity: 55.0"


class TestParseCot:
    def test_roundtrip_device(self):
        xml_str = device_to_cot(
            "dev-001", lat=38.8977, lng=-77.0365, alt=50.0,
            capabilities=["camera", "imu"], callsign="Test Node",
        )
        parsed = parse_cot(xml_str)
        assert parsed is not None
        assert parsed["uid"] == "tritium-edge-dev-001"
        assert parsed["device_id"] == "dev-001"
        assert abs(parsed["lat"] - 38.8977) < 0.0001
        assert abs(parsed["lng"] - (-77.0365)) < 0.0001
        assert parsed["callsign"] == "Test Node"
        assert "camera" in parsed["capabilities"]

    def test_invalid_xml(self):
        assert parse_cot("not xml") is None

    def test_non_event_root(self):
        assert parse_cot("<other/>") is None
