# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Extended CoT codec tests — round-trip encoding, alliance types, sensor reports."""

import xml.etree.ElementTree as ET

import pytest

from tritium_lib.cot.codec import (
    _infer_role,
    device_to_cot,
    parse_cot,
    sensor_to_cot,
)


# ── Role inference ──────────────────────────────────────────────────

class TestInferRole:
    """Tests for _infer_role()."""

    def test_camera_capability(self):
        assert _infer_role(["ble", "camera", "wifi"]) == "camera"

    def test_lora_capability(self):
        assert _infer_role(["lora"]) == "relay"

    def test_mesh_capability(self):
        assert _infer_role(["mesh"]) == "relay"

    def test_display_capability(self):
        assert _infer_role(["display"]) == "display"

    def test_default_sensor(self):
        assert _infer_role(["ble", "wifi"]) == "sensor"

    def test_empty_capabilities(self):
        assert _infer_role([]) == "sensor"

    def test_camera_takes_priority(self):
        assert _infer_role(["camera", "lora", "display"]) == "camera"


# ── device_to_cot ──────────────────────────────────────────────────

class TestDeviceToCot:
    """Tests for device_to_cot() XML generation."""

    def test_returns_valid_xml(self):
        xml_str = device_to_cot("esp32-001", lat=40.0, lng=-74.0)
        root = ET.fromstring(xml_str)
        assert root.tag == "event"

    def test_uid_contains_device_id(self):
        xml_str = device_to_cot("node-42", lat=0, lng=0)
        root = ET.fromstring(xml_str)
        assert root.get("uid") == "tritium-edge-node-42"

    def test_friendly_type_code(self):
        xml_str = device_to_cot("n1", lat=0, lng=0, alliance="friendly")
        root = ET.fromstring(xml_str)
        assert root.get("type", "").startswith("a-f-")

    def test_hostile_type_code(self):
        xml_str = device_to_cot("n1", lat=0, lng=0, alliance="hostile")
        root = ET.fromstring(xml_str)
        assert root.get("type", "").startswith("a-h-")

    def test_neutral_type_code(self):
        xml_str = device_to_cot("n1", lat=0, lng=0, alliance="neutral")
        root = ET.fromstring(xml_str)
        assert root.get("type", "").startswith("a-n-")

    def test_unknown_type_code(self):
        xml_str = device_to_cot("n1", lat=0, lng=0, alliance="unknown")
        root = ET.fromstring(xml_str)
        assert root.get("type", "").startswith("a-u-")

    def test_point_element_present(self):
        xml_str = device_to_cot("n1", lat=40.1234567, lng=-74.9876543, alt=100.5)
        root = ET.fromstring(xml_str)
        point = root.find("point")
        assert point is not None
        assert float(point.get("lat")) == pytest.approx(40.1234567, abs=1e-6)
        assert float(point.get("lon")) == pytest.approx(-74.9876543, abs=1e-6)
        assert float(point.get("hae")) == pytest.approx(100.5, abs=0.5)

    def test_callsign_defaults_to_device_id(self):
        xml_str = device_to_cot("my-node", lat=0, lng=0)
        root = ET.fromstring(xml_str)
        contact = root.find(".//contact")
        assert contact is not None
        assert contact.get("callsign") == "my-node"

    def test_custom_callsign(self):
        xml_str = device_to_cot("n1", lat=0, lng=0, callsign="Alpha-1")
        root = ET.fromstring(xml_str)
        contact = root.find(".//contact")
        assert contact.get("callsign") == "Alpha-1"

    def test_team_color_for_friendly(self):
        xml_str = device_to_cot("n1", lat=0, lng=0, alliance="friendly")
        root = ET.fromstring(xml_str)
        group = root.find(".//__group")
        assert group is not None
        assert group.get("name") == "Cyan"

    def test_team_color_for_hostile(self):
        xml_str = device_to_cot("n1", lat=0, lng=0, alliance="hostile")
        root = ET.fromstring(xml_str)
        group = root.find(".//__group")
        assert group.get("name") == "Red"

    def test_capabilities_in_detail(self):
        xml_str = device_to_cot("n1", lat=0, lng=0, capabilities=["ble", "wifi"])
        root = ET.fromstring(xml_str)
        edge = root.find(".//tritium_edge")
        assert edge is not None
        assert edge.get("capabilities") == "ble,wifi"

    def test_extra_fields_in_detail(self):
        xml_str = device_to_cot(
            "n1", lat=0, lng=0,
            extra={"firmware": "1.2.3", "battery": "85"},
        )
        root = ET.fromstring(xml_str)
        edge = root.find(".//tritium_edge")
        assert edge.get("firmware") == "1.2.3"
        assert edge.get("battery") == "85"

    def test_how_is_machine_gps(self):
        xml_str = device_to_cot("n1", lat=0, lng=0)
        root = ET.fromstring(xml_str)
        assert root.get("how") == "m-g"

    def test_version_is_2_0(self):
        xml_str = device_to_cot("n1", lat=0, lng=0)
        root = ET.fromstring(xml_str)
        assert root.get("version") == "2.0"

    def test_camera_type_code(self):
        xml_str = device_to_cot("n1", lat=0, lng=0, capabilities=["camera"])
        root = ET.fromstring(xml_str)
        assert root.get("type") == "a-f-G-E-S-C"


# ── sensor_to_cot ──────────────────────────────────────────────────

class TestSensorToCot:
    """Tests for sensor_to_cot() XML generation."""

    def test_returns_valid_xml(self):
        xml_str = sensor_to_cot("s1", "temperature", 23.5, lat=40.0, lng=-74.0)
        root = ET.fromstring(xml_str)
        assert root.tag == "event"

    def test_uid_format(self):
        xml_str = sensor_to_cot("s1", "humidity", 65.0, lat=0, lng=0)
        root = ET.fromstring(xml_str)
        assert root.get("uid") == "tritium-sensor-s1-humidity"

    def test_remarks_contain_value(self):
        xml_str = sensor_to_cot("s1", "temperature", 23.5, lat=0, lng=0, unit="C")
        root = ET.fromstring(xml_str)
        remarks = root.find(".//remarks")
        assert remarks is not None
        assert "23.5" in remarks.text
        assert "C" in remarks.text

    def test_remarks_without_unit(self):
        xml_str = sensor_to_cot("s1", "rssi", -72, lat=0, lng=0)
        root = ET.fromstring(xml_str)
        remarks = root.find(".//remarks")
        assert "rssi" in remarks.text

    def test_how_is_machine_reported(self):
        xml_str = sensor_to_cot("s1", "temp", 20.0, lat=0, lng=0)
        root = ET.fromstring(xml_str)
        assert root.get("how") == "m-r"

    def test_point_coordinates(self):
        xml_str = sensor_to_cot("s1", "temp", 20.0, lat=51.5, lng=-0.1, alt=50.0)
        root = ET.fromstring(xml_str)
        point = root.find("point")
        assert float(point.get("lat")) == pytest.approx(51.5, abs=1e-5)
        assert float(point.get("lon")) == pytest.approx(-0.1, abs=1e-5)
        assert float(point.get("hae")) == pytest.approx(50.0, abs=0.5)


# ── parse_cot ───────────────────────────────────────────────────────

class TestParseCot:
    """Tests for parse_cot() XML parsing."""

    def test_round_trip_device(self):
        xml_str = device_to_cot(
            "esp32-001", lat=40.7, lng=-74.0, alt=10.0,
            capabilities=["ble", "wifi"], callsign="Bravo",
        )
        result = parse_cot(xml_str)
        assert result is not None
        assert result["uid"] == "tritium-edge-esp32-001"
        assert result["callsign"] == "Bravo"
        assert result["device_id"] == "esp32-001"
        assert result["capabilities"] == ["ble", "wifi"]
        assert result["lat"] == pytest.approx(40.7, abs=1e-5)
        assert result["lng"] == pytest.approx(-74.0, abs=1e-5)

    def test_round_trip_sensor(self):
        xml_str = sensor_to_cot("s1", "temperature", 23.5, lat=40.0, lng=-74.0)
        result = parse_cot(xml_str)
        assert result is not None
        assert result["uid"] == "tritium-sensor-s1-temperature"

    def test_invalid_xml_returns_none(self):
        assert parse_cot("not xml at all") is None

    def test_non_event_root_returns_none(self):
        xml_str = "<data><value>123</value></data>"
        assert parse_cot(xml_str) is None

    def test_missing_point_element(self):
        xml_str = '<event version="2.0" uid="x" type="a-f-G" how="m-g" time="" stale=""><detail/></event>'
        result = parse_cot(xml_str)
        assert result is not None
        assert "lat" not in result

    def test_missing_detail_element(self):
        xml_str = '<event version="2.0" uid="x" type="a-f-G" how="m-g" time="" stale=""><point lat="0" lon="0" hae="0" ce="10" le="10"/></event>'
        result = parse_cot(xml_str)
        assert result is not None
        assert "callsign" not in result

    def test_empty_capabilities_parsed_as_empty_list(self):
        xml_str = device_to_cot("n1", lat=0, lng=0)
        result = parse_cot(xml_str)
        assert result is not None
        # No capabilities passed, so tritium_edge element won't have capabilities attr
        # or it will be empty
        caps = result.get("capabilities", [])
        assert isinstance(caps, list)
