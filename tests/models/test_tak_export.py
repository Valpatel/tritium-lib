# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for CoT export models — tak_export.py."""

import xml.etree.ElementTree as ET

import pytest

from tritium_lib.models.tak_export import (
    CoTExportEvent,
    CoTExportPoint,
    targets_to_cot_xml,
    targets_to_cot_file,
)


class TestCoTExportPoint:
    """CoTExportPoint model tests."""

    def test_defaults(self):
        p = CoTExportPoint()
        assert p.lat == 0.0
        assert p.lon == 0.0
        assert p.hae == 0.0
        assert p.ce == 10.0
        assert p.le == 10.0

    def test_custom_values(self):
        p = CoTExportPoint(lat=33.5, lon=-117.2, hae=50.0, ce=5.0, le=5.0)
        assert p.lat == 33.5
        assert p.lon == -117.2


class TestCoTExportEvent:
    """CoTExportEvent model tests."""

    def test_from_friendly_person(self):
        target = {
            "target_id": "ble_aa:bb:cc:dd:ee:ff",
            "name": "Alice",
            "alliance": "friendly",
            "asset_type": "person",
            "lat": 33.5,
            "lng": -117.2,
            "source": "ble",
            "battery": 0.85,
            "speed": 1.2,
            "heading": 90.0,
            "status": "active",
        }
        evt = CoTExportEvent.from_target_dict(target)
        assert evt.uid == "ble_aa:bb:cc:dd:ee:ff"
        assert evt.cot_type == "a-f-G-U-C"
        assert evt.callsign == "Alice"
        assert evt.team_color == "Cyan"
        assert evt.point.lat == 33.5
        assert evt.point.lon == -117.2
        assert evt.battery_pct == 85.0
        assert evt.speed == 1.2
        assert evt.course == 90.0

    def test_from_hostile_drone(self):
        target = {
            "target_id": "det_drone_1",
            "name": "Hostile Drone",
            "alliance": "hostile",
            "asset_type": "drone",
            "lat": 34.0,
            "lng": -118.0,
            "source": "yolo",
        }
        evt = CoTExportEvent.from_target_dict(target)
        assert evt.cot_type == "a-h-A-M-F-Q"
        assert evt.team_color == "Red"
        assert evt.how == "m-r"

    def test_to_xml_valid(self):
        target = {
            "target_id": "mesh_abc123",
            "name": "Relay Node",
            "alliance": "friendly",
            "asset_type": "mesh_radio",
            "lat": 33.0,
            "lng": -117.0,
            "source": "mesh",
        }
        evt = CoTExportEvent.from_target_dict(target)
        xml_str = evt.to_xml()

        # Parse the XML to verify it's valid
        root = ET.fromstring(xml_str)
        assert root.tag == "event"
        assert root.get("uid") == "mesh_abc123"
        assert root.get("type") == "a-f-G-E-S"

        # Check point
        point = root.find("point")
        assert point is not None
        assert float(point.get("lat")) == pytest.approx(33.0, abs=0.001)

        # Check detail
        detail = root.find("detail")
        assert detail is not None
        contact = detail.find("contact")
        assert contact.get("callsign") == "Relay Node"

    def test_unknown_asset_type_fallback(self):
        target = {
            "target_id": "test_1",
            "alliance": "unknown",
            "asset_type": "alien_craft",
        }
        evt = CoTExportEvent.from_target_dict(target)
        assert evt.cot_type == "a-u-G"

    def test_position_fallback_to_xy(self):
        target = {
            "target_id": "test_2",
            "position": {"x": -117.5, "y": 33.8},
        }
        evt = CoTExportEvent.from_target_dict(target)
        assert evt.point.lon == -117.5
        assert evt.point.lat == 33.8


class TestTargetsToCotXml:
    """Test batch export functions."""

    def test_empty_list(self):
        result = targets_to_cot_xml([])
        assert result == ""

    def test_single_target(self):
        targets = [{
            "target_id": "t1",
            "name": "Target 1",
            "alliance": "friendly",
            "asset_type": "person",
            "lat": 33.0,
            "lng": -117.0,
        }]
        result = targets_to_cot_xml(targets)
        assert "<event" in result
        assert 'uid="t1"' in result

    def test_multiple_targets(self):
        targets = [
            {"target_id": f"t{i}", "alliance": "friendly", "lat": 33.0 + i * 0.01, "lng": -117.0}
            for i in range(5)
        ]
        result = targets_to_cot_xml(targets)
        for i in range(5):
            assert f'uid="t{i}"' in result


class TestTargetsToCotFile:
    """Test file export wrapper."""

    def test_file_format(self):
        targets = [
            {"target_id": "t1", "alliance": "friendly", "asset_type": "person"},
        ]
        result = targets_to_cot_file(targets)
        assert result.startswith('<?xml version="1.0"')
        assert '<cot-events' in result
        assert 'count="1"' in result
        assert '</cot-events>' in result
        assert '<event' in result
