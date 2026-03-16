# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for floorplan models — floor plans, rooms, indoor positions."""

import pytest
from tritium_lib.models.floorplan import (
    BuildingOccupancy,
    FloorPlan,
    FloorPlanBounds,
    FloorPlanStatus,
    GeoAnchor,
    IndoorPosition,
    PolygonPoint,
    Room,
    RoomOccupancy,
    RoomType,
    WiFiRSSIFingerprint,
)


class TestRoom:
    """Room polygon containment tests."""

    def test_contains_point_inside(self):
        room = Room(
            room_id="r1",
            name="Test Room",
            polygon=[
                PolygonPoint(lat=0.0, lon=0.0),
                PolygonPoint(lat=0.0, lon=1.0),
                PolygonPoint(lat=1.0, lon=1.0),
                PolygonPoint(lat=1.0, lon=0.0),
            ],
        )
        assert room.contains_point(0.5, 0.5) is True

    def test_contains_point_outside(self):
        room = Room(
            room_id="r1",
            name="Test Room",
            polygon=[
                PolygonPoint(lat=0.0, lon=0.0),
                PolygonPoint(lat=0.0, lon=1.0),
                PolygonPoint(lat=1.0, lon=1.0),
                PolygonPoint(lat=1.0, lon=0.0),
            ],
        )
        assert room.contains_point(2.0, 2.0) is False

    def test_contains_point_empty_polygon(self):
        room = Room(room_id="r1", name="Empty", polygon=[])
        assert room.contains_point(0.5, 0.5) is False

    def test_contains_point_two_points(self):
        room = Room(
            room_id="r1",
            name="Line",
            polygon=[
                PolygonPoint(lat=0.0, lon=0.0),
                PolygonPoint(lat=1.0, lon=1.0),
            ],
        )
        assert room.contains_point(0.5, 0.5) is False

    def test_room_types(self):
        for rt in RoomType:
            room = Room(room_id="r1", name="Test", room_type=rt)
            assert room.room_type == rt


class TestFloorPlan:
    """FloorPlan model tests."""

    def test_create_basic(self):
        fp = FloorPlan(
            plan_id="fp_001",
            name="Office Floor 1",
            building="HQ",
            floor_level=1,
        )
        assert fp.plan_id == "fp_001"
        assert fp.name == "Office Floor 1"
        assert fp.building == "HQ"
        assert fp.floor_level == 1
        assert fp.status == FloorPlanStatus.DRAFT

    def test_find_room(self):
        fp = FloorPlan(
            plan_id="fp_001",
            name="Test",
            rooms=[
                Room(
                    room_id="conf_a",
                    name="Conference A",
                    polygon=[
                        PolygonPoint(lat=0.0, lon=0.0),
                        PolygonPoint(lat=0.0, lon=1.0),
                        PolygonPoint(lat=1.0, lon=1.0),
                        PolygonPoint(lat=1.0, lon=0.0),
                    ],
                ),
                Room(
                    room_id="conf_b",
                    name="Conference B",
                    polygon=[
                        PolygonPoint(lat=2.0, lon=2.0),
                        PolygonPoint(lat=2.0, lon=3.0),
                        PolygonPoint(lat=3.0, lon=3.0),
                        PolygonPoint(lat=3.0, lon=2.0),
                    ],
                ),
            ],
        )
        r = fp.find_room(0.5, 0.5)
        assert r is not None
        assert r.room_id == "conf_a"

        r = fp.find_room(2.5, 2.5)
        assert r is not None
        assert r.room_id == "conf_b"

        r = fp.find_room(5.0, 5.0)
        assert r is None

    def test_get_room_by_id(self):
        fp = FloorPlan(
            plan_id="fp_001",
            name="Test",
            rooms=[
                Room(room_id="r1", name="Room 1"),
                Room(room_id="r2", name="Room 2"),
            ],
        )
        assert fp.get_room_by_id("r1").name == "Room 1"
        assert fp.get_room_by_id("r2").name == "Room 2"
        assert fp.get_room_by_id("r3") is None


class TestFloorPlanBounds:
    """FloorPlanBounds tests."""

    def test_contains(self):
        b = FloorPlanBounds(north=40.0, south=39.0, east=-74.0, west=-75.0)
        assert b.contains(39.5, -74.5) is True
        assert b.contains(41.0, -74.5) is False

    def test_center(self):
        b = FloorPlanBounds(north=40.0, south=38.0, east=-74.0, west=-76.0)
        assert b.center_lat == 39.0
        assert b.center_lon == -75.0


class TestIndoorPosition:
    """IndoorPosition model tests."""

    def test_create(self):
        pos = IndoorPosition(
            target_id="ble_AA:BB:CC:DD:EE:FF",
            plan_id="fp_001",
            room_id="conf_a",
            floor_level=1,
            lat=39.5,
            lon=-74.5,
            confidence=0.85,
            method="trilateration",
        )
        assert pos.target_id == "ble_AA:BB:CC:DD:EE:FF"
        assert pos.confidence == 0.85


class TestRoomOccupancy:
    """RoomOccupancy model tests."""

    def test_occupancy_ratio(self):
        occ = RoomOccupancy(
            room_id="r1",
            room_name="Test",
            person_count=3,
            capacity=10,
        )
        assert occ.occupancy_ratio == 0.3

    def test_occupancy_ratio_no_capacity(self):
        occ = RoomOccupancy(
            room_id="r1",
            room_name="Test",
            person_count=3,
        )
        assert occ.occupancy_ratio is None

    def test_occupancy_ratio_full(self):
        occ = RoomOccupancy(
            room_id="r1",
            room_name="Test",
            person_count=10,
            capacity=10,
        )
        assert occ.occupancy_ratio == 1.0


class TestWiFiRSSIFingerprint:
    """WiFi fingerprint model tests."""

    def test_create(self):
        fp = WiFiRSSIFingerprint(
            fingerprint_id="wfp_001",
            plan_id="fp_001",
            lat=39.5,
            lon=-74.5,
            rssi_map={
                "AA:BB:CC:DD:EE:01": -45.0,
                "AA:BB:CC:DD:EE:02": -67.0,
                "AA:BB:CC:DD:EE:03": -82.0,
            },
            device_id="tritium_43c_01",
        )
        assert len(fp.rssi_map) == 3
        assert fp.rssi_map["AA:BB:CC:DD:EE:01"] == -45.0


class TestGeoAnchor:
    """GeoAnchor model tests."""

    def test_create(self):
        anchor = GeoAnchor(
            pixel_x=100,
            pixel_y=200,
            lat=39.5,
            lon=-74.5,
            label="NW corner",
        )
        assert anchor.pixel_x == 100
        assert anchor.label == "NW corner"


class TestImports:
    """Verify all exports are importable from tritium_lib.models."""

    def test_import_from_models(self):
        from tritium_lib.models import (
            BuildingOccupancy,
            FloorPlan,
            FloorPlanBounds,
            FloorPlanStatus,
            GeoAnchor,
            IndoorPosition,
            PolygonPoint,
            Room,
            RoomOccupancy,
            RoomType,
            WiFiRSSIFingerprint,
        )
        assert FloorPlan is not None
        assert Room is not None
        assert IndoorPosition is not None
