# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.indoor — WiFi/BLE location fingerprinting.

Covers: Fingerprint, FingerprintDB, PositionEstimator, FloorPlan, ZoneMapper.
"""

import math
import pytest

from tritium_lib.indoor import (
    Fingerprint,
    FingerprintDB,
    PositionEstimator,
    PositionResult,
    FloorPlan,
    Room,
    Door,
    RoomType,
    ZoneMapper,
    Zone,
)


# ===========================================================================
# Helper: build a small reference database
# ===========================================================================

def _make_test_db() -> FingerprintDB:
    """Create a small fingerprint DB simulating a hallway with 3 APs."""
    db = FingerprintDB(building_id="test-building")
    # Position near AP1 (strong AP1, weak AP2/AP3)
    db.add(Fingerprint(x=0.0, y=0.0, floor=0, label="entrance",
                        rssi={"ap1": -30, "ap2": -70, "ap3": -80}))
    # Position in the middle
    db.add(Fingerprint(x=5.0, y=0.0, floor=0, label="middle",
                        rssi={"ap1": -55, "ap2": -50, "ap3": -60}))
    # Position near AP2 (weak AP1, strong AP2)
    db.add(Fingerprint(x=10.0, y=0.0, floor=0, label="far-end",
                        rssi={"ap1": -75, "ap2": -30, "ap3": -45}))
    # Second floor
    db.add(Fingerprint(x=5.0, y=5.0, floor=1, label="upstairs",
                        rssi={"ap1": -80, "ap2": -80, "ap4": -40}))
    return db


# ===========================================================================
# Fingerprint Tests
# ===========================================================================

class TestFingerprint:
    """Tests for the Fingerprint data class."""

    def test_create_basic(self):
        fp = Fingerprint(x=1.0, y=2.0, rssi={"ap1": -50, "ap2": -60})
        assert fp.x == 1.0
        assert fp.y == 2.0
        assert fp.floor == 0
        assert fp.ap_count == 2

    def test_auto_id(self):
        """Each fingerprint gets a unique auto-generated ID."""
        fp1 = Fingerprint(x=0, y=0)
        fp2 = Fingerprint(x=0, y=0)
        assert fp1.fingerprint_id != fp2.fingerprint_id

    def test_mean_rssi(self):
        fp = Fingerprint(x=0, y=0, rssi={"a": -40, "b": -60})
        assert fp.mean_rssi == -50.0

    def test_mean_rssi_empty(self):
        fp = Fingerprint(x=0, y=0)
        assert fp.mean_rssi == -100.0

    def test_strongest_ap(self):
        fp = Fingerprint(x=0, y=0, rssi={"a": -70, "b": -30, "c": -50})
        assert fp.strongest_ap == "b"

    def test_strongest_ap_empty(self):
        fp = Fingerprint(x=0, y=0)
        assert fp.strongest_ap is None

    def test_rssi_distance_identical(self):
        """Distance to identical RSSI should be 0."""
        fp = Fingerprint(x=0, y=0, rssi={"a": -50, "b": -60})
        dist = fp.rssi_distance({"a": -50, "b": -60})
        assert dist == 0.0

    def test_rssi_distance_different(self):
        """Distance increases with RSSI difference."""
        fp = Fingerprint(x=0, y=0, rssi={"a": -50, "b": -60})
        dist = fp.rssi_distance({"a": -50, "b": -70})
        assert dist == 10.0  # sqrt((0)^2 + (10)^2) = 10

    def test_rssi_distance_no_overlap(self):
        """No common APs => infinite distance."""
        fp = Fingerprint(x=0, y=0, rssi={"a": -50})
        dist = fp.rssi_distance({"b": -60})
        assert dist == float("inf")

    def test_rssi_distance_partial_overlap(self):
        """Only common APs contribute to distance."""
        fp = Fingerprint(x=0, y=0, rssi={"a": -50, "b": -60, "c": -70})
        dist = fp.rssi_distance({"a": -50, "d": -40})
        assert dist == 0.0  # only 'a' is common, and it matches

    def test_weighted_rssi_distance(self):
        """Weighted distance penalises missing APs."""
        fp = Fingerprint(x=0, y=0, rssi={"a": -50, "b": -60})
        # Normal distance (only common AP 'a')
        normal = fp.rssi_distance({"a": -50, "c": -40})
        # Weighted distance (penalises missing 'b' and 'c')
        weighted = fp.weighted_rssi_distance({"a": -50, "c": -40})
        assert weighted > normal

    def test_to_dict_and_from_dict(self):
        """Round-trip serialisation."""
        fp = Fingerprint(x=3.5, y=7.2, floor=2, label="lab",
                          rssi={"ap1": -45, "ap2": -67})
        d = fp.to_dict()
        assert d["x"] == 3.5
        assert d["floor"] == 2
        assert d["ap_count"] == 2

        fp2 = Fingerprint.from_dict(d)
        assert fp2.x == fp.x
        assert fp2.y == fp.y
        assert fp2.floor == fp.floor
        assert fp2.rssi == fp.rssi
        assert fp2.label == fp.label


# ===========================================================================
# FingerprintDB Tests
# ===========================================================================

class TestFingerprintDB:
    """Tests for the fingerprint database."""

    def test_add_and_count(self):
        db = FingerprintDB("test")
        assert db.count == 0
        db.add(Fingerprint(x=0, y=0, rssi={"a": -50}))
        assert db.count == 1

    def test_add_replaces_duplicate_id(self):
        db = FingerprintDB("test")
        fp = Fingerprint(x=0, y=0, fingerprint_id="fp1", rssi={"a": -50})
        db.add(fp)
        fp2 = Fingerprint(x=1, y=1, fingerprint_id="fp1", rssi={"a": -30})
        db.add(fp2)
        assert db.count == 1
        assert db.get("fp1").x == 1.0

    def test_add_many(self):
        db = FingerprintDB("test")
        fps = [Fingerprint(x=i, y=0) for i in range(5)]
        added = db.add_many(fps)
        assert added == 5
        assert db.count == 5

    def test_remove(self):
        db = FingerprintDB("test")
        fp = Fingerprint(x=0, y=0, fingerprint_id="fp1")
        db.add(fp)
        assert db.remove("fp1") is True
        assert db.count == 0
        assert db.remove("fp1") is False

    def test_clear(self):
        db = _make_test_db()
        assert db.count > 0
        db.clear()
        assert db.count == 0

    def test_get(self):
        db = FingerprintDB("test")
        fp = Fingerprint(x=1, y=2, fingerprint_id="fp1")
        db.add(fp)
        assert db.get("fp1") is fp
        assert db.get("nonexistent") is None

    def test_filter_by_floor(self):
        db = _make_test_db()
        floor0 = db.filter_by_floor(0)
        floor1 = db.filter_by_floor(1)
        assert len(floor0) == 3
        assert len(floor1) == 1
        assert all(fp.floor == 0 for fp in floor0)

    def test_get_all_aps(self):
        db = _make_test_db()
        aps = db.get_all_aps()
        assert "ap1" in aps
        assert "ap2" in aps
        assert "ap3" in aps
        assert "ap4" in aps

    def test_get_floors(self):
        db = _make_test_db()
        assert db.get_floors() == [0, 1]

    def test_find_nearest_basic(self):
        db = _make_test_db()
        # Query near the entrance (strong AP1)
        results = db.find_nearest({"ap1": -32, "ap2": -68, "ap3": -78}, k=2)
        assert len(results) == 2
        # Closest should be the entrance
        assert results[0][0].label == "entrance"
        # Distances should be ascending
        assert results[0][1] <= results[1][1]

    def test_find_nearest_floor_filter(self):
        db = _make_test_db()
        results = db.find_nearest({"ap4": -42}, k=5, floor=1)
        assert len(results) == 1
        assert results[0][0].label == "upstairs"

    def test_find_nearest_max_distance(self):
        db = _make_test_db()
        # Very tight max_distance should filter out distant matches
        results = db.find_nearest({"ap1": -30, "ap2": -70, "ap3": -80},
                                   k=10, max_distance=5.0)
        assert len(results) >= 1
        assert all(d <= 5.0 for _, d in results)

    def test_to_dict_and_from_dict(self):
        db = _make_test_db()
        d = db.to_dict()
        assert d["building_id"] == "test-building"
        assert d["count"] == 4

        db2 = FingerprintDB.from_dict(d)
        assert db2.building_id == "test-building"
        assert db2.count == 4

    def test_get_status(self):
        db = _make_test_db()
        status = db.get_status()
        assert status["fingerprint_count"] == 4
        assert status["floor_count"] == 2
        assert status["ap_count"] == 4


# ===========================================================================
# PositionEstimator Tests
# ===========================================================================

class TestPositionEstimator:
    """Tests for k-NN position estimation."""

    def test_estimate_exact_match(self):
        """When live RSSI exactly matches a reference, return that position."""
        db = _make_test_db()
        est = PositionEstimator(db, k=3)
        result = est.estimate({"ap1": -30, "ap2": -70, "ap3": -80})
        # Should be very close to entrance (0, 0)
        assert abs(result.x - 0.0) < 2.0
        assert abs(result.y - 0.0) < 1.0
        assert result.confidence > 0.5

    def test_estimate_between_points(self):
        """RSSI between two references should estimate a position between them."""
        db = _make_test_db()
        est = PositionEstimator(db, k=3)
        # RSSI midway between entrance and middle
        result = est.estimate({"ap1": -42, "ap2": -60, "ap3": -70})
        # Should be somewhere between x=0 and x=5
        assert 0.0 <= result.x <= 6.0

    def test_estimate_empty_rssi(self):
        """Empty RSSI returns zero-confidence result."""
        db = _make_test_db()
        est = PositionEstimator(db, k=3)
        result = est.estimate({})
        assert result.confidence == 0.0

    def test_estimate_empty_db(self):
        """Empty database returns zero-confidence result."""
        db = FingerprintDB("empty")
        est = PositionEstimator(db, k=3)
        result = est.estimate({"ap1": -50})
        assert result.confidence == 0.0

    def test_estimate_floor_restriction(self):
        """Floor filter should restrict to that floor only."""
        db = _make_test_db()
        est = PositionEstimator(db, k=3, min_common_aps=1)
        result = est.estimate({"ap4": -42}, floor=1)
        assert result.floor == 1
        assert result.confidence > 0.0

    def test_estimate_single_match(self):
        """With only one valid match, return that position directly."""
        db = FingerprintDB("single")
        db.add(Fingerprint(x=7.0, y=3.0, rssi={"ap1": -50}))
        est = PositionEstimator(db, k=3, min_common_aps=1)
        result = est.estimate({"ap1": -52})
        assert result.x == 7.0
        assert result.y == 3.0
        assert result.k_used == 1

    def test_estimate_no_common_aps(self):
        """No common APs with any reference => zero confidence."""
        db = _make_test_db()
        est = PositionEstimator(db, k=3, min_common_aps=2)
        result = est.estimate({"unknown_ap": -50})
        assert result.confidence == 0.0

    def test_k_setter(self):
        db = _make_test_db()
        est = PositionEstimator(db, k=5)
        assert est.k == 5
        est.k = 1
        assert est.k == 1
        est.k = -1  # should clamp to 1
        assert est.k == 1

    def test_estimate_floor_detection(self):
        """estimate_floor should return the correct floor."""
        db = _make_test_db()
        est = PositionEstimator(db, k=3, min_common_aps=1)
        # AP4 is only on floor 1
        floor = est.estimate_floor({"ap4": -42})
        assert floor == 1

    def test_estimate_floor_empty(self):
        db = FingerprintDB("empty")
        est = PositionEstimator(db, k=3)
        assert est.estimate_floor({}) is None

    def test_result_to_dict(self):
        result = PositionResult(x=3.14, y=2.72, floor=1, confidence=0.85)
        d = result.to_dict()
        assert d["x"] == 3.14
        assert d["y"] == 2.72
        assert d["floor"] == 1
        assert d["confidence"] == 0.85

    def test_closer_match_wins(self):
        """The position should be pulled toward the closer RSSI match."""
        db = FingerprintDB("pull-test")
        db.add(Fingerprint(x=0.0, y=0.0, rssi={"a": -30, "b": -70}))
        db.add(Fingerprint(x=10.0, y=0.0, rssi={"a": -70, "b": -30}))
        est = PositionEstimator(db, k=2)
        # RSSI closer to x=0 reference
        result = est.estimate({"a": -33, "b": -68})
        assert result.x < 5.0  # should be pulled toward x=0

    def test_weighted_distance_mode(self):
        """Weighted distance mode should still produce valid estimates."""
        db = _make_test_db()
        est = PositionEstimator(db, k=3, use_weighted_distance=True)
        result = est.estimate({"ap1": -50, "ap2": -55, "ap3": -65})
        assert result.confidence > 0.0


# ===========================================================================
# FloorPlan Tests
# ===========================================================================

class TestFloorPlan:
    """Tests for the building floor plan model."""

    def _make_plan(self) -> FloorPlan:
        plan = FloorPlan(building_id="hq", name="Headquarters", floor=0)
        plan.add_room(Room(
            room_id="lobby", name="Lobby", room_type=RoomType.LOBBY,
            x_min=0, y_min=0, x_max=10, y_max=5,
        ))
        plan.add_room(Room(
            room_id="office1", name="Office 1", room_type=RoomType.OFFICE,
            x_min=10, y_min=0, x_max=20, y_max=5,
        ))
        plan.add_room(Room(
            room_id="hallway", name="Main Hallway", room_type=RoomType.HALLWAY,
            x_min=0, y_min=5, x_max=20, y_max=7,
        ))
        plan.add_door(Door(door_id="d1", room_a="lobby", room_b="hallway",
                            x=5.0, y=5.0))
        plan.add_door(Door(door_id="d2", room_a="office1", room_b="hallway",
                            x=15.0, y=5.0))
        return plan

    def test_room_count(self):
        plan = self._make_plan()
        assert plan.room_count == 3

    def test_door_count(self):
        plan = self._make_plan()
        assert plan.door_count == 2

    def test_find_room_at(self):
        plan = self._make_plan()
        room = plan.find_room_at(5.0, 2.5)
        assert room is not None
        assert room.room_id == "lobby"

    def test_find_room_at_outside(self):
        plan = self._make_plan()
        room = plan.find_room_at(50.0, 50.0)
        assert room is None

    def test_find_nearest_room(self):
        plan = self._make_plan()
        room = plan.find_nearest_room(25.0, 2.5)
        assert room is not None
        assert room.room_id == "office1"

    def test_room_contains(self):
        room = Room(room_id="r1", name="R1", x_min=0, y_min=0, x_max=5, y_max=5)
        assert room.contains(2.5, 2.5) is True
        assert room.contains(6.0, 2.5) is False

    def test_room_area(self):
        room = Room(room_id="r1", name="R1", x_min=0, y_min=0, x_max=4, y_max=3)
        assert room.area == 12.0

    def test_room_center(self):
        room = Room(room_id="r1", name="R1", x_min=2, y_min=4, x_max=6, y_max=8)
        assert room.center_x == 4.0
        assert room.center_y == 6.0

    def test_room_distance_to_inside(self):
        room = Room(room_id="r1", name="R1", x_min=0, y_min=0, x_max=5, y_max=5)
        assert room.distance_to(2.5, 2.5) == 0.0

    def test_room_distance_to_outside(self):
        room = Room(room_id="r1", name="R1", x_min=0, y_min=0, x_max=5, y_max=5)
        dist = room.distance_to(8.0, 0.0)
        assert abs(dist - 3.0) < 0.01  # 3m east of room

    def test_adjacent_rooms(self):
        plan = self._make_plan()
        adj = plan.get_adjacent_rooms("hallway")
        assert "lobby" in adj
        assert "office1" in adj

    def test_door_connects(self):
        door = Door(door_id="d1", room_a="lobby", room_b="hallway")
        assert door.connects("lobby") is True
        assert door.connects("hallway") is True
        assert door.connects("office") is False

    def test_door_other_room(self):
        door = Door(door_id="d1", room_a="lobby", room_b="hallway")
        assert door.other_room("lobby") == "hallway"
        assert door.other_room("hallway") == "lobby"
        assert door.other_room("office") is None

    def test_remove_room_cascades_doors(self):
        plan = self._make_plan()
        plan.remove_room("lobby")
        assert plan.room_count == 2
        # Door d1 (lobby<->hallway) should be removed
        assert plan.get_door("d1") is None
        # Door d2 (office1<->hallway) should remain
        assert plan.get_door("d2") is not None

    def test_total_area(self):
        plan = self._make_plan()
        expected = (10 * 5) + (10 * 5) + (20 * 2)  # lobby + office1 + hallway
        assert plan.total_area == expected

    def test_to_dict(self):
        plan = self._make_plan()
        d = plan.to_dict()
        assert d["building_id"] == "hq"
        assert d["room_count"] == 3
        assert d["door_count"] == 2
        assert len(d["rooms"]) == 3

    def test_room_type_enum(self):
        assert RoomType.OFFICE.value == "office"
        assert RoomType.SERVER_ROOM.value == "server_room"

    def test_get_doors_for_room(self):
        plan = self._make_plan()
        doors = plan.get_doors_for_room("hallway")
        assert len(doors) == 2

    def test_find_nearest_room_empty(self):
        plan = FloorPlan()
        assert plan.find_nearest_room(0, 0) is None


# ===========================================================================
# ZoneMapper Tests
# ===========================================================================

class TestZoneMapper:
    """Tests for zone mapping and resolution."""

    def _make_mapper(self) -> ZoneMapper:
        mapper = ZoneMapper("test-building")
        mapper.add_zone(Zone(
            zone_id="lobby", name="Lobby",
            x_min=0, y_min=0, x_max=10, y_max=5, floor=0,
        ))
        mapper.add_zone(Zone(
            zone_id="office1", name="Office 1",
            x_min=10, y_min=0, x_max=20, y_max=5, floor=0,
        ))
        mapper.add_zone(Zone(
            zone_id="server", name="Server Room",
            x_min=0, y_min=5, x_max=10, y_max=10, floor=0,
            tags=["restricted"],
        ))
        return mapper

    def test_resolve_inside(self):
        mapper = self._make_mapper()
        zone = mapper.resolve(5.0, 2.5, floor=0)
        assert zone is not None
        assert zone.name == "Lobby"

    def test_resolve_outside(self):
        mapper = self._make_mapper()
        zone = mapper.resolve(50.0, 50.0, floor=0)
        assert zone is None

    def test_resolve_wrong_floor(self):
        mapper = self._make_mapper()
        zone = mapper.resolve(5.0, 2.5, floor=5)
        assert zone is None

    def test_resolve_name(self):
        mapper = self._make_mapper()
        assert mapper.resolve_name(5.0, 2.5) == "Lobby"
        assert mapper.resolve_name(50.0, 50.0) == "Unknown"

    def test_resolve_nearest(self):
        mapper = self._make_mapper()
        # Point outside all zones, closest to Office 1
        zone = mapper.resolve_nearest(22.0, 2.5, floor=0)
        assert zone is not None
        assert zone.name == "Office 1"

    def test_resolve_nearest_max_distance(self):
        mapper = self._make_mapper()
        zone = mapper.resolve_nearest(100.0, 100.0, floor=0, max_distance=5.0)
        assert zone is None

    def test_resolve_overlapping_prefers_smallest(self):
        """When a point is inside overlapping zones, smallest wins."""
        mapper = ZoneMapper("test")
        mapper.add_zone(Zone(zone_id="big", name="Big Zone",
                              x_min=0, y_min=0, x_max=100, y_max=100))
        mapper.add_zone(Zone(zone_id="small", name="Small Zone",
                              x_min=4, y_min=4, x_max=6, y_max=6))
        zone = mapper.resolve(5.0, 5.0)
        assert zone.name == "Small Zone"

    def test_zone_count(self):
        mapper = self._make_mapper()
        assert mapper.zone_count == 3

    def test_add_and_remove_zone(self):
        mapper = ZoneMapper("test")
        zone = Zone(zone_id="z1", name="Zone 1")
        mapper.add_zone(zone)
        assert mapper.zone_count == 1
        assert mapper.remove_zone("z1") is True
        assert mapper.zone_count == 0
        assert mapper.remove_zone("z1") is False

    def test_clear(self):
        mapper = self._make_mapper()
        mapper.clear()
        assert mapper.zone_count == 0

    def test_create_zones_from_floorplan(self):
        plan = FloorPlan(building_id="hq", floor=0)
        plan.add_room(Room(room_id="lobby", name="Lobby",
                            room_type=RoomType.LOBBY,
                            x_min=0, y_min=0, x_max=10, y_max=5))
        plan.add_room(Room(room_id="office", name="Office",
                            room_type=RoomType.OFFICE,
                            x_min=10, y_min=0, x_max=20, y_max=5))
        mapper = ZoneMapper("hq")
        count = mapper.create_zones_from_floorplan(plan)
        assert count == 2
        assert mapper.zone_count == 2
        zone = mapper.get_zone("lobby")
        assert zone is not None
        assert zone.name == "Lobby"
        assert "lobby" in zone.tags

    def test_map_fingerprints(self):
        mapper = self._make_mapper()
        db = FingerprintDB("test")
        db.add(Fingerprint(x=5.0, y=2.5, floor=0,
                            fingerprint_id="fp1", rssi={"a": -50}))
        db.add(Fingerprint(x=50.0, y=50.0, floor=0,
                            fingerprint_id="fp2", rssi={"a": -80}))
        mapping = mapper.map_fingerprints(db)
        assert mapping["fp1"] == "Lobby"
        assert mapping["fp2"] == "Unknown"

    def test_zone_occupancy(self):
        mapper = self._make_mapper()
        positions = [
            (5.0, 2.5, 0),   # Lobby
            (3.0, 1.0, 0),   # Lobby
            (15.0, 2.5, 0),  # Office 1
            (50.0, 50.0, 0), # Unknown
        ]
        occ = mapper.zone_occupancy(positions)
        assert occ["Lobby"] == 2
        assert occ["Office 1"] == 1
        assert occ["Unknown"] == 1

    def test_zone_to_dict(self):
        zone = Zone(zone_id="z1", name="Test Zone",
                     x_min=0, y_min=0, x_max=5, y_max=5, tags=["vip"])
        d = zone.to_dict()
        assert d["zone_id"] == "z1"
        assert d["area"] == 25.0
        assert "vip" in d["tags"]

    def test_zone_contains(self):
        zone = Zone(zone_id="z1", name="Z",
                     x_min=0, y_min=0, x_max=10, y_max=10)
        assert zone.contains(5, 5) is True
        assert zone.contains(15, 5) is False

    def test_zone_distance_to(self):
        zone = Zone(zone_id="z1", name="Z",
                     x_min=0, y_min=0, x_max=10, y_max=10)
        assert zone.distance_to(5, 5) == 0.0
        assert abs(zone.distance_to(13, 0) - 3.0) < 0.01

    def test_mapper_to_dict(self):
        mapper = self._make_mapper()
        d = mapper.to_dict()
        assert d["building_id"] == "test-building"
        assert d["zone_count"] == 3
        assert len(d["zones"]) == 3


# ===========================================================================
# Integration: Estimator + ZoneMapper end-to-end
# ===========================================================================

class TestIntegration:
    """End-to-end tests: fingerprint -> estimate -> zone resolution."""

    def test_estimate_to_zone(self):
        """Full pipeline: estimate position then resolve to zone."""
        db = FingerprintDB("hq")
        # Lobby fingerprints
        for i in range(5):
            db.add(Fingerprint(
                x=float(i * 2), y=2.0, floor=0,
                rssi={"ap_lobby": -30 - i * 5, "ap_office": -70 + i * 3},
            ))
        # Office fingerprints
        for i in range(5):
            db.add(Fingerprint(
                x=10.0 + float(i * 2), y=2.0, floor=0,
                rssi={"ap_lobby": -70 - i * 3, "ap_office": -30 - i * 5},
            ))

        mapper = ZoneMapper("hq")
        mapper.add_zone(Zone(zone_id="lobby", name="Lobby",
                              x_min=0, y_min=0, x_max=10, y_max=5))
        mapper.add_zone(Zone(zone_id="office", name="Office",
                              x_min=10, y_min=0, x_max=20, y_max=5))

        est = PositionEstimator(db, k=3)
        # Scan near the lobby
        result = est.estimate({"ap_lobby": -35, "ap_office": -65})
        zone_name = mapper.resolve_name(result.x, result.y, result.floor)
        assert zone_name == "Lobby"

    def test_multi_floor_pipeline(self):
        """Estimate floor then position on that floor."""
        db = FingerprintDB("building")
        # Floor 0
        db.add(Fingerprint(x=5, y=5, floor=0,
                            rssi={"ap_ground": -30, "ap_upper": -80}))
        # Floor 1
        db.add(Fingerprint(x=5, y=5, floor=1,
                            rssi={"ap_ground": -80, "ap_upper": -30}))

        est = PositionEstimator(db, k=2, min_common_aps=1)
        # Scan on upper floor
        floor = est.estimate_floor({"ap_ground": -78, "ap_upper": -33})
        assert floor == 1

        result = est.estimate({"ap_ground": -78, "ap_upper": -33}, floor=1)
        assert result.floor == 1
        assert result.confidence > 0.0
