# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Standalone unit tests for buildings.py — CQB room clearing engine.

Tests RoomClearingEngine layout generation, room clearing mechanics,
unit placement, adjacency queries, and to_three_js serialization.
"""

from __future__ import annotations

import random

import pytest

from tritium_lib.sim_engine.buildings import (
    RoomClearingEngine,
    BuildingLayout,
    Room,
    RoomType,
)


# ---------------------------------------------------------------------------
# Layout generation
# ---------------------------------------------------------------------------

class TestLayoutGeneration:
    def test_generate_single_floor(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=4,
            building_pos=(100.0, 100.0), template="house",
        )
        assert isinstance(layout, BuildingLayout)
        assert layout.floors == 1
        assert len(layout.rooms) == 4
        assert layout.total_rooms == 4

    def test_generate_multi_floor(self) -> None:
        eng = RoomClearingEngine()
        # "office" template overrides floors to 5 and rooms_per_floor to 4
        layout = eng.generate_layout(
            floors=3, rooms_per_floor=5,
            building_pos=(0.0, 0.0), template="office",
        )
        assert layout.floors == 5  # template overrides caller
        assert len(layout.rooms) >= 20  # 5 floors * 4 rooms + stairwells

    def test_building_registered(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=3,
            building_pos=(0.0, 0.0), template="house",
        )
        assert layout.building_id in eng.buildings

    def test_rooms_have_valid_types(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=6,
            building_pos=(0.0, 0.0), template="house",
        )
        for room in layout.rooms:
            assert isinstance(room.room_type, RoomType)

    def test_rooms_have_positions(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=4,
            building_pos=(50.0, 50.0), template="house",
        )
        for room in layout.rooms:
            assert isinstance(room.position, tuple)
            assert len(room.position) == 2

    def test_rooms_have_doors(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=4,
            building_pos=(0.0, 0.0), template="house",
        )
        # At least some rooms should have doors
        rooms_with_doors = [r for r in layout.rooms if len(r.doors) > 0]
        assert len(rooms_with_doors) > 0

    def test_entry_points_exist(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=4,
            building_pos=(0.0, 0.0), template="house",
        )
        assert len(layout.entry_points) >= 1

    def test_building_starts_not_cleared(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=4,
            building_pos=(0.0, 0.0), template="house",
        )
        assert layout.is_fully_cleared is False
        assert layout.cleared_rooms == 0


# ---------------------------------------------------------------------------
# Room dataclass
# ---------------------------------------------------------------------------

class TestRoom:
    def test_area(self) -> None:
        room = Room(
            room_id="r1", room_type=RoomType.ROOM,
            position=(0.0, 0.0), size=(4.0, 5.0),
        )
        assert room.area == 20.0

    def test_center(self) -> None:
        room = Room(
            room_id="r1", room_type=RoomType.ROOM,
            position=(10.0, 20.0), size=(4.0, 5.0),
        )
        assert room.center == (10.0, 20.0)

    def test_has_hostile_true(self) -> None:
        room = Room(
            room_id="r1", room_type=RoomType.ROOM,
            position=(0.0, 0.0), size=(4.0, 4.0),
            occupants=["h1", "f1"],
        )
        assert room.has_hostile({"h1", "h2"}) is True

    def test_has_hostile_false(self) -> None:
        room = Room(
            room_id="r1", room_type=RoomType.ROOM,
            position=(0.0, 0.0), size=(4.0, 4.0),
            occupants=["f1"],
        )
        assert room.has_hostile({"h1"}) is False

    def test_has_hostile_none_ids(self) -> None:
        room = Room(
            room_id="r1", room_type=RoomType.ROOM,
            position=(0.0, 0.0), size=(4.0, 4.0),
            occupants=["h1"],
        )
        assert room.has_hostile(None) is False


# ---------------------------------------------------------------------------
# BuildingLayout
# ---------------------------------------------------------------------------

class TestBuildingLayout:
    def test_room_by_id(self) -> None:
        rooms = [
            Room(room_id="r1", room_type=RoomType.ROOM, position=(0.0, 0.0), size=(4.0, 4.0)),
            Room(room_id="r2", room_type=RoomType.HALLWAY, position=(5.0, 0.0), size=(2.0, 8.0)),
        ]
        layout = BuildingLayout(
            building_id="b1", position=(0.0, 0.0), floors=1, rooms=rooms,
        )
        assert layout.room_by_id("r1") is rooms[0]
        assert layout.room_by_id("nonexistent") is None

    def test_is_fully_cleared(self) -> None:
        rooms = [
            Room(room_id="r1", room_type=RoomType.ROOM, position=(0.0, 0.0),
                 size=(4.0, 4.0), is_cleared=True),
            Room(room_id="r2", room_type=RoomType.ROOM, position=(5.0, 0.0),
                 size=(4.0, 4.0), is_cleared=True),
        ]
        layout = BuildingLayout(
            building_id="b1", position=(0.0, 0.0), floors=1, rooms=rooms,
        )
        # cleared_rooms counter must be set manually (tick or clear_room updates it)
        layout.cleared_rooms = sum(1 for r in rooms if r.is_cleared)
        assert layout.is_fully_cleared is True

    def test_not_fully_cleared(self) -> None:
        rooms = [
            Room(room_id="r1", room_type=RoomType.ROOM, position=(0.0, 0.0),
                 size=(4.0, 4.0), is_cleared=True),
            Room(room_id="r2", room_type=RoomType.ROOM, position=(5.0, 0.0),
                 size=(4.0, 4.0), is_cleared=False),
        ]
        layout = BuildingLayout(
            building_id="b1", position=(0.0, 0.0), floors=1, rooms=rooms,
        )
        assert layout.is_fully_cleared is False

    def test_post_init_sets_total_rooms(self) -> None:
        rooms = [
            Room(room_id=f"r{i}", room_type=RoomType.ROOM,
                 position=(float(i), 0.0), size=(4.0, 4.0))
            for i in range(5)
        ]
        layout = BuildingLayout(
            building_id="b1", position=(0.0, 0.0), floors=1, rooms=rooms,
        )
        assert layout.total_rooms == 5


# ---------------------------------------------------------------------------
# Room clearing CQB
# ---------------------------------------------------------------------------

class TestClearRoom:
    def _setup_engine_with_hostiles(self) -> tuple[RoomClearingEngine, str, str]:
        """Create engine with one building, hostiles in room r1."""
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=4,
            building_pos=(0.0, 0.0), template="house",
        )
        bid = layout.building_id
        room = layout.rooms[0]
        rid = room.room_id

        # Place hostiles
        eng.hostile_ids.add("h1")
        eng.hostile_ids.add("h2")
        room.occupants.extend(["h1", "h2"])
        eng._unit_locations["h1"] = (bid, rid)
        eng._unit_locations["h2"] = (bid, rid)

        return eng, bid, rid

    def test_clear_room_success(self) -> None:
        eng, bid, rid = self._setup_engine_with_hostiles()
        result = eng.clear_room(["f1", "f2", "f3"], rid, bid)
        assert result["success"] is True
        assert result["room_id"] == rid
        assert result["building_id"] == bid
        assert result["hostiles_found"] == 2
        assert result["room_cleared"] is True
        assert "accuracy" in result

    def test_clear_room_with_flashbang(self) -> None:
        eng, bid, rid = self._setup_engine_with_hostiles()
        result = eng.clear_room(["f1", "f2", "f3"], rid, bid, flashbang=True)
        assert result["success"] is True
        assert result["flashbang_used"] is True
        # Flashbang gives +0.25 accuracy bonus, capped at 1.0
        assert result["accuracy"] >= 0.85

    def test_clear_room_no_building(self) -> None:
        eng = RoomClearingEngine()
        result = eng.clear_room(["f1"], "fake_room")
        assert result["success"] is False
        assert result["error"] == "no_building"

    def test_clear_room_building_not_found(self) -> None:
        eng = RoomClearingEngine()
        eng.generate_layout(
            floors=1, rooms_per_floor=2,
            building_pos=(0.0, 0.0), template="house",
        )
        # Place a unit so building_id can be inferred, but use wrong building_id
        result = eng.clear_room(["f1"], "fake_room", building_id="nonexistent")
        assert result["success"] is False
        assert result["error"] == "building_not_found"

    def test_clear_empty_room(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=2,
            building_pos=(0.0, 0.0), template="house",
        )
        bid = layout.building_id
        rid = layout.rooms[0].room_id
        result = eng.clear_room(["f1", "f2"], rid, bid)
        assert result["success"] is True
        assert result["hostiles_found"] == 0
        assert result["room_cleared"] is True

    def test_clearing_updates_building_cleared_count(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=3,
            building_pos=(0.0, 0.0), template="house",
        )
        bid = layout.building_id
        for room in layout.rooms:
            eng.clear_room(["f1", "f2"], room.room_id, bid)
        assert layout.is_fully_cleared is True
        assert layout.cleared_rooms == layout.total_rooms


# ---------------------------------------------------------------------------
# Unit location queries
# ---------------------------------------------------------------------------

class TestUnitQueries:
    def test_get_unit_room(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=2,
            building_pos=(0.0, 0.0), template="house",
        )
        bid = layout.building_id
        rid = layout.rooms[0].room_id
        layout.rooms[0].occupants.append("u1")
        eng._unit_locations["u1"] = (bid, rid)
        room = eng.get_unit_room("u1")
        assert room is not None
        assert room.room_id == rid

    def test_get_unit_room_not_found(self) -> None:
        eng = RoomClearingEngine()
        assert eng.get_unit_room("nonexistent") is None

    def test_get_unit_building(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=2,
            building_pos=(0.0, 0.0), template="house",
        )
        bid = layout.building_id
        rid = layout.rooms[0].room_id
        eng._unit_locations["u1"] = (bid, rid)
        bld = eng.get_unit_building("u1")
        assert bld is not None
        assert bld.building_id == bid

    def test_get_adjacent_rooms(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=4,
            building_pos=(0.0, 0.0), template="house",
        )
        bid = layout.building_id
        # Find a room with doors
        room_with_doors = None
        for r in layout.rooms:
            if r.doors:
                room_with_doors = r
                break
        if room_with_doors:
            adjacent = eng.get_adjacent_rooms(room_with_doors.room_id, bid)
            assert isinstance(adjacent, list)

    def test_get_uncleared_rooms(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=4,
            building_pos=(0.0, 0.0), template="house",
        )
        bid = layout.building_id
        uncleared = eng.get_uncleared_rooms(bid)
        assert len(uncleared) == 4  # all uncleared initially

    def test_get_uncleared_rooms_after_clearing(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=4,
            building_pos=(0.0, 0.0), template="house",
        )
        bid = layout.building_id
        # Clear one room
        eng.clear_room(["f1"], layout.rooms[0].room_id, bid)
        uncleared = eng.get_uncleared_rooms(bid)
        assert len(uncleared) == 3


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------

class TestBuildingsTick:
    def test_tick_no_crash(self) -> None:
        eng = RoomClearingEngine()
        eng.generate_layout(
            floors=2, rooms_per_floor=4,
            building_pos=(0.0, 0.0), template="office",
        )
        # Should not raise
        eng.tick(0.1)
        eng.tick(1.0)

    def test_tick_updates_cleared_count(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=3,
            building_pos=(0.0, 0.0), template="house",
        )
        layout.rooms[0].is_cleared = True
        eng.tick(0.1)
        assert layout.cleared_rooms == 1


# ---------------------------------------------------------------------------
# to_three_js
# ---------------------------------------------------------------------------

class TestBuildingsToThreeJs:
    def test_serialization(self) -> None:
        eng = RoomClearingEngine()
        layout = eng.generate_layout(
            floors=1, rooms_per_floor=4,
            building_pos=(50.0, 50.0), template="house",
        )
        data = eng.to_three_js(layout.building_id)
        assert "building_id" in data
        assert "floors" in data  # list of floor dicts
        assert "total_floors" in data
        assert "total_rooms" in data
        assert isinstance(data["floors"], list)
        # Each floor has a "rooms" key
        total_rooms = sum(len(f["rooms"]) for f in data["floors"])
        assert total_rooms == 4

    def test_nonexistent_building(self) -> None:
        eng = RoomClearingEngine()
        data = eng.to_three_js("nonexistent")
        # Should return an empty/error dict, not crash
        assert isinstance(data, dict)
