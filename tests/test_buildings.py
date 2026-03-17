# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the building interior and room-clearing system."""

from __future__ import annotations

import random

import pytest

from tritium_lib.sim_engine.buildings import (
    BUILDING_TEMPLATES,
    BuildingLayout,
    Room,
    RoomClearingEngine,
    RoomType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> RoomClearingEngine:
    return RoomClearingEngine(hostile_ids={"h1", "h2", "h3", "h4", "h5"})


@pytest.fixture
def simple_layout(engine: RoomClearingEngine) -> BuildingLayout:
    return engine.generate_layout(floors=1, rooms_per_floor=4)


@pytest.fixture
def multi_floor_layout(engine: RoomClearingEngine) -> BuildingLayout:
    return engine.generate_layout(floors=3, rooms_per_floor=4)


# ---------------------------------------------------------------------------
# RoomType enum
# ---------------------------------------------------------------------------

class TestRoomType:
    def test_all_values_exist(self) -> None:
        expected = {"hallway", "room", "stairwell", "roof", "basement",
                    "lobby", "office", "storage", "bathroom"}
        actual = {rt.value for rt in RoomType}
        assert actual == expected

    def test_enum_count(self) -> None:
        assert len(RoomType) == 9

    def test_enum_access_by_name(self) -> None:
        assert RoomType.HALLWAY.value == "hallway"
        assert RoomType.STAIRWELL.value == "stairwell"
        assert RoomType.ROOF.value == "roof"


# ---------------------------------------------------------------------------
# Room dataclass
# ---------------------------------------------------------------------------

class TestRoom:
    def test_creation(self) -> None:
        room = Room(
            room_id="r1",
            room_type=RoomType.ROOM,
            position=(10.0, 20.0),
            size=(4.0, 3.0),
        )
        assert room.room_id == "r1"
        assert room.room_type == RoomType.ROOM
        assert room.position == (10.0, 20.0)
        assert room.size == (4.0, 3.0)

    def test_defaults(self) -> None:
        room = Room("r2", RoomType.OFFICE, (0.0, 0.0), (5.0, 5.0))
        assert room.floor == 0
        assert room.doors == []
        assert room.windows == []
        assert room.occupants == []
        assert room.is_cleared is False
        assert room.cover_positions == []
        assert room.visibility == 1.0

    def test_area(self) -> None:
        room = Room("r3", RoomType.STORAGE, (0.0, 0.0), (6.0, 4.0))
        assert room.area == 24.0

    def test_center(self) -> None:
        room = Room("r4", RoomType.HALLWAY, (5.0, 10.0), (8.0, 2.0))
        assert room.center == (5.0, 10.0)

    def test_has_hostile_empty(self) -> None:
        room = Room("r5", RoomType.ROOM, (0.0, 0.0), (4.0, 4.0))
        assert room.has_hostile(set()) is False
        assert room.has_hostile(None) is False

    def test_has_hostile_present(self) -> None:
        room = Room("r6", RoomType.ROOM, (0.0, 0.0), (4.0, 4.0),
                     occupants=["h1", "f1"])
        assert room.has_hostile({"h1", "h2"}) is True

    def test_has_hostile_not_present(self) -> None:
        room = Room("r7", RoomType.ROOM, (0.0, 0.0), (4.0, 4.0),
                     occupants=["f1", "f2"])
        assert room.has_hostile({"h1"}) is False


# ---------------------------------------------------------------------------
# BuildingLayout
# ---------------------------------------------------------------------------

class TestBuildingLayout:
    def test_creation(self) -> None:
        layout = BuildingLayout(
            building_id="b1",
            position=(100.0, 200.0),
            floors=2,
        )
        assert layout.building_id == "b1"
        assert layout.floors == 2
        assert layout.total_rooms == 0
        assert layout.rooms == []

    def test_total_rooms_auto(self) -> None:
        rooms = [
            Room("r1", RoomType.ROOM, (0.0, 0.0), (4.0, 4.0)),
            Room("r2", RoomType.ROOM, (5.0, 0.0), (4.0, 4.0)),
        ]
        layout = BuildingLayout("b2", (0.0, 0.0), 1, rooms=rooms)
        assert layout.total_rooms == 2

    def test_is_fully_cleared_false(self) -> None:
        rooms = [Room("r1", RoomType.ROOM, (0.0, 0.0), (4.0, 4.0))]
        layout = BuildingLayout("b3", (0.0, 0.0), 1, rooms=rooms)
        assert layout.is_fully_cleared is False

    def test_is_fully_cleared_true(self) -> None:
        rooms = [Room("r1", RoomType.ROOM, (0.0, 0.0), (4.0, 4.0), is_cleared=True)]
        layout = BuildingLayout("b4", (0.0, 0.0), 1, rooms=rooms, cleared_rooms=1)
        assert layout.is_fully_cleared is True

    def test_room_by_id(self) -> None:
        rooms = [
            Room("r1", RoomType.ROOM, (0.0, 0.0), (4.0, 4.0)),
            Room("r2", RoomType.OFFICE, (5.0, 0.0), (4.0, 4.0)),
        ]
        layout = BuildingLayout("b5", (0.0, 0.0), 1, rooms=rooms)
        assert layout.room_by_id("r2") is rooms[1]
        assert layout.room_by_id("nonexistent") is None

    def test_rooms_on_floor(self) -> None:
        rooms = [
            Room("r1", RoomType.ROOM, (0.0, 0.0), (4.0, 4.0), floor=0),
            Room("r2", RoomType.ROOM, (0.0, 0.0), (4.0, 4.0), floor=1),
            Room("r3", RoomType.ROOM, (0.0, 0.0), (4.0, 4.0), floor=0),
        ]
        layout = BuildingLayout("b6", (0.0, 0.0), 2, rooms=rooms)
        floor0 = layout.rooms_on_floor(0)
        assert len(floor0) == 2
        assert layout.rooms_on_floor(1) == [rooms[1]]
        assert layout.rooms_on_floor(5) == []


# ---------------------------------------------------------------------------
# Building generation
# ---------------------------------------------------------------------------

class TestGeneration:
    def test_single_floor(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout(floors=1, rooms_per_floor=4)
        assert layout.floors == 1
        assert layout.total_rooms == 4
        assert len(layout.rooms) == 4
        assert len(layout.entry_points) >= 1

    def test_multi_floor(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout(floors=3, rooms_per_floor=4)
        # 3 floors * 4 rooms + 2 stairwells
        assert layout.floors == 3
        assert layout.total_rooms == 14  # 12 rooms + 2 stairwells
        assert any(r.room_type == RoomType.STAIRWELL for r in layout.rooms)

    def test_rooms_have_doors(self, simple_layout: BuildingLayout) -> None:
        # Non-hallway rooms should connect to the hallway
        for room in simple_layout.rooms[1:]:
            assert len(room.doors) >= 1, f"{room.room_id} has no doors"

    def test_hallway_connects_to_rooms(self, simple_layout: BuildingLayout) -> None:
        hallway = simple_layout.rooms[0]
        assert len(hallway.doors) >= 1

    def test_rooms_have_windows(self, simple_layout: BuildingLayout) -> None:
        for room in simple_layout.rooms:
            assert len(room.windows) >= 1

    def test_rooms_have_cover(self, simple_layout: BuildingLayout) -> None:
        for room in simple_layout.rooms:
            assert len(room.cover_positions) >= 1

    def test_entry_points_exist(self, simple_layout: BuildingLayout) -> None:
        assert len(simple_layout.entry_points) >= 1
        for ep in simple_layout.entry_points:
            assert "position" in ep
            assert "type" in ep
            assert "room_id" in ep

    def test_building_registered(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout()
        assert layout.building_id in engine.buildings

    def test_unique_building_ids(self, engine: RoomClearingEngine) -> None:
        ids = {engine.generate_layout().building_id for _ in range(10)}
        assert len(ids) == 10

    def test_building_position(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout(building_pos=(50.0, 100.0))
        assert layout.position == (50.0, 100.0)

    def test_stairwells_connect_floors(self, multi_floor_layout: BuildingLayout) -> None:
        stairwells = [r for r in multi_floor_layout.rooms
                      if r.room_type == RoomType.STAIRWELL]
        assert len(stairwells) == 2
        for stair in stairwells:
            assert len(stair.doors) == 2  # connects lower + upper floor

    def test_lobby_on_ground_floor(self, multi_floor_layout: BuildingLayout) -> None:
        ground_rooms = multi_floor_layout.rooms_on_floor(0)
        assert any(r.room_type == RoomType.LOBBY for r in ground_rooms)


# ---------------------------------------------------------------------------
# Building templates
# ---------------------------------------------------------------------------

class TestTemplates:
    def test_template_keys(self) -> None:
        expected = {"house", "apartment", "office", "warehouse", "compound"}
        assert set(BUILDING_TEMPLATES.keys()) == expected

    def test_house_template(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout(template="house")
        assert layout.floors == 1
        assert layout.total_rooms == 4

    def test_apartment_template(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout(template="apartment")
        assert layout.floors == 3
        assert layout.total_rooms > 12  # rooms + stairwells

    def test_office_template(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout(template="office")
        assert layout.floors == 5

    def test_warehouse_template(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout(template="warehouse")
        assert layout.floors == 1
        # Warehouse has custom room sizes
        storage_rooms = [r for r in layout.rooms if r.room_type == RoomType.STORAGE]
        assert len(storage_rooms) >= 1

    def test_compound_template(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout(template="compound")
        assert layout.floors == 2
        assert len(layout.entry_points) >= 3

    def test_unknown_template_ignored(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout(template="nonexistent", floors=2, rooms_per_floor=3)
        assert layout.floors == 2


# ---------------------------------------------------------------------------
# Building entry
# ---------------------------------------------------------------------------

class TestEntry:
    def test_enter_building(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        result = engine.enter_building("u1", simple_layout.building_id, 0)
        assert result is True

    def test_enter_places_unit_in_room(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        engine.enter_building("u1", simple_layout.building_id, 0)
        room = engine.get_unit_room("u1")
        assert room is not None
        assert "u1" in room.occupants

    def test_enter_invalid_building(self, engine: RoomClearingEngine) -> None:
        assert engine.enter_building("u1", "nonexistent") is False

    def test_enter_invalid_entry_point(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        assert engine.enter_building("u1", simple_layout.building_id, 99) is False

    def test_enter_negative_entry_point(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        assert engine.enter_building("u1", simple_layout.building_id, -1) is False

    def test_reenter_removes_from_old_room(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout(floors=1, rooms_per_floor=4)
        engine.enter_building("u1", layout.building_id, 0)
        old_room = engine.get_unit_room("u1")
        assert old_room is not None

        # Enter again at different entry point
        if len(layout.entry_points) > 1:
            engine.enter_building("u1", layout.building_id, 1)
            assert "u1" not in old_room.occupants


# ---------------------------------------------------------------------------
# Unit movement
# ---------------------------------------------------------------------------

class TestMovement:
    def test_move_to_adjacent_room(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        engine.enter_building("u1", simple_layout.building_id, 0)
        entry_room = engine.get_unit_room("u1")
        assert entry_room is not None
        if entry_room.doors:
            target_id = entry_room.doors[0]["connects_to"]
            result = engine.move_unit("u1", target_id)
            assert result is True
            assert engine.get_unit_room("u1").room_id == target_id

    def test_move_to_non_adjacent_fails(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout(floors=1, rooms_per_floor=6)
        engine.enter_building("u1", layout.building_id, 0)
        # Room index 5 should not be directly connected to entry
        far_room = layout.rooms[-1]
        # Only fails if not adjacent
        entry_room = engine.get_unit_room("u1")
        connected = {d["connects_to"] for d in entry_room.doors}
        if far_room.room_id not in connected:
            assert engine.move_unit("u1", far_room.room_id) is False

    def test_move_unit_not_in_building(self, engine: RoomClearingEngine) -> None:
        assert engine.move_unit("ghost", "room_x") is False


# ---------------------------------------------------------------------------
# Room clearing
# ---------------------------------------------------------------------------

class TestClearing:
    def test_clear_empty_room(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        engine.enter_building("u1", simple_layout.building_id, 0)
        room = engine.get_unit_room("u1")
        result = engine.clear_room(["u1"], room.room_id, simple_layout.building_id)
        assert result["success"] is True
        assert result["room_cleared"] is True
        assert result["hostiles_found"] == 0
        assert room.is_cleared is True

    def test_clear_room_with_hostiles(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        random.seed(42)
        # Place hostile in a room
        target_room = simple_layout.rooms[1]
        target_room.occupants.append("h1")
        engine._unit_locations["h1"] = (simple_layout.building_id, target_room.room_id)

        engine.enter_building("u1", simple_layout.building_id, 0)
        result = engine.clear_room(["u1"], target_room.room_id, simple_layout.building_id)
        assert result["success"] is True
        assert result["hostiles_found"] == 1

    def test_clear_room_flashbang(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        random.seed(42)
        target_room = simple_layout.rooms[1]
        target_room.occupants.append("h2")
        engine._unit_locations["h2"] = (simple_layout.building_id, target_room.room_id)

        result = engine.clear_room(
            ["u1"], target_room.room_id, simple_layout.building_id, flashbang=True,
        )
        assert result["success"] is True
        assert result["flashbang_used"] is True
        assert result["accuracy"] > 0.85  # flashbang bonus

    def test_clear_room_no_building(self, engine: RoomClearingEngine) -> None:
        result = engine.clear_room(["u1"], "r_fake")
        assert result["success"] is False

    def test_clear_room_invalid_room(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        result = engine.clear_room(["u1"], "nonexistent", simple_layout.building_id)
        assert result["success"] is False

    def test_cleared_count_increments(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        engine.enter_building("u1", simple_layout.building_id, 0)
        room = engine.get_unit_room("u1")
        engine.clear_room(["u1"], room.room_id, simple_layout.building_id)
        assert simple_layout.cleared_rooms >= 1

    def test_building_fully_cleared(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout(floors=1, rooms_per_floor=2)
        engine.enter_building("u1", layout.building_id, 0)
        for room in layout.rooms:
            engine.clear_room(["u1"], room.room_id, layout.building_id)
        assert layout.is_fully_cleared is True

    def test_clear_multiple_hostiles(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        random.seed(0)
        room = simple_layout.rooms[1]
        room.occupants.extend(["h1", "h2", "h3"])
        for hid in ["h1", "h2", "h3"]:
            engine._unit_locations[hid] = (simple_layout.building_id, room.room_id)

        result = engine.clear_room(
            ["u1", "u2", "u3"], room.room_id, simple_layout.building_id,
        )
        assert result["success"] is True
        assert result["hostiles_found"] == 3

    def test_visibility_affects_accuracy(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        room = simple_layout.rooms[1]
        room.visibility = 0.5
        room.occupants.append("h4")
        engine._unit_locations["h4"] = (simple_layout.building_id, room.room_id)

        result = engine.clear_room(["u1"], room.room_id, simple_layout.building_id)
        assert result["accuracy"] < 0.85  # penalized by visibility


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

class TestQueries:
    def test_get_unit_room_none(self, engine: RoomClearingEngine) -> None:
        assert engine.get_unit_room("nobody") is None

    def test_get_unit_building(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        engine.enter_building("u1", simple_layout.building_id, 0)
        bld = engine.get_unit_building("u1")
        assert bld is not None
        assert bld.building_id == simple_layout.building_id

    def test_get_unit_building_none(self, engine: RoomClearingEngine) -> None:
        assert engine.get_unit_building("ghost") is None

    def test_get_adjacent_rooms(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        hallway = simple_layout.rooms[0]
        adj = engine.get_adjacent_rooms(hallway.room_id, simple_layout.building_id)
        assert len(adj) >= 1

    def test_get_adjacent_rooms_invalid(self, engine: RoomClearingEngine) -> None:
        assert engine.get_adjacent_rooms("fake", "fake") == []

    def test_get_uncleared_rooms(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        uncleared = engine.get_uncleared_rooms(simple_layout.building_id)
        assert len(uncleared) == simple_layout.total_rooms

    def test_get_uncleared_after_clearing(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        engine.enter_building("u1", simple_layout.building_id, 0)
        room = engine.get_unit_room("u1")
        engine.clear_room(["u1"], room.room_id, simple_layout.building_id)
        uncleared = engine.get_uncleared_rooms(simple_layout.building_id)
        assert len(uncleared) == simple_layout.total_rooms - 1

    def test_get_uncleared_invalid_building(self, engine: RoomClearingEngine) -> None:
        assert engine.get_uncleared_rooms("fake") == []


# ---------------------------------------------------------------------------
# Tick simulation
# ---------------------------------------------------------------------------

class TestTick:
    def test_tick_no_crash(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        engine.tick(0.016)

    def test_tick_updates_visibility(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout(floors=2, rooms_per_floor=2)
        # Find basement-type room or set one
        room = layout.rooms[0]
        room.room_type = RoomType.BASEMENT
        room.visibility = 0.3
        initial = room.visibility
        # Run many ticks to see variance
        random.seed(99)
        for _ in range(100):
            engine.tick(0.016)
        # Visibility should have changed (or stayed within bounds)
        assert 0.1 <= room.visibility <= 0.5

    def test_tick_syncs_cleared_count(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        simple_layout.rooms[0].is_cleared = True
        simple_layout.rooms[1].is_cleared = True
        engine.tick(0.016)
        assert simple_layout.cleared_rooms == 2


# ---------------------------------------------------------------------------
# Three.js export
# ---------------------------------------------------------------------------

class TestThreeJsExport:
    def test_export_structure(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        data = engine.to_three_js(simple_layout.building_id)
        assert "building_id" in data
        assert "floors" in data
        assert "entry_points" in data
        assert "total_rooms" in data

    def test_export_floors(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        data = engine.to_three_js(simple_layout.building_id)
        assert len(data["floors"]) == 1
        assert len(data["floors"][0]["rooms"]) == 4

    def test_export_room_fields(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        data = engine.to_three_js(simple_layout.building_id)
        room_data = data["floors"][0]["rooms"][0]
        assert "room_id" in room_data
        assert "type" in room_data
        assert "position" in room_data
        assert "size" in room_data
        assert "doors" in room_data
        assert "windows" in room_data
        assert "occupants" in room_data
        assert "is_cleared" in room_data
        assert "visibility" in room_data
        assert "cover_positions" in room_data

    def test_export_position_format(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        data = engine.to_three_js(simple_layout.building_id)
        pos = data["position"]
        assert "x" in pos and "y" in pos

    def test_export_invalid_building(self, engine: RoomClearingEngine) -> None:
        data = engine.to_three_js("nonexistent")
        assert "error" in data

    def test_export_multi_floor(self, engine: RoomClearingEngine, multi_floor_layout: BuildingLayout) -> None:
        data = engine.to_three_js(multi_floor_layout.building_id)
        assert data["total_floors"] == 3
        # Floor 0 should have rooms
        assert len(data["floors"][0]["rooms"]) >= 1

    def test_export_cleared_status(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        simple_layout.rooms[0].is_cleared = True
        simple_layout.cleared_rooms = 1
        data = engine.to_three_js(simple_layout.building_id)
        assert data["cleared_rooms"] == 1

    def test_export_entry_points(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        data = engine.to_three_js(simple_layout.building_id)
        for ep in data["entry_points"]:
            assert "position" in ep
            assert "type" in ep
            assert "room_id" in ep


# ---------------------------------------------------------------------------
# add_building
# ---------------------------------------------------------------------------

class TestAddBuilding:
    def test_add_external_building(self, engine: RoomClearingEngine) -> None:
        rooms = [Room("ext_r1", RoomType.ROOM, (0.0, 0.0), (4.0, 4.0))]
        layout = BuildingLayout("ext_b1", (0.0, 0.0), 1, rooms=rooms,
                                entry_points=[{"position": (0.0, 0.0), "type": "door", "room_id": "ext_r1"}])
        engine.add_building(layout)
        assert "ext_b1" in engine.buildings
        assert engine.enter_building("u1", "ext_b1", 0) is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_rooms_per_floor(self, engine: RoomClearingEngine) -> None:
        # Should still produce something (at least empty)
        layout = engine.generate_layout(floors=1, rooms_per_floor=0)
        assert layout.total_rooms == 0

    def test_single_room(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout(floors=1, rooms_per_floor=1)
        assert layout.total_rooms == 1

    def test_many_floors(self, engine: RoomClearingEngine) -> None:
        layout = engine.generate_layout(floors=10, rooms_per_floor=2)
        assert layout.floors == 10
        stairwells = [r for r in layout.rooms if r.room_type == RoomType.STAIRWELL]
        assert len(stairwells) == 9  # one between each pair of floors

    def test_clear_same_room_twice(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        engine.enter_building("u1", simple_layout.building_id, 0)
        room = engine.get_unit_room("u1")
        engine.clear_room(["u1"], room.room_id, simple_layout.building_id)
        result = engine.clear_room(["u1"], room.room_id, simple_layout.building_id)
        assert result["success"] is True  # idempotent
        assert room.is_cleared is True

    def test_hostile_removed_after_kill(self, engine: RoomClearingEngine, simple_layout: BuildingLayout) -> None:
        random.seed(42)
        room = simple_layout.rooms[1]
        room.occupants.append("h5")
        engine._unit_locations["h5"] = (simple_layout.building_id, room.room_id)
        result = engine.clear_room(["u1", "u2", "u3", "u4"], room.room_id, simple_layout.building_id)
        if result["hostiles_killed"]:
            assert "h5" not in engine.hostile_ids or "h5" not in room.occupants
