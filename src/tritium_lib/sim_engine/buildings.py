# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Building interior and room-clearing system for urban combat.

Procedural building generation, room-by-room CQB clearing, occupant
tracking, and Three.js-ready floor-plan export.  Each building is a
:class:`BuildingLayout` composed of :class:`Room` instances connected
by doors.  :class:`RoomClearingEngine` orchestrates entry, clearing,
and per-tick simulation updates.

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import enum
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RoomType(enum.Enum):
    """Categories of rooms inside a building."""

    HALLWAY = "hallway"
    ROOM = "room"
    STAIRWELL = "stairwell"
    ROOF = "roof"
    BASEMENT = "basement"
    LOBBY = "lobby"
    OFFICE = "office"
    STORAGE = "storage"
    BATHROOM = "bathroom"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Room:
    """A single room inside a building."""

    room_id: str
    room_type: RoomType
    position: Vec2  # center (x, y) in local building coords
    size: tuple[float, float]  # (width, height) in meters
    floor: int = 0
    doors: list[dict] = field(default_factory=list)
    # Each door: {"position": Vec2, "connects_to": room_id}
    windows: list[dict] = field(default_factory=list)
    # Each window: {"position": Vec2, "facing": float}  (radians)
    occupants: list[str] = field(default_factory=list)  # unit IDs
    is_cleared: bool = False
    cover_positions: list[Vec2] = field(default_factory=list)
    visibility: float = 1.0  # 0 = dark, 1 = fully lit

    @property
    def area(self) -> float:
        return self.size[0] * self.size[1]

    @property
    def center(self) -> Vec2:
        return self.position

    def has_hostile(self, hostile_ids: set[str] | None = None) -> bool:
        """Return True if room contains occupants considered hostile."""
        if hostile_ids is None:
            return False
        return bool(set(self.occupants) & hostile_ids)


@dataclass
class BuildingLayout:
    """Full layout of a building with one or more floors."""

    building_id: str
    position: Vec2  # world position of the building origin
    floors: int
    rooms: list[Room] = field(default_factory=list)
    entry_points: list[dict] = field(default_factory=list)
    # Each entry: {"position": Vec2, "type": "door"|"window", "room_id": str}
    is_hostile: bool = False
    cleared_rooms: int = 0
    total_rooms: int = 0

    def __post_init__(self) -> None:
        if self.total_rooms == 0:
            self.total_rooms = len(self.rooms)

    @property
    def is_fully_cleared(self) -> bool:
        return self.cleared_rooms >= self.total_rooms > 0

    def room_by_id(self, room_id: str) -> Room | None:
        for room in self.rooms:
            if room.room_id == room_id:
                return room
        return None

    def rooms_on_floor(self, floor: int) -> list[Room]:
        return [r for r in self.rooms if r.floor == floor]


# ---------------------------------------------------------------------------
# Building templates
# ---------------------------------------------------------------------------

BUILDING_TEMPLATES: dict[str, dict[str, Any]] = {
    "house": {
        "floors": 1,
        "rooms_per_floor": 4,
        "room_types": [RoomType.LOBBY, RoomType.ROOM, RoomType.BATHROOM, RoomType.ROOM],
        "entry_count": 2,
    },
    "apartment": {
        "floors": 3,
        "rooms_per_floor": 4,
        "room_types": [RoomType.HALLWAY, RoomType.ROOM, RoomType.ROOM, RoomType.BATHROOM],
        "entry_count": 2,
    },
    "office": {
        "floors": 5,
        "rooms_per_floor": 4,
        "room_types": [RoomType.HALLWAY, RoomType.OFFICE, RoomType.OFFICE, RoomType.BATHROOM],
        "entry_count": 3,
    },
    "warehouse": {
        "floors": 1,
        "rooms_per_floor": 4,
        "room_types": [RoomType.ROOM, RoomType.ROOM, RoomType.STORAGE, RoomType.STORAGE],
        "room_sizes": [(12.0, 10.0), (12.0, 10.0), (8.0, 6.0), (8.0, 6.0)],
        "entry_count": 3,
    },
    "compound": {
        "floors": 2,
        "rooms_per_floor": 4,
        "room_types": [RoomType.HALLWAY, RoomType.ROOM, RoomType.STORAGE, RoomType.ROOM],
        "entry_count": 4,
    },
}


# ---------------------------------------------------------------------------
# Room-clearing engine
# ---------------------------------------------------------------------------

class RoomClearingEngine:
    """Orchestrates building entry and room-by-room CQB clearing.

    Parameters
    ----------
    hostile_ids:
        Set of unit IDs that count as hostiles.  Passed at construction
        so that ``clear_room`` can determine engagement outcomes.
    """

    def __init__(self, hostile_ids: set[str] | None = None) -> None:
        self.buildings: dict[str, BuildingLayout] = {}
        self.hostile_ids: set[str] = hostile_ids or set()
        # Track which unit is in which room: unit_id -> (building_id, room_id)
        self._unit_locations: dict[str, tuple[str, str]] = {}

    # -- generation ---------------------------------------------------------

    def generate_layout(
        self,
        floors: int = 1,
        rooms_per_floor: int = 4,
        building_pos: Vec2 = (0.0, 0.0),
        template: str | None = None,
    ) -> BuildingLayout:
        """Procedurally generate a building with connected rooms.

        Each floor gets a hallway running left-to-right with rooms
        branching off it.  Stairwells connect floors vertically.
        """
        bid = f"bld_{uuid.uuid4().hex[:8]}"

        # Apply template overrides
        tpl: dict[str, Any] = {}
        if template and template in BUILDING_TEMPLATES:
            tpl = BUILDING_TEMPLATES[template]
            floors = tpl.get("floors", floors)
            rooms_per_floor = tpl.get("rooms_per_floor", rooms_per_floor)

        room_types_cycle: list[RoomType] = tpl.get(
            "room_types",
            [RoomType.HALLWAY] + [RoomType.ROOM] * (rooms_per_floor - 1),
        )
        room_sizes_override: list[tuple[float, float]] | None = tpl.get("room_sizes")
        entry_count: int = tpl.get("entry_count", 2)

        all_rooms: list[Room] = []
        entry_points: list[dict] = []

        hallway_ids: dict[int, str] = {}

        for fl in range(floors):
            floor_rooms: list[Room] = []
            y_offset = fl * 15.0  # vertical separation per floor in viz

            for ri in range(rooms_per_floor):
                rid = f"{bid}_f{fl}_r{ri}"
                rtype = room_types_cycle[ri % len(room_types_cycle)]

                # First room on each floor is the hallway/corridor
                if ri == 0:
                    rtype = RoomType.HALLWAY if floors == 1 else (
                        RoomType.LOBBY if fl == 0 else RoomType.HALLWAY
                    )

                if room_sizes_override and ri < len(room_sizes_override):
                    rsize = room_sizes_override[ri]
                else:
                    rsize = (6.0, 4.0) if rtype == RoomType.HALLWAY else (4.0, 4.0)

                rx = ri * (rsize[0] + 1.0) + building_pos[0]
                ry = y_offset + building_pos[1]

                doors: list[dict] = []
                windows: list[dict] = []

                # Connect to hallway (first room index 0 on this floor)
                if ri > 0 and floor_rooms:
                    hallway = floor_rooms[0]
                    door_pos: Vec2 = (
                        (hallway.position[0] + rx) / 2.0,
                        (hallway.position[1] + ry) / 2.0,
                    )
                    doors.append({"position": door_pos, "connects_to": hallway.room_id})
                    hallway.doors.append({"position": door_pos, "connects_to": rid})

                # Exterior window on the far wall
                window_pos: Vec2 = (rx + rsize[0] / 2.0, ry)
                windows.append({"position": window_pos, "facing": 0.0})

                # Cover positions — furniture
                cover: list[Vec2] = [
                    (rx - rsize[0] * 0.3, ry - rsize[1] * 0.3),
                    (rx + rsize[0] * 0.3, ry + rsize[1] * 0.3),
                ]

                room = Room(
                    room_id=rid,
                    room_type=rtype,
                    position=(rx, ry),
                    size=rsize,
                    floor=fl,
                    doors=doors,
                    windows=windows,
                    cover_positions=cover,
                    visibility=1.0 if rtype != RoomType.BASEMENT else 0.3,
                )
                floor_rooms.append(room)

                if ri == 0:
                    hallway_ids[fl] = rid

            all_rooms.extend(floor_rooms)

        # Connect floors via stairwells
        if floors > 1:
            for fl in range(floors - 1):
                stair_id = f"{bid}_stair_{fl}_{fl + 1}"
                sx = building_pos[0] + (rooms_per_floor + 0.5) * 5.0
                sy_lower = fl * 15.0 + building_pos[1]
                sy_upper = (fl + 1) * 15.0 + building_pos[1]
                sy = (sy_lower + sy_upper) / 2.0

                stair_doors: list[dict] = []
                # Connect to hallway on lower floor
                if fl in hallway_ids:
                    stair_doors.append({
                        "position": (sx, sy_lower),
                        "connects_to": hallway_ids[fl],
                    })
                    lower_hall = next(
                        (r for r in all_rooms if r.room_id == hallway_ids[fl]), None
                    )
                    if lower_hall:
                        lower_hall.doors.append({
                            "position": (sx, sy_lower),
                            "connects_to": stair_id,
                        })
                # Connect to hallway on upper floor
                upper_fl = fl + 1
                if upper_fl in hallway_ids:
                    stair_doors.append({
                        "position": (sx, sy_upper),
                        "connects_to": hallway_ids[upper_fl],
                    })
                    upper_hall = next(
                        (r for r in all_rooms if r.room_id == hallway_ids[upper_fl]),
                        None,
                    )
                    if upper_hall:
                        upper_hall.doors.append({
                            "position": (sx, sy_upper),
                            "connects_to": stair_id,
                        })

                stair_room = Room(
                    room_id=stair_id,
                    room_type=RoomType.STAIRWELL,
                    position=(sx, sy),
                    size=(3.0, 3.0),
                    floor=fl,
                    doors=stair_doors,
                )
                all_rooms.append(stair_room)

        # Entry points — first N ground-floor rooms
        ground_rooms = [r for r in all_rooms if r.floor == 0]
        for i, room in enumerate(ground_rooms[:entry_count]):
            entry_points.append({
                "position": room.position,
                "type": "door" if i == 0 else "window",
                "room_id": room.room_id,
            })

        layout = BuildingLayout(
            building_id=bid,
            position=building_pos,
            floors=floors,
            rooms=all_rooms,
            entry_points=entry_points,
            total_rooms=len(all_rooms),
        )
        self.buildings[bid] = layout
        return layout

    def add_building(self, layout: BuildingLayout) -> None:
        """Register an externally-created building layout."""
        self.buildings[layout.building_id] = layout

    # -- entry & movement ---------------------------------------------------

    def enter_building(
        self,
        unit_id: str,
        building_id: str,
        entry_point: int = 0,
    ) -> bool:
        """Move *unit_id* into the entry-point room of *building_id*.

        Parameters
        ----------
        entry_point:
            Index into the building's ``entry_points`` list.

        Returns ``True`` on success.
        """
        layout = self.buildings.get(building_id)
        if layout is None:
            return False
        if entry_point < 0 or entry_point >= len(layout.entry_points):
            return False

        ep = layout.entry_points[entry_point]
        room_id: str = ep["room_id"]
        room = layout.room_by_id(room_id)
        if room is None:
            return False

        # Remove from previous location if any
        self._remove_unit(unit_id)

        room.occupants.append(unit_id)
        self._unit_locations[unit_id] = (building_id, room_id)
        return True

    def move_unit(self, unit_id: str, target_room_id: str) -> bool:
        """Move a unit to an adjacent room (connected by door)."""
        loc = self._unit_locations.get(unit_id)
        if loc is None:
            return False
        building_id, current_room_id = loc
        layout = self.buildings.get(building_id)
        if layout is None:
            return False

        current_room = layout.room_by_id(current_room_id)
        if current_room is None:
            return False

        # Check adjacency via doors
        connected_ids = {d["connects_to"] for d in current_room.doors}
        if target_room_id not in connected_ids:
            return False

        target_room = layout.room_by_id(target_room_id)
        if target_room is None:
            return False

        current_room.occupants.remove(unit_id)
        target_room.occupants.append(unit_id)
        self._unit_locations[unit_id] = (building_id, target_room_id)
        return True

    # -- clearing -----------------------------------------------------------

    def clear_room(
        self,
        unit_ids: list[str],
        room_id: str,
        building_id: str | None = None,
        flashbang: bool = False,
    ) -> dict[str, Any]:
        """Execute CQB room clearing with the given units.

        All *unit_ids* must be inside the building (or will be moved to
        the room).  Returns a result dict with engagement details.

        Parameters
        ----------
        flashbang:
            If True, hostiles in the room are stunned, giving a +0.25
            accuracy bonus to the clearing team.
        """
        # Resolve building
        if building_id is None:
            # Infer from first unit's location
            for uid in unit_ids:
                loc = self._unit_locations.get(uid)
                if loc:
                    building_id = loc[0]
                    break
        if building_id is None:
            return {"success": False, "error": "no_building"}

        layout = self.buildings.get(building_id)
        if layout is None:
            return {"success": False, "error": "building_not_found"}

        room = layout.room_by_id(room_id)
        if room is None:
            return {"success": False, "error": "room_not_found"}

        # Move clearing units into the room
        for uid in unit_ids:
            loc = self._unit_locations.get(uid)
            if loc is None or loc[1] != room_id:
                # Place them in the room directly for clearing
                self._remove_unit(uid)
                room.occupants.append(uid)
                self._unit_locations[uid] = (building_id, room_id)

        # Identify hostiles present
        hostiles_in_room = [u for u in room.occupants if u in self.hostile_ids]
        friendlies_in_room = [u for u in unit_ids if u not in self.hostile_ids]

        # CQB engagement
        base_accuracy = 0.85  # CQB is close range -> high accuracy
        if flashbang:
            base_accuracy += 0.25
        base_accuracy = min(base_accuracy, 1.0)

        # Visibility penalty
        accuracy = base_accuracy * room.visibility

        hostiles_killed: list[str] = []
        friendly_casualties: list[str] = []

        # Clearing team fires first (initiative advantage)
        for hostile in hostiles_in_room:
            # Each friendly gets a shot at each hostile
            for _ in friendlies_in_room:
                if hostile in hostiles_killed:
                    break
                if random.random() < accuracy:
                    hostiles_killed.append(hostile)
                    break

        # Surviving hostiles fire back
        surviving_hostiles = [h for h in hostiles_in_room if h not in hostiles_killed]
        hostile_accuracy = 0.4 * room.visibility
        if flashbang:
            hostile_accuracy *= 0.3  # severely impaired

        for hostile in surviving_hostiles:
            target = random.choice(friendlies_in_room) if friendlies_in_room else None
            if target and random.random() < hostile_accuracy:
                if target not in friendly_casualties:
                    friendly_casualties.append(target)

        # Remove killed units
        for uid in hostiles_killed:
            if uid in room.occupants:
                room.occupants.remove(uid)
            self.hostile_ids.discard(uid)
            self._unit_locations.pop(uid, None)

        for uid in friendly_casualties:
            if uid in room.occupants:
                room.occupants.remove(uid)
            self._unit_locations.pop(uid, None)

        # Mark room cleared
        room.is_cleared = True
        layout.cleared_rooms = sum(1 for r in layout.rooms if r.is_cleared)

        return {
            "success": True,
            "room_id": room_id,
            "building_id": building_id,
            "hostiles_found": len(hostiles_in_room),
            "hostiles_killed": hostiles_killed,
            "friendly_casualties": friendly_casualties,
            "room_cleared": True,
            "building_cleared": layout.is_fully_cleared,
            "cleared_count": layout.cleared_rooms,
            "total_rooms": layout.total_rooms,
            "flashbang_used": flashbang,
            "accuracy": accuracy,
        }

    # -- queries ------------------------------------------------------------

    def get_unit_room(self, unit_id: str) -> Room | None:
        """Return the room a unit currently occupies, or None."""
        loc = self._unit_locations.get(unit_id)
        if loc is None:
            return None
        building_id, room_id = loc
        layout = self.buildings.get(building_id)
        if layout is None:
            return None
        return layout.room_by_id(room_id)

    def get_unit_building(self, unit_id: str) -> BuildingLayout | None:
        """Return the building a unit is inside, or None."""
        loc = self._unit_locations.get(unit_id)
        if loc is None:
            return None
        return self.buildings.get(loc[0])

    def get_adjacent_rooms(self, room_id: str, building_id: str) -> list[Room]:
        """Return rooms connected to *room_id* by doors."""
        layout = self.buildings.get(building_id)
        if layout is None:
            return []
        room = layout.room_by_id(room_id)
        if room is None:
            return []
        connected_ids = [d["connects_to"] for d in room.doors]
        return [r for r in layout.rooms if r.room_id in connected_ids]

    def get_uncleared_rooms(self, building_id: str) -> list[Room]:
        """Return rooms that have not yet been cleared."""
        layout = self.buildings.get(building_id)
        if layout is None:
            return []
        return [r for r in layout.rooms if not r.is_cleared]

    # -- simulation ---------------------------------------------------------

    def tick(self, dt: float) -> None:
        """Per-frame update: visibility decay, occupant bookkeeping."""
        for layout in self.buildings.values():
            for room in layout.rooms:
                # Stairwells and basements flicker slightly
                if room.room_type == RoomType.BASEMENT:
                    room.visibility = max(0.1, min(0.5, room.visibility + random.uniform(-0.02, 0.02)))
                elif room.room_type == RoomType.STAIRWELL:
                    room.visibility = max(0.3, min(0.8, room.visibility + random.uniform(-0.01, 0.01)))

                # Sync cleared count
            layout.cleared_rooms = sum(1 for r in layout.rooms if r.is_cleared)

    # -- visualization ------------------------------------------------------

    def to_three_js(self, building_id: str) -> dict[str, Any]:
        """Export a building as a Three.js-ready JSON structure.

        Returns a dict with floor plans, room geometries, door positions,
        occupant markers, and cleared-status overlays.
        """
        layout = self.buildings.get(building_id)
        if layout is None:
            return {"error": "building_not_found"}

        floors_data: list[dict[str, Any]] = []
        for fl in range(layout.floors):
            floor_rooms = layout.rooms_on_floor(fl)
            rooms_data: list[dict[str, Any]] = []
            for room in floor_rooms:
                rooms_data.append({
                    "room_id": room.room_id,
                    "type": room.room_type.value,
                    "position": {"x": room.position[0], "y": room.position[1]},
                    "size": {"width": room.size[0], "height": room.size[1]},
                    "doors": [
                        {
                            "position": {"x": d["position"][0], "y": d["position"][1]},
                            "connects_to": d["connects_to"],
                        }
                        for d in room.doors
                    ],
                    "windows": [
                        {
                            "position": {"x": w["position"][0], "y": w["position"][1]},
                            "facing": w["facing"],
                        }
                        for w in room.windows
                    ],
                    "occupants": list(room.occupants),
                    "is_cleared": room.is_cleared,
                    "visibility": room.visibility,
                    "cover_positions": [
                        {"x": c[0], "y": c[1]} for c in room.cover_positions
                    ],
                })
            floors_data.append({
                "floor": fl,
                "rooms": rooms_data,
            })

        return {
            "building_id": layout.building_id,
            "position": {"x": layout.position[0], "y": layout.position[1]},
            "floors": floors_data,
            "total_floors": layout.floors,
            "entry_points": [
                {
                    "position": {"x": ep["position"][0], "y": ep["position"][1]},
                    "type": ep["type"],
                    "room_id": ep["room_id"],
                }
                for ep in layout.entry_points
            ],
            "is_hostile": layout.is_hostile,
            "cleared_rooms": layout.cleared_rooms,
            "total_rooms": layout.total_rooms,
            "is_fully_cleared": layout.is_fully_cleared,
        }

    # -- internals ----------------------------------------------------------

    def _remove_unit(self, unit_id: str) -> None:
        """Remove a unit from whatever room it is currently in."""
        loc = self._unit_locations.pop(unit_id, None)
        if loc is None:
            return
        building_id, room_id = loc
        layout = self.buildings.get(building_id)
        if layout is None:
            return
        room = layout.room_by_id(room_id)
        if room and unit_id in room.occupants:
            room.occupants.remove(unit_id)
