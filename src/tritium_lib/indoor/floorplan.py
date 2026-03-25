# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Lightweight building floor plan model for indoor positioning.

This is an algorithmic spatial model — rooms as axis-aligned rectangles
with doors connecting them. Designed for zone containment checks, path
reasoning, and integration with the fingerprint-based position estimator.

For the geo-referenced Pydantic model used by the SC frontend/backend,
see :mod:`tritium_lib.models.floorplan`.

Pure Python — no numpy required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Room types
# ---------------------------------------------------------------------------

class RoomType(str, Enum):
    """Classification for rooms/zones."""
    OFFICE = "office"
    CONFERENCE = "conference"
    HALLWAY = "hallway"
    CORRIDOR = "corridor"
    BATHROOM = "bathroom"
    KITCHEN = "kitchen"
    LOBBY = "lobby"
    STORAGE = "storage"
    SERVER_ROOM = "server_room"
    STAIRWELL = "stairwell"
    ELEVATOR = "elevator"
    OPEN_AREA = "open_area"
    RESTRICTED = "restricted"
    LAB = "lab"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Door
# ---------------------------------------------------------------------------

@dataclass
class Door:
    """A door connecting two rooms.

    Attributes:
        door_id: Unique door identifier.
        room_a: ID of the first room.
        room_b: ID of the second room.
        x: X-coordinate of the door in local metres.
        y: Y-coordinate of the door in local metres.
        width: Door width in metres (default 0.9m).
        is_open: Whether the door is currently open.
    """
    door_id: str
    room_a: str
    room_b: str
    x: float = 0.0
    y: float = 0.0
    width: float = 0.9
    is_open: bool = True

    def connects(self, room_id: str) -> bool:
        """Check if this door connects to a given room."""
        return room_id in (self.room_a, self.room_b)

    def other_room(self, room_id: str) -> Optional[str]:
        """Return the room on the other side, or None if not connected."""
        if room_id == self.room_a:
            return self.room_b
        if room_id == self.room_b:
            return self.room_a
        return None

    def to_dict(self) -> dict:
        return {
            "door_id": self.door_id,
            "room_a": self.room_a,
            "room_b": self.room_b,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "is_open": self.is_open,
        }


# ---------------------------------------------------------------------------
# Room
# ---------------------------------------------------------------------------

@dataclass
class Room:
    """A rectangular room in a building floor plan.

    Defined by an axis-aligned bounding box (x_min, y_min) to
    (x_max, y_max) in local metres.

    Attributes:
        room_id: Unique room identifier.
        name: Human-readable room name.
        room_type: Type classification.
        x_min: Left edge X-coordinate.
        y_min: Bottom edge Y-coordinate.
        x_max: Right edge X-coordinate.
        y_max: Top edge Y-coordinate.
        floor: Floor level (0 = ground).
        capacity: Maximum occupancy (None = unknown).
    """
    room_id: str
    name: str
    room_type: RoomType = RoomType.OTHER
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 0.0
    y_max: float = 0.0
    floor: int = 0
    capacity: Optional[int] = None

    @property
    def width(self) -> float:
        """Width (X extent) in metres."""
        return abs(self.x_max - self.x_min)

    @property
    def height(self) -> float:
        """Height (Y extent) in metres."""
        return abs(self.y_max - self.y_min)

    @property
    def area(self) -> float:
        """Floor area in square metres."""
        return self.width * self.height

    @property
    def center_x(self) -> float:
        """X-coordinate of room centre."""
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        """Y-coordinate of room centre."""
        return (self.y_min + self.y_max) / 2.0

    def contains(self, x: float, y: float) -> bool:
        """Check if a point (x, y) is inside this room."""
        return (self.x_min <= x <= self.x_max
                and self.y_min <= y <= self.y_max)

    def distance_to(self, x: float, y: float) -> float:
        """Distance from a point to the nearest edge of this room.

        Returns 0 if the point is inside the room.
        """
        dx = max(self.x_min - x, 0.0, x - self.x_max)
        dy = max(self.y_min - y, 0.0, y - self.y_max)
        return (dx ** 2 + dy ** 2) ** 0.5

    def to_dict(self) -> dict:
        return {
            "room_id": self.room_id,
            "name": self.name,
            "room_type": self.room_type.value,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "floor": self.floor,
            "capacity": self.capacity,
            "area": round(self.area, 2),
        }


# ---------------------------------------------------------------------------
# FloorPlan
# ---------------------------------------------------------------------------

class FloorPlan:
    """Building floor plan with rooms, corridors, and doors.

    Lightweight spatial model for indoor positioning zone containment
    checks, adjacency queries, and path reasoning.

    Args:
        building_id: Unique building identifier.
        name: Human-readable building name.
        floor: Floor level this plan represents.
    """

    def __init__(
        self,
        building_id: str = "default",
        name: str = "",
        floor: int = 0,
    ) -> None:
        self.building_id = building_id
        self.name = name
        self.floor = floor
        self._rooms: dict[str, Room] = {}
        self._doors: dict[str, Door] = {}

    # -- Mutation -----------------------------------------------------------

    def add_room(self, room: Room) -> None:
        """Add a room to the floor plan."""
        self._rooms[room.room_id] = room

    def add_door(self, door: Door) -> None:
        """Add a door connecting two rooms."""
        self._doors[door.door_id] = door

    def remove_room(self, room_id: str) -> bool:
        """Remove a room and any doors connected to it."""
        if room_id not in self._rooms:
            return False
        del self._rooms[room_id]
        # Remove doors connected to this room
        to_remove = [
            did for did, d in self._doors.items()
            if d.connects(room_id)
        ]
        for did in to_remove:
            del self._doors[did]
        return True

    def remove_door(self, door_id: str) -> bool:
        """Remove a door by ID."""
        return self._doors.pop(door_id, None) is not None

    # -- Queries ------------------------------------------------------------

    @property
    def room_count(self) -> int:
        return len(self._rooms)

    @property
    def door_count(self) -> int:
        return len(self._doors)

    def get_room(self, room_id: str) -> Optional[Room]:
        """Get a room by ID."""
        return self._rooms.get(room_id)

    def get_door(self, door_id: str) -> Optional[Door]:
        """Get a door by ID."""
        return self._doors.get(door_id)

    def all_rooms(self) -> list[Room]:
        """Return all rooms."""
        return list(self._rooms.values())

    def all_doors(self) -> list[Door]:
        """Return all doors."""
        return list(self._doors.values())

    def find_room_at(self, x: float, y: float) -> Optional[Room]:
        """Find which room contains the given point.

        Returns the first matching room, or None if the point is
        outside all rooms.
        """
        for room in self._rooms.values():
            if room.contains(x, y):
                return room
        return None

    def find_nearest_room(self, x: float, y: float) -> Optional[Room]:
        """Find the room nearest to a point (even if the point is outside).

        Returns the room whose boundary is closest, or None if no rooms.
        """
        if not self._rooms:
            return None
        return min(self._rooms.values(), key=lambda r: r.distance_to(x, y))

    def get_adjacent_rooms(self, room_id: str) -> list[str]:
        """Return IDs of rooms connected to a room by doors."""
        adjacent: list[str] = []
        for door in self._doors.values():
            other = door.other_room(room_id)
            if other is not None:
                adjacent.append(other)
        return adjacent

    def get_doors_for_room(self, room_id: str) -> list[Door]:
        """Return all doors connected to a room."""
        return [d for d in self._doors.values() if d.connects(room_id)]

    @property
    def total_area(self) -> float:
        """Total floor area across all rooms in square metres."""
        return sum(r.area for r in self._rooms.values())

    # -- Serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """Export as a JSON-serialisable dict."""
        return {
            "building_id": self.building_id,
            "name": self.name,
            "floor": self.floor,
            "room_count": self.room_count,
            "door_count": self.door_count,
            "total_area": round(self.total_area, 2),
            "rooms": [r.to_dict() for r in self._rooms.values()],
            "doors": [d.to_dict() for d in self._doors.values()],
        }

    def get_status(self) -> dict:
        """Return summary status."""
        return {
            "building_id": self.building_id,
            "name": self.name,
            "floor": self.floor,
            "room_count": self.room_count,
            "door_count": self.door_count,
            "total_area": round(self.total_area, 2),
        }
