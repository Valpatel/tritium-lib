# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Zone mapper — map fingerprint positions to named zones.

A :class:`Zone` is a named region (lobby, office-A, server-room, ...) that
covers a set of reference fingerprint positions. :class:`ZoneMapper` provides:

- Manual zone assignment (map fingerprints to zones by label or bounding box).
- Automatic zone detection from :class:`FloorPlan` rooms.
- Live position-to-zone resolution: given an estimated (x, y), return
  which zone the target is in.

Pure Python — no numpy required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .fingerprint import Fingerprint, FingerprintDB
from .floorplan import FloorPlan, Room


# ---------------------------------------------------------------------------
# Zone
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    """A named indoor zone (a logical region inside a building).

    Defined by a bounding box in local metres. Fingerprints and position
    estimates can be resolved to zones for human-readable location.

    Attributes:
        zone_id: Unique zone identifier.
        name: Human-readable zone name (e.g. "Lobby", "Office 201").
        x_min: Left edge X-coordinate.
        y_min: Bottom edge Y-coordinate.
        x_max: Right edge X-coordinate.
        y_max: Top edge Y-coordinate.
        floor: Floor level.
        tags: Arbitrary metadata tags.
    """
    zone_id: str
    name: str
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 0.0
    y_max: float = 0.0
    floor: int = 0
    tags: list[str] = field(default_factory=list)

    def contains(self, x: float, y: float) -> bool:
        """Check if a point falls inside this zone."""
        return (self.x_min <= x <= self.x_max
                and self.y_min <= y <= self.y_max)

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def area(self) -> float:
        return abs(self.x_max - self.x_min) * abs(self.y_max - self.y_min)

    def distance_to(self, x: float, y: float) -> float:
        """Distance from a point to the nearest edge. 0 if inside."""
        dx = max(self.x_min - x, 0.0, x - self.x_max)
        dy = max(self.y_min - y, 0.0, y - self.y_max)
        return (dx ** 2 + dy ** 2) ** 0.5

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "floor": self.floor,
            "area": round(self.area, 2),
            "tags": list(self.tags),
        }


# ---------------------------------------------------------------------------
# ZoneMapper
# ---------------------------------------------------------------------------

class ZoneMapper:
    """Map fingerprint positions and live estimates to named zones.

    Supports manual zone definitions and automatic zone creation from
    a :class:`FloorPlan`.

    Args:
        building_id: Identifier for the building.
    """

    def __init__(self, building_id: str = "default") -> None:
        self.building_id = building_id
        self._zones: dict[str, Zone] = {}

    # -- Zone management ----------------------------------------------------

    def add_zone(self, zone: Zone) -> None:
        """Add or replace a zone."""
        self._zones[zone.zone_id] = zone

    def remove_zone(self, zone_id: str) -> bool:
        """Remove a zone by ID. Returns True if it existed."""
        return self._zones.pop(zone_id, None) is not None

    def get_zone(self, zone_id: str) -> Optional[Zone]:
        """Look up a zone by ID."""
        return self._zones.get(zone_id)

    def all_zones(self) -> list[Zone]:
        """Return all zones."""
        return list(self._zones.values())

    @property
    def zone_count(self) -> int:
        return len(self._zones)

    def clear(self) -> None:
        """Remove all zones."""
        self._zones.clear()

    # -- Zone resolution ----------------------------------------------------

    def resolve(self, x: float, y: float, floor: int = 0) -> Optional[Zone]:
        """Find which zone contains the given position.

        If the point is inside multiple overlapping zones, the smallest
        (by area) is returned — this favours more specific zones.

        Args:
            x: X-coordinate in local metres.
            y: Y-coordinate in local metres.
            floor: Floor level to match.

        Returns:
            The containing :class:`Zone`, or None if outside all zones.
        """
        candidates = [
            z for z in self._zones.values()
            if z.floor == floor and z.contains(x, y)
        ]
        if not candidates:
            return None
        # Prefer smallest zone (most specific)
        return min(candidates, key=lambda z: z.area)

    def resolve_nearest(
        self,
        x: float,
        y: float,
        floor: int = 0,
        max_distance: float = float("inf"),
    ) -> Optional[Zone]:
        """Find the nearest zone, even if the point is outside all zones.

        Args:
            x: X-coordinate.
            y: Y-coordinate.
            floor: Floor level.
            max_distance: Maximum distance in metres to consider.

        Returns:
            Nearest zone within max_distance, or None.
        """
        same_floor = [z for z in self._zones.values() if z.floor == floor]
        if not same_floor:
            return None

        nearest = min(same_floor, key=lambda z: z.distance_to(x, y))
        if nearest.distance_to(x, y) <= max_distance:
            return nearest
        return None

    def resolve_name(self, x: float, y: float, floor: int = 0) -> str:
        """Return the zone name for a position, or "Unknown" if outside."""
        zone = self.resolve(x, y, floor)
        if zone is not None:
            return zone.name
        return "Unknown"

    # -- Bulk operations ----------------------------------------------------

    def map_fingerprints(
        self,
        db: FingerprintDB,
    ) -> dict[str, str]:
        """Assign each fingerprint in a database to a zone.

        Returns a mapping {fingerprint_id: zone_name}. Fingerprints not
        inside any zone are mapped to "Unknown".
        """
        result: dict[str, str] = {}
        for fp in db.all():
            name = self.resolve_name(fp.x, fp.y, fp.floor)
            result[fp.fingerprint_id] = name
        return result

    def create_zones_from_floorplan(self, floorplan: FloorPlan) -> int:
        """Auto-create zones from a floor plan's rooms.

        Each room becomes a zone with the same bounds and name.
        Existing zones with the same ID are overwritten.

        Args:
            floorplan: The floor plan to extract rooms from.

        Returns:
            Number of zones created.
        """
        count = 0
        for room in floorplan.all_rooms():
            zone = Zone(
                zone_id=room.room_id,
                name=room.name,
                x_min=room.x_min,
                y_min=room.y_min,
                x_max=room.x_max,
                y_max=room.y_max,
                floor=room.floor,
                tags=[room.room_type.value],
            )
            self.add_zone(zone)
            count += 1
        return count

    # -- Zone statistics ----------------------------------------------------

    def zone_occupancy(
        self,
        positions: list[tuple[float, float, int]],
    ) -> dict[str, int]:
        """Count how many positions fall in each zone.

        Args:
            positions: List of (x, y, floor) tuples.

        Returns:
            Mapping {zone_name: count}. Positions outside all zones are
            counted under "Unknown".
        """
        counts: dict[str, int] = {}
        for x, y, fl in positions:
            name = self.resolve_name(x, y, fl)
            counts[name] = counts.get(name, 0) + 1
        return counts

    # -- Serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "building_id": self.building_id,
            "zone_count": self.zone_count,
            "zones": [z.to_dict() for z in self._zones.values()],
        }

    def get_status(self) -> dict:
        return {
            "building_id": self.building_id,
            "zone_count": self.zone_count,
        }
