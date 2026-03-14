# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Floor plan models for indoor spatial intelligence.

Supports uploaded SVG/PNG floor plans geo-referenced to map coordinates,
room/zone definitions as polygons, and indoor position estimates for
target localization within buildings.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FloorPlanStatus(str, Enum):
    """Status of a floor plan."""
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class GeoAnchor(BaseModel):
    """A geo-reference anchor point mapping image pixels to lat/lon.

    At least 2 anchors are needed to geo-reference a floor plan.
    3+ anchors allow affine correction for rotation/skew.
    """
    pixel_x: float = Field(description="X coordinate in image pixels")
    pixel_y: float = Field(description="Y coordinate in image pixels")
    lat: float = Field(description="WGS84 latitude")
    lon: float = Field(description="WGS84 longitude")
    label: str = ""


class FloorPlanBounds(BaseModel):
    """Geographic bounding box for a floor plan overlay."""
    north: float
    south: float
    east: float
    west: float

    @property
    def center_lat(self) -> float:
        return (self.north + self.south) / 2.0

    @property
    def center_lon(self) -> float:
        return (self.east + self.west) / 2.0

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a point falls within these bounds."""
        return (self.south <= lat <= self.north
                and self.west <= lon <= self.east)


class RoomType(str, Enum):
    """Type classification for rooms/zones."""
    OFFICE = "office"
    CONFERENCE = "conference"
    HALLWAY = "hallway"
    BATHROOM = "bathroom"
    KITCHEN = "kitchen"
    LOBBY = "lobby"
    STORAGE = "storage"
    SERVER_ROOM = "server_room"
    STAIRWELL = "stairwell"
    ELEVATOR = "elevator"
    OPEN_AREA = "open_area"
    RESTRICTED = "restricted"
    OTHER = "other"


class PolygonPoint(BaseModel):
    """A point in a polygon, in lat/lon coordinates."""
    lat: float
    lon: float


class Room(BaseModel):
    """A room or zone within a floor plan.

    Defined by a polygon in geographic coordinates for containment checks.
    """
    room_id: str = Field(description="Unique room identifier")
    name: str = Field(description="Human-readable room name")
    room_type: RoomType = RoomType.OTHER
    floor_level: int = 0
    polygon: list[PolygonPoint] = Field(
        default_factory=list,
        description="Polygon vertices in lat/lon defining the room boundary",
    )
    capacity: Optional[int] = Field(
        None, description="Maximum occupancy for this room"
    )
    tags: list[str] = Field(default_factory=list)
    color: str = Field(
        default="#00f0ff",
        description="Display color for this room (hex)",
    )

    def contains_point(self, lat: float, lon: float) -> bool:
        """Ray-casting point-in-polygon test.

        Returns True if the given lat/lon falls inside this room's polygon.
        """
        if len(self.polygon) < 3:
            return False

        n = len(self.polygon)
        inside = False
        j = n - 1
        for i in range(n):
            pi = self.polygon[i]
            pj = self.polygon[j]
            if ((pi.lon > lon) != (pj.lon > lon)) and (
                lat < (pj.lat - pi.lat) * (lon - pi.lon) / (pj.lon - pi.lon) + pi.lat
            ):
                inside = not inside
            j = i
        return inside


class FloorPlan(BaseModel):
    """A geo-referenced floor plan image.

    Represents an uploaded SVG/PNG floor plan that has been geo-referenced
    to map coordinates, with optional room definitions for indoor
    target localization.
    """
    plan_id: str = Field(description="Unique floor plan identifier")
    name: str = Field(description="Human-readable name")
    building: str = Field(default="", description="Building name or identifier")
    floor_level: int = Field(default=0, description="Floor number (0=ground)")
    image_path: str = Field(
        default="",
        description="Path to the floor plan image file (relative to data/floorplans/)",
    )
    image_format: str = Field(default="png", description="Image format: png, svg, jpg")
    image_width: int = Field(default=0, description="Image width in pixels")
    image_height: int = Field(default=0, description="Image height in pixels")
    bounds: Optional[FloorPlanBounds] = Field(
        None, description="Geographic bounds for map overlay positioning"
    )
    anchors: list[GeoAnchor] = Field(
        default_factory=list,
        description="Geo-reference anchor points (image pixels -> lat/lon)",
    )
    rooms: list[Room] = Field(
        default_factory=list,
        description="Room/zone definitions within this floor plan",
    )
    status: FloorPlanStatus = FloorPlanStatus.DRAFT
    opacity: float = Field(
        default=0.7,
        description="Overlay opacity (0.0-1.0) for map rendering",
        ge=0.0,
        le=1.0,
    )
    rotation: float = Field(
        default=0.0,
        description="Rotation angle in degrees for map alignment",
    )
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def find_room(self, lat: float, lon: float) -> Optional[Room]:
        """Find which room contains the given lat/lon point.

        Returns the first matching room, or None if the point is not
        inside any defined room.
        """
        for room in self.rooms:
            if room.contains_point(lat, lon):
                return room
        return None

    def get_room_by_id(self, room_id: str) -> Optional[Room]:
        """Look up a room by its ID."""
        for room in self.rooms:
            if room.room_id == room_id:
                return room
        return None


class IndoorPosition(BaseModel):
    """Indoor position estimate for a tracked target.

    Links a target to a room/zone based on BLE trilateration or
    WiFi fingerprint positioning within a building.
    """
    target_id: str = Field(description="Target identifier (e.g., ble_{mac})")
    plan_id: str = Field(description="Floor plan this position is within")
    room_id: Optional[str] = Field(
        None, description="Room ID if target is localized to a room"
    )
    floor_level: int = Field(default=0, description="Floor level estimate")
    lat: Optional[float] = Field(None, description="Estimated latitude")
    lon: Optional[float] = Field(None, description="Estimated longitude")
    confidence: float = Field(
        default=0.0,
        description="Position confidence (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    method: str = Field(
        default="trilateration",
        description="Positioning method: trilateration, fingerprint, proximity",
    )
    timestamp: Optional[datetime] = None


class RoomOccupancy(BaseModel):
    """Occupancy summary for a single room."""
    room_id: str
    room_name: str
    room_type: RoomType = RoomType.OTHER
    floor_level: int = 0
    person_count: int = 0
    device_count: int = 0
    target_ids: list[str] = Field(default_factory=list)
    capacity: Optional[int] = None
    updated_at: Optional[datetime] = None

    @property
    def occupancy_ratio(self) -> Optional[float]:
        """Ratio of person count to capacity, or None if no capacity set."""
        if self.capacity and self.capacity > 0:
            return self.person_count / self.capacity
        return None


class BuildingOccupancy(BaseModel):
    """Occupancy summary for an entire building/floor plan."""
    plan_id: str
    building: str = ""
    floor_level: int = 0
    total_persons: int = 0
    total_devices: int = 0
    rooms: list[RoomOccupancy] = Field(default_factory=list)
    updated_at: Optional[datetime] = None


class WiFiRSSIFingerprint(BaseModel):
    """A WiFi RSSI fingerprint at a known position.

    Collected by walking through a building with an edge device and
    recording RSSI values from visible access points at known positions.
    """
    fingerprint_id: str = Field(description="Unique fingerprint ID")
    plan_id: str = Field(description="Floor plan this fingerprint belongs to")
    room_id: Optional[str] = Field(None, description="Room where this was collected")
    lat: float = Field(description="Collection position latitude")
    lon: float = Field(description="Collection position longitude")
    floor_level: int = 0
    rssi_map: dict[str, float] = Field(
        default_factory=dict,
        description="BSSID -> RSSI value mapping at this position",
    )
    collected_at: Optional[datetime] = None
    device_id: str = Field(
        default="",
        description="Edge device that collected this fingerprint",
    )
