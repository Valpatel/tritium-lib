# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Privacy zones — geographic areas where tracking is suppressed.

A ``PrivacyZone`` is a polygon on the map where some or all sensor
data collection and target tracking is suppressed.  This supports
compliance with privacy regulations that restrict surveillance in
certain areas (residential zones, schools, hospitals, etc.).

Uses ray-casting for point-in-polygon tests (same algorithm as
:mod:`tritium_lib.tracking.geofence`).
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("privacy.zone")


# ---------------------------------------------------------------------------
# Suppression levels
# ---------------------------------------------------------------------------

class SuppressionLevel(str, Enum):
    """How aggressively to suppress data in the zone."""
    NONE = "none"                  # zone is inactive
    ANONYMIZE = "anonymize"        # collect but anonymize identifiers
    SUPPRESS_PII = "suppress_pii"  # suppress PII fields, keep sensor data
    FULL = "full"                  # suppress all tracking in zone


# ---------------------------------------------------------------------------
# PrivacyZone
# ---------------------------------------------------------------------------

@dataclass
class PrivacyZone:
    """A geographic area where tracking is suppressed or anonymized.

    Attributes
    ----------
    zone_id : str
        Unique identifier.
    name : str
        Human-readable name (e.g. "School Zone - Lincoln Elementary").
    polygon : list[tuple[float, float]]
        Ordered vertices as (lat, lng) pairs.
    suppression : str
        How aggressively to suppress data in this zone.
    reason : str
        Why this zone exists (regulatory, policy, user request).
    enabled : bool
        Whether the zone is currently active.
    created_at : float
        When the zone was created.
    expires_at : float
        When the zone expires (0 = never).
    created_by : str
        Who created this zone.
    affected_sensors : list[str]
        Sensor types affected (empty = all). E.g. ["camera", "ble"].
    notes : str
        Additional context.
    """

    zone_id: str = ""
    name: str = ""
    polygon: list[tuple[float, float]] = field(default_factory=list)
    suppression: str = "full"
    reason: str = ""
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    created_by: str = ""
    affected_sensors: list[str] = field(default_factory=list)
    notes: str = ""

    def is_active(self, now: Optional[float] = None) -> bool:
        """Return True if the zone is currently active."""
        if not self.enabled:
            return False
        if self.suppression == SuppressionLevel.NONE:
            return False
        if self.expires_at > 0:
            now = now if now is not None else time.time()
            if now > self.expires_at:
                return False
        return True

    def contains_point(self, lat: float, lng: float) -> bool:
        """Test if a (lat, lng) point falls inside this zone.

        Uses ray-casting algorithm for point-in-polygon test.
        """
        if len(self.polygon) < 3:
            return False

        inside = False
        n = len(self.polygon)
        j = n - 1

        for i in range(n):
            lat_i, lng_i = self.polygon[i]
            lat_j, lng_j = self.polygon[j]

            if ((lat_i > lat) != (lat_j > lat)) and (
                lng < (lng_j - lng_i) * (lat - lat_i) / (lat_j - lat_i) + lng_i
            ):
                inside = not inside
            j = i

        return inside

    def affects_sensor(self, sensor_type: str) -> bool:
        """Check if this zone suppresses a specific sensor type.

        If ``affected_sensors`` is empty, all sensors are affected.
        """
        if not self.affected_sensors:
            return True
        return sensor_type.lower() in [s.lower() for s in self.affected_sensors]

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "polygon": [list(p) for p in self.polygon],
            "suppression": self.suppression,
            "reason": self.reason,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "created_by": self.created_by,
            "affected_sensors": list(self.affected_sensors),
            "notes": self.notes,
        }

    @staticmethod
    def create(
        name: str,
        polygon: list[tuple[float, float]],
        suppression: str = "full",
        reason: str = "",
        created_by: str = "",
        affected_sensors: Optional[list[str]] = None,
        expires_at: float = 0.0,
        notes: str = "",
    ) -> PrivacyZone:
        """Factory: create a new active privacy zone."""
        return PrivacyZone(
            zone_id=str(uuid.uuid4()),
            name=name,
            polygon=polygon,
            suppression=suppression,
            reason=reason,
            enabled=True,
            created_by=created_by,
            affected_sensors=affected_sensors or [],
            expires_at=expires_at,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# PrivacyZoneManager
# ---------------------------------------------------------------------------

class PrivacyZoneManager:
    """Manage privacy zones and check coordinates against them.

    Usage
    -----
    ::

        mgr = PrivacyZoneManager()
        zone = mgr.add_zone("School", polygon, suppression="full")
        result = mgr.check_point(40.7128, -74.0060)
        if result.suppressed:
            # Don't track / anonymize data
            ...
    """

    def __init__(self) -> None:
        self._zones: dict[str, PrivacyZone] = {}

    # -- zone management ----------------------------------------------------

    def add_zone(
        self,
        name: str,
        polygon: list[tuple[float, float]],
        suppression: str = "full",
        reason: str = "",
        created_by: str = "",
        affected_sensors: Optional[list[str]] = None,
        expires_at: float = 0.0,
    ) -> PrivacyZone:
        """Create and register a new privacy zone."""
        zone = PrivacyZone.create(
            name=name,
            polygon=polygon,
            suppression=suppression,
            reason=reason,
            created_by=created_by,
            affected_sensors=affected_sensors,
            expires_at=expires_at,
        )
        self._zones[zone.zone_id] = zone
        logger.info("Privacy zone added: %s (%s)", name, zone.zone_id)
        return zone

    def remove_zone(self, zone_id: str) -> bool:
        """Remove a privacy zone.  Returns True if it existed."""
        return self._zones.pop(zone_id, None) is not None

    def get_zone(self, zone_id: str) -> Optional[PrivacyZone]:
        """Get a zone by ID."""
        return self._zones.get(zone_id)

    def list_zones(self, active_only: bool = False) -> list[PrivacyZone]:
        """List all zones, optionally filtered to active-only."""
        zones = list(self._zones.values())
        if active_only:
            zones = [z for z in zones if z.is_active()]
        return sorted(zones, key=lambda z: z.name)

    def update_zone(self, zone_id: str, **kwargs: Any) -> Optional[PrivacyZone]:
        """Update zone fields.  Returns updated zone or None if not found."""
        zone = self._zones.get(zone_id)
        if zone is None:
            return None
        d = zone.to_dict()
        d.update(kwargs)
        # Reconstruct with updated fields
        polygon = d.get("polygon", zone.polygon)
        if polygon and isinstance(polygon[0], list):
            polygon = [tuple(p) for p in polygon]
        updated = PrivacyZone(
            zone_id=zone.zone_id,
            name=d.get("name", zone.name),
            polygon=polygon,
            suppression=d.get("suppression", zone.suppression),
            reason=d.get("reason", zone.reason),
            enabled=d.get("enabled", zone.enabled),
            created_at=zone.created_at,
            expires_at=d.get("expires_at", zone.expires_at),
            created_by=zone.created_by,
            affected_sensors=d.get("affected_sensors", zone.affected_sensors),
            notes=d.get("notes", zone.notes),
        )
        self._zones[zone_id] = updated
        return updated

    # -- point checking -----------------------------------------------------

    def check_point(
        self,
        lat: float,
        lng: float,
        sensor_type: str = "",
        now: Optional[float] = None,
    ) -> ZoneCheckResult:
        """Check if a point falls inside any active privacy zone.

        Parameters
        ----------
        lat, lng :
            Coordinates to check.
        sensor_type :
            Optional sensor type for zone filtering.
        now :
            Current time for expiry checks.

        Returns a :class:`ZoneCheckResult`.
        """
        matching_zones: list[PrivacyZone] = []
        max_suppression = SuppressionLevel.NONE

        for zone in self._zones.values():
            if not zone.is_active(now=now):
                continue
            if sensor_type and not zone.affects_sensor(sensor_type):
                continue
            if zone.contains_point(lat, lng):
                matching_zones.append(zone)
                # Track highest suppression level
                zone_level = _suppression_rank(zone.suppression)
                current_max = _suppression_rank(max_suppression)
                if zone_level > current_max:
                    max_suppression = zone.suppression

        return ZoneCheckResult(
            suppressed=len(matching_zones) > 0,
            suppression_level=max_suppression,
            matching_zones=matching_zones,
            lat=lat,
            lng=lng,
        )

    def check_target(
        self,
        target_id: str,
        lat: float,
        lng: float,
        sensor_type: str = "",
    ) -> ZoneCheckResult:
        """Convenience: check a target position against all zones."""
        result = self.check_point(lat, lng, sensor_type=sensor_type)
        result.target_id = target_id
        return result

    # -- export -------------------------------------------------------------

    def export(self) -> dict[str, Any]:
        """Export all zones as a serializable dict."""
        return {
            "zones": {zid: z.to_dict() for zid, z in self._zones.items()},
            "total": len(self._zones),
            "active": sum(1 for z in self._zones.values() if z.is_active()),
        }


# ---------------------------------------------------------------------------
# ZoneCheckResult
# ---------------------------------------------------------------------------

@dataclass
class ZoneCheckResult:
    """Result of checking a point against privacy zones."""

    suppressed: bool = False
    suppression_level: str = "none"
    matching_zones: list[PrivacyZone] = field(default_factory=list)
    lat: float = 0.0
    lng: float = 0.0
    target_id: str = ""

    @property
    def zone_count(self) -> int:
        return len(self.matching_zones)

    @property
    def zone_names(self) -> list[str]:
        return [z.name for z in self.matching_zones]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suppressed": self.suppressed,
            "suppression_level": self.suppression_level,
            "matching_zones": [z.zone_id for z in self.matching_zones],
            "zone_names": self.zone_names,
            "zone_count": self.zone_count,
            "lat": self.lat,
            "lng": self.lng,
            "target_id": self.target_id,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUPPRESSION_RANK = {
    SuppressionLevel.NONE: 0,
    SuppressionLevel.ANONYMIZE: 1,
    SuppressionLevel.SUPPRESS_PII: 2,
    SuppressionLevel.FULL: 3,
}


def _suppression_rank(level: str) -> int:
    """Return numeric rank for a suppression level."""
    return _SUPPRESSION_RANK.get(level, 0)
