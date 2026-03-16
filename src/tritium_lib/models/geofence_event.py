# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Geofence crossing event model.

Structured record for when a target enters or exits a geofence zone.
Used by tritium-sc geofence engine and edge firmware for zone monitoring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class GeofenceEvent:
    """A structured geofence crossing record.

    Attributes:
        target_id: Unique target identifier (e.g. ble_AA:BB:CC, det_person_3).
        zone_id: Unique geofence zone identifier.
        direction: Crossing direction — 'enter' or 'exit'.
        timestamp: Unix epoch time of the crossing.
        target_alliance: Alliance of the target — 'friendly', 'hostile', or 'unknown'.
        zone_type: Type of zone — 'restricted', 'monitored', or 'safe'.
        zone_name: Human-readable zone name (optional convenience field).
        position: Target position at crossing time as (lat, lng) tuple, if available.
    """

    target_id: str
    zone_id: str
    direction: str  # "enter" or "exit"
    timestamp: float = field(default_factory=time.time)
    target_alliance: str = "unknown"  # "friendly", "hostile", "unknown"
    zone_type: str = "monitored"  # "restricted", "monitored", "safe"
    zone_name: str = ""
    position: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        """Serialize to dict for JSON transport."""
        d: dict = {
            "target_id": self.target_id,
            "zone_id": self.zone_id,
            "direction": self.direction,
            "timestamp": self.timestamp,
            "target_alliance": self.target_alliance,
            "zone_type": self.zone_type,
        }
        if self.zone_name:
            d["zone_name"] = self.zone_name
        if self.position is not None:
            d["position"] = list(self.position)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> GeofenceEvent:
        """Deserialize from dict."""
        pos = data.get("position")
        return cls(
            target_id=data["target_id"],
            zone_id=data["zone_id"],
            direction=data["direction"],
            timestamp=data.get("timestamp", time.time()),
            target_alliance=data.get("target_alliance", "unknown"),
            zone_type=data.get("zone_type", "monitored"),
            zone_name=data.get("zone_name", ""),
            position=tuple(pos) if pos else None,
        )
