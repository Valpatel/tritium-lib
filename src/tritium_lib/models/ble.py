# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""BLE presence models — used when edge nodes scan for nearby BLE devices
and report sightings to the fleet server for presence tracking."""

import re
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

_MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


class BleDevice(BaseModel):
    """A BLE device discovered by one or more nodes."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "mac": "AA:BB:CC:DD:EE:FF",
                    "rssi": -65,
                    "name": "iPhone",
                    "seen_count": 3,
                }
            ]
        }
    )

    mac: str
    rssi: int = Field(..., ge=-127, le=0)
    name: str = ""
    seen_count: int = Field(1, ge=0)
    is_known: bool = False
    first_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("mac")
    @classmethod
    def _validate_mac(cls, v: str) -> str:
        """Validate and normalize MAC address to uppercase."""
        v = v.upper()
        if not _MAC_RE.match(v):
            raise ValueError(
                f"Invalid MAC address '{v}' — expected format AA:BB:CC:DD:EE:FF"
            )
        return v

    def to_summary(self) -> str:
        """Human-readable one-line summary."""
        name_str = f" ({self.name})" if self.name else ""
        return f"BLE {self.mac}{name_str} rssi={self.rssi} seen={self.seen_count}"


class BleSighting(BaseModel):
    """A single BLE sighting reported by a specific node."""
    device: BleDevice
    node_id: str = Field(..., min_length=1)
    node_ip: str = ""
    node_wifi_rssi: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"Sighting {self.device.mac} by {self.node_id} "
            f"rssi={self.device.rssi} at {self.timestamp.isoformat()}"
        )


class BlePresence(BaseModel):
    """Aggregated presence for a single BLE device across multiple nodes."""
    mac: str
    name: str = ""
    sightings: list[BleSighting] = Field(default_factory=list)
    strongest_rssi: int = Field(-100, ge=-127, le=0)
    node_count: int = Field(0, ge=0)

    @field_validator("mac")
    @classmethod
    def _validate_mac(cls, v: str) -> str:
        v = v.upper()
        if not _MAC_RE.match(v):
            raise ValueError(
                f"Invalid MAC address '{v}' — expected format AA:BB:CC:DD:EE:FF"
            )
        return v

    def to_summary(self) -> str:
        """Human-readable one-line summary."""
        name_str = f" ({self.name})" if self.name else ""
        return (
            f"Presence {self.mac}{name_str} best_rssi={self.strongest_rssi} "
            f"nodes={self.node_count} sightings={len(self.sightings)}"
        )


class BlePresenceMap(BaseModel):
    """Full presence map — all BLE devices seen across the fleet."""
    devices: dict[str, BlePresence] = Field(default_factory=dict)
    total_devices: int = 0
    total_nodes: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Known node positions for triangulation: node_id -> (x, y)
# Populated by the fleet server from node config.
_node_positions: dict[str, tuple[float, float]] = {}


def set_node_positions(positions: dict[str, tuple[float, float]]) -> None:
    """Register known node positions for triangulation."""
    _node_positions.clear()
    _node_positions.update(positions)


def triangulate_position(
    sightings: list[BleSighting],
) -> Optional[tuple[float, float]]:
    """Estimate device position using RSSI-weighted centroid.

    Requires 3+ sightings from nodes with known positions.
    Returns (x, y) or None if insufficient data.
    """
    positioned = []
    for s in sightings:
        pos = _node_positions.get(s.node_id)
        if pos is not None:
            positioned.append((pos, s.device.rssi))

    if len(positioned) < 3:
        return None

    # Convert RSSI to linear weight (higher RSSI = closer = more weight).
    # RSSI is negative; -30 is strong, -90 is weak.
    weights = []
    for pos, rssi in positioned:
        # Shift to positive and use power scale for weight
        w = 10 ** ((rssi + 100) / 20.0)
        weights.append((pos, w))

    total_w = sum(w for _, w in weights)
    if total_w == 0:
        return None

    x = sum(pos[0] * w for pos, w in weights) / total_w
    y = sum(pos[1] * w for pos, w in weights) / total_w
    return (round(x, 2), round(y, 2))
