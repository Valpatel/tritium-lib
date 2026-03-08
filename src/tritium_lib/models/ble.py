# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""BLE presence models — used when edge nodes scan for nearby BLE devices
and report sightings to the fleet server for presence tracking."""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class BleDevice(BaseModel):
    """A BLE device discovered by one or more nodes."""
    mac: str
    rssi: int
    name: str = ""
    seen_count: int = 1
    is_known: bool = False
    first_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BleSighting(BaseModel):
    """A single BLE sighting reported by a specific node."""
    device: BleDevice
    node_id: str
    node_ip: str = ""
    node_wifi_rssi: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BlePresence(BaseModel):
    """Aggregated presence for a single BLE device across multiple nodes."""
    mac: str
    name: str = ""
    sightings: list[BleSighting] = Field(default_factory=list)
    strongest_rssi: int = -100
    node_count: int = 0


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
