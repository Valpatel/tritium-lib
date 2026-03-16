# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Meshtastic integration models — BLE bridge to LoRa mesh radios.

These models represent Meshtastic nodes, messages, waypoints, and
connection status.  Used by the edge firmware (BLE bridge) and the
command center (fleet map + message display).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MeshtasticConnectionType(str, Enum):
    """How the Tritium device connects to a Meshtastic radio."""
    BLE = "ble"
    SERIAL = "serial"
    TCP = "tcp"


class MeshtasticNode(BaseModel):
    """A node discovered on the Meshtastic LoRa mesh.

    Meshtastic radios broadcast node info periodically.  The Tritium BLE
    bridge collects these and forwards them via MQTT to the command center.
    """
    node_id: str  # Meshtastic node number as hex string, e.g. "!aabbccdd"
    long_name: str = ""
    short_name: str = ""
    hw_model: str = ""  # hardware model string, e.g. "TBEAM", "HELTEC_V3"
    lat: Optional[float] = None
    lng: Optional[float] = None
    alt: Optional[float] = None  # altitude in meters
    battery_level: Optional[int] = None  # 0-100 percent
    snr: Optional[float] = None  # signal-to-noise ratio in dB
    last_heard: Optional[datetime] = None

    @property
    def has_position(self) -> bool:
        """True if the node has reported a GPS position."""
        return self.lat is not None and self.lng is not None


class MeshtasticMessage(BaseModel):
    """A text message from the Meshtastic mesh.

    Messages are bridged from the LoRa mesh through BLE into the Tritium
    mesh and MQTT, extending effective range dramatically.
    """
    from_id: str  # sender node_id
    to_id: str = "^all"  # recipient node_id or "^all" for broadcast
    text: str = ""
    channel: int = 0  # Meshtastic channel index
    timestamp: Optional[datetime] = None
    hop_count: int = 0  # number of hops the message has traversed

    @property
    def is_broadcast(self) -> bool:
        """True if this message was sent to all nodes."""
        return self.to_id == "^all"


class MeshtasticWaypoint(BaseModel):
    """A shared waypoint on the Meshtastic mesh.

    Waypoints are GPS coordinates with a label that nodes can broadcast
    for shared situational awareness.
    """
    lat: float
    lng: float
    name: str = ""
    description: str = ""
    expire_time: Optional[datetime] = None  # when the waypoint expires

    @property
    def is_expired(self) -> bool:
        """True if the waypoint has passed its expiration time."""
        if self.expire_time is None:
            return False
        return datetime.now() > self.expire_time


class MeshtasticStatus(BaseModel):
    """Connection status of the Meshtastic BLE bridge.

    Published periodically in the device heartbeat so the command center
    knows whether LoRa mesh connectivity is available.
    """
    connected: bool = False
    connection_type: Optional[MeshtasticConnectionType] = None
    node_count: int = 0  # number of nodes seen on the mesh
    my_node_id: Optional[str] = None  # our node_id on the Meshtastic mesh
