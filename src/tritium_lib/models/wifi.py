# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""WiFi probe request and network models for passive device fingerprinting.

Edge nodes passively observe WiFi probe requests broadcast by nearby devices.
These probes reveal preferred SSIDs which can fingerprint device types (corporate
laptops, IoT gadgets, mobile hotspots, etc.) without any active interaction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class WiFiNetworkType(str, Enum):
    """Classification of a WiFi network by its typical usage pattern."""
    CORPORATE = "corporate"
    HOME = "home"
    HOTSPOT = "hotspot"
    IOT = "iot"
    MESH = "mesh"
    GUEST = "guest"
    PUBLIC = "public"
    UNKNOWN = "unknown"


class WiFiProbeRequest(BaseModel):
    """A single WiFi probe request observed by an edge node.

    Devices broadcast probe requests to discover known networks.  Each probe
    reveals the MAC address of the sender and (optionally) the SSID it is
    looking for.
    """
    mac: str
    ssid_probed: str = ""
    rssi: int = -100
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    channel: int = 0
    observer_id: str = ""


class WiFiNetwork(BaseModel):
    """A WiFi network (access point) observed during a scan.

    Edge nodes periodically scan visible APs and report them so that the
    command center can build a radio-frequency picture of the environment.
    """
    bssid: str
    ssid: str = ""
    rssi: int = -100
    channel: int = 0
    auth_type: str = "open"
    network_type: WiFiNetworkType = WiFiNetworkType.UNKNOWN
    observer_id: str = ""
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class WiFiFingerprint(BaseModel):
    """Aggregated WiFi fingerprint for a single device (by MAC).

    Built from accumulated probe requests and network associations over time.
    The list of probed SSIDs is a strong behavioural fingerprint — a corporate
    laptop probing for ``CORP-5G`` and ``eduroam`` looks very different from an
    IoT thermostat probing for ``SmartHome_2G``.
    """
    mac: str
    probed_ssids: list[str] = Field(default_factory=list)
    network_associations: list[str] = Field(default_factory=list)
    first_seen: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    last_seen: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    device_type_hint: str = "unknown"
    observer_id: str = ""
    probe_count: int = 0
