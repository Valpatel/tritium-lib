# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""WiFi probe request and network models for passive device fingerprinting.

Edge nodes passively observe WiFi probe requests broadcast by nearby devices.
These probes reveal preferred SSIDs which can fingerprint device types (corporate
laptops, IoT gadgets, mobile hotspots, etc.) without any active interaction.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

_MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


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

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "mac": "AA:BB:CC:DD:EE:FF",
                    "ssid_probed": "MyNetwork",
                    "rssi": -65,
                    "channel": 6,
                    "observer_id": "esp32-001",
                }
            ]
        }
    )

    mac: str
    ssid_probed: str = ""
    rssi: int = Field(-100, ge=-127, le=0)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    channel: int = Field(0, ge=0, le=196)
    observer_id: str = ""

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
        ssid = self.ssid_probed or "(broadcast)"
        return f"Probe {self.mac} -> {ssid} rssi={self.rssi} ch={self.channel}"


class WiFiNetwork(BaseModel):
    """A WiFi network (access point) observed during a scan.

    Edge nodes periodically scan visible APs and report them so that the
    command center can build a radio-frequency picture of the environment.
    """
    bssid: str
    ssid: str = ""
    rssi: int = Field(-100, ge=-127, le=0)
    channel: int = Field(0, ge=0, le=196)
    auth_type: str = "open"
    network_type: WiFiNetworkType = WiFiNetworkType.UNKNOWN
    observer_id: str = ""
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @field_validator("bssid")
    @classmethod
    def _validate_bssid(cls, v: str) -> str:
        v = v.upper()
        if not _MAC_RE.match(v):
            raise ValueError(
                f"Invalid BSSID '{v}' — expected format AA:BB:CC:DD:EE:FF"
            )
        return v

    def to_summary(self) -> str:
        """Human-readable one-line summary."""
        ssid = self.ssid or "(hidden)"
        return (
            f"AP {self.bssid} {ssid} rssi={self.rssi} ch={self.channel} "
            f"auth={self.auth_type} type={self.network_type.value}"
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
    probe_count: int = Field(0, ge=0)

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
        ssids = ", ".join(self.probed_ssids[:3])
        if len(self.probed_ssids) > 3:
            ssids += f" (+{len(self.probed_ssids) - 3} more)"
        return (
            f"Fingerprint {self.mac} type={self.device_type_hint} "
            f"probes={self.probe_count} ssids=[{ssids}]"
        )
