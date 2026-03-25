# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Device models — shared between tritium-edge (fleet management) and
tritium-sc (sensor node integration)."""

import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Canonical MAC address pattern: AA:BB:CC:DD:EE:FF (case-insensitive)
_MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")
_VALID_STATUSES = {"online", "offline", "updating", "error"}


class DeviceCapabilities(BaseModel):
    """What a device can do — reported in heartbeat, drives server UI."""
    ble: bool = False
    wifi: bool = False
    camera: bool = False
    audio: bool = False
    imu: bool = False
    display: bool = False
    touch: bool = False
    rtc: bool = False
    power: bool = False
    mesh: bool = False
    lora: bool = False
    gps: bool = False
    temperature: bool = False
    humidity: bool = False
    # Extensible: plugins can add custom capabilities
    custom: dict[str, bool] = Field(default_factory=dict)

    @classmethod
    def from_list(cls, caps: list[str]) -> "DeviceCapabilities":
        """Create from a list of capability strings (heartbeat format)."""
        known = cls.model_fields.keys() - {"custom"}
        kwargs = {}
        custom = {}
        for cap in caps:
            if cap in known:
                kwargs[cap] = True
            else:
                custom[cap] = True
        return cls(custom=custom, **kwargs)

    def to_list(self) -> list[str]:
        """Convert to list of capability strings for heartbeat."""
        caps = [k for k, v in self.model_dump().items()
                if k != "custom" and v is True]
        caps.extend(k for k, v in self.custom.items() if v)
        return sorted(caps)


class DeviceHeartbeat(BaseModel):
    """Heartbeat payload from device — v2 protocol.

    Used by both tritium-edge (fleet server) and tritium-sc (sensor node).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "device_id": "esp32-001",
                    "firmware_version": "1.2.3",
                    "board": "touch-lcd-43c-box",
                    "family": "esp32",
                    "uptime_s": 3600,
                    "free_heap": 180000,
                    "wifi_rssi": -55,
                    "capabilities": ["camera", "imu"],
                }
            ]
        }
    )

    device_id: str = Field(..., min_length=1)
    device_token: Optional[str] = None
    firmware_version: str = "unknown"
    firmware_hash: Optional[str] = None
    board: str = "unknown"
    family: str = "esp32"
    uptime_s: Optional[int] = Field(None, ge=0)
    free_heap: Optional[int] = Field(None, ge=0)
    wifi_rssi: Optional[int] = Field(None, ge=-127, le=0)
    ip_address: Optional[str] = None
    boot_count: Optional[int] = Field(None, ge=0)
    reported_config: Optional[dict] = None
    capabilities: list[str] = Field(default_factory=list)
    ota_status: Optional[str] = None
    ota_result: Optional[dict] = None
    command_acks: list[dict] = Field(default_factory=list)
    mesh_peers: Optional[int] = Field(None, ge=0)
    timestamp: Optional[int] = None
    device_group: str = ""  # perimeter, interior, mobile, reserve

    @field_validator("ip_address")
    @classmethod
    def _validate_ip(cls, v: Optional[str]) -> Optional[str]:
        """Basic IPv4 format check when provided."""
        if v is None or v == "":
            return v
        parts = v.split(".")
        if len(parts) != 4:
            raise ValueError(f"Invalid IPv4 address: {v}")
        for p in parts:
            if not p.isdigit() or not 0 <= int(p) <= 255:
                raise ValueError(f"Invalid IPv4 address: {v}")
        return v

    def to_summary(self) -> str:
        """Human-readable one-line summary."""
        caps = ", ".join(self.capabilities) if self.capabilities else "none"
        rssi = f"{self.wifi_rssi} dBm" if self.wifi_rssi is not None else "n/a"
        return (
            f"[{self.device_id}] {self.board} fw={self.firmware_version} "
            f"up={self.uptime_s}s heap={self.free_heap} rssi={rssi} caps=[{caps}]"
        )


class Device(BaseModel):
    """Device record — stored by fleet server, consumed by both systems."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "device_id": "esp32-001",
                    "device_name": "Front Door Sensor",
                    "mac": "AA:BB:CC:DD:EE:FF",
                    "board": "touch-lcd-43c-box",
                    "status": "online",
                    "capabilities": ["camera", "ble", "wifi"],
                }
            ]
        }
    )

    device_id: str = Field(..., min_length=1)
    device_name: str = ""
    mac: str = ""
    board: str = "unknown"
    family: str = "esp32"
    firmware_version: str = "unknown"
    firmware_hash: Optional[str] = None
    ip_address: Optional[str] = None
    capabilities: list[str] = Field(default_factory=list)
    status: str = "offline"  # online, offline, updating, error
    last_seen: Optional[datetime] = None
    registered_at: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("mac")
    @classmethod
    def _validate_mac(cls, v: str) -> str:
        """Validate MAC address format when provided."""
        if v == "":
            return v
        v = v.upper()
        if not _MAC_RE.match(v):
            raise ValueError(
                f"Invalid MAC address '{v}' — expected format AA:BB:CC:DD:EE:FF"
            )
        return v

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        """Ensure status is one of the allowed values."""
        if v not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{v}' — must be one of {sorted(_VALID_STATUSES)}"
            )
        return v

    @field_validator("ip_address")
    @classmethod
    def _validate_ip(cls, v: Optional[str]) -> Optional[str]:
        """Basic IPv4 format check when provided."""
        if v is None or v == "":
            return v
        parts = v.split(".")
        if len(parts) != 4:
            raise ValueError(f"Invalid IPv4 address: {v}")
        for p in parts:
            if not p.isdigit() or not 0 <= int(p) <= 255:
                raise ValueError(f"Invalid IPv4 address: {v}")
        return v

    def to_summary(self) -> str:
        """Human-readable one-line summary."""
        name = self.device_name or self.device_id
        mac_str = f" mac={self.mac}" if self.mac else ""
        return (
            f"[{name}] {self.board} fw={self.firmware_version} "
            f"status={self.status}{mac_str}"
        )


class DeviceGroup(BaseModel):
    """A named group of devices with shared configuration."""
    id: str
    name: str
    devices: list[str] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
