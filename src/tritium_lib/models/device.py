# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Device models — shared between tritium-edge (fleet management) and
tritium-sc (sensor node integration)."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


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
    device_id: str
    device_token: Optional[str] = None
    firmware_version: str = "unknown"
    firmware_hash: Optional[str] = None
    board: str = "unknown"
    family: str = "esp32"
    uptime_s: Optional[int] = None
    free_heap: Optional[int] = None
    wifi_rssi: Optional[int] = None
    ip_address: Optional[str] = None
    boot_count: Optional[int] = None
    reported_config: Optional[dict] = None
    capabilities: list[str] = Field(default_factory=list)
    ota_status: Optional[str] = None
    ota_result: Optional[dict] = None
    command_acks: list[dict] = Field(default_factory=list)
    mesh_peers: Optional[int] = None
    timestamp: Optional[int] = None
    device_group: str = ""  # perimeter, interior, mobile, reserve


class Device(BaseModel):
    """Device record — stored by fleet server, consumed by both systems."""
    device_id: str
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


class DeviceGroup(BaseModel):
    """A named group of devices with shared configuration."""
    id: str
    name: str
    devices: list[str] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
