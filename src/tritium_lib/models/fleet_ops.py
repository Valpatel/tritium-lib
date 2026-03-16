# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fleet operations models — group commands, config templates, analytics.

Standardizes fleet-level management operations across tritium-sc and
tritium-edge. Used by the fleet coordination API and edge group command
handler.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FleetCommandType(str, Enum):
    """Types of commands that can be sent to device groups."""
    REBOOT = "reboot"
    SCAN_BURST = "scan_burst"
    INCREASE_RATE = "increase_rate"
    DECREASE_RATE = "decrease_rate"
    OTA_UPDATE = "ota_update"
    APPLY_TEMPLATE = "apply_template"
    SET_GROUP = "set_group"
    IDENTIFY = "identify"
    SLEEP = "sleep"


class FleetCommandStatus(str, Enum):
    """Status of a fleet-wide command."""
    PENDING = "pending"
    BROADCASTING = "broadcasting"
    PARTIAL = "partial"
    COMPLETE = "complete"
    FAILED = "failed"
    EXPIRED = "expired"


class FleetCommand(BaseModel):
    """A command targeted at a group of devices.

    Sent via MQTT broadcast to all devices in the target_group.
    Each device verifies its group membership before executing.
    """
    id: str
    command_type: FleetCommandType
    target_group: str
    payload: dict = Field(default_factory=dict)
    status: FleetCommandStatus = FleetCommandStatus.PENDING
    created_at: Optional[datetime] = None
    broadcast_at: Optional[datetime] = None
    expected_targets: int = 0
    acked_targets: int = 0
    failed_targets: int = 0
    ack_device_ids: list[str] = Field(default_factory=list)
    fail_device_ids: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    expires_at: Optional[datetime] = None


class ConfigTemplateName(str, Enum):
    """Built-in configuration template names."""
    PERIMETER_HIGH_SECURITY = "perimeter_high_security"
    INDOOR_NORMAL = "indoor_normal"
    POWER_SAVER_MOBILE = "power_saver_mobile"
    CUSTOM = "custom"


class ConfigTemplate(BaseModel):
    """A named configuration template for edge devices.

    Templates define scan intervals, report rates, and power modes.
    Can be applied to device groups via fleet commands.
    """
    id: str
    name: str
    description: str = ""
    template_type: ConfigTemplateName = ConfigTemplateName.CUSTOM
    # Scan intervals in milliseconds
    ble_scan_interval_ms: int = 10000
    wifi_scan_interval_ms: int = 30000
    # Report rates in milliseconds
    heartbeat_interval_ms: int = 30000
    sighting_interval_ms: int = 15000
    # Power mode: "normal", "low_power", "high_performance"
    power_mode: str = "normal"
    # Additional settings
    settings: dict = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# Built-in templates
BUILTIN_TEMPLATES: dict[str, ConfigTemplate] = {
    "perimeter_high_security": ConfigTemplate(
        id="perimeter_high_security",
        name="Perimeter - High Security",
        description="Fast scan rates for perimeter nodes. Maximum detection speed.",
        template_type=ConfigTemplateName.PERIMETER_HIGH_SECURITY,
        ble_scan_interval_ms=5000,
        wifi_scan_interval_ms=15000,
        heartbeat_interval_ms=15000,
        sighting_interval_ms=5000,
        power_mode="high_performance",
    ),
    "indoor_normal": ConfigTemplate(
        id="indoor_normal",
        name="Indoor - Normal",
        description="Balanced scan rates for indoor monitoring.",
        template_type=ConfigTemplateName.INDOOR_NORMAL,
        ble_scan_interval_ms=10000,
        wifi_scan_interval_ms=30000,
        heartbeat_interval_ms=30000,
        sighting_interval_ms=15000,
        power_mode="normal",
    ),
    "power_saver_mobile": ConfigTemplate(
        id="power_saver_mobile",
        name="Power Saver - Mobile",
        description="Reduced scan rates for battery-powered mobile nodes.",
        template_type=ConfigTemplateName.POWER_SAVER_MOBILE,
        ble_scan_interval_ms=30000,
        wifi_scan_interval_ms=60000,
        heartbeat_interval_ms=60000,
        sighting_interval_ms=30000,
        power_mode="low_power",
    ),
}


class DeviceUptimeRecord(BaseModel):
    """A single uptime data point for a device."""
    device_id: str
    timestamp: float
    uptime_s: int = 0
    online: bool = True


class SightingRateRecord(BaseModel):
    """Sighting rate data point for a device."""
    device_id: str
    timestamp: float
    ble_rate: float = 0.0  # sightings per minute
    wifi_rate: float = 0.0


class CoveragePoint(BaseModel):
    """A point in the coverage map showing sensor overlap."""
    lat: float
    lng: float
    sensor_count: int = 0
    device_ids: list[str] = Field(default_factory=list)


class FleetAnalyticsSnapshot(BaseModel):
    """A snapshot of fleet-wide analytics.

    Computed from heartbeat history and device positions.
    Used by the fleet analytics dashboard.
    """
    timestamp: float
    total_devices: int = 0
    online_devices: int = 0
    offline_devices: int = 0
    avg_uptime_s: float = 0.0
    avg_battery_pct: Optional[float] = None
    total_ble_sightings: int = 0
    total_wifi_sightings: int = 0
    uptime_records: list[DeviceUptimeRecord] = Field(default_factory=list)
    sighting_rates: list[SightingRateRecord] = Field(default_factory=list)
    coverage_points: list[CoveragePoint] = Field(default_factory=list)
    groups: dict[str, int] = Field(default_factory=dict)  # group_name -> device_count
