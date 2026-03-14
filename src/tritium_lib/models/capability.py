# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Device capability advertisement models.

Standardizes how edge devices advertise their compiled-in HALs and
capabilities to the command center. Each capability has a type, version,
and optional configuration, enabling SC to know exactly what each device
can do and how to configure it.

Edge devices publish a CapabilityAdvertisement on boot via MQTT topic:
    tritium/{device_id}/capabilities

SC uses this to:
  - Show device capabilities in fleet dashboard
  - Enable/disable relevant UI panels per device
  - Route commands only to devices that support them
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CapabilityType(str, Enum):
    """Standard capability types matching edge HAL names."""
    BLE_SCANNER = "ble_scanner"
    WIFI_SCANNER = "wifi_scanner"
    WIFI_PROBE = "wifi_probe"
    CAMERA = "camera"
    AUDIO = "audio"
    ACOUSTIC = "acoustic"
    GPS = "gps"
    IMU = "imu"
    MESH_ESPNOW = "mesh_espnow"
    MESH_LORA = "mesh_lora"
    MESHTASTIC = "meshtastic"
    DISPLAY = "display"
    TOUCH = "touch"
    SDCARD = "sdcard"
    RTC = "rtc"
    POWER_MGMT = "power_mgmt"
    OTA = "ota"
    COT = "cot"
    RF_MONITOR = "rf_monitor"
    CONFIG_SYNC = "config_sync"
    DIAGLOG = "diaglog"
    WEBSERVER = "webserver"
    RADIO_SCHEDULER = "radio_scheduler"
    SLEEP = "sleep"
    PROVISION = "provision"
    HEARTBEAT = "heartbeat"


class DeviceCapability(BaseModel):
    """A single capability with version and configuration metadata.

    Richer than a boolean flag: includes what version of the HAL is
    compiled in and any relevant configuration parameters.
    """
    cap_type: CapabilityType
    version: str = "1.0"
    enabled: bool = True
    config: dict = Field(default_factory=dict)
    # Optional human-readable description
    description: str = ""

    def to_summary(self) -> str:
        """Short summary string for display."""
        status = "ON" if self.enabled else "OFF"
        return f"{self.cap_type.value} v{self.version} [{status}]"


class CapabilityAdvertisement(BaseModel):
    """Full capability advertisement from an edge device.

    Published on boot and whenever capabilities change (e.g., after
    a radio scheduler switches modes).
    """
    device_id: str
    board: str = "unknown"
    firmware_version: str = "unknown"
    capabilities: list[DeviceCapability] = Field(default_factory=list)
    timestamp: Optional[int] = None

    def has_capability(self, cap_type: CapabilityType) -> bool:
        """Check if device has a specific capability."""
        return any(
            c.cap_type == cap_type and c.enabled
            for c in self.capabilities
        )

    def get_capability(self, cap_type: CapabilityType) -> Optional[DeviceCapability]:
        """Get a specific capability if present and enabled."""
        for c in self.capabilities:
            if c.cap_type == cap_type and c.enabled:
                return c
        return None

    def capability_types(self) -> list[str]:
        """List of enabled capability type strings (for heartbeat compat)."""
        return sorted(
            c.cap_type.value for c in self.capabilities if c.enabled
        )

    def to_heartbeat_list(self) -> list[str]:
        """Convert to simple list format compatible with DeviceCapabilities.from_list()."""
        # Map capability types to the short names used in DeviceCapabilities
        _type_to_short = {
            CapabilityType.BLE_SCANNER: "ble",
            CapabilityType.WIFI_SCANNER: "wifi",
            CapabilityType.CAMERA: "camera",
            CapabilityType.AUDIO: "audio",
            CapabilityType.GPS: "gps",
            CapabilityType.IMU: "imu",
            CapabilityType.MESH_ESPNOW: "mesh",
            CapabilityType.MESH_LORA: "lora",
            CapabilityType.DISPLAY: "display",
            CapabilityType.TOUCH: "touch",
            CapabilityType.RTC: "rtc",
            CapabilityType.POWER_MGMT: "power",
        }
        result = []
        for c in self.capabilities:
            if c.enabled:
                short = _type_to_short.get(c.cap_type, c.cap_type.value)
                result.append(short)
        return sorted(result)
