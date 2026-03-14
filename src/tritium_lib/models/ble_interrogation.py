# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""BLE GATT interrogation models — active device profiling via GATT connections.

When an edge node connects to an unknown BLE device and reads its GATT
services/characteristics, these models carry the results through the
system for enrichment and classification.
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


# Standard BLE 16-bit service UUID lookup
STANDARD_SERVICE_UUIDS: dict[int, str] = {
    0x1800: "Generic Access (GAP)",
    0x1801: "Generic Attribute (GATT)",
    0x1802: "Immediate Alert",
    0x1803: "Link Loss",
    0x1804: "TX Power",
    0x1805: "Current Time",
    0x1808: "Glucose",
    0x1809: "Health Thermometer",
    0x180A: "Device Information",
    0x180D: "Heart Rate",
    0x180F: "Battery Service",
    0x1810: "Blood Pressure",
    0x1811: "Alert Notification",
    0x1812: "Human Interface Device (HID)",
    0x1813: "Scan Parameters",
    0x1814: "Running Speed and Cadence",
    0x1815: "Automation IO",
    0x1816: "Cycling Speed and Cadence",
    0x1818: "Cycling Power",
    0x1819: "Location and Navigation",
    0x181A: "Environmental Sensing",
    0x181C: "User Data",
    0x181D: "Weight Scale",
    0x181E: "Bond Management",
    0x1820: "Internet Protocol Support",
    0x1821: "Indoor Positioning",
    0x1822: "Pulse Oximeter",
    0x1823: "HTTP Proxy",
    0x1824: "Transport Discovery",
    0x1825: "Object Transfer",
    0x1826: "Fitness Machine",
    0x1827: "Mesh Provisioning",
    0x1828: "Mesh Proxy",
    0x1829: "Reconnection Configuration",
    0x183A: "Insulin Delivery",
    0x183B: "Binary Sensor",
    0x183C: "Emergency Configuration",
    0x183E: "Physical Activity Monitor",
    0x1843: "Audio Input Control",
    0x1844: "Volume Control",
    0x1845: "Volume Offset Control",
    0x1846: "Coordinated Set Identification",
    0x1848: "Media Control",
    0x1849: "Generic Media Control",
    0x184A: "Constant Tone Extension",
    0x184B: "Telephone Bearer",
    0x184C: "Generic Telephone Bearer",
    0x184D: "Microphone Control",
    0x184E: "Audio Stream Control",
    0x184F: "Broadcast Audio Scan",
    0x1850: "Published Audio Capabilities",
    0x1851: "Basic Audio Announcement",
    0x1852: "Broadcast Audio Announcement",
    0x1853: "Common Audio",
    0x1854: "Hearing Access",
    0x1856: "Public Broadcast Announcement",
    0x1858: "Gaming Audio",
    0xFE2C: "Google Fast Pair",
    0xFEAA: "Google Eddystone",
    0xFEED: "Tile Tracker",
    0xFD6F: "Apple Exposure Notification",
}


def lookup_service_name(uuid: int) -> str:
    """Look up the human-readable name of a 16-bit BLE service UUID.

    Returns the name if known, or 'Unknown (0xNNNN)' if not recognized.
    """
    name = STANDARD_SERVICE_UUIDS.get(uuid)
    if name:
        return name
    return f"Unknown (0x{uuid:04X})"


class BleGATTCharacteristic(BaseModel):
    """A single GATT characteristic discovered during interrogation."""
    uuid: str = Field(..., description="UUID string (16-bit as '0xNNNN' or full 128-bit)")
    name: str = Field("", description="Human-readable name if known")
    value: Optional[str] = Field(None, description="Read value as string, if readable")
    properties: list[str] = Field(
        default_factory=list,
        description="Characteristic properties: read, write, notify, indicate, etc.",
    )


class BleGATTService(BaseModel):
    """A single GATT service discovered during interrogation."""
    uuid: str = Field(..., description="UUID string (16-bit as '0xNNNN' or full 128-bit)")
    uuid16: Optional[int] = Field(None, description="16-bit UUID if standard, else None")
    name: str = Field("", description="Human-readable service name")
    is_standard: bool = Field(False, description="True if this is a recognized standard service")
    characteristics: list[BleGATTCharacteristic] = Field(
        default_factory=list,
        description="Characteristics discovered under this service",
    )


class BleDeviceProfile(BaseModel):
    """Complete device profile obtained via active GATT interrogation.

    This is the richest classification data possible -- the device's
    own self-description read directly from its GATT services.
    """
    mac: str = Field(..., description="MAC address AA:BB:CC:DD:EE:FF")
    addr_type: int = Field(0, description="Address type: 0=public, 1=random")

    # Discovered services
    services: list[BleGATTService] = Field(
        default_factory=list,
        description="All GATT services discovered on the device",
    )

    # Device Information Service (0x180A) fields
    manufacturer: str = Field("", description="Manufacturer Name String")
    model: str = Field("", description="Model Number String")
    firmware_rev: str = Field("", description="Firmware Revision String")
    hardware_rev: str = Field("", description="Hardware Revision String")
    software_rev: str = Field("", description="Software Revision String")
    serial_number: str = Field("", description="Serial Number String")

    # GAP (0x1800)
    device_name: str = Field("", description="GAP Device Name")
    appearance: Optional[int] = Field(None, description="GAP Appearance value (16-bit)")

    # Battery Service (0x180F)
    battery_level: Optional[int] = Field(
        None, description="Battery level 0-100, None if not available"
    )

    # Metadata
    interrogated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the interrogation completed",
    )
    connection_duration_ms: int = Field(
        0, description="How long the GATT connection took in milliseconds"
    )

    @property
    def has_device_info(self) -> bool:
        """True if any Device Information Service fields were populated."""
        return bool(
            self.manufacturer or self.model or self.firmware_rev
            or self.hardware_rev or self.software_rev or self.serial_number
        )

    @property
    def service_uuids_16bit(self) -> list[int]:
        """Return list of 16-bit service UUIDs found."""
        return [s.uuid16 for s in self.services if s.uuid16 is not None]

    @property
    def service_names(self) -> list[str]:
        """Return list of human-readable service names."""
        return [s.name for s in self.services if s.name]

    def has_service(self, uuid16: int) -> bool:
        """Check if a specific 16-bit service UUID was discovered."""
        return uuid16 in self.service_uuids_16bit

    def to_enrichment_dict(self) -> dict:
        """Convert to a dict suitable for target dossier enrichment."""
        result: dict = {
            "mac": self.mac,
            "source": "gatt_interrogation",
            "interrogated_at": self.interrogated_at.isoformat(),
            "connection_duration_ms": self.connection_duration_ms,
            "services": [s.name or s.uuid for s in self.services],
            "service_uuids": [s.uuid for s in self.services],
        }
        if self.manufacturer:
            result["manufacturer"] = self.manufacturer
        if self.model:
            result["model"] = self.model
        if self.firmware_rev:
            result["firmware_rev"] = self.firmware_rev
        if self.hardware_rev:
            result["hardware_rev"] = self.hardware_rev
        if self.software_rev:
            result["software_rev"] = self.software_rev
        if self.serial_number:
            result["serial_number"] = self.serial_number
        if self.device_name:
            result["device_name"] = self.device_name
        if self.appearance is not None:
            result["appearance"] = self.appearance
        if self.battery_level is not None:
            result["battery_level"] = self.battery_level
        return result


class BleInterrogationResult(BaseModel):
    """Result of a BLE interrogation attempt — success or failure."""
    mac: str = Field(..., description="Target MAC address")
    success: bool = Field(..., description="True if interrogation succeeded")
    profile: Optional[BleDeviceProfile] = Field(
        None, description="Device profile if successful"
    )
    error: str = Field("", description="Error message if failed")
    duration_ms: int = Field(0, description="Total attempt duration in milliseconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the attempt completed",
    )
    node_id: str = Field("", description="Edge node that performed the interrogation")


class BleInterrogationQueue(BaseModel):
    """Status of the interrogation queue on an edge node."""
    pending: list[str] = Field(
        default_factory=list, description="MAC addresses waiting to be interrogated"
    )
    completed: int = Field(0, description="Total completed interrogations")
    failed: int = Field(0, description="Total failed interrogations")
    on_cooldown: int = Field(0, description="MACs currently on cooldown")
    active: bool = Field(False, description="Whether the interrogator is running")


def classify_device_from_profile(profile: BleDeviceProfile) -> str:
    """Classify a device type from its GATT profile.

    Uses service UUIDs, manufacturer name, and model to determine
    the most likely device category.

    Returns a device type string like 'phone', 'watch', 'headphones', etc.
    """
    uuids = set(profile.service_uuids_16bit)
    mfr = profile.manufacturer.lower()
    model = profile.model.lower()
    name = profile.device_name.lower()

    # Heart rate + running/cycling → fitness tracker or watch
    if 0x180D in uuids:
        if 0x1814 in uuids or 0x1816 in uuids or 0x1818 in uuids:
            return "fitness_tracker"
        return "watch"

    # HID → keyboard, mouse, or gamepad
    if 0x1812 in uuids:
        if any(k in name for k in ("keyboard", "kb")):
            return "keyboard"
        if any(k in name for k in ("mouse", "trackpad")):
            return "mouse"
        if any(k in name for k in ("gamepad", "controller", "joystick")):
            return "gamepad"
        return "peripheral"

    # Blood pressure, glucose, health thermometer → medical
    if uuids & {0x1810, 0x1808, 0x1809}:
        return "medical_device"

    # Weight scale
    if 0x181D in uuids:
        return "scale"

    # Environmental sensing
    if 0x181A in uuids:
        return "environmental_sensor"

    # Mesh provisioning/proxy → mesh device
    if uuids & {0x1827, 0x1828}:
        return "mesh_device"

    # Audio services
    if uuids & {0x1843, 0x1844, 0x184E, 0x184F, 0x1850, 0x1851}:
        if any(k in name for k in ("buds", "pods", "airpod", "earphone", "earbud")):
            return "earbuds"
        if any(k in name for k in ("speaker", "soundbar", "boom")):
            return "speaker"
        return "headphones"

    # Fitness machine
    if 0x1826 in uuids:
        return "fitness_machine"

    # Location and navigation
    if 0x1819 in uuids:
        return "gps_device"

    # Manufacturer-based classification
    if "apple" in mfr:
        if "watch" in model or "watch" in name:
            return "watch"
        if "iphone" in model or "iphone" in name:
            return "phone"
        if "ipad" in model or "ipad" in name:
            return "tablet"
        if "macbook" in model or "mac" in name:
            return "laptop"
        if "airpods" in model or "airpods" in name:
            return "earbuds"
        if "pencil" in model or "pencil" in name:
            return "stylus"
        return "apple_device"

    if "samsung" in mfr:
        if "buds" in model or "buds" in name:
            return "earbuds"
        if "watch" in model or "watch" in name:
            return "watch"
        if "galaxy tab" in model:
            return "tablet"
        if "galaxy" in model:
            return "phone"
        return "samsung_device"

    if "fitbit" in mfr or "fitbit" in name:
        return "fitness_tracker"

    if "garmin" in mfr or "garmin" in name:
        return "fitness_tracker"

    if "tile" in mfr or "tile" in name:
        return "tracker"

    if any(k in mfr for k in ("bose", "sony", "jabra", "jbl", "beats", "sennheiser")):
        return "headphones"

    # Fallback: if battery service only, likely a simple peripheral
    if uuids == {0x1800, 0x1801, 0x180F}:
        return "simple_peripheral"

    return "unknown"
