# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Device provisioning models for fleet management.

Covers the full device lifecycle: discovery, commissioning, configuration,
and decommissioning. Supports 5 commissioning paths: Web Portal, BLE,
USB Serial, SD Card, and Peer Seeding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class ProvisionSource(str, Enum):
    """How a device was commissioned."""
    WEB_PORTAL = "web_portal"    # Captive portal AP mode
    BLE = "ble"                  # BLE GATT provisioning
    USB_SERIAL = "usb_serial"    # USB serial console
    SD_CARD = "sd_card"          # SD card config file
    PEER_SEED = "peer_seed"      # Peer-to-peer seeding
    MANUAL = "manual"            # Manual API registration
    AUTO = "auto"                # Auto-discovered via heartbeat


class ProvisionState(str, Enum):
    """Lifecycle state of a device."""
    DISCOVERED = "discovered"      # Seen but not commissioned
    PENDING = "pending"            # Awaiting approval
    COMMISSIONED = "commissioned"  # Active in fleet
    SUSPENDED = "suspended"        # Temporarily deactivated
    DECOMMISSIONED = "decommissioned"  # Removed from fleet


@dataclass
class ProvisionData:
    """Configuration data pushed to a device during commissioning."""
    wifi_ssid: Optional[str] = None
    wifi_password: Optional[str] = None
    server_url: Optional[str] = None
    device_id: Optional[str] = None
    device_name: Optional[str] = None
    mqtt_broker: Optional[str] = None
    ca_pem: Optional[str] = None          # TLS certificate
    extra_config: dict[str, Any] = field(default_factory=dict)

    @property
    def has_wifi(self) -> bool:
        return bool(self.wifi_ssid)

    @property
    def has_tls(self) -> bool:
        return bool(self.ca_pem)


@dataclass
class ProvisionRecord:
    """Record of a device's provisioning event."""
    device_id: str
    source: ProvisionSource
    state: ProvisionState
    provisioned_at: Optional[datetime] = None
    provisioned_by: Optional[str] = None  # User or system that performed it
    data: Optional[ProvisionData] = None
    mac_address: Optional[str] = None
    board_type: Optional[str] = None
    firmware_version: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.state == ProvisionState.COMMISSIONED

    @property
    def is_pending(self) -> bool:
        return self.state in (ProvisionState.DISCOVERED, ProvisionState.PENDING)


@dataclass
class FleetProvisionStatus:
    """Fleet-wide provisioning summary."""
    total_devices: int
    commissioned: int
    pending: int
    discovered: int
    suspended: int
    decommissioned: int
    devices: list[ProvisionRecord] = field(default_factory=list)

    @property
    def active_ratio(self) -> float:
        if self.total_devices == 0:
            return 1.0
        return self.commissioned / self.total_devices

    @property
    def needs_attention(self) -> int:
        """Devices requiring admin action (pending + discovered)."""
        return self.pending + self.discovered


def compute_provision_status(
    records: list[ProvisionRecord],
) -> FleetProvisionStatus:
    """Aggregate provisioning status across the fleet."""
    counts = {state: 0 for state in ProvisionState}
    for r in records:
        counts[r.state] = counts.get(r.state, 0) + 1

    return FleetProvisionStatus(
        total_devices=len(records),
        commissioned=counts[ProvisionState.COMMISSIONED],
        pending=counts[ProvisionState.PENDING],
        discovered=counts[ProvisionState.DISCOVERED],
        suspended=counts[ProvisionState.SUSPENDED],
        decommissioned=counts[ProvisionState.DECOMMISSIONED],
        devices=records,
    )


def validate_provision_data(data: ProvisionData) -> list[str]:
    """Validate provisioning data, return list of issues (empty = valid)."""
    issues = []
    if data.server_url and not (
        data.server_url.startswith("http://") or data.server_url.startswith("https://")
    ):
        issues.append("server_url must start with http:// or https://")
    if data.wifi_password and len(data.wifi_password) < 8:
        issues.append("wifi_password must be at least 8 characters")
    if data.ca_pem and "BEGIN CERTIFICATE" not in data.ca_pem:
        issues.append("ca_pem does not look like a PEM certificate")
    return issues
