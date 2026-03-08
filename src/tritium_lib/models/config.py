# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Configuration sync models for fleet management.

Tracks desired vs reported device configuration and detects drift.
The fleet server pushes desired_config, devices report reported_config
in their heartbeats, and drift is calculated as the diff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ConfigDriftSeverity(str, Enum):
    """How serious a config drift is."""
    NONE = "none"          # No drift
    MINOR = "minor"        # Cosmetic or non-critical keys differ
    MODERATE = "moderate"  # Functional keys differ (e.g., heartbeat interval)
    CRITICAL = "critical"  # Security-relevant keys differ (e.g., server URL, certs)


@dataclass
class ConfigKey:
    """A single configuration key-value pair."""
    key: str
    value: Any
    source: str = ""  # "desired" or "reported"


@dataclass
class ConfigDrift:
    """A single configuration key that differs between desired and reported."""
    key: str
    desired_value: Any
    reported_value: Any
    severity: ConfigDriftSeverity = ConfigDriftSeverity.MINOR

    @property
    def is_missing(self) -> bool:
        """True if the key is desired but not reported at all."""
        return self.reported_value is None

    @property
    def is_extra(self) -> bool:
        """True if the key is reported but not in desired config."""
        return self.desired_value is None


@dataclass
class DeviceConfig:
    """Full device configuration state."""
    device_id: str
    desired: dict[str, Any] = field(default_factory=dict)
    reported: dict[str, Any] = field(default_factory=dict)
    last_sync: datetime | None = None
    drifts: list[ConfigDrift] = field(default_factory=list)

    @property
    def is_synced(self) -> bool:
        """True if desired and reported match."""
        return len(self.drifts) == 0

    @property
    def drift_count(self) -> int:
        return len(self.drifts)

    @property
    def max_severity(self) -> ConfigDriftSeverity:
        if not self.drifts:
            return ConfigDriftSeverity.NONE
        severities = [d.severity for d in self.drifts]
        if ConfigDriftSeverity.CRITICAL in severities:
            return ConfigDriftSeverity.CRITICAL
        if ConfigDriftSeverity.MODERATE in severities:
            return ConfigDriftSeverity.MODERATE
        return ConfigDriftSeverity.MINOR


@dataclass
class FleetConfigStatus:
    """Fleet-wide configuration sync status."""
    total_devices: int = 0
    synced_count: int = 0
    drifted_count: int = 0
    critical_drift_count: int = 0
    devices: list[DeviceConfig] = field(default_factory=list)

    @property
    def sync_ratio(self) -> float:
        if self.total_devices == 0:
            return 1.0
        return self.synced_count / self.total_devices


# Keys that are security-critical and warrant CRITICAL severity
_CRITICAL_KEYS = frozenset({
    "server_url", "mqtt_broker", "mqtt_port",
    "ca_pem", "client_crt", "client_key",
    "ota_url", "firmware_url",
})

# Keys that affect device behavior and warrant MODERATE severity
_MODERATE_KEYS = frozenset({
    "heartbeat_interval_s", "scan_interval_s",
    "wifi_ssid", "wifi_password",
    "ble_enabled", "lora_enabled",
    "sleep_enabled", "diag_enabled",
})


def classify_drift_severity(key: str) -> ConfigDriftSeverity:
    """Classify how serious a config drift is based on the key name."""
    if key in _CRITICAL_KEYS:
        return ConfigDriftSeverity.CRITICAL
    if key in _MODERATE_KEYS:
        return ConfigDriftSeverity.MODERATE
    return ConfigDriftSeverity.MINOR


def compute_config_drift(
    desired: dict[str, Any],
    reported: dict[str, Any],
) -> list[ConfigDrift]:
    """Compare desired and reported config, return list of drifts."""
    drifts = []
    all_keys = set(desired.keys()) | set(reported.keys())

    for key in sorted(all_keys):
        d_val = desired.get(key)
        r_val = reported.get(key)

        if d_val != r_val:
            drifts.append(ConfigDrift(
                key=key,
                desired_value=d_val,
                reported_value=r_val,
                severity=classify_drift_severity(key),
            ))

    return drifts


def compute_fleet_config_status(
    devices: list[dict],
) -> FleetConfigStatus:
    """Compute fleet-wide config sync status from device list.

    Each device dict should have 'device_id', 'desired_config', 'reported_config'.
    """
    status = FleetConfigStatus(total_devices=len(devices))

    for dev in devices:
        device_id = dev.get("device_id", "unknown")
        desired = dev.get("desired_config", {})
        reported = dev.get("reported_config", {})
        drifts = compute_config_drift(desired, reported)

        dc = DeviceConfig(
            device_id=device_id,
            desired=desired,
            reported=reported,
            drifts=drifts,
        )
        status.devices.append(dc)

        if dc.is_synced:
            status.synced_count += 1
        else:
            status.drifted_count += 1
            if dc.max_severity == ConfigDriftSeverity.CRITICAL:
                status.critical_drift_count += 1

    return status
