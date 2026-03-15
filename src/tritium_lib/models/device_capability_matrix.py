# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Device capability matrix model for the capability grid panel.

A CapabilityMatrix maps devices to their capabilities in a 2D boolean
matrix. Used by the Command Center capability grid panel to show at a
glance which devices support which features (BLE scanning, WiFi, camera,
GPS, mesh radio, acoustic, etc.). Also supports querying devices by
capability and identifying coverage gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeviceCapabilityEntry:
    """A single device's entry in the capability matrix."""
    device_id: str = ""
    device_name: str = ""
    device_type: str = ""  # e.g. "43c", "esp32-s3", "rpi4"
    online: bool = False

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "online": self.online,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DeviceCapabilityEntry:
        return cls(
            device_id=data.get("device_id", ""),
            device_name=data.get("device_name", ""),
            device_type=data.get("device_type", ""),
            online=data.get("online", False),
        )


@dataclass
class CapabilityMatrix:
    """2D boolean matrix mapping devices to capabilities.

    Attributes:
        devices: List of device entries (rows).
        capabilities: List of capability names (columns).
        matrix: 2D list of bools, matrix[device_idx][cap_idx] = True/False.
    """
    devices: list[DeviceCapabilityEntry] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    matrix: list[list[bool]] = field(default_factory=list)

    def add_device(
        self,
        device: DeviceCapabilityEntry,
        caps: Optional[list[bool]] = None,
    ) -> None:
        """Add a device row to the matrix."""
        self.devices.append(device)
        if caps is None:
            caps = [False] * len(self.capabilities)
        # Pad or truncate to match capability count
        while len(caps) < len(self.capabilities):
            caps.append(False)
        self.matrix.append(caps[: len(self.capabilities)])

    def add_capability(self, name: str, default: bool = False) -> None:
        """Add a capability column to the matrix."""
        self.capabilities.append(name)
        for row in self.matrix:
            row.append(default)

    def get_device_capabilities(self, device_id: str) -> list[str]:
        """Return list of capability names a device supports."""
        for idx, dev in enumerate(self.devices):
            if dev.device_id == device_id:
                return [
                    cap
                    for cap_idx, cap in enumerate(self.capabilities)
                    if idx < len(self.matrix) and cap_idx < len(self.matrix[idx]) and self.matrix[idx][cap_idx]
                ]
        return []

    def get_devices_with_capability(self, capability: str) -> list[str]:
        """Return device IDs that have a given capability."""
        if capability not in self.capabilities:
            return []
        cap_idx = self.capabilities.index(capability)
        return [
            dev.device_id
            for dev_idx, dev in enumerate(self.devices)
            if dev_idx < len(self.matrix) and cap_idx < len(self.matrix[dev_idx]) and self.matrix[dev_idx][cap_idx]
        ]

    def coverage_gaps(self) -> list[str]:
        """Return capabilities that no online device supports."""
        gaps = []
        for cap_idx, cap in enumerate(self.capabilities):
            has_coverage = False
            for dev_idx, dev in enumerate(self.devices):
                if dev.online and dev_idx < len(self.matrix) and cap_idx < len(self.matrix[dev_idx]) and self.matrix[dev_idx][cap_idx]:
                    has_coverage = True
                    break
            if not has_coverage:
                gaps.append(cap)
        return gaps

    @property
    def device_count(self) -> int:
        return len(self.devices)

    @property
    def capability_count(self) -> int:
        return len(self.capabilities)

    def to_dict(self) -> dict:
        return {
            "devices": [d.to_dict() for d in self.devices],
            "capabilities": self.capabilities,
            "matrix": self.matrix,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CapabilityMatrix:
        devices = [DeviceCapabilityEntry.from_dict(d) for d in data.get("devices", [])]
        return cls(
            devices=devices,
            capabilities=data.get("capabilities", []),
            matrix=data.get("matrix", []),
        )
