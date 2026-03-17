# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""DeviceRegistry for the Tritium Addon SDK.

Tracks N devices per addon with transport, status, and metadata.
Each addon instance owns one DeviceRegistry that manages its devices.
"""

import enum
import time
from dataclasses import dataclass, field
from typing import Any


class DeviceState(enum.Enum):
    """Lifecycle state of a registered device."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class RegisteredDevice:
    """A device tracked by the registry."""

    device_id: str
    device_type: str  # e.g. "hackrf", "meshtastic", "rtl-sdr"
    transport_type: str  # "local", "mqtt", "ssh"
    state: DeviceState = DeviceState.DISCONNECTED
    metadata: dict = field(default_factory=dict)
    last_seen: float = 0.0
    error: str = ""
    transport: Any = None  # DeviceTransport instance, not serialized

    def to_dict(self) -> dict:
        """JSON-serializable representation (excludes transport object)."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "transport_type": self.transport_type,
            "state": self.state.value,
            "metadata": dict(self.metadata),
            "last_seen": self.last_seen,
            "error": self.error,
        }

    @property
    def is_local(self) -> bool:
        """Whether the device uses a local transport."""
        return self.transport_type == "local"

    @property
    def is_remote(self) -> bool:
        """Whether the device uses a remote transport."""
        return self.transport_type != "local"


class DeviceRegistry:
    """Manages devices for a single addon instance.

    Usage::

        registry = DeviceRegistry("sdr_monitor")
        registry.add_device("hackrf-001", "hackrf", transport_type="local")
        registry.set_state("hackrf-001", DeviceState.CONNECTED)
        if "hackrf-001" in registry:
            dev = registry.get_device("hackrf-001")
    """

    def __init__(self, addon_id: str) -> None:
        self.addon_id = addon_id
        self._devices: dict[str, RegisteredDevice] = {}

    # --- CRUD ---

    def add_device(
        self,
        device_id: str,
        device_type: str,
        transport_type: str = "local",
        metadata: dict | None = None,
        transport: Any = None,
    ) -> RegisteredDevice:
        """Register a new device. Raises ValueError if device_id already exists."""
        if device_id in self._devices:
            raise ValueError(f"Device '{device_id}' already registered in {self.addon_id}")
        dev = RegisteredDevice(
            device_id=device_id,
            device_type=device_type,
            transport_type=transport_type,
            metadata=metadata or {},
            transport=transport,
        )
        self._devices[device_id] = dev
        return dev

    def remove_device(self, device_id: str) -> bool:
        """Remove a device. Returns False if not found."""
        if device_id not in self._devices:
            return False
        del self._devices[device_id]
        return True

    def get_device(self, device_id: str) -> RegisteredDevice | None:
        """Look up a device by ID."""
        return self._devices.get(device_id)

    # --- Listing ---

    def list_devices(self) -> list[RegisteredDevice]:
        """Return all registered devices."""
        return list(self._devices.values())

    def list_connected(self) -> list[RegisteredDevice]:
        """Return only devices in CONNECTED state."""
        return [d for d in self._devices.values() if d.state == DeviceState.CONNECTED]

    def list_by_type(self, device_type: str) -> list[RegisteredDevice]:
        """Return devices matching a given device_type."""
        return [d for d in self._devices.values() if d.device_type == device_type]

    # --- State management ---

    def set_state(self, device_id: str, state: DeviceState, error: str = "") -> bool:
        """Update device state. Returns False if device not found."""
        dev = self._devices.get(device_id)
        if dev is None:
            return False
        dev.state = state
        dev.error = error
        return True

    def update_metadata(self, device_id: str, **kwargs: Any) -> bool:
        """Merge key-value pairs into device metadata. Returns False if not found."""
        dev = self._devices.get(device_id)
        if dev is None:
            return False
        dev.metadata.update(kwargs)
        return True

    def touch(self, device_id: str) -> bool:
        """Update last_seen to current time. Returns False if not found."""
        dev = self._devices.get(device_id)
        if dev is None:
            return False
        dev.last_seen = time.time()
        return True

    # --- Bulk transport operations ---

    async def connect_all(self) -> dict[str, bool]:
        """Connect all devices that have a transport. Returns {device_id: success}."""
        results: dict[str, bool] = {}
        for dev_id, dev in self._devices.items():
            if dev.transport is None:
                results[dev_id] = False
                continue
            dev.state = DeviceState.CONNECTING
            try:
                ok = await dev.transport.connect()
                if ok:
                    dev.state = DeviceState.CONNECTED
                    dev.last_seen = time.time()
                    dev.error = ""
                else:
                    dev.state = DeviceState.ERROR
                    dev.error = "connect() returned False"
                results[dev_id] = bool(ok)
            except Exception as exc:
                dev.state = DeviceState.ERROR
                dev.error = str(exc)
                results[dev_id] = False
        return results

    async def disconnect_all(self) -> None:
        """Disconnect all devices that have a transport."""
        for dev in self._devices.values():
            if dev.transport is not None:
                try:
                    await dev.transport.disconnect()
                except Exception:
                    pass
            dev.state = DeviceState.DISCONNECTED

    # --- Properties ---

    @property
    def device_count(self) -> int:
        """Total number of registered devices."""
        return len(self._devices)

    @property
    def connected_count(self) -> int:
        """Number of devices currently in CONNECTED state."""
        return sum(1 for d in self._devices.values() if d.state == DeviceState.CONNECTED)

    # --- Serialization ---

    def to_dict(self) -> dict:
        """Full registry state for API serialization."""
        return {
            "addon_id": self.addon_id,
            "device_count": self.device_count,
            "connected_count": self.connected_count,
            "devices": {did: dev.to_dict() for did, dev in self._devices.items()},
        }

    # --- Dunder methods ---

    def __contains__(self, device_id: object) -> bool:
        return device_id in self._devices

    def __len__(self) -> int:
        return self.device_count
