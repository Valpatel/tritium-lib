# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fleet device management — register, heartbeat, group, command, monitor.

Builds on the data models in ``tritium_lib.models.device`` and
``tritium_lib.models.fleet`` to provide pure-logic management of a fleet
of edge devices (ESP32 nodes, cameras, sensors).  No network I/O — callers
feed in heartbeats and read back state.

Quick start::

    from tritium_lib.fleet import FleetManager

    mgr = FleetManager()
    mgr.register("esp32-001", device_type="esp32-s3", capabilities=["ble", "wifi"])
    mgr.heartbeat("esp32-001", firmware_version="1.2.0", wifi_rssi=-48, free_heap=200_000)
    stale = mgr.monitor.check_stale(timeout_s=60)
    mgr.commands.enqueue("esp32-001", "reboot")
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional


# ---------------------------------------------------------------------------
# FleetDevice
# ---------------------------------------------------------------------------

class DeviceStatus(str, Enum):
    """Connectivity status of a fleet device."""
    ONLINE = "online"
    STALE = "stale"
    OFFLINE = "offline"
    UPDATING = "updating"
    ERROR = "error"


@dataclass
class FleetDevice:
    """A single edge device in the fleet.

    Tracks identity, health telemetry, capabilities, group membership,
    and connectivity status.  Designed to be updated from heartbeat
    payloads arriving over MQTT or REST.
    """
    device_id: str
    device_type: str = "esp32"
    device_name: str = ""
    status: DeviceStatus = DeviceStatus.OFFLINE
    capabilities: list[str] = field(default_factory=list)
    group: str = ""
    tags: list[str] = field(default_factory=list)

    # Telemetry (updated on heartbeat)
    firmware_version: str = "unknown"
    ip_address: str = ""
    wifi_rssi: int = 0
    free_heap: int = 0
    uptime_s: int = 0
    battery_pct: Optional[float] = None

    # Timestamps
    registered_at: float = 0.0
    last_heartbeat: float = 0.0

    # Metadata bag for caller-defined extras
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- helpers ----------------------------------------------------------

    @property
    def is_online(self) -> bool:
        return self.status == DeviceStatus.ONLINE

    @property
    def has_capability(self) -> bool:
        """True if the device has at least one capability."""
        return len(self.capabilities) > 0

    def has_cap(self, cap: str) -> bool:
        """Check if device has a specific capability."""
        return cap in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "device_name": self.device_name,
            "status": self.status.value,
            "capabilities": list(self.capabilities),
            "group": self.group,
            "tags": list(self.tags),
            "firmware_version": self.firmware_version,
            "ip_address": self.ip_address,
            "wifi_rssi": self.wifi_rssi,
            "free_heap": self.free_heap,
            "uptime_s": self.uptime_s,
            "battery_pct": self.battery_pct,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# DeviceGroup
# ---------------------------------------------------------------------------

@dataclass
class DeviceGroup:
    """Logical grouping of fleet devices.

    Devices are grouped by location, type, or mission.  Groups can have
    shared configuration applied via ``CommandQueue``.
    """
    group_id: str
    name: str = ""
    description: str = ""
    device_ids: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    @property
    def size(self) -> int:
        return len(self.device_ids)

    def add(self, device_id: str) -> bool:
        """Add a device.  Returns False if already present."""
        if device_id in self.device_ids:
            return False
        self.device_ids.append(device_id)
        return True

    def remove(self, device_id: str) -> bool:
        """Remove a device.  Returns False if not present."""
        if device_id not in self.device_ids:
            return False
        self.device_ids.remove(device_id)
        return True

    def __contains__(self, device_id: object) -> bool:
        return device_id in self.device_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "description": self.description,
            "device_ids": list(self.device_ids),
            "config": dict(self.config),
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# CommandQueue
# ---------------------------------------------------------------------------

class CommandPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class CommandStatus(str, Enum):
    QUEUED = "queued"
    DISPATCHED = "dispatched"
    ACKED = "acked"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class QueuedCommand:
    """A command waiting to be dispatched to a device."""
    command_id: str
    device_id: str
    command_type: str  # reboot, config_update, ota, scan_burst, identify, sleep
    payload: dict[str, Any] = field(default_factory=dict)
    priority: CommandPriority = CommandPriority.NORMAL
    status: CommandStatus = CommandStatus.QUEUED
    created_at: float = 0.0
    dispatched_at: float = 0.0
    acked_at: float = 0.0
    expires_at: float = 0.0  # 0 = never
    error: str = ""

    @property
    def is_expired(self) -> bool:
        if self.expires_at <= 0:
            return False
        return time.monotonic() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "device_id": self.device_id,
            "command_type": self.command_type,
            "payload": dict(self.payload),
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "dispatched_at": self.dispatched_at,
            "acked_at": self.acked_at,
            "expires_at": self.expires_at,
            "error": self.error,
        }


_PRIORITY_ORDER = {
    CommandPriority.CRITICAL: 0,
    CommandPriority.HIGH: 1,
    CommandPriority.NORMAL: 2,
    CommandPriority.LOW: 3,
}


class CommandQueue:
    """Priority queue of commands destined for fleet devices.

    Commands are enqueued with a priority and dequeued highest-first.
    Expired commands are automatically pruned on dequeue.
    """

    def __init__(self) -> None:
        self._queue: list[QueuedCommand] = []
        self._by_id: dict[str, QueuedCommand] = {}
        self._history: deque[QueuedCommand] = deque(maxlen=500)

    # ---- enqueue / dequeue ------------------------------------------------

    def enqueue(
        self,
        device_id: str,
        command_type: str,
        payload: dict[str, Any] | None = None,
        priority: CommandPriority = CommandPriority.NORMAL,
        ttl_s: float = 0,
    ) -> QueuedCommand:
        """Add a command to the queue.  Returns the created command."""
        now = time.monotonic()
        cmd = QueuedCommand(
            command_id=uuid.uuid4().hex[:12],
            device_id=device_id,
            command_type=command_type,
            payload=payload or {},
            priority=priority,
            created_at=now,
            expires_at=(now + ttl_s) if ttl_s > 0 else 0,
        )
        self._queue.append(cmd)
        self._by_id[cmd.command_id] = cmd
        # Keep sorted by priority (stable — preserves FIFO within priority)
        self._queue.sort(key=lambda c: _PRIORITY_ORDER.get(c.priority, 2))
        return cmd

    def dequeue(self, device_id: str | None = None) -> QueuedCommand | None:
        """Pop the highest-priority queued command, optionally for a device.

        Expired commands are pruned silently.  Returns None when empty.
        """
        self._prune_expired()
        for i, cmd in enumerate(self._queue):
            if cmd.status != CommandStatus.QUEUED:
                continue
            if device_id is not None and cmd.device_id != device_id:
                continue
            cmd.status = CommandStatus.DISPATCHED
            cmd.dispatched_at = time.monotonic()
            self._queue.pop(i)
            self._history.append(cmd)
            return cmd
        return None

    def peek(self, device_id: str | None = None) -> QueuedCommand | None:
        """Look at the next command without removing it."""
        self._prune_expired()
        for cmd in self._queue:
            if cmd.status != CommandStatus.QUEUED:
                continue
            if device_id is not None and cmd.device_id != device_id:
                continue
            return cmd
        return None

    # ---- ack / fail -------------------------------------------------------

    def ack(self, command_id: str) -> bool:
        """Mark a dispatched command as acknowledged.  Returns False if not found."""
        cmd = self._by_id.get(command_id)
        if cmd is None:
            return False
        cmd.status = CommandStatus.ACKED
        cmd.acked_at = time.monotonic()
        return True

    def fail(self, command_id: str, error: str = "") -> bool:
        """Mark a dispatched command as failed.  Returns False if not found."""
        cmd = self._by_id.get(command_id)
        if cmd is None:
            return False
        cmd.status = CommandStatus.FAILED
        cmd.error = error
        return True

    # ---- query ------------------------------------------------------------

    def pending_for(self, device_id: str) -> list[QueuedCommand]:
        """All queued commands for a specific device."""
        return [
            c for c in self._queue
            if c.device_id == device_id and c.status == CommandStatus.QUEUED
        ]

    def get(self, command_id: str) -> QueuedCommand | None:
        return self._by_id.get(command_id)

    @property
    def pending_count(self) -> int:
        return sum(1 for c in self._queue if c.status == CommandStatus.QUEUED)

    @property
    def history(self) -> list[QueuedCommand]:
        """Recent dispatched/acked/failed commands (up to 500)."""
        return list(self._history)

    # ---- internal ---------------------------------------------------------

    def _prune_expired(self) -> None:
        """Move expired commands out of the active queue."""
        survivors: list[QueuedCommand] = []
        for cmd in self._queue:
            if cmd.status == CommandStatus.QUEUED and cmd.is_expired:
                cmd.status = CommandStatus.EXPIRED
                self._history.append(cmd)
            else:
                survivors.append(cmd)
        self._queue = survivors


# ---------------------------------------------------------------------------
# HeartbeatMonitor
# ---------------------------------------------------------------------------

@dataclass
class StaleDevice:
    """A device that has not sent a heartbeat within the timeout."""
    device_id: str
    last_heartbeat: float
    seconds_since: float


class HeartbeatMonitor:
    """Detect stale and offline devices based on heartbeat timing.

    The monitor does not keep its own device list — it reads from the
    owning ``FleetManager`` instance.
    """

    def __init__(self, devices: dict[str, FleetDevice]) -> None:
        self._devices = devices

    def check_stale(
        self,
        timeout_s: float = 60.0,
        now: float | None = None,
    ) -> list[StaleDevice]:
        """Return devices whose last heartbeat exceeds *timeout_s*.

        Devices that have never sent a heartbeat (last_heartbeat == 0)
        are included if they are not in OFFLINE status already.
        """
        if now is None:
            now = time.monotonic()
        stale: list[StaleDevice] = []
        for dev in self._devices.values():
            if dev.status == DeviceStatus.OFFLINE:
                continue
            if dev.last_heartbeat == 0:
                stale.append(StaleDevice(
                    device_id=dev.device_id,
                    last_heartbeat=0,
                    seconds_since=float("inf"),
                ))
                continue
            elapsed = now - dev.last_heartbeat
            if elapsed > timeout_s:
                stale.append(StaleDevice(
                    device_id=dev.device_id,
                    last_heartbeat=dev.last_heartbeat,
                    seconds_since=round(elapsed, 2),
                ))
        return stale

    def mark_stale(self, timeout_s: float = 60.0, now: float | None = None) -> list[str]:
        """Transition overdue devices to STALE.  Returns affected device IDs."""
        stale_list = self.check_stale(timeout_s=timeout_s, now=now)
        affected: list[str] = []
        for s in stale_list:
            dev = self._devices.get(s.device_id)
            if dev is not None and dev.status not in (DeviceStatus.STALE, DeviceStatus.OFFLINE):
                dev.status = DeviceStatus.STALE
                affected.append(dev.device_id)
        return affected

    def mark_offline(self, timeout_s: float = 300.0, now: float | None = None) -> list[str]:
        """Transition long-overdue devices to OFFLINE.  Returns affected IDs."""
        stale_list = self.check_stale(timeout_s=timeout_s, now=now)
        affected: list[str] = []
        for s in stale_list:
            dev = self._devices.get(s.device_id)
            if dev is not None and dev.status != DeviceStatus.OFFLINE:
                dev.status = DeviceStatus.OFFLINE
                affected.append(dev.device_id)
        return affected


# ---------------------------------------------------------------------------
# FleetManager
# ---------------------------------------------------------------------------

class FleetManager:
    """Central fleet management — register, heartbeat, group, command, monitor.

    Pure logic, no network I/O.  Callers feed in heartbeats and read back
    state.  Provides sub-managers for commands and heartbeat monitoring.

    Example::

        mgr = FleetManager()
        mgr.register("esp32-001", device_type="esp32-s3", capabilities=["ble"])
        mgr.heartbeat("esp32-001", wifi_rssi=-48, free_heap=200_000)
        print(mgr.get("esp32-001").status)  # DeviceStatus.ONLINE
    """

    def __init__(self) -> None:
        self._devices: dict[str, FleetDevice] = {}
        self._groups: dict[str, DeviceGroup] = {}
        self.commands = CommandQueue()
        self.monitor = HeartbeatMonitor(self._devices)

    # ---- registration -----------------------------------------------------

    def register(
        self,
        device_id: str,
        *,
        device_type: str = "esp32",
        device_name: str = "",
        capabilities: list[str] | None = None,
        group: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FleetDevice:
        """Register a new device.  Raises ``ValueError`` if already registered."""
        if device_id in self._devices:
            raise ValueError(f"Device '{device_id}' is already registered")
        now = time.monotonic()
        dev = FleetDevice(
            device_id=device_id,
            device_type=device_type,
            device_name=device_name or device_id,
            capabilities=list(capabilities or []),
            group=group,
            tags=list(tags or []),
            metadata=metadata or {},
            registered_at=now,
        )
        self._devices[device_id] = dev

        # Auto-add to group if specified
        if group:
            self._ensure_group(group)
            self._groups[group].add(device_id)

        return dev

    def unregister(self, device_id: str) -> bool:
        """Remove a device.  Returns False if not found."""
        dev = self._devices.pop(device_id, None)
        if dev is None:
            return False
        # Remove from any groups
        for grp in self._groups.values():
            grp.remove(device_id)
        return True

    # ---- heartbeat --------------------------------------------------------

    def heartbeat(
        self,
        device_id: str,
        *,
        firmware_version: str | None = None,
        ip_address: str | None = None,
        wifi_rssi: int | None = None,
        free_heap: int | None = None,
        uptime_s: int | None = None,
        battery_pct: float | None = None,
        capabilities: list[str] | None = None,
        now: float | None = None,
    ) -> bool:
        """Process an incoming heartbeat.  Updates telemetry and marks ONLINE.

        Returns False if the device is not registered.
        """
        dev = self._devices.get(device_id)
        if dev is None:
            return False

        if now is None:
            now = time.monotonic()
        dev.last_heartbeat = now
        dev.status = DeviceStatus.ONLINE

        if firmware_version is not None:
            dev.firmware_version = firmware_version
        if ip_address is not None:
            dev.ip_address = ip_address
        if wifi_rssi is not None:
            dev.wifi_rssi = wifi_rssi
        if free_heap is not None:
            dev.free_heap = free_heap
        if uptime_s is not None:
            dev.uptime_s = uptime_s
        if battery_pct is not None:
            dev.battery_pct = battery_pct
        if capabilities is not None:
            dev.capabilities = list(capabilities)

        return True

    # ---- lookup -----------------------------------------------------------

    def get(self, device_id: str) -> FleetDevice | None:
        return self._devices.get(device_id)

    def list_devices(
        self,
        *,
        status: DeviceStatus | None = None,
        group: str | None = None,
        capability: str | None = None,
        device_type: str | None = None,
    ) -> list[FleetDevice]:
        """List devices with optional filters."""
        devices = list(self._devices.values())
        if status is not None:
            devices = [d for d in devices if d.status == status]
        if group is not None:
            devices = [d for d in devices if d.group == group]
        if capability is not None:
            devices = [d for d in devices if capability in d.capabilities]
        if device_type is not None:
            devices = [d for d in devices if d.device_type == device_type]
        return devices

    @property
    def device_count(self) -> int:
        return len(self._devices)

    @property
    def online_count(self) -> int:
        return sum(1 for d in self._devices.values() if d.status == DeviceStatus.ONLINE)

    @property
    def offline_count(self) -> int:
        return sum(1 for d in self._devices.values() if d.status == DeviceStatus.OFFLINE)

    # ---- groups -----------------------------------------------------------

    def create_group(
        self,
        group_id: str,
        name: str = "",
        description: str = "",
        config: dict[str, Any] | None = None,
    ) -> DeviceGroup:
        """Create a named device group.  Raises ValueError if exists."""
        if group_id in self._groups:
            raise ValueError(f"Group '{group_id}' already exists")
        grp = DeviceGroup(
            group_id=group_id,
            name=name or group_id,
            description=description,
            config=config or {},
            created_at=time.monotonic(),
        )
        self._groups[group_id] = grp
        return grp

    def delete_group(self, group_id: str) -> bool:
        """Delete a group and unset group membership on its devices."""
        grp = self._groups.pop(group_id, None)
        if grp is None:
            return False
        for did in grp.device_ids:
            dev = self._devices.get(did)
            if dev is not None and dev.group == group_id:
                dev.group = ""
        return True

    def get_group(self, group_id: str) -> DeviceGroup | None:
        return self._groups.get(group_id)

    def list_groups(self) -> list[DeviceGroup]:
        return list(self._groups.values())

    def assign_to_group(self, device_id: str, group_id: str) -> bool:
        """Assign a device to a group.  Creates the group if it doesn't exist.

        Removes the device from its previous group first.
        Returns False if the device is not registered.
        """
        dev = self._devices.get(device_id)
        if dev is None:
            return False

        # Remove from old group
        if dev.group and dev.group in self._groups:
            self._groups[dev.group].remove(device_id)

        # Add to new group
        self._ensure_group(group_id)
        self._groups[group_id].add(device_id)
        dev.group = group_id
        return True

    def remove_from_group(self, device_id: str) -> bool:
        """Remove a device from its current group.  Returns False if not in a group."""
        dev = self._devices.get(device_id)
        if dev is None or not dev.group:
            return False
        grp = self._groups.get(dev.group)
        if grp is not None:
            grp.remove(device_id)
        dev.group = ""
        return True

    # ---- health summary ---------------------------------------------------

    def health_score(self) -> float:
        """Fleet-wide health score 0.0-1.0.

        Factors: online ratio (50%), avg RSSI (25%), avg heap (25%).
        """
        if not self._devices:
            return 0.0

        total = len(self._devices)
        online = [d for d in self._devices.values() if d.status == DeviceStatus.ONLINE]
        online_ratio = len(online) / total

        if not online:
            return round(online_ratio * 0.5, 3)

        # RSSI: map [-90, -30] -> [0.0, 1.0]
        avg_rssi = sum(d.wifi_rssi for d in online) / len(online)
        rssi_score = max(0.0, min(1.0, (avg_rssi + 90) / 60.0))

        # Heap: fraction of 300KB typical max
        avg_heap = sum(d.free_heap for d in online) / len(online)
        heap_score = max(0.0, min(1.0, avg_heap / 300_000))

        score = (online_ratio * 0.5) + (rssi_score * 0.25) + (heap_score * 0.25)
        return round(max(0.0, min(1.0, score)), 3)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable fleet summary."""
        by_status: dict[str, int] = {}
        for dev in self._devices.values():
            by_status[dev.status.value] = by_status.get(dev.status.value, 0) + 1

        by_group: dict[str, int] = {}
        for grp in self._groups.values():
            by_group[grp.group_id] = grp.size

        return {
            "total_devices": self.device_count,
            "online": self.online_count,
            "offline": self.offline_count,
            "by_status": by_status,
            "groups": by_group,
            "health_score": self.health_score(),
            "pending_commands": self.commands.pending_count,
        }

    # ---- iteration --------------------------------------------------------

    def __contains__(self, device_id: object) -> bool:
        return device_id in self._devices

    def __len__(self) -> int:
        return len(self._devices)

    def __iter__(self) -> Iterator[FleetDevice]:
        return iter(self._devices.values())

    # ---- internal ---------------------------------------------------------

    def _ensure_group(self, group_id: str) -> None:
        """Create a group if it doesn't already exist."""
        if group_id not in self._groups:
            self._groups[group_id] = DeviceGroup(
                group_id=group_id,
                name=group_id,
                created_at=time.monotonic(),
            )


__all__ = [
    "CommandPriority",
    "CommandQueue",
    "CommandStatus",
    "DeviceGroup",
    "DeviceStatus",
    "FleetDevice",
    "FleetManager",
    "HeartbeatMonitor",
    "QueuedCommand",
    "StaleDevice",
]
