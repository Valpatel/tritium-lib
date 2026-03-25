# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Command and Control (C2) protocol — send orders to edge devices, receive status.

Pure protocol and data model layer.  No network transport — callers back
this with MQTT, HTTP, WebSocket, or any other channel.

Command types
-------------
- **ScanCommand** — configure BLE/WiFi scan parameters
- **PatrolCommand** — set patrol route for mobile devices
- **ObserveCommand** — focus sensors on a specific target
- **ConfigCommand** — update device configuration key/values
- **DiagnosticCommand** — request a diagnostic dump

Addressing
----------
- **DeviceCommand** — addressed to a single device by ID
- **BroadcastCommand** — addressed to all devices in a group (or all)

Audit
-----
- **CommandHistory** — full audit trail of every command issued, with
  results, timestamps, and operator attribution.

Quick start::

    from tritium_lib.c2 import (
        C2Controller, ScanCommand, PatrolCommand, ObserveCommand,
        ConfigCommand, DiagnosticCommand, CommandResult,
    )

    c2 = C2Controller()

    # Send a scan command to a specific device
    cmd = c2.send_device(
        device_id="esp32-001",
        command=ScanCommand(ble_channels=[37, 38, 39], scan_duration_s=10.0),
        operator="amy",
    )

    # Broadcast a config change to all devices in "perimeter" group
    cmd = c2.send_broadcast(
        group="perimeter",
        command=ConfigCommand(updates={"heartbeat_interval_s": 5}),
        operator="amy",
    )

    # Record a result when a device responds
    c2.record_result(cmd.command_id, CommandResult(success=True, detail="scan started"))

    # Query audit trail
    history = c2.history.query(device_id="esp32-001")
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class C2Priority(str, Enum):
    """Priority level for C2 commands."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class C2Status(str, Enum):
    """Lifecycle status of a C2 command."""
    DRAFT = "draft"
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class C2CommandType(str, Enum):
    """High-level command categories."""
    SCAN = "scan"
    PATROL = "patrol"
    OBSERVE = "observe"
    CONFIGURE = "configure"
    DIAGNOSTIC = "diagnostic"


# ---------------------------------------------------------------------------
# Command payloads
# ---------------------------------------------------------------------------

@dataclass
class ScanCommand:
    """Configure scan parameters on an edge device.

    Supports BLE channel selection, WiFi band selection, scan duration,
    and scan interval.  Sensible defaults allow sending a bare ScanCommand
    to trigger a default scan burst.
    """
    type: C2CommandType = field(default=C2CommandType.SCAN, init=False)

    ble_channels: list[int] = field(default_factory=lambda: [37, 38, 39])
    wifi_bands: list[str] = field(default_factory=lambda: ["2.4GHz"])
    scan_duration_s: float = 10.0
    scan_interval_s: float = 0.0  # 0 = one-shot
    active_scan: bool = False
    filter_rssi_min: int = -100  # ignore weaker signals

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "ble_channels": list(self.ble_channels),
            "wifi_bands": list(self.wifi_bands),
            "scan_duration_s": self.scan_duration_s,
            "scan_interval_s": self.scan_interval_s,
            "active_scan": self.active_scan,
            "filter_rssi_min": self.filter_rssi_min,
        }


@dataclass
class PatrolCommand:
    """Set a patrol route for a mobile device (robot, drone, rover).

    The route is an ordered list of waypoints.  Each waypoint is a dict
    with at least ``lat`` and ``lng`` keys, plus optional ``alt``,
    ``dwell_s`` (how long to stay), and ``action`` (what to do there).
    """
    type: C2CommandType = field(default=C2CommandType.PATROL, init=False)

    waypoints: list[dict[str, Any]] = field(default_factory=list)
    loop: bool = False  # repeat the route
    speed_mps: float = 1.0  # meters per second
    resume_on_obstacle: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "waypoints": [dict(wp) for wp in self.waypoints],
            "loop": self.loop,
            "speed_mps": self.speed_mps,
            "resume_on_obstacle": self.resume_on_obstacle,
        }


@dataclass
class ObserveCommand:
    """Focus sensors on a specific target.

    Instructs the device to prioritize tracking a particular target ID,
    optionally adjusting scan parameters for better resolution.
    """
    type: C2CommandType = field(default=C2CommandType.OBSERVE, init=False)

    target_id: str = ""
    sensor_modes: list[str] = field(default_factory=lambda: ["ble", "wifi"])
    duration_s: float = 60.0
    report_interval_s: float = 5.0
    priority_boost: bool = True  # elevate scan priority for this target

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "target_id": self.target_id,
            "sensor_modes": list(self.sensor_modes),
            "duration_s": self.duration_s,
            "report_interval_s": self.report_interval_s,
            "priority_boost": self.priority_boost,
        }


@dataclass
class ConfigCommand:
    """Update device configuration key/value pairs.

    The ``updates`` dict is merged into the device's running configuration.
    Keys not present in ``updates`` are left unchanged.  Set a key to
    ``None`` to delete it.
    """
    type: C2CommandType = field(default=C2CommandType.CONFIGURE, init=False)

    updates: dict[str, Any] = field(default_factory=dict)
    restart_required: bool = False
    validate_before_apply: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "updates": dict(self.updates),
            "restart_required": self.restart_required,
            "validate_before_apply": self.validate_before_apply,
        }


@dataclass
class DiagnosticCommand:
    """Request a diagnostic dump from a device.

    The device collects the requested sections and sends them back
    as a diagnostic report.
    """
    type: C2CommandType = field(default=C2CommandType.DIAGNOSTIC, init=False)

    sections: list[str] = field(default_factory=lambda: [
        "system", "network", "sensors", "memory",
    ])
    include_logs: bool = False
    log_lines: int = 100
    include_config: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "sections": list(self.sections),
            "include_logs": self.include_logs,
            "log_lines": self.log_lines,
            "include_config": self.include_config,
        }


# Union of all command payload types
CommandPayload = ScanCommand | PatrolCommand | ObserveCommand | ConfigCommand | DiagnosticCommand


# ---------------------------------------------------------------------------
# CommandResult
# ---------------------------------------------------------------------------

@dataclass
class CommandResult:
    """Result reported by a device after executing a command."""
    success: bool = True
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error_code: str = ""
    received_at: float = 0.0

    def __post_init__(self) -> None:
        if self.received_at <= 0:
            self.received_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "detail": self.detail,
            "data": dict(self.data),
            "error_code": self.error_code,
            "received_at": self.received_at,
        }


# ---------------------------------------------------------------------------
# C2Channel — abstract bidirectional command channel
# ---------------------------------------------------------------------------

class C2Channel:
    """Abstract bidirectional command channel.

    Subclasses implement the actual transport (MQTT, HTTP, WebSocket, etc.).
    This base provides the interface contract and a no-op implementation
    useful for testing and dry-run mode.
    """

    def send(self, envelope: "C2Envelope") -> bool:
        """Send a command envelope to the device(s).

        Returns True if the send was accepted by the transport layer.
        This does NOT mean the device received or acknowledged the command.
        """
        return True

    def on_result(self, command_id: str, result: CommandResult) -> None:
        """Called when a result arrives from a device.

        Override to wire into the C2Controller's result recording.
        """
        pass

    @property
    def channel_name(self) -> str:
        """Human-readable name for this channel (e.g., 'mqtt', 'http')."""
        return "noop"

    @property
    def is_connected(self) -> bool:
        """Whether the underlying transport is connected."""
        return True


# ---------------------------------------------------------------------------
# C2Envelope — a command addressed and ready for delivery
# ---------------------------------------------------------------------------

@dataclass
class C2Envelope:
    """A fully addressed C2 command ready for delivery.

    Wraps a command payload with addressing, priority, TTL, and audit
    metadata.  This is the unit that flows through the C2 system.
    """
    command_id: str = ""
    command: Optional[CommandPayload] = None
    device_id: str = ""  # empty for broadcast
    group: str = ""  # non-empty for broadcast
    is_broadcast: bool = False
    priority: C2Priority = C2Priority.NORMAL
    status: C2Status = C2Status.DRAFT
    operator: str = ""  # who issued the command
    ttl_s: float = 300.0  # time to live (seconds)
    created_at: float = 0.0
    sent_at: float = 0.0
    delivered_at: float = 0.0
    completed_at: float = 0.0
    result: Optional[CommandResult] = None
    retries: int = 0
    max_retries: int = 3
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.command_id:
            self.command_id = uuid.uuid4().hex[:16]
        if self.created_at <= 0:
            self.created_at = time.time()

    @property
    def is_expired(self) -> bool:
        """True if the command has exceeded its TTL."""
        if self.ttl_s <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl_s

    @property
    def is_terminal(self) -> bool:
        """True if the command is in a final state."""
        return self.status in (
            C2Status.COMPLETED,
            C2Status.FAILED,
            C2Status.EXPIRED,
            C2Status.CANCELLED,
        )

    @property
    def command_type(self) -> C2CommandType | None:
        """The type of the wrapped command, or None if no command set."""
        if self.command is None:
            return None
        return self.command.type

    @property
    def target_description(self) -> str:
        """Human-readable description of who this targets."""
        if self.is_broadcast:
            return f"group:{self.group}" if self.group else "all"
        return self.device_id or "unknown"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot of the envelope."""
        result: dict[str, Any] = {
            "command_id": self.command_id,
            "command_type": self.command_type.value if self.command_type else None,
            "command": self.command.to_dict() if self.command else None,
            "device_id": self.device_id,
            "group": self.group,
            "is_broadcast": self.is_broadcast,
            "priority": self.priority.value,
            "status": self.status.value,
            "operator": self.operator,
            "ttl_s": self.ttl_s,
            "created_at": self.created_at,
            "sent_at": self.sent_at,
            "delivered_at": self.delivered_at,
            "completed_at": self.completed_at,
            "result": self.result.to_dict() if self.result else None,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }
        return result


# ---------------------------------------------------------------------------
# CommandHistory — audit trail
# ---------------------------------------------------------------------------

class CommandHistory:
    """Audit trail of all C2 commands issued.

    Stores envelopes in a bounded deque and supports querying by device,
    operator, command type, status, and time range.
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._entries: deque[C2Envelope] = deque(maxlen=max_size)
        self._by_id: dict[str, C2Envelope] = {}

    def record(self, envelope: C2Envelope) -> None:
        """Add an envelope to the history."""
        self._entries.append(envelope)
        self._by_id[envelope.command_id] = envelope
        # Prune the ID index when deque has evicted old entries
        if len(self._by_id) > len(self._entries):
            live_ids = {e.command_id for e in self._entries}
            stale = [k for k in self._by_id if k not in live_ids]
            for k in stale:
                del self._by_id[k]

    def get(self, command_id: str) -> C2Envelope | None:
        """Look up a command by ID."""
        return self._by_id.get(command_id)

    def query(
        self,
        *,
        device_id: str | None = None,
        group: str | None = None,
        operator: str | None = None,
        command_type: C2CommandType | None = None,
        status: C2Status | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> list[C2Envelope]:
        """Query the history with optional filters.

        Returns most-recent-first, up to ``limit`` entries.
        """
        results: list[C2Envelope] = []
        for envelope in reversed(self._entries):
            if device_id is not None and envelope.device_id != device_id:
                continue
            if group is not None and envelope.group != group:
                continue
            if operator is not None and envelope.operator != operator:
                continue
            if command_type is not None and envelope.command_type != command_type:
                continue
            if status is not None and envelope.status != status:
                continue
            if since is not None and envelope.created_at < since:
                continue
            if until is not None and envelope.created_at > until:
                continue
            results.append(envelope)
            if len(results) >= limit:
                break
        return results

    @property
    def total(self) -> int:
        """Total number of entries in history."""
        return len(self._entries)

    def summary(self) -> dict[str, Any]:
        """Return counts grouped by status and command type."""
        by_status: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for envelope in self._entries:
            by_status[envelope.status.value] = by_status.get(envelope.status.value, 0) + 1
            ct = envelope.command_type
            if ct is not None:
                by_type[ct.value] = by_type.get(ct.value, 0) + 1
        return {
            "total": self.total,
            "by_status": by_status,
            "by_type": by_type,
        }

    def clear(self) -> None:
        """Clear all history entries."""
        self._entries.clear()
        self._by_id.clear()


# ---------------------------------------------------------------------------
# C2Controller — main orchestrator
# ---------------------------------------------------------------------------

_PRIORITY_ORDER = {
    C2Priority.CRITICAL: 0,
    C2Priority.HIGH: 1,
    C2Priority.NORMAL: 2,
    C2Priority.LOW: 3,
}


class C2Controller:
    """Central C2 orchestrator — create, queue, send, and track commands.

    Pure logic, no network I/O.  Attach a ``C2Channel`` to wire up actual
    transport.  Without a channel, commands are queued and tracked but
    never sent over the wire.

    Example::

        c2 = C2Controller()
        env = c2.send_device("esp32-001", ScanCommand(), operator="amy")
        assert env.status == C2Status.SENT

        c2.record_result(env.command_id, CommandResult(success=True))
        assert env.status == C2Status.COMPLETED
    """

    def __init__(
        self,
        channel: C2Channel | None = None,
        history_size: int = 10000,
    ) -> None:
        self.channel = channel or C2Channel()
        self.history = CommandHistory(max_size=history_size)
        self._queue: list[C2Envelope] = []

    # ---- send commands ----------------------------------------------------

    def send_device(
        self,
        device_id: str,
        command: CommandPayload,
        *,
        operator: str = "",
        priority: C2Priority = C2Priority.NORMAL,
        ttl_s: float = 300.0,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> C2Envelope:
        """Send a command to a specific device.

        Creates a C2Envelope, sends it through the channel, and records
        it in history.  Returns the envelope (which is mutated in place
        as status changes).
        """
        envelope = C2Envelope(
            command=command,
            device_id=device_id,
            is_broadcast=False,
            priority=priority,
            operator=operator,
            ttl_s=ttl_s,
            tags=list(tags or []),
            metadata=metadata or {},
        )
        return self._dispatch(envelope)

    def send_broadcast(
        self,
        command: CommandPayload,
        *,
        group: str = "",
        operator: str = "",
        priority: C2Priority = C2Priority.NORMAL,
        ttl_s: float = 300.0,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> C2Envelope:
        """Broadcast a command to all devices in a group (or all devices).

        If ``group`` is empty, the command targets all devices.
        """
        envelope = C2Envelope(
            command=command,
            group=group,
            is_broadcast=True,
            priority=priority,
            operator=operator,
            ttl_s=ttl_s,
            tags=list(tags or []),
            metadata=metadata or {},
        )
        return self._dispatch(envelope)

    def queue_command(
        self,
        device_id: str,
        command: CommandPayload,
        *,
        operator: str = "",
        priority: C2Priority = C2Priority.NORMAL,
        ttl_s: float = 300.0,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> C2Envelope:
        """Queue a command for later dispatch (e.g., when device comes online).

        The command sits in QUEUED status until ``flush_queue`` is called.
        """
        envelope = C2Envelope(
            command=command,
            device_id=device_id,
            is_broadcast=False,
            priority=priority,
            status=C2Status.QUEUED,
            operator=operator,
            ttl_s=ttl_s,
            tags=list(tags or []),
            metadata=metadata or {},
        )
        self._queue.append(envelope)
        self._queue.sort(key=lambda e: _PRIORITY_ORDER.get(e.priority, 2))
        self.history.record(envelope)
        return envelope

    def flush_queue(self, device_id: str | None = None) -> list[C2Envelope]:
        """Dispatch all queued commands, optionally filtering by device.

        Returns the list of envelopes that were dispatched.  Expired
        commands are pruned instead of dispatched.
        """
        dispatched: list[C2Envelope] = []
        remaining: list[C2Envelope] = []

        for envelope in self._queue:
            if device_id is not None and envelope.device_id != device_id:
                remaining.append(envelope)
                continue
            if envelope.is_expired:
                envelope.status = C2Status.EXPIRED
                continue
            self._send_via_channel(envelope)
            dispatched.append(envelope)

        self._queue = remaining
        return dispatched

    # ---- result handling --------------------------------------------------

    def record_result(
        self,
        command_id: str,
        result: CommandResult,
    ) -> bool:
        """Record a result from a device for a previously sent command.

        Updates the envelope's status to COMPLETED or FAILED based on
        ``result.success``.  Returns False if the command_id is unknown.
        """
        envelope = self.history.get(command_id)
        if envelope is None:
            return False
        if envelope.is_terminal:
            return False  # already in a final state

        envelope.result = result
        envelope.completed_at = time.time()
        envelope.status = C2Status.COMPLETED if result.success else C2Status.FAILED
        return True

    def mark_delivered(self, command_id: str) -> bool:
        """Mark a command as delivered (transport confirmed receipt).

        Returns False if the command is unknown or already terminal.
        """
        envelope = self.history.get(command_id)
        if envelope is None:
            return False
        if envelope.is_terminal:
            return False
        envelope.status = C2Status.DELIVERED
        envelope.delivered_at = time.time()
        return True

    def mark_acknowledged(self, command_id: str) -> bool:
        """Mark a command as acknowledged (device confirmed it will execute).

        Returns False if the command is unknown or already terminal.
        """
        envelope = self.history.get(command_id)
        if envelope is None:
            return False
        if envelope.is_terminal:
            return False
        envelope.status = C2Status.ACKNOWLEDGED
        return True

    def mark_executing(self, command_id: str) -> bool:
        """Mark a command as currently executing on the device.

        Returns False if the command is unknown or already terminal.
        """
        envelope = self.history.get(command_id)
        if envelope is None:
            return False
        if envelope.is_terminal:
            return False
        envelope.status = C2Status.EXECUTING
        return True

    def cancel(self, command_id: str) -> bool:
        """Cancel a command.  Works on queued or in-flight commands.

        Returns False if the command is unknown or already terminal.
        """
        envelope = self.history.get(command_id)
        if envelope is None:
            return False
        if envelope.is_terminal:
            return False
        envelope.status = C2Status.CANCELLED
        # Also remove from queue if still there
        self._queue = [e for e in self._queue if e.command_id != command_id]
        return True

    def retry(self, command_id: str) -> bool:
        """Retry a failed command if retries remain.

        Re-sends through the channel and increments the retry counter.
        Returns False if the command is unknown, not failed, or out of retries.
        """
        envelope = self.history.get(command_id)
        if envelope is None:
            return False
        if envelope.status != C2Status.FAILED:
            return False
        if envelope.retries >= envelope.max_retries:
            return False
        envelope.retries += 1
        self._send_via_channel(envelope)
        return True

    # ---- query ------------------------------------------------------------

    def pending_for(self, device_id: str) -> list[C2Envelope]:
        """All queued commands for a specific device."""
        return [
            e for e in self._queue
            if e.device_id == device_id and e.status == C2Status.QUEUED
        ]

    @property
    def queue_size(self) -> int:
        """Number of commands in the queue."""
        return len(self._queue)

    def active_commands(self) -> list[C2Envelope]:
        """All non-terminal commands currently tracked."""
        return [
            e for e in self.history.query(limit=10000)
            if not e.is_terminal
        ]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of C2 state."""
        return {
            "queue_size": self.queue_size,
            "channel": self.channel.channel_name,
            "channel_connected": self.channel.is_connected,
            "history": self.history.summary(),
        }

    # ---- internal ---------------------------------------------------------

    def _dispatch(self, envelope: C2Envelope) -> C2Envelope:
        """Send via channel and record in history."""
        self._send_via_channel(envelope)
        self.history.record(envelope)
        return envelope

    def _send_via_channel(self, envelope: C2Envelope) -> None:
        """Push the envelope through the attached channel."""
        sent = self.channel.send(envelope)
        if sent:
            envelope.status = C2Status.SENT
            envelope.sent_at = time.time()
        else:
            envelope.status = C2Status.FAILED


__all__ = [
    # Enums
    "C2CommandType",
    "C2Priority",
    "C2Status",
    # Command payloads
    "ScanCommand",
    "PatrolCommand",
    "ObserveCommand",
    "ConfigCommand",
    "DiagnosticCommand",
    "CommandPayload",
    # Result
    "CommandResult",
    # Channel
    "C2Channel",
    # Envelope
    "C2Envelope",
    # History
    "CommandHistory",
    # Controller
    "C2Controller",
]
