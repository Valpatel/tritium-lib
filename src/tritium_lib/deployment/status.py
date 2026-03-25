# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""ComponentStatus — runtime status of a deployed Tritium component."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StatusLevel(str, Enum):
    """Runtime status level for a component."""

    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    STARTING = "starting"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class ComponentStatus:
    """Runtime status of a single deployed component.

    Attributes
    ----------
    name:
        Component identifier (e.g., "sc", "mqtt").
    status:
        Current status level.
    pid:
        Process ID if running (0 = unknown/not running).
    uptime_seconds:
        How long the component has been running.
    port:
        Port the component is listening on (0 = N/A).
    version:
        Currently running version string.
    memory_mb:
        Current memory usage in megabytes.
    cpu_percent:
        Current CPU usage as a percentage (0-100).
    error_message:
        Error message if status is ERROR.
    last_checked:
        Timestamp of the last status check.
    details:
        Arbitrary key-value pairs with component-specific info.
    """

    name: str
    status: StatusLevel = StatusLevel.UNKNOWN
    pid: int = 0
    uptime_seconds: float = 0.0
    port: int = 0
    version: str = ""
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    error_message: str = ""
    last_checked: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_running(self) -> bool:
        """True if the component is running."""
        return self.status == StatusLevel.RUNNING

    @property
    def is_healthy(self) -> bool:
        """True if the component is running or starting."""
        return self.status in (StatusLevel.RUNNING, StatusLevel.STARTING)

    @property
    def needs_attention(self) -> bool:
        """True if the component is in error or degraded state."""
        return self.status in (StatusLevel.ERROR, StatusLevel.DEGRADED)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "pid": self.pid,
            "uptime_seconds": self.uptime_seconds,
            "port": self.port,
            "version": self.version,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "error_message": self.error_message,
            "last_checked": self.last_checked,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComponentStatus:
        """Deserialize from a dictionary."""
        return cls(
            name=data["name"],
            status=StatusLevel(data.get("status", "unknown")),
            pid=data.get("pid", 0),
            uptime_seconds=data.get("uptime_seconds", 0.0),
            port=data.get("port", 0),
            version=data.get("version", ""),
            memory_mb=data.get("memory_mb", 0.0),
            cpu_percent=data.get("cpu_percent", 0.0),
            error_message=data.get("error_message", ""),
            last_checked=data.get("last_checked", time.time()),
            details=data.get("details", {}),
        )

    def __str__(self) -> str:
        """Human-readable status string."""
        parts = [f"{self.name}: {self.status.value}"]
        if self.pid:
            parts.append(f"pid={self.pid}")
        if self.version:
            parts.append(f"v={self.version}")
        if self.error_message:
            parts.append(f"err={self.error_message}")
        return " ".join(parts)
