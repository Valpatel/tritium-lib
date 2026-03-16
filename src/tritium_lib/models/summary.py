# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""System summary model for point-in-time snapshots of entire system state.

Used by /api/health endpoints and system dashboards to show a single
unified view of targets, dossiers, plugins, alerts, investigations,
and fleet status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class TargetCounts:
    """Target counts broken down by alliance and source."""
    total: int = 0
    # By alliance
    friendly: int = 0
    hostile: int = 0
    unknown: int = 0
    # By source
    ble: int = 0
    yolo: int = 0
    mesh: int = 0
    wifi: int = 0
    rf_motion: int = 0
    manual: int = 0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "by_alliance": {
                "friendly": self.friendly,
                "hostile": self.hostile,
                "unknown": self.unknown,
            },
            "by_source": {
                "ble": self.ble,
                "yolo": self.yolo,
                "mesh": self.mesh,
                "wifi": self.wifi,
                "rf_motion": self.rf_motion,
                "manual": self.manual,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> TargetCounts:
        by_alliance = data.get("by_alliance", {})
        by_source = data.get("by_source", {})
        return cls(
            total=data.get("total", 0),
            friendly=by_alliance.get("friendly", 0),
            hostile=by_alliance.get("hostile", 0),
            unknown=by_alliance.get("unknown", 0),
            ble=by_source.get("ble", 0),
            yolo=by_source.get("yolo", 0),
            mesh=by_source.get("mesh", 0),
            wifi=by_source.get("wifi", 0),
            rf_motion=by_source.get("rf_motion", 0),
            manual=by_source.get("manual", 0),
        )


@dataclass
class FleetSummary:
    """Fleet device summary."""
    total_devices: int = 0
    online: int = 0
    offline: int = 0
    low_battery: int = 0

    def to_dict(self) -> dict:
        return {
            "total_devices": self.total_devices,
            "online": self.online,
            "offline": self.offline,
            "low_battery": self.low_battery,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FleetSummary:
        return cls(
            total_devices=data.get("total_devices", 0),
            online=data.get("online", 0),
            offline=data.get("offline", 0),
            low_battery=data.get("low_battery", 0),
        )


@dataclass
class SystemSummary:
    """Point-in-time snapshot of entire system state.

    Captures target counts, dossier counts, active plugins, alerts,
    investigations, and fleet status. Designed for /api/health and
    system dashboards.

    Attributes
    ----------
    timestamp:
        When this snapshot was taken.
    targets:
        Target counts by alliance and source.
    dossier_count:
        Number of active target dossiers.
    active_plugins:
        List of active plugin names.
    plugin_count:
        Total number of loaded plugins.
    active_alerts:
        Number of currently active (unresolved) alerts.
    active_investigations:
        Number of open investigations.
    fleet:
        Fleet device summary.
    demo_active:
        Whether demo/synthetic data mode is running.
    uptime_seconds:
        Server uptime in seconds.
    mqtt_connected:
        Whether the MQTT broker connection is active.
    version:
        Software version string.
    extra:
        Arbitrary extension data for plugins to add their own metrics.
    """
    timestamp: Optional[datetime] = None
    targets: TargetCounts = field(default_factory=TargetCounts)
    dossier_count: int = 0
    active_plugins: list[str] = field(default_factory=list)
    plugin_count: int = 0
    active_alerts: int = 0
    active_investigations: int = 0
    fleet: FleetSummary = field(default_factory=FleetSummary)
    demo_active: bool = False
    uptime_seconds: float = 0.0
    mqtt_connected: bool = False
    version: str = "0.1.0"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSON/REST transport."""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "targets": self.targets.to_dict(),
            "dossier_count": self.dossier_count,
            "active_plugins": self.active_plugins,
            "plugin_count": self.plugin_count,
            "active_alerts": self.active_alerts,
            "active_investigations": self.active_investigations,
            "fleet": self.fleet.to_dict(),
            "demo_active": self.demo_active,
            "uptime_seconds": self.uptime_seconds,
            "mqtt_connected": self.mqtt_connected,
            "version": self.version,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SystemSummary:
        """Deserialize from plain dict."""
        targets = TargetCounts.from_dict(data.get("targets", {}))
        fleet = FleetSummary.from_dict(data.get("fleet", {}))

        summary = cls(
            targets=targets,
            dossier_count=data.get("dossier_count", 0),
            active_plugins=data.get("active_plugins", []),
            plugin_count=data.get("plugin_count", 0),
            active_alerts=data.get("active_alerts", 0),
            active_investigations=data.get("active_investigations", 0),
            fleet=fleet,
            demo_active=data.get("demo_active", False),
            uptime_seconds=data.get("uptime_seconds", 0.0),
            mqtt_connected=data.get("mqtt_connected", False),
            version=data.get("version", "0.1.0"),
            extra=data.get("extra", {}),
        )

        if data.get("timestamp"):
            summary.timestamp = datetime.fromisoformat(data["timestamp"])

        return summary
