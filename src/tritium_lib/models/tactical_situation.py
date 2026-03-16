# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tactical situation model for situation banners and SITREP generation.

A TacticalSituation captures a point-in-time snapshot of the operational
environment: threat level, target counts, active alerts and investigations,
fleet health, and Amy AI commander status. Used by the situation banner
in the Command Center UI and as the data source for automated SITREP
generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class ThreatLevel(str, Enum):
    """Overall threat level for the operational area."""
    GREEN = "green"        # Normal operations, no threats
    YELLOW = "yellow"      # Elevated awareness, potential threats
    ORANGE = "orange"      # High threat, confirmed hostile activity
    RED = "red"            # Critical, active engagement
    BLACK = "black"        # Catastrophic, system compromise


class AmyStatus(str, Enum):
    """Status of the Amy AI commander."""
    ONLINE = "online"
    OFFLINE = "offline"
    THINKING = "thinking"
    ALERTING = "alerting"
    DEGRADED = "degraded"


@dataclass
class FleetHealth:
    """Summary health metrics for the device fleet."""
    total_devices: int = 0
    online: int = 0
    offline: int = 0
    degraded: int = 0
    avg_battery_pct: float = 100.0
    avg_uptime_hours: float = 0.0

    @property
    def health_pct(self) -> float:
        """Percentage of devices online or degraded (not offline)."""
        if self.total_devices == 0:
            return 100.0
        return ((self.online + self.degraded) / self.total_devices) * 100.0

    def to_dict(self) -> dict:
        return {
            "total_devices": self.total_devices,
            "online": self.online,
            "offline": self.offline,
            "degraded": self.degraded,
            "avg_battery_pct": self.avg_battery_pct,
            "avg_uptime_hours": self.avg_uptime_hours,
            "health_pct": self.health_pct,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FleetHealth:
        return cls(
            total_devices=data.get("total_devices", 0),
            online=data.get("online", 0),
            offline=data.get("offline", 0),
            degraded=data.get("degraded", 0),
            avg_battery_pct=data.get("avg_battery_pct", 100.0),
            avg_uptime_hours=data.get("avg_uptime_hours", 0.0),
        )


@dataclass
class TargetCountsSummary:
    """Target counts for the situation banner."""
    total: int = 0
    friendly: int = 0
    hostile: int = 0
    unknown: int = 0
    new_last_hour: int = 0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "friendly": self.friendly,
            "hostile": self.hostile,
            "unknown": self.unknown,
            "new_last_hour": self.new_last_hour,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TargetCountsSummary:
        return cls(
            total=data.get("total", 0),
            friendly=data.get("friendly", 0),
            hostile=data.get("hostile", 0),
            unknown=data.get("unknown", 0),
            new_last_hour=data.get("new_last_hour", 0),
        )


@dataclass
class TacticalSituation:
    """Point-in-time tactical situation snapshot.

    Captures everything needed for the situation banner and SITREP:
    threat level, target counts, active alerts/investigations, fleet
    health, and Amy AI status.

    Attributes:
        threat_level: Current threat level for the operational area.
        target_counts: Breakdown of tracked targets.
        active_alerts: Number of unresolved alerts.
        active_investigations: Number of open investigations.
        fleet_health: Device fleet health summary.
        amy_status: Current AI commander status.
        timestamp: When this snapshot was captured.
        site_id: Site this situation applies to.
        summary_text: Human-readable one-line summary.
        notes: Additional commander notes.
    """
    threat_level: ThreatLevel = ThreatLevel.GREEN
    target_counts: TargetCountsSummary = field(default_factory=TargetCountsSummary)
    active_alerts: int = 0
    active_investigations: int = 0
    fleet_health: FleetHealth = field(default_factory=FleetHealth)
    amy_status: AmyStatus = AmyStatus.ONLINE
    timestamp: Optional[datetime] = None
    site_id: str = ""
    summary_text: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def escalate(self) -> None:
        """Raise threat level by one step."""
        levels = list(ThreatLevel)
        idx = levels.index(self.threat_level)
        if idx < len(levels) - 1:
            self.threat_level = levels[idx + 1]

    def deescalate(self) -> None:
        """Lower threat level by one step."""
        levels = list(ThreatLevel)
        idx = levels.index(self.threat_level)
        if idx > 0:
            self.threat_level = levels[idx - 1]

    @property
    def is_critical(self) -> bool:
        """True if threat level is RED or BLACK."""
        return self.threat_level in (ThreatLevel.RED, ThreatLevel.BLACK)

    def generate_sitrep(self) -> str:
        """Generate a human-readable SITREP string."""
        lines = [
            f"SITREP — {self.timestamp.isoformat() if self.timestamp else 'N/A'}",
            f"Threat Level: {self.threat_level.value.upper()}",
            f"Targets: {self.target_counts.total} total "
            f"({self.target_counts.friendly}F / {self.target_counts.hostile}H / "
            f"{self.target_counts.unknown}U)",
            f"New targets (1h): {self.target_counts.new_last_hour}",
            f"Active alerts: {self.active_alerts}",
            f"Investigations: {self.active_investigations}",
            f"Fleet: {self.fleet_health.online}/{self.fleet_health.total_devices} online "
            f"({self.fleet_health.health_pct:.0f}% healthy)",
            f"Amy: {self.amy_status.value}",
        ]
        if self.summary_text:
            lines.append(f"Summary: {self.summary_text}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSON transport."""
        return {
            "threat_level": self.threat_level.value,
            "target_counts": self.target_counts.to_dict(),
            "active_alerts": self.active_alerts,
            "active_investigations": self.active_investigations,
            "fleet_health": self.fleet_health.to_dict(),
            "amy_status": self.amy_status.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "site_id": self.site_id,
            "summary_text": self.summary_text,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TacticalSituation:
        """Deserialize from plain dict."""
        sit = cls(
            threat_level=ThreatLevel(data.get("threat_level", "green")),
            target_counts=TargetCountsSummary.from_dict(data.get("target_counts", {})),
            active_alerts=data.get("active_alerts", 0),
            active_investigations=data.get("active_investigations", 0),
            fleet_health=FleetHealth.from_dict(data.get("fleet_health", {})),
            amy_status=AmyStatus(data.get("amy_status", "online")),
            site_id=data.get("site_id", ""),
            summary_text=data.get("summary_text", ""),
            notes=data.get("notes", ""),
        )
        if data.get("timestamp"):
            sit.timestamp = datetime.fromisoformat(data["timestamp"])
        return sit
