# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Daily analytics model for tracking system performance over time.

Captures new targets discovered, correlations made, threats detected,
zone events, sightings by source, and top devices for a given date.
Used by /api/picture-of-day and performance dashboards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional


@dataclass
class DeviceActivity:
    """Activity summary for a single device in a time period."""
    device_id: str = ""
    sighting_count: int = 0
    target_count: int = 0
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "sighting_count": self.sighting_count,
            "target_count": self.target_count,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DeviceActivity:
        return cls(
            device_id=data.get("device_id", ""),
            sighting_count=data.get("sighting_count", 0),
            target_count=data.get("target_count", 0),
            last_seen=data.get("last_seen", 0.0),
        )


@dataclass
class DailyAnalytics:
    """Analytics snapshot for a single day.

    Attributes
    ----------
    report_date:
        The date this report covers (YYYY-MM-DD).
    generated_at:
        When this report was generated.
    new_targets:
        Number of new unique targets discovered.
    correlations:
        Number of target correlation/fusion events.
    threats:
        Number of threats detected (hostile classifications).
    zone_events:
        Number of geofence enter/exit events.
    investigations_opened:
        Number of new investigations opened.
    total_sightings:
        Total sighting events across all sources.
    sightings_by_source:
        Breakdown of sightings by source type (ble, wifi, yolo, mesh, etc.).
    top_devices:
        Most active devices by sighting count.
    threat_level:
        Overall threat level for the day (GREEN/YELLOW/ORANGE/RED).
    uptime_percent:
        System uptime as percentage (0-100).
    extra:
        Arbitrary extension data.
    """
    report_date: Optional[str] = None
    generated_at: Optional[datetime] = None
    new_targets: int = 0
    correlations: int = 0
    threats: int = 0
    zone_events: int = 0
    investigations_opened: int = 0
    total_sightings: int = 0
    sightings_by_source: dict[str, int] = field(default_factory=dict)
    top_devices: list[DeviceActivity] = field(default_factory=list)
    threat_level: str = "GREEN"
    uptime_percent: float = 100.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.generated_at is None:
            self.generated_at = datetime.now(timezone.utc)
        if self.report_date is None:
            self.report_date = date.today().isoformat()

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSON/REST transport."""
        return {
            "report_date": self.report_date,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "new_targets": self.new_targets,
            "correlations": self.correlations,
            "threats": self.threats,
            "zone_events": self.zone_events,
            "investigations_opened": self.investigations_opened,
            "total_sightings": self.total_sightings,
            "sightings_by_source": self.sightings_by_source,
            "top_devices": [d.to_dict() for d in self.top_devices],
            "threat_level": self.threat_level,
            "uptime_percent": self.uptime_percent,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DailyAnalytics:
        """Deserialize from plain dict."""
        top_devices = [
            DeviceActivity.from_dict(d)
            for d in data.get("top_devices", [])
        ]

        analytics = cls(
            report_date=data.get("report_date"),
            new_targets=data.get("new_targets", 0),
            correlations=data.get("correlations", 0),
            threats=data.get("threats", 0),
            zone_events=data.get("zone_events", 0),
            investigations_opened=data.get("investigations_opened", 0),
            total_sightings=data.get("total_sightings", 0),
            sightings_by_source=data.get("sightings_by_source", {}),
            top_devices=top_devices,
            threat_level=data.get("threat_level", "GREEN"),
            uptime_percent=data.get("uptime_percent", 100.0),
            extra=data.get("extra", {}),
        )

        if data.get("generated_at"):
            analytics.generated_at = datetime.fromisoformat(data["generated_at"])

        return analytics
