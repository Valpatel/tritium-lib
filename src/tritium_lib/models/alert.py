# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Alert and webhook notification models.

Defines the shared contract for fleet alerting — webhook registrations,
alert payloads, and delivery tracking. Used by both the fleet server's
AlertService and client dashboards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class AlertEventType(str, Enum):
    """Standard alert event types in the Tritium fleet."""
    NODE_ANOMALY = "node_anomaly"        # Anomaly detected on a node
    NODE_OFFLINE = "node_offline"        # Node stopped reporting
    NODE_REBOOT = "node_reboot"          # Node rebooted unexpectedly
    CONFIG_DRIFT = "config_drift"        # Config mismatch detected
    OTA_FAILURE = "ota_failure"          # OTA update failed
    BATTERY_LOW = "battery_low"          # Battery below threshold
    FLEET_DEGRADED = "fleet_degraded"    # Fleet health score dropped


class AlertSeverity(str, Enum):
    """Severity classification for alerts."""
    INFO = "info"          # Informational, no action needed
    WARNING = "warning"    # Attention recommended
    CRITICAL = "critical"  # Immediate action required


@dataclass
class WebhookConfig:
    """A registered webhook endpoint for receiving alerts."""
    id: str
    url: str
    name: str = ""
    severity_min: float = 0.0
    device_ids: list[str] = field(default_factory=list)
    event_types: list[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    fire_count: int = 0
    last_fired_at: Optional[datetime] = None

    @property
    def is_filtered_by_device(self) -> bool:
        return len(self.device_ids) > 0

    @property
    def is_filtered_by_event(self) -> bool:
        return len(self.event_types) > 0

    def matches(self, event_type: str, device_id: str, severity: float) -> bool:
        """Check if an event matches this webhook's filters."""
        if severity < self.severity_min:
            return False
        if self.device_ids and device_id not in self.device_ids:
            return False
        if self.event_types and event_type not in self.event_types:
            return False
        return True


@dataclass
class AlertDelivery:
    """Result of delivering an alert to a single webhook."""
    webhook_id: str
    url: str
    status_code: Optional[int] = None
    ok: bool = False
    error: Optional[str] = None


@dataclass
class Alert:
    """A fleet alert event with delivery tracking."""
    id: str
    timestamp: datetime
    event_type: str
    device_id: str
    detail: str
    severity: float
    deliveries: list[AlertDelivery] = field(default_factory=list)

    @property
    def severity_level(self) -> AlertSeverity:
        if self.severity >= 0.7:
            return AlertSeverity.CRITICAL
        elif self.severity >= 0.4:
            return AlertSeverity.WARNING
        return AlertSeverity.INFO

    @property
    def delivery_count(self) -> int:
        return len(self.deliveries)

    @property
    def successful_deliveries(self) -> int:
        return sum(1 for d in self.deliveries if d.ok)


@dataclass
class AlertHistory:
    """Summary of recent alert activity."""
    total_alerts: int
    critical_count: int
    warning_count: int
    info_count: int
    recent: list[Alert] = field(default_factory=list)


def classify_alert_severity(severity_score: float) -> AlertSeverity:
    """Classify a numeric severity score into an AlertSeverity level."""
    if severity_score >= 0.7:
        return AlertSeverity.CRITICAL
    elif severity_score >= 0.4:
        return AlertSeverity.WARNING
    return AlertSeverity.INFO


def summarize_alerts(alerts: list[Alert]) -> AlertHistory:
    """Build an AlertHistory summary from a list of alerts."""
    critical = sum(1 for a in alerts if a.severity_level == AlertSeverity.CRITICAL)
    warning = sum(1 for a in alerts if a.severity_level == AlertSeverity.WARNING)
    info = sum(1 for a in alerts if a.severity_level == AlertSeverity.INFO)
    return AlertHistory(
        total_alerts=len(alerts),
        critical_count=critical,
        warning_count=warning,
        info_count=info,
        recent=alerts,
    )
