# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fleet event correlation models.

Cross-device pattern detection: synchronized reboots, cascading failures,
environmental correlations, and periodic failure patterns. These models
match the output of tritium-edge's correlation_service.py.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CorrelationType(str, Enum):
    """Types of cross-device correlation patterns."""
    SYNCHRONIZED_REBOOT = "synchronized_reboot"
    CASCADING_FAILURE = "cascading_failure"
    CASCADING_WIFI_FAILURE = "cascading_wifi_failure"
    ENVIRONMENTAL = "environmental"
    ENVIRONMENTAL_FAULT = "environmental_fault"
    PERIODIC_FAILURE = "periodic_failure"
    CORRELATED_MEMORY_LEAK = "correlated_memory_leak"
    FIRMWARE_ANOMALY = "firmware_anomaly"
    FLEET_UPDATE = "fleet_update"


class CorrelationEvent(BaseModel):
    """A detected cross-device event correlation.

    Produced by the fleet server's correlation service when diagnostic
    snapshots from multiple devices reveal a pattern.
    """
    type: CorrelationType
    description: str = Field(..., description="Human-readable summary of the pattern")
    devices_involved: list[str] = Field(
        default_factory=list,
        description="Device IDs participating in the pattern",
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Detection confidence (0=low, 1=certain)",
    )
    severity: str = Field(
        "info",
        description="Alert severity: critical, warning, or info",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the earliest related event",
    )


class CorrelationSummary(BaseModel):
    """Summary of all detected correlations for a fleet snapshot window."""
    events: list[CorrelationEvent] = Field(default_factory=list)
    snapshot_count: int = Field(0, description="Number of diagnostic snapshots analyzed")
    device_count: int = Field(0, description="Number of distinct devices in the analysis")
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None

    @property
    def total_events(self) -> int:
        return len(self.events)

    @property
    def high_confidence_events(self) -> list[CorrelationEvent]:
        """Events with confidence >= 0.7."""
        return [e for e in self.events if e.confidence >= 0.7]

    def events_by_type(self, type: CorrelationType) -> list[CorrelationEvent]:
        """Filter events by correlation type."""
        return [e for e in self.events if e.type == type]

    def affected_devices(self) -> set[str]:
        """All unique device IDs involved in any correlation."""
        devices: set[str] = set()
        for event in self.events:
            devices.update(event.devices_involved)
        return devices


def classify_correlation_severity(event: CorrelationEvent) -> str:
    """Map a correlation event to an alert severity level.

    Returns one of: "critical", "warning", "info".
    """
    # Use pre-set severity if explicitly set to non-default
    if event.severity and event.severity not in ("info", ""):
        return event.severity

    if event.type == CorrelationType.SYNCHRONIZED_REBOOT:
        if len(event.devices_involved) >= 5:
            return "critical"
        return "warning"

    critical_types = {
        CorrelationType.CASCADING_FAILURE,
        CorrelationType.CASCADING_WIFI_FAILURE,
        CorrelationType.FIRMWARE_ANOMALY,
        CorrelationType.CORRELATED_MEMORY_LEAK,
    }
    if event.type in critical_types:
        if event.confidence >= 0.7 or len(event.devices_involved) >= 5:
            return "critical"
        return "warning"
    if event.type in (CorrelationType.ENVIRONMENTAL, CorrelationType.ENVIRONMENTAL_FAULT):
        return "warning"
    if event.type == CorrelationType.FLEET_UPDATE:
        return "info"
    return "info"


def summarize_correlations(events: list[CorrelationEvent]) -> dict:
    """Produce a compact summary dict for API responses.

    Returns:
        Dict with counts by type, total, high-confidence count,
        and affected device count.
    """
    by_type: dict[str, int] = {}
    devices: set[str] = set()
    high_conf = 0

    for e in events:
        by_type[e.type.value] = by_type.get(e.type.value, 0) + 1
        devices.update(e.devices_involved)
        if e.confidence >= 0.7:
            high_conf += 1

    return {
        "total": len(events),
        "high_confidence": high_conf,
        "by_type": by_type,
        "affected_devices": len(devices),
    }
