# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tactical event model — unified model for ALL events on the tactical picture.

A TacticalEvent represents any significant occurrence that should appear
on the tactical map or in the event feed: target detections, alerts,
geofence breaches, acoustic events, communications, fleet status changes,
etc. This is the single canonical event type that the Command Center UI
consumes for rendering events on the map and timeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class TacticalEventType(str, Enum):
    """Categories of tactical events."""
    DETECTION = "detection"             # New target detected
    CLASSIFICATION = "classification"   # Target classified/reclassified
    ALERT = "alert"                     # Alert triggered
    GEOFENCE = "geofence"              # Geofence entry/exit
    ACOUSTIC = "acoustic"              # Acoustic event (gunshot, voice, etc.)
    RF_ANOMALY = "rf_anomaly"          # RF environment anomaly
    FLEET = "fleet"                     # Device online/offline/degraded
    COMMUNICATION = "communication"     # Message received/sent
    ENGAGEMENT = "engagement"           # Active response/engagement
    CORRELATION = "correlation"         # Targets correlated/fused
    INVESTIGATION = "investigation"     # Investigation opened/closed
    SYSTEM = "system"                   # System-level event
    MANUAL = "manual"                   # Operator-created event


class TacticalSeverity(str, Enum):
    """Severity levels for tactical events."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EventPosition:
    """Geographic position associated with an event."""
    lat: float = 0.0
    lng: float = 0.0
    alt_m: Optional[float] = None
    accuracy_m: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "lat": self.lat,
            "lng": self.lng,
            "alt_m": self.alt_m,
            "accuracy_m": self.accuracy_m,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EventPosition:
        return cls(
            lat=data.get("lat", 0.0),
            lng=data.get("lng", 0.0),
            alt_m=data.get("alt_m"),
            accuracy_m=data.get("accuracy_m"),
        )


@dataclass
class TacticalEvent:
    """A unified event that appears on the tactical picture.

    All event types in the system — detections, alerts, geofence
    breaches, acoustic events, fleet changes — are normalized into
    this single model for consistent rendering on the map and timeline.

    Attributes:
        event_id: Unique event identifier.
        event_type: Category of event.
        severity: How critical this event is.
        position: Where the event occurred (if spatial).
        description: Human-readable description.
        source: What produced this event (plugin name, device ID, etc.).
        entities: Target IDs or entity IDs involved in this event.
        timestamp: When the event occurred.
        site_id: Site this event belongs to.
        acknowledged: Whether an operator has acknowledged this event.
        acknowledged_by: Who acknowledged it.
        resolved: Whether this event has been resolved.
        ttl_sec: Time-to-live in seconds (0 = permanent).
        tags: Arbitrary tags for filtering.
        metadata: Extra data specific to the event type.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: TacticalEventType = TacticalEventType.SYSTEM
    severity: TacticalSeverity = TacticalSeverity.INFO
    position: Optional[EventPosition] = None
    description: str = ""
    source: str = ""
    entities: list[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    site_id: str = ""
    acknowledged: bool = False
    acknowledged_by: str = ""
    resolved: bool = False
    ttl_sec: int = 0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def acknowledge(self, operator: str) -> None:
        """Mark this event as acknowledged by an operator."""
        self.acknowledged = True
        self.acknowledged_by = operator

    def resolve(self) -> None:
        """Mark this event as resolved."""
        self.resolved = True

    @property
    def is_active(self) -> bool:
        """True if the event is unresolved and unacknowledged."""
        return not self.resolved and not self.acknowledged

    @property
    def is_spatial(self) -> bool:
        """True if this event has a geographic position."""
        return self.position is not None

    @property
    def is_expired(self) -> bool:
        """True if TTL has elapsed (always False if ttl_sec == 0)."""
        if self.ttl_sec <= 0 or self.timestamp is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return elapsed > self.ttl_sec

    def involves_entity(self, entity_id: str) -> bool:
        """Check if a specific entity is involved in this event."""
        return entity_id in self.entities

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSON transport."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "position": self.position.to_dict() if self.position else None,
            "description": self.description,
            "source": self.source,
            "entities": self.entities,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "site_id": self.site_id,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "resolved": self.resolved,
            "ttl_sec": self.ttl_sec,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TacticalEvent:
        """Deserialize from plain dict."""
        pos = None
        if data.get("position"):
            pos = EventPosition.from_dict(data["position"])

        evt = cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=TacticalEventType(data.get("event_type", "system")),
            severity=TacticalSeverity(data.get("severity", "info")),
            position=pos,
            description=data.get("description", ""),
            source=data.get("source", ""),
            entities=data.get("entities", []),
            site_id=data.get("site_id", ""),
            acknowledged=data.get("acknowledged", False),
            acknowledged_by=data.get("acknowledged_by", ""),
            resolved=data.get("resolved", False),
            ttl_sec=data.get("ttl_sec", 0),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )
        if data.get("timestamp"):
            evt.timestamp = datetime.fromisoformat(data["timestamp"])
        return evt


def filter_events(
    events: list[TacticalEvent],
    event_type: Optional[TacticalEventType] = None,
    severity: Optional[TacticalSeverity] = None,
    source: Optional[str] = None,
    entity_id: Optional[str] = None,
    active_only: bool = False,
    spatial_only: bool = False,
) -> list[TacticalEvent]:
    """Filter a list of tactical events by various criteria."""
    result = events
    if event_type is not None:
        result = [e for e in result if e.event_type == event_type]
    if severity is not None:
        result = [e for e in result if e.severity == severity]
    if source is not None:
        result = [e for e in result if e.source == source]
    if entity_id is not None:
        result = [e for e in result if e.involves_entity(entity_id)]
    if active_only:
        result = [e for e in result if e.is_active]
    if spatial_only:
        result = [e for e in result if e.is_spatial]
    return result
