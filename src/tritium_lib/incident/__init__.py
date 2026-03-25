# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.incident — Incident management from detection through resolution.

Manages the full lifecycle of incidents: detected -> investigating -> responding
-> resolved. Integrates with the AlertEngine so that alerts can automatically
create incidents, and with the EventBus to publish incident lifecycle events.

Core classes:
  - Incident         — the incident record with state machine lifecycle
  - IncidentManager  — create, update, escalate, resolve incidents
  - Timeline         — ordered events within an incident
  - TimelineEntry    — a single timestamped event in a timeline
  - AssignedResource — who/what is assigned to the incident
  - Resolution       — how the incident was resolved + lessons learned

Quick start::

    from tritium_lib.incident import IncidentManager, IncidentSeverity

    mgr = IncidentManager()

    # Create an incident
    inc = mgr.create(
        title="Hostile detected in Zone Alpha",
        severity=IncidentSeverity.HIGH,
        source="alerting",
        target_ids=["ble_aabbccdd"],
        zone_id="zone-alpha",
    )

    # Assign a resource
    mgr.assign_resource(inc.incident_id, "drone-01", role="surveillance")

    # Add timeline entries as the situation develops
    mgr.add_timeline_entry(
        inc.incident_id,
        description="Drone dispatched to Zone Alpha",
        author="operator",
    )

    # Escalate
    mgr.escalate(inc.incident_id, IncidentSeverity.CRITICAL, reason="Second hostile")

    # Resolve
    mgr.resolve(
        inc.incident_id,
        summary="Hostiles neutralized. Zone clear.",
        resolution_type="neutralized",
        lessons_learned="Faster drone dispatch needed.",
    )

Integration with AlertEngine::

    from tritium_lib.alerting import AlertEngine, AlertRecord
    from tritium_lib.incident import IncidentManager

    mgr = IncidentManager()
    alert_engine = AlertEngine()

    # Wire alert escalations to incident creation
    mgr.connect_alert_engine(alert_engine)

Architecture
------------
- **IncidentState** — enum lifecycle: DETECTED -> INVESTIGATING -> RESPONDING -> RESOLVED / CLOSED
- **IncidentSeverity** — LOW / MEDIUM / HIGH / CRITICAL
- **TimelineEntry** — timestamped event within an incident
- **Timeline** — ordered collection of timeline entries
- **AssignedResource** — resource (person, drone, unit) assigned to an incident
- **Resolution** — final outcome: summary, type, lessons learned
- **Incident** — the full incident record, owns timeline + resources + resolution
- **IncidentManager** — CRUD + lifecycle operations, alert integration, querying
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("tritium.incident")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IncidentState(str, Enum):
    """Lifecycle states for an incident."""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    RESPONDING = "responding"
    RESOLVED = "resolved"
    CLOSED = "closed"


class IncidentSeverity(str, Enum):
    """Severity classification for incidents."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Valid state transitions
_VALID_TRANSITIONS: dict[IncidentState, set[IncidentState]] = {
    IncidentState.DETECTED: {IncidentState.INVESTIGATING, IncidentState.RESPONDING, IncidentState.RESOLVED, IncidentState.CLOSED},
    IncidentState.INVESTIGATING: {IncidentState.RESPONDING, IncidentState.RESOLVED, IncidentState.CLOSED},
    IncidentState.RESPONDING: {IncidentState.RESOLVED, IncidentState.CLOSED},
    IncidentState.RESOLVED: {IncidentState.CLOSED, IncidentState.INVESTIGATING},  # can reopen
    IncidentState.CLOSED: set(),  # terminal
}


# ---------------------------------------------------------------------------
# TimelineEntry
# ---------------------------------------------------------------------------

@dataclass
class TimelineEntry:
    """A single timestamped event in an incident timeline.

    Attributes
    ----------
    entry_id:
        Unique identifier for this entry.
    timestamp:
        Unix timestamp when this entry was created.
    description:
        Human-readable description of what happened.
    author:
        Who or what created this entry (operator name, system, plugin).
    entry_type:
        Category: "note", "state_change", "escalation", "assignment",
        "alert", "action", "resolution".
    metadata:
        Arbitrary key-value data attached to this entry.
    """
    entry_id: str
    timestamp: float
    description: str
    author: str = "system"
    entry_type: str = "note"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "description": self.description,
            "author": self.author,
            "entry_type": self.entry_type,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TimelineEntry:
        return cls(
            entry_id=data.get("entry_id", uuid.uuid4().hex[:12]),
            timestamp=data.get("timestamp", time.time()),
            description=data.get("description", ""),
            author=data.get("author", "system"),
            entry_type=data.get("entry_type", "note"),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

class Timeline:
    """Ordered collection of timeline entries for an incident.

    Entries are maintained in chronological order (oldest first).
    Thread-safe when used through IncidentManager.
    """

    def __init__(self) -> None:
        self._entries: list[TimelineEntry] = []

    def add(
        self,
        description: str,
        author: str = "system",
        entry_type: str = "note",
        metadata: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> TimelineEntry:
        """Add a new entry to the timeline. Returns the created entry."""
        entry = TimelineEntry(
            entry_id=uuid.uuid4().hex[:12],
            timestamp=timestamp if timestamp is not None else time.time(),
            description=description,
            author=author,
            entry_type=entry_type,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._entries.sort(key=lambda e: e.timestamp)
        return entry

    def get_entries(
        self,
        entry_type: str = "",
        since: float = 0.0,
        limit: int = 0,
    ) -> list[TimelineEntry]:
        """Return entries with optional filtering.

        Parameters
        ----------
        entry_type:
            Filter by entry type. Empty string returns all.
        since:
            Only return entries after this timestamp.
        limit:
            Maximum entries to return. 0 = unlimited.

        Returns entries in chronological order.
        """
        result = list(self._entries)
        if entry_type:
            result = [e for e in result if e.entry_type == entry_type]
        if since > 0:
            result = [e for e in result if e.timestamp >= since]
        if limit > 0:
            result = result[:limit]
        return result

    @property
    def count(self) -> int:
        return len(self._entries)

    @property
    def latest(self) -> TimelineEntry | None:
        """Return the most recent entry, or None if empty."""
        return self._entries[-1] if self._entries else None

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    @classmethod
    def from_list(cls, data: list[dict]) -> Timeline:
        tl = cls()
        for item in data:
            tl._entries.append(TimelineEntry.from_dict(item))
        tl._entries.sort(key=lambda e: e.timestamp)
        return tl


# ---------------------------------------------------------------------------
# AssignedResource
# ---------------------------------------------------------------------------

@dataclass
class AssignedResource:
    """A resource (person, drone, unit) assigned to an incident.

    Attributes
    ----------
    resource_id:
        Unique identifier for this resource (e.g., "drone-01", "operator-smith").
    resource_type:
        Category: "person", "drone", "unit", "vehicle", "sensor", "other".
    role:
        What this resource is doing: "lead", "surveillance", "response",
        "support", "investigation".
    assigned_at:
        Unix timestamp when the resource was assigned.
    released_at:
        Unix timestamp when the resource was released, or 0.0 if still assigned.
    notes:
        Additional context about the assignment.
    """
    resource_id: str
    resource_type: str = "other"
    role: str = "support"
    assigned_at: float = 0.0
    released_at: float = 0.0
    notes: str = ""

    @property
    def is_active(self) -> bool:
        """True if the resource is currently assigned (not released)."""
        return self.released_at == 0.0

    def release(self, timestamp: float | None = None) -> None:
        """Mark this resource as released."""
        self.released_at = timestamp if timestamp is not None else time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "role": self.role,
            "assigned_at": self.assigned_at,
            "released_at": self.released_at,
            "notes": self.notes,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AssignedResource:
        return cls(
            resource_id=data["resource_id"],
            resource_type=data.get("resource_type", "other"),
            role=data.get("role", "support"),
            assigned_at=data.get("assigned_at", 0.0),
            released_at=data.get("released_at", 0.0),
            notes=data.get("notes", ""),
        )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

@dataclass
class Resolution:
    """How an incident was resolved, including lessons learned.

    Attributes
    ----------
    summary:
        Brief description of how the incident was resolved.
    resolution_type:
        Category: "neutralized", "false_alarm", "contained", "escalated_external",
        "timed_out", "deferred", "other".
    resolved_by:
        Who resolved it (operator name, system, automated).
    resolved_at:
        Unix timestamp of resolution.
    lessons_learned:
        Free-text field for post-incident improvement notes.
    follow_up_actions:
        List of recommended follow-up actions.
    """
    summary: str
    resolution_type: str = "other"
    resolved_by: str = "system"
    resolved_at: float = 0.0
    lessons_learned: str = ""
    follow_up_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "resolution_type": self.resolution_type,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at,
            "lessons_learned": self.lessons_learned,
            "follow_up_actions": list(self.follow_up_actions),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Resolution:
        return cls(
            summary=data.get("summary", ""),
            resolution_type=data.get("resolution_type", "other"),
            resolved_by=data.get("resolved_by", "system"),
            resolved_at=data.get("resolved_at", 0.0),
            lessons_learned=data.get("lessons_learned", ""),
            follow_up_actions=data.get("follow_up_actions", []),
        )


# ---------------------------------------------------------------------------
# Incident
# ---------------------------------------------------------------------------

@dataclass
class Incident:
    """A tracked incident from detection through resolution.

    Owns a Timeline, a list of AssignedResources, and an optional Resolution.
    State transitions are enforced via the IncidentManager.

    Attributes
    ----------
    incident_id:
        Unique identifier.
    title:
        Short human-readable description.
    state:
        Current lifecycle state.
    severity:
        Current severity level (can be escalated).
    source:
        What created this incident (e.g., "alerting", "operator", "automation").
    created_at:
        Unix timestamp of creation.
    updated_at:
        Unix timestamp of last modification.
    target_ids:
        Target IDs involved in this incident.
    zone_id:
        Primary zone where the incident occurred.
    alert_ids:
        Alert record IDs that contributed to this incident.
    tags:
        Arbitrary tags for filtering/grouping.
    description:
        Longer description of the incident.
    timeline:
        Ordered timeline of events.
    resources:
        Resources assigned to this incident.
    resolution:
        How the incident was resolved (None if unresolved).
    """
    incident_id: str
    title: str
    state: IncidentState = IncidentState.DETECTED
    severity: IncidentSeverity = IncidentSeverity.MEDIUM
    source: str = "system"
    created_at: float = 0.0
    updated_at: float = 0.0
    target_ids: list[str] = field(default_factory=list)
    zone_id: str = ""
    alert_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    description: str = ""
    timeline: Timeline = field(default_factory=Timeline)
    resources: list[AssignedResource] = field(default_factory=list)
    resolution: Resolution | None = None

    @property
    def is_open(self) -> bool:
        """True if the incident is not resolved or closed."""
        return self.state not in (IncidentState.RESOLVED, IncidentState.CLOSED)

    @property
    def active_resources(self) -> list[AssignedResource]:
        """Return resources that are currently assigned (not released)."""
        return [r for r in self.resources if r.is_active]

    @property
    def duration_seconds(self) -> float:
        """Elapsed time from creation to resolution (or now if still open)."""
        end = self.updated_at if not self.is_open else time.time()
        return max(0.0, end - self.created_at)

    def can_transition_to(self, new_state: IncidentState) -> bool:
        """Check if transitioning to new_state is valid."""
        return new_state in _VALID_TRANSITIONS.get(self.state, set())

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "state": self.state.value,
            "severity": self.severity.value,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "target_ids": list(self.target_ids),
            "zone_id": self.zone_id,
            "alert_ids": list(self.alert_ids),
            "tags": list(self.tags),
            "description": self.description,
            "timeline": self.timeline.to_list(),
            "resources": [r.to_dict() for r in self.resources],
            "resolution": self.resolution.to_dict() if self.resolution else None,
            "is_open": self.is_open,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Incident:
        state = data.get("state", "detected")
        if isinstance(state, str):
            state = IncidentState(state)

        severity = data.get("severity", "medium")
        if isinstance(severity, str):
            severity = IncidentSeverity(severity)

        resolution = None
        if data.get("resolution"):
            resolution = Resolution.from_dict(data["resolution"])

        inc = cls(
            incident_id=data["incident_id"],
            title=data.get("title", ""),
            state=state,
            severity=severity,
            source=data.get("source", "system"),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            target_ids=data.get("target_ids", []),
            zone_id=data.get("zone_id", ""),
            alert_ids=data.get("alert_ids", []),
            tags=data.get("tags", []),
            description=data.get("description", ""),
            timeline=Timeline.from_list(data.get("timeline", [])),
            resources=[
                AssignedResource.from_dict(r)
                for r in data.get("resources", [])
            ],
            resolution=resolution,
        )
        return inc


# ---------------------------------------------------------------------------
# IncidentManager
# ---------------------------------------------------------------------------

class IncidentManager:
    """Create, update, escalate, and resolve incidents.

    Thread-safe. All public methods acquire the internal lock.

    Parameters
    ----------
    event_bus:
        Optional EventBus instance for publishing incident lifecycle events.
    max_incidents:
        Maximum number of incidents to retain (oldest closed incidents
        are evicted first).
    on_incident_created:
        Optional callback invoked when a new incident is created.
    on_incident_resolved:
        Optional callback invoked when an incident is resolved.
    auto_create_from_alerts:
        If True and an AlertEngine is connected, automatically create
        incidents from alert escalations.
    """

    def __init__(
        self,
        event_bus=None,
        *,
        max_incidents: int = 10000,
        on_incident_created: Callable[[Incident], None] | None = None,
        on_incident_resolved: Callable[[Incident], None] | None = None,
        auto_create_from_alerts: bool = True,
    ) -> None:
        self._event_bus = event_bus
        self._lock = threading.Lock()
        self._max_incidents = max_incidents
        self._on_incident_created = on_incident_created
        self._on_incident_resolved = on_incident_resolved
        self._auto_create_from_alerts = auto_create_from_alerts

        # Storage: incident_id -> Incident
        self._incidents: dict[str, Incident] = {}

        # Alert engine reference for integration
        self._alert_engine = None

        # Counters
        self._total_created = 0
        self._total_resolved = 0
        self._total_escalations = 0

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        title: str,
        severity: IncidentSeverity = IncidentSeverity.MEDIUM,
        source: str = "system",
        target_ids: list[str] | None = None,
        zone_id: str = "",
        alert_ids: list[str] | None = None,
        tags: list[str] | None = None,
        description: str = "",
    ) -> Incident:
        """Create a new incident in the DETECTED state.

        Parameters
        ----------
        title:
            Short human-readable description.
        severity:
            Initial severity level.
        source:
            What created this incident.
        target_ids:
            Target IDs involved.
        zone_id:
            Primary zone where the incident occurred.
        alert_ids:
            Alert record IDs that contributed.
        tags:
            Arbitrary tags.
        description:
            Longer description.

        Returns the created Incident.
        """
        now = time.time()
        incident_id = f"inc_{uuid.uuid4().hex[:12]}"

        incident = Incident(
            incident_id=incident_id,
            title=title,
            state=IncidentState.DETECTED,
            severity=severity,
            source=source,
            created_at=now,
            updated_at=now,
            target_ids=list(target_ids or []),
            zone_id=zone_id,
            alert_ids=list(alert_ids or []),
            tags=list(tags or []),
            description=description,
        )

        # Initial timeline entry
        incident.timeline.add(
            description=f"Incident created: {title}",
            author=source,
            entry_type="state_change",
            metadata={"new_state": IncidentState.DETECTED.value, "severity": severity.value},
            timestamp=now,
        )

        with self._lock:
            self._incidents[incident_id] = incident
            self._total_created += 1
            self._evict_if_needed()

        logger.info(
            "Incident created: %s [%s] %s",
            incident_id, severity.value, title,
        )

        # Publish event
        self._publish_event("incident.created", incident)

        # Callback
        if self._on_incident_created:
            try:
                self._on_incident_created(incident)
            except Exception:
                logger.debug("on_incident_created callback failed", exc_info=True)

        return incident

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def transition(
        self,
        incident_id: str,
        new_state: IncidentState,
        *,
        reason: str = "",
        author: str = "system",
    ) -> Incident | None:
        """Transition an incident to a new state.

        Returns the updated Incident, or None if the incident was not
        found or the transition is invalid.
        """
        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident is None:
                logger.warning("Incident not found: %s", incident_id)
                return None

            if not incident.can_transition_to(new_state):
                logger.warning(
                    "Invalid transition: %s -> %s for incident %s",
                    incident.state.value, new_state.value, incident_id,
                )
                return None

            old_state = incident.state
            incident.state = new_state
            incident.updated_at = time.time()

            incident.timeline.add(
                description=f"State changed: {old_state.value} -> {new_state.value}"
                + (f" — {reason}" if reason else ""),
                author=author,
                entry_type="state_change",
                metadata={
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "reason": reason,
                },
            )

        logger.info(
            "Incident %s: %s -> %s%s",
            incident_id, old_state.value, new_state.value,
            f" ({reason})" if reason else "",
        )

        self._publish_event("incident.state_changed", incident)
        return incident

    def investigate(self, incident_id: str, reason: str = "", author: str = "system") -> Incident | None:
        """Move incident to INVESTIGATING state."""
        return self.transition(incident_id, IncidentState.INVESTIGATING, reason=reason, author=author)

    def respond(self, incident_id: str, reason: str = "", author: str = "system") -> Incident | None:
        """Move incident to RESPONDING state."""
        return self.transition(incident_id, IncidentState.RESPONDING, reason=reason, author=author)

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

    def escalate(
        self,
        incident_id: str,
        new_severity: IncidentSeverity,
        *,
        reason: str = "",
        author: str = "system",
    ) -> Incident | None:
        """Escalate an incident's severity.

        Only escalates upward (LOW -> MEDIUM -> HIGH -> CRITICAL).
        Returns the updated Incident, or None if not found or not escalatable.
        """
        _severity_order = {
            IncidentSeverity.LOW: 0,
            IncidentSeverity.MEDIUM: 1,
            IncidentSeverity.HIGH: 2,
            IncidentSeverity.CRITICAL: 3,
        }

        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident is None:
                return None

            if not incident.is_open:
                logger.warning("Cannot escalate closed incident: %s", incident_id)
                return None

            old_severity = incident.severity
            if _severity_order.get(new_severity, 0) <= _severity_order.get(old_severity, 0):
                logger.debug(
                    "Ignoring non-escalation: %s -> %s",
                    old_severity.value, new_severity.value,
                )
                return incident

            incident.severity = new_severity
            incident.updated_at = time.time()
            self._total_escalations += 1

            incident.timeline.add(
                description=f"Escalated: {old_severity.value} -> {new_severity.value}"
                + (f" — {reason}" if reason else ""),
                author=author,
                entry_type="escalation",
                metadata={
                    "old_severity": old_severity.value,
                    "new_severity": new_severity.value,
                    "reason": reason,
                },
            )

        logger.info(
            "Incident %s escalated: %s -> %s",
            incident_id, old_severity.value, new_severity.value,
        )

        self._publish_event("incident.escalated", incident)
        return incident

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        incident_id: str,
        summary: str,
        *,
        resolution_type: str = "other",
        resolved_by: str = "system",
        lessons_learned: str = "",
        follow_up_actions: list[str] | None = None,
    ) -> Incident | None:
        """Resolve an incident.

        Transitions to RESOLVED state, attaches a Resolution, and
        releases all active resources.

        Returns the updated Incident, or None if not found or already closed.
        """
        now = time.time()

        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident is None:
                return None

            if not incident.is_open:
                logger.warning("Cannot resolve non-open incident: %s", incident_id)
                return None

            old_state = incident.state
            incident.state = IncidentState.RESOLVED
            incident.updated_at = now

            incident.resolution = Resolution(
                summary=summary,
                resolution_type=resolution_type,
                resolved_by=resolved_by,
                resolved_at=now,
                lessons_learned=lessons_learned,
                follow_up_actions=list(follow_up_actions or []),
            )

            # Release all active resources
            for resource in incident.resources:
                if resource.is_active:
                    resource.release(timestamp=now)

            incident.timeline.add(
                description=f"Resolved: {summary}",
                author=resolved_by,
                entry_type="resolution",
                metadata={
                    "old_state": old_state.value,
                    "resolution_type": resolution_type,
                    "lessons_learned": lessons_learned,
                },
                timestamp=now,
            )

            self._total_resolved += 1

        logger.info("Incident %s resolved: %s", incident_id, summary)

        self._publish_event("incident.resolved", incident)

        # Callback
        if self._on_incident_resolved:
            try:
                self._on_incident_resolved(incident)
            except Exception:
                logger.debug("on_incident_resolved callback failed", exc_info=True)

        return incident

    def close(self, incident_id: str, author: str = "system") -> Incident | None:
        """Close an incident (terminal state). Usually done after resolution."""
        return self.transition(incident_id, IncidentState.CLOSED, reason="Closed", author=author)

    def reopen(self, incident_id: str, reason: str = "", author: str = "system") -> Incident | None:
        """Reopen a resolved incident back to INVESTIGATING."""
        return self.transition(incident_id, IncidentState.INVESTIGATING, reason=reason or "Reopened", author=author)

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def add_timeline_entry(
        self,
        incident_id: str,
        description: str,
        *,
        author: str = "system",
        entry_type: str = "note",
        metadata: dict[str, Any] | None = None,
    ) -> TimelineEntry | None:
        """Add a timeline entry to an incident.

        Returns the created TimelineEntry, or None if incident not found.
        """
        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident is None:
                return None

            entry = incident.timeline.add(
                description=description,
                author=author,
                entry_type=entry_type,
                metadata=metadata,
            )
            incident.updated_at = time.time()

        return entry

    def get_timeline(self, incident_id: str) -> list[TimelineEntry]:
        """Return the full timeline for an incident."""
        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident is None:
                return []
            return incident.timeline.get_entries()

    # ------------------------------------------------------------------
    # Resource assignment
    # ------------------------------------------------------------------

    def assign_resource(
        self,
        incident_id: str,
        resource_id: str,
        *,
        resource_type: str = "other",
        role: str = "support",
        notes: str = "",
    ) -> AssignedResource | None:
        """Assign a resource to an incident.

        Returns the AssignedResource, or None if incident not found.
        """
        now = time.time()

        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident is None:
                return None

            # Check if already assigned
            for existing in incident.resources:
                if existing.resource_id == resource_id and existing.is_active:
                    logger.debug(
                        "Resource %s already assigned to %s",
                        resource_id, incident_id,
                    )
                    return existing

            resource = AssignedResource(
                resource_id=resource_id,
                resource_type=resource_type,
                role=role,
                assigned_at=now,
                notes=notes,
            )
            incident.resources.append(resource)
            incident.updated_at = now

            incident.timeline.add(
                description=f"Resource assigned: {resource_id} ({role})",
                author="system",
                entry_type="assignment",
                metadata={
                    "resource_id": resource_id,
                    "resource_type": resource_type,
                    "role": role,
                },
                timestamp=now,
            )

        logger.info(
            "Resource %s assigned to incident %s as %s",
            resource_id, incident_id, role,
        )

        return resource

    def release_resource(
        self,
        incident_id: str,
        resource_id: str,
    ) -> bool:
        """Release a resource from an incident.

        Returns True if the resource was found and released.
        """
        now = time.time()

        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident is None:
                return False

            for resource in incident.resources:
                if resource.resource_id == resource_id and resource.is_active:
                    resource.release(timestamp=now)
                    incident.updated_at = now

                    incident.timeline.add(
                        description=f"Resource released: {resource_id}",
                        author="system",
                        entry_type="assignment",
                        metadata={"resource_id": resource_id, "action": "released"},
                        timestamp=now,
                    )

                    logger.info(
                        "Resource %s released from incident %s",
                        resource_id, incident_id,
                    )
                    return True

        return False

    # ------------------------------------------------------------------
    # Alert integration
    # ------------------------------------------------------------------

    def add_alert(self, incident_id: str, alert_id: str) -> bool:
        """Link an alert record to an incident.

        Returns True if the alert was added (or already linked).
        """
        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident is None:
                return False
            if alert_id not in incident.alert_ids:
                incident.alert_ids.append(alert_id)
                incident.updated_at = time.time()
                incident.timeline.add(
                    description=f"Alert linked: {alert_id}",
                    author="alerting",
                    entry_type="alert",
                    metadata={"alert_id": alert_id},
                )
            return True

    def connect_alert_engine(self, alert_engine) -> None:
        """Connect to an AlertEngine to auto-create incidents from escalations.

        Registers an action handler on the AlertEngine so that ESCALATE
        actions automatically create incidents. Also subscribes to the
        EventBus (if available) for alert.escalation events.
        """
        self._alert_engine = alert_engine

        if not self._auto_create_from_alerts:
            return

        # Import here to avoid circular imports
        try:
            from tritium_lib.alerting import DispatchAction
        except ImportError:
            logger.debug("Could not import alerting module for integration")
            return

        def _on_escalation(record) -> None:
            """Create an incident from an escalation alert record."""
            severity_map = {
                "critical": IncidentSeverity.CRITICAL,
                "error": IncidentSeverity.HIGH,
                "warning": IncidentSeverity.MEDIUM,
                "info": IncidentSeverity.LOW,
                "debug": IncidentSeverity.LOW,
            }
            sev = severity_map.get(
                getattr(record, "severity", "medium"),
                IncidentSeverity.MEDIUM,
            )

            target_ids = []
            target_id = getattr(record, "target_id", "")
            if target_id:
                target_ids.append(target_id)

            self.create(
                title=getattr(record, "message", getattr(record, "rule_name", "Alert escalation")),
                severity=sev,
                source="alerting",
                target_ids=target_ids,
                zone_id=getattr(record, "zone_id", ""),
                alert_ids=[getattr(record, "record_id", "")],
                tags=["auto-created", "from-alert"],
            )

        alert_engine.register_action_handler(DispatchAction.ESCALATE, _on_escalation)
        logger.info("Incident manager connected to AlertEngine")

    def create_from_alert(
        self,
        alert_record,
    ) -> Incident:
        """Manually create an incident from an AlertRecord.

        Parameters
        ----------
        alert_record:
            An AlertRecord (or any object with record_id, rule_name,
            severity, message, target_id, zone_id attributes).

        Returns the created Incident.
        """
        severity_map = {
            "critical": IncidentSeverity.CRITICAL,
            "error": IncidentSeverity.HIGH,
            "warning": IncidentSeverity.MEDIUM,
            "info": IncidentSeverity.LOW,
            "debug": IncidentSeverity.LOW,
        }
        sev = severity_map.get(
            getattr(alert_record, "severity", "medium"),
            IncidentSeverity.MEDIUM,
        )

        target_ids = []
        target_id = getattr(alert_record, "target_id", "")
        if target_id:
            target_ids.append(target_id)

        return self.create(
            title=getattr(alert_record, "message", getattr(alert_record, "rule_name", "Alert")),
            severity=sev,
            source="alerting",
            target_ids=target_ids,
            zone_id=getattr(alert_record, "zone_id", ""),
            alert_ids=[getattr(alert_record, "record_id", "")],
            tags=["from-alert"],
        )

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get(self, incident_id: str) -> Incident | None:
        """Get an incident by ID."""
        with self._lock:
            return self._incidents.get(incident_id)

    def get_all(
        self,
        *,
        state: IncidentState | None = None,
        severity: IncidentSeverity | None = None,
        source: str = "",
        zone_id: str = "",
        target_id: str = "",
        tag: str = "",
        open_only: bool = False,
        limit: int = 100,
    ) -> list[Incident]:
        """Query incidents with optional filtering.

        Parameters
        ----------
        state:
            Filter by exact state.
        severity:
            Filter by exact severity.
        source:
            Filter by source.
        zone_id:
            Filter by zone ID.
        target_id:
            Filter by target ID (incident must involve this target).
        tag:
            Filter by tag (incident must have this tag).
        open_only:
            If True, only return open incidents.
        limit:
            Maximum results.

        Returns incidents sorted by updated_at descending (newest first).
        """
        with self._lock:
            results = list(self._incidents.values())

        if state is not None:
            results = [i for i in results if i.state == state]
        if severity is not None:
            results = [i for i in results if i.severity == severity]
        if source:
            results = [i for i in results if i.source == source]
        if zone_id:
            results = [i for i in results if i.zone_id == zone_id]
        if target_id:
            results = [i for i in results if target_id in i.target_ids]
        if tag:
            results = [i for i in results if tag in i.tags]
        if open_only:
            results = [i for i in results if i.is_open]

        results.sort(key=lambda i: i.updated_at, reverse=True)
        return results[:limit]

    def get_open(self, limit: int = 100) -> list[Incident]:
        """Shortcut to get all open incidents."""
        return self.get_all(open_only=True, limit=limit)

    def get_by_target(self, target_id: str, limit: int = 100) -> list[Incident]:
        """Get all incidents involving a specific target."""
        return self.get_all(target_id=target_id, limit=limit)

    def get_stats(self) -> dict[str, Any]:
        """Return incident management statistics."""
        with self._lock:
            by_state: dict[str, int] = {}
            by_severity: dict[str, int] = {}
            for inc in self._incidents.values():
                by_state[inc.state.value] = by_state.get(inc.state.value, 0) + 1
                by_severity[inc.severity.value] = by_severity.get(inc.severity.value, 0) + 1

            return {
                "total_incidents": len(self._incidents),
                "total_created": self._total_created,
                "total_resolved": self._total_resolved,
                "total_escalations": self._total_escalations,
                "open_count": sum(
                    1 for i in self._incidents.values() if i.is_open
                ),
                "by_state": by_state,
                "by_severity": by_severity,
                "max_incidents": self._max_incidents,
                "alert_engine_connected": self._alert_engine is not None,
            }

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def merge(
        self,
        primary_id: str,
        secondary_id: str,
        *,
        author: str = "system",
    ) -> Incident | None:
        """Merge a secondary incident into a primary incident.

        The secondary incident is resolved as "merged" and its target IDs,
        alert IDs, and timeline entries are copied to the primary.

        Returns the primary Incident, or None if either not found.
        """
        with self._lock:
            primary = self._incidents.get(primary_id)
            secondary = self._incidents.get(secondary_id)

            if primary is None or secondary is None:
                return None

            # Copy target IDs
            for tid in secondary.target_ids:
                if tid not in primary.target_ids:
                    primary.target_ids.append(tid)

            # Copy alert IDs
            for aid in secondary.alert_ids:
                if aid not in primary.alert_ids:
                    primary.alert_ids.append(aid)

            # Copy tags
            for tag in secondary.tags:
                if tag not in primary.tags:
                    primary.tags.append(tag)

            now = time.time()
            primary.updated_at = now

            primary.timeline.add(
                description=f"Merged with incident {secondary_id}: {secondary.title}",
                author=author,
                entry_type="note",
                metadata={"merged_incident_id": secondary_id},
                timestamp=now,
            )

            # Resolve the secondary
            secondary.state = IncidentState.RESOLVED
            secondary.updated_at = now
            secondary.resolution = Resolution(
                summary=f"Merged into incident {primary_id}",
                resolution_type="deferred",
                resolved_by=author,
                resolved_at=now,
            )
            secondary.timeline.add(
                description=f"Merged into incident {primary_id}",
                author=author,
                entry_type="resolution",
                metadata={"merged_into": primary_id},
                timestamp=now,
            )

            self._total_resolved += 1

        logger.info("Incident %s merged into %s", secondary_id, primary_id)
        return primary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _publish_event(self, topic: str, incident: Incident) -> None:
        """Publish an incident lifecycle event to the EventBus."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish(
                topic,
                data=incident.to_dict(),
                source="incident",
            )
        except Exception:
            logger.debug("Failed to publish %s", topic, exc_info=True)

    def _evict_if_needed(self) -> None:
        """Evict oldest closed/resolved incidents if over capacity.

        Must be called while holding self._lock.
        """
        if len(self._incidents) <= self._max_incidents:
            return

        # Find closed incidents, sorted oldest first
        closed = sorted(
            [i for i in self._incidents.values() if not i.is_open],
            key=lambda i: i.updated_at,
        )

        to_remove = len(self._incidents) - self._max_incidents
        for i in range(min(to_remove, len(closed))):
            del self._incidents[closed[i].incident_id]

    def reset(self) -> None:
        """Clear all state."""
        with self._lock:
            self._incidents.clear()
            self._total_created = 0
            self._total_resolved = 0
            self._total_escalations = 0


__all__ = [
    "Incident",
    "IncidentManager",
    "IncidentSeverity",
    "IncidentState",
    "Timeline",
    "TimelineEntry",
    "AssignedResource",
    "Resolution",
]
