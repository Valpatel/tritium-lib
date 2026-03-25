# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.mission — Plan and coordinate surveillance/security operations.

Manages the full lifecycle of missions: planning -> briefing -> active ->
paused -> completed/aborted. Integrates with the AlertEngine for mission-scoped
alerts, the EventBus for lifecycle events, and the geo module for area-of-interest
definitions.

Core classes:
  - Mission            — a named operation with objectives, resources, constraints
  - MissionObjective   — what to achieve (surveil area, track target, monitor zone)
  - ResourceAllocation — assign sensors/devices to objectives
  - MissionSchedule    — when to start/end, shift rotations
  - MissionBrief       — human-readable summary of mission parameters
  - MissionStatus      — live status tracking during execution
  - MissionPlanner     — create, manage, and track missions

Mission types:
  - SURVEILLANCE   — monitor an area for activity
  - TRACKING       — follow a specific target
  - PERIMETER      — maintain perimeter security
  - INVESTIGATION  — focused investigation of an incident
  - PATROL         — recurring patrol of checkpoints

Quick start::

    from tritium_lib.mission import MissionPlanner, MissionType, MissionObjective

    planner = MissionPlanner()

    # Create a surveillance mission
    mission = planner.create_mission(
        name="Overwatch Alpha",
        mission_type=MissionType.SURVEILLANCE,
        objectives=[
            MissionObjective(
                description="Monitor parking lot for unauthorized vehicles",
                area_id="zone-parking-a",
            ),
        ],
    )

    # Assign sensors
    planner.allocate_resource(
        mission.mission_id,
        resource_id="cam-01",
        resource_type="camera",
        objective_index=0,
    )

    # Activate the mission
    planner.activate(mission.mission_id)

    # Generate a brief
    brief = planner.generate_brief(mission.mission_id)
    print(brief.summary)

    # Check live status
    status = planner.get_status(mission.mission_id)

    # Complete the mission
    planner.complete(mission.mission_id, summary="No incidents observed.")

Integration with AlertEngine::

    from tritium_lib.alerting import AlertEngine
    from tritium_lib.mission import MissionPlanner

    planner = MissionPlanner(alert_engine=alert_engine, event_bus=bus)
    # Alerts scoped to mission areas are automatically tracked

Architecture
------------
- **MissionType**         — enum: SURVEILLANCE, TRACKING, PERIMETER, INVESTIGATION, PATROL
- **MissionState**        — enum lifecycle: PLANNING -> BRIEFED -> ACTIVE -> PAUSED -> COMPLETED / ABORTED
- **MissionPriority**     — enum: LOW, MEDIUM, HIGH, CRITICAL
- **MissionObjective**    — what to achieve, with optional area/target constraints
- **ObjectiveStatus**     — tracking progress on individual objectives
- **ResourceAllocation**  — sensor/device assigned to an objective with role and time window
- **MissionSchedule**     — start/end times, shift rotations, duration constraints
- **MissionConstraint**   — operational constraints (ROE, weather, comms, etc.)
- **MissionBrief**        — human-readable summary for operators
- **MissionStatus**       — real-time aggregated status of a running mission
- **Mission**             — the full mission record
- **MissionPlanner**      — CRUD + lifecycle operations, resource allocation, querying
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("tritium.mission")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MissionType(str, Enum):
    """Types of missions the system can plan and execute."""
    SURVEILLANCE = "surveillance"       # Monitor an area for activity
    TRACKING = "tracking"               # Follow a specific target
    PERIMETER = "perimeter"             # Maintain perimeter security
    INVESTIGATION = "investigation"     # Focused investigation of an incident
    PATROL = "patrol"                   # Recurring patrol of checkpoints


class MissionState(str, Enum):
    """Lifecycle states for a mission."""
    PLANNING = "planning"       # Being defined, objectives set
    BRIEFED = "briefed"         # Brief generated, ready for activation
    ACTIVE = "active"           # Currently executing
    PAUSED = "paused"           # Temporarily suspended
    COMPLETED = "completed"     # Successfully finished
    ABORTED = "aborted"         # Terminated before completion


class MissionPriority(str, Enum):
    """Priority classification for missions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Valid state transitions
_VALID_TRANSITIONS: dict[MissionState, set[MissionState]] = {
    MissionState.PLANNING: {MissionState.BRIEFED, MissionState.ABORTED},
    MissionState.BRIEFED: {MissionState.ACTIVE, MissionState.PLANNING, MissionState.ABORTED},
    MissionState.ACTIVE: {MissionState.PAUSED, MissionState.COMPLETED, MissionState.ABORTED},
    MissionState.PAUSED: {MissionState.ACTIVE, MissionState.ABORTED, MissionState.COMPLETED},
    MissionState.COMPLETED: set(),  # terminal
    MissionState.ABORTED: set(),    # terminal
}


# ---------------------------------------------------------------------------
# MissionObjective — what to achieve
# ---------------------------------------------------------------------------

@dataclass
class MissionObjective:
    """A single objective within a mission.

    Objectives define what the mission aims to achieve. Each objective
    can be scoped to an area (zone/geofence) or a specific target.

    Attributes
    ----------
    description:
        Human-readable description of the objective.
    area_id:
        Optional zone/geofence ID this objective covers.
    target_ids:
        Optional list of target IDs this objective focuses on.
    checkpoint_coords:
        Optional list of (lat, lng) waypoints for patrol objectives.
    priority:
        Priority of this objective within the mission.
    required_sensors:
        Types of sensors needed (e.g., "camera", "ble", "acoustic").
    success_criteria:
        Description of what constitutes success for this objective.
    metadata:
        Arbitrary key-value data.
    """
    description: str
    area_id: str = ""
    target_ids: list[str] = field(default_factory=list)
    checkpoint_coords: list[tuple[float, float]] = field(default_factory=list)
    priority: MissionPriority = MissionPriority.MEDIUM
    required_sensors: list[str] = field(default_factory=list)
    success_criteria: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "area_id": self.area_id,
            "target_ids": list(self.target_ids),
            "checkpoint_coords": [list(c) for c in self.checkpoint_coords],
            "priority": self.priority.value,
            "required_sensors": list(self.required_sensors),
            "success_criteria": self.success_criteria,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> MissionObjective:
        return cls(
            description=data.get("description", ""),
            area_id=data.get("area_id", ""),
            target_ids=data.get("target_ids", []),
            checkpoint_coords=[
                tuple(c) for c in data.get("checkpoint_coords", [])
            ],
            priority=MissionPriority(data.get("priority", "medium")),
            required_sensors=data.get("required_sensors", []),
            success_criteria=data.get("success_criteria", ""),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# ObjectiveStatus — live progress on an objective
# ---------------------------------------------------------------------------

@dataclass
class ObjectiveStatus:
    """Real-time status for a single objective.

    Attributes
    ----------
    objective_index:
        Index of the objective in the mission's objective list.
    status:
        Current status: "pending", "active", "completed", "failed".
    progress_pct:
        Estimated progress percentage (0-100).
    notes:
        Operator or system notes on progress.
    started_at:
        When this objective started being worked.
    completed_at:
        When this objective was completed (0 if not yet).
    detections:
        Number of relevant detections/sightings during this objective.
    alerts_fired:
        Number of alerts fired related to this objective.
    """
    objective_index: int
    status: str = "pending"
    progress_pct: float = 0.0
    notes: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    detections: int = 0
    alerts_fired: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_index": self.objective_index,
            "status": self.status,
            "progress_pct": self.progress_pct,
            "notes": self.notes,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "detections": self.detections,
            "alerts_fired": self.alerts_fired,
        }


# ---------------------------------------------------------------------------
# ResourceAllocation — assign sensors/devices to objectives
# ---------------------------------------------------------------------------

@dataclass
class ResourceAllocation:
    """A sensor or device assigned to a mission objective.

    Tracks which resource is allocated, what role it plays, which
    objective it supports, and the time window of the allocation.

    Attributes
    ----------
    allocation_id:
        Unique ID for this allocation.
    resource_id:
        ID of the sensor, device, or unit being allocated.
    resource_type:
        Type of resource (e.g., "camera", "drone", "ble_scanner", "operator").
    objective_index:
        Index of the objective this resource supports (-1 = mission-wide).
    role:
        Role this resource plays (e.g., "primary", "backup", "relay").
    assigned_at:
        When this allocation was made.
    released_at:
        When the resource was released (0.0 if still allocated).
    shift_start:
        Scheduled start time for this allocation (0 = immediate).
    shift_end:
        Scheduled end time for this allocation (0 = until mission ends).
    metadata:
        Arbitrary key-value data.
    """
    allocation_id: str = ""
    resource_id: str = ""
    resource_type: str = ""
    objective_index: int = -1
    role: str = "primary"
    assigned_at: float = 0.0
    released_at: float = 0.0
    shift_start: float = 0.0
    shift_end: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.allocation_id:
            self.allocation_id = uuid.uuid4().hex[:12]
        if self.assigned_at == 0.0:
            self.assigned_at = time.time()

    @property
    def is_active(self) -> bool:
        """True if the resource is still allocated (not released)."""
        return self.released_at == 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocation_id": self.allocation_id,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "objective_index": self.objective_index,
            "role": self.role,
            "assigned_at": self.assigned_at,
            "released_at": self.released_at,
            "shift_start": self.shift_start,
            "shift_end": self.shift_end,
            "is_active": self.is_active,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResourceAllocation:
        return cls(
            allocation_id=data.get("allocation_id", ""),
            resource_id=data.get("resource_id", ""),
            resource_type=data.get("resource_type", ""),
            objective_index=data.get("objective_index", -1),
            role=data.get("role", "primary"),
            assigned_at=data.get("assigned_at", 0.0),
            released_at=data.get("released_at", 0.0),
            shift_start=data.get("shift_start", 0.0),
            shift_end=data.get("shift_end", 0.0),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# MissionSchedule — when to start/end, shift rotations
# ---------------------------------------------------------------------------

@dataclass
class MissionSchedule:
    """Scheduling parameters for a mission.

    Defines the time window for the mission, shift rotations, and
    any recurring schedule for patrol-type missions.

    Attributes
    ----------
    planned_start:
        Planned start time (Unix timestamp). 0 = start immediately on activation.
    planned_end:
        Planned end time (Unix timestamp). 0 = no planned end.
    max_duration_hours:
        Maximum mission duration in hours. 0 = unlimited.
    shift_duration_hours:
        Duration of each shift rotation in hours. 0 = no rotation.
    shifts:
        List of shift definitions, each with label and resource overrides.
    recurring:
        If True, the mission repeats (for patrols). False = one-time.
    recurrence_interval_hours:
        Hours between recurring patrol starts.
    """
    planned_start: float = 0.0
    planned_end: float = 0.0
    max_duration_hours: float = 0.0
    shift_duration_hours: float = 0.0
    shifts: list[dict[str, Any]] = field(default_factory=list)
    recurring: bool = False
    recurrence_interval_hours: float = 0.0

    @property
    def has_time_window(self) -> bool:
        """True if the schedule defines a finite time window."""
        return self.planned_start > 0 or self.planned_end > 0

    @property
    def duration_seconds(self) -> float:
        """Planned duration in seconds, or 0 if open-ended."""
        if self.planned_start > 0 and self.planned_end > 0:
            return max(0.0, self.planned_end - self.planned_start)
        if self.max_duration_hours > 0:
            return self.max_duration_hours * 3600.0
        return 0.0

    def is_within_window(self, ts: float | None = None) -> bool:
        """Check if a timestamp falls within the scheduled window.

        Parameters
        ----------
        ts:
            Timestamp to check. Defaults to now.

        Returns True if no window is defined (always valid).
        """
        if not self.has_time_window:
            return True
        now = ts if ts is not None else time.time()
        if self.planned_start > 0 and now < self.planned_start:
            return False
        if self.planned_end > 0 and now > self.planned_end:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "planned_start": self.planned_start,
            "planned_end": self.planned_end,
            "max_duration_hours": self.max_duration_hours,
            "shift_duration_hours": self.shift_duration_hours,
            "shifts": list(self.shifts),
            "recurring": self.recurring,
            "recurrence_interval_hours": self.recurrence_interval_hours,
            "has_time_window": self.has_time_window,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MissionSchedule:
        return cls(
            planned_start=data.get("planned_start", 0.0),
            planned_end=data.get("planned_end", 0.0),
            max_duration_hours=data.get("max_duration_hours", 0.0),
            shift_duration_hours=data.get("shift_duration_hours", 0.0),
            shifts=data.get("shifts", []),
            recurring=data.get("recurring", False),
            recurrence_interval_hours=data.get("recurrence_interval_hours", 0.0),
        )


# ---------------------------------------------------------------------------
# MissionConstraint — operational constraints
# ---------------------------------------------------------------------------

@dataclass
class MissionConstraint:
    """An operational constraint on a mission.

    Constraints capture rules of engagement, weather limitations,
    communication requirements, or any other operational boundary.

    Attributes
    ----------
    constraint_type:
        Category: "roe" (rules of engagement), "weather", "comms",
        "legal", "resource", "geographic", "temporal".
    description:
        Human-readable description of the constraint.
    severity:
        How critical: "advisory", "mandatory", "hard_stop".
    parameters:
        Constraint-specific parameters (e.g., max wind speed, frequency band).
    """
    constraint_type: str
    description: str
    severity: str = "advisory"
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_type": self.constraint_type,
            "description": self.description,
            "severity": self.severity,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: dict) -> MissionConstraint:
        return cls(
            constraint_type=data.get("constraint_type", ""),
            description=data.get("description", ""),
            severity=data.get("severity", "advisory"),
            parameters=data.get("parameters", {}),
        )


# ---------------------------------------------------------------------------
# MissionBrief — human-readable summary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MissionBrief:
    """A human-readable briefing summary of a mission.

    Generated by MissionPlanner.generate_brief(). Contains all the
    information an operator needs to understand and execute the mission.

    Attributes
    ----------
    mission_id:
        ID of the mission this brief covers.
    mission_name:
        Name of the mission.
    mission_type:
        Type of mission.
    priority:
        Mission priority.
    summary:
        High-level summary paragraph.
    objectives_text:
        Formatted text describing all objectives.
    resources_text:
        Formatted text describing resource allocations.
    schedule_text:
        Formatted text describing the schedule.
    constraints_text:
        Formatted text describing constraints.
    generated_at:
        When this brief was generated.
    """
    mission_id: str
    mission_name: str
    mission_type: str
    priority: str
    summary: str
    objectives_text: str
    resources_text: str
    schedule_text: str
    constraints_text: str
    generated_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "mission_name": self.mission_name,
            "mission_type": self.mission_type,
            "priority": self.priority,
            "summary": self.summary,
            "objectives_text": self.objectives_text,
            "resources_text": self.resources_text,
            "schedule_text": self.schedule_text,
            "constraints_text": self.constraints_text,
            "generated_at": self.generated_at,
        }

    @property
    def full_text(self) -> str:
        """Return the complete brief as a formatted text block."""
        lines = [
            f"MISSION BRIEF: {self.mission_name}",
            f"Type: {self.mission_type.upper()} | Priority: {self.priority.upper()}",
            f"ID: {self.mission_id}",
            "",
            "SUMMARY",
            self.summary,
            "",
            "OBJECTIVES",
            self.objectives_text,
            "",
            "RESOURCES",
            self.resources_text,
            "",
            "SCHEDULE",
            self.schedule_text,
            "",
            "CONSTRAINTS",
            self.constraints_text,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MissionStatus — live status during execution
# ---------------------------------------------------------------------------

@dataclass
class MissionStatus:
    """Real-time aggregated status of a running mission.

    Combines objective progress, resource health, elapsed time,
    and alert/detection counts into a single status snapshot.

    Attributes
    ----------
    mission_id:
        ID of the mission.
    state:
        Current mission state.
    elapsed_seconds:
        Seconds since mission activation.
    objective_statuses:
        Per-objective status list.
    active_resources:
        Count of currently allocated resources.
    total_resources:
        Total resources ever allocated.
    total_detections:
        Sum of detections across all objectives.
    total_alerts:
        Sum of alerts fired across all objectives.
    overall_progress_pct:
        Weighted average of objective progress.
    health:
        Overall health: "green", "yellow", "red".
    timestamp:
        When this status snapshot was generated.
    """
    mission_id: str
    state: str
    elapsed_seconds: float = 0.0
    objective_statuses: list[ObjectiveStatus] = field(default_factory=list)
    active_resources: int = 0
    total_resources: int = 0
    total_detections: int = 0
    total_alerts: int = 0
    overall_progress_pct: float = 0.0
    health: str = "green"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "state": self.state,
            "elapsed_seconds": self.elapsed_seconds,
            "objective_statuses": [os.to_dict() for os in self.objective_statuses],
            "active_resources": self.active_resources,
            "total_resources": self.total_resources,
            "total_detections": self.total_detections,
            "total_alerts": self.total_alerts,
            "overall_progress_pct": self.overall_progress_pct,
            "health": self.health,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Mission — the full mission record
# ---------------------------------------------------------------------------

@dataclass
class Mission:
    """A named operation with objectives, resources, constraints, and schedule.

    This is the central data structure for mission planning. A Mission
    progresses through states: PLANNING -> BRIEFED -> ACTIVE -> COMPLETED.

    Attributes
    ----------
    mission_id:
        Unique identifier.
    name:
        Human-readable mission name.
    mission_type:
        Type of mission (SURVEILLANCE, TRACKING, etc.).
    priority:
        Mission priority level.
    state:
        Current lifecycle state.
    description:
        Detailed description of the mission.
    objectives:
        List of MissionObjective instances.
    resources:
        List of ResourceAllocation instances.
    schedule:
        MissionSchedule defining time windows and shifts.
    constraints:
        List of MissionConstraint instances.
    created_at:
        When the mission was created.
    activated_at:
        When the mission was activated (0 if not yet).
    completed_at:
        When the mission was completed/aborted (0 if not yet).
    created_by:
        Who or what created this mission.
    incident_id:
        Optional linked incident ID (for investigation missions).
    area_of_interest:
        Optional GeoJSON-like geometry defining the mission area.
    tags:
        Free-form tags for categorization.
    metadata:
        Arbitrary key-value data.
    """
    mission_id: str = ""
    name: str = ""
    mission_type: MissionType = MissionType.SURVEILLANCE
    priority: MissionPriority = MissionPriority.MEDIUM
    state: MissionState = MissionState.PLANNING
    description: str = ""
    objectives: list[MissionObjective] = field(default_factory=list)
    resources: list[ResourceAllocation] = field(default_factory=list)
    schedule: MissionSchedule = field(default_factory=MissionSchedule)
    constraints: list[MissionConstraint] = field(default_factory=list)
    created_at: float = 0.0
    activated_at: float = 0.0
    completed_at: float = 0.0
    created_by: str = "system"
    incident_id: str = ""
    area_of_interest: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mission_id:
            self.mission_id = f"msn_{uuid.uuid4().hex[:10]}"
        if self.created_at == 0.0:
            self.created_at = time.time()

    @property
    def is_terminal(self) -> bool:
        """True if the mission is in a terminal state."""
        return self.state in (MissionState.COMPLETED, MissionState.ABORTED)

    @property
    def active_resources(self) -> list[ResourceAllocation]:
        """Return only currently allocated (not released) resources."""
        return [r for r in self.resources if r.is_active]

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since activation, or 0 if not yet activated."""
        if self.activated_at == 0.0:
            return 0.0
        end = self.completed_at if self.completed_at > 0 else time.time()
        return end - self.activated_at

    def can_transition_to(self, new_state: MissionState) -> bool:
        """Check if the mission can transition to the given state."""
        return new_state in _VALID_TRANSITIONS.get(self.state, set())

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "name": self.name,
            "mission_type": self.mission_type.value,
            "priority": self.priority.value,
            "state": self.state.value,
            "description": self.description,
            "objectives": [o.to_dict() for o in self.objectives],
            "resources": [r.to_dict() for r in self.resources],
            "schedule": self.schedule.to_dict(),
            "constraints": [c.to_dict() for c in self.constraints],
            "created_at": self.created_at,
            "activated_at": self.activated_at,
            "completed_at": self.completed_at,
            "created_by": self.created_by,
            "incident_id": self.incident_id,
            "area_of_interest": dict(self.area_of_interest),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "is_terminal": self.is_terminal,
            "elapsed_seconds": self.elapsed_seconds,
            "active_resource_count": len(self.active_resources),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Mission:
        return cls(
            mission_id=data.get("mission_id", ""),
            name=data.get("name", ""),
            mission_type=MissionType(data.get("mission_type", "surveillance")),
            priority=MissionPriority(data.get("priority", "medium")),
            state=MissionState(data.get("state", "planning")),
            description=data.get("description", ""),
            objectives=[
                MissionObjective.from_dict(o)
                for o in data.get("objectives", [])
            ],
            resources=[
                ResourceAllocation.from_dict(r)
                for r in data.get("resources", [])
            ],
            schedule=MissionSchedule.from_dict(data.get("schedule", {})),
            constraints=[
                MissionConstraint.from_dict(c)
                for c in data.get("constraints", [])
            ],
            created_at=data.get("created_at", 0.0),
            activated_at=data.get("activated_at", 0.0),
            completed_at=data.get("completed_at", 0.0),
            created_by=data.get("created_by", "system"),
            incident_id=data.get("incident_id", ""),
            area_of_interest=data.get("area_of_interest", {}),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# MissionPlanner — CRUD + lifecycle operations
# ---------------------------------------------------------------------------

class MissionPlanner:
    """Manages mission planning, execution, and tracking.

    Thread-safe. All public methods acquire the internal lock.

    Parameters
    ----------
    event_bus:
        Optional EventBus for publishing mission lifecycle events.
    alert_engine:
        Optional AlertEngine for mission-scoped alert tracking.
    max_missions:
        Maximum number of missions to retain in memory.
    """

    def __init__(
        self,
        event_bus=None,
        alert_engine=None,
        *,
        max_missions: int = 1000,
    ) -> None:
        self._event_bus = event_bus
        self._alert_engine = alert_engine
        self._lock = threading.Lock()
        self._max_missions = max_missions

        # Storage: mission_id -> Mission
        self._missions: dict[str, Mission] = {}

        # Per-mission objective status tracking
        self._objective_statuses: dict[str, list[ObjectiveStatus]] = {}

        # Counters
        self._total_created = 0
        self._total_completed = 0
        self._total_aborted = 0

    # ------------------------------------------------------------------
    # Mission CRUD
    # ------------------------------------------------------------------

    def create_mission(
        self,
        name: str,
        mission_type: MissionType | str = MissionType.SURVEILLANCE,
        *,
        priority: MissionPriority | str = MissionPriority.MEDIUM,
        description: str = "",
        objectives: list[MissionObjective] | None = None,
        schedule: MissionSchedule | None = None,
        constraints: list[MissionConstraint] | None = None,
        created_by: str = "system",
        incident_id: str = "",
        area_of_interest: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Mission:
        """Create a new mission in PLANNING state.

        Parameters
        ----------
        name:
            Human-readable mission name.
        mission_type:
            Type of mission.
        priority:
            Mission priority.
        description:
            Detailed description.
        objectives:
            Initial list of objectives.
        schedule:
            Scheduling parameters.
        constraints:
            Operational constraints.
        created_by:
            Creator identity.
        incident_id:
            Optional linked incident ID.
        area_of_interest:
            Optional GeoJSON geometry for mission area.
        tags:
            Free-form tags.
        metadata:
            Arbitrary key-value data.

        Returns the created Mission.
        """
        if isinstance(mission_type, str):
            mission_type = MissionType(mission_type)
        if isinstance(priority, str):
            priority = MissionPriority(priority)

        mission = Mission(
            name=name,
            mission_type=mission_type,
            priority=priority,
            description=description,
            objectives=objectives or [],
            schedule=schedule or MissionSchedule(),
            constraints=constraints or [],
            created_by=created_by,
            incident_id=incident_id,
            area_of_interest=area_of_interest or {},
            tags=tags or [],
            metadata=metadata or {},
        )

        with self._lock:
            self._missions[mission.mission_id] = mission
            self._objective_statuses[mission.mission_id] = [
                ObjectiveStatus(objective_index=i)
                for i in range(len(mission.objectives))
            ]
            self._total_created += 1

            # Enforce max missions (remove oldest completed/aborted first)
            self._enforce_limit()

        logger.info(
            "Mission created: %s (%s) type=%s priority=%s",
            mission.name, mission.mission_id,
            mission.mission_type.value, mission.priority.value,
        )

        self._publish_event("mission.created", mission)
        return mission

    def get_mission(self, mission_id: str) -> Mission | None:
        """Get a mission by ID. Returns None if not found."""
        with self._lock:
            return self._missions.get(mission_id)

    def get_missions(
        self,
        *,
        state: MissionState | str | None = None,
        mission_type: MissionType | str | None = None,
        priority: MissionPriority | str | None = None,
        tag: str = "",
    ) -> list[Mission]:
        """Return missions with optional filtering.

        Parameters
        ----------
        state:
            Filter by state.
        mission_type:
            Filter by type.
        priority:
            Filter by priority.
        tag:
            Filter by tag (must contain this tag).

        Returns missions sorted by created_at (newest first).
        """
        if isinstance(state, str):
            state = MissionState(state)
        if isinstance(mission_type, str):
            mission_type = MissionType(mission_type)
        if isinstance(priority, str):
            priority = MissionPriority(priority)

        with self._lock:
            missions = list(self._missions.values())

        if state is not None:
            missions = [m for m in missions if m.state == state]
        if mission_type is not None:
            missions = [m for m in missions if m.mission_type == mission_type]
        if priority is not None:
            missions = [m for m in missions if m.priority == priority]
        if tag:
            missions = [m for m in missions if tag in m.tags]

        missions.sort(key=lambda m: m.created_at, reverse=True)
        return missions

    def update_mission(
        self,
        mission_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        priority: MissionPriority | str | None = None,
        area_of_interest: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Mission | None:
        """Update mutable fields of a mission. Returns updated mission or None.

        Only missions in PLANNING or BRIEFED state can be updated.
        """
        if isinstance(priority, str):
            priority = MissionPriority(priority)

        with self._lock:
            mission = self._missions.get(mission_id)
            if mission is None:
                return None
            if mission.state not in (MissionState.PLANNING, MissionState.BRIEFED):
                logger.warning(
                    "Cannot update mission %s in state %s",
                    mission_id, mission.state.value,
                )
                return None

            if name is not None:
                mission.name = name
            if description is not None:
                mission.description = description
            if priority is not None:
                mission.priority = priority
            if area_of_interest is not None:
                mission.area_of_interest = area_of_interest
            if tags is not None:
                mission.tags = tags
            if metadata is not None:
                mission.metadata = metadata

        logger.info("Mission updated: %s (%s)", mission.name, mission_id)
        self._publish_event("mission.updated", mission)
        return mission

    def delete_mission(self, mission_id: str) -> bool:
        """Delete a mission. Only terminal or planning missions can be deleted.

        Returns True if deleted.
        """
        with self._lock:
            mission = self._missions.get(mission_id)
            if mission is None:
                return False
            if mission.state in (MissionState.ACTIVE, MissionState.PAUSED):
                logger.warning(
                    "Cannot delete active/paused mission %s", mission_id,
                )
                return False
            del self._missions[mission_id]
            self._objective_statuses.pop(mission_id, None)

        logger.info("Mission deleted: %s", mission_id)
        return True

    # ------------------------------------------------------------------
    # Objective management
    # ------------------------------------------------------------------

    def add_objective(
        self,
        mission_id: str,
        objective: MissionObjective,
    ) -> bool:
        """Add an objective to a mission. Returns True if added.

        Only missions in PLANNING or BRIEFED state accept new objectives.
        """
        with self._lock:
            mission = self._missions.get(mission_id)
            if mission is None:
                return False
            if mission.state not in (MissionState.PLANNING, MissionState.BRIEFED):
                return False
            mission.objectives.append(objective)
            self._objective_statuses.setdefault(mission_id, []).append(
                ObjectiveStatus(objective_index=len(mission.objectives) - 1)
            )
        return True

    def update_objective_status(
        self,
        mission_id: str,
        objective_index: int,
        *,
        status: str | None = None,
        progress_pct: float | None = None,
        notes: str | None = None,
        detections_delta: int = 0,
        alerts_delta: int = 0,
    ) -> ObjectiveStatus | None:
        """Update the status of a specific objective.

        Returns the updated ObjectiveStatus or None if not found.
        """
        with self._lock:
            statuses = self._objective_statuses.get(mission_id, [])
            if objective_index < 0 or objective_index >= len(statuses):
                return None
            os = statuses[objective_index]
            if status is not None:
                os.status = status
                if status == "active" and os.started_at == 0.0:
                    os.started_at = time.time()
                elif status in ("completed", "failed") and os.completed_at == 0.0:
                    os.completed_at = time.time()
            if progress_pct is not None:
                os.progress_pct = max(0.0, min(100.0, progress_pct))
            if notes is not None:
                os.notes = notes
            os.detections += detections_delta
            os.alerts_fired += alerts_delta
            return os

    # ------------------------------------------------------------------
    # Resource allocation
    # ------------------------------------------------------------------

    def allocate_resource(
        self,
        mission_id: str,
        resource_id: str,
        resource_type: str = "",
        *,
        objective_index: int = -1,
        role: str = "primary",
        shift_start: float = 0.0,
        shift_end: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ResourceAllocation | None:
        """Allocate a resource to a mission.

        Parameters
        ----------
        mission_id:
            Target mission.
        resource_id:
            ID of the sensor/device/unit.
        resource_type:
            Type of resource.
        objective_index:
            Which objective this supports (-1 = mission-wide).
        role:
            Role: "primary", "backup", "relay", etc.
        shift_start:
            Scheduled start time (0 = immediate).
        shift_end:
            Scheduled end time (0 = until mission ends).
        metadata:
            Extra data.

        Returns the allocation or None if mission not found.
        """
        with self._lock:
            mission = self._missions.get(mission_id)
            if mission is None:
                return None
            if mission.is_terminal:
                logger.warning(
                    "Cannot allocate to terminal mission %s", mission_id,
                )
                return None

            allocation = ResourceAllocation(
                resource_id=resource_id,
                resource_type=resource_type,
                objective_index=objective_index,
                role=role,
                shift_start=shift_start,
                shift_end=shift_end,
                metadata=metadata or {},
            )
            mission.resources.append(allocation)

        logger.info(
            "Resource %s (%s) allocated to mission %s",
            resource_id, resource_type, mission_id,
        )
        self._publish_event("mission.resource_allocated", mission, extra={
            "resource_id": resource_id,
            "allocation_id": allocation.allocation_id,
        })
        return allocation

    def release_resource(
        self,
        mission_id: str,
        allocation_id: str,
    ) -> bool:
        """Release a resource allocation. Returns True if found and released."""
        with self._lock:
            mission = self._missions.get(mission_id)
            if mission is None:
                return False
            for alloc in mission.resources:
                if alloc.allocation_id == allocation_id and alloc.is_active:
                    alloc.released_at = time.time()
                    logger.info(
                        "Resource %s released from mission %s",
                        alloc.resource_id, mission_id,
                    )
                    return True
        return False

    def get_resource_allocations(
        self,
        mission_id: str,
        *,
        active_only: bool = False,
    ) -> list[ResourceAllocation]:
        """Get resource allocations for a mission."""
        with self._lock:
            mission = self._missions.get(mission_id)
            if mission is None:
                return []
            allocs = list(mission.resources)
        if active_only:
            allocs = [a for a in allocs if a.is_active]
        return allocs

    # ------------------------------------------------------------------
    # Constraint management
    # ------------------------------------------------------------------

    def add_constraint(
        self,
        mission_id: str,
        constraint: MissionConstraint,
    ) -> bool:
        """Add a constraint to a mission. Returns True if added."""
        with self._lock:
            mission = self._missions.get(mission_id)
            if mission is None:
                return False
            mission.constraints.append(constraint)
        return True

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def _transition(
        self,
        mission_id: str,
        new_state: MissionState,
        *,
        reason: str = "",
    ) -> Mission | None:
        """Attempt a state transition. Returns the mission or None on failure."""
        with self._lock:
            mission = self._missions.get(mission_id)
            if mission is None:
                logger.warning("Mission not found: %s", mission_id)
                return None
            if not mission.can_transition_to(new_state):
                logger.warning(
                    "Invalid transition %s -> %s for mission %s",
                    mission.state.value, new_state.value, mission_id,
                )
                return None

            old_state = mission.state
            mission.state = new_state

            if new_state == MissionState.ACTIVE and mission.activated_at == 0.0:
                mission.activated_at = time.time()
            elif new_state in (MissionState.COMPLETED, MissionState.ABORTED):
                mission.completed_at = time.time()
                if new_state == MissionState.COMPLETED:
                    self._total_completed += 1
                else:
                    self._total_aborted += 1

        logger.info(
            "Mission %s (%s): %s -> %s%s",
            mission.name, mission_id,
            old_state.value, new_state.value,
            f" ({reason})" if reason else "",
        )
        self._publish_event(
            f"mission.state.{new_state.value}",
            mission,
            extra={"old_state": old_state.value, "reason": reason},
        )
        return mission

    def brief(self, mission_id: str) -> Mission | None:
        """Transition mission to BRIEFED state (ready for activation)."""
        return self._transition(mission_id, MissionState.BRIEFED)

    def activate(self, mission_id: str) -> Mission | None:
        """Activate a mission (BRIEFED -> ACTIVE or PAUSED -> ACTIVE).

        Sets activated_at timestamp on first activation.
        Marks all pending objectives as active.
        """
        mission = self._transition(mission_id, MissionState.ACTIVE)
        if mission is not None:
            # Activate pending objectives
            with self._lock:
                statuses = self._objective_statuses.get(mission_id, [])
                for os in statuses:
                    if os.status == "pending":
                        os.status = "active"
                        os.started_at = time.time()
        return mission

    def pause(self, mission_id: str, reason: str = "") -> Mission | None:
        """Pause an active mission."""
        return self._transition(
            mission_id, MissionState.PAUSED, reason=reason,
        )

    def complete(
        self,
        mission_id: str,
        summary: str = "",
    ) -> Mission | None:
        """Complete a mission successfully.

        Parameters
        ----------
        summary:
            Optional completion summary stored in metadata.
        """
        mission = self._transition(mission_id, MissionState.COMPLETED)
        if mission is not None and summary:
            with self._lock:
                mission.metadata["completion_summary"] = summary
        return mission

    def abort(
        self,
        mission_id: str,
        reason: str = "",
    ) -> Mission | None:
        """Abort a mission.

        Parameters
        ----------
        reason:
            Why the mission was aborted.
        """
        mission = self._transition(
            mission_id, MissionState.ABORTED, reason=reason,
        )
        if mission is not None and reason:
            with self._lock:
                mission.metadata["abort_reason"] = reason
        return mission

    def replan(self, mission_id: str) -> Mission | None:
        """Send a BRIEFED mission back to PLANNING for changes."""
        return self._transition(mission_id, MissionState.PLANNING)

    # ------------------------------------------------------------------
    # Brief generation
    # ------------------------------------------------------------------

    def generate_brief(self, mission_id: str) -> MissionBrief | None:
        """Generate a human-readable brief for a mission.

        Returns a MissionBrief or None if the mission is not found.
        """
        with self._lock:
            mission = self._missions.get(mission_id)
            if mission is None:
                return None
            # Take a snapshot under lock
            mission_dict = mission.to_dict()
            objectives = list(mission.objectives)
            resources = list(mission.resources)
            schedule = mission.schedule
            constraints = list(mission.constraints)

        # Build summary
        obj_count = len(objectives)
        res_count = len(resources)
        summary = (
            f"Mission '{mission_dict['name']}' is a {mission_dict['mission_type']} operation "
            f"at {mission_dict['priority']} priority. "
            f"{obj_count} objective{'s' if obj_count != 1 else ''} defined, "
            f"{res_count} resource{'s' if res_count != 1 else ''} allocated."
        )
        if mission_dict.get("description"):
            summary += f" {mission_dict['description']}"

        # Objectives text
        obj_lines = []
        for i, obj in enumerate(objectives):
            line = f"  {i + 1}. [{obj.priority.value.upper()}] {obj.description}"
            if obj.area_id:
                line += f" (area: {obj.area_id})"
            if obj.target_ids:
                line += f" (targets: {', '.join(obj.target_ids)})"
            if obj.success_criteria:
                line += f"\n     Success: {obj.success_criteria}"
            obj_lines.append(line)
        objectives_text = "\n".join(obj_lines) if obj_lines else "  No objectives defined."

        # Resources text
        res_lines = []
        for alloc in resources:
            status = "ACTIVE" if alloc.is_active else "RELEASED"
            line = (
                f"  - {alloc.resource_id} ({alloc.resource_type}) "
                f"role={alloc.role} obj={alloc.objective_index} [{status}]"
            )
            res_lines.append(line)
        resources_text = "\n".join(res_lines) if res_lines else "  No resources allocated."

        # Schedule text
        sched_lines = []
        if schedule.has_time_window:
            if schedule.planned_start > 0:
                sched_lines.append(
                    f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(schedule.planned_start))}"
                )
            if schedule.planned_end > 0:
                sched_lines.append(
                    f"  End: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(schedule.planned_end))}"
                )
        if schedule.max_duration_hours > 0:
            sched_lines.append(f"  Max duration: {schedule.max_duration_hours}h")
        if schedule.shift_duration_hours > 0:
            sched_lines.append(f"  Shift rotation: every {schedule.shift_duration_hours}h")
        if schedule.recurring:
            sched_lines.append(
                f"  Recurring: every {schedule.recurrence_interval_hours}h"
            )
        schedule_text = "\n".join(sched_lines) if sched_lines else "  No schedule constraints."

        # Constraints text
        con_lines = []
        for con in constraints:
            con_lines.append(
                f"  - [{con.severity.upper()}] {con.constraint_type}: {con.description}"
            )
        constraints_text = "\n".join(con_lines) if con_lines else "  No constraints."

        return MissionBrief(
            mission_id=mission_id,
            mission_name=mission_dict["name"],
            mission_type=mission_dict["mission_type"],
            priority=mission_dict["priority"],
            summary=summary,
            objectives_text=objectives_text,
            resources_text=resources_text,
            schedule_text=schedule_text,
            constraints_text=constraints_text,
            generated_at=time.time(),
        )

    # ------------------------------------------------------------------
    # Status query
    # ------------------------------------------------------------------

    def get_status(self, mission_id: str) -> MissionStatus | None:
        """Generate a real-time status snapshot for a mission.

        Returns MissionStatus or None if mission not found.
        """
        with self._lock:
            mission = self._missions.get(mission_id)
            if mission is None:
                return None

            obj_statuses = list(self._objective_statuses.get(mission_id, []))
            active_res = len(mission.active_resources)
            total_res = len(mission.resources)

        total_detections = sum(os.detections for os in obj_statuses)
        total_alerts = sum(os.alerts_fired for os in obj_statuses)

        # Calculate overall progress
        if obj_statuses:
            overall_progress = sum(os.progress_pct for os in obj_statuses) / len(obj_statuses)
        else:
            overall_progress = 0.0

        # Determine health
        health = "green"
        if any(os.status == "failed" for os in obj_statuses):
            health = "red"
        elif active_res == 0 and mission.state == MissionState.ACTIVE:
            health = "red"
        elif any(os.progress_pct < 10.0 and os.status == "active" for os in obj_statuses):
            health = "yellow"

        return MissionStatus(
            mission_id=mission_id,
            state=mission.state.value,
            elapsed_seconds=mission.elapsed_seconds,
            objective_statuses=obj_statuses,
            active_resources=active_res,
            total_resources=total_res,
            total_detections=total_detections,
            total_alerts=total_alerts,
            overall_progress_pct=overall_progress,
            health=health,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return planner-wide statistics."""
        with self._lock:
            by_state: dict[str, int] = {}
            by_type: dict[str, int] = {}
            by_priority: dict[str, int] = {}
            for m in self._missions.values():
                by_state[m.state.value] = by_state.get(m.state.value, 0) + 1
                by_type[m.mission_type.value] = by_type.get(m.mission_type.value, 0) + 1
                by_priority[m.priority.value] = by_priority.get(m.priority.value, 0) + 1

            return {
                "total_missions": len(self._missions),
                "total_created": self._total_created,
                "total_completed": self._total_completed,
                "total_aborted": self._total_aborted,
                "by_state": by_state,
                "by_type": by_type,
                "by_priority": by_priority,
                "max_missions": self._max_missions,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _publish_event(
        self,
        topic: str,
        mission: Mission,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Publish a mission lifecycle event to the EventBus."""
        if self._event_bus is None:
            return
        data = {
            "mission_id": mission.mission_id,
            "mission_name": mission.name,
            "mission_type": mission.mission_type.value,
            "state": mission.state.value,
            "priority": mission.priority.value,
        }
        if extra:
            data.update(extra)
        try:
            self._event_bus.publish(topic, data=data, source="mission")
        except Exception:
            logger.debug("Failed to publish event %s", topic, exc_info=True)

    def _enforce_limit(self) -> None:
        """Remove oldest terminal missions if over the limit. Must hold lock."""
        if len(self._missions) <= self._max_missions:
            return

        # Collect terminal missions sorted by completed_at (oldest first)
        terminal = [
            m for m in self._missions.values() if m.is_terminal
        ]
        terminal.sort(key=lambda m: m.completed_at)

        while len(self._missions) > self._max_missions and terminal:
            old = terminal.pop(0)
            del self._missions[old.mission_id]
            self._objective_statuses.pop(old.mission_id, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "MissionType",
    "MissionState",
    "MissionPriority",
    "MissionObjective",
    "ObjectiveStatus",
    "ResourceAllocation",
    "MissionSchedule",
    "MissionConstraint",
    "MissionBrief",
    "MissionStatus",
    "Mission",
    "MissionPlanner",
]
