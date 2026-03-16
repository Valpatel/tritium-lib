# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Mission management models for structured operations.

Missions represent coordinated multi-asset operations with objectives,
assigned assets, geofence zones, and lifecycle tracking. Used by the
Command Center to manage patrols, surveillance sweeps, investigations,
and tactical responses beyond simple point-to-point dispatches.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class MissionType(str, Enum):
    """Classification of mission purpose."""
    PATROL = "patrol"
    SURVEILLANCE = "surveillance"
    INVESTIGATION = "investigation"
    RESPONSE = "response"
    ESCORT = "escort"
    SEARCH = "search"
    PERIMETER = "perimeter"
    CUSTOM = "custom"


class MissionStatus(str, Enum):
    """Lifecycle state of a mission."""
    DRAFT = "draft"
    PLANNED = "planned"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class MissionObjective:
    """A single objective within a mission."""
    objective_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    completed: bool = False
    priority: int = 1  # 1=highest
    completed_at: Optional[datetime] = None


@dataclass
class GeofenceZone:
    """Geographic boundary for a mission area of operations."""
    zone_id: str = ""
    name: str = ""
    # Polygon vertices as list of (lat, lng) tuples
    vertices: list[tuple[float, float]] = field(default_factory=list)
    # Simple circle alternative: center + radius_m
    center_lat: Optional[float] = None
    center_lng: Optional[float] = None
    radius_m: Optional[float] = None

    @property
    def is_circle(self) -> bool:
        return (
            self.center_lat is not None
            and self.center_lng is not None
            and self.radius_m is not None
        )


@dataclass
class Mission:
    """A structured mission with assets, objectives, and geographic scope.

    Missions coordinate multiple assets toward defined objectives within
    a geographic area. They have a lifecycle from draft through completion
    or abort, and track which objectives have been met.
    """
    mission_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    type: MissionType = MissionType.CUSTOM
    status: MissionStatus = MissionStatus.DRAFT
    description: str = ""

    # Assets assigned to this mission (target_ids or asset_ids)
    assigned_assets: list[str] = field(default_factory=list)

    # Mission objectives
    objectives: list[MissionObjective] = field(default_factory=list)

    # Area of operations
    geofence_zone: Optional[GeofenceZone] = None

    # Timestamps
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started: Optional[datetime] = None
    completed: Optional[datetime] = None

    # Mission commander / creator
    created_by: str = ""

    # Priority (1 = highest)
    priority: int = 3

    # Tags for filtering/categorization
    tags: list[str] = field(default_factory=list)

    def start(self) -> None:
        """Transition mission to active status."""
        if self.status in (MissionStatus.DRAFT, MissionStatus.PLANNED, MissionStatus.PAUSED):
            self.status = MissionStatus.ACTIVE
            if self.started is None:
                self.started = datetime.now(timezone.utc)

    def pause(self) -> None:
        """Pause an active mission."""
        if self.status == MissionStatus.ACTIVE:
            self.status = MissionStatus.PAUSED

    def complete(self) -> None:
        """Mark mission as completed."""
        if self.status in (MissionStatus.ACTIVE, MissionStatus.PAUSED):
            self.status = MissionStatus.COMPLETED
            self.completed = datetime.now(timezone.utc)

    def abort(self, reason: str = "") -> None:
        """Abort the mission."""
        if self.status not in (MissionStatus.COMPLETED, MissionStatus.FAILED):
            self.status = MissionStatus.ABORTED
            self.completed = datetime.now(timezone.utc)
            if reason and not self.description.endswith(reason):
                self.description += f" [ABORTED: {reason}]"

    def complete_objective(self, objective_id: str) -> bool:
        """Mark an objective as completed. Returns True if found."""
        for obj in self.objectives:
            if obj.objective_id == objective_id:
                obj.completed = True
                obj.completed_at = datetime.now(timezone.utc)
                return True
        return False

    @property
    def progress(self) -> float:
        """Fraction of objectives completed (0.0 to 1.0)."""
        if not self.objectives:
            return 0.0
        done = sum(1 for o in self.objectives if o.completed)
        return done / len(self.objectives)

    @property
    def is_terminal(self) -> bool:
        """True if mission is in a terminal state."""
        return self.status in (
            MissionStatus.COMPLETED,
            MissionStatus.ABORTED,
            MissionStatus.FAILED,
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON transport."""
        return {
            "mission_id": self.mission_id,
            "title": self.title,
            "type": self.type.value,
            "status": self.status.value,
            "description": self.description,
            "assigned_assets": self.assigned_assets,
            "objectives": [
                {
                    "objective_id": o.objective_id,
                    "description": o.description,
                    "completed": o.completed,
                    "priority": o.priority,
                    "completed_at": o.completed_at.isoformat() if o.completed_at else None,
                }
                for o in self.objectives
            ],
            "geofence_zone": {
                "zone_id": self.geofence_zone.zone_id,
                "name": self.geofence_zone.name,
                "vertices": self.geofence_zone.vertices,
                "center_lat": self.geofence_zone.center_lat,
                "center_lng": self.geofence_zone.center_lng,
                "radius_m": self.geofence_zone.radius_m,
            } if self.geofence_zone else None,
            "created": self.created.isoformat(),
            "started": self.started.isoformat() if self.started else None,
            "completed": self.completed.isoformat() if self.completed else None,
            "created_by": self.created_by,
            "priority": self.priority,
            "progress": self.progress,
            "tags": self.tags,
        }
