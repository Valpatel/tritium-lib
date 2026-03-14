# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tactical scenario models for repeatable test scenarios and training exercises.

A TacticalScenario defines a structured, repeatable exercise with actors,
events, a timeline, and objectives. Used for training, regression testing,
and after-action review of the Tritium system's detection and response
capabilities.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class ScenarioStatus(str, Enum):
    """Lifecycle state of a tactical scenario."""
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class ActorType(str, Enum):
    """Type of actor in a scenario."""
    PERSON = "person"
    VEHICLE = "vehicle"
    DEVICE = "device"
    DRONE = "drone"
    SENSOR_NODE = "sensor_node"
    MESH_RADIO = "mesh_radio"
    CAMERA = "camera"
    ROBOT = "robot"
    UNKNOWN = "unknown"


class ActorAlliance(str, Enum):
    """Alliance classification for scenario actors."""
    FRIENDLY = "friendly"
    HOSTILE = "hostile"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class ScenarioEventType(str, Enum):
    """Types of events that can occur in a scenario timeline."""
    SPAWN = "spawn"
    MOVE = "move"
    DETECT = "detect"
    CLASSIFY = "classify"
    CORRELATE = "correlate"
    GEOFENCE_ENTER = "geofence_enter"
    GEOFENCE_EXIT = "geofence_exit"
    ALERT = "alert"
    DISPATCH = "dispatch"
    ENGAGE = "engage"
    DEPART = "depart"
    ENRICH = "enrich"
    CUSTOM = "custom"


@dataclass
class ScenarioActor:
    """An actor participating in a scenario."""
    actor_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    actor_type: ActorType = ActorType.UNKNOWN
    alliance: ActorAlliance = ActorAlliance.UNKNOWN
    # Starting position (lat/lng or local x/y)
    start_lat: Optional[float] = None
    start_lng: Optional[float] = None
    # BLE/WiFi simulation properties
    mac_address: Optional[str] = None
    device_class: Optional[str] = None
    manufacturer: Optional[str] = None
    # Movement path (list of {lat, lng, time_offset_s})
    waypoints: list[dict[str, float]] = field(default_factory=list)
    # Extra properties for custom actor types
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioEvent:
    """A single event in the scenario timeline."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type: ScenarioEventType = ScenarioEventType.CUSTOM
    time_offset_s: float = 0.0  # seconds from scenario start
    actor_id: Optional[str] = None  # which actor this event involves
    target_actor_id: Optional[str] = None  # second actor (for correlations)
    description: str = ""
    # Position (optional, for movement/spawn events)
    lat: Optional[float] = None
    lng: Optional[float] = None
    # Expected outcome (for validation)
    expected_result: Optional[str] = None
    # Extra data
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioObjective:
    """An objective to achieve during the scenario."""
    objective_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    completed: bool = False
    priority: int = 1  # 1=highest
    # Validation criteria
    success_criteria: str = ""
    # Time limit in seconds (0 = no limit)
    time_limit_s: float = 0.0
    # Score value when completed
    score_value: int = 100


@dataclass
class TacticalScenario:
    """A complete tactical scenario definition.

    Defines a repeatable test scenario with actors, events, a timeline,
    and objectives. Can be serialized to JSON for sharing and replay.
    """
    scenario_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    status: ScenarioStatus = ScenarioStatus.DRAFT
    # Actors in this scenario
    actors: list[ScenarioActor] = field(default_factory=list)
    # Timeline of events
    events: list[ScenarioEvent] = field(default_factory=list)
    # Objectives to achieve
    objectives: list[ScenarioObjective] = field(default_factory=list)
    # Duration in seconds (0 = derived from events)
    duration_s: float = 0.0
    # Map center for this scenario
    center_lat: float = 0.0
    center_lng: float = 0.0
    zoom_level: float = 16.0
    # Tags for categorization
    tags: list[str] = field(default_factory=list)
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    version: int = 1

    def computed_duration(self) -> float:
        """Return explicit duration or max event time offset."""
        if self.duration_s > 0:
            return self.duration_s
        if not self.events:
            return 0.0
        return max(e.time_offset_s for e in self.events)

    def actor_by_id(self, actor_id: str) -> Optional[ScenarioActor]:
        """Find an actor by ID."""
        for a in self.actors:
            if a.actor_id == actor_id:
                return a
        return None

    def events_for_actor(self, actor_id: str) -> list[ScenarioEvent]:
        """Get all events involving a specific actor."""
        return [e for e in self.events if e.actor_id == actor_id or e.target_actor_id == actor_id]

    def sorted_events(self) -> list[ScenarioEvent]:
        """Return events sorted by time offset."""
        return sorted(self.events, key=lambda e: e.time_offset_s)

    def completion_pct(self) -> float:
        """Percentage of objectives completed."""
        if not self.objectives:
            return 100.0
        done = sum(1 for o in self.objectives if o.completed)
        return round(100.0 * done / len(self.objectives), 1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        from dataclasses import asdict
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        return d
