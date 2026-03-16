# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Operational period models for structuring operations into defined time blocks.

An OperationalPeriod represents a bounded time window during which a specific
commander is responsible, with defined objectives, weather conditions, and
personnel counts. Used by the Command Center to structure multi-shift
operations, briefings, and after-action reviews.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class OperationalPhase(str, Enum):
    """Phase of the operational period lifecycle."""
    PLANNED = "planned"
    BRIEFING = "briefing"
    ACTIVE = "active"
    TRANSITION = "transition"
    DEBRIEFING = "debriefing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class WeatherInfo:
    """Weather conditions during an operational period."""
    condition: str = ""  # clear, rain, snow, fog, overcast, etc.
    temperature_c: Optional[float] = None
    wind_speed_kph: Optional[float] = None
    wind_direction: Optional[str] = None  # N, NE, E, etc.
    visibility_m: Optional[float] = None
    humidity_pct: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "temperature_c": self.temperature_c,
            "wind_speed_kph": self.wind_speed_kph,
            "wind_direction": self.wind_direction,
            "visibility_m": self.visibility_m,
            "humidity_pct": self.humidity_pct,
        }


@dataclass
class OperationalObjective:
    """A single objective within an operational period."""
    objective_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    priority: int = 1  # 1=highest
    completed: bool = False
    completed_at: Optional[datetime] = None
    assigned_to: str = ""  # who is responsible
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "description": self.description,
            "priority": self.priority,
            "completed": self.completed,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "assigned_to": self.assigned_to,
            "notes": self.notes,
        }


@dataclass
class OperationalPeriod:
    """A defined time period for structuring operations.

    Operational periods segment continuous operations into manageable blocks
    with clear commanders, objectives, and conditions. They enable shift
    handoffs, briefings, and after-action reviews.

    Attributes:
        period_id: Unique identifier for this operational period.
        start: Start time of the period.
        end: End time of the period (None if open-ended).
        commander: Name/ID of the period commander.
        objectives: List of objectives for this period.
        weather: Weather conditions during the period.
        personnel_count: Number of personnel active during the period.
        phase: Current lifecycle phase.
        site_id: Site this period applies to.
        notes: Free-form operational notes.
        tags: Tags for filtering/categorization.
    """
    period_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end: Optional[datetime] = None
    commander: str = ""
    objectives: list[OperationalObjective] = field(default_factory=list)
    weather: Optional[WeatherInfo] = None
    personnel_count: int = 0
    phase: OperationalPhase = OperationalPhase.PLANNED
    site_id: str = ""
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    # Tracking
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""

    def activate(self) -> None:
        """Transition to active phase."""
        if self.phase in (OperationalPhase.PLANNED, OperationalPhase.BRIEFING):
            self.phase = OperationalPhase.ACTIVE

    def complete(self) -> None:
        """Mark period as completed."""
        if self.phase in (OperationalPhase.ACTIVE, OperationalPhase.TRANSITION):
            self.phase = OperationalPhase.COMPLETED
            if self.end is None:
                self.end = datetime.now(timezone.utc)

    def cancel(self) -> None:
        """Cancel the operational period."""
        if not self.is_terminal:
            self.phase = OperationalPhase.CANCELLED
            if self.end is None:
                self.end = datetime.now(timezone.utc)

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
        """True if period is in a terminal state."""
        return self.phase in (
            OperationalPhase.COMPLETED,
            OperationalPhase.CANCELLED,
        )

    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration in seconds, or None if no end time."""
        if self.end is None:
            return None
        return (self.end - self.start).total_seconds()

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON transport."""
        return {
            "period_id": self.period_id,
            "start": self.start.isoformat(),
            "end": self.end.isoformat() if self.end else None,
            "commander": self.commander,
            "objectives": [o.to_dict() for o in self.objectives],
            "weather": self.weather.to_dict() if self.weather else None,
            "personnel_count": self.personnel_count,
            "phase": self.phase.value,
            "site_id": self.site_id,
            "notes": self.notes,
            "tags": self.tags,
            "created": self.created.isoformat(),
            "created_by": self.created_by,
            "progress": self.progress,
            "duration_seconds": self.duration_seconds,
        }
