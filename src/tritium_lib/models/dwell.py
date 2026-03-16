# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Dwell event models — tracking how long targets stay in specific locations.

A DwellEvent is generated when a target remains stationary (or within a small
radius) for longer than a configurable threshold (default 5 minutes). Used for
loitering detection, pattern analysis, and behavioral intelligence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DwellState(str, Enum):
    """Current state of a dwell event."""
    ACTIVE = "active"       # Target is still dwelling
    ENDED = "ended"         # Target has moved away
    EXPIRED = "expired"     # Dwell timed out (max duration reached)


class DwellSeverity(str, Enum):
    """Severity classification based on dwell duration."""
    NORMAL = "normal"       # 5-15 minutes
    EXTENDED = "extended"   # 15-60 minutes
    PROLONGED = "prolonged" # 1-4 hours
    CRITICAL = "critical"   # 4+ hours


class DwellEvent(BaseModel):
    """A dwell event — a target staying in one location for an extended period.

    Generated when a target remains within a configurable radius for longer
    than the dwell threshold. Updated as the target continues to dwell.
    """
    target_id: str = Field(description="Unique target identifier")
    event_id: str = Field(default="", description="Unique dwell event ID")
    position_lat: Optional[float] = Field(None, description="Latitude of dwell center")
    position_lng: Optional[float] = Field(None, description="Longitude of dwell center")
    position_x: Optional[float] = Field(None, description="Local X coordinate")
    position_y: Optional[float] = Field(None, description="Local Y coordinate")
    start_time: Optional[datetime] = Field(None, description="When the target started dwelling")
    end_time: Optional[datetime] = Field(None, description="When the target stopped dwelling")
    duration_s: float = Field(0.0, description="Duration of dwell in seconds")
    zone_id: Optional[str] = Field(None, description="Zone ID if the dwell is within a defined zone")
    zone_name: Optional[str] = Field(None, description="Human-readable zone name")
    state: DwellState = DwellState.ACTIVE
    severity: DwellSeverity = DwellSeverity.NORMAL
    radius_m: float = Field(10.0, description="Radius within which the target is considered dwelling")
    target_name: Optional[str] = Field(None, description="Display name of the target")
    target_alliance: Optional[str] = Field(None, description="Target alliance: friendly/hostile/unknown")
    target_type: Optional[str] = Field(None, description="Target asset type: person/vehicle/phone etc")

    model_config = {"populate_by_name": True}

    @property
    def duration_minutes(self) -> float:
        """Duration in minutes."""
        return self.duration_s / 60.0

    @property
    def duration_display(self) -> str:
        """Human-readable duration string."""
        mins = int(self.duration_s // 60)
        secs = int(self.duration_s % 60)
        if mins >= 60:
            hours = mins // 60
            mins = mins % 60
            return f"{hours}h {mins}m"
        return f"{mins}m {secs}s"


def classify_dwell_severity(duration_s: float) -> DwellSeverity:
    """Classify dwell severity based on duration."""
    if duration_s >= 14400:   # 4 hours
        return DwellSeverity.CRITICAL
    elif duration_s >= 3600:  # 1 hour
        return DwellSeverity.PROLONGED
    elif duration_s >= 900:   # 15 minutes
        return DwellSeverity.EXTENDED
    return DwellSeverity.NORMAL


DWELL_THRESHOLD_S = 300  # 5 minutes default threshold
DWELL_RADIUS_M = 15.0    # meters — movement within this radius counts as dwelling
