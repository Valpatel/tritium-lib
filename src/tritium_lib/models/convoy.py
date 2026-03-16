# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Convoy model — coordinated target movement detection.

When 3+ targets move together at similar speed in the same direction,
they form a convoy. Convoys are suspicious because they indicate
coordinated movement — vehicles in formation, a group of people
walking together, or devices being transported together.

The convoy model tracks:
  - Member target IDs (3+ required)
  - Average speed and heading of the group
  - Formation type (line, cluster, spread)
  - Suspicion score based on coordination tightness and duration
  - First/last seen timestamps for duration tracking
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ConvoyFormation(str, Enum):
    """How convoy members are arranged spatially."""

    LINE = "line"            # Targets in a line (vehicle convoy)
    CLUSTER = "cluster"      # Targets tightly grouped (walking group)
    SPREAD = "spread"        # Targets moving together but spread out
    UNKNOWN = "unknown"


class ConvoyStatus(str, Enum):
    """Current status of the convoy."""

    ACTIVE = "active"        # Convoy is currently moving together
    DISPERSED = "dispersed"  # Members have separated
    STOPPED = "stopped"      # Convoy has stopped but members still together
    MERGED = "merged"        # Convoy merged with another convoy


class Convoy(BaseModel):
    """A group of 3+ targets moving together in coordinated fashion.

    Attributes:
        convoy_id: Unique identifier for this convoy.
        member_target_ids: List of target IDs in this convoy (minimum 3).
        speed_avg_mps: Average speed of the convoy in meters per second.
        heading_avg_deg: Average heading in degrees (0=north, 90=east).
        formation: Spatial arrangement of convoy members.
        status: Current convoy status.
        first_seen: When the convoy was first detected.
        last_seen: When the convoy was last updated.
        suspicious_score: 0.0-1.0 score based on coordination and duration.
        heading_variance_deg: Variance in member headings (low = more coordinated).
        speed_variance_mps: Variance in member speeds (low = more coordinated).
        center_lat: Latitude of convoy center.
        center_lng: Longitude of convoy center.
        duration_s: How long the convoy has been active in seconds.
        source_node_id: Which sensor node first detected the convoy.
    """

    convoy_id: str = ""
    member_target_ids: list[str] = Field(default_factory=list)
    speed_avg_mps: float = 0.0
    heading_avg_deg: float = 0.0
    formation: ConvoyFormation = ConvoyFormation.UNKNOWN
    status: ConvoyStatus = ConvoyStatus.ACTIVE
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    suspicious_score: float = 0.0
    heading_variance_deg: float = 0.0
    speed_variance_mps: float = 0.0
    center_lat: float = 0.0
    center_lng: float = 0.0
    duration_s: float = 0.0
    source_node_id: str = ""

    def model_post_init(self, __context) -> None:
        now = datetime.now(timezone.utc)
        if self.first_seen is None:
            self.first_seen = now
        if self.last_seen is None:
            self.last_seen = now

    @property
    def member_count(self) -> int:
        """Number of targets in this convoy."""
        return len(self.member_target_ids)

    @property
    def is_valid(self) -> bool:
        """A convoy requires at least 3 members."""
        return len(self.member_target_ids) >= 3

    def add_member(self, target_id: str) -> bool:
        """Add a target to the convoy. Returns True if added."""
        if target_id not in self.member_target_ids:
            self.member_target_ids.append(target_id)
            self.last_seen = datetime.now(timezone.utc)
            return True
        return False

    def remove_member(self, target_id: str) -> bool:
        """Remove a target from the convoy. Returns True if removed."""
        if target_id in self.member_target_ids:
            self.member_target_ids.remove(target_id)
            self.last_seen = datetime.now(timezone.utc)
            if len(self.member_target_ids) < 3:
                self.status = ConvoyStatus.DISPERSED
            return True
        return False

    def compute_suspicious_score(self) -> float:
        """Compute suspicion score based on coordination tightness and duration.

        Factors:
          - Low heading variance = high coordination (0.3 weight)
          - Low speed variance = high coordination (0.3 weight)
          - Long duration = more suspicious (0.2 weight)
          - More members = more suspicious (0.2 weight)
        """
        # Heading coordination (0-1): lower variance = higher score
        heading_score = max(0.0, 1.0 - (self.heading_variance_deg / 45.0))

        # Speed coordination (0-1): lower variance = higher score
        speed_score = max(0.0, 1.0 - (self.speed_variance_mps / 2.0))

        # Duration factor (0-1): ramps up over 10 minutes
        duration_score = min(1.0, self.duration_s / 600.0)

        # Member count factor (0-1): 3 members = 0.5, 6+ = 1.0
        member_score = min(1.0, (self.member_count - 2) / 4.0)

        self.suspicious_score = (
            heading_score * 0.3
            + speed_score * 0.3
            + duration_score * 0.2
            + member_score * 0.2
        )
        return self.suspicious_score


class ConvoySummary(BaseModel):
    """Summary statistics for all active convoys."""

    total_convoys: int = 0
    active_convoys: int = 0
    total_members: int = 0
    avg_suspicious_score: float = 0.0
    highest_suspicious_score: float = 0.0
    largest_convoy_size: int = 0
