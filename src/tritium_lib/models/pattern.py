# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Behavioral pattern learning models for target intelligence.

Defines patterns detected from target movement history: daily routines,
regular commutes, co-presence relationships, and anomaly detection when
established patterns are broken.

MQTT topics:
    tritium/{site}/patterns/{target_id}/detected  — new pattern detected
    tritium/{site}/patterns/{target_id}/broken     — pattern violation
    tritium/{site}/patterns/relationships          — co-presence edges
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class PatternType(str, Enum):
    """Types of behavioral patterns detected from movement history."""

    DAILY_ROUTINE = "daily_routine"  # same time, same place, daily
    WEEKLY_ROUTINE = "weekly_routine"  # same time, same place, weekly
    COMMUTE = "commute"  # same route repeated between locations
    DWELL_PATTERN = "dwell_pattern"  # regular dwell at specific location
    ARRIVAL_PATTERN = "arrival_pattern"  # regular arrival time at location
    DEPARTURE_PATTERN = "departure_pattern"  # regular departure from location
    CO_PRESENCE = "co_presence"  # two devices always together
    PERIODIC_VISIT = "periodic_visit"  # visits location on a schedule


class PatternStatus(str, Enum):
    """Lifecycle of a detected pattern."""

    EMERGING = "emerging"  # seen a few times, not yet confident
    ESTABLISHED = "established"  # high confidence, stable pattern
    BREAKING = "breaking"  # pattern recently violated
    STALE = "stale"  # pattern not reinforced recently


class DeviationType(str, Enum):
    """Types of pattern deviations (anomalies)."""

    MISSING = "missing"  # expected target not seen
    EARLY = "early"  # target arrived/departed earlier than usual
    LATE = "late"  # target arrived/departed later than usual
    WRONG_LOCATION = "wrong_location"  # target at unexpected location
    UNUSUAL_DURATION = "unusual_duration"  # dwell time different from pattern
    NEW_COMPANION = "new_companion"  # co-presence with unknown device
    LOST_COMPANION = "lost_companion"  # expected co-present device missing
    ROUTE_DEVIATION = "route_deviation"  # different path than usual commute


class TimeSlot(BaseModel):
    """A time window in a recurring schedule."""

    hour_start: int = Field(0, ge=0, le=23, description="Start hour (0-23)")
    hour_end: int = Field(23, ge=0, le=23, description="End hour (0-23)")
    minute_start: int = Field(0, ge=0, le=59, description="Start minute")
    minute_end: int = Field(59, ge=0, le=59, description="End minute")
    days_of_week: list[int] = Field(
        default_factory=lambda: list(range(7)),
        description="Days when this slot is active (0=Mon, 6=Sun)",
    )

    def contains_time(self, dt: datetime) -> bool:
        """Check if a datetime falls within this time slot."""
        if dt.weekday() not in self.days_of_week:
            return False
        t = dt.hour * 60 + dt.minute
        start = self.hour_start * 60 + self.minute_start
        end = self.hour_end * 60 + self.minute_end
        if start <= end:
            return start <= t <= end
        # Wraps midnight
        return t >= start or t <= end


class LocationCluster(BaseModel):
    """A cluster of positions representing a frequently visited area."""

    center_lat: float = 0.0
    center_lng: float = 0.0
    radius_m: float = 50.0
    visit_count: int = 0
    avg_dwell_s: float = 0.0
    label: str = ""  # user-assigned or auto-generated label


class BehaviorPattern(BaseModel):
    """A detected behavioral pattern for a target.

    Represents a recurring behavior extracted from movement history:
    daily routines, commute routes, dwell patterns, arrival/departure
    schedules, and co-presence relationships.
    """

    pattern_id: str = Field("", description="Unique pattern identifier")
    target_id: str = Field("", description="Target this pattern belongs to")
    pattern_type: PatternType = PatternType.DAILY_ROUTINE
    status: PatternStatus = PatternStatus.EMERGING
    confidence: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Pattern confidence (0=weak, 1=certain)",
    )
    frequency: str = Field(
        "daily",
        description="How often the pattern repeats (daily, weekly, weekday, etc.)",
    )
    schedule: TimeSlot = Field(
        default_factory=TimeSlot,
        description="When this pattern typically occurs",
    )
    locations: list[LocationCluster] = Field(
        default_factory=list,
        description="Location clusters involved in this pattern",
    )
    observation_count: int = Field(
        0, description="How many times the pattern has been observed",
    )
    first_seen: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp(),
        description="Epoch when pattern was first detected",
    )
    last_seen: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp(),
        description="Epoch when pattern was last reinforced",
    )
    last_expected: float = Field(
        0.0, description="Epoch when the pattern was last expected to occur",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra data (route waypoints, companion IDs, etc.)",
    )

    @property
    def is_established(self) -> bool:
        """Pattern has enough observations to be reliable."""
        return self.confidence >= 0.7 and self.observation_count >= 5

    @property
    def age_days(self) -> float:
        """How many days since the pattern was first detected."""
        now = datetime.now(timezone.utc).timestamp()
        return (now - self.first_seen) / 86400.0

    def reinforce(self) -> None:
        """Reinforce the pattern with a new observation."""
        self.observation_count += 1
        self.last_seen = datetime.now(timezone.utc).timestamp()
        # Increase confidence toward 1.0
        self.confidence = min(1.0, self.confidence + 0.05)
        if self.confidence >= 0.7 and self.observation_count >= 5:
            self.status = PatternStatus.ESTABLISHED


class PatternAnomaly(BaseModel):
    """A detected deviation from an established behavioral pattern.

    Generated when a known pattern is broken: regular commuter doesn't
    appear, device seen at unusual time/place, companion device missing.
    """

    anomaly_id: str = Field("", description="Unique anomaly identifier")
    target_id: str = Field("", description="Target whose pattern was broken")
    pattern_id: str = Field("", description="The pattern that was violated")
    deviation_type: DeviationType = DeviationType.MISSING
    deviation_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="How severe the deviation is (0=minor, 1=extreme)",
    )
    expected_behavior: str = Field(
        "",
        description="Human-readable description of what was expected",
    )
    actual_behavior: str = Field(
        "",
        description="Human-readable description of what was observed",
    )
    timestamp: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp(),
        description="When the anomaly was detected",
    )
    location_lat: float = 0.0
    location_lng: float = 0.0
    acknowledged: bool = Field(
        False, description="Whether an operator has seen this anomaly",
    )
    alert_generated: bool = Field(
        False, description="Whether an alert was created from this anomaly",
    )


class CoPresenceRelationship(BaseModel):
    """Inferred relationship between two devices seen together.

    When two BLE devices have temporal correlation > threshold,
    we infer they are carried by the same person or traveling together.
    Creates TRAVELS_WITH graph edges.
    """

    target_a: str = Field("", description="First target ID")
    target_b: str = Field("", description="Second target ID")
    temporal_correlation: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Fraction of time they appear together",
    )
    spatial_correlation: float = Field(
        0.0, ge=0.0, le=1.0,
        description="How close they are when both detected",
    )
    co_occurrence_count: int = Field(
        0, description="Number of times seen together",
    )
    total_observations: int = Field(
        0, description="Total observations of either target",
    )
    avg_distance_m: float = Field(
        0.0, description="Average distance between detections",
    )
    relationship_type: str = Field(
        "travels_with",
        description="Inferred relationship type (travels_with, carried_by, etc.)",
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Overall confidence in this relationship",
    )
    first_seen: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp(),
    )
    last_seen: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp(),
    )
    graph_edge_created: bool = Field(
        False, description="Whether the TRAVELS_WITH graph edge exists",
    )

    def compute_confidence(self) -> float:
        """Compute overall confidence from temporal and spatial correlation."""
        if self.total_observations < 3:
            return 0.0
        self.confidence = (
            0.6 * self.temporal_correlation
            + 0.3 * self.spatial_correlation
            + 0.1 * min(1.0, self.co_occurrence_count / 20.0)
        )
        return self.confidence


class PatternAlert(BaseModel):
    """An alert rule tied to a specific pattern.

    When the associated pattern is broken, this alert fires.
    Users can create alert rules from detected patterns.
    """

    alert_id: str = Field("", description="Unique alert identifier")
    pattern_id: str = Field("", description="Pattern this alert monitors")
    target_id: str = Field("", description="Target being monitored")
    name: str = Field("", description="Human-readable alert name")
    description: str = Field(
        "",
        description="What this alert watches for",
    )
    enabled: bool = True
    severity: str = Field(
        "medium",
        description="Alert severity: low, medium, high, critical",
    )
    deviation_threshold: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Minimum deviation score to trigger",
    )
    cooldown_seconds: float = Field(
        3600.0,
        description="Minimum time between alert firings",
    )
    last_fired: float = Field(
        0.0, description="Epoch when this alert last fired",
    )
    fire_count: int = Field(
        0, description="Total number of times this alert has fired",
    )
    created_at: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp(),
    )

    def can_fire(self) -> bool:
        """Check if the alert can fire (not in cooldown)."""
        if not self.enabled:
            return False
        now = datetime.now(timezone.utc).timestamp()
        return (now - self.last_fired) >= self.cooldown_seconds

    def fire(self) -> None:
        """Record that the alert has fired."""
        self.last_fired = datetime.now(timezone.utc).timestamp()
        self.fire_count += 1


def compute_temporal_correlation(
    sightings_a: list[float],
    sightings_b: list[float],
    window_s: float = 60.0,
) -> float:
    """Compute temporal correlation between two devices.

    Two devices are temporally correlated if they tend to be seen
    within `window_s` seconds of each other.

    Args:
        sightings_a: Sorted list of epoch timestamps for device A.
        sightings_b: Sorted list of epoch timestamps for device B.
        window_s: Time window for co-occurrence (seconds).

    Returns:
        Fraction of sightings that are temporally correlated (0.0-1.0).
    """
    if not sightings_a or not sightings_b:
        return 0.0

    matches = 0
    j = 0
    for t_a in sightings_a:
        while j < len(sightings_b) and sightings_b[j] < t_a - window_s:
            j += 1
        if j < len(sightings_b) and abs(sightings_b[j] - t_a) <= window_s:
            matches += 1

    total = max(len(sightings_a), len(sightings_b))
    return matches / total if total > 0 else 0.0


def detect_time_regularity(
    timestamps: list[float],
    tolerance_minutes: int = 30,
) -> Optional[TimeSlot]:
    """Detect if timestamps cluster around a regular daily time.

    Args:
        timestamps: List of epoch timestamps.
        tolerance_minutes: Acceptable deviation from the mean time.

    Returns:
        A TimeSlot if a regular pattern is detected, else None.
    """
    if len(timestamps) < 3:
        return None

    # Extract hours and minutes
    times = []
    days = set()
    for ts in timestamps:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        times.append(dt.hour * 60 + dt.minute)
        days.add(dt.weekday())

    # Compute mean time of day (circular mean for clock times)
    import math
    angles = [t / 1440.0 * 2 * math.pi for t in times]
    sin_sum = sum(math.sin(a) for a in angles)
    cos_sum = sum(math.cos(a) for a in angles)
    mean_angle = math.atan2(sin_sum, cos_sum)
    if mean_angle < 0:
        mean_angle += 2 * math.pi
    mean_minutes = int(mean_angle / (2 * math.pi) * 1440)

    # Check if all times are within tolerance of mean
    for t in times:
        diff = abs(t - mean_minutes)
        if diff > 720:
            diff = 1440 - diff  # wrap around midnight
        if diff > tolerance_minutes:
            return None

    mean_hour = mean_minutes // 60
    mean_min = mean_minutes % 60
    start_min = max(0, mean_minutes - tolerance_minutes)
    end_min = min(1439, mean_minutes + tolerance_minutes)

    return TimeSlot(
        hour_start=start_min // 60,
        minute_start=start_min % 60,
        hour_end=end_min // 60,
        minute_end=end_min % 60,
        days_of_week=sorted(days),
    )
