# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Behavioral pattern recognition models for target analysis.

Tracks movement patterns, routine detection, and anomaly alerting
to identify suspicious or unusual behavior in the unified picture.

MQTT topics:
    tritium/{site}/behavior/{target_id}/pattern  — detected patterns
    tritium/{site}/behavior/anomalies             — anomaly alerts
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BehaviorType(str, Enum):
    """Types of detected behavioral patterns."""

    ROUTINE = "routine"  # regular/predictable movement
    LOITERING = "loitering"  # staying in one area unusually long
    PATROL = "patrol"  # regular patrol-like movement
    SURVEILLANCE = "surveillance"  # circling/observing a location
    APPROACH = "approach"  # moving toward a specific target/location
    RETREAT = "retreat"  # moving away quickly
    ERRATIC = "erratic"  # unpredictable movement
    STATIONARY = "stationary"  # not moving for extended time
    CONVOY = "convoy"  # multiple targets moving together
    UNKNOWN = "unknown"


class AnomalyType(str, Enum):
    """Types of behavioral anomalies."""

    NEW_DEVICE = "new_device"  # previously unseen device
    UNUSUAL_TIME = "unusual_time"  # activity outside normal hours
    UNUSUAL_LOCATION = "unusual_location"  # target in unexpected area
    SPEED_ANOMALY = "speed_anomaly"  # moving unusually fast/slow
    PATTERN_BREAK = "pattern_break"  # deviation from established routine
    DWELL_ANOMALY = "dwell_anomaly"  # unusual dwell time
    ASSOCIATION_ANOMALY = "association_anomaly"  # unusual target co-occurrence
    FREQUENCY_ANOMALY = "frequency_anomaly"  # unusual visit frequency


class AnomalySeverity(str, Enum):
    """Severity of a behavioral anomaly."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class PositionSample(BaseModel):
    """A single position sample for pattern analysis."""

    latitude: float
    longitude: float
    timestamp: float
    speed_mps: float = 0.0
    heading_deg: float = 0.0
    source: str = ""  # ble, gps, trilateration


class BehaviorPattern(BaseModel):
    """A detected behavioral pattern for a target."""

    target_id: str
    behavior_type: BehaviorType = BehaviorType.UNKNOWN
    confidence: float = 0.0  # 0.0-1.0
    start_time: float = 0.0
    end_time: float = 0.0
    duration_s: float = 0.0
    center_lat: float = 0.0
    center_lng: float = 0.0
    radius_m: float = 0.0  # area of activity
    samples: int = 0  # number of position samples
    metadata: dict = Field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Pattern is still ongoing (end_time not set or recent)."""
        if self.end_time == 0:
            return True
        return (datetime.now().timestamp() - self.end_time) < 60


class BehaviorAnomaly(BaseModel):
    """A detected behavioral anomaly."""

    target_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity = AnomalySeverity.INFO
    confidence: float = 0.0
    description: str = ""
    timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp()
    )
    location_lat: float = 0.0
    location_lng: float = 0.0
    baseline_value: str = ""  # what was expected
    observed_value: str = ""  # what was actually seen
    pattern_id: str = ""  # related pattern if any


class TargetRoutine(BaseModel):
    """Learned routine for a target (daily/weekly pattern)."""

    target_id: str
    name: str = ""
    typical_locations: list[dict] = Field(default_factory=list)  # [{lat, lng, time_range, frequency}]
    active_hours: list[int] = Field(default_factory=list)  # hours 0-23 when typically seen
    active_days: list[int] = Field(default_factory=list)  # days 0-6 (Mon-Sun) when typically seen
    avg_speed_mps: float = 0.0
    avg_dwell_time_s: float = 0.0
    total_observations: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    confidence: float = 0.0  # how established the routine is


class CorrelationScore(BaseModel):
    """Correlation score between two targets (for fusion)."""

    target_a: str
    target_b: str
    score: float = 0.0  # 0.0 = unrelated, 1.0 = same entity
    reasons: list[str] = Field(default_factory=list)
    temporal_overlap: float = 0.0  # fraction of time seen together
    spatial_proximity_m: float = 0.0  # average distance between detections
    co_movement_score: float = 0.0  # do they move together?
    source_a: str = ""  # e.g., "ble"
    source_b: str = ""  # e.g., "camera"


def classify_anomaly_severity(anomaly_type: AnomalyType) -> AnomalySeverity:
    """Default severity for each anomaly type."""
    severity_map = {
        AnomalyType.NEW_DEVICE: AnomalySeverity.LOW,
        AnomalyType.UNUSUAL_TIME: AnomalySeverity.MEDIUM,
        AnomalyType.UNUSUAL_LOCATION: AnomalySeverity.MEDIUM,
        AnomalyType.SPEED_ANOMALY: AnomalySeverity.LOW,
        AnomalyType.PATTERN_BREAK: AnomalySeverity.MEDIUM,
        AnomalyType.DWELL_ANOMALY: AnomalySeverity.LOW,
        AnomalyType.ASSOCIATION_ANOMALY: AnomalySeverity.HIGH,
        AnomalyType.FREQUENCY_ANOMALY: AnomalySeverity.LOW,
    }
    return severity_map.get(anomaly_type, AnomalySeverity.INFO)


def compute_correlation_score(
    temporal_overlap: float,
    spatial_proximity_m: float,
    co_movement: float,
    max_proximity_m: float = 50.0,
) -> float:
    """Compute a correlation score from individual factors.

    Returns a score between 0.0 (unrelated) and 1.0 (same entity).
    """
    # Spatial component: closer = higher score
    spatial_score = max(0, 1.0 - spatial_proximity_m / max(max_proximity_m, 1))

    # Weighted combination
    score = (
        0.4 * temporal_overlap
        + 0.35 * spatial_score
        + 0.25 * co_movement
    )
    return min(1.0, max(0.0, score))
