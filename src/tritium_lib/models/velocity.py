# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Velocity profile models for target movement consistency tracking.

Captures per-target velocity metrics (speed, acceleration, heading changes)
and computes an anomaly score when movement patterns deviate from expected
behavior (e.g., sudden acceleration, erratic heading changes).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class VelocityProfile(BaseModel):
    """Velocity profile for a single tracked target.

    Tracks current and historical speed/acceleration metrics and computes
    an anomaly score based on deviation from typical movement patterns.

    Attributes
    ----------
    target_id:
        Unique target identifier (e.g., ``ble_aa:bb:cc``, ``det_person_3``).
    current_speed:
        Current estimated speed in meters per second.
    max_speed:
        Maximum observed speed in m/s over the analysis window.
    avg_speed:
        Average speed in m/s over the analysis window.
    acceleration:
        Current acceleration in m/s^2 (positive = speeding up, negative = slowing).
    heading_change_rate:
        Rate of heading change in degrees per second. High values indicate
        erratic or evasive movement.
    anomaly_score:
        0.0 to 1.0 score indicating how anomalous the current velocity profile
        is compared to expected behavior. 0.0 = normal, 1.0 = highly anomalous.
    heading_deg:
        Current heading in degrees (0=north, clockwise).
    min_speed:
        Minimum observed speed in m/s over the analysis window.
    speed_variance:
        Variance of speed samples over the analysis window. High variance
        indicates inconsistent movement (stop-and-go, evasion).
    sample_count:
        Number of position samples used to compute this profile.
    analysis_window_s:
        Time window in seconds over which metrics are computed.
    is_stationary:
        True if the target is currently stationary (speed below threshold).
    generated_at:
        When this profile was generated.
    """
    target_id: str
    current_speed: float = 0.0
    max_speed: float = 0.0
    avg_speed: float = 0.0
    acceleration: float = 0.0
    heading_change_rate: float = 0.0
    anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
    heading_deg: float = 0.0
    min_speed: float = 0.0
    speed_variance: float = 0.0
    sample_count: int = 0
    analysis_window_s: float = 300.0
    is_stationary: bool = True
    generated_at: Optional[datetime] = None

    def model_post_init(self, __context: object) -> None:
        if self.generated_at is None:
            self.generated_at = datetime.now(timezone.utc)

    def is_anomalous(self, threshold: float = 0.5) -> bool:
        """True if anomaly score exceeds the given threshold."""
        return self.anomaly_score >= threshold

    def speed_consistency(self) -> float:
        """Return 0.0-1.0 score for speed consistency.

        1.0 means perfectly consistent speed (zero variance relative to avg).
        0.0 means highly erratic speed.
        """
        if self.avg_speed <= 0.0 or self.sample_count < 2:
            return 1.0
        relative_variance = self.speed_variance / (self.avg_speed ** 2)
        # Clamp: variance equal to avg^2 gives 0.0 consistency
        return max(0.0, 1.0 - min(1.0, relative_variance))

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "target_id": self.target_id,
            "current_speed": round(self.current_speed, 3),
            "max_speed": round(self.max_speed, 3),
            "avg_speed": round(self.avg_speed, 3),
            "min_speed": round(self.min_speed, 3),
            "acceleration": round(self.acceleration, 4),
            "heading_change_rate": round(self.heading_change_rate, 2),
            "heading_deg": round(self.heading_deg, 1),
            "anomaly_score": round(self.anomaly_score, 3),
            "speed_variance": round(self.speed_variance, 4),
            "sample_count": self.sample_count,
            "analysis_window_s": self.analysis_window_s,
            "is_stationary": self.is_stationary,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> VelocityProfile:
        """Deserialize from a plain dictionary."""
        vp = cls(
            target_id=data.get("target_id", ""),
            current_speed=data.get("current_speed", 0.0),
            max_speed=data.get("max_speed", 0.0),
            avg_speed=data.get("avg_speed", 0.0),
            min_speed=data.get("min_speed", 0.0),
            acceleration=data.get("acceleration", 0.0),
            heading_change_rate=data.get("heading_change_rate", 0.0),
            heading_deg=data.get("heading_deg", 0.0),
            anomaly_score=data.get("anomaly_score", 0.0),
            speed_variance=data.get("speed_variance", 0.0),
            sample_count=data.get("sample_count", 0),
            analysis_window_s=data.get("analysis_window_s", 300.0),
            is_stationary=data.get("is_stationary", True),
        )
        if data.get("generated_at"):
            vp.generated_at = datetime.fromisoformat(data["generated_at"])
        return vp


def compute_anomaly_score(
    speed_variance: float,
    avg_speed: float,
    acceleration: float,
    heading_change_rate: float,
    max_expected_speed: float = 30.0,
    max_expected_accel: float = 5.0,
    max_expected_heading_rate: float = 45.0,
) -> float:
    """Compute a 0.0-1.0 anomaly score from velocity metrics.

    The score combines three factors:
    - Speed consistency (variance relative to average)
    - Acceleration magnitude relative to expected maximum
    - Heading change rate relative to expected maximum

    Args:
        speed_variance: Variance of speed samples.
        avg_speed: Average speed in m/s.
        acceleration: Current acceleration in m/s^2.
        heading_change_rate: Heading change rate in deg/s.
        max_expected_speed: Maximum expected speed for this target class.
        max_expected_accel: Maximum expected acceleration.
        max_expected_heading_rate: Maximum expected heading change rate.

    Returns:
        Anomaly score between 0.0 and 1.0.
    """
    # Speed consistency factor
    if avg_speed > 0.0:
        speed_factor = min(1.0, speed_variance / (avg_speed ** 2))
    else:
        speed_factor = 0.0

    # Acceleration factor
    accel_factor = min(1.0, abs(acceleration) / max_expected_accel)

    # Heading erraticism factor
    heading_factor = min(1.0, abs(heading_change_rate) / max_expected_heading_rate)

    # Weighted combination
    score = 0.4 * speed_factor + 0.3 * accel_factor + 0.3 * heading_factor
    return min(1.0, max(0.0, round(score, 4)))
