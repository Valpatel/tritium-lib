# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Target prediction model for trajectory prediction visualization.

A TargetPrediction captures the predicted future positions of a tracked
target based on current heading, speed, and behavioral patterns. The
predicted_positions list contains future position estimates at increasing
time offsets, with a growing confidence cone radius representing
uncertainty. Used by the tactical map to render trajectory prediction
lines and confidence cones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class PredictedPosition:
    """A single predicted future position at a time offset."""
    lat: float = 0.0
    lng: float = 0.0
    time_offset_sec: float = 0.0   # seconds into the future
    confidence: float = 1.0         # 0.0-1.0, decays with time
    radius_m: float = 0.0           # uncertainty radius in meters

    def to_dict(self) -> dict:
        return {
            "lat": self.lat,
            "lng": self.lng,
            "time_offset_sec": self.time_offset_sec,
            "confidence": self.confidence,
            "radius_m": self.radius_m,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PredictedPosition:
        return cls(
            lat=data.get("lat", 0.0),
            lng=data.get("lng", 0.0),
            time_offset_sec=data.get("time_offset_sec", 0.0),
            confidence=data.get("confidence", 1.0),
            radius_m=data.get("radius_m", 0.0),
        )


@dataclass
class TargetPrediction:
    """Predicted trajectory for a tracked target.

    Attributes:
        target_id: The target being predicted.
        current_lat: Current latitude.
        current_lng: Current longitude.
        predicted_positions: Future positions at increasing time offsets.
        heading_deg: Current heading in degrees (0=N, 90=E).
        speed_mps: Current speed in meters per second.
        confidence_decay_rate: How fast confidence decays per second.
        model: Prediction model used (linear, kalman, behavioral).
        timestamp: When this prediction was generated.
    """
    target_id: str = ""
    current_lat: float = 0.0
    current_lng: float = 0.0
    predicted_positions: list[PredictedPosition] = field(default_factory=list)
    heading_deg: float = 0.0
    speed_mps: float = 0.0
    confidence_decay_rate: float = 0.01  # per second
    model: str = "linear"
    timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def generate_linear_predictions(
        self,
        steps: int = 5,
        interval_sec: float = 30.0,
        base_radius_m: float = 5.0,
    ) -> None:
        """Generate linear trajectory predictions from current state.

        Projects the target forward along its current heading at its
        current speed, with growing uncertainty radius.

        Args:
            steps: Number of future positions to predict.
            interval_sec: Time between each prediction step.
            base_radius_m: Starting uncertainty radius in meters.
        """
        self.predicted_positions.clear()
        heading_rad = math.radians(self.heading_deg)
        # Approximate meters-per-degree at current latitude
        m_per_deg_lat = 111_320.0
        m_per_deg_lng = 111_320.0 * math.cos(math.radians(self.current_lat))

        for i in range(1, steps + 1):
            t = interval_sec * i
            dist_m = self.speed_mps * t
            dlat = (dist_m * math.cos(heading_rad)) / m_per_deg_lat
            dlng = (dist_m * math.sin(heading_rad)) / max(m_per_deg_lng, 1.0)
            confidence = max(0.0, 1.0 - self.confidence_decay_rate * t)
            radius = base_radius_m + (1.0 - confidence) * dist_m * 0.5
            self.predicted_positions.append(PredictedPosition(
                lat=self.current_lat + dlat,
                lng=self.current_lng + dlng,
                time_offset_sec=t,
                confidence=round(confidence, 4),
                radius_m=round(radius, 2),
            ))

    @property
    def max_prediction_time_sec(self) -> float:
        """Maximum time offset in the prediction set."""
        if not self.predicted_positions:
            return 0.0
        return max(p.time_offset_sec for p in self.predicted_positions)

    @property
    def min_confidence(self) -> float:
        """Lowest confidence in the prediction set."""
        if not self.predicted_positions:
            return 0.0
        return min(p.confidence for p in self.predicted_positions)

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "current_lat": self.current_lat,
            "current_lng": self.current_lng,
            "predicted_positions": [p.to_dict() for p in self.predicted_positions],
            "heading_deg": self.heading_deg,
            "speed_mps": self.speed_mps,
            "confidence_decay_rate": self.confidence_decay_rate,
            "model": self.model,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TargetPrediction:
        pred = cls(
            target_id=data.get("target_id", ""),
            current_lat=data.get("current_lat", 0.0),
            current_lng=data.get("current_lng", 0.0),
            predicted_positions=[
                PredictedPosition.from_dict(p)
                for p in data.get("predicted_positions", [])
            ],
            heading_deg=data.get("heading_deg", 0.0),
            speed_mps=data.get("speed_mps", 0.0),
            confidence_decay_rate=data.get("confidence_decay_rate", 0.01),
            model=data.get("model", "linear"),
        )
        if data.get("timestamp"):
            pred.timestamp = datetime.fromisoformat(data["timestamp"])
        return pred
