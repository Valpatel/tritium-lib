# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Vehicle behavior tracking models for YOLO-detected vehicles.

Standardized vehicle behavior analysis: speed, heading, road segment,
stopped duration, and suspicious scoring. When YOLO detects a vehicle,
consecutive frame positions are used to compute speed and direction.
Vehicles moving >30mph on roads are normal; vehicles stopping in unusual
locations or exhibiting erratic behavior are flagged as suspicious.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class VehicleTrack(BaseModel):
    """Tracked vehicle behavior profile from consecutive YOLO detections.

    Attributes
    ----------
    target_id:
        Unique target identifier (e.g., ``det_car_5``, ``det_truck_2``).
    speed_mph:
        Estimated speed in miles per hour from consecutive frame positions.
    heading:
        Heading in degrees (0=north, clockwise).
    road_segment:
        Identifier of the road segment the vehicle is on, or empty if unknown.
    stopped_duration_s:
        How long the vehicle has been stopped (speed < 2mph) in seconds.
        0.0 if currently moving.
    suspicious_score:
        0.0 to 1.0 score indicating how suspicious the vehicle behavior is.
        High scores for: stopping in unusual locations, loitering, erratic speed.
    vehicle_class:
        YOLO class name (car, truck, bus, motorcycle, bicycle).
    track_id:
        YOLO tracker ID for cross-frame association.
    position:
        Current position as (lat, lng) or (x, y) in local coords.
    last_positions:
        Recent position history for trail rendering (up to 20 entries).
        Each entry is (x, y, timestamp).
    direction_label:
        Human-readable direction (N, NE, E, SE, S, SW, W, NW).
    is_parked:
        True if vehicle has been stopped for >60 seconds.
    generated_at:
        When this track was last updated.
    """

    target_id: str
    speed_mph: float = 0.0
    heading: float = 0.0
    road_segment: str = ""
    stopped_duration_s: float = 0.0
    suspicious_score: float = Field(default=0.0, ge=0.0, le=1.0)
    vehicle_class: str = "car"
    track_id: Optional[int] = None
    position: tuple[float, float] = (0.0, 0.0)
    last_positions: list[tuple[float, float, float]] = Field(default_factory=list)
    direction_label: str = ""
    is_parked: bool = False
    generated_at: Optional[datetime] = None

    def model_post_init(self, __context: object) -> None:
        if self.generated_at is None:
            self.generated_at = datetime.now(timezone.utc)
        if not self.direction_label:
            self.direction_label = heading_to_label(self.heading)

    def is_suspicious(self, threshold: float = 0.5) -> bool:
        """True if suspicious score exceeds the given threshold."""
        return self.suspicious_score >= threshold

    def is_moving(self, speed_threshold_mph: float = 2.0) -> bool:
        """True if vehicle is currently moving above the threshold."""
        return self.speed_mph >= speed_threshold_mph

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "target_id": self.target_id,
            "speed_mph": round(self.speed_mph, 1),
            "heading": round(self.heading, 1),
            "road_segment": self.road_segment,
            "stopped_duration_s": round(self.stopped_duration_s, 1),
            "suspicious_score": round(self.suspicious_score, 3),
            "vehicle_class": self.vehicle_class,
            "track_id": self.track_id,
            "position": self.position,
            "direction_label": self.direction_label,
            "is_parked": self.is_parked,
            "generated_at": (
                self.generated_at.isoformat() if self.generated_at else None
            ),
        }


def heading_to_label(heading_deg: float) -> str:
    """Convert heading in degrees to compass label (N, NE, E, etc.)."""
    normalized = heading_deg % 360
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((normalized + 22.5) / 45.0) % 8
    return directions[idx]


def compute_speed_mph(
    pos_a: tuple[float, float],
    pos_b: tuple[float, float],
    time_delta_s: float,
    meters_per_unit: float = 1.0,
) -> float:
    """Compute speed in mph from two positions and time delta.

    Args:
        pos_a: First position (x, y) or (lat, lng).
        pos_b: Second position (x, y) or (lat, lng).
        time_delta_s: Time between positions in seconds.
        meters_per_unit: Conversion factor (1.0 if positions are in meters).

    Returns:
        Speed in miles per hour.
    """
    if time_delta_s <= 0:
        return 0.0
    import math

    dx = (pos_b[0] - pos_a[0]) * meters_per_unit
    dy = (pos_b[1] - pos_a[1]) * meters_per_unit
    distance_m = math.hypot(dx, dy)
    speed_mps = distance_m / time_delta_s
    return speed_mps * 2.23694  # m/s to mph


def compute_heading(
    pos_a: tuple[float, float],
    pos_b: tuple[float, float],
) -> float:
    """Compute heading in degrees from pos_a to pos_b (0=north, clockwise).

    Args:
        pos_a: Starting position (x, y).
        pos_b: Ending position (x, y).

    Returns:
        Heading in degrees [0, 360).
    """
    import math

    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]
    if dx == 0 and dy == 0:
        return 0.0
    angle_rad = math.atan2(dx, dy)  # atan2(east, north) for compass heading
    heading = math.degrees(angle_rad) % 360
    return heading


def compute_suspicious_score(
    speed_mph: float,
    stopped_duration_s: float,
    is_unusual_location: bool = False,
    speed_variance: float = 0.0,
    heading_change_rate: float = 0.0,
) -> float:
    """Compute suspicious score for a vehicle.

    Normal behavior: >30mph on roads, smooth heading, no loitering.
    Suspicious behavior: stopping in unusual places, erratic movement,
    very slow crawling through areas.

    Args:
        speed_mph: Current speed in mph.
        stopped_duration_s: How long vehicle has been stopped.
        is_unusual_location: Whether the stop location is unusual (not a
            parking lot, intersection, etc.).
        speed_variance: Variance in speed over recent history.
        heading_change_rate: Rate of heading change in degrees/second.

    Returns:
        Suspicious score between 0.0 and 1.0.
    """
    score = 0.0

    # Loitering: stopped for a long time
    if stopped_duration_s > 300:  # 5+ minutes
        score += 0.3
    elif stopped_duration_s > 60:  # 1-5 minutes
        score += 0.15

    # Unusual location amplifier
    if is_unusual_location and stopped_duration_s > 30:
        score += 0.25

    # Very slow crawling (2-10 mph) — potential surveillance behavior
    if 2.0 < speed_mph < 10.0:
        score += 0.15

    # Erratic speed changes
    if speed_variance > 100:  # High variance in mph^2
        score += 0.15
    elif speed_variance > 25:
        score += 0.08

    # Erratic heading (lots of turns)
    if heading_change_rate > 30:  # degrees/second
        score += 0.15
    elif heading_change_rate > 10:
        score += 0.08

    return min(1.0, max(0.0, round(score, 3)))
