# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Movement analytics models for target velocity, direction, and dwell analysis.

Captures per-target movement metrics (speed, direction, dwell times) and
fleet-wide aggregates. Used by /api/analytics/movement/{target_id} and
fleet analytics dashboards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class ActivityPeriod:
    """A time window when the target was active (moving)."""
    start_epoch: float = 0.0
    end_epoch: float = 0.0
    avg_speed_mps: float = 0.0
    distance_m: float = 0.0

    def to_dict(self) -> dict:
        return {
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
            "avg_speed_mps": round(self.avg_speed_mps, 3),
            "distance_m": round(self.distance_m, 2),
            "duration_s": round(self.end_epoch - self.start_epoch, 1),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ActivityPeriod:
        return cls(
            start_epoch=data.get("start_epoch", 0.0),
            end_epoch=data.get("end_epoch", 0.0),
            avg_speed_mps=data.get("avg_speed_mps", 0.0),
            distance_m=data.get("distance_m", 0.0),
        )


@dataclass
class DwellTime:
    """Time spent in a named zone or area."""
    zone_id: str = ""
    zone_name: str = ""
    total_seconds: float = 0.0
    entry_count: int = 0
    last_entry_epoch: float = 0.0
    last_exit_epoch: float = 0.0

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "total_seconds": round(self.total_seconds, 1),
            "entry_count": self.entry_count,
            "last_entry_epoch": self.last_entry_epoch,
            "last_exit_epoch": self.last_exit_epoch,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DwellTime:
        return cls(
            zone_id=data.get("zone_id", ""),
            zone_name=data.get("zone_name", ""),
            total_seconds=data.get("total_seconds", 0.0),
            entry_count=data.get("entry_count", 0),
            last_entry_epoch=data.get("last_entry_epoch", 0.0),
            last_exit_epoch=data.get("last_exit_epoch", 0.0),
        )


@dataclass
class MovementAnalytics:
    """Movement analytics for a single tracked target.

    Attributes
    ----------
    target_id:
        The unique target identifier.
    avg_speed_mps:
        Average speed in meters per second over the analysis window.
    max_speed_mps:
        Maximum observed speed in meters per second.
    total_distance_m:
        Total distance traveled in meters.
    dwell_times:
        List of dwell-time records per zone.
    direction_histogram:
        Distribution of heading directions (8 compass bins: N, NE, E, SE, S, SW, W, NW).
        Values are fractions summing to 1.0.
    activity_periods:
        List of time windows when the target was actively moving.
    current_speed_mps:
        Most recent speed estimate.
    current_heading_deg:
        Most recent heading (0=north, clockwise).
    is_stationary:
        Whether the target is currently stationary.
    analysis_window_s:
        Time window used for analysis in seconds.
    generated_at:
        When this analysis was generated.
    """
    target_id: str = ""
    avg_speed_mps: float = 0.0
    max_speed_mps: float = 0.0
    total_distance_m: float = 0.0
    dwell_times: list[DwellTime] = field(default_factory=list)
    direction_histogram: dict[str, float] = field(default_factory=dict)
    activity_periods: list[ActivityPeriod] = field(default_factory=list)
    current_speed_mps: float = 0.0
    current_heading_deg: float = 0.0
    is_stationary: bool = True
    analysis_window_s: float = 3600.0
    generated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.generated_at is None:
            self.generated_at = datetime.now(timezone.utc)
        if not self.direction_histogram:
            self.direction_histogram = {
                d: 0.0 for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            }

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "avg_speed_mps": round(self.avg_speed_mps, 3),
            "max_speed_mps": round(self.max_speed_mps, 3),
            "total_distance_m": round(self.total_distance_m, 2),
            "dwell_times": [d.to_dict() for d in self.dwell_times],
            "direction_histogram": {
                k: round(v, 4) for k, v in self.direction_histogram.items()
            },
            "activity_periods": [a.to_dict() for a in self.activity_periods],
            "current_speed_mps": round(self.current_speed_mps, 3),
            "current_heading_deg": round(self.current_heading_deg, 1),
            "is_stationary": self.is_stationary,
            "analysis_window_s": self.analysis_window_s,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MovementAnalytics:
        dwell_times = [
            DwellTime.from_dict(d) for d in data.get("dwell_times", [])
        ]
        activity_periods = [
            ActivityPeriod.from_dict(a) for a in data.get("activity_periods", [])
        ]
        ma = cls(
            target_id=data.get("target_id", ""),
            avg_speed_mps=data.get("avg_speed_mps", 0.0),
            max_speed_mps=data.get("max_speed_mps", 0.0),
            total_distance_m=data.get("total_distance_m", 0.0),
            dwell_times=dwell_times,
            direction_histogram=data.get("direction_histogram", {}),
            activity_periods=activity_periods,
            current_speed_mps=data.get("current_speed_mps", 0.0),
            current_heading_deg=data.get("current_heading_deg", 0.0),
            is_stationary=data.get("is_stationary", True),
            analysis_window_s=data.get("analysis_window_s", 3600.0),
        )
        if data.get("generated_at"):
            ma.generated_at = datetime.fromisoformat(data["generated_at"])
        return ma


@dataclass
class FleetMetrics:
    """Fleet-wide movement aggregate metrics.

    Attributes
    ----------
    total_targets:
        Number of tracked targets in the fleet.
    moving_targets:
        Number of currently moving targets.
    stationary_targets:
        Number of currently stationary targets.
    avg_fleet_speed_mps:
        Average speed across all moving targets.
    max_fleet_speed_mps:
        Maximum speed observed across all targets.
    total_fleet_distance_m:
        Sum of all distances traveled by all targets.
    busiest_zone:
        Zone with the most dwell time across all targets.
    dominant_direction:
        Most common direction of movement across fleet.
    generated_at:
        When these metrics were generated.
    """
    total_targets: int = 0
    moving_targets: int = 0
    stationary_targets: int = 0
    avg_fleet_speed_mps: float = 0.0
    max_fleet_speed_mps: float = 0.0
    total_fleet_distance_m: float = 0.0
    busiest_zone: str = ""
    dominant_direction: str = ""
    generated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.generated_at is None:
            self.generated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "total_targets": self.total_targets,
            "moving_targets": self.moving_targets,
            "stationary_targets": self.stationary_targets,
            "avg_fleet_speed_mps": round(self.avg_fleet_speed_mps, 3),
            "max_fleet_speed_mps": round(self.max_fleet_speed_mps, 3),
            "total_fleet_distance_m": round(self.total_fleet_distance_m, 2),
            "busiest_zone": self.busiest_zone,
            "dominant_direction": self.dominant_direction,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FleetMetrics:
        fm = cls(
            total_targets=data.get("total_targets", 0),
            moving_targets=data.get("moving_targets", 0),
            stationary_targets=data.get("stationary_targets", 0),
            avg_fleet_speed_mps=data.get("avg_fleet_speed_mps", 0.0),
            max_fleet_speed_mps=data.get("max_fleet_speed_mps", 0.0),
            total_fleet_distance_m=data.get("total_fleet_distance_m", 0.0),
            busiest_zone=data.get("busiest_zone", ""),
            dominant_direction=data.get("dominant_direction", ""),
        )
        if data.get("generated_at"):
            fm.generated_at = datetime.fromisoformat(data["generated_at"])
        return fm

    @classmethod
    def from_analytics(cls, analytics_list: list[MovementAnalytics]) -> FleetMetrics:
        """Compute fleet metrics from a list of per-target analytics."""
        if not analytics_list:
            return cls()

        total = len(analytics_list)
        moving = [a for a in analytics_list if not a.is_stationary]
        stationary = total - len(moving)

        speeds = [a.current_speed_mps for a in moving if a.current_speed_mps > 0]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        max_speed = max((a.max_speed_mps for a in analytics_list), default=0.0)
        total_dist = sum(a.total_distance_m for a in analytics_list)

        # Find busiest zone across all targets
        zone_dwell: dict[str, float] = {}
        for a in analytics_list:
            for dw in a.dwell_times:
                zone_dwell[dw.zone_name or dw.zone_id] = (
                    zone_dwell.get(dw.zone_name or dw.zone_id, 0.0) + dw.total_seconds
                )
        busiest = max(zone_dwell, key=zone_dwell.get, default="") if zone_dwell else ""

        # Dominant direction across fleet
        dir_totals: dict[str, float] = {}
        for a in analytics_list:
            for d, v in a.direction_histogram.items():
                dir_totals[d] = dir_totals.get(d, 0.0) + v
        dominant = max(dir_totals, key=dir_totals.get, default="") if dir_totals else ""

        return cls(
            total_targets=total,
            moving_targets=len(moving),
            stationary_targets=stationary,
            avg_fleet_speed_mps=avg_speed,
            max_fleet_speed_mps=max_speed,
            total_fleet_distance_m=total_dist,
            busiest_zone=busiest,
            dominant_direction=dominant,
        )
