# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Sensor health monitoring models — track per-sensor sighting rates
and flag when a sensor goes quiet (possible failure, obstruction, or tampering)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SensorHealthStatus(str, Enum):
    """Health status of a sensor based on sighting rate deviation."""
    HEALTHY = "healthy"           # Within normal operating range
    DEGRADED = "degraded"         # 25-50% below baseline
    CRITICAL = "critical"         # >50% below baseline
    OFFLINE = "offline"           # No sightings received
    UNKNOWN = "unknown"           # Insufficient data for baseline


class SensorHealthMetrics(BaseModel):
    """Health metrics for a single sensor in the array.

    Tracks sighting rates against a learned baseline and flags
    deviations that may indicate sensor failure, obstruction, or tampering.
    """
    sensor_id: str
    sighting_rate: float = 0.0          # Current sightings per minute
    baseline_rate: float = 0.0          # Learned baseline sightings per minute
    deviation_pct: float = 0.0          # % deviation from baseline (negative = fewer sightings)
    status: SensorHealthStatus = SensorHealthStatus.UNKNOWN
    last_seen: Optional[datetime] = None
    sighting_count: int = 0             # Total sightings in current window
    window_seconds: float = 300.0       # Measurement window (default 5 min)
    alert_message: Optional[str] = None # Human-readable alert if degraded/critical

    def is_healthy(self) -> bool:
        """True if sensor is operating within normal parameters."""
        return self.status == SensorHealthStatus.HEALTHY

    def to_alert_dict(self) -> dict:
        """Return alert-worthy fields for notification pipeline."""
        return {
            "sensor_id": self.sensor_id,
            "status": self.status.value,
            "deviation_pct": round(self.deviation_pct, 1),
            "sighting_rate": round(self.sighting_rate, 2),
            "baseline_rate": round(self.baseline_rate, 2),
            "alert_message": self.alert_message,
        }


class SensorArrayHealth(BaseModel):
    """Aggregate health status for an entire sensor array."""
    sensors: list[SensorHealthMetrics] = Field(default_factory=list)
    healthy_count: int = 0
    degraded_count: int = 0
    critical_count: int = 0
    offline_count: int = 0
    overall_status: SensorHealthStatus = SensorHealthStatus.UNKNOWN

    def compute_overall(self) -> None:
        """Recompute aggregate counts and overall status from sensor list."""
        self.healthy_count = sum(1 for s in self.sensors if s.status == SensorHealthStatus.HEALTHY)
        self.degraded_count = sum(1 for s in self.sensors if s.status == SensorHealthStatus.DEGRADED)
        self.critical_count = sum(1 for s in self.sensors if s.status == SensorHealthStatus.CRITICAL)
        self.offline_count = sum(1 for s in self.sensors if s.status == SensorHealthStatus.OFFLINE)

        if not self.sensors:
            self.overall_status = SensorHealthStatus.UNKNOWN
        elif self.critical_count > 0 or self.offline_count > 0:
            self.overall_status = SensorHealthStatus.CRITICAL
        elif self.degraded_count > 0:
            self.overall_status = SensorHealthStatus.DEGRADED
        else:
            self.overall_status = SensorHealthStatus.HEALTHY


class SensorBaseline(BaseModel):
    """Learned baseline for a sensor's normal operating parameters.

    Built from historical observations over a configurable training window.
    Used to detect deviations that may indicate failure, obstruction, or tampering.
    """
    sensor_id: str
    sighting_rate_mean: float = 0.0       # Mean sightings per minute during baseline window
    sighting_rate_stddev: float = 0.0     # Standard deviation of sighting rate
    min_sighting_rate: float = 0.0        # Minimum observed sighting rate
    max_sighting_rate: float = 0.0        # Maximum observed sighting rate
    training_window_hours: float = 24.0   # Hours of data used to build baseline
    sample_count: int = 0                 # Number of samples used
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_valid: bool = False                # True once enough samples collected

    def deviation_from(self, current_rate: float) -> float:
        """Compute how many standard deviations the current rate is from the mean.

        Returns negative values when below the mean, positive when above.
        Returns 0.0 if baseline is not valid or stddev is zero.
        """
        if not self.is_valid or self.sighting_rate_stddev <= 0.0:
            return 0.0
        return (current_rate - self.sighting_rate_mean) / self.sighting_rate_stddev


class SensorAlert(BaseModel):
    """Alert generated when a sensor deviates from its baseline.

    Captures the sensor, the nature of the deviation, and recommended action.
    """
    sensor_id: str
    alert_type: str = "deviation"          # deviation, offline, tamper, degraded
    severity: SensorHealthStatus = SensorHealthStatus.UNKNOWN
    message: str = ""
    sighting_rate: float = 0.0             # Current sighting rate at time of alert
    baseline_rate: float = 0.0             # Expected baseline rate
    deviation_pct: float = 0.0             # Percentage deviation from baseline
    deviation_sigma: float = 0.0           # Standard deviations from mean
    timestamp: Optional[datetime] = None
    acknowledged: bool = False
    recommended_action: str = ""           # e.g., "inspect sensor", "check for jamming"

    def to_notification_dict(self) -> dict:
        """Return fields suitable for the notification pipeline."""
        return {
            "sensor_id": self.sensor_id,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "message": self.message,
            "deviation_pct": round(self.deviation_pct, 1),
            "deviation_sigma": round(self.deviation_sigma, 2),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "recommended_action": self.recommended_action,
        }


def classify_sensor_health(
    sighting_rate: float,
    baseline_rate: float,
    offline_threshold_seconds: float = 300.0,
    seconds_since_last: Optional[float] = None,
) -> SensorHealthStatus:
    """Classify sensor health based on sighting rate deviation.

    Args:
        sighting_rate: Current sightings per minute.
        baseline_rate: Learned baseline sightings per minute.
        offline_threshold_seconds: Seconds of silence before marking offline.
        seconds_since_last: Seconds since last sighting (None = unknown).

    Returns:
        SensorHealthStatus classification.
    """
    if seconds_since_last is not None and seconds_since_last >= offline_threshold_seconds:
        return SensorHealthStatus.OFFLINE

    if baseline_rate <= 0.0:
        return SensorHealthStatus.UNKNOWN

    deviation = (sighting_rate - baseline_rate) / baseline_rate * 100.0
    if deviation >= -25.0:
        return SensorHealthStatus.HEALTHY
    elif deviation >= -50.0:
        return SensorHealthStatus.DEGRADED
    else:
        return SensorHealthStatus.CRITICAL
