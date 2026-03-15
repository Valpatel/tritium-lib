# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Acoustic TDoA (Time Difference of Arrival) models for standardized computation.

Provides lightweight data containers for multi-node acoustic event timing.
When 3+ edge nodes detect the same acoustic event within a short time window,
arrival time differences can be used to compute the sound source position.

These models complement the AcousticTrilateration model from acoustic_intelligence
by providing standardized input/output types for TDoA pipelines.

MQTT topics:
    tritium/{site}/acoustic/{sensor_id}/tdoa     -- TDoA observation from a sensor
    tritium/{site}/acoustic/tdoa/result           -- computed TDoA localization result
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TDoAObservation(BaseModel):
    """A single sensor's observation of an acoustic event for TDoA computation.

    Each edge node that detects an acoustic event publishes a TDoAObservation
    with its NTP-synced arrival time. The SC backend collects observations
    within a time window and runs TDoA computation when 3+ arrive.
    """

    sensor_id: str = Field(..., description="Edge node identifier")
    arrival_time_ms: float = Field(
        ...,
        description="NTP-synced arrival time in milliseconds since epoch",
    )
    signal_strength: float = Field(
        0.0,
        description="Signal strength / amplitude in dB (higher = closer to source)",
    )
    # Optional metadata
    event_type: str = Field("unknown", description="Classified event type if available")
    confidence: float = Field(
        1.0,
        description="Detection confidence at this sensor (0.0-1.0)",
    )
    ntp_sync_quality: float = Field(
        0.0,
        description="NTP sync quality indicator (0.0=unsync, 1.0=<1ms jitter)",
    )
    lat: float = Field(0.0, description="Sensor latitude")
    lon: float = Field(0.0, description="Sensor longitude")
    timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp(),
        description="When this observation was created (epoch seconds)",
    )


class TDoAResult(BaseModel):
    """Result of TDoA sound source position computation.

    Computed by the SC backend from 3+ TDoAObservation records.
    Includes position estimate, confidence, residual error, and
    which sensors contributed to the computation.
    """

    position: tuple[float, float] = Field(
        (0.0, 0.0),
        description="Estimated source position (lat, lon)",
    )
    confidence: float = Field(
        0.0,
        description="Overall confidence (0.0-1.0) combining geometry + sync quality",
    )
    residual_error_m: float = Field(
        0.0,
        description="Residual error in meters from the TDoA solution",
    )
    sensors_used: list[str] = Field(
        default_factory=list,
        description="Sensor IDs that contributed to this result",
    )
    method: str = Field(
        "tdoa_weighted_centroid",
        description="Algorithm used (tdoa_weighted_centroid, tdoa_leastsq, etc.)",
    )
    event_type: str = Field("unknown", description="Event type if classified")
    event_id: str = Field("", description="Unique event correlation ID")
    timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp(),
        description="When this result was computed (epoch seconds)",
    )

    @property
    def lat(self) -> float:
        return self.position[0]

    @property
    def lon(self) -> float:
        return self.position[1]


# Speed of sound at 20C in air
SPEED_OF_SOUND_MPS = 343.0


def compute_tdoa_position(
    observations: list[TDoAObservation],
    sound_speed: float = SPEED_OF_SOUND_MPS,
) -> Optional[TDoAResult]:
    """Compute sound source position from 3+ TDoA observations.

    Uses arrival time differences with inverse-distance weighted centroid.
    The earliest observation is assumed closest to the source.

    Requires NTP-synced timestamps on each observation. Sync quality is
    used to weight observations (poorly synced sensors contribute less).

    Args:
        observations: List of TDoAObservation from different sensors.
        sound_speed: Speed of sound in m/s (default 343 for 20C air).

    Returns:
        TDoAResult with estimated position and confidence, or None if
        fewer than 3 observations.
    """
    if len(observations) < 3:
        return None

    # Sort by arrival time — earliest = closest to source
    sorted_obs = sorted(observations, key=lambda o: o.arrival_time_ms)
    t0_ms = sorted_obs[0].arrival_time_ms

    # Build weighted anchors from TDoA
    anchors: list[tuple[float, float, float]] = []  # lat, lon, weight
    for obs in sorted_obs:
        dt_s = (obs.arrival_time_ms - t0_ms) / 1000.0  # convert ms to seconds

        # Weight: inversely proportional to delay, scaled by sync quality and confidence
        sync_factor = max(0.1, obs.ntp_sync_quality)
        conf_factor = max(0.1, obs.confidence)

        if dt_s < 0.001:
            # Closest sensor — high weight
            weight = 10.0 * sync_factor * conf_factor
        else:
            weight = (sync_factor * conf_factor) / max(dt_s, 0.001)

        anchors.append((obs.lat, obs.lon, weight))

    # Weighted centroid
    total_w = sum(w for _, _, w in anchors)
    if total_w <= 0:
        return None

    est_lat = sum(lat * w for lat, _, w in anchors) / total_w
    est_lon = sum(lon * w for _, lon, w in anchors) / total_w

    # Confidence scoring
    n = len(sorted_obs)
    count_score = min(1.0, 0.4 + (n - 3) * 0.2)  # 3 sensors = 0.4, 6+ = 1.0

    # Geometric spread (wider sensor array = better geometry)
    lats = [o.lat for o in sorted_obs]
    lons = [o.lon for o in sorted_obs]
    spread = math.sqrt(
        (max(lats) - min(lats)) ** 2 + (max(lons) - min(lons)) ** 2
    )
    geometry_score = min(1.0, spread * 111_000 / max(sound_speed, 1.0))

    # Sync quality average
    avg_sync = sum(o.ntp_sync_quality for o in sorted_obs) / n

    confidence = round(
        0.35 * count_score + 0.35 * geometry_score + 0.30 * avg_sync,
        3,
    )
    confidence = min(1.0, max(0.0, confidence))

    # Residual error estimate: max TDoA distance mismatch
    max_dt_s = (sorted_obs[-1].arrival_time_ms - t0_ms) / 1000.0
    residual_m = max_dt_s * sound_speed * (1.0 - avg_sync)

    return TDoAResult(
        position=(round(est_lat, 8), round(est_lon, 8)),
        confidence=confidence,
        residual_error_m=round(residual_m, 2),
        sensors_used=[o.sensor_id for o in sorted_obs],
        event_type=sorted_obs[0].event_type,
        method="tdoa_weighted_centroid",
    )
