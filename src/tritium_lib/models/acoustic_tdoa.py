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


# --- Real hyperbolic TDoA multilateration (least-squares) --------------------
#
# The weighted-centroid above can only place a source *between* the sensors and
# reports a fabricated residual. This solver inverts the actual TDoA hyperbolae
# (|x - s_i| - |x - s_ref| = c * (t_i - t_ref)) via Gauss-Newton, so it
# localizes sources outside the sensor hull and returns a REAL RMS fit error.
# Pure NumPy (no SciPy); falls back to the centroid if NumPy is unavailable.

# Local-tangent-plane scale factors (equirectangular approx — exact enough for a
# sensor array spanning at most a few km).
_M_PER_DEG_LAT = 110_540.0


def _ll_to_local_m(lat: float, lon: float, lat0: float, lon0: float):
    """(lat, lon) -> local east/north meters about (lat0, lon0)."""
    import math as _m

    m_per_deg_lon = 111_320.0 * _m.cos(_m.radians(lat0))
    east = (lon - lon0) * m_per_deg_lon
    north = (lat - lat0) * _M_PER_DEG_LAT
    return east, north


def _local_m_to_ll(east: float, north: float, lat0: float, lon0: float):
    """Local east/north meters about (lat0, lon0) -> (lat, lon)."""
    import math as _m

    m_per_deg_lon = 111_320.0 * _m.cos(_m.radians(lat0))
    lat = lat0 + north / _M_PER_DEG_LAT
    lon = lon0 + east / m_per_deg_lon
    return lat, lon


def compute_tdoa_position_leastsq(
    observations: list[TDoAObservation],
    sound_speed: float = SPEED_OF_SOUND_MPS,
    max_iter: int = 50,
    tol_m: float = 0.01,
) -> Optional[TDoAResult]:
    """Localize a sound source by least-squares TDoA multilateration.

    Solves the hyperbolic system |x - s_i| - |x - s_ref| = c * (t_i - t_ref)
    with Gauss-Newton, seeded from the weighted centroid. Unlike
    :func:`compute_tdoa_position`, this can place the source OUTSIDE the convex
    hull of the sensors and returns ``residual_error_m`` as the real RMS misfit
    of the TDoA equations (meters), not a heuristic.

    Args:
        observations: 3+ TDoAObservation from different sensors with synced
            ``arrival_time_ms`` and ``lat``/``lon``.
        sound_speed: Speed of sound, m/s.
        max_iter: Maximum Gauss-Newton iterations.
        tol_m: Convergence tolerance on the position step, meters.

    Returns:
        TDoAResult(method="tdoa_leastsq") with a real residual, or None if
        fewer than 3 observations. Falls back to the centroid solver if NumPy
        is unavailable or the normal equations are singular.
    """
    if len(observations) < 3:
        return None

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy absent in core-only install
        return compute_tdoa_position(observations, sound_speed)

    # Reference = earliest arrival (closest to source); origin = sensor centroid.
    sorted_obs = sorted(observations, key=lambda o: o.arrival_time_ms)
    lat0 = sum(o.lat for o in sorted_obs) / len(sorted_obs)
    lon0 = sum(o.lon for o in sorted_obs) / len(sorted_obs)

    sensors = np.array(
        [_ll_to_local_m(o.lat, o.lon, lat0, lon0) for o in sorted_obs],
        dtype=np.float64,
    )
    t_s = np.array([o.arrival_time_ms / 1000.0 for o in sorted_obs], dtype=np.float64)

    ref = 0  # earliest
    s_ref = sensors[ref]
    t_ref = t_s[ref]
    others = [i for i in range(len(sorted_obs)) if i != ref]
    # Range differences d_i = c * (t_i - t_ref)  (>= 0 since ref is earliest)
    d = np.array([sound_speed * (t_s[i] - t_ref) for i in others], dtype=np.float64)
    s_oth = sensors[others]

    # Seed from the weighted centroid solution.
    seed = compute_tdoa_position(observations, sound_speed)
    if seed is not None:
        x = np.array(_ll_to_local_m(seed.lat, seed.lon, lat0, lon0), dtype=np.float64)
    else:
        x = sensors.mean(axis=0)

    eps = 1e-9
    converged = False
    for _ in range(max_iter):
        r_oth = np.linalg.norm(x - s_oth, axis=1)  # |x - s_i|
        r_ref = np.linalg.norm(x - s_ref)          # |x - s_ref|
        r_ref = max(r_ref, eps)
        r_oth = np.maximum(r_oth, eps)

        residual = (r_oth - r_ref) - d  # want -> 0
        # Jacobian rows: (x - s_i)/|x - s_i| - (x - s_ref)/|x - s_ref|
        jac = (x - s_oth) / r_oth[:, None] - (x - s_ref) / r_ref

        jtj = jac.T @ jac
        jtr = jac.T @ residual
        try:
            step = np.linalg.solve(jtj + 1e-6 * np.eye(2), jtr)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate geometry
            return compute_tdoa_position(observations, sound_speed)

        x = x - step
        if float(np.linalg.norm(step)) < tol_m:
            converged = True
            break

    # Real RMS residual in meters.
    r_oth = np.maximum(np.linalg.norm(x - s_oth, axis=1), eps)
    r_ref = max(float(np.linalg.norm(x - s_ref)), eps)
    final_res = (r_oth - r_ref) - d
    residual_m = float(np.sqrt(np.mean(final_res ** 2)))

    est_lat, est_lon = _local_m_to_ll(float(x[0]), float(x[1]), lat0, lon0)

    # Confidence: geometry (sensor count + spread), sync quality, and fit.
    n = len(sorted_obs)
    count_score = min(1.0, 0.4 + (n - 3) * 0.2)
    avg_sync = sum(o.ntp_sync_quality for o in sorted_obs) / n
    fit_score = 1.0 / (1.0 + residual_m / 10.0)  # 0 m -> 1.0, 10 m -> 0.5
    if not converged:
        fit_score *= 0.5
    confidence = round(
        0.30 * count_score + 0.25 * avg_sync + 0.45 * fit_score, 3
    )
    confidence = min(1.0, max(0.0, confidence))

    return TDoAResult(
        position=(round(est_lat, 8), round(est_lon, 8)),
        confidence=confidence,
        residual_error_m=round(residual_m, 2),
        sensors_used=[o.sensor_id for o in sorted_obs],
        event_type=sorted_obs[0].event_type,
        method="tdoa_leastsq",
    )
