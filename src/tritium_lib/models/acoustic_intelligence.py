# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Acoustic intelligence models for ML-based sound classification and localization.

Extends the basic acoustic_event models with:
- SoundSignature: frequency/energy profile for ML classification
- AcousticTrilateration: multi-node sound source localization via TDoA
- SoundClassification: ML model output with confidence and model version
- AudioFeatureVector: MFCC + spectral features from edge devices

MQTT topics:
    tritium/{site}/acoustic/{sensor_id}/features   — audio feature vectors
    tritium/{site}/acoustic/{sensor_id}/classify    — ML classification results
    tritium/{site}/acoustic/localization            — triangulated source positions
"""

import math
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AudioFeatureVector(BaseModel):
    """Audio features extracted on edge devices for SC-side ML classification.

    Contains MFCCs (13 coefficients), spectral centroid, zero-crossing rate,
    and other features computed from I2S microphone data on ESP32.
    """

    sensor_id: str = ""
    timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp()
    )

    # MFCC coefficients (13 is standard for speech/environmental audio)
    mfcc: list[float] = Field(
        default_factory=lambda: [0.0] * 13,
        description="Mel-frequency cepstral coefficients (13 values)",
    )

    # Spectral features
    spectral_centroid: float = 0.0  # Hz — center of mass of spectrum
    spectral_bandwidth: float = 0.0  # Hz — spread around centroid
    spectral_rolloff: float = 0.0  # Hz — frequency below which 85% energy
    spectral_flatness: float = 0.0  # 0.0-1.0 — tonality measure

    # Time-domain features
    zero_crossing_rate: float = 0.0  # crossings per sample
    rms_energy: float = 0.0  # root mean square energy 0.0-1.0
    peak_amplitude: float = 0.0  # peak amplitude 0.0-1.0

    # Temporal
    duration_ms: int = 0  # segment duration in milliseconds
    sample_rate_hz: int = 16000  # sample rate used for extraction


class SoundSignature(BaseModel):
    """Frequency and energy profile for a class of sounds.

    Used as training data / reference signatures for the ML classifier.
    Each sound class has a characteristic signature that the classifier
    learns to recognize.
    """

    class_name: str  # e.g. "gunshot", "voice", "vehicle"
    frequencies: list[float] = Field(
        default_factory=list,
        description="Characteristic frequency bins in Hz",
    )
    duration_range_ms: tuple[float, float] = (0.0, 10000.0)
    energy_profile: list[float] = Field(
        default_factory=list,
        description="Normalized energy envelope over time (0.0-1.0 values)",
    )
    typical_mfcc: list[float] = Field(
        default_factory=lambda: [0.0] * 13,
        description="Average MFCC profile for this sound class",
    )
    spectral_centroid_range: tuple[float, float] = (0.0, 22050.0)
    zero_crossing_range: tuple[float, float] = (0.0, 1.0)
    rms_energy_range: tuple[float, float] = (0.0, 1.0)
    description: str = ""

    def matches_features(self, features: AudioFeatureVector) -> float:
        """Compute a rough similarity score (0.0-1.0) against features.

        This is a simple heuristic matcher for when the ML model is unavailable.
        """
        score = 0.0
        checks = 0

        # Spectral centroid range check
        lo, hi = self.spectral_centroid_range
        if lo <= features.spectral_centroid <= hi:
            score += 1.0
        checks += 1

        # Zero crossing range
        lo, hi = self.zero_crossing_range
        if lo <= features.zero_crossing_rate <= hi:
            score += 1.0
        checks += 1

        # RMS energy range
        lo, hi = self.rms_energy_range
        if lo <= features.rms_energy <= hi:
            score += 1.0
        checks += 1

        # Duration range
        lo, hi = self.duration_range_ms
        if lo <= features.duration_ms <= hi:
            score += 1.0
        checks += 1

        return score / checks if checks > 0 else 0.0


class SoundClassification(BaseModel):
    """Result of ML-based sound classification.

    Produced by the acoustic intelligence pipeline on SC when it receives
    an AudioFeatureVector from an edge device.
    """

    event_type: str = "unknown"  # matches AcousticEventType values
    confidence: float = 0.0  # 0.0-1.0 classification confidence
    model_version: str = "rule_based_v1"  # model identifier
    model_type: str = "rule_based"  # "rule_based", "mfcc_knn", "mfcc_rf"

    # Top-N predictions for multi-class output
    predictions: list[dict] = Field(
        default_factory=list,
        description="Top predictions: [{class_name, confidence}, ...]",
    )

    # Source info
    sensor_id: str = ""
    timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp()
    )
    processing_time_ms: float = 0.0  # how long classification took


class AcousticObserver(BaseModel):
    """An acoustic sensor node that observed a sound event.

    Used for TDoA (Time Difference of Arrival) localization.
    """

    sensor_id: str
    lat: float
    lon: float
    arrival_time: float  # epoch seconds with microsecond precision
    amplitude_db: float = 0.0  # received amplitude at this sensor
    confidence: float = 1.0  # detection confidence at this sensor


class AcousticTrilateration(BaseModel):
    """Multi-node acoustic source localization via Time Difference of Arrival.

    When 2+ edge nodes detect the same acoustic event, the time difference
    between arrivals can be used to estimate the source position.
    Sound speed (~343 m/s at 20C) gives distance differences between
    observer pairs, constraining the source location.
    """

    event_id: str = ""
    observers: list[AcousticObserver] = Field(default_factory=list)
    arrival_times: list[float] = Field(
        default_factory=list,
        description="Arrival timestamps in epoch seconds (one per observer)",
    )

    # Estimated position
    estimated_lat: float = 0.0
    estimated_lon: float = 0.0
    confidence: float = 0.0  # 0.0-1.0

    # Metadata
    sound_speed_mps: float = 343.0  # speed of sound in m/s
    method: str = "tdoa_weighted_centroid"
    timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp()
    )

    @property
    def estimated_position(self) -> tuple[float, float]:
        return (self.estimated_lat, self.estimated_lon)


# Speed of sound in air at 20C
SPEED_OF_SOUND_MPS = 343.0


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two lat/lon points in meters."""
    R = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def acoustic_trilaterate(
    observers: list[dict],
    sound_speed: float = SPEED_OF_SOUND_MPS,
) -> Optional[dict]:
    """Estimate sound source position from multi-node TDoA.

    Uses time-difference-of-arrival with inverse-distance weighted centroid.
    The earliest observer is assumed closest to the source.

    Args:
        observers: List of dicts with keys: sensor_id, lat, lon, arrival_time,
                   and optionally amplitude_db, confidence.
        sound_speed: Speed of sound in m/s (default 343 for 20C air).

    Returns:
        Dict with estimated position, confidence, and metadata.
        None if fewer than 2 observers.
    """
    if len(observers) < 2:
        return None

    # Parse observers
    obs_list: list[AcousticObserver] = []
    for o in observers:
        obs_list.append(AcousticObserver(
            sensor_id=o.get("sensor_id", ""),
            lat=o.get("lat", 0.0),
            lon=o.get("lon", 0.0),
            arrival_time=o.get("arrival_time", 0.0),
            amplitude_db=o.get("amplitude_db", 0.0),
            confidence=o.get("confidence", 1.0),
        ))

    # Sort by arrival time — earliest = closest to source
    obs_list.sort(key=lambda x: x.arrival_time)
    t0 = obs_list[0].arrival_time

    # Compute distance estimates from TDoA relative to first observer
    # Each observer gives a maximum distance from the first observer's location
    # The actual source is closer to the first observer by (dt * sound_speed)
    anchors: list[tuple[float, float, float]] = []  # lat, lon, weight
    for obs in obs_list:
        dt = obs.arrival_time - t0
        # Distance from first observer that this TDoA implies
        distance_diff = dt * sound_speed

        if dt < 0.001:
            # This is the closest observer — high weight, small distance
            weight = 10.0 * obs.confidence
        else:
            # Weight inversely proportional to time delay
            weight = obs.confidence / max(dt, 0.001)

        anchors.append((obs.lat, obs.lon, weight))

    # Weighted centroid
    total_w = sum(w for _, _, w in anchors)
    if total_w <= 0:
        return None

    est_lat = sum(lat * w for lat, _, w in anchors) / total_w
    est_lon = sum(lon * w for _, lon, w in anchors) / total_w

    # Confidence based on observer count and geometric spread
    n = len(obs_list)
    count_score = min(1.0, 0.3 + (n - 2) * 0.25)

    lats = [o.lat for o in obs_list]
    lons = [o.lon for o in obs_list]
    spread = math.sqrt(
        (max(lats) - min(lats)) ** 2 + (max(lons) - min(lons)) ** 2
    )
    # Spread in degrees — 0.001 deg ~ 111m
    geometry_score = min(1.0, spread * 111_000 / max(sound_speed, 1.0))

    confidence = round(0.5 * count_score + 0.5 * geometry_score, 3)
    confidence = min(1.0, max(0.0, confidence))

    result = AcousticTrilateration(
        observers=obs_list,
        arrival_times=[o.arrival_time for o in obs_list],
        estimated_lat=round(est_lat, 8),
        estimated_lon=round(est_lon, 8),
        confidence=confidence,
        sound_speed_mps=sound_speed,
    )

    return result.model_dump()
