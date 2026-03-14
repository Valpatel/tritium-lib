# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Acoustic event classification models for environmental sound detection.

These models represent classified audio events from edge microphones or
SC audio processing pipelines. Events feed into TargetTracker as acoustic
detections (e.g., gunshot at location, vehicle engine near sensor).

Distinct from acoustic_modem.py which handles data-over-audio communication.

MQTT topics:
    tritium/{site}/acoustic/{sensor_id}/event      — classified audio events
    tritium/{site}/acoustic/{sensor_id}/spectrum    — FFT spectrum snapshots
    tritium/{site}/acoustic/alerts                  — high-priority acoustic alerts
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AcousticEventType(str, Enum):
    """Classification of detected audio events."""

    GUNSHOT = "gunshot"
    EXPLOSION = "explosion"
    GLASS_BREAK = "glass_break"
    VOICE = "voice"
    SCREAM = "scream"
    SIREN = "siren"
    ALARM = "alarm"
    VEHICLE_ENGINE = "vehicle_engine"
    VEHICLE_HORN = "vehicle_horn"
    VEHICLE_CRASH = "vehicle_crash"
    DOG_BARK = "dog_bark"
    FOOTSTEPS = "footsteps"
    DOOR = "door"
    MACHINERY = "machinery"
    MUSIC = "music"
    SILENCE = "silence"
    AMBIENT = "ambient"
    UNKNOWN = "unknown"


class AcousticSeverity(str, Enum):
    """Severity level of an acoustic event."""

    CRITICAL = "critical"  # gunshot, explosion
    HIGH = "high"  # glass_break, scream, crash
    MEDIUM = "medium"  # siren, alarm, vehicle_horn
    LOW = "low"  # voice, footsteps, door
    INFO = "info"  # ambient, music, silence


class AcousticEvent(BaseModel):
    """A classified acoustic event from a sensor.

    Produced by the acoustic classification pipeline (edge HAL or SC plugin).
    """

    event_id: str = ""
    event_type: AcousticEventType = AcousticEventType.UNKNOWN
    severity: AcousticSeverity = AcousticSeverity.INFO
    confidence: float = 0.0  # classification confidence 0.0-1.0

    # Audio characteristics
    peak_frequency_hz: float = 0.0  # dominant frequency
    peak_amplitude_db: float = 0.0  # peak amplitude in dBFS
    duration_ms: float = 0.0  # event duration in milliseconds
    bandwidth_hz: float = 0.0  # frequency bandwidth of the event

    # Source sensor
    sensor_id: str = ""
    site_id: str = ""
    latitude: float = 0.0
    longitude: float = 0.0

    # Timing
    timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp()
    )
    detection_latency_ms: float = 0.0  # time from sound to classification

    # Optional audio context
    ambient_noise_db: float = 0.0  # background noise level
    snr_db: float = 0.0  # signal-to-noise ratio

    def to_target_dict(self) -> dict:
        """Convert to dict for TargetTracker ingestion."""
        return {
            "target_id": f"acoustic_{self.sensor_id}_{self.event_type.value}",
            "name": f"{self.event_type.value} @ {self.sensor_id}",
            "source": "acoustic",
            "asset_type": self.event_type.value,
            "alliance": "unknown",
            "position": {
                "lat": self.latitude,
                "lng": self.longitude,
            },
            "classification": self.event_type.value,
            "metadata": {
                "confidence": self.confidence,
                "severity": self.severity.value,
                "peak_frequency_hz": self.peak_frequency_hz,
                "peak_amplitude_db": self.peak_amplitude_db,
                "duration_ms": self.duration_ms,
                "sensor_id": self.sensor_id,
            },
        }


class AcousticSpectrum(BaseModel):
    """FFT spectrum snapshot from a sensor for visualization."""

    sensor_id: str
    timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp()
    )
    frequencies: list[float] = Field(default_factory=list)  # Hz bins
    magnitudes: list[float] = Field(default_factory=list)  # dB values
    sample_rate_hz: int = 44100
    fft_size: int = 1024
    window: str = "hann"


class AcousticSensorConfig(BaseModel):
    """Configuration for an acoustic classification sensor."""

    sensor_id: str
    site_id: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    enabled: bool = True
    sample_rate_hz: int = 16000
    fft_size: int = 1024
    detection_threshold_db: float = -40.0  # minimum amplitude to trigger
    min_confidence: float = 0.5  # minimum classification confidence
    report_ambient: bool = False  # whether to report ambient/silence events
    classification_model: str = "rule_based"  # or "ml_yamnet", "ml_custom"


class AcousticStats(BaseModel):
    """Statistics for an acoustic sensor or site."""

    total_events: int = 0
    events_by_type: dict[str, int] = Field(default_factory=dict)
    events_by_severity: dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0
    avg_ambient_db: float = 0.0
    sensors_active: int = 0
    last_event_time: float = 0.0


def classify_event_severity(event_type: AcousticEventType) -> AcousticSeverity:
    """Map an acoustic event type to its default severity level."""
    severity_map = {
        AcousticEventType.GUNSHOT: AcousticSeverity.CRITICAL,
        AcousticEventType.EXPLOSION: AcousticSeverity.CRITICAL,
        AcousticEventType.GLASS_BREAK: AcousticSeverity.HIGH,
        AcousticEventType.SCREAM: AcousticSeverity.HIGH,
        AcousticEventType.VEHICLE_CRASH: AcousticSeverity.HIGH,
        AcousticEventType.SIREN: AcousticSeverity.MEDIUM,
        AcousticEventType.ALARM: AcousticSeverity.MEDIUM,
        AcousticEventType.VEHICLE_HORN: AcousticSeverity.MEDIUM,
        AcousticEventType.VOICE: AcousticSeverity.LOW,
        AcousticEventType.FOOTSTEPS: AcousticSeverity.LOW,
        AcousticEventType.DOOR: AcousticSeverity.LOW,
        AcousticEventType.DOG_BARK: AcousticSeverity.LOW,
        AcousticEventType.VEHICLE_ENGINE: AcousticSeverity.LOW,
        AcousticEventType.MACHINERY: AcousticSeverity.LOW,
        AcousticEventType.MUSIC: AcousticSeverity.INFO,
        AcousticEventType.AMBIENT: AcousticSeverity.INFO,
        AcousticEventType.SILENCE: AcousticSeverity.INFO,
        AcousticEventType.UNKNOWN: AcousticSeverity.INFO,
    }
    return severity_map.get(event_type, AcousticSeverity.INFO)
