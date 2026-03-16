# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Radar integration models for track data and configuration.

Supports ground-based and airborne radar systems that feed tracks
into the unified operating picture via MQTT or REST.

MQTT topics:
    tritium/{site}/radar/{radar_id}/scan   — complete scan with tracks
    tritium/{site}/radar/{radar_id}/track  — individual track updates
    tritium/{site}/radar/{radar_id}/config — radar configuration
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RadarMode(str, Enum):
    """Radar operating mode."""

    SURVEILLANCE = "surveillance"
    TRACKING = "tracking"
    WEATHER = "weather"
    GROUND_MAP = "ground_map"
    SAR = "sar"  # synthetic aperture radar


class RadarClassification(str, Enum):
    """Radar target classification."""

    UNKNOWN = "unknown"
    PERSON = "person"
    VEHICLE = "vehicle"
    AIRCRAFT = "aircraft"
    ROTORCRAFT = "rotorcraft"
    UAV = "uav"
    SHIP = "ship"
    ANIMAL = "animal"
    CLUTTER = "clutter"
    WEATHER = "weather"


class RadarTrack(BaseModel):
    """A single radar track (detected object).

    Represents a target detected by the radar with range, bearing,
    and optional velocity/classification data.
    """

    track_id: str
    range_m: float  # distance in meters
    azimuth_deg: float  # bearing from radar, 0-360 degrees true north
    elevation_deg: float = 0.0  # elevation angle in degrees
    velocity_mps: float = 0.0  # radial velocity in m/s (positive = away)
    rcs_dbsm: float = 0.0  # radar cross section in dBsm
    classification: RadarClassification = RadarClassification.UNKNOWN
    confidence: float = 1.0  # 0.0-1.0
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    source_id: str = ""  # radar that produced this track

    def to_target_dict(self) -> dict:
        """Convert to a dict suitable for TargetTracker ingestion."""
        return {
            "target_id": f"radar_{self.source_id}_{self.track_id}",
            "source": "radar",
            "classification": self.classification.value,
            "alliance": "unknown",
            "confidence": self.confidence,
            "metadata": {
                "range_m": self.range_m,
                "azimuth_deg": self.azimuth_deg,
                "elevation_deg": self.elevation_deg,
                "velocity_mps": self.velocity_mps,
                "rcs_dbsm": self.rcs_dbsm,
                "track_id": self.track_id,
            },
        }


class RadarScan(BaseModel):
    """A complete radar scan containing multiple tracks.

    One scan represents a full antenna rotation or dwell period.
    """

    scan_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    tracks: list[RadarTrack] = Field(default_factory=list)
    mode: RadarMode = RadarMode.SURVEILLANCE
    rotation_rate_rpm: float = 0.0  # antenna rotation rate


class RadarConfig(BaseModel):
    """Radar system configuration and location.

    Describes a radar sensor's capabilities and physical position
    for integration into the sensor fusion pipeline.
    """

    radar_id: str
    name: str = ""
    frequency_ghz: float = 0.0  # operating frequency
    max_range_m: float = 0.0  # maximum detection range
    beam_width_deg: float = 0.0  # 3dB beamwidth
    latitude: float = 0.0
    longitude: float = 0.0
    altitude_m: float = 0.0
    enabled: bool = True
