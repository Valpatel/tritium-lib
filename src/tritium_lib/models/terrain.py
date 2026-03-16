# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Terrain analysis and geospatial intelligence models.

Provides elevation data, line-of-sight calculations, RF propagation
modeling, and coverage analysis for sensor placement optimization.

MQTT topics:
    tritium/{site}/terrain/elevation — elevation query results
    tritium/{site}/terrain/coverage  — sensor coverage analysis results
"""

import math
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TerrainType(str, Enum):
    """Terrain classification for RF propagation modeling."""

    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"
    FOREST = "forest"
    WATER = "water"
    DESERT = "desert"
    MOUNTAIN = "mountain"
    INDOOR = "indoor"
    UNKNOWN = "unknown"


class ElevationPoint(BaseModel):
    """A single elevation point."""

    latitude: float
    longitude: float
    elevation_m: float = 0.0  # meters above sea level
    terrain_type: TerrainType = TerrainType.UNKNOWN
    source: str = "srtm"  # srtm, lidar, manual


class ElevationProfile(BaseModel):
    """Elevation profile between two points for line-of-sight analysis."""

    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    points: list[ElevationPoint] = Field(default_factory=list)
    distance_m: float = 0.0
    min_elevation_m: float = 0.0
    max_elevation_m: float = 0.0
    has_line_of_sight: bool = True
    obstruction_points: list[int] = Field(default_factory=list)  # indices of obstructing points


class CoverageCell(BaseModel):
    """A single cell in a coverage analysis grid."""

    latitude: float
    longitude: float
    signal_strength_dbm: float = -100.0
    covered: bool = False
    distance_m: float = 0.0
    elevation_m: float = 0.0
    has_los: bool = True  # line of sight to sensor


class CoverageAnalysis(BaseModel):
    """Coverage analysis result for a sensor placement."""

    sensor_id: str
    sensor_lat: float
    sensor_lng: float
    sensor_height_m: float = 2.0  # height above ground
    frequency_mhz: float = 2400.0  # operating frequency
    tx_power_dbm: float = 0.0  # transmit power
    terrain_type: TerrainType = TerrainType.SUBURBAN
    range_m: float = 100.0
    cells: list[CoverageCell] = Field(default_factory=list)
    coverage_percent: float = 0.0
    grid_resolution_m: float = 10.0


class SensorPlacement(BaseModel):
    """Recommended sensor placement with coverage score."""

    latitude: float
    longitude: float
    height_m: float = 2.0
    score: float = 0.0  # 0.0-1.0 coverage quality
    coverage_area_m2: float = 0.0
    overlapping_sensors: list[str] = Field(default_factory=list)
    terrain_type: TerrainType = TerrainType.UNKNOWN


class WeatherConditions(BaseModel):
    """Weather conditions affecting RF propagation."""

    temperature_c: float = 20.0
    humidity_percent: float = 50.0
    rain_rate_mm_h: float = 0.0
    wind_speed_ms: float = 0.0
    visibility_km: float = 10.0
    atmospheric_pressure_hpa: float = 1013.25

    @property
    def rain_attenuation_db_km(self) -> float:
        """Estimate rain attenuation in dB/km at 2.4 GHz.

        Based on ITU-R P.838 simplified model.
        """
        if self.rain_rate_mm_h <= 0:
            return 0.0
        # Simplified: attenuation ~ k * R^alpha
        # At 2.4 GHz: k ~ 0.0001, alpha ~ 1.0
        return 0.0001 * self.rain_rate_mm_h


# --- Free-space path loss calculation ---

def free_space_path_loss_db(distance_m: float, frequency_mhz: float) -> float:
    """Calculate free-space path loss (FSPL) in dB.

    FSPL(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
    where d is in meters and f is in Hz.
    """
    if distance_m <= 0 or frequency_mhz <= 0:
        return 0.0
    freq_hz = frequency_mhz * 1e6
    c = 299792458.0  # speed of light m/s
    fspl = 20 * math.log10(distance_m) + 20 * math.log10(freq_hz) + 20 * math.log10(4 * math.pi / c)
    return fspl


def terrain_path_loss_db(
    distance_m: float,
    frequency_mhz: float,
    terrain: TerrainType,
) -> float:
    """Estimate total path loss including terrain effects.

    Adds terrain-specific loss factor on top of free-space path loss.
    """
    fspl = free_space_path_loss_db(distance_m, frequency_mhz)

    # Terrain loss factors (empirical, dB per doubling of distance)
    terrain_factor = {
        TerrainType.URBAN: 30.0,
        TerrainType.SUBURBAN: 20.0,
        TerrainType.RURAL: 10.0,
        TerrainType.FOREST: 25.0,
        TerrainType.WATER: 5.0,
        TerrainType.DESERT: 8.0,
        TerrainType.MOUNTAIN: 15.0,
        TerrainType.INDOOR: 35.0,
        TerrainType.UNKNOWN: 20.0,
    }

    extra_loss = terrain_factor.get(terrain, 20.0)
    if distance_m > 1:
        extra_loss *= math.log2(max(distance_m, 2)) / 10.0

    return fspl + extra_loss


def estimate_signal_strength(
    tx_power_dbm: float,
    distance_m: float,
    frequency_mhz: float = 2400.0,
    terrain: TerrainType = TerrainType.SUBURBAN,
) -> float:
    """Estimate received signal strength at a given distance.

    Returns estimated RSSI in dBm.
    """
    if distance_m <= 0:
        return tx_power_dbm
    loss = terrain_path_loss_db(distance_m, frequency_mhz, terrain)
    return tx_power_dbm - loss
