# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Sensor placement and configuration models.

Standardizes how sensors (cameras, BLE radios, WiFi APs, microphones,
etc.) are described across edge firmware and the command center.

MQTT topics:
    tritium/{site}/sensors/{sensor_id}/config — sensor placement config
    tritium/{site}/sensors/{sensor_id}/status — sensor operational status
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SensorType(str, Enum):
    """Physical sensor type classification."""

    BLE_RADIO = "ble_radio"
    WIFI_RADIO = "wifi_radio"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    RADAR = "radar"
    LIDAR = "lidar"
    PIR = "pir"
    ACOUSTIC = "acoustic"
    RF_MONITOR = "rf_monitor"
    MESH_RADIO = "mesh_radio"
    IMU = "imu"
    GPS = "gps"
    ENVIRONMENTAL = "environmental"  # temp, humidity, pressure
    UNKNOWN = "unknown"


class MountingType(str, Enum):
    """How the sensor is physically mounted."""

    WALL = "wall"
    CEILING = "ceiling"
    POLE = "pole"
    TRIPOD = "tripod"
    VEHICLE = "vehicle"
    DRONE = "drone"
    HANDHELD = "handheld"
    DESK = "desk"
    GROUND = "ground"
    EMBEDDED = "embedded"  # built into another device
    UNKNOWN = "unknown"


class SensorStatus(str, Enum):
    """Operational status of a sensor."""

    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    CALIBRATING = "calibrating"
    ERROR = "error"
    UNKNOWN = "unknown"


class SensorPosition(BaseModel):
    """Geographic position of a sensor."""

    latitude: float = 0.0
    longitude: float = 0.0
    altitude_m: float = 0.0  # meters above sea level
    x: float = 0.0  # local coordinate (meters from origin)
    y: float = 0.0  # local coordinate (meters from origin)
    z: float = 0.0  # local height above ground (meters)


class SensorPlacement(BaseModel):
    """Full sensor placement and configuration descriptor.

    Used by both edge devices (to report their sensor capabilities) and
    the command center (to reason about coverage, triangulation, etc.).
    """

    sensor_id: str = Field(..., description="Unique sensor identifier")
    device_id: Optional[str] = Field(
        None, description="Parent device hosting this sensor"
    )
    sensor_type: SensorType = SensorType.UNKNOWN
    position: SensorPosition = Field(default_factory=SensorPosition)
    height_m: float = Field(
        2.0, ge=0.0, description="Height above ground in meters"
    )
    fov_degrees: float = Field(
        360.0,
        ge=0.0,
        le=360.0,
        description="Field of view in degrees (360 = omnidirectional)",
    )
    rotation_degrees: float = Field(
        0.0,
        ge=0.0,
        lt=360.0,
        description="Rotation/heading in degrees from north (clockwise)",
    )
    tilt_degrees: float = Field(
        0.0,
        ge=-90.0,
        le=90.0,
        description="Vertical tilt in degrees (0 = horizontal, -90 = down)",
    )
    coverage_radius_m: float = Field(
        50.0, ge=0.0, description="Effective detection range in meters"
    )
    mounting_type: MountingType = MountingType.UNKNOWN
    status: SensorStatus = SensorStatus.UNKNOWN

    # Operational parameters
    frequency_mhz: float = Field(
        0.0, ge=0.0, description="Operating frequency (0 = N/A)"
    )
    tx_power_dbm: float = Field(
        0.0, description="Transmit power in dBm (0 = passive)"
    )
    sensitivity_dbm: float = Field(
        -90.0, description="Receive sensitivity in dBm"
    )
    sample_rate_hz: float = Field(
        0.0, ge=0.0, description="Sampling rate in Hz (0 = event-driven)"
    )

    # Metadata
    label: str = Field("", description="Human-readable label")
    firmware_version: str = ""
    model: str = ""
    manufacturer: str = ""
    notes: str = ""

    @property
    def is_omnidirectional(self) -> bool:
        """True if sensor has 360-degree field of view."""
        return self.fov_degrees >= 360.0

    @property
    def is_directional(self) -> bool:
        """True if sensor has a limited field of view."""
        return self.fov_degrees < 360.0

    @property
    def is_passive(self) -> bool:
        """True if sensor only receives (no transmission)."""
        return self.tx_power_dbm <= 0.0

    def coverage_area_m2(self) -> float:
        """Estimate coverage area in square meters.

        For omnidirectional sensors: full circle.
        For directional sensors: sector of circle.
        """
        import math

        r = self.coverage_radius_m
        if self.is_omnidirectional:
            return math.pi * r * r
        fraction = self.fov_degrees / 360.0
        return math.pi * r * r * fraction

    def contains_bearing(self, bearing_deg: float) -> bool:
        """Check if a bearing (degrees from north) falls within the sensor FOV.

        Args:
            bearing_deg: Bearing in degrees from north (0-360).

        Returns:
            True if the bearing is within the sensor's field of view.
        """
        if self.is_omnidirectional:
            return True

        half_fov = self.fov_degrees / 2.0
        diff = (bearing_deg - self.rotation_degrees + 360.0) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff
        return diff <= half_fov


class SensorArray(BaseModel):
    """A collection of sensors forming a detection array.

    Used for multi-sensor setups like trilateration arrays,
    camera grids, or distributed microphone arrays.
    """

    array_id: str = Field(..., description="Array identifier")
    sensors: list[SensorPlacement] = Field(default_factory=list)
    label: str = ""
    purpose: str = ""  # "trilateration", "coverage", "surveillance", etc.

    @property
    def sensor_count(self) -> int:
        return len(self.sensors)

    def sensor_ids(self) -> list[str]:
        return [s.sensor_id for s in self.sensors]

    def by_type(self, sensor_type: SensorType) -> list[SensorPlacement]:
        """Filter sensors by type."""
        return [s for s in self.sensors if s.sensor_type == sensor_type]

    def online_sensors(self) -> list[SensorPlacement]:
        """Get sensors that are currently online."""
        return [s for s in self.sensors if s.status == SensorStatus.ONLINE]

    def total_coverage_area_m2(self) -> float:
        """Sum of individual sensor coverage areas (ignoring overlap)."""
        return sum(s.coverage_area_m2() for s in self.sensors)
