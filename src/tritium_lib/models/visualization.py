# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""3D visualization models for standardizing scene rendering parameters.

Provides configuration models for trajectory ribbons, sensor coverage volumes,
timeline scrubbers, and overall 3D scene settings. Used by tritium-sc frontend
(war3d.js, map3d.js) and any future AR/VR visualization tools.

MQTT topics:
    tritium/{site}/visualization/config — scene configuration updates
    tritium/{site}/visualization/timeline — timeline state broadcasts
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AllianceColor(str, Enum):
    """Standard alliance color mapping for 3D rendering."""

    FRIENDLY = "#05ffa1"
    HOSTILE = "#ff2a6d"
    NEUTRAL = "#00a0ff"
    UNKNOWN = "#fcee0a"


class SensorVolumeType(str, Enum):
    """Type of sensor coverage volume geometry."""

    CONE = "cone"        # Cameras with directional FOV
    SPHERE = "sphere"    # BLE/WiFi omnidirectional
    CYLINDER = "cylinder"  # Acoustic vertical detection
    FRUSTUM = "frustum"  # Cameras with near/far planes


class TrajectoryRibbon(BaseModel):
    """Configuration for a 3D target movement trail rendered as a ribbon.

    Width encodes confidence, color encodes alliance, and the ribbon
    extends through 3D space-time (x, y mapped to ground, z/height
    mapped to time progression).
    """

    target_id: str
    alliance: str = "unknown"  # friendly, hostile, neutral, unknown
    color: str = "#fcee0a"  # Override color (defaults to alliance color)
    min_width: float = Field(default=0.1, ge=0.01, le=5.0, description="Min ribbon width (low confidence)")
    max_width: float = Field(default=0.5, ge=0.01, le=5.0, description="Max ribbon width (high confidence)")
    opacity: float = Field(default=0.6, ge=0.0, le=1.0)
    fade_tail: bool = True  # Fade opacity toward oldest points
    time_height_scale: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description="How much to elevate ribbon per second of history (0=flat on ground)",
    )
    max_points: int = Field(default=200, ge=2, le=2000, description="Max trail points to render")
    glow_intensity: float = Field(default=0.15, ge=0.0, le=1.0, description="Emissive glow strength")
    visible: bool = True


class CoverageVolume(BaseModel):
    """Configuration for a sensor coverage volume rendered in 3D.

    Renders translucent geometric shapes showing actual detection range
    and field of view for each sensor type.
    """

    sensor_id: str
    sensor_type: str = "ble"  # ble, wifi, camera, acoustic, rf
    volume_type: SensorVolumeType = SensorVolumeType.SPHERE
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0  # Height above ground
    range_m: float = Field(default=10.0, ge=0.1, le=10000.0, description="Detection range in meters")
    fov_horizontal_deg: float = Field(default=360.0, ge=1.0, le=360.0, description="Horizontal FOV degrees")
    fov_vertical_deg: float = Field(default=180.0, ge=1.0, le=180.0, description="Vertical FOV degrees")
    heading_deg: float = Field(default=0.0, ge=0.0, le=360.0, description="Sensor pointing direction")
    tilt_deg: float = Field(default=0.0, ge=-90.0, le=90.0, description="Sensor tilt angle")
    color: str = "#00f0ff"
    opacity: float = Field(default=0.08, ge=0.0, le=1.0)
    wireframe: bool = True  # Show wireframe edges
    wireframe_opacity: float = Field(default=0.2, ge=0.0, le=1.0)
    pulse_animation: bool = False  # Animate detection pulses
    visible: bool = True

    @staticmethod
    def for_camera(
        sensor_id: str,
        x: float, y: float, z: float = 3.0,
        range_m: float = 30.0,
        fov_h: float = 90.0, fov_v: float = 60.0,
        heading: float = 0.0, tilt: float = -15.0,
    ) -> "CoverageVolume":
        """Factory for camera-type sensor coverage (cone/frustum)."""
        return CoverageVolume(
            sensor_id=sensor_id,
            sensor_type="camera",
            volume_type=SensorVolumeType.CONE,
            position_x=x, position_y=y, position_z=z,
            range_m=range_m,
            fov_horizontal_deg=fov_h,
            fov_vertical_deg=fov_v,
            heading_deg=heading,
            tilt_deg=tilt,
            color="#ff2a6d",
            opacity=0.06,
        )

    @staticmethod
    def for_ble(
        sensor_id: str,
        x: float, y: float, z: float = 1.0,
        range_m: float = 15.0,
    ) -> "CoverageVolume":
        """Factory for BLE sensor coverage (sphere)."""
        return CoverageVolume(
            sensor_id=sensor_id,
            sensor_type="ble",
            volume_type=SensorVolumeType.SPHERE,
            position_x=x, position_y=y, position_z=z,
            range_m=range_m,
            fov_horizontal_deg=360.0,
            fov_vertical_deg=180.0,
            color="#05ffa1",
            opacity=0.05,
            pulse_animation=True,
        )

    @staticmethod
    def for_wifi(
        sensor_id: str,
        x: float, y: float, z: float = 2.0,
        range_m: float = 50.0,
    ) -> "CoverageVolume":
        """Factory for WiFi sensor coverage (sphere)."""
        return CoverageVolume(
            sensor_id=sensor_id,
            sensor_type="wifi",
            volume_type=SensorVolumeType.SPHERE,
            position_x=x, position_y=y, position_z=z,
            range_m=range_m,
            fov_horizontal_deg=360.0,
            fov_vertical_deg=180.0,
            color="#00f0ff",
            opacity=0.04,
        )


class TimelineConfig(BaseModel):
    """Configuration for the 3D timeline scrubber.

    Controls temporal playback in the 3D view, allowing users to
    scrub through time and see target positions animate.
    """

    enabled: bool = False
    start_time: float = 0.0  # Unix timestamp or monotonic
    end_time: float = 0.0
    current_time: float = 0.0
    playback_speed: float = Field(default=1.0, ge=0.1, le=100.0, description="Playback speed multiplier")
    playing: bool = False
    loop: bool = False
    show_trails: bool = True  # Show ribbons during playback
    trail_duration_s: float = Field(
        default=30.0, ge=1.0, le=3600.0,
        description="How many seconds of trail to show behind current time",
    )
    step_size_s: float = Field(default=1.0, ge=0.1, le=60.0, description="Time step per scrub increment")
    show_timestamps: bool = True
    show_speed_indicator: bool = True

    @property
    def duration(self) -> float:
        """Total timeline duration in seconds."""
        return max(0.0, self.end_time - self.start_time)

    @property
    def progress(self) -> float:
        """Current position as 0.0-1.0 fraction."""
        d = self.duration
        if d <= 0:
            return 0.0
        return max(0.0, min(1.0, (self.current_time - self.start_time) / d))


class Scene3DConfig(BaseModel):
    """Top-level 3D scene configuration aggregating all visualization settings.

    Sent to the frontend to configure the Three.js renderer with
    trajectory ribbons, coverage volumes, and timeline state.
    """

    ribbons: list[TrajectoryRibbon] = Field(default_factory=list)
    coverage_volumes: list[CoverageVolume] = Field(default_factory=list)
    timeline: TimelineConfig = Field(default_factory=TimelineConfig)

    # Scene-level settings
    show_grid: bool = True
    grid_opacity: float = Field(default=0.15, ge=0.0, le=1.0)
    show_buildings: bool = True
    building_opacity: float = Field(default=0.6, ge=0.0, le=1.0)
    fog_near: float = 80.0
    fog_far: float = 120.0
    ambient_light_intensity: float = Field(default=0.35, ge=0.0, le=2.0)
    directional_light_intensity: float = Field(default=0.65, ge=0.0, le=2.0)
    background_color: str = "#0d0d1a"
    enable_shadows: bool = True
    max_fps: int = Field(default=60, ge=10, le=120)
