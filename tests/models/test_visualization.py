# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for 3D visualization models."""

import pytest

from tritium_lib.models.visualization import (
    AllianceColor,
    CoverageVolume,
    Scene3DConfig,
    SensorVolumeType,
    TimelineConfig,
    TrajectoryRibbon,
)


class TestTrajectoryRibbon:
    """Tests for TrajectoryRibbon model."""

    def test_defaults(self):
        r = TrajectoryRibbon(target_id="ble_aa:bb:cc")
        assert r.target_id == "ble_aa:bb:cc"
        assert r.alliance == "unknown"
        assert r.min_width == 0.1
        assert r.max_width == 0.5
        assert r.opacity == 0.6
        assert r.fade_tail is True
        assert r.visible is True

    def test_alliance_color(self):
        r = TrajectoryRibbon(target_id="t1", alliance="hostile", color="#ff2a6d")
        assert r.alliance == "hostile"
        assert r.color == "#ff2a6d"

    def test_serialization(self):
        r = TrajectoryRibbon(target_id="t1", max_points=50)
        d = r.model_dump()
        assert d["target_id"] == "t1"
        assert d["max_points"] == 50


class TestCoverageVolume:
    """Tests for CoverageVolume model."""

    def test_defaults(self):
        v = CoverageVolume(sensor_id="sensor_1")
        assert v.sensor_type == "ble"
        assert v.volume_type == SensorVolumeType.SPHERE
        assert v.range_m == 10.0
        assert v.fov_horizontal_deg == 360.0

    def test_camera_factory(self):
        v = CoverageVolume.for_camera("cam_1", 10, 20, 3, 30, 90, 60, 45, -15)
        assert v.sensor_type == "camera"
        assert v.volume_type == SensorVolumeType.CONE
        assert v.range_m == 30.0
        assert v.fov_horizontal_deg == 90.0
        assert v.heading_deg == 45.0
        assert v.tilt_deg == -15.0

    def test_ble_factory(self):
        v = CoverageVolume.for_ble("ble_1", 5, 5, 1, 15)
        assert v.sensor_type == "ble"
        assert v.volume_type == SensorVolumeType.SPHERE
        assert v.range_m == 15.0
        assert v.pulse_animation is True

    def test_wifi_factory(self):
        v = CoverageVolume.for_wifi("wifi_1", 0, 0, 2, 50)
        assert v.sensor_type == "wifi"
        assert v.range_m == 50.0

    def test_serialization(self):
        v = CoverageVolume.for_camera("cam_1", 0, 0)
        d = v.model_dump()
        assert d["sensor_id"] == "cam_1"
        assert d["volume_type"] == "cone"


class TestTimelineConfig:
    """Tests for TimelineConfig model."""

    def test_defaults(self):
        t = TimelineConfig()
        assert t.enabled is False
        assert t.playback_speed == 1.0
        assert t.duration == 0.0
        assert t.progress == 0.0

    def test_duration_and_progress(self):
        t = TimelineConfig(start_time=100, end_time=200, current_time=150)
        assert t.duration == 100.0
        assert t.progress == 0.5

    def test_progress_at_start(self):
        t = TimelineConfig(start_time=100, end_time=200, current_time=100)
        assert t.progress == 0.0

    def test_progress_at_end(self):
        t = TimelineConfig(start_time=100, end_time=200, current_time=200)
        assert t.progress == 1.0

    def test_zero_duration(self):
        t = TimelineConfig(start_time=100, end_time=100)
        assert t.duration == 0.0
        assert t.progress == 0.0


class TestScene3DConfig:
    """Tests for Scene3DConfig model."""

    def test_defaults(self):
        s = Scene3DConfig()
        assert s.ribbons == []
        assert s.coverage_volumes == []
        assert isinstance(s.timeline, TimelineConfig)
        assert s.show_grid is True
        assert s.background_color == "#0d0d1a"

    def test_with_ribbons_and_volumes(self):
        r = TrajectoryRibbon(target_id="t1")
        v = CoverageVolume.for_ble("s1", 0, 0)
        s = Scene3DConfig(
            ribbons=[r],
            coverage_volumes=[v],
            timeline=TimelineConfig(enabled=True, start_time=0, end_time=60),
        )
        assert len(s.ribbons) == 1
        assert len(s.coverage_volumes) == 1
        assert s.timeline.enabled is True
        assert s.timeline.duration == 60.0

    def test_full_serialization(self):
        s = Scene3DConfig(
            ribbons=[TrajectoryRibbon(target_id="t1")],
            coverage_volumes=[CoverageVolume.for_camera("c1", 0, 0)],
            enable_shadows=False,
        )
        d = s.model_dump()
        assert len(d["ribbons"]) == 1
        assert len(d["coverage_volumes"]) == 1
        assert d["enable_shadows"] is False


class TestAllianceColor:
    """Tests for AllianceColor enum."""

    def test_values(self):
        assert AllianceColor.FRIENDLY == "#05ffa1"
        assert AllianceColor.HOSTILE == "#ff2a6d"
        assert AllianceColor.NEUTRAL == "#00a0ff"
        assert AllianceColor.UNKNOWN == "#fcee0a"


class TestSensorVolumeType:
    """Tests for SensorVolumeType enum."""

    def test_values(self):
        assert SensorVolumeType.CONE == "cone"
        assert SensorVolumeType.SPHERE == "sphere"
        assert SensorVolumeType.CYLINDER == "cylinder"
        assert SensorVolumeType.FRUSTUM == "frustum"
