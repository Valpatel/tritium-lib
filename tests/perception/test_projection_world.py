# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for perception.projection world placement — camera_xyz -> map.

Every expectation is closed-form: a camera at a known pose, a point at a
known camera-frame offset, and the ENU / lat-lng position it MUST land at.
Lat/lng expectations come from tritium_lib.geo (destination_point /
haversine_distance) so the test and the implementation share one geodesy.
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.geo import destination_point, haversine_distance
from tritium_lib.models.camera import BoundingBox, CameraDetection
from tritium_lib.perception.projection import (
    CameraWorldPose,
    place_detections_on_map,
    world_from_camera_xyz,
)

LAT0, LNG0 = 30.2672, -97.7431  # Austin, TX — arbitrary known anchor


# --------------------------------------------------- world_from_camera_xyz

class TestWorldFromCameraXyz:
    def test_point_ahead_of_north_facing_camera_lands_north(self):
        pose = CameraWorldPose(lat=LAT0, lng=LNG0, heading_deg=0.0, height_m=2.0)
        w = world_from_camera_xyz((0.0, 0.0, 4.0), pose)
        assert w is not None
        assert w["east"] == pytest.approx(0.0, abs=1e-9)
        assert w["north"] == pytest.approx(4.0, abs=1e-9)
        assert w["up"] == pytest.approx(2.0, abs=1e-9)
        # Lat/lng: 4 m due north of the camera, per the shared geodesy.
        exp_lat, exp_lng = destination_point(LAT0, LNG0, 0.0, 4.0)
        assert w["lat"] == pytest.approx(exp_lat, abs=1e-9)
        assert w["lng"] == pytest.approx(exp_lng, abs=1e-9)
        assert haversine_distance(LAT0, LNG0, w["lat"], w["lng"]) == pytest.approx(
            4.0, abs=0.01
        )
        assert w["lat"] > LAT0  # north = latitude increases

    def test_heading_rotation_east(self):
        pose = CameraWorldPose(lat=LAT0, lng=LNG0, heading_deg=90.0)
        w = world_from_camera_xyz((0.0, 0.0, 4.0), pose)
        assert w["east"] == pytest.approx(4.0, abs=1e-9)
        assert w["north"] == pytest.approx(0.0, abs=1e-9)
        exp_lat, exp_lng = destination_point(LAT0, LNG0, 90.0, 4.0)
        assert w["lat"] == pytest.approx(exp_lat, abs=1e-9)
        assert w["lng"] == pytest.approx(exp_lng, abs=1e-9)
        assert w["lng"] > LNG0  # east = longitude increases

    def test_heading_rotation_south_and_right_offset(self):
        # Facing south, a point 3 m to the camera's RIGHT is 3 m WEST.
        pose = CameraWorldPose(lat=LAT0, lng=LNG0, heading_deg=180.0)
        w = world_from_camera_xyz((3.0, 0.0, 4.0), pose)
        assert w["north"] == pytest.approx(-4.0, abs=1e-9)
        assert w["east"] == pytest.approx(-3.0, abs=1e-9)

    def test_pitch_down_shortens_ground_range_and_drops_height(self):
        # Camera 2 m up, pitched 30 deg DOWN, point 4 m along the optical
        # axis: ground range = 4*cos30, height drop = 4*sin30 = 2 -> up = 0
        # (the point is ON the ground — exactly the flat-ground handoff).
        pose = CameraWorldPose(
            lat=LAT0, lng=LNG0, heading_deg=0.0, pitch_deg=-30.0, height_m=2.0
        )
        w = world_from_camera_xyz((0.0, 0.0, 4.0), pose)
        assert w["north"] == pytest.approx(4.0 * math.cos(math.radians(30.0)), abs=1e-9)
        assert w["east"] == pytest.approx(0.0, abs=1e-9)
        assert w["up"] == pytest.approx(0.0, abs=1e-9)
        exp_lat, exp_lng = destination_point(
            LAT0, LNG0, 0.0, 4.0 * math.cos(math.radians(30.0))
        )
        assert w["lat"] == pytest.approx(exp_lat, abs=1e-9)
        assert w["lng"] == pytest.approx(exp_lng, abs=1e-9)

    def test_pitch_up_raises_point(self):
        pose = CameraWorldPose(heading_deg=0.0, pitch_deg=30.0, height_m=1.0)
        w = world_from_camera_xyz((0.0, 0.0, 4.0), pose)
        assert w["up"] == pytest.approx(1.0 + 4.0 * math.sin(math.radians(30.0)), abs=1e-9)

    def test_optical_y_down_maps_to_world_down(self):
        # +y in the optical frame is DOWN: a point 1 m "below" the optical
        # centre loses 1 m of height, at the same ground spot.
        pose = CameraWorldPose(heading_deg=0.0, height_m=3.0)
        w = world_from_camera_xyz((0.0, 1.0, 4.0), pose)
        assert w["north"] == pytest.approx(4.0, abs=1e-9)
        assert w["up"] == pytest.approx(2.0, abs=1e-9)

    def test_roll_rotates_right_into_down(self):
        # Roll +90 (right side down): the camera's +x (right) axis now
        # points at the ground.
        pose = CameraWorldPose(heading_deg=0.0, roll_deg=90.0, height_m=5.0)
        w = world_from_camera_xyz((1.0, 0.0, 4.0), pose)
        assert w["north"] == pytest.approx(4.0, abs=1e-9)
        assert w["east"] == pytest.approx(0.0, abs=1e-9)
        assert w["up"] == pytest.approx(4.0, abs=1e-9)  # 5 - 1

    def test_no_geo_pose_yields_enu_only(self):
        pose = CameraWorldPose(heading_deg=45.0, height_m=2.0)
        w = world_from_camera_xyz((0.0, 0.0, math.sqrt(2.0)), pose)
        assert w is not None
        assert w["east"] == pytest.approx(1.0, abs=1e-9)
        assert w["north"] == pytest.approx(1.0, abs=1e-9)
        assert w["lat"] is None and w["lng"] is None

    def test_point_at_camera_maps_to_camera_latlng(self):
        pose = CameraWorldPose(lat=LAT0, lng=LNG0, height_m=2.0)
        w = world_from_camera_xyz((0.0, 0.0, 0.0), pose)
        assert w["lat"] == pytest.approx(LAT0)
        assert w["lng"] == pytest.approx(LNG0)

    @pytest.mark.parametrize(
        "bad", [None, (), (1.0,), (1.0, 2.0), (float("nan"), 0.0, 1.0),
                (0.0, 0.0, float("inf")), "not-a-point"]
    )
    def test_unusable_point_returns_none(self, bad):
        assert world_from_camera_xyz(bad, CameraWorldPose()) is None


# ------------------------------------------------- place_detections_on_map

class TestPlaceDetectionsOnMap:
    def test_dict_detections_annotated(self):
        pose = CameraWorldPose(lat=LAT0, lng=LNG0, heading_deg=0.0, height_m=2.0)
        dets = [
            {"class_name": "person", "camera_xyz": (0.0, 0.0, 4.0)},
            {"class_name": "no-depth"},  # graceful passthrough
        ]
        out = place_detections_on_map(dets, pose)
        assert out is dets
        e, n, u = out[0]["world_enu"]
        assert n == pytest.approx(4.0, abs=1e-9)
        assert u == pytest.approx(2.0, abs=1e-9)
        exp_lat, exp_lng = destination_point(LAT0, LNG0, 0.0, 4.0)
        assert out[0]["world_lat"] == pytest.approx(exp_lat, abs=1e-9)
        assert out[0]["world_lng"] == pytest.approx(exp_lng, abs=1e-9)
        assert "world_enu" not in out[1]
        assert "world_lat" not in out[1]

    def test_model_detections_annotated_in_place(self):
        pose = CameraWorldPose(lat=LAT0, lng=LNG0, heading_deg=90.0)
        det = CameraDetection(
            source_id="cam-1",
            class_name="person",
            bbox=BoundingBox(x=10, y=10, w=20, h=40),
            camera_xyz=(0.0, 0.0, 10.0),
        )
        place_detections_on_map([det], pose)
        assert det.world_enu is not None
        assert det.world_enu[0] == pytest.approx(10.0, abs=1e-9)  # east
        assert det.world_enu[1] == pytest.approx(0.0, abs=1e-9)
        exp_lat, exp_lng = destination_point(LAT0, LNG0, 90.0, 10.0)
        assert det.world_lat == pytest.approx(exp_lat, abs=1e-9)
        assert det.world_lng == pytest.approx(exp_lng, abs=1e-9)

    def test_model_without_depth_passthrough(self):
        det = CameraDetection(source_id="cam-1", class_name="car")
        place_detections_on_map([det], CameraWorldPose(lat=LAT0, lng=LNG0))
        assert det.world_enu is None
        assert det.world_lat is None and det.world_lng is None

    def test_no_geo_pose_sets_enu_but_not_latlng(self):
        dets = [{"camera_xyz": (0.0, 0.0, 4.0)}]
        place_detections_on_map(dets, CameraWorldPose(heading_deg=0.0))
        assert dets[0]["world_enu"][1] == pytest.approx(4.0, abs=1e-9)
        assert "world_lat" not in dets[0]

    def test_empty_list_passthrough(self):
        assert place_detections_on_map([], CameraWorldPose()) == []


# --------------------------------------------------- end-to-end depth chain

class TestDepthToMapChain:
    def test_deproject_then_place(self):
        """Full chain: pixel + depth -> camera_xyz -> world lat/lng."""
        from tritium_lib.perception.depth import CameraIntrinsics, deproject_pixel

        k = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0)
        # Centre pixel at 8 m: camera_xyz = (0, 0, 8).
        xyz = deproject_pixel(50.0, 50.0, 8.0, k)
        pose = CameraWorldPose(lat=LAT0, lng=LNG0, heading_deg=0.0, height_m=2.5)
        w = world_from_camera_xyz(xyz, pose)
        assert haversine_distance(LAT0, LNG0, w["lat"], w["lng"]) == pytest.approx(
            8.0, abs=0.01
        )
        assert w["up"] == pytest.approx(2.5, abs=1e-9)
