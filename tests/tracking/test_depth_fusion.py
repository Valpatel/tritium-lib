# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Depth fusion — world-placed camera detections -> unique TrackedTargets.

The contract under test is the last link of the vision -> map -> unique-ID
chain: repeated detections of one entity at a stable world position must map
to ONE ``det_*`` id carrying the depth-derived 3D position; distinct entities
must get distinct ids; detections without a world position pass through
gracefully.  Same contract for an Isaac Sim depth camera and a real RGB-D
sensor — the payloads are identical by design.
"""

from __future__ import annotations

import numpy as np
import pytest

import tritium_lib.geo as geo
from tritium_lib.perception import (
    CameraIntrinsics,
    CameraWorldPose,
    enrich_detections_with_depth,
    place_detections_on_map,
)
from tritium_lib.models.camera import BoundingBox, CameraDetection
from tritium_lib.tracking import TargetTracker, fuse_depth_detections


@pytest.fixture(autouse=True)
def _clean_geo():
    """Each test starts and ends without a global geo reference."""
    geo.reset()
    yield
    geo.reset()


def _person_at(east: float, north: float, up: float = 1.0, **extra) -> dict:
    det = {
        "class_name": "person",
        "confidence": 0.9,
        "bbox": {"x": 300.0, "y": 200.0, "w": 40.0, "h": 90.0},
        "range_m": float(np.hypot(east, north)),
        "world_enu": (east, north, up),
    }
    det.update(extra)
    return det


# ---------------------------------------------------------------- stable id


class TestStableUniqueId:
    def test_two_frames_one_object_one_id(self):
        tracker = TargetTracker()
        frame1 = [_person_at(12.0, 8.0)]
        frame2 = [_person_at(12.3, 8.2)]  # same entity, slight jitter

        ids1 = fuse_depth_detections(
            tracker, frame1, source="isaac_depth", cam_id="cam-depth-01")
        ids2 = fuse_depth_detections(
            tracker, frame2, source="isaac_depth", cam_id="cam-depth-01")

        assert ids1[0] is not None
        assert ids2[0] == ids1[0], "stable entity must keep ONE unique id"
        vision = [t for t in tracker.get_all() if t.source == "camera"]
        assert len(vision) == 1

    def test_position_is_depth_derived(self):
        tracker = TargetTracker()
        (tid,) = fuse_depth_detections(
            tracker, [_person_at(12.0, 8.0, up=1.5)],
            source="isaac_depth", cam_id="cam-depth-01")

        target = tracker.get_target(tid)
        assert target is not None
        # ENU east/north become local x/y (camera at local origin).
        assert target.position == pytest.approx((12.0, 8.0))
        # Measured 3D data rides on kinematics for the dossier/map.
        kin = target.kinematics
        assert kin["camera_id"] == "cam-depth-01"
        assert kin["range_m"] == pytest.approx(np.hypot(12.0, 8.0))
        assert kin["world_enu"] == pytest.approx([12.0, 8.0, 1.5])
        assert kin["elevation_m"] == pytest.approx(1.5)
        assert kin["depth_source"] == "isaac_depth"
        assert kin["distance_m"] == pytest.approx(np.hypot(12.0, 8.0), abs=0.01)
        assert kin["bearing_deg"] == pytest.approx(
            np.degrees(np.arctan2(12.0, 8.0)), abs=0.01)

    def test_signal_count_accumulates_on_one_track(self):
        tracker = TargetTracker()
        tid = None
        for _ in range(4):
            (tid,) = fuse_depth_detections(
                tracker, [_person_at(20.0, -5.0)], cam_id="cam-1")
        target = tracker.get_target(tid)
        assert target.signal_count == 4


# ------------------------------------------------------------- distinct ids


class TestDistinctObjects:
    def test_two_objects_two_ids(self):
        tracker = TargetTracker()
        ids = fuse_depth_detections(
            tracker,
            [_person_at(10.0, 5.0), _person_at(40.0, -20.0)],
            source="realsense", cam_id="cam-depth-01")

        assert None not in ids
        assert ids[0] != ids[1], "distinct entities must get distinct ids"
        vision = [t for t in tracker.get_all() if t.source == "camera"]
        assert len(vision) == 2

    def test_different_classes_never_merge(self):
        tracker = TargetTracker()
        person = _person_at(10.0, 5.0)
        car = _person_at(10.5, 5.2)
        car["class_name"] = "car"
        ids = fuse_depth_detections(tracker, [person, car], cam_id="cam-1")
        assert ids[0] != ids[1]


# ------------------------------------------------------- graceful passthrough


class TestGracefulPassthrough:
    def test_no_world_position_yields_none_and_no_track(self):
        tracker = TargetTracker()
        bare = {  # detected, depth-less: no world_enu / world_lat / world_lng
            "class_name": "person",
            "confidence": 0.9,
            "bbox": {"x": 10.0, "y": 10.0, "w": 40.0, "h": 90.0},
        }
        ids = fuse_depth_detections(tracker, [bare], cam_id="cam-1")
        assert ids == [None]
        assert tracker.get_all() == []

    def test_junk_world_annotations_yield_none(self):
        tracker = TargetTracker()
        junk = {
            "class_name": "person",
            "confidence": 0.9,
            "world_enu": "not-a-tuple",
            "world_lat": float("nan"),
            "world_lng": None,
        }
        ids = fuse_depth_detections(tracker, [junk], cam_id="cam-1")
        assert ids == [None]
        assert tracker.get_all() == []

    def test_mixed_batch_fuses_only_placed_detections(self):
        tracker = TargetTracker()
        placed = _person_at(15.0, 15.0)
        unplaced = {"class_name": "person", "confidence": 0.9}
        ids = fuse_depth_detections(tracker, [placed, unplaced], cam_id="cam-1")
        assert ids[0] is not None
        assert ids[1] is None
        assert len(tracker.get_all()) == 1

    def test_empty_list(self):
        tracker = TargetTracker()
        assert fuse_depth_detections(tracker, []) == []
        assert tracker.get_all() == []

    def test_low_confidence_rejected_by_tracker(self):
        tracker = TargetTracker()
        weak = _person_at(10.0, 5.0)
        weak["confidence"] = 0.2  # below the tracker's 0.4 vision gate
        ids = fuse_depth_detections(tracker, [weak], cam_id="cam-1")
        assert ids == [None]
        assert tracker.get_all() == []


# ---------------------------------------------------------------- geo frame


class TestGeoPlacement:
    def test_world_latlng_places_via_geo_reference(self):
        geo.init_reference(37.7749, -122.4194)
        tracker = TargetTracker()
        # An absolute fix ~55 m north-east of the reference.
        lat, lng = geo.destination_point(37.7749, -122.4194, 45.0, 55.0)
        det = _person_at(0.0, 0.0)  # ENU says origin — lat/lng must win
        det["world_lat"] = lat
        det["world_lng"] = lng

        (tid,) = fuse_depth_detections(tracker, [det], cam_id="cam-geo")
        target = tracker.get_target(tid)
        expected_x, expected_y, _ = geo.latlng_to_local(lat, lng)
        assert target.position == pytest.approx((expected_x, expected_y), abs=0.5)
        assert target.kinematics["world_lat"] == pytest.approx(lat)
        assert target.kinematics["world_lng"] == pytest.approx(lng)

    def test_enu_offset_by_camera_origin(self):
        tracker = TargetTracker()
        (tid,) = fuse_depth_detections(
            tracker, [_person_at(5.0, 5.0)],
            cam_id="cam-1", camera_origin_xy=(100.0, 200.0))
        target = tracker.get_target(tid)
        assert target.position == pytest.approx((105.0, 205.0))


# ----------------------------------------------------------- full 3D chain


class TestFullChain:
    """detector bbox -> depth -> world pose -> tracker, end to end."""

    def test_isaac_style_depth_camera_yields_unique_map_target(self):
        # A 640x480 depth frame: everything 10 m out (Isaac
        # distance_to_image_plane style), one person mid-frame.
        depth = np.full((480, 640), 10.0)
        intr = CameraIntrinsics.from_fov(640, 480, horizontal_fov_deg=90.0)
        pose = CameraWorldPose(heading_deg=0.0, height_m=2.0)  # facing North
        tracker = TargetTracker()

        tid = None
        for _ in range(3):  # three frames of the same standing person
            dets = [CameraDetection(
                source_id="isaac-cam-01",
                class_name="person",
                confidence=0.95,
                bbox=BoundingBox(x=300.0, y=195.0, w=40.0, h=90.0),
            )]
            enrich_detections_with_depth(dets, depth, intr)
            place_detections_on_map(dets, pose)
            assert dets[0].range_m == pytest.approx(10.0)
            assert dets[0].world_enu is not None
            (new_tid,) = fuse_depth_detections(
                tracker, dets, source="isaac_depth", cam_id="isaac-cam-01")
            assert new_tid is not None
            assert tid is None or new_tid == tid
            tid = new_tid

        vision = [t for t in tracker.get_all() if t.source == "camera"]
        assert len(vision) == 1
        target = vision[0]
        # Camera faces North: the detection sits ~10 m north, near-zero east.
        assert target.position[1] == pytest.approx(10.0, abs=0.5)
        assert abs(target.position[0]) < 1.0
        assert target.kinematics["range_m"] == pytest.approx(10.0)
        assert target.kinematics["depth_source"] == "isaac_depth"
        assert target.asset_type == "person"
        assert target.target_id.startswith("det_person_")
