# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""One-call depth pipeline — RGB + depth -> unique 3D map targets.

The contract under test: a SINGLE :func:`process_depth_frame` call (or
``DepthCameraPipeline.process``) runs the whole detect -> enrich -> place ->
fuse chain that consumers previously wired by hand.  With a fake injected
detector, synthetic depth, and a known camera pose, one call must yield a
unique ``det_*`` id at the correct world position; ``depth=None`` must
degrade to the documented flat-ground path (never crash); repeated frames
of one entity must keep ONE id; an empty or crashing detector must yield an
empty frame, not an exception.
"""

from __future__ import annotations

import numpy as np
import pytest

import tritium_lib.geo as geo
from tritium_lib.models.camera import BoundingBox, CameraDetection
from tritium_lib.perception import (
    CameraIntrinsics,
    CameraWorldPose,
    DepthCameraPipeline,
    GroundCameraModel,
    process_depth_frame,
)
from tritium_lib.perception.depth_pipeline import GROUND_FALLBACK_SOURCE
from tritium_lib.tracking import TargetTracker


@pytest.fixture(autouse=True)
def _clean_geo():
    """Each test starts and ends without a global geo reference."""
    geo.reset()
    yield
    geo.reset()


# A 640x480 rig facing North, mounted 2 m up, 90 deg HFOV.
W, H = 640, 480
INTR = CameraIntrinsics.from_fov(W, H, horizontal_fov_deg=90.0)
POSE = CameraWorldPose(heading_deg=0.0, height_m=2.0)
# Person bbox centred on the principal point: centre pixel (320, 240).
PERSON_BOX = {"x": 300.0, "y": 195.0, "w": 40.0, "h": 90.0}


class FakeDetector:
    """Injected duck-typed detector: fresh CameraDetection models per call."""

    def __init__(self, boxes: list[dict] | None = None):
        self.boxes = boxes if boxes is not None else [PERSON_BOX]
        self.calls = 0

    def detect(self, frame, source_id=""):
        self.calls += 1
        return [
            CameraDetection(
                source_id=source_id or "fake-cam",
                class_name=b.get("class_name", "person"),
                confidence=b.get("confidence", 0.95),
                bbox=BoundingBox(
                    x=b["x"], y=b["y"], w=b["w"], h=b["h"],
                ),
            )
            for b in self.boxes
        ]


class FakeDictDetector:
    """Detector returning plain dicts — the other supported detection shape."""

    def detect(self, frame, source_id=""):
        return [{
            "class_name": "person",
            "confidence": 0.95,
            "bbox": dict(PERSON_BOX),
        }]


class CrashingDetector:
    def detect(self, frame, source_id=""):
        raise RuntimeError("sensor exploded")


def _rgb() -> np.ndarray:
    return np.zeros((H, W, 3), dtype=np.uint8)


def _depth(at: float = 10.0) -> np.ndarray:
    return np.full((H, W), float(at))


# ----------------------------------------------------------- one-call chain


class TestOneCallChain:
    def test_single_call_yields_unique_id_at_correct_world_position(self):
        tracker = TargetTracker()
        ids = process_depth_frame(
            _rgb(), _depth(10.0), INTR, POSE, tracker, FakeDetector(),
            source="isaac_depth", cam_id="cam-depth-01",
        )

        assert len(ids) == 1 and ids[0] is not None
        assert ids[0].startswith("det_person_")
        target = tracker.get_target(ids[0])
        # Camera faces North, box on the optical axis, 10 m of depth:
        # the person stands ~10 m due north of the camera.
        assert target.position[1] == pytest.approx(10.0, abs=0.5)
        assert abs(target.position[0]) < 1.0
        kin = target.kinematics
        assert kin["range_m"] == pytest.approx(10.0)
        assert kin["depth_source"] == "isaac_depth"
        assert kin["camera_id"] == "cam-depth-01"
        assert kin["world_enu"][2] == pytest.approx(2.0, abs=0.1)  # mount height
        assert target.asset_type == "person"

    def test_dict_detections_flow_through_the_same_call(self):
        tracker = TargetTracker()
        ids = process_depth_frame(
            _rgb(), _depth(10.0), INTR, POSE, tracker, FakeDictDetector(),
            cam_id="cam-1",
        )
        assert ids[0] is not None
        assert tracker.get_target(ids[0]).position[1] == pytest.approx(10.0, abs=0.5)

    def test_two_objects_two_unique_ids(self):
        left = dict(PERSON_BOX, x=80.0)     # well left of centre
        right = dict(PERSON_BOX, x=520.0)   # well right of centre
        tracker = TargetTracker()
        ids = process_depth_frame(
            _rgb(), _depth(10.0), INTR, POSE, tracker, FakeDetector([left, right]),
            cam_id="cam-1",
        )
        assert None not in ids
        assert ids[0] != ids[1]
        assert len([t for t in tracker.get_all() if t.source == "camera"]) == 2

    def test_millimetre_depth_via_depth_scale(self):
        depth_mm = np.full((H, W), 10000, dtype=np.uint16)  # 10 m in mm
        tracker = TargetTracker()
        ids = process_depth_frame(
            _rgb(), depth_mm, INTR, POSE, tracker, FakeDetector(),
            cam_id="cam-1", depth_scale=0.001,
        )
        assert tracker.get_target(ids[0]).kinematics["range_m"] == pytest.approx(10.0)


# ------------------------------------------------------------- stable ids


class TestStableIdAcrossFrames:
    def test_repeated_frames_one_entity_one_id(self):
        tracker = TargetTracker()
        detector = FakeDetector()
        seen: set[str] = set()
        for _ in range(3):
            ids = process_depth_frame(
                _rgb(), _depth(10.0), INTR, POSE, tracker, detector,
                source="isaac_depth", cam_id="cam-depth-01",
            )
            assert ids[0] is not None
            seen.add(ids[0])
        assert len(seen) == 1, "stable entity must keep ONE unique id"
        vision = [t for t in tracker.get_all() if t.source == "camera"]
        assert len(vision) == 1
        assert vision[0].signal_count == 3

    def test_depth_dropout_keeps_id_when_positions_agree(self):
        """A frame with depth then a frame without must not fork the track,
        provided the 2D ground guess lands within the association budget."""
        tracker = TargetTracker()
        detector = FakeDetector()
        # Ground-model position for this box — place the depth frame there
        # so the two paths agree spatially.
        model = GroundCameraModel(
            lat=None, lng=None, heading_deg=0.0, fov_deg=90.0,
            image_w=W, image_h=H,
        )
        det = detector.detect(None, "cam-1")[0]
        ground = model.project(det)
        (tid1,) = process_depth_frame(
            _rgb(), _depth(ground["distance_m"]), INTR, POSE, tracker, detector,
            cam_id="cam-1",
        )
        (tid2,) = process_depth_frame(
            _rgb(), None, INTR, POSE, tracker, detector, cam_id="cam-1",
        )
        assert tid1 is not None and tid2 == tid1


# --------------------------------------------------------- depth=None path


class TestDepthNoneGraceful:
    def test_depth_none_still_places_via_ground_path(self):
        tracker = TargetTracker()
        ids = process_depth_frame(
            _rgb(), None, INTR, POSE, tracker, FakeDetector(), cam_id="cam-1",
        )
        assert len(ids) == 1 and ids[0] is not None
        target = tracker.get_target(ids[0])
        # Same placement the documented 2D GroundCameraModel produces.
        model = GroundCameraModel(
            lat=None, lng=None, heading_deg=0.0, fov_deg=90.0,
            image_w=W, image_h=H,
        )
        det = FakeDetector().detect(None, "cam-1")[0]
        expected = model.project(det)
        assert target.position == pytest.approx(
            (expected["x"], expected["y"]), abs=0.05)
        # Honest provenance: a guess is labelled a guess, and no fabricated
        # measured range rides on the track.
        assert target.kinematics["depth_source"] == GROUND_FALLBACK_SOURCE
        assert "range_m" not in target.kinematics

    def test_depth_none_with_fallback_off_is_pure_passthrough(self):
        tracker = TargetTracker()
        ids = process_depth_frame(
            _rgb(), None, INTR, POSE, tracker, FakeDetector(),
            cam_id="cam-1", ground_fallback=False,
        )
        assert ids == [None]
        assert tracker.get_all() == []

    def test_no_pose_no_depth_never_crashes(self):
        tracker = TargetTracker()
        ids = process_depth_frame(
            _rgb(), None, None, None, tracker, FakeDetector(), cam_id="cam-1",
        )
        assert ids == [None]
        assert tracker.get_all() == []


# ------------------------------------------------------------ empty frames


class TestEmptyAndBroken:
    def test_no_detections_empty_list(self):
        tracker = TargetTracker()
        ids = process_depth_frame(
            _rgb(), _depth(10.0), INTR, POSE, tracker, FakeDetector([]),
        )
        assert ids == []
        assert tracker.get_all() == []

    def test_crashing_detector_yields_empty_frame_not_exception(self):
        tracker = TargetTracker()
        ids = process_depth_frame(
            _rgb(), _depth(10.0), INTR, POSE, tracker, CrashingDetector(),
        )
        assert ids == []

    def test_none_detector_yields_empty_frame(self):
        assert process_depth_frame(
            _rgb(), _depth(10.0), INTR, POSE, TargetTracker(), None,
        ) == []

    def test_low_confidence_rejected(self):
        weak = dict(PERSON_BOX, confidence=0.2)  # below the 0.4 vision gate
        tracker = TargetTracker()
        ids = process_depth_frame(
            _rgb(), _depth(10.0), INTR, POSE, tracker, FakeDetector([weak]),
            cam_id="cam-1",
        )
        assert ids == [None]
        assert tracker.get_all() == []


# --------------------------------------------------------------- geo frame


class TestGeoPlacement:
    def test_posed_geo_camera_yields_absolute_fix(self):
        geo.init_reference(37.7749, -122.4194)
        pose = CameraWorldPose(
            lat=37.7749, lng=-122.4194, heading_deg=90.0, height_m=2.0,
        )
        tracker = TargetTracker()
        (tid,) = process_depth_frame(
            _rgb(), _depth(10.0), INTR, pose, tracker, FakeDetector(),
            cam_id="cam-geo",
        )
        target = tracker.get_target(tid)
        # Camera faces East: the fix sits ~10 m east of the reference.
        assert target.position[0] == pytest.approx(10.0, abs=0.5)
        assert abs(target.position[1]) < 1.0
        assert target.kinematics["world_lat"] == pytest.approx(37.7749, abs=1e-3)


# ---------------------------------------------------------- pipeline class


class TestDepthCameraPipeline:
    def test_per_frame_one_liner_and_counters(self):
        tracker = TargetTracker()
        pipe = DepthCameraPipeline(
            FakeDetector(), tracker, INTR, POSE,
            source="realsense", cam_id="rgbd-01",
        )
        ids1 = pipe.process(_rgb(), _depth(10.0))
        ids2 = pipe.process(_rgb(), _depth(10.2))
        assert ids1[0] is not None and ids2[0] == ids1[0]
        assert pipe.frames_processed == 2
        assert pipe.tracked_total == 2
        assert pipe.ids_last == ids2
        target = tracker.get_target(ids1[0])
        assert target.kinematics["depth_source"] == "realsense"
        assert target.kinematics["camera_id"] == "rgbd-01"

    def test_rgb_only_camera_uses_same_pipeline(self):
        pipe = DepthCameraPipeline(
            FakeDetector(), TargetTracker(), INTR, POSE, cam_id="rgb-01",
        )
        ids = pipe.process(_rgb())  # no depth argument at all
        assert ids[0] is not None
