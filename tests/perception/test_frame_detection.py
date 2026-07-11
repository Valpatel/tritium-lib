# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Deterministic frame -> detection -> world -> sink chain (no GPU/network).

Proves the reusable perception primitive that turns camera pixels into
world-positioned detections the TargetTracker can consume as det_* tracks.
Fixture frames only: no ultralytics, no torch, no RTSP, no live server.
"""

from __future__ import annotations

import numpy as np
import pytest

from tritium_lib.models.camera import BoundingBox, CameraDetection
from tritium_lib.perception import (
    BackgroundMotionDetector,
    FrameDetectionPipeline,
    GroundCameraModel,
    available_backends,
    build_frame_detector,
)


# ------------------------------------------------------------------ helpers

def _blank(w: int = 320, h: int = 240, val: int = 40) -> np.ndarray:
    """A uniform BGR background frame."""
    return np.full((h, w, 3), val, dtype=np.uint8)


def _with_person(frame: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Stamp a tall bright rectangle (a 'person') centred at (cx, cy)."""
    f = frame.copy()
    pw, ph = 24, 70  # tall -> person aspect
    x1, y1 = cx - pw // 2, cy - ph // 2
    x2, y2 = cx + pw // 2, cy + ph // 2
    f[max(0, y1):y2, max(0, x1):x2] = (230, 230, 230)
    return f


def _with_car(frame: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Stamp a wide bright rectangle (a 'car') centred at (cx, cy)."""
    f = frame.copy()
    cw, ch = 90, 34  # wide -> vehicle aspect
    f[cy - ch // 2:cy + ch // 2, cx - cw // 2:cx + cw // 2] = (200, 210, 220)
    return f


def _learn_background(det: BackgroundMotionDetector, frame: np.ndarray, n: int = 20):
    """Feed n copies of the empty background so MOG2 models it as static."""
    for _ in range(n):
        det.detect(frame, "fixture")


# ------------------------------------------------------------------ detector

def test_backend_availability_reports_motion_always():
    backends = available_backends()
    assert backends["motion"] is True
    assert "yolo" in backends  # bool either way


def test_motion_detector_finds_a_person_blob():
    det = BackgroundMotionDetector(min_area=200)
    bg = _blank()
    _learn_background(det, bg)

    dets = det.detect(_with_person(bg, 160, 120), "cam1")
    assert dets, "no detection on a clear foreground object"
    person = max(dets, key=lambda d: d.bbox.area)
    assert person.class_name == "person"
    assert person.source_id == "cam1"
    assert 0.4 <= person.confidence <= 1.0
    # Box centre lands near the injected object centre.
    ccx, ccy = person.bbox.center
    assert abs(ccx - 160) < 30
    assert abs(ccy - 120) < 40


def test_motion_detector_classifies_wide_blob_as_car():
    det = BackgroundMotionDetector(min_area=200)
    bg = _blank()
    _learn_background(det, bg)
    dets = det.detect(_with_car(bg, 160, 120), "cam1")
    assert dets
    assert max(dets, key=lambda d: d.bbox.area).class_name == "car"


def test_motion_detector_quiet_scene_yields_nothing():
    det = BackgroundMotionDetector(min_area=200)
    bg = _blank()
    _learn_background(det, bg)
    # Same background again -> no foreground.
    assert det.detect(bg, "cam1") == []


def test_motion_detector_is_deterministic():
    def run():
        det = BackgroundMotionDetector(min_area=200)
        bg = _blank()
        _learn_background(det, bg)
        out = det.detect(_with_person(bg, 160, 120), "cam1")
        return [(d.class_name, round(d.bbox.x), round(d.bbox.y),
                 round(d.bbox.w), round(d.bbox.h), d.confidence) for d in out]

    assert run() == run(), "detector output diverged across identical runs"


def test_empty_frame_is_safe():
    det = BackgroundMotionDetector()
    assert det.detect(None, "cam1") == []
    assert det.detect(np.zeros((0, 0, 3), dtype=np.uint8), "cam1") == []


def test_build_frame_detector_falls_back_to_motion_when_no_yolo():
    # On a box without ultralytics/torch this must degrade, never raise.
    det = build_frame_detector(prefer="auto")
    assert det.backend_name in ("motion",) or det.backend_name.startswith("yolo:")
    # Forcing motion always yields the classical backend.
    assert build_frame_detector(prefer="motion").backend_name == "motion"


# ---------------------------------------------------------------- projection

def _det(x, y, w, h, cls="person") -> CameraDetection:
    return CameraDetection(
        source_id="cam", class_name=cls, confidence=0.9,
        bbox=BoundingBox(x=float(x), y=float(y), w=float(w), h=float(h)),
    )


def test_projection_centre_bottom_is_near_and_on_heading():
    model = GroundCameraModel(
        lat=40.0, lng=-105.0, heading_deg=90.0, fov_deg=70.0,
        range_m=80.0, image_w=640, image_h=480, near_m=3.0,
    )
    # Foot point at horizontal centre, bottom of frame -> straight ahead, near.
    out = model.bearing_range(_det(300, 400, 40, 80))
    bearing, distance = out
    assert abs(bearing - 90.0) < 1.0, "centre column should be on heading"
    assert distance == pytest.approx(model.near_m, abs=1.0)


def test_projection_high_in_frame_is_far():
    model = GroundCameraModel(
        lat=40.0, lng=-105.0, heading_deg=0.0, fov_deg=70.0,
        range_m=80.0, image_w=640, image_h=480,
    )
    near = model.bearing_range(_det(300, 470, 40, 10))[1]   # feet at bottom
    far = model.bearing_range(_det(300, 170, 40, 10))[1]    # feet near horizon
    assert far > near, "objects higher in frame must project farther"


def test_projection_left_of_centre_bears_less_than_heading():
    model = GroundCameraModel(
        lat=40.0, lng=-105.0, heading_deg=90.0, fov_deg=70.0,
        image_w=640, image_h=480,
    )
    left = model.bearing_range(_det(40, 400, 40, 80))[0]
    right = model.bearing_range(_det(560, 400, 40, 80))[0]
    assert left < 90.0 < right, "FOV should span the heading"
    assert (right - left) == pytest.approx(70.0 * (520 / 640), abs=2.0)


def test_projection_yields_world_xy():
    model = GroundCameraModel(lat=40.0, lng=-105.0, heading_deg=90.0)
    world = model.project(_det(320, 400, 40, 80))
    assert "x" in world and "y" in world
    assert isinstance(world["x"], float) and isinstance(world["y"], float)
    # East-facing near detection: mostly +x (east) offset from the camera.
    assert world["distance_m"] > 0


# ------------------------------------------------------------------ pipeline

def test_pipeline_full_chain_frame_to_sink():
    """feed frame -> detect -> project -> sink payload the tracker consumes."""
    det = BackgroundMotionDetector(min_area=200)
    bg = _blank()
    _learn_background(det, bg)

    # A one-shot frame provider: hands back the person frame once.
    frames = [_with_person(bg, 160, 120)]

    def frame_provider():
        return frames[0] if frames else None

    model = GroundCameraModel(
        lat=40.0, lng=-105.0, heading_deg=90.0, fov_deg=70.0,
        range_m=80.0, image_w=320, image_h=240,
    )
    captured: list[dict] = []

    pipe = FrameDetectionPipeline(
        detector=det,
        frame_provider=frame_provider,
        detection_sink=captured.append,
        model_provider=lambda: model,
        source_id="cam-front-01",
        min_confidence=0.4,
    )

    emitted = pipe.tick()
    assert emitted >= 1, "pipeline emitted no detections for a clear object"
    assert pipe.detections_total == emitted
    payload = captured[0]
    # Exactly the shape TargetTracker.update_from_detection wants.
    assert payload["class_name"] == "person"
    assert payload["source_camera"] == "cam-front-01"
    assert "center_x" in payload and "center_y" in payload
    assert isinstance(payload["center_x"], float)
    assert payload["confidence"] >= 0.4


def test_pipeline_no_frame_is_noop():
    det = BackgroundMotionDetector()
    calls: list[dict] = []
    pipe = FrameDetectionPipeline(
        detector=det,
        frame_provider=lambda: None,
        detection_sink=calls.append,
        source_id="cam",
    )
    assert pipe.tick() == 0
    assert calls == []


def test_pipeline_without_pose_emits_normalized_coords():
    det = BackgroundMotionDetector(min_area=200)
    bg = _blank()
    _learn_background(det, bg)
    captured: list[dict] = []
    pipe = FrameDetectionPipeline(
        detector=det,
        frame_provider=lambda: _with_person(bg, 160, 120),
        detection_sink=captured.append,
        model_provider=None,
        source_id="cam",
    )
    assert pipe.tick() >= 1
    p = captured[0]
    # Normalized image coords are in [0, 1].
    assert 0.0 <= p["center_x"] <= 1.0
    assert 0.0 <= p["center_y"] <= 1.0


def test_pipeline_feeds_target_tracker_as_det_track():
    """End-to-end into the real TargetTracker -> a det_* vision track."""
    from tritium_lib.tracking.target_tracker import TargetTracker

    det = BackgroundMotionDetector(min_area=200)
    bg = _blank()
    _learn_background(det, bg)
    tracker = TargetTracker()
    model = GroundCameraModel(
        lat=40.0, lng=-105.0, heading_deg=90.0, range_m=80.0,
        image_w=320, image_h=240,
    )
    pipe = FrameDetectionPipeline(
        detector=det,
        frame_provider=lambda: _with_person(bg, 160, 120),
        detection_sink=tracker.update_from_detection,
        model_provider=lambda: model,
        source_id="cam-front-01",
    )
    assert pipe.tick() >= 1
    det_tracks = [t for t in tracker.get_all()
                  if str(t.target_id).startswith("det_")]
    assert det_tracks, "no det_* vision track created in the TargetTracker"
    t = det_tracks[0]
    # The pipeline stamps source_camera, so the track carries CAMERA
    # provenance: source="camera" plus which camera saw it (kinematics).
    assert t.source == "camera"
    assert "camera" in t.confirming_sources
    assert t.kinematics is not None
    assert t.kinematics["camera_id"] == "cam-front-01"
    assert "bearing_deg" in t.kinematics
    assert "distance_m" in t.kinematics
    assert "bbox" in t.kinematics
    # Operator surface: to_dict must expose the provenance for map/dossier.
    d = t.to_dict()
    assert d["source"] == "camera"
    assert d["kinematics"]["camera_id"] == "cam-front-01"
