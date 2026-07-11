# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Camera provenance on vision tracks (UX Loop 8).

A detection that arrives with ``source_camera`` (set by
FrameDetectionPipeline and the SC camera bridges) must produce a track the
operator can trace back to the observing camera: ``source="camera"`` and
``kinematics.camera_id``.  Detections without camera identity keep the
legacy ``source="yolo"`` contract.  Both live in ONE vision track regime —
a camera detection refreshes a yolo track instead of duplicating it.

Deterministic: no frames, no network, no GPU — bare detection dicts.
"""

import time

import pytest

from tritium_lib.tracking.target_tracker import TargetTracker, _decayed_confidence


def _cam_det(cx=10.0, cy=5.0, camera="cam-alpha", **extra):
    det = {
        "class_name": "person",
        "confidence": 0.9,
        "center_x": cx,
        "center_y": cy,
        "source_camera": camera,
        "bearing_deg": 45.0,
        "distance_m": 12.5,
        "bbox": {"x": 100.0, "y": 50.0, "w": 40.0, "h": 120.0},
    }
    det.update(extra)
    return det


def _yolo_det(cx=10.0, cy=5.0):
    return {
        "class_name": "person",
        "confidence": 0.9,
        "center_x": cx,
        "center_y": cy,
    }


class TestCameraProvenanceCreate:
    def test_source_camera_creates_camera_track(self):
        tracker = TargetTracker()
        tracker.update_from_detection(_cam_det())
        (t,) = tracker.get_all()
        assert t.target_id.startswith("det_person_")
        assert t.source == "camera"
        assert t.position_source == "camera"
        assert t.confirming_sources == {"camera"}
        assert t.kinematics["camera_id"] == "cam-alpha"
        assert t.kinematics["bearing_deg"] == 45.0
        assert t.kinematics["distance_m"] == 12.5
        assert t.kinematics["bbox"] == {
            "x": 100.0, "y": 50.0, "w": 40.0, "h": 120.0,
        }

    def test_no_source_camera_keeps_legacy_yolo(self):
        tracker = TargetTracker()
        tracker.update_from_detection(_yolo_det())
        (t,) = tracker.get_all()
        assert t.source == "yolo"
        assert t.confirming_sources == {"yolo"}
        assert t.kinematics is None

    def test_to_dict_exposes_provenance(self):
        tracker = TargetTracker()
        tracker.update_from_detection(_cam_det())
        (t,) = tracker.get_all()
        d = t.to_dict()
        assert d["source"] == "camera"
        assert d["kinematics"]["camera_id"] == "cam-alpha"


class TestOneVisionRegime:
    def test_camera_detection_refreshes_yolo_track(self):
        """Same physical contact: yolo track first, camera det refreshes it."""
        tracker = TargetTracker()
        tracker.update_from_detection(_yolo_det(cx=10.0, cy=5.0))
        tracker.update_from_detection(_cam_det(cx=10.5, cy=5.2))
        targets = tracker.get_all()
        assert len(targets) == 1, "camera det must fuse, not duplicate"
        t = targets[0]
        assert t.source == "yolo"  # created as yolo; source is birth identity
        assert t.signal_count == 2
        # Provenance stamped even on a refresh — the dossier can still
        # answer "which camera saw this".
        assert t.kinematics["camera_id"] == "cam-alpha"

    def test_yolo_detection_refreshes_camera_track(self):
        tracker = TargetTracker()
        tracker.update_from_detection(_cam_det(cx=10.0, cy=5.0))
        tracker.update_from_detection(_yolo_det(cx=10.5, cy=5.2))
        targets = tracker.get_all()
        assert len(targets) == 1
        assert targets[0].source == "camera"
        assert targets[0].signal_count == 2

    def test_vision_pair_is_not_cross_modal_confirmation(self):
        """camera+yolo must NOT inflate multi-source fusion metrics."""
        tracker = TargetTracker()
        tracker.update_from_detection(_yolo_det())
        tracker.update_from_detection(_cam_det(cx=10.2, cy=5.1))
        (t,) = tracker.get_all()
        assert t.confirming_sources == {"yolo"}
        assert t.to_dict()["source_count"] == 1

    def test_track_moves_with_consecutive_camera_detections(self):
        """One id, moving position — the map glyph should travel."""
        tracker = TargetTracker()
        for i in range(5):
            tracker.update_from_detection(_cam_det(cx=10.0 + i, cy=5.0))
        (t,) = tracker.get_all()
        assert t.signal_count == 5
        assert t.position == (14.0, 5.0)

    def test_kinematics_from_other_source_preserved(self):
        tracker = TargetTracker()
        tracker.update_from_detection(_yolo_det())
        (t,) = tracker.get_all()
        t.kinematics = {"radar_range_m": 99.0}
        tracker.update_from_detection(_cam_det(cx=10.2, cy=5.1))
        assert t.kinematics["radar_range_m"] == 99.0
        assert t.kinematics["camera_id"] == "cam-alpha"


class TestUpdateFromCameraDetection:
    _FLAT = staticmethod(lambda lat, lng: (lat, lng, 0.0))

    def test_camera_id_param_gives_camera_provenance(self):
        tracker = TargetTracker()
        tracker.update_from_camera_detection(
            {"label": "person", "confidence": 0.8,
             "bbox": {"x": 0.5, "y": 0.5}},
            camera_lat=1.0, camera_lng=2.0,
            latlng_to_local_fn=self._FLAT,
            camera_id="demo-cam-01",
        )
        (t,) = tracker.get_all()
        assert t.source == "camera"
        assert t.kinematics["camera_id"] == "demo-cam-01"

    def test_detection_dict_camera_id_fallback(self):
        tracker = TargetTracker()
        tracker.update_from_camera_detection(
            {"label": "person", "confidence": 0.8,
             "camera_id": "demo-cam-02", "bbox": {"x": 0.5, "y": 0.5}},
            camera_lat=1.0, camera_lng=2.0,
            latlng_to_local_fn=self._FLAT,
        )
        (t,) = tracker.get_all()
        assert t.source == "camera"
        assert t.kinematics["camera_id"] == "demo-cam-02"

    def test_legacy_call_stays_yolo(self):
        tracker = TargetTracker()
        tracker.update_from_camera_detection(
            {"label": "person", "confidence": 0.8,
             "bbox": {"x": 0.5, "y": 0.5}},
            camera_lat=1.0, camera_lng=2.0,
            latlng_to_local_fn=self._FLAT,
        )
        (t,) = tracker.get_all()
        assert t.source == "yolo"
        assert t.kinematics is None


class TestLifecycle:
    def test_camera_track_prunes_like_vision(self):
        tracker = TargetTracker()
        tracker.update_from_detection(_cam_det())
        (t,) = tracker.get_all()
        t.last_seen = time.monotonic() - (TargetTracker.STALE_TIMEOUT + 1.0)
        tracker._prune_stale()
        assert tracker.get_all() == []

    def test_camera_confidence_decays_like_yolo(self):
        assert _decayed_confidence("camera", 1.0, 15.0) == pytest.approx(
            _decayed_confidence("yolo", 1.0, 15.0)
        )
        assert _decayed_confidence("camera", 1.0, 15.0) == pytest.approx(0.5)

    def test_clear_source_camera(self):
        tracker = TargetTracker()
        tracker.update_from_detection(_cam_det())
        tracker.update_from_detection(_yolo_det(cx=500.0, cy=500.0))
        assert tracker.clear_source("camera") == 1
        (t,) = tracker.get_all()
        assert t.source == "yolo"
