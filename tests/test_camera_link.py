# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for CameraDetectionLink model."""

import time

import pytest

from tritium_lib.models.camera_link import (
    CameraDetectionLink,
    CameraLinkSummary,
    FramePosition,
)


class TestFramePosition:
    def test_defaults(self):
        fp = FramePosition()
        assert fp.x == 0.0
        assert fp.y == 0.0

    def test_custom_values(self):
        fp = FramePosition(x=0.5, y=0.3)
        assert fp.x == 0.5
        assert fp.y == 0.3


class TestCameraDetectionLink:
    def test_defaults(self):
        link = CameraDetectionLink()
        assert link.link_id  # UUID generated
        assert link.detection_id == ""
        assert link.camera_id == ""
        assert link.target_id == ""
        assert link.confidence == 0.0
        assert link.timestamp > 0

    def test_full_construction(self):
        link = CameraDetectionLink(
            detection_id="det_person_42",
            camera_id="cam_front",
            target_id="ble_aabbccddeeff",
            class_name="person",
            position_in_frame=FramePosition(x=0.5, y=0.6),
            confidence=0.85,
            camera_fov_degrees=90.0,
            camera_rotation=45.0,
            bbox_area=0.12,
        )
        assert link.detection_id == "det_person_42"
        assert link.camera_id == "cam_front"
        assert link.target_id == "ble_aabbccddeeff"
        assert link.class_name == "person"
        assert link.position_in_frame.x == 0.5
        assert link.confidence == 0.85
        assert link.is_high_confidence is True
        assert link.camera_fov_degrees == 90.0

    def test_low_confidence(self):
        link = CameraDetectionLink(confidence=0.3)
        assert link.is_high_confidence is False

    def test_to_signal_dict(self):
        link = CameraDetectionLink(
            detection_id="det1",
            camera_id="cam1",
            class_name="vehicle",
            confidence=0.9,
            camera_fov_degrees=120.0,
        )
        d = link.to_signal_dict()
        assert d["detection_id"] == "det1"
        assert d["camera_id"] == "cam1"
        assert d["class_name"] == "vehicle"
        assert d["confidence"] == 0.9
        assert d["camera_fov_degrees"] == 120.0
        assert "position_in_frame" in d

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            CameraDetectionLink(confidence=1.5)
        with pytest.raises(Exception):
            CameraDetectionLink(confidence=-0.1)

    def test_serialization(self):
        link = CameraDetectionLink(
            detection_id="d1",
            camera_id="c1",
            target_id="t1",
            confidence=0.5,
        )
        d = link.model_dump()
        assert d["detection_id"] == "d1"
        assert d["camera_id"] == "c1"
        assert d["target_id"] == "t1"

    def test_import_from_models_init(self):
        """Verify the model is importable from the top-level models package."""
        from tritium_lib.models import CameraDetectionLink as CDL
        assert CDL is not None


class TestCameraLinkSummary:
    def test_defaults(self):
        s = CameraLinkSummary()
        assert s.total_links == 0
        assert s.unique_targets == 0
        assert s.unique_cameras == 0
        assert s.avg_confidence == 0.0
        assert s.class_distribution == {}

    def test_with_data(self):
        s = CameraLinkSummary(
            entity_id="cam_front",
            total_links=15,
            unique_targets=8,
            unique_cameras=1,
            avg_confidence=0.78,
            first_link=1000.0,
            last_link=2000.0,
            class_distribution={"person": 10, "vehicle": 5},
        )
        assert s.total_links == 15
        assert s.class_distribution["person"] == 10
