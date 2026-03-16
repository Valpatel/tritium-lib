# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for camera source and detection models."""

from datetime import datetime

import pytest

from tritium_lib.models.camera import (
    BoundingBox,
    CameraDetection,
    CameraFrame,
    CameraFrameFormat,
    CameraPosition,
    CameraSource,
    CameraSourceType,
)


# ── CameraPosition ──────────────────────────────────────────────────

class TestCameraPosition:
    def test_defaults(self):
        pos = CameraPosition()
        assert pos.lat is None
        assert pos.lng is None
        assert pos.alt is None

    def test_full(self):
        pos = CameraPosition(lat=37.0, lng=-122.0, alt=100.0)
        assert pos.lat == 37.0
        assert pos.alt == 100.0


# ── CameraSource ────────────────────────────────────────────────────

class TestCameraSource:
    def test_minimal(self):
        src = CameraSource(source_id="cam-01")
        assert src.source_id == "cam-01"
        assert src.name == ""
        assert src.source_type == CameraSourceType.SYNTHETIC
        assert src.url is None
        assert src.enabled is True
        assert src.fov_degrees is None
        assert src.rotation_degrees == 0.0

    def test_full(self):
        src = CameraSource(
            source_id="cam-02",
            name="Front Door",
            source_type=CameraSourceType.RTSP,
            url="rtsp://192.168.1.100:554/stream",
            enabled=True,
            position=CameraPosition(lat=37.0, lng=-122.0, alt=5.0),
            fov_degrees=120.0,
            rotation_degrees=45.0,
        )
        assert src.name == "Front Door"
        assert src.source_type == CameraSourceType.RTSP
        assert src.url == "rtsp://192.168.1.100:554/stream"
        assert src.has_position is True
        assert src.fov_degrees == 120.0
        assert src.rotation_degrees == 45.0

    def test_has_position_false(self):
        src = CameraSource(source_id="cam-01")
        assert src.has_position is False

    def test_has_position_partial(self):
        src = CameraSource(
            source_id="cam-01",
            position=CameraPosition(lat=1.0),
        )
        assert src.has_position is False

    def test_all_source_types(self):
        for st in CameraSourceType:
            src = CameraSource(source_id="t", source_type=st)
            assert src.source_type == st

    def test_source_type_values(self):
        assert CameraSourceType.SYNTHETIC.value == "synthetic"
        assert CameraSourceType.RTSP.value == "rtsp"
        assert CameraSourceType.MJPEG.value == "mjpeg"
        assert CameraSourceType.MQTT.value == "mqtt"
        assert CameraSourceType.USB.value == "usb"

    def test_serialization_roundtrip(self):
        src = CameraSource(
            source_id="cam-03",
            name="Backyard",
            source_type=CameraSourceType.MJPEG,
            url="http://cam/mjpeg",
            position=CameraPosition(lat=1.0, lng=2.0),
            fov_degrees=90.0,
        )
        data = src.model_dump()
        restored = CameraSource.model_validate(data)
        assert restored == src

    def test_json_roundtrip(self):
        src = CameraSource(source_id="cam-04", name="Test")
        restored = CameraSource.model_validate_json(src.model_dump_json())
        assert restored == src

    def test_disabled_source(self):
        src = CameraSource(source_id="cam-05", enabled=False)
        assert src.enabled is False


# ── CameraFrame ─────────────────────────────────────────────────────

class TestCameraFrame:
    def test_defaults(self):
        frame = CameraFrame(source_id="cam-01")
        assert frame.width == 0
        assert frame.height == 0
        assert frame.format == CameraFrameFormat.JPEG
        assert frame.timestamp is None

    def test_full(self):
        now = datetime.now()
        frame = CameraFrame(
            source_id="cam-01",
            timestamp=now,
            width=1920,
            height=1080,
            format=CameraFrameFormat.JPEG,
        )
        assert frame.resolution == "1920x1080"
        assert frame.timestamp == now

    def test_rgb565_format(self):
        frame = CameraFrame(
            source_id="cam-01",
            width=320,
            height=240,
            format=CameraFrameFormat.RGB565,
        )
        assert frame.format == CameraFrameFormat.RGB565
        assert frame.resolution == "320x240"

    def test_serialization_roundtrip(self):
        frame = CameraFrame(source_id="cam-01", width=640, height=480)
        restored = CameraFrame.model_validate(frame.model_dump())
        assert restored == frame

    def test_json_roundtrip(self):
        frame = CameraFrame(source_id="cam-01", width=100, height=100)
        restored = CameraFrame.model_validate_json(frame.model_dump_json())
        assert restored == frame


# ── BoundingBox ──────────────────────────────────────────────────────

class TestBoundingBox:
    def test_defaults(self):
        bbox = BoundingBox()
        assert bbox.x == 0.0
        assert bbox.y == 0.0
        assert bbox.w == 0.0
        assert bbox.h == 0.0

    def test_area(self):
        bbox = BoundingBox(x=10, y=20, w=100, h=50)
        assert bbox.area == 5000.0

    def test_center(self):
        bbox = BoundingBox(x=0, y=0, w=100, h=200)
        assert bbox.center == (50.0, 100.0)

    def test_zero_area(self):
        bbox = BoundingBox(x=5, y=5, w=0, h=0)
        assert bbox.area == 0.0

    def test_serialization_roundtrip(self):
        bbox = BoundingBox(x=1.5, y=2.5, w=3.0, h=4.0)
        restored = BoundingBox.model_validate(bbox.model_dump())
        assert restored == bbox


# ── CameraDetection ─────────────────────────────────────────────────

class TestCameraDetection:
    def test_defaults(self):
        det = CameraDetection(source_id="cam-01")
        assert det.class_name == ""
        assert det.confidence == 0.0
        assert det.timestamp is None
        assert det.is_high_confidence is False

    def test_high_confidence(self):
        det = CameraDetection(
            source_id="cam-01",
            class_name="person",
            confidence=0.95,
        )
        assert det.is_high_confidence is True
        assert det.class_name == "person"

    def test_low_confidence(self):
        det = CameraDetection(source_id="cam-01", confidence=0.3)
        assert det.is_high_confidence is False

    def test_boundary_confidence(self):
        det = CameraDetection(source_id="cam-01", confidence=0.7)
        assert det.is_high_confidence is False  # > 0.7, not >=

    def test_full(self):
        now = datetime.now()
        det = CameraDetection(
            source_id="cam-02",
            class_name="vehicle",
            confidence=0.88,
            bbox=BoundingBox(x=100, y=200, w=50, h=30),
            timestamp=now,
        )
        assert det.bbox.area == 1500.0
        assert det.bbox.center == (125.0, 215.0)
        assert det.timestamp == now

    def test_serialization_roundtrip(self):
        det = CameraDetection(
            source_id="cam-01",
            class_name="drone",
            confidence=0.82,
            bbox=BoundingBox(x=10, y=20, w=30, h=40),
        )
        data = det.model_dump()
        restored = CameraDetection.model_validate(data)
        assert restored == det

    def test_json_roundtrip(self):
        det = CameraDetection(
            source_id="cam-01",
            class_name="cat",
            confidence=0.99,
        )
        restored = CameraDetection.model_validate_json(det.model_dump_json())
        assert restored == det


# ── Import from top-level ───────────────────────────────────────────

class TestTopLevelImport:
    def test_importable_from_models(self):
        from tritium_lib.models import (
            BoundingBox,
            CameraDetection,
            CameraFrame,
            CameraFrameFormat,
            CameraPosition,
            CameraSource,
            CameraSourceType,
        )
        assert CameraSource(source_id="x").source_id == "x"
