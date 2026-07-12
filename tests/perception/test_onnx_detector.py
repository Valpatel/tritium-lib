# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""ONNX-YOLO detector backend — letterbox/decode math + graceful fallback.

The letterbox / decode / NMS / resolve tests are pure math: no model file,
no onnxruntime, no network — they run everywhere.  The real-model
integration test runs only when onnxruntime is importable AND
TRITIUM_YOLO_ONNX points at an existing .onnx file (no absolute model
paths are hardcoded here — env var only).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tritium_lib.perception import (
    BackgroundMotionDetector,
    available_backends,
    build_frame_detector,
    decode_yolo_predictions,
    letterbox_frame,
    onnx_available,
    resolve_onnx_model,
)

FIXTURES = Path(__file__).parent / "fixtures"

# Reference letterbox geometry used by the decode tests: a 320x240 frame
# into a 640 square -> ratio 2.0, no x pad, 80 px top pad.
RATIO, PAD_X, PAD_Y = 2.0, 0, 80


# ------------------------------------------------------------------ helpers

def _v8_tensor(
    anchor: int,
    cx: float, cy: float, w: float, h: float,
    cls_id: int, score: float,
    nc: int = 80, anchors: int = 8400,
) -> np.ndarray:
    """A zero (1, 4+nc, anchors) YOLOv8 tensor with ONE planted box."""
    pred = np.zeros((1, 4 + nc, anchors), dtype=np.float32)
    pred[0, 0, anchor] = cx
    pred[0, 1, anchor] = cy
    pred[0, 2, anchor] = w
    pred[0, 3, anchor] = h
    pred[0, 4 + cls_id, anchor] = score
    return pred


def _v5_tensor(
    anchor: int,
    cx: float, cy: float, w: float, h: float,
    obj: float, cls_id: int, cls_score: float,
    nc: int = 80, anchors: int = 25200,
) -> np.ndarray:
    """A zero (1, anchors, 5+nc) YOLOv5 tensor with ONE planted box."""
    pred = np.zeros((1, anchors, 5 + nc), dtype=np.float32)
    pred[0, anchor, 0:4] = (cx, cy, w, h)
    pred[0, anchor, 4] = obj
    pred[0, anchor, 5 + cls_id] = cls_score
    return pred


# ------------------------------------------------------------------ letterbox

class TestLetterboxFrame:
    def test_640x480_pads_top_and_bottom(self):
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        blob, ratio, pad_x, pad_y = letterbox_frame(frame, 640)
        assert blob.shape == (1, 3, 640, 640)
        assert blob.dtype == np.float32
        assert ratio == pytest.approx(1.0)
        assert (pad_x, pad_y) == (0, 80)
        assert float(blob.min()) >= 0.0
        assert float(blob.max()) <= 1.0
        # Pad rows carry the YOLO-standard grey (114).
        assert float(blob[0, 0, 0, 0]) == pytest.approx(114 / 255.0)
        # Image region carries the frame value.
        assert float(blob[0, 0, 320, 320]) == pytest.approx(200 / 255.0)

    def test_320x240_upscales_by_two(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        _, ratio, pad_x, pad_y = letterbox_frame(frame, 640)
        assert ratio == pytest.approx(2.0)
        assert (pad_x, pad_y) == (0, 80)

    def test_tall_frame_pads_left_and_right(self):
        frame = np.zeros((640, 320, 3), dtype=np.uint8)
        blob, ratio, pad_x, pad_y = letterbox_frame(frame, 640)
        assert blob.shape == (1, 3, 640, 640)
        assert ratio == pytest.approx(1.0)
        assert (pad_x, pad_y) == (160, 0)

    def test_square_frame_no_pads(self):
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        _, ratio, pad_x, pad_y = letterbox_frame(frame, 640)
        assert ratio == pytest.approx(2.0)
        assert (pad_x, pad_y) == (0, 0)


# ------------------------------------------------------------------ decode v8

class TestDecodeV8Layout:
    def test_single_box_unletterboxed_to_original_coords(self):
        # Original box x=50, y=30, w=40, h=60 on a 320x240 frame ->
        # letterbox coords cx=140, cy=200, w=80, h=120 (ratio 2, pad_y 80).
        pred = _v8_tensor(100, 140.0, 200.0, 80.0, 120.0, cls_id=0, score=0.9)
        out = decode_yolo_predictions(pred, RATIO, PAD_X, PAD_Y, 0.4, 0.45)
        assert len(out) == 1
        cls_id, conf, x, y, w, h = out[0]
        assert cls_id == 0
        assert conf == pytest.approx(0.9, abs=1e-5)
        assert (x, y, w, h) == pytest.approx((50.0, 30.0, 40.0, 60.0), abs=1e-3)

    def test_accepts_pred_without_batch_dim(self):
        pred = _v8_tensor(7, 140.0, 200.0, 80.0, 120.0, cls_id=2, score=0.8)[0]
        out = decode_yolo_predictions(pred, RATIO, PAD_X, PAD_Y, 0.4, 0.45)
        assert len(out) == 1
        assert out[0][0] == 2

    def test_below_threshold_dropped(self):
        pred = _v8_tensor(5, 140.0, 200.0, 80.0, 120.0, cls_id=0, score=0.2)
        assert decode_yolo_predictions(pred, RATIO, PAD_X, PAD_Y, 0.4, 0.45) == []

    def test_all_zeros_empty(self):
        pred = np.zeros((1, 84, 8400), dtype=np.float32)
        assert decode_yolo_predictions(pred, RATIO, PAD_X, PAD_Y, 0.4, 0.45) == []

    def test_normalized_coordinate_export_rescaled(self):
        # Some ONNX exports emit cx/cy/w/h normalized to [0, 1] instead of
        # letterbox pixels (the shipped yolov8n.onnx does).  Same planted
        # box as test_single_box_unletterboxed_to_original_coords, divided
        # by the 640 input size -> must decode to identical original coords.
        pred = _v8_tensor(
            100, 140.0 / 640, 200.0 / 640, 80.0 / 640, 120.0 / 640,
            cls_id=0, score=0.9,
        )
        out = decode_yolo_predictions(
            pred, RATIO, PAD_X, PAD_Y, 0.4, 0.45, input_size=640,
        )
        assert len(out) == 1
        cls_id, conf, x, y, w, h = out[0]
        assert cls_id == 0
        assert conf == pytest.approx(0.9, abs=1e-5)
        assert (x, y, w, h) == pytest.approx((50.0, 30.0, 40.0, 60.0), abs=1e-3)


# ------------------------------------------------------------------ decode v5

class TestDecodeV5Layout:
    def test_objectness_times_class_score(self):
        pred = _v5_tensor(
            500, 140.0, 200.0, 80.0, 120.0, obj=0.8, cls_id=0, cls_score=0.9,
        )
        out = decode_yolo_predictions(pred, RATIO, PAD_X, PAD_Y, 0.4, 0.45)
        assert len(out) == 1
        cls_id, conf, x, y, w, h = out[0]
        assert cls_id == 0
        assert conf == pytest.approx(0.8 * 0.9, abs=1e-5)  # obj * cls
        assert (x, y, w, h) == pytest.approx((50.0, 30.0, 40.0, 60.0), abs=1e-3)

    def test_high_cls_low_obj_dropped(self):
        # cls 0.9 but obj 0.3 -> 0.27 < 0.4 threshold: the multiplication
        # (not the raw class score) must gate the detection.
        pred = _v5_tensor(
            42, 140.0, 200.0, 80.0, 120.0, obj=0.3, cls_id=0, cls_score=0.9,
        )
        assert decode_yolo_predictions(pred, RATIO, PAD_X, PAD_Y, 0.4, 0.45) == []

    def test_accepts_pred_without_batch_dim(self):
        pred = _v5_tensor(
            9, 140.0, 200.0, 80.0, 120.0, obj=0.9, cls_id=5, cls_score=0.9,
        )[0]
        out = decode_yolo_predictions(pred, RATIO, PAD_X, PAD_Y, 0.4, 0.45)
        assert len(out) == 1
        assert out[0][0] == 5


# ------------------------------------------------------------------ NMS

class TestNms:
    def test_overlapping_same_class_one_survivor(self):
        pred = np.zeros((1, 84, 8400), dtype=np.float32)
        # Two heavily-overlapping class-0 boxes (4 px apart, 80 px wide).
        for anchor, cx, conf in ((10, 140.0, 0.9), (11, 144.0, 0.8)):
            pred[0, 0, anchor] = cx
            pred[0, 1, anchor] = 200.0
            pred[0, 2, anchor] = 80.0
            pred[0, 3, anchor] = 120.0
            pred[0, 4, anchor] = conf
        out = decode_yolo_predictions(pred, RATIO, PAD_X, PAD_Y, 0.4, 0.45)
        assert len(out) == 1
        assert out[0][1] == pytest.approx(0.9, abs=1e-5)  # strongest survives

    def test_distant_same_class_both_survive(self):
        pred = np.zeros((1, 84, 8400), dtype=np.float32)
        for anchor, cx in ((10, 140.0), (11, 400.0)):
            pred[0, 0, anchor] = cx
            pred[0, 1, anchor] = 200.0
            pred[0, 2, anchor] = 80.0
            pred[0, 3, anchor] = 120.0
            pred[0, 4, anchor] = 0.9
        out = decode_yolo_predictions(pred, RATIO, PAD_X, PAD_Y, 0.4, 0.45)
        assert len(out) == 2


# ------------------------------------------------------------------ resolve

class TestResolveOnnxModel:
    def test_explicit_existing_wins_over_env(self, monkeypatch, tmp_path):
        explicit = tmp_path / "explicit.onnx"
        explicit.write_bytes(b"stub")
        env_model = tmp_path / "env.onnx"
        env_model.write_bytes(b"stub")
        monkeypatch.setenv("TRITIUM_YOLO_ONNX", str(env_model))
        assert resolve_onnx_model(str(explicit)) == str(explicit)

    def test_explicit_missing_falls_to_env(self, monkeypatch, tmp_path):
        env_model = tmp_path / "env.onnx"
        env_model.write_bytes(b"stub")
        monkeypatch.setenv("TRITIUM_YOLO_ONNX", str(env_model))
        assert resolve_onnx_model("/nonexistent/model.onnx") == str(env_model)

    def test_env_pointing_at_missing_file_is_none(self, monkeypatch):
        monkeypatch.setenv("TRITIUM_YOLO_ONNX", "/nonexistent/env.onnx")
        assert resolve_onnx_model(None) is None

    def test_nothing_set_is_none(self, monkeypatch):
        monkeypatch.delenv("TRITIUM_YOLO_ONNX", raising=False)
        assert resolve_onnx_model() is None


# ------------------------------------------------------------------ factory

class TestFactoryFallback:
    def test_onnx_prefer_with_missing_model_falls_to_motion(self, monkeypatch):
        monkeypatch.delenv("TRITIUM_YOLO_ONNX", raising=False)
        det = build_frame_detector(
            prefer="onnx", onnx_model_path="/nonexistent/model.onnx",
        )
        assert isinstance(det, BackgroundMotionDetector)
        assert det.backend_name == "motion"

    def test_available_backends_reports_onnx(self):
        caps = available_backends()
        assert "onnx" in caps
        assert caps["onnx"] == onnx_available()


# ------------------------------------------------------------------ real model

_ENV_MODEL = os.environ.get("TRITIUM_YOLO_ONNX", "")


@pytest.mark.skipif(
    not onnx_available() or not (_ENV_MODEL and os.path.isfile(_ENV_MODEL)),
    reason="needs onnxruntime + TRITIUM_YOLO_ONNX pointing at a real model",
)
class TestRealModelIntegration:
    def test_street_scene_persons_and_bus(self):
        import cv2

        from tritium_lib.perception import OnnxYoloDetector

        frame = cv2.imread(str(FIXTURES / "street_scene.jpg"))
        assert frame is not None, "street_scene.jpg fixture missing/unreadable"
        frame_h, frame_w = frame.shape[:2]

        det = OnnxYoloDetector(_ENV_MODEL)
        assert det.backend_name.startswith("onnx:")

        dets = det.detect(frame, source_id="cam-test")
        names = [d.class_name for d in dets]
        assert names.count("person") >= 2, f"expected >=2 persons, got {names}"
        assert names.count("bus") >= 1, f"expected >=1 bus, got {names}"

        for d in dets:
            assert d.confidence >= 0.4
            assert d.source_id == "cam-test"
            assert 0.0 <= d.bbox.x <= frame_w
            assert 0.0 <= d.bbox.y <= frame_h
            assert d.bbox.x + d.bbox.w <= frame_w + 1e-6
            assert d.bbox.y + d.bbox.h <= frame_h + 1e-6

        confs = [d.confidence for d in dets]
        assert confs == sorted(confs, reverse=True)
