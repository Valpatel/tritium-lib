# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""End-to-end: depth16 wire bytes -> decode -> DepthCameraPipeline -> range.

The codec tests and the pipeline tests were independent islands: nothing
asserted that what comes off the wire feeds the pipeline in the units the
pipeline expects.  That seam is a 1000x trap.  ``decode_depth16_png`` returns
**metres**, so the consumer must pass ``depth_scale=1.0``; passing the 0.001
that raw uint16-millimetre input would need turns a 5 m contact into 5 mm and
drops a phantom on the operator's own position.

These tests pin the seam in the units an operator would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from tritium_lib.perception import (
    CameraWorldPose,
    CameraIntrinsics,
    decode_depth16_png,
    encode_depth16_png,
)
from tritium_lib.perception.depth_pipeline import (
    DepthCameraPipeline,
    process_depth_frame,
)

W, H = 64, 48
TRUE_RANGE_M = 7.25


class _OneBoxDetector:
    """Detects a single fixed box in the middle of the frame."""

    def detect(self, frame, source_id=""):
        return [{
            "class_name": "person",
            "confidence": 0.9,
            "bbox": {"x": 24, "y": 16, "w": 16, "h": 16},
        }]


class _RecordingTracker:
    def __init__(self):
        self.seen = []

    def update_from_detection(self, payload):
        self.seen.append(payload)
        return f"det_person_{len(self.seen)}"


def _wire_frame(range_m: float = TRUE_RANGE_M) -> bytes:
    """A depth16 PNG exactly as camera_server's /depth16 would emit it."""
    depth = np.full((H, W), range_m, dtype=np.float32)
    return encode_depth16_png(depth)


def _rgb() -> np.ndarray:
    return np.zeros((H, W, 3), dtype=np.uint8)


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics.from_fov(W, H, 60.0)


class TestWireUnitsReachThePipeline:
    def test_decoded_wire_depth_yields_true_range(self):
        """depth_scale=1.0 on decoded metres reproduces the true range."""
        depth_m = decode_depth16_png(_wire_frame())
        det = _OneBoxDetector()
        tracker = _RecordingTracker()
        process_depth_frame(
            _rgb(), depth_m, _intrinsics(), None, tracker, det,
            depth_scale=1.0,
        )
        assert tracker.seen, "detection never reached the tracker"
        # mm quantisation is the only error budget
        assert tracker.seen[0]["range_m"] == pytest.approx(TRUE_RANGE_M, abs=0.002)

    def test_wrong_scale_is_catastrophically_wrong(self):
        """Guard-rail: document what the 0.001 mistake actually costs.

        If someone 'fixes' the seam by passing the raw-millimetre scale to
        already-decoded metres, the contact lands ~1000x too close. This test
        fails loudly if the seam is ever rewired that way.
        """
        depth_m = decode_depth16_png(_wire_frame())
        tracker = _RecordingTracker()
        process_depth_frame(
            _rgb(), depth_m, _intrinsics(), None, tracker, _OneBoxDetector(),
            depth_scale=0.001,
        )
        assert tracker.seen[0]["range_m"] == pytest.approx(TRUE_RANGE_M / 1000.0, abs=1e-4)

    def test_pipeline_class_over_the_wire(self):
        """The class seam a live source uses, not just the function."""
        tracker = _RecordingTracker()
        pipe = DepthCameraPipeline(
            detector=_OneBoxDetector(), tracker=tracker,
            intrinsics=_intrinsics(), cam_id="isaac_depth",
            depth_scale=1.0, source="depth16",
        )
        ids = pipe.process(_rgb(), decode_depth16_png(_wire_frame()))
        assert ids and ids[0] is not None
        assert pipe.frames_processed == 1
        assert tracker.seen[0]["range_m"] == pytest.approx(TRUE_RANGE_M, abs=0.002)

    def test_all_hole_box_is_rejected_not_placed_at_zero(self):
        """A detection over pure no-return must yield NO contact at all.

        The 0-sentinel decodes to NaN, and with the ground fallback off the
        pipeline must drop the detection entirely rather than place it at 0 m
        on the operator's own lens. Asserts the tracker is never called — the
        weaker 'range_m != 0' form would pass vacuously here.
        """
        depth = np.full((H, W), TRUE_RANGE_M, dtype=np.float32)
        depth[16:32, 24:40] = np.nan  # the whole detection box is a hole
        tracker = _RecordingTracker()
        pipe = DepthCameraPipeline(
            detector=_OneBoxDetector(), tracker=tracker,
            intrinsics=_intrinsics(), depth_scale=1.0,
            ground_fallback=False,
        )
        ids = pipe.process(_rgb(), decode_depth16_png(encode_depth16_png(depth)))
        assert ids == [None], f"all-hole box should place no contact, got {ids}"
        assert tracker.seen == [], "a no-return box reached the tracker"

    @pytest.mark.parametrize("cam_id", ["isaac_cam", ""])
    def test_hole_box_still_placed_when_ground_fallback_is_on(self, cam_id):
        """Same frame, fallback ON: the contact survives via flat-ground.

        Complement of the test above — proves the rejection is the fallback
        flag talking, not the detection silently vanishing.

        Parametrised on ``cam_id`` because of a real defect this test caught:
        the fallback builds a ``CameraDetection`` to reuse the ground model,
        that model requires ``source_id`` min_length=1, and the builder passed
        ``cam_id or ""``. With the DEFAULT empty cam_id the construction raised,
        a bare ``except`` swallowed it, and the fallback was silently dead —
        so every depth dropout dropped the contact instead of degrading to
        flat-ground. An unnamed camera must degrade exactly like a named one.
        """
        depth = np.full((H, W), TRUE_RANGE_M, dtype=np.float32)
        depth[16:32, 24:40] = np.nan
        tracker = _RecordingTracker()
        pipe = DepthCameraPipeline(
            detector=_OneBoxDetector(), tracker=tracker,
            intrinsics=_intrinsics(), depth_scale=1.0, cam_id=cam_id,
            pose=CameraWorldPose(lat=37.0, lng=-122.0, heading_deg=0.0,
                                 pitch_deg=-20.0, height_m=3.0),
            ground_fallback=True,
        )
        ids = pipe.process(_rgb(), decode_depth16_png(encode_depth16_png(depth)))
        assert ids != [None], "ground fallback should still place the contact"
        assert tracker.seen, "fallback contact never reached the tracker"
        # Provenance must say 'guess', never 'measurement'.
        assert tracker.seen[0].get("depth_source") != "depth16"

    def test_far_geometry_saturates_rather_than_wrapping(self):
        """Beyond the uint16-mm ceiling, range must clamp, never wrap.

        A wrapped 70 m sky pixel reading as ~4.5 m would put a phantom contact
        in the operator's lap — strictly worse than reporting the ceiling.
        """
        depth_m = decode_depth16_png(_wire_frame(range_m=70.0))
        tracker = _RecordingTracker()
        process_depth_frame(
            _rgb(), depth_m, _intrinsics(), None, tracker, _OneBoxDetector(),
            depth_scale=1.0,
        )
        assert tracker.seen[0]["range_m"] == pytest.approx(65.535, abs=0.01)
