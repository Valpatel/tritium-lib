# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.perception.depth — bbox range + pinhole deprojection.

Uses synthetic depth ramps with analytically-known values so every
assertion has a closed-form expectation.  This is the production half of
Isaac Sim depth-stream consumption: the same math runs against a real
RGB-D sensor.
"""

from __future__ import annotations

import numpy as np
import pytest

from tritium_lib.models.camera import BoundingBox, CameraDetection
from tritium_lib.perception.depth import (
    CameraIntrinsics,
    deproject_pixel,
    enrich_detections_with_depth,
    range_for_bbox,
)


def make_ramp(h: int = 100, w: int = 100) -> np.ndarray:
    """Depth ramp: depth[v, u] = 1 + 0.1 * u metres (constant per column)."""
    u = np.arange(w, dtype=np.float64)
    return np.tile(1.0 + 0.1 * u, (h, 1))


K = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100)


# ------------------------------------------------------------- range_for_bbox

class TestRangeForBbox:
    def test_constant_patch_exact(self):
        depth = np.full((100, 100), 7.5)
        rng = range_for_bbox(depth, BoundingBox(x=10, y=10, w=20, h=20))
        assert rng == pytest.approx(7.5)

    def test_ramp_median_at_bbox_center(self):
        # bbox centred at u=40 -> median column depth = 1 + 0.1*40 = 5.0
        depth = make_ramp()
        rng = range_for_bbox(depth, BoundingBox(x=30, y=30, w=20, h=20))
        assert rng == pytest.approx(5.0, abs=0.15)

    def test_bbox_tuple_and_dict_forms(self):
        depth = np.full((50, 50), 3.0)
        assert range_for_bbox(depth, (5, 5, 10, 10)) == pytest.approx(3.0)
        assert range_for_bbox(
            depth, {"x": 5, "y": 5, "w": 10, "h": 10}
        ) == pytest.approx(3.0)

    def test_invalid_pixels_ignored(self):
        # Center region peppered with 0 / NaN / inf; valid pixels all 4.0.
        depth = np.full((60, 60), 4.0)
        depth[20:40:3, 20:40:3] = 0.0
        depth[21:40:4, 21:40:4] = np.nan
        depth[22:40:5, 22:40:5] = np.inf
        rng = range_for_bbox(depth, BoundingBox(x=20, y=20, w=20, h=20))
        assert rng == pytest.approx(4.0)

    def test_all_invalid_returns_none(self):
        depth = np.zeros((40, 40))
        assert range_for_bbox(depth, BoundingBox(x=5, y=5, w=10, h=10)) is None
        depth_nan = np.full((40, 40), np.nan)
        assert range_for_bbox(depth_nan, BoundingBox(x=5, y=5, w=10, h=10)) is None

    def test_bbox_fully_off_frame_returns_none(self):
        depth = np.full((40, 40), 2.0)
        assert range_for_bbox(depth, BoundingBox(x=100, y=100, w=10, h=10)) is None
        assert range_for_bbox(depth, BoundingBox(x=-30, y=-30, w=10, h=10)) is None

    def test_degenerate_bbox_returns_none(self):
        depth = np.full((40, 40), 2.0)
        assert range_for_bbox(depth, BoundingBox(x=5, y=5, w=0, h=10)) is None
        assert range_for_bbox(depth, BoundingBox(x=5, y=5, w=10, h=0)) is None

    def test_none_and_garbage_frames(self):
        box = BoundingBox(x=5, y=5, w=10, h=10)
        assert range_for_bbox(None, box) is None
        assert range_for_bbox(np.zeros((0, 0)), box) is None
        assert range_for_bbox(np.zeros(10), box) is None  # 1-D
        assert range_for_bbox("not a frame", box) is None

    def test_hxwx1_frame_accepted(self):
        depth = np.full((40, 40, 1), 6.0)
        rng = range_for_bbox(depth, BoundingBox(x=10, y=10, w=10, h=10))
        assert rng == pytest.approx(6.0)

    def test_depth_scale_millimetres(self):
        depth = np.full((40, 40), 2500, dtype=np.uint16)  # 2.5 m in mm
        rng = range_for_bbox(
            depth, BoundingBox(x=10, y=10, w=10, h=10), depth_scale=0.001
        )
        assert rng == pytest.approx(2.5)

    def test_inner_sampling_prefers_object_over_background(self):
        # Object (3 m) fills the central half; background (30 m) surrounds it.
        depth = np.full((100, 100), 30.0)
        depth[35:65, 35:65] = 3.0
        rng = range_for_bbox(depth, BoundingBox(x=25, y=25, w=50, h=50))
        assert rng == pytest.approx(3.0, abs=0.01)

    def test_percentile_bias_near(self):
        depth = make_ramp()
        near = range_for_bbox(
            depth, BoundingBox(x=20, y=20, w=40, h=40), percentile=10.0
        )
        far = range_for_bbox(
            depth, BoundingBox(x=20, y=20, w=40, h=40), percentile=90.0
        )
        assert near is not None and far is not None
        assert near < far


# ------------------------------------------------------------ deproject_pixel

class TestDeprojectPixel:
    def test_principal_point_on_axis(self):
        xyz = deproject_pixel(50.0, 50.0, 4.0, K)
        assert xyz == pytest.approx((0.0, 0.0, 4.0))

    def test_known_offsets(self):
        # (u-cx)=10 px at fx=100 and depth 2 m -> x = 0.2 m; same for y.
        xyz = deproject_pixel(60.0, 50.0, 2.0, K)
        assert xyz == pytest.approx((0.2, 0.0, 2.0))
        xyz = deproject_pixel(50.0, 30.0, 2.0, K)
        assert xyz == pytest.approx((0.0, -0.4, 2.0))

    def test_intrinsics_dict_and_sequence_forms(self):
        as_dict = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}
        as_seq = (100.0, 100.0, 50.0, 50.0)
        assert deproject_pixel(60, 50, 2.0, as_dict) == pytest.approx((0.2, 0.0, 2.0))
        assert deproject_pixel(60, 50, 2.0, as_seq) == pytest.approx((0.2, 0.0, 2.0))

    def test_from_fov_round_trip(self):
        # 90 deg hfov at 100 px wide -> fx = 50 / tan(45) = 50.
        k = CameraIntrinsics.from_fov(100, 100, 90.0)
        assert k.fx == pytest.approx(50.0)
        xyz = deproject_pixel(100.0, 50.0, 5.0, k)
        # Right edge at 90 deg fov -> x == z at the half-fov angle: x = 5.0
        assert xyz == pytest.approx((5.0, 0.0, 5.0))

    def test_invalid_inputs_return_none(self):
        assert deproject_pixel(50, 50, 0.0, K) is None
        assert deproject_pixel(50, 50, -1.0, K) is None
        assert deproject_pixel(50, 50, float("nan"), K) is None
        assert deproject_pixel(50, 50, float("inf"), K) is None
        assert deproject_pixel(50, 50, 2.0, None) is None
        assert deproject_pixel(50, 50, 2.0, {"fx": 0.0, "fy": 1, "cx": 0, "cy": 0}) is None
        assert deproject_pixel(float("nan"), 50, 2.0, K) is None


# ------------------------------------- enrich_detections_with_depth

def _det(x: float, y: float, w: float, h: float) -> CameraDetection:
    return CameraDetection(
        source_id="cam-test",
        class_name="person",
        confidence=0.9,
        bbox=BoundingBox(x=x, y=y, w=w, h=h),
    )


class TestEnrichDetections:
    def test_annotates_range_and_xyz(self):
        depth = np.full((100, 100), 5.0)
        dets = [_det(50, 40, 20, 20)]  # centre pixel (60, 50)
        out = enrich_detections_with_depth(dets, depth, K)
        assert out is dets
        assert out[0].range_m == pytest.approx(5.0)
        # (60-50)*5/100 = 0.5 right, (50-50) -> 0 down, z = 5.
        assert out[0].camera_xyz == pytest.approx((0.5, 0.0, 5.0))

    def test_ramp_ranges_per_detection(self):
        depth = make_ramp()
        dets = [_det(10, 40, 20, 20), _det(60, 40, 20, 20)]  # centres u=20, u=70
        out = enrich_detections_with_depth(dets, depth, K)
        assert out[0].range_m == pytest.approx(3.0, abs=0.15)   # 1 + 0.1*20
        assert out[1].range_m == pytest.approx(8.0, abs=0.15)   # 1 + 0.1*70
        assert out[0].range_m < out[1].range_m

    def test_no_intrinsics_gives_range_only(self):
        depth = np.full((100, 100), 5.0)
        dets = [_det(40, 40, 20, 20)]
        out = enrich_detections_with_depth(dets, depth, None)
        assert out[0].range_m == pytest.approx(5.0)
        assert out[0].camera_xyz is None

    def test_none_depth_passthrough_unchanged(self):
        dets = [_det(40, 40, 20, 20)]
        out = enrich_detections_with_depth(dets, None, K)
        assert out is dets
        assert out[0].range_m is None
        assert out[0].camera_xyz is None

    def test_shape_mismatch_passthrough_unchanged(self):
        dets = [_det(40, 40, 20, 20)]
        for bad in (np.zeros(10), np.zeros((2, 2, 3)), np.zeros((0, 0)), "junk"):
            out = enrich_detections_with_depth(dets, bad, K)
            assert out is dets
            assert out[0].range_m is None

    def test_invalid_depth_region_leaves_detection_untouched(self):
        depth = np.full((100, 100), 5.0)
        depth[30:70, 30:70] = 0.0  # dead zone under the first bbox
        dets = [_det(40, 40, 20, 20), _det(0, 0, 20, 20)]
        out = enrich_detections_with_depth(dets, depth, K)
        assert out[0].range_m is None            # no fabricated data
        assert out[1].range_m == pytest.approx(5.0)

    def test_empty_list_passthrough(self):
        assert enrich_detections_with_depth([], np.ones((10, 10)), K) == []

    def test_dict_detections_supported(self):
        depth = np.full((100, 100), 2.0)
        dets = [{"bbox": {"x": 40, "y": 40, "w": 20, "h": 20}, "class_name": "car"}]
        out = enrich_detections_with_depth(dets, depth, K)
        assert out[0]["range_m"] == pytest.approx(2.0)
        assert out[0]["camera_xyz"] == pytest.approx((0.0, 0.0, 2.0))

    def test_model_serializes_depth_fields(self):
        depth = np.full((100, 100), 5.0)
        dets = enrich_detections_with_depth([_det(40, 40, 20, 20)], depth, K)
        payload = dets[0].model_dump()
        assert payload["range_m"] == pytest.approx(5.0)
        assert tuple(payload["camera_xyz"]) == pytest.approx((0.0, 0.0, 5.0))
