# Copyright (c) 2026 Matthew Valancy / Valpatel Software LLC
# SPDX-License-Identifier: AGPL-3.0-only
"""Metric depth transport — uint16 PNG in millimetres (the ROS ``16UC1``
convention shared by RealSense, ZED, Kinect and ``depth_image_proc``).

The whole point of this codec is that RANGE SURVIVES THE WIRE. A colorized
depth JPEG looks the same to an operator but is metrically dead: the colormap
is many-to-one and JPEG is lossy, so a consumer cannot recover metres from it.
These tests pin the property that matters downstream — decode(encode(d)) is d
back, to millimetre quantization — because that is what lets
``DepthCameraPipeline`` place a detection at a true range instead of guessing
flat ground.
"""

import numpy as np
import pytest

from tritium_lib.perception.depth_codec import (
    DEPTH_SCALE_MM,
    decode_depth16_png,
    encode_depth16_png,
)


def _ramp(h: int = 24, w: int = 32, near: float = 0.5, far: float = 40.0):
    """A horizontal metre ramp — the simplest depth image with real structure."""
    row = np.linspace(near, far, w, dtype=np.float32)
    return np.repeat(row[None, :], h, axis=0)


class TestRoundTrip:
    def test_round_trip_preserves_metres_to_mm(self):
        depth = _ramp()
        out = decode_depth16_png(encode_depth16_png(depth))
        assert out.shape == depth.shape
        assert out.dtype == np.float32
        # Quantization is 1 mm; allow a shade over half a step for rounding.
        assert np.max(np.abs(out - depth)) <= 0.001

    def test_scalar_range_survives(self):
        """A single known distance comes back as that distance — the claim an
        operator actually relies on when a target is called out at 12.5 m."""
        depth = np.full((4, 4), 12.5, dtype=np.float32)
        assert decode_depth16_png(encode_depth16_png(depth)) == pytest.approx(12.5, abs=0.001)

    def test_encoding_is_a_real_png(self):
        blob = encode_depth16_png(_ramp())
        assert blob[:8] == b"\x89PNG\r\n\x1a\n"

    def test_lossless_across_repeated_generations(self):
        """Re-encoding a decoded frame must not drift — PNG is lossless, so a
        relay hop (Isaac -> connector -> SC) cannot silently erode range."""
        first = encode_depth16_png(_ramp())
        second = encode_depth16_png(decode_depth16_png(first))
        assert first == second


class TestInvalidReturns:
    def test_nan_and_inf_become_zero_no_return(self):
        """0 is the 16UC1 'no return' sentinel — sky, glass, out-of-range."""
        depth = np.array([[np.nan, np.inf, 5.0, -np.inf]], dtype=np.float32)
        out = decode_depth16_png(encode_depth16_png(depth))
        assert np.isnan(out[0, 0]) and np.isnan(out[0, 1]) and np.isnan(out[0, 3])
        assert out[0, 2] == pytest.approx(5.0, abs=0.001)

    def test_zero_and_negative_are_no_return(self):
        """A true 0.0 m reading is not physical for a depth camera; both it and
        negatives fold onto the no-return sentinel rather than lying as 0 m."""
        out = decode_depth16_png(encode_depth16_png(np.array([[0.0, -3.0]], dtype=np.float32)))
        assert np.isnan(out).all()


class TestSaturation:
    def test_beyond_range_saturates_not_wraps(self):
        """65535 mm is ~65.5 m. Anything past it must CLAMP, never wrap around
        to a near value — a wrapped 70 m sky pixel reading as 4 m would put a
        phantom contact in the operator's lap."""
        depth = np.array([[70.0, 1000.0]], dtype=np.float32)
        out = decode_depth16_png(encode_depth16_png(depth))
        assert np.all(out >= 65.0)

    def test_custom_scale_extends_range(self):
        """Centimetre scale trades resolution for a ~655 m ceiling — the knob a
        LiDAR-range consumer needs."""
        depth = np.array([[300.0]], dtype=np.float32)
        out = decode_depth16_png(encode_depth16_png(depth, scale=100.0), scale=100.0)
        assert out[0, 0] == pytest.approx(300.0, abs=0.01)


class TestContract:
    def test_default_scale_is_millimetres(self):
        assert DEPTH_SCALE_MM == 1000.0

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            encode_depth16_png(np.zeros((4, 4, 3), dtype=np.float32))

    def test_decode_rejects_non_png(self):
        with pytest.raises(ValueError):
            decode_depth16_png(b"\xff\xd8not a png\xff\xd9")
