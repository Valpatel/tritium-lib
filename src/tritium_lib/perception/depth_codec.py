# Copyright (c) 2026 Matthew Valancy / Valpatel Software LLC
# SPDX-License-Identifier: AGPL-3.0-only
"""Metric depth transport — uint16 PNG in millimetres.

A depth camera's whole value is the *number*: this pixel is 12.5 m away. The
usual way to stream depth (colorize it and send MJPEG) destroys that number —
the colormap is many-to-one and JPEG is lossy, so the receiver gets a picture
of depth rather than depth. That is fine for a human looking at a panel and
useless for :mod:`tritium_lib.perception.depth_pipeline`, which needs metres to
place a detection at a true range instead of assuming flat ground.

This module carries the number instead, using the convention the rest of the
robotics world already agreed on: **ROS ``16UC1``** — a single-channel uint16
image holding millimetres, with **0 as the "no return" sentinel** (sky, glass,
absorbed beam, out of range). RealSense, ZED, Kinect/OpenNI and ROS'
``depth_image_proc`` all speak it, so an Isaac depth frame and a real ZED frame
arrive at the perception stack in the same units with the same holes. PNG
because it is lossless, universally decodable, and compresses depth well.

Deliberate design points:

* **Saturation, never wrap.** Beyond the representable ceiling (~65.5 m at mm
  scale) values clamp to the max. A wrapped sky pixel reading as 4 m would put
  a phantom contact in the operator's lap.
* **Invalid folds to 0, and 0 decodes to NaN** — not to 0 m. A consumer that
  forgets to mask holes gets loud NaNs rather than a silent wall of contacts
  standing on the lens.
* ``scale`` is the divisor, so ``scale=100`` (centimetres) trades resolution
  for a ~655 m ceiling when a LiDAR-range consumer needs it.

Encoder/decoder are pure numpy + an image codec (cv2 when present, else
Pillow), keeping this importable on a bare aarch64 Jetson.
"""

from __future__ import annotations

import io

import numpy as np

#: Millimetres — the ROS ``16UC1`` default and what every consumer assumes
#: unless a frame explicitly says otherwise.
DEPTH_SCALE_MM = 1000.0

#: uint16 ceiling. At mm scale this is 65.535 m.
_UINT16_MAX = 65535

#: The 16UC1 "no return" sentinel.
_NO_RETURN = 0

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def encode_depth16_png(depth: np.ndarray, scale: float = DEPTH_SCALE_MM) -> bytes:
    """Encode a HxW float depth image (metres) as a uint16 PNG.

    ``NaN``/``inf``/zero/negative all become the 0 no-return sentinel; finite
    values are rounded to ``1/scale`` m and clamped at the uint16 ceiling.
    """
    d = np.asarray(depth)
    if d.ndim != 2:
        raise ValueError(f"depth must be a 2-D HxW array of metres, got shape {d.shape}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    d = d.astype(np.float32, copy=True)
    # Anything not a real forward distance is a hole, not a reading.
    invalid = ~np.isfinite(d) | (d <= 0.0)
    d[invalid] = 0.0

    units = np.rint(d * scale)
    # Clamp BEFORE the uint16 cast — casting an out-of-range float wraps.
    units = np.clip(units, 0, _UINT16_MAX)
    # A valid-but-sub-half-unit reading would round to 0 and masquerade as a
    # hole; lift it to the smallest representable distance instead.
    units[(units == 0) & ~invalid] = 1
    units[invalid] = _NO_RETURN

    return _encode_png16(units.astype(np.uint16))


def decode_depth16_png(blob: bytes, scale: float = DEPTH_SCALE_MM) -> np.ndarray:
    """Decode a uint16 PNG back to a HxW float32 depth image in metres.

    No-return pixels (0) decode to ``NaN`` so a consumer that forgets to mask
    them fails loudly rather than reporting contacts at 0 m.
    """
    if not blob[:8] == _PNG_MAGIC:
        raise ValueError("not a PNG — depth16 frames must be lossless PNG")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    units = _decode_png16(blob)
    metres = units.astype(np.float32) / np.float32(scale)
    metres[units == _NO_RETURN] = np.nan
    return metres


def colorize_depth_bgr(
    depth: np.ndarray, near: float = 0.5, far: float = 60.0,
) -> np.ndarray:
    """Render HxW float depth (metres) as an HxWx3 uint8 **BGR** image.

    This is the *picture* side of the contract this module exists to protect:
    a viewable tile for the operator, produced from — never instead of — the
    metric frame.  The colormap is many-to-one and whatever encodes it is
    usually lossy, so a consumer that needs the number must use
    :func:`decode_depth16_png`, not this.

    ``BGR`` is in the name deliberately.  The channel order of a depth
    colormap is invisible on inspection — a near/far ramp looks equally
    plausible either way round — so a silent RGB/BGR swap survives review and
    shows up only as an operator wondering why close things are blue.  Callers
    wanting RGB reverse the last axis at the call site, where it is legible.

    ``NaN``/``inf`` (no-return: sky, glass, absorbed beam) render at ``far``:
    a hole is unknown-therefore-distant, and rendering it near would paint a
    wall of phantom close contacts across the sky.
    """
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"depth must be a 2-D HxW array of metres, got shape {d.shape}")
    if not far > near:
        raise ValueError(f"far must exceed near, got near={near} far={far}")

    d = np.nan_to_num(d, nan=far, posinf=far, neginf=far)
    d = np.clip(d, near, far)
    inv = 1.0 - (d - near) / (far - near)          # 1.0 near .. 0.0 far
    gray = (inv * 255.0).astype(np.uint8)
    try:
        import cv2
    except ImportError:
        return np.repeat(gray[:, :, None], 3, axis=2)
    return np.ascontiguousarray(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO))


# --------------------------------------------------------------------------- #
# Image codec — cv2 when available (fast), Pillow otherwise.  The two emit
# DIFFERENT (both valid) PNG byte streams for the same pixels — compression
# differs — but each must decode the other's blob to identical uint16 VALUES,
# so a relay hop cannot drift metrically.  Value identity is the contract;
# byte identity holds only within one codec (pinned by the determinism suite).
# --------------------------------------------------------------------------- #

def _encode_png16(units: np.ndarray) -> bytes:
    try:
        import cv2
    except ImportError:
        pass
    else:
        ok, buf = cv2.imencode(".png", units)
        if not ok:
            raise ValueError("cv2 failed to encode the depth PNG")
        return buf.tobytes()

    from PIL import Image

    out = io.BytesIO()
    # A uint16 array auto-selects Pillow's 16-bit single-channel mode
    # ("I;16").  Passing mode= explicitly is deprecated and removed in
    # Pillow 13 — on a cv2-less Jetson that removal would have killed the
    # whole depth path.
    Image.fromarray(units).save(out, format="PNG")
    return out.getvalue()


def _decode_png16(blob: bytes) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        pass
    else:
        arr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError("cv2 failed to decode the depth PNG")
        if arr.ndim != 2:
            raise ValueError(f"depth PNG must be single-channel, got shape {arr.shape}")
        return arr.astype(np.uint16, copy=False)

    from PIL import Image

    img = Image.open(io.BytesIO(blob))
    arr = np.array(img)
    if arr.ndim != 2:
        raise ValueError(f"depth PNG must be single-channel, got shape {arr.shape}")
    return arr.astype(np.uint16, copy=False)
