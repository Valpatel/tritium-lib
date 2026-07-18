# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Depth-aware perception primitives — bbox range + pinhole deprojection.

A 2D detector answers *what* and *where in the image*; a depth stream
answers *how far*.  This module is the pure-numpy seam that fuses the two:
given a per-pixel depth frame (metres) aligned with the RGB frame the
detector ran on, it attaches a robust range and a camera-frame 3D point to
each detection.

Producers of the depth frame are interchangeable by design:

  * **Isaac Sim** ``distance_to_image_plane`` / depth annotator frames
    published by the ``tritium-addons/isaac_sim`` bridge (the FUN half —
    the digital twin's virtual depth camera).
  * A **real RGB-D sensor** (RealSense, OAK-D, ZED, a robot's stereo pair)
    on the production track — same math, different frame source.

Conventions:

  * ``depth_frame`` is an ``HxW`` (or ``HxWx1``) array of ranges **in
    metres** along the optical axis, aligned pixel-for-pixel with the RGB
    frame the detections came from.  Invalid pixels are ``0``, ``NaN``, or
    ``inf`` (the union of Isaac / RealSense / stereo no-return encodings).
    Sensors that report integer millimetres pass ``depth_scale=0.001``.
  * Camera-frame axes follow the OpenCV / ROS ``optical_frame`` pinhole
    convention: **+x right, +y down, +z forward** (z == depth).

Everything here is numpy + stdlib — no torch, no isaac, no ros — so it
imports on a bare Jetson exactly like the rest of the lib.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger("tritium.perception.depth")

__all__ = [
    "CameraIntrinsics",
    "range_for_bbox",
    "deproject_pixel",
    "enrich_detections_with_depth",
]


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics (pixels).

    Attributes:
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point (optical centre) in pixels.
        width, height: Optional sensor resolution the intrinsics were
            calibrated at (informational; not required for deprojection).
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0
    height: int = 0

    @classmethod
    def from_fov(
        cls, width: int, height: int, horizontal_fov_deg: float,
    ) -> "CameraIntrinsics":
        """Build square-pixel intrinsics from a horizontal FOV.

        This is how an Isaac Sim virtual camera (which exposes FOV, not a
        calibration matrix) becomes deprojectable: ``fx = (W/2) /
        tan(hfov/2)``, ``fy = fx`` (square pixels), principal point at the
        image centre.
        """
        w = max(1, int(width))
        h = max(1, int(height))
        half = math.radians(max(1e-3, min(179.9, float(horizontal_fov_deg)))) / 2.0
        fx = (w / 2.0) / math.tan(half)
        return cls(fx=fx, fy=fx, cx=w / 2.0, cy=h / 2.0, width=w, height=h)

    def is_valid(self) -> bool:
        return (
            math.isfinite(self.fx) and math.isfinite(self.fy)
            and math.isfinite(self.cx) and math.isfinite(self.cy)
            and self.fx > 0.0 and self.fy > 0.0
        )


# ------------------------------------------------------------------ helpers

def _coerce_depth(depth_frame: Any) -> Optional[np.ndarray]:
    """Coerce input to a float HxW depth array, or None if unusable."""
    if depth_frame is None:
        return None
    try:
        arr = np.asarray(depth_frame)
    except Exception:
        return None
    if arr.ndim == 3 and arr.shape[2] == 1:  # HxWx1 (common ROS/Isaac layout)
        arr = arr[:, :, 0]
    if arr.ndim != 2 or arr.size == 0:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return arr.astype(np.float64, copy=False)


def _coerce_bbox(bbox: Any) -> Optional[tuple[float, float, float, float]]:
    """Accept a BoundingBox model, a mapping, or an (x, y, w, h) sequence."""
    if bbox is None:
        return None
    x = getattr(bbox, "x", None)
    if x is not None:
        vals = (bbox.x, bbox.y, bbox.w, bbox.h)
    elif isinstance(bbox, dict):
        try:
            vals = (bbox["x"], bbox["y"], bbox["w"], bbox["h"])
        except KeyError:
            return None
    elif isinstance(bbox, Sequence) and len(bbox) >= 4:
        vals = tuple(bbox[:4])
    else:
        return None
    try:
        x, y, w, h = (float(v) for v in vals)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    return x, y, w, h


def _coerce_intrinsics(intrinsics: Any) -> Optional[CameraIntrinsics]:
    """Accept CameraIntrinsics, a mapping with fx/fy/cx/cy, or a 4-seq."""
    if intrinsics is None:
        return None
    if isinstance(intrinsics, CameraIntrinsics):
        k = intrinsics
    elif isinstance(intrinsics, dict):
        try:
            k = CameraIntrinsics(
                fx=float(intrinsics["fx"]), fy=float(intrinsics["fy"]),
                cx=float(intrinsics["cx"]), cy=float(intrinsics["cy"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    elif isinstance(intrinsics, Sequence) and len(intrinsics) >= 4:
        try:
            fx, fy, cx, cy = (float(v) for v in intrinsics[:4])
        except (TypeError, ValueError):
            return None
        k = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
    else:
        return None
    return k if k.is_valid() else None


# ------------------------------------------------------------------ core API

def range_for_bbox(
    depth_frame: Any,
    bbox: Any,
    *,
    percentile: float = 50.0,
    inner: float = 0.6,
    depth_scale: float = 1.0,
) -> Optional[float]:
    """Robust range (metres) for a pixel-space bbox over a depth frame.

    Samples the **central** ``inner`` fraction of the box (a detector box
    always includes background around the object's silhouette; the centre
    is far more likely to actually be the object), drops invalid pixels
    (``<= 0``, ``NaN``, ``inf``), and returns the requested percentile of
    what survives — the median by default, which shrugs off both stray
    background pixels and specular dropouts.

    Args:
        depth_frame: HxW (or HxWx1) depth image; invalid pixels are
            0 / NaN / inf.
        bbox: ``BoundingBox`` model, ``{"x","y","w","h"}`` mapping, or an
            ``(x, y, w, h)`` sequence, in the depth frame's pixel space.
        percentile: Percentile of valid depths to report (50 = median;
            lower biases toward the nearest surface in the box).
        inner: Central fraction of the box to sample (0 < inner <= 1).
        depth_scale: Multiplier applied to raw depth values (e.g. 0.001
            for a uint16 millimetre depth image).

    Returns:
        Range in metres, or ``None`` when the frame/box is unusable or no
        valid depth pixel falls inside the sampled region.
    """
    depth = _coerce_depth(depth_frame)
    box = _coerce_bbox(bbox)
    if depth is None or box is None:
        return None
    x, y, w, h = box
    if w <= 0.0 or h <= 0.0:
        return None

    # Shrink to the central `inner` fraction of the box.
    inner = min(1.0, max(1e-3, float(inner)))
    cx = x + w / 2.0
    cy = y + h / 2.0
    sw = w * inner
    sh = h * inner

    fh, fw = depth.shape
    x0 = int(np.clip(math.floor(cx - sw / 2.0), 0, fw))
    x1 = int(np.clip(math.ceil(cx + sw / 2.0), 0, fw))
    y0 = int(np.clip(math.floor(cy - sh / 2.0), 0, fh))
    y1 = int(np.clip(math.ceil(cy + sh / 2.0), 0, fh))
    # Degenerate after clipping (box entirely off-frame) -> at least try the
    # single nearest in-frame pixel only if the box actually overlaps.
    if x1 <= x0 or y1 <= y0:
        return None

    patch = depth[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    pct = float(np.clip(percentile, 0.0, 100.0))
    return float(np.percentile(valid, pct)) * float(depth_scale)


def deproject_pixel(
    u: float,
    v: float,
    depth: float,
    intrinsics: Any,
) -> Optional[tuple[float, float, float]]:
    """Deproject a pixel + depth to a camera-frame 3D point.

    Standard pinhole back-projection in the OpenCV / ROS optical-frame
    convention (+x right, +y down, +z forward):

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

    Args:
        u, v: Pixel coordinates.
        depth: Range along the optical axis in metres (must be finite, > 0).
        intrinsics: ``CameraIntrinsics``, an fx/fy/cx/cy mapping, or an
            ``(fx, fy, cx, cy)`` sequence.

    Returns:
        ``(x, y, z)`` in metres, or ``None`` for invalid depth/intrinsics.
    """
    k = _coerce_intrinsics(intrinsics)
    if k is None:
        return None
    try:
        d = float(depth)
        uu = float(u)
        vv = float(v)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(d) and d > 0.0 and math.isfinite(uu) and math.isfinite(vv)):
        return None
    return ((uu - k.cx) * d / k.fx, (vv - k.cy) * d / k.fy, d)


def enrich_detections_with_depth(
    detections: list,
    depth_frame: Any,
    intrinsics: Any = None,
    *,
    percentile: float = 50.0,
    inner: float = 0.6,
    depth_scale: float = 1.0,
) -> list:
    """Annotate detections with ``range_m`` + camera-frame ``camera_xyz``.

    For each detection whose bbox overlaps valid depth, sets:

      * ``range_m`` — robust range from :func:`range_for_bbox`.
      * ``camera_xyz`` — the bbox centre deprojected at that range via
        :func:`deproject_pixel` (only when usable ``intrinsics`` are
        given; +x right, +y down, +z forward).

    GRACEFUL by contract: if ``depth_frame`` is ``None``, empty, or not an
    HxW numeric array, the input list is returned **unchanged** — no crash,
    no fabricated data.  Detections whose boxes hold no valid depth are
    likewise left untouched.  Works on :class:`~tritium_lib.models.camera.
    CameraDetection` models (annotated in place) and on plain dicts.

    Args:
        detections: ``CameraDetection`` models or bbox-bearing dicts.
        depth_frame: Depth image aligned with the detections' RGB frame.
        intrinsics: Optional pinhole intrinsics for 3D deprojection.
        percentile / inner / depth_scale: See :func:`range_for_bbox`.

    Returns:
        The same list, with detections annotated where depth allowed.
    """
    if not detections:
        return detections
    depth = _coerce_depth(depth_frame)
    if depth is None:
        logger.debug("Depth frame unusable; returning detections unchanged")
        return detections
    k = _coerce_intrinsics(intrinsics)

    for det in detections:
        bbox = det.get("bbox") if isinstance(det, dict) else getattr(det, "bbox", None)
        box = _coerce_bbox(bbox)
        if box is None:
            continue
        rng = range_for_bbox(
            depth, box,
            percentile=percentile, inner=inner, depth_scale=depth_scale,
        )
        if rng is None:
            continue
        xyz: Optional[tuple[float, float, float]] = None
        if k is not None:
            x, y, w, h = box
            xyz = deproject_pixel(x + w / 2.0, y + h / 2.0, rng, k)
        if isinstance(det, dict):
            det["range_m"] = rng
            if xyz is not None:
                det["camera_xyz"] = xyz
        else:
            try:
                det.range_m = rng
                if xyz is not None:
                    det.camera_xyz = xyz
            except (AttributeError, ValueError, TypeError):
                # Frozen/foreign detection type — leave it untouched.
                logger.debug("Detection %r not annotatable; skipped", type(det))
    return detections
