# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""One-call depth-camera pipeline: RGB + depth -> unique 3D map targets.

The lib already ships every link of the chain:

    detector.detect(rgb)        -> bbox detections       (injected, duck-typed)
    enrich_detections_with_depth -> range_m + camera_xyz  (perception.depth)
    place_detections_on_map      -> world ENU + lat/lng   (perception.projection)
    fuse_depth_detections        -> unique ``det_*`` ids  (tracking.depth_fusion)

Until now every consumer (SC camera_feeds, the Isaac Sim bridge, a real
RGB-D robot) had to wire those four calls itself, in the right order, with
the right graceful fallbacks.  :func:`process_depth_frame` is that wiring,
once, tested: one call per frame, returns the unique tracked target ids.
:class:`DepthCameraPipeline` is the same call with the camera's fixed
geometry (intrinsics, world pose, tracker, detector) held so per-frame call
sites stay one-liners.

Design contract (matches the rest of the perception seam):

  * The detector is **injected and duck-typed** — anything with
    ``detect(rgb, cam_id) -> detections`` (or ``detect(rgb)``); dicts and
    :class:`~tritium_lib.models.camera.CameraDetection` models both work.
    No heavy detector import lives here.
  * GRACEFUL end to end: no depth frame, no intrinsics, a crashing
    detector, junk boxes — never an exception, never fabricated *measured*
    data.  With ``depth=None`` detections still reach the map through the
    documented :class:`~tritium_lib.perception.projection.GroundCameraModel`
    flat-ground approximation (provenance-labelled ``2d_ground``), so an
    RGB-only camera and a depth camera share this one entry point.
  * Same call for both halves of the north star: the Isaac Sim virtual
    depth camera (FUN) and a RealSense/OAK-D/ZED on a deployed robot
    (production) — identical code path, identical tests.

Pure numpy/stdlib + the lib's own modules — no torch, no isaac, no ros.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from tritium_lib.models.camera import BoundingBox, CameraDetection
from tritium_lib.perception.depth import (
    _coerce_bbox,
    _coerce_intrinsics,
    enrich_detections_with_depth,
)
from tritium_lib.perception.projection import (
    CameraWorldPose,
    GroundCameraModel,
    place_detections_on_map,
)
from tritium_lib.tracking.depth_fusion import fuse_depth_detections

logger = logging.getLogger("tritium.perception.depth_pipeline")

__all__ = ["DepthCameraPipeline", "process_depth_frame"]

#: ``kinematics.depth_source`` label stamped on tracks placed by the
#: ground-plane fallback (no measured depth) — honest provenance so the
#: dossier can tell a measured 3D fix from the flat-ground approximation.
GROUND_FALLBACK_SOURCE = "2d_ground"


# ------------------------------------------------------------------ helpers

def _safe_detect(detector: Any, rgb: Any, cam_id: str) -> list:
    """Run the injected detector, duck-typed, never raising.

    Tries ``detect(rgb, cam_id)`` first (the lib's ``FrameObjectDetector``
    signature), then plain ``detect(rgb)``.  Any detector failure yields an
    empty frame, not a crash — one bad frame must never kill a feed loop.
    """
    if detector is None:
        return []
    try:
        return list(detector.detect(rgb, cam_id) or [])
    except TypeError:
        try:
            return list(detector.detect(rgb) or [])
        except Exception as exc:
            logger.warning("Detector failed on %s: %s", cam_id or "frame", exc)
            return []
    except Exception as exc:
        logger.warning("Detector failed on %s: %s", cam_id or "frame", exc)
        return []


def _annotate(det: Any, key: str, value: Any) -> None:
    """Set a field on a detection dict OR model; foreign types stay untouched."""
    if isinstance(det, dict):
        det[key] = value
        return
    try:
        setattr(det, key, value)
    except (AttributeError, ValueError, TypeError):
        logger.debug("Detection %r not annotatable; skipped", type(det))


def _as_camera_detection(det: Any, cam_id: str) -> Optional[CameraDetection]:
    """View a detection as a CameraDetection for ground-model projection."""
    bbox = det.get("bbox") if isinstance(det, dict) else getattr(det, "bbox", None)
    if not isinstance(det, dict) and bbox is not None and hasattr(bbox, "x"):
        return det  # already model-shaped (bbox attribute access works)
    box = _coerce_bbox(bbox)
    if box is None:
        return None
    x, y, w, h = box
    name = det.get("class_name") if isinstance(det, dict) else None
    conf = det.get("confidence") if isinstance(det, dict) else None
    try:
        return CameraDetection(
            source_id=cam_id or "",
            class_name=str(name or "unknown"),
            confidence=float(conf) if conf is not None else 0.5,
            bbox=BoundingBox(x=x, y=y, w=w, h=h),
        )
    except Exception:
        return None


def _ground_model(
    pose: CameraWorldPose,
    intrinsics: Any,
    rgb_shape: Any,
) -> GroundCameraModel:
    """Build the 2D flat-ground projector from the camera's known geometry."""
    k = _coerce_intrinsics(intrinsics)
    if rgb_shape is not None and len(rgb_shape) >= 2:
        image_h, image_w = int(rgb_shape[0]), int(rgb_shape[1])
    elif k is not None and k.width > 0 and k.height > 0:
        image_w, image_h = k.width, k.height
    else:
        image_w, image_h = 640, 480
    if k is not None:
        w_calib = k.width if k.width > 0 else image_w
        fov_deg = 2.0 * math.degrees(math.atan((w_calib / 2.0) / k.fx))
        fov_deg = min(179.0, max(1.0, fov_deg))
    else:
        fov_deg = 70.0
    return GroundCameraModel(
        lat=pose.lat if pose.has_geo else None,  # type: ignore[arg-type]
        lng=pose.lng if pose.has_geo else None,  # type: ignore[arg-type]
        heading_deg=pose.heading_deg,
        fov_deg=fov_deg,
        image_w=image_w,
        image_h=image_h,
    )


def _place_on_ground(
    detections: list,
    indices: list[int],
    pose: CameraWorldPose,
    intrinsics: Any,
    rgb_shape: Any,
    cam_id: str,
) -> list[int]:
    """Annotate ``world_enu`` (+lat/lng) via the ground model; return placed."""
    model = _ground_model(pose, intrinsics, rgb_shape)
    placed: list[int] = []
    for i in indices:
        cam_det = _as_camera_detection(detections[i], cam_id)
        if cam_det is None:
            continue
        try:
            world = model.project(cam_det)
        except Exception as exc:  # junk box — skip, never crash the frame
            logger.debug("Ground projection failed: %s", exc)
            continue
        rad = math.radians(world["bearing_deg"])
        dist = world["distance_m"]
        _annotate(
            detections[i], "world_enu",
            (dist * math.sin(rad), dist * math.cos(rad), 0.0),
        )
        if world.get("lat") is not None:
            _annotate(detections[i], "world_lat", world["lat"])
            _annotate(detections[i], "world_lng", world["lng"])
        placed.append(i)
    return placed


# ------------------------------------------------------------------ core API

def process_depth_frame(
    rgb: Any,
    depth: Any,
    intrinsics: Any,
    pose: Optional[CameraWorldPose],
    tracker: Any,
    detector: Any,
    *,
    source: str = "depth",
    cam_id: str = "",
    depth_scale: float = 1.0,
    percentile: float = 50.0,
    inner: float = 0.6,
    camera_origin_xy: tuple[float, float] = (0.0, 0.0),
    ground_fallback: bool = True,
) -> list[Optional[str]]:
    """Run the whole RGB+depth -> unique 3D map target chain in ONE call.

    Orchestrates the existing seam — detect, then
    :func:`~tritium_lib.perception.depth.enrich_detections_with_depth`, then
    :func:`~tritium_lib.perception.projection.place_detections_on_map`, then
    :func:`~tritium_lib.tracking.depth_fusion.fuse_depth_detections` — with
    nothing reimplemented, so every graceful-degradation contract those
    functions document holds here too.

    Detections whose bbox holds valid depth are placed at their MEASURED 3D
    position (``kinematics.depth_source = source``).  Detections without
    usable depth (no depth frame at all, or a per-box dropout) fall back to
    the documented flat-ground 2D projection when ``ground_fallback`` is on
    and a ``pose`` exists — provenance-labelled ``2d_ground`` so the fused
    picture never mistakes a guess for a measurement.  Because both paths
    feed the tracker's one vision-association ingest, an entity keeps ONE
    unique ``det_*`` id across frames — including across depth dropouts.

    Args:
        rgb: BGR/RGB frame handed to the injected detector (opaque here).
        depth: Aligned HxW depth frame in metres (``None`` -> 2D fallback).
        intrinsics: :class:`~tritium_lib.perception.depth.CameraIntrinsics`,
            an fx/fy/cx/cy mapping, a 4-sequence, or ``None``.
        pose: The camera's :class:`~tritium_lib.perception.projection.
            CameraWorldPose`; ``None`` disables world placement (detections
            without measured world positions then return ``None`` slots).
        tracker: A :class:`~tritium_lib.tracking.TargetTracker`.
        detector: Duck-typed — ``detect(rgb, cam_id)`` or ``detect(rgb)``
            returning ``CameraDetection`` models or bbox dicts.
        source: Depth-producer label ("isaac_depth", "realsense", ...).
        cam_id: Observing camera id, stamped as camera provenance.
        depth_scale: Multiplier for raw depth units (0.001 for uint16 mm).
        percentile / inner: Bbox depth sampling, see
            :func:`~tritium_lib.perception.depth.range_for_bbox`.
        camera_origin_xy: Camera's local metric position for ENU-only
            placement (no geo frame).
        ground_fallback: Allow the flat-ground 2D path for depth-less
            detections (off -> honest passthrough, ``None`` slots).

    Returns:
        A list parallel to the frame's detections: the unique ``det_*``
        target id each detection matched or created, or ``None`` where it
        could not be placed or was rejected (e.g. low confidence).
    """
    detections = _safe_detect(detector, rgb, cam_id)
    if not detections:
        return []

    world_pose = pose if pose is not None else CameraWorldPose()

    # Measured-depth path: range + camera-frame 3D + true world placement.
    enrich_detections_with_depth(
        detections, depth, intrinsics,
        percentile=percentile, inner=inner, depth_scale=depth_scale,
    )
    place_detections_on_map(detections, world_pose)
    ids = fuse_depth_detections(
        tracker, detections,
        source=source, cam_id=cam_id, camera_origin_xy=camera_origin_xy,
    )

    # Ground fallback: whatever depth could not place, the documented 2D
    # flat-ground approximation still can — same tracker seam, honest label.
    if ground_fallback and pose is not None:
        unplaced = [i for i, tid in enumerate(ids) if tid is None]
        if unplaced:
            rgb_shape = getattr(rgb, "shape", None)
            placed = _place_on_ground(
                detections, unplaced, world_pose, intrinsics, rgb_shape, cam_id,
            )
            if placed:
                fallback_ids = fuse_depth_detections(
                    tracker, [detections[i] for i in placed],
                    source=GROUND_FALLBACK_SOURCE, cam_id=cam_id,
                    camera_origin_xy=camera_origin_xy,
                )
                for i, tid in zip(placed, fallback_ids):
                    ids[i] = tid

    return ids


class DepthCameraPipeline:
    """A posed depth camera bound to its detector + tracker.

    Holds the per-camera constants (intrinsics, world pose, detector,
    tracker, labels) so the per-frame call site is one line::

        pipe = DepthCameraPipeline(detector, tracker, intrinsics, pose,
                                   source="isaac_depth", cam_id="cam-01")
        ids = pipe.process(rgb, depth)   # every frame

    All arguments and graceful contracts match :func:`process_depth_frame`.
    """

    def __init__(
        self,
        detector: Any,
        tracker: Any,
        intrinsics: Any = None,
        pose: Optional[CameraWorldPose] = None,
        *,
        source: str = "depth",
        cam_id: str = "",
        depth_scale: float = 1.0,
        percentile: float = 50.0,
        inner: float = 0.6,
        camera_origin_xy: tuple[float, float] = (0.0, 0.0),
        ground_fallback: bool = True,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.intrinsics = intrinsics
        self.pose = pose
        self.source = source
        self.cam_id = cam_id
        self.depth_scale = float(depth_scale)
        self.percentile = float(percentile)
        self.inner = float(inner)
        self.camera_origin_xy = camera_origin_xy
        self.ground_fallback = bool(ground_fallback)

        self.frames_processed = 0
        self.tracked_total = 0
        self.ids_last: list[Optional[str]] = []

    def process(self, rgb: Any, depth: Any = None) -> list[Optional[str]]:
        """Process one aligned RGB(+depth) frame; return tracked target ids."""
        ids = process_depth_frame(
            rgb, depth, self.intrinsics, self.pose, self.tracker, self.detector,
            source=self.source, cam_id=self.cam_id,
            depth_scale=self.depth_scale, percentile=self.percentile,
            inner=self.inner, camera_origin_xy=self.camera_origin_xy,
            ground_fallback=self.ground_fallback,
        )
        self.frames_processed += 1
        self.ids_last = ids
        self.tracked_total += sum(1 for tid in ids if tid is not None)
        return ids
