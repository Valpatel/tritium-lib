# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Depth-placed camera detections -> unique TrackedTargets.

The last link in the vision -> map -> unique-ID chain:

    detector          -> bbox detections                (perception.detector)
    depth frame       -> range_m + camera_xyz           (perception.depth)
    camera world pose -> world ENU + lat/lng            (perception.projection)
    THIS MODULE       -> unique ``det_{class}_{n}`` ids (tracking.TargetTracker)

:func:`fuse_depth_detections` feeds world-enriched detections into the
tracker's EXISTING vision ingest (:meth:`TargetTracker.update_from_detection`)
— the same nearest-within-motion-budget association every other camera path
uses — so repeated sightings of one entity at a stable world position update
ONE track instead of spawning duplicates, and the measured 3D data (slant
range, ENU offset, elevation, lat/lng) rides on the track's ``kinematics``
for the dossier and tactical map.

Same call for both halves of the north star: the Isaac Sim virtual depth
camera (the FUN half — digital-twin detections placed at true range) and a
real RGB-D sensor on a deployed robot (the production half) produce the
identical detection dicts, so this one seam validates both.

Pure stdlib + the lib's own modules — no torch, no isaac, no ros.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Sequence

logger = logging.getLogger("tritium.tracking.depth_fusion")

__all__ = ["fuse_depth_detections"]


# ------------------------------------------------------------------ helpers

def _field(det: Any, name: str) -> Any:
    """Read a field from a detection dict OR a CameraDetection model."""
    if isinstance(det, dict):
        return det.get(name)
    return getattr(det, name, None)


def _finite(value: Any) -> Optional[float]:
    """Coerce to a finite float, else None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _coerce_enu(value: Any) -> Optional[tuple[float, float, float]]:
    """Coerce a world_enu annotation to a finite (east, north, up) tuple."""
    if value is None or isinstance(value, (str, bytes)):
        return None
    if not isinstance(value, Sequence) or len(value) < 3:
        return None
    e = _finite(value[0])
    n = _finite(value[1])
    u = _finite(value[2])
    if e is None or n is None or u is None:
        return None
    return e, n, u


def _bbox_dict(det: Any) -> Optional[dict]:
    """Extract a detection's bbox as an {x, y, w, h} dict, else None."""
    bbox = _field(det, "bbox")
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        vals = {k: _finite(bbox.get(k)) for k in ("x", "y", "w", "h")}
    else:  # BoundingBox model
        vals = {k: _finite(getattr(bbox, k, None)) for k in ("x", "y", "w", "h")}
    if any(v is None for v in vals.values()):
        return None
    return vals


def _class_name(det: Any) -> str:
    name = _field(det, "class_name") or _field(det, "label")
    return str(name) if name else "unknown"


def _confidence(det: Any) -> float:
    conf = _finite(_field(det, "confidence"))
    return conf if conf is not None else 0.5


def _resolve_local_xy(
    det: Any,
    camera_origin_xy: tuple[float, float],
) -> Optional[tuple[float, float, Optional[tuple[float, float, float]]]]:
    """Resolve a detection's local metric (x, y) from its world annotations.

    Preference order:

      1. ``world_lat`` / ``world_lng`` through the shared geo reference —
         absolute placement, when both the annotation and the reference
         exist.
      2. ``world_enu`` offset from ``camera_origin_xy`` — the honest
         fallback when there is no geo frame (matches the rest of the lib:
         without a reference, the camera IS the local origin).

    Returns ``(x, y, enu)`` or ``None`` when the detection carries no
    usable world position (graceful passthrough — the caller skips it).
    """
    enu = _coerce_enu(_field(det, "world_enu"))
    lat = _finite(_field(det, "world_lat"))
    lng = _finite(_field(det, "world_lng"))

    if lat is not None and lng is not None:
        try:
            from tritium_lib.geo import is_initialized, latlng_to_local
            if is_initialized():
                x, y, _ = latlng_to_local(lat, lng)
                return x, y, enu
        except Exception:  # geo unavailable — fall through to ENU
            pass

    if enu is not None:
        east, north, _up = enu
        return camera_origin_xy[0] + east, camera_origin_xy[1] + north, enu

    return None


# ------------------------------------------------------------------ core API

def fuse_depth_detections(
    tracker,
    detections: list,
    *,
    source: str = "depth",
    cam_id: str = "",
    camera_origin_xy: tuple[float, float] = (0.0, 0.0),
) -> list[Optional[str]]:
    """Fuse depth-placed detections into unique ``det_*`` TrackedTargets.

    Each detection that carries a world position (``world_lat``/``world_lng``
    and/or ``world_enu``, as annotated by
    :func:`tritium_lib.perception.place_detections_on_map`) is fed through
    :meth:`TargetTracker.update_from_detection` — the tracker's existing
    vision association (closest existing track of the same asset type within
    a motion-aware radius) — so a stable entity keeps ONE unique id across
    frames.  The measured 3D data is stamped into the track's ``kinematics``:

      * ``range_m`` — measured slant range from the depth frame.
      * ``world_enu`` / ``elevation_m`` — ENU offset from the camera and
        height above the local ground plane.
      * ``world_lat`` / ``world_lng`` — absolute geographic fix, when known.
      * ``bearing_deg`` / ``distance_m`` — camera->target bearing and ground
        distance derived from the ENU offset (the provenance keys the
        dossier and map already consume).
      * ``depth_source`` — the ``source`` label ("isaac_depth",
        "realsense", ...), so the fused picture records WHICH depth
        producer measured the fix.

    GRACEFUL by contract: detections without any world annotation are
    passed over — no crash, no fabricated position, ``None`` in the result
    slot.  The 2D pipeline (``FrameDetectionPipeline`` + ground-plane
    projection) remains their path onto the map.

    Args:
        tracker: A :class:`~tritium_lib.tracking.TargetTracker`.
        detections: ``CameraDetection`` models or dicts, ideally annotated
            by ``enrich_detections_with_depth`` + ``place_detections_on_map``.
        source: Depth-producer label recorded as ``kinematics.depth_source``.
        cam_id: Observing camera id — stamped as camera provenance
            (``source="camera"``, ``kinematics.camera_id``) by the tracker.
        camera_origin_xy: The camera's local metric position, used to place
            ENU-only detections (no geo frame) at ``origin + (east, north)``.
            Default keeps the camera at the local origin.

    Returns:
        A list parallel to ``detections``: the unique ``det_*`` target id
        each detection matched or created, or ``None`` where the detection
        had no world position or was rejected (e.g. low confidence).
    """
    ids: list[Optional[str]] = []
    if not detections:
        return ids

    for det in detections:
        resolved = _resolve_local_xy(det, camera_origin_xy)
        if resolved is None:
            logger.debug("Detection without world position skipped: %r", det)
            ids.append(None)
            continue
        x, y, enu = resolved

        payload: dict = {
            "class_name": _class_name(det),
            "confidence": _confidence(det),
            "center_x": x,
            "center_y": y,
            "depth_source": str(source),
        }
        if cam_id:
            payload["source_camera"] = str(cam_id)

        bbox = _bbox_dict(det)
        if bbox is not None:
            payload["bbox"] = bbox

        range_m = _finite(_field(det, "range_m"))
        if range_m is not None:
            payload["range_m"] = range_m

        lat = _finite(_field(det, "world_lat"))
        lng = _finite(_field(det, "world_lng"))
        if lat is not None and lng is not None:
            payload["world_lat"] = lat
            payload["world_lng"] = lng

        if enu is not None:
            east, north, up = enu
            payload["world_enu"] = [east, north, up]
            payload["elevation_m"] = up
            ground = math.hypot(east, north)
            payload["distance_m"] = round(ground, 2)
            if ground > 1e-9:
                payload["bearing_deg"] = round(
                    math.degrees(math.atan2(east, north)) % 360.0, 2
                )

        try:
            ids.append(tracker.update_from_detection(payload))
        except Exception as exc:  # never let one bad detection kill the batch
            logger.warning("Tracker rejected depth detection: %s", exc)
            ids.append(None)

    return ids
