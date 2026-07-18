# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Pixel -> world projection for a posed ground camera.

A detection is a box in image pixels; the tactical map needs a world
position.  :class:`GroundCameraModel` is a deliberately simple, documented
pin-hole ground-plane model: it maps a detection's *foot point* (bottom
centre of its box) to a bearing + range from the camera, then to local
metric (x, y) and geographic (lat, lng).

Assumptions (honest, and correct for a fixed CCTV-style camera):
  * The camera looks out roughly horizontally / slightly down at a flat
    ground plane.
  * Horizontal image position maps linearly to bearing across the FOV.
  * Vertical image position maps monotonically to range: an object whose
    feet are low in the frame is near; feet near the horizon line are far.

This is an approximation, not photogrammetry — but it is reversible enough
to unit-test and good enough to place a track on the map, and it is the
SAME model whether the frame came from a synthetic demo feed or a real RTSP
security camera.  Bearing convention matches the embodiment SDK and the
virtual-camera emitter: 0 deg = +y (North), 90 deg = +x (East).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from tritium_lib.models.camera import CameraDetection


@dataclass
class GroundCameraModel:
    """A posed, calibrated-enough ground camera for pixel->world projection.

    Attributes:
        lat, lng: Camera geographic position.
        heading_deg: Compass heading the camera points (0=N, 90=E).
        fov_deg: Horizontal field of view in degrees.
        range_m: Ground distance (m) that the horizon line maps to — the
            camera's effective sight range.
        image_w, image_h: Frame dimensions in pixels.
        near_m: Ground distance (m) that the bottom of the frame maps to.
        horizon_frac: Normalized image-y (0=top) of the ground horizon line;
            detections above it are treated as at the horizon (max range).
    """

    lat: float
    lng: float
    heading_deg: float
    fov_deg: float = 70.0
    range_m: float = 80.0
    image_w: int = 640
    image_h: int = 480
    near_m: float = 3.0
    horizon_frac: float = 0.35

    def bearing_range(self, det: CameraDetection) -> tuple[float, float]:
        """Return (bearing_deg, distance_m) for a detection's foot point."""
        w = self.image_w or 1
        h = self.image_h or 1
        # Foot point: horizontal centre, vertical bottom of the box.
        foot_x = det.bbox.x + det.bbox.w / 2.0
        foot_y = det.bbox.y + det.bbox.h
        cx_norm = _clamp(foot_x / w, 0.0, 1.0)
        by_norm = _clamp(foot_y / h, 0.0, 1.0)

        # Horizontal -> bearing across the FOV, centred on the heading.
        bearing = self.heading_deg + (cx_norm - 0.5) * self.fov_deg

        # Vertical -> range. t=0 at the horizon line, t=1 at the bottom edge.
        denom = max(1e-6, 1.0 - self.horizon_frac)
        t = _clamp((by_norm - self.horizon_frac) / denom, 0.0, 1.0)
        distance = self.range_m - (self.range_m - self.near_m) * t
        return bearing % 360.0, distance

    def project(self, det: CameraDetection) -> dict:
        """Project a detection to a world position.

        Returns a dict with local metric ``x``/``y`` (metres, the units the
        TargetTracker consumes), geographic ``lat``/``lng`` when a geo
        reference is available, and the intermediate ``bearing_deg`` /
        ``distance_m``.
        """
        bearing, distance = self.bearing_range(det)
        rad = math.radians(bearing)
        # 0deg=North=+y, 90deg=East=+x.
        d_east = distance * math.sin(rad)
        d_north = distance * math.cos(rad)

        cam_x, cam_y = self._camera_local()
        x = cam_x + d_east
        y = cam_y + d_north
        lat, lng = self._local_to_latlng(x, y)
        return {
            "x": x,
            "y": y,
            "lat": lat,
            "lng": lng,
            "bearing_deg": round(bearing, 2),
            "distance_m": round(distance, 2),
        }

    # -- geo helpers (graceful — projection still yields x/y offsets without
    #    a geo reference, using the camera as the local origin) -------------

    def _camera_local(self) -> tuple[float, float]:
        try:
            from tritium_lib.geo import latlng_to_local
            cx, cy, _ = latlng_to_local(self.lat, self.lng)
            return cx, cy
        except Exception:
            return 0.0, 0.0

    def _local_to_latlng(self, x: float, y: float) -> tuple[float | None, float | None]:
        try:
            from tritium_lib.geo import local_to_latlng_2d
            return local_to_latlng_2d(x, y)
        except Exception:
            return None, None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Depth-derived 3D -> world placement
# ---------------------------------------------------------------------------
#
# GroundCameraModel above guesses range from a detection's image position —
# the flat-ground approximation.  When the detection carries a *measured*
# camera-frame 3D point (``camera_xyz`` from ``perception.depth`` — Isaac Sim
# depth annotator or a real RGB-D sensor), we can do better: rotate that
# point through the camera's true pose and place it at its TRUE position on
# the tactical map, elevation included.  Same math for the virtual depth
# camera (the FUN half) and a RealSense on a robot (the production half).


@dataclass(frozen=True)
class CameraWorldPose:
    """World pose of a camera for camera-frame -> world projection.

    (Named ``CameraWorldPose`` because ``perception.CameraPose`` is the PTZ
    pan/tilt estimate — this is the camera's placement in the WORLD.)

    Orientation follows the aerospace yaw-pitch-roll convention applied to
    the camera's optical axis:

      * ``heading_deg`` — compass heading the optical axis points
        (0 = North, 90 = East, clockwise), matching :class:`GroundCameraModel`
        and ``tritium_lib.geo``.
      * ``pitch_deg`` — elevation of the optical axis (positive = up;
        a ground camera tilted 20 deg down has ``pitch_deg=-20``).
      * ``roll_deg`` — rotation about the optical axis (positive = camera's
        right side rolls down).

    Attributes:
        lat, lng: Geographic position of the camera (optional — without
            them projection still yields ENU offsets, no lat/lng).
        heading_deg: Compass heading of the optical axis.
        pitch_deg: Optical-axis elevation (positive up).
        roll_deg: Rotation about the optical axis (positive right-down).
        height_m: Mount height of the optical centre above the local
            ground plane (metres).
    """

    lat: Optional[float] = None
    lng: Optional[float] = None
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    height_m: float = 0.0

    @property
    def has_geo(self) -> bool:
        return self.lat is not None and self.lng is not None


def _coerce_xyz(camera_xyz: Any) -> Optional[tuple[float, float, float]]:
    """Coerce a camera-frame point to a finite (x, y, z) tuple, else None."""
    if camera_xyz is None:
        return None
    if not isinstance(camera_xyz, Sequence) or len(camera_xyz) < 3:
        return None
    try:
        x, y, z = (float(v) for v in camera_xyz[:3])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, z)):
        return None
    return x, y, z


def world_from_camera_xyz(
    camera_xyz: Any,
    camera_pose: CameraWorldPose,
) -> Optional[dict]:
    """Project a camera-frame 3D point to world ENU + lat/lng.

    The input point is in the OpenCV / ROS optical frame produced by
    :func:`tritium_lib.perception.depth.deproject_pixel` (+x right, +y down,
    +z forward, metres).  It is rotated through the camera's yaw/pitch/roll
    into the world East-North-Up frame and lifted by the mount height, then
    geolocated from the camera's lat/lng via
    :func:`tritium_lib.geo.destination_point` (pure function — no dependence
    on the module-level geo reference singleton).

    Args:
        camera_xyz: ``(x, y, z)`` point in the camera optical frame.
        camera_pose: The camera's world pose.

    Returns:
        Dict with ``east`` / ``north`` (metres from the camera's ground
        footprint), ``up`` (metres above the local ground plane, mount
        height included), and ``lat`` / ``lng`` (``None`` when the pose has
        no geographic position).  ``None`` for an unusable input point.
    """
    xyz = _coerce_xyz(camera_xyz)
    if xyz is None:
        return None
    x_opt, y_opt, z_opt = xyz

    # Optical frame -> body FRD (forward, right, down).
    f, r, d0 = z_opt, x_opt, y_opt

    # Body FRD -> NED via the aerospace ZYX direction-cosine matrix
    # (yaw = heading CW from north, pitch positive up, roll positive
    # right-side-down).
    cy = math.cos(math.radians(camera_pose.heading_deg))
    sy = math.sin(math.radians(camera_pose.heading_deg))
    cp = math.cos(math.radians(camera_pose.pitch_deg))
    sp = math.sin(math.radians(camera_pose.pitch_deg))
    cr = math.cos(math.radians(camera_pose.roll_deg))
    sr = math.sin(math.radians(camera_pose.roll_deg))

    n = (cy * cp) * f + (cy * sp * sr - sy * cr) * r + (cy * sp * cr + sy * sr) * d0
    e = (sy * cp) * f + (sy * sp * sr + cy * cr) * r + (sy * sp * cr - cy * sr) * d0
    d = (-sp) * f + (cp * sr) * r + (cp * cr) * d0

    east = e
    north = n
    up = camera_pose.height_m - d

    lat: Optional[float] = None
    lng: Optional[float] = None
    if camera_pose.has_geo:
        ground_range = math.hypot(east, north)
        if ground_range < 1e-9:
            lat, lng = float(camera_pose.lat), float(camera_pose.lng)
        else:
            from tritium_lib.geo import destination_point
            bearing = math.degrees(math.atan2(east, north)) % 360.0
            lat, lng = destination_point(
                float(camera_pose.lat), float(camera_pose.lng),
                bearing, ground_range,
            )

    return {"east": east, "north": north, "up": up, "lat": lat, "lng": lng}


def place_detections_on_map(
    detections: list,
    camera_pose: CameraWorldPose,
) -> list:
    """Annotate depth-enriched detections with their TRUE world position.

    For each detection carrying a ``camera_xyz`` (set by
    :func:`tritium_lib.perception.depth.enrich_detections_with_depth`), sets:

      * ``world_enu`` — ``(east, north, up)`` metres from the camera's
        ground footprint (up includes the mount height).
      * ``world_lat`` / ``world_lng`` — geographic position, when the pose
        has one.

    GRACEFUL by contract: detections without a usable ``camera_xyz`` are
    passed through untouched — no crash, no fabricated position.  Works on
    :class:`~tritium_lib.models.camera.CameraDetection` models (annotated in
    place) and on plain dicts.

    Args:
        detections: ``CameraDetection`` models or dicts, ideally
            depth-enriched.
        camera_pose: The producing camera's world pose.

    Returns:
        The same list, annotated where ``camera_xyz`` allowed.
    """
    if not detections:
        return detections
    for det in detections:
        xyz = (
            det.get("camera_xyz") if isinstance(det, dict)
            else getattr(det, "camera_xyz", None)
        )
        world = world_from_camera_xyz(xyz, camera_pose)
        if world is None:
            continue
        enu = (world["east"], world["north"], world["up"])
        if isinstance(det, dict):
            det["world_enu"] = enu
            if world["lat"] is not None:
                det["world_lat"] = world["lat"]
                det["world_lng"] = world["lng"]
        else:
            try:
                det.world_enu = enu
                if world["lat"] is not None:
                    det.world_lat = world["lat"]
                    det.world_lng = world["lng"]
            except (AttributeError, ValueError, TypeError):
                # Frozen/foreign detection type — leave it untouched.
                pass
    return detections
