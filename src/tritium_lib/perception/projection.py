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
