# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""A camera carried BY a moving body, rather than bolted to a wall.

Every camera Tritium has consumed until now has a fixed pose: the operator
types a lat/lon and a heading once, and both are constants forever.  A camera
on a robot breaks that assumption in a way that is not a small correction.
The mount is rigid in the BODY's frame, so when the dog turns in place the
lens physically swings through an arc and the field of view sweeps across the
map.  "Where is this camera looking" becomes a function of the body's pose,
re-evaluated every telemetry update.

This module is that function, and nothing else.  It knows no simulator, no
HTTP, and no robot -- it takes a body pose and returns a camera pose, so the
same code serves an Isaac-simulated quadruped, a real Go2 reporting odometry,
a rover, and a PTZ camera on a vehicle roof.  Keeping it here (rather than in
the Isaac addon where the first caller lives) is what stops the second body
from re-deriving the trigonometry and getting a sign wrong.

Conventions:

* **Body frame** is ROS ``base_link`` / REP-103: +X forward out the nose,
  +Y out the port (left) side, +Z up.  A real robot's URDF mount transform
  therefore transcribes directly into ``forward_m``/``left_m``/``up_m``.
* **Output** is Tritium local ENU metres plus a compass heading (0 = north,
  increasing clockwise), which is what the tactical map and ``TrackedTarget``
  already consume.

The one subtlety worth stating out loud: pan is a body-frame slew measured
counter-clockwise (positive = left, as a turret operator means it), while
compass heading increases clockwise.  Composing them is therefore a
SUBTRACTION.  Both conventions are individually standard; it is only where
they meet that the sign flips, so that is where the bug lives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .isaac_frame import LocalPose

__all__ = ["AdvertisedMount", "CameraMount", "parse_advertised_mount"]

# Points per arc when tessellating the field-of-view sector.  Sixteen segments
# hold a 180-degree arc to well under a pixel at map zoom, and the polygon is
# rebuilt on every telemetry tick, so this is deliberately cheap rather than
# smooth.
_ARC_SEGMENTS = 16

_UP_AXES = ("Z", "Y")


@dataclass(frozen=True)
class CameraMount:
    """Where a camera sits on a body, and what it can see from there.

    Args:
        forward_m: mount offset out the nose, in body frame.
        left_m: mount offset out the port side.
        up_m: mount offset above the body origin.
        pan_deg: fixed or commanded slew about the body's up axis, positive to
            the LEFT (counter-clockwise viewed from above).
        tilt_deg: elevation of the boresight, positive UP.
        hfov_deg: horizontal field of view (full angle, not half).
        vfov_deg: vertical field of view (full angle).
        range_m: useful sensor range -- how far out the operator should trust
            a detection.  Caps the drawn footprint even when geometry would
            let the ray run to the horizon.
    """

    forward_m: float = 0.0
    left_m: float = 0.0
    up_m: float = 0.0
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    hfov_deg: float = 90.0
    vfov_deg: float = 60.0
    range_m: float = 50.0

    def __post_init__(self) -> None:
        if not 0.0 < self.hfov_deg < 360.0:
            raise ValueError(
                f"hfov_deg must be in (0, 360), got {self.hfov_deg!r}"
            )
        if not 0.0 <= self.vfov_deg < 180.0:
            raise ValueError(
                f"vfov_deg must be in [0, 180), got {self.vfov_deg!r}"
            )
        if self.range_m < 0.0:
            raise ValueError(f"range_m must be non-negative, got {self.range_m!r}")
        if not -90.0 <= self.tilt_deg <= 90.0:
            raise ValueError(
                f"tilt_deg must be within [-90, 90], got {self.tilt_deg!r}"
            )

    # -- the camera's own pose --------------------------------------------

    def world_pose(self, body: LocalPose) -> LocalPose:
        """Body pose -> the lens's pose in Tritium local ENU.

        The offset is rotated by the body's heading before being added.  Doing
        it the other way round -- adding metres of "forward" straight onto the
        north axis -- happens to be correct when the body faces north and is
        wrong everywhere else, which is why it survives so many test suites.
        """
        heading = float(body.heading_deg)
        rad = math.radians(heading)
        sin_h, cos_h = math.sin(rad), math.cos(rad)

        # Body +forward is the heading direction; body +left is 90 deg
        # counter-clockwise from it, i.e. heading - 90 in compass terms.
        east = body.east_m + self.forward_m * sin_h - self.left_m * cos_h
        north = body.north_m + self.forward_m * cos_h + self.left_m * sin_h

        return LocalPose(
            east_m=east,
            north_m=north,
            up_m=body.up_m + self.up_m,
            heading_deg=(heading - self.pan_deg) % 360.0,
        )

    # -- what it sees ------------------------------------------------------

    def ground_footprint(
        self, body: LocalPose, ground_up_m: float = 0.0
    ) -> list[tuple[float, float]]:
        """The patch of ground inside the field of view, as a closed polygon.

        Returns ``(east_m, north_m)`` vertices tracing the near arc and far
        arc of a sector centred on the camera's boresight -- the shape the
        tactical map already draws as an FOV cone, but now recomputed as the
        robot moves.

        The near and far edges come from intersecting the bottom and top of
        the vertical FOV with the ground plane, each capped at ``range_m``.
        A camera aimed entirely above horizontal sees no ground at all and
        returns an empty list: the naive ``height / tan(elevation)`` gives a
        negative distance there, which would draw a phantom cone BEHIND the
        robot rather than no cone at all.
        """
        cam = self.world_pose(body)
        height = cam.up_m - float(ground_up_m)

        half_v = self.vfov_deg / 2.0
        bottom_elev = self.tilt_deg - half_v
        if bottom_elev > 0.0 or height <= 0.0:
            return []

        far = self._ground_distance(self.tilt_deg + half_v, height)
        near = self._ground_distance(bottom_elev, height)
        if far < near:  # pathological mount; nothing sensible to draw
            return []

        half_h = self.hfov_deg / 2.0
        left = cam.heading_deg - half_h
        right = cam.heading_deg + half_h

        ring: list[tuple[float, float]] = []
        for i in range(_ARC_SEGMENTS + 1):
            bearing = left + (right - left) * i / _ARC_SEGMENTS
            ring.append(self._offset(cam, bearing, far))
        for i in range(_ARC_SEGMENTS + 1):
            bearing = right + (left - right) * i / _ARC_SEGMENTS
            ring.append(self._offset(cam, bearing, near))
        ring.append(ring[0])
        return ring

    def _ground_distance(self, elevation_deg: float, height_m: float) -> float:
        """Horizontal distance at which a ray meets the ground, capped at range.

        A ray at or above horizontal never descends to the ground plane; its
        useful extent is the sensor's own range rather than infinity.
        """
        if elevation_deg >= 0.0:
            return self.range_m
        drop = math.tan(math.radians(-elevation_deg))
        return min(height_m / drop, self.range_m)

    @staticmethod
    def _offset(
        cam: LocalPose, bearing_deg: float, distance_m: float
    ) -> tuple[float, float]:
        rad = math.radians(bearing_deg)
        return (
            cam.east_m + distance_m * math.sin(rad),
            cam.north_m + distance_m * math.cos(rad),
        )

    # -- handing the mount to a USD stage ----------------------------------

    def stage_offset(
        self, up_axis: str = "Z", meters_per_unit: float = 1.0
    ) -> tuple[float, float, float]:
        """The mount offset expressed as a USD stage translation.

        Lets a simulator parent a camera prim under the body prim using the
        SAME numbers the map projection uses, so the rendered image and the
        drawn FOV cone cannot drift apart.  Axis handling mirrors
        :class:`~tritium_lib.geo.isaac_frame.IsaacFrame`: Z-up stages put
        north on +Y, Y-up stages put north on -Z.
        """
        axis = str(up_axis).upper()
        if axis not in _UP_AXES:
            raise ValueError(f"up_axis must be one of {_UP_AXES}, got {up_axis!r}")
        if meters_per_unit <= 0.0:
            raise ValueError(
                f"meters_per_unit must be positive, got {meters_per_unit!r}"
            )
        scale = 1.0 / meters_per_unit
        if axis == "Z":
            xyz = (self.forward_m, self.left_m, self.up_m)
        else:
            xyz = (self.forward_m, self.up_m, -self.left_m)
        return (xyz[0] * scale, xyz[1] * scale, xyz[2] * scale)


# --- Self-describing camera servers ----------------------------------------
#
# A camera process that renders from a body-parented lens is the only party
# that knows the mount for certain -- it is the geometry it just rendered
# with.  Making the consumer re-enter those numbers by hand creates a second
# copy of one truth, and the two drift silently: the operator sees a picture
# taken from the robot's nose and a cone drawn from its tail, with nothing in
# the system able to notice.  So the server ADVERTISES its mount and this is
# the parse of that advertisement.
#
# The parse is deliberately forgiving about everything except the one fact
# that matters -- whether this camera rides a body at all.  A malformed offset
# should degrade to zero, not discard the attachment; but a document with no
# body named in it must NOT produce a mount, because binding a feed to an
# empty target id makes the map cone vanish with no error anywhere.

#: Keys a parsed mount writes into a feed's stored config, matching what the
#: camera-feeds registration and its attachment PATCH already accept.
_EXTRA_KEYS = {
    "forward_m": "mount_forward_m",
    "left_m": "mount_left_m",
    "up_m": "mount_up_m",
    "pan_deg": "mount_pan_deg",
    "tilt_deg": "mount_tilt_deg",
}


@dataclass(frozen=True)
class AdvertisedMount:
    """What a camera server said about the body it is bolted to.

    Attributes:
        mount: the rigid body->lens transform, ready to pose a FOV cone.
        attach_to: tracked-target id of the body, when the server knows it.
            ``None`` means "rides a body we cannot name yet" -- the operator
            still has to say which tracked target that prim corresponds to.
        prim: the simulator scene path the render camera is parented under,
            carried through for provenance and for operator display.
    """

    mount: CameraMount
    attach_to: str | None = None
    prim: str | None = None

    def to_feed_extra(self) -> dict:
        """The mount as camera-feed config keys.

        ``attach_to`` is OMITTED rather than set to ``None`` when unknown: a
        present-but-null key reads downstream as an attachment whose target
        lookup fails forever, which shows as a permanently stale cone.
        """
        extra: dict = {
            "fov_angle": self.mount.hfov_deg,
            "fov_range": self.mount.range_m,
        }
        for field, key in _EXTRA_KEYS.items():
            extra[key] = getattr(self.mount, field)
        if self.attach_to:
            extra["attach_to"] = self.attach_to
        return extra


def _num(source: dict, key: str, default: float) -> float:
    """A float from an untrusted document, or the default.

    Covers the three ways this goes wrong off a socket: the key is absent,
    the value is JSON null (an unset command-line default serializes that
    way), or the value is a string a different version wrote.
    """
    try:
        value = source.get(key)
        return default if value is None else float(value)
    except (AttributeError, TypeError, ValueError):
        return default


def _text(source: dict, key: str) -> str | None:
    value = source.get(key)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def parse_advertised_mount(status: object) -> AdvertisedMount | None:
    """Read a camera server's ``/status`` document as a body mount.

    Returns ``None`` -- meaning "an ordinary fixed camera" -- for any
    document that does not name a body, including junk that is not a mapping
    at all.  Callers treat ``None`` as "leave the operator's pose alone",
    so this must never raise on a response it did not expect.
    """
    if not isinstance(status, dict):
        return None
    mount_doc = status.get("mount")
    if not isinstance(mount_doc, dict):
        return None

    attach_to = _text(mount_doc, "attach_to")
    prim = _text(mount_doc, "prim")
    if not attach_to and not prim:
        # Offsets with nothing to hang them off. Not a mount.
        return None

    # FOV describes the render, not the bracket, so it sits at the top level
    # of /status next to width/height -- the same keys a wall camera uses.
    hfov = _num(status, "fov_angle", 90.0)
    if not 0.0 < hfov < 360.0:
        hfov = 90.0
    range_m = _num(status, "fov_range", 50.0)
    if range_m < 0.0:
        range_m = 50.0

    mount = CameraMount(
        forward_m=_num(mount_doc, "forward_m", 0.0),
        left_m=_num(mount_doc, "left_m", 0.0),
        up_m=_num(mount_doc, "up_m", 0.0),
        pan_deg=_num(mount_doc, "pan_deg", 0.0),
        tilt_deg=_num(mount_doc, "tilt_deg", 0.0),
        hfov_deg=hfov,
        vfov_deg=hfov * 0.6,
        range_m=range_m,
    )
    return AdvertisedMount(mount=mount, attach_to=attach_to, prim=prim)
