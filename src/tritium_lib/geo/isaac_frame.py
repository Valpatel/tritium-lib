# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Isaac/USD stage frame <-> Tritium local ENU frame.

A simulated body's pose only helps the operator if the icon on the tactical
map lands where the robot actually is inside the simulator.  That requires
one conversion, applied identically by every consumer:

    USD stage (author's up axis, author's units, yaw CCW from +X)
        -> Tritium local (metres, +X east / +Y north / +Z up, heading CW from north)

This lived as a prose comment inside the Isaac addon's quadruped server and
was applied by hand at two call sites.  It is pure arithmetic with no Isaac
import, so it belongs here: the Isaac pose bridge, a ROS2 ``/odom`` relay, and
any future rover or aerial body all need the same maths, and only one of them
runs on a GPU box.

The headline identity::

    heading_deg = (90 - yaw_deg) mod 360

is a REFLECTION, not an offset, because the two frames disagree about which
way an angle grows: Isaac yaw increases counter-clockwise from east, Tritium
heading increases clockwise from north.  ``yaw + 90`` maps north correctly and
is wrong everywhere else, which is exactly why the tests check all four
cardinals in both directions.

Downstream, ``tritium_lib.geo.local_to_latlng`` turns the local metres this
module produces into the lat/lng an operator API response carries.

Usage::

    frame = IsaacFrame.from_stage_metadata(bridge_health["result"])
    pose = frame.pose_to_local(translation, quaternion_wxyz)
    lat, lng = local_to_latlng(pose.east_m, pose.north_m)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "IsaacFrame",
    "LocalPose",
    "quat_to_yaw_deg",
]

# USD authors an up axis as a token; Isaac Sim defaults to Z-up, but assets
# imported from DCC tools are frequently Y-up.  Guessing wrong silently swaps
# north for altitude, so both are handled explicitly and anything else raises.
_UP_AXES = ("Z", "Y")

# A quaternion shorter than this is not a rotation -- it is a dropped or
# zero-initialised pose.  Treating it as identity would report "facing north"
# for a body whose pose we never actually received.
_MIN_QUAT_NORM = 1e-9


@dataclass(frozen=True)
class LocalPose:
    """A body pose in Tritium local coordinates.

    Attributes mirror what the tactical map and ``TrackedTarget`` consume:
    metres east/north/up of the geo-reference origin, plus a compass heading.
    """

    east_m: float
    north_m: float
    up_m: float
    heading_deg: float


def quat_to_yaw_deg(quat_wxyz: Sequence[float]) -> float:
    """Extract rotation about the vertical axis from a quaternion, in degrees.

    ``quat_wxyz`` is ``(w, x, y, z)`` -- the order USD's ``Gf.Quatd`` reports
    when you take ``GetReal()`` then ``GetImaginary()``.

    Roll and pitch are discarded on purpose.  A walking quadruped's torso
    pitches and rolls continuously; if that leaked into the map heading the
    operator would watch the icon shimmy while the robot walked in a straight
    line.  The standard ZYX-Euler yaw term isolates the heading component.

    Raises:
        ValueError: if the quaternion has effectively zero length (a dropped
            pose), rather than silently reporting yaw 0.
    """
    w, x, y, z = (float(v) for v in quat_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < _MIN_QUAT_NORM:
        raise ValueError(
            f"degenerate quaternion (norm={norm!r}); refusing to report a heading"
        )
    # Normalise -- Isaac hands back float32 and composed rotations drift.
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


@dataclass(frozen=True)
class IsaacFrame:
    """Converts poses between a USD stage and Tritium local ENU metres.

    Args:
        meters_per_unit: the stage's ``metersPerUnit`` metadata.  Isaac
            defaults to 1.0; centimetre-authored stages use 0.01.
        up_axis: the stage's ``upAxis`` metadata -- ``"Z"`` or ``"Y"``.
        origin_offset: where the Tritium geo-reference origin sits within the
            stage, **in stage units**.  Lets the operator's (0, 0) be a city
            block rather than the stage origin.
    """

    meters_per_unit: float = 1.0
    up_axis: str = "Z"
    origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.up_axis not in _UP_AXES:
            raise ValueError(
                f"up_axis must be one of {_UP_AXES}, got {self.up_axis!r}"
            )
        if self.meters_per_unit <= 0.0:
            raise ValueError(
                f"meters_per_unit must be positive, got {self.meters_per_unit!r}"
            )

    @classmethod
    def from_stage_metadata(cls, meta: Mapping[str, Any]) -> "IsaacFrame":
        """Build a frame from a stage-metadata payload.

        The key names match the Isaac MCP bridge's ``/health`` result
        (``up_axis``, ``meters_per_unit``) so a live bridge can hand its reply
        straight in.  Missing keys fall back to Isaac's own defaults.
        """
        return cls(
            meters_per_unit=float(meta.get("meters_per_unit", 1.0) or 1.0),
            up_axis=str(meta.get("up_axis", "Z") or "Z").upper(),
            origin_offset=tuple(meta.get("origin_offset", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
        )

    # -- position ---------------------------------------------------------

    def stage_to_local(
        self, xyz: Sequence[float]
    ) -> tuple[float, float, float]:
        """Stage translation -> ``(east_m, north_m, up_m)``.

        The origin offset is subtracted in stage units first, then the result
        is scaled to metres -- offsets authored by looking at the USD viewport
        are naturally in the stage's own units.
        """
        sx, sy, sz = (float(v) for v in xyz)
        ox, oy, oz = self.origin_offset
        sx, sy, sz = sx - ox, sy - oy, sz - oz
        if self.up_axis == "Z":
            east, north, up = sx, sy, sz
        else:  # Y-up: ground plane is XZ and north is -Z (right-handed)
            east, north, up = sx, -sz, sy
        s = self.meters_per_unit
        return (east * s, north * s, up * s)

    def local_to_stage(
        self, enu: Sequence[float]
    ) -> tuple[float, float, float]:
        """``(east_m, north_m, up_m)`` -> stage translation.  Inverse of above."""
        east, north, up = (float(v) / self.meters_per_unit for v in enu)
        if self.up_axis == "Z":
            sx, sy, sz = east, north, up
        else:
            sx, sy, sz = east, up, -north
        ox, oy, oz = self.origin_offset
        return (sx + ox, sy + oy, sz + oz)

    # -- heading ----------------------------------------------------------

    def yaw_to_heading(self, yaw_deg: float) -> float:
        """Isaac yaw (CCW from +X/east) -> Tritium heading (CW from north)."""
        return (90.0 - float(yaw_deg)) % 360.0

    def heading_to_yaw(self, heading_deg: float) -> float:
        """Tritium heading -> Isaac yaw.  Self-inverse with ``yaw_to_heading``."""
        return (90.0 - float(heading_deg)) % 360.0

    # -- the whole pose ---------------------------------------------------

    def pose_to_local(
        self,
        translation: Sequence[float],
        quat_wxyz: Sequence[float],
    ) -> LocalPose:
        """Full prim pose -> operator pose.  The call a pose bridge makes."""
        east, north, up = self.stage_to_local(translation)
        heading = self.yaw_to_heading(quat_to_yaw_deg(quat_wxyz))
        return LocalPose(east_m=east, north_m=north, up_m=up, heading_deg=heading)
