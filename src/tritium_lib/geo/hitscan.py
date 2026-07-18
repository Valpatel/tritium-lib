# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Where the barrel points, and what the ray reaches.

A weapon on a body needs two things that have nothing to do with any
simulator: the world pose of the muzzle TIP given the body's pose and the
turret's slew, and the first thing a ray from that tip runs into.  Both are
geometry.  Neither belongs in a connector, and until now the only 3-D version
in the repo lived inside the Isaac quadruped server, where a real turret on a
real robot could not reach it.

This module is that geometry and nothing else.  It knows no engine, holds no
state, rolls no dice, and does no damage -- it answers *what did the ray
touch*, and leaves *what that costs the target* to
:mod:`tritium_lib.sim_engine.damage`.  Splitting it there is deliberate: the
damage model is probabilistic by design (it rolls against a hit chance), and a
hitscan weapon aimed at a simulated body must be **deterministic**, or the same
shot from the same pose grades differently on replay and no A/B is possible.

**Hitscan, not ballistics.** The ray is analytic and instantaneous, which is
how nearly every game and most short-range fire-control systems resolve a
shot, and it deliberately assumes no PhysX/Newton/Bullet raycast API so the
same call works against a live simulator, a recorded trace, or a real robot's
own target list.  Travel-time projectiles with dispersion and terrain
occlusion already exist in :mod:`tritium_lib.sim_engine.combat` -- reach for
those when the round takes time to arrive; reach for this when it does not.

Frame conventions, matching :mod:`tritium_lib.geo.camera_mount` and
:class:`~tritium_lib.geo.isaac_frame.LocalPose`:

* **World** is Tritium local ENU: ``east``/``north``/``up`` in metres.
* **Heading** is a COMPASS bearing -- 0 = north, 90 = east, increasing
  clockwise.  This differs from the counter-clockwise yaw used in
  :mod:`tritium_lib.control`, which is why the conversion lives at exactly one
  boundary (:func:`muzzle_from_body`) instead of in every caller's head.
* **Elevation** is positive UP, and turret **pan is positive to the LEFT**,
  both matching :class:`~tritium_lib.geo.camera_mount.CameraMount`.

Standard library only, so it imports on a bare aarch64 brain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .camera_mount import CameraMount
from .isaac_frame import LocalPose

__all__ = [
    "BoxTarget",
    "Muzzle",
    "ShotResult",
    "SphereTarget",
    "muzzle_from_body",
    "ray_aabb",
    "ray_sphere",
    "resolve_shot",
]

Vec3 = tuple[float, float, float]

# Below this, a direction vector is numerically meaningless and normalising it
# amplifies float noise into an arbitrary aim.  Callers get an error rather
# than a shot that goes somewhere unrepeatable.
_MIN_DIRECTION_NORM = 1e-9


# --- targets --------------------------------------------------------------


@dataclass(frozen=True)
class SphereTarget:
    """A target approximated by a bounding sphere.

    A sphere is the right first approximation for a person, a drone, or a
    range dummy: it is rotation-invariant, so a target that spins does not
    change how hard it is to hit, and the test is a handful of flops.
    """

    target_id: str
    east_m: float
    north_m: float
    up_m: float
    radius_m: float = 0.5

    def center(self) -> Vec3:
        return (self.east_m, self.north_m, self.up_m)


@dataclass(frozen=True)
class BoxTarget:
    """A target approximated by an axis-aligned box.

    Walls, vehicles and shipping containers are badly served by a sphere --
    its bounding radius juts far past the corners, so shots that visibly miss
    register as hits.  Axis-aligned is the honest limit here: a box rotated
    off the world axes needs an OBB test, which this deliberately is not.
    """

    target_id: str
    min_east_m: float
    min_north_m: float
    min_up_m: float
    max_east_m: float
    max_north_m: float
    max_up_m: float

    def bounds(self) -> tuple[Vec3, Vec3]:
        return (
            (self.min_east_m, self.min_north_m, self.min_up_m),
            (self.max_east_m, self.max_north_m, self.max_up_m),
        )


Target = SphereTarget | BoxTarget


# --- the barrel -----------------------------------------------------------


@dataclass(frozen=True)
class Muzzle:
    """The world pose of the barrel tip and the direction it points."""

    east_m: float
    north_m: float
    up_m: float
    heading_deg: float
    elevation_deg: float

    def origin(self) -> Vec3:
        return (self.east_m, self.north_m, self.up_m)

    def direction(self) -> Vec3:
        """Unit aim vector in ENU.

        Compass heading means east takes the sine and north the cosine -- the
        opposite of the mathematical convention, and the single most common
        place a fire-control rewrite puts its sign bug.
        """
        head = math.radians(self.heading_deg)
        elev = math.radians(self.elevation_deg)
        horizontal = math.cos(elev)
        return (
            horizontal * math.sin(head),
            horizontal * math.cos(head),
            math.sin(elev),
        )

    def to_dict(self) -> dict:
        return {
            "east_m": self.east_m,
            "north_m": self.north_m,
            "up_m": self.up_m,
            "heading_deg": self.heading_deg,
            "elevation_deg": self.elevation_deg,
        }


def muzzle_from_body(
    body: LocalPose, mount: CameraMount, barrel_m: float = 0.0
) -> Muzzle:
    """Body pose + turret mount -> the pose of the barrel tip.

    The mount transform is not re-derived here.  A weapon mount and a camera
    mount are the same problem -- an offset in body frame, rotated by the
    body's heading, plus a pan/tilt slew off the boresight -- and
    :meth:`CameraMount.world_pose` is already checked against USD's own
    transform to 0.000000 m over the live Go2.  Reimplementing it beside that
    would give the shot and the picture two chances to disagree about where
    the robot's nose is.

    ``barrel_m`` then runs from the turret pivot out along the boresight, so
    the round leaves the muzzle TIP.  It matters more than its size suggests:
    firing from the pivot puts the origin inside the robot's own hull, and any
    self-collision test then reports the body shooting itself.
    """
    pivot = mount.world_pose(body)
    elevation_deg = float(mount.tilt_deg)

    tip = Muzzle(
        east_m=pivot.east_m,
        north_m=pivot.north_m,
        up_m=pivot.up_m,
        heading_deg=pivot.heading_deg,
        elevation_deg=elevation_deg,
    )
    if barrel_m == 0.0:
        return tip

    de, dn, du = tip.direction()
    return Muzzle(
        east_m=tip.east_m + de * barrel_m,
        north_m=tip.north_m + dn * barrel_m,
        up_m=tip.up_m + du * barrel_m,
        heading_deg=tip.heading_deg,
        elevation_deg=elevation_deg,
    )


# --- intersection ---------------------------------------------------------


def _unit(direction: Vec3) -> Vec3:
    dx, dy, dz = direction
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < _MIN_DIRECTION_NORM:
        raise ValueError(f"direction must be non-degenerate, got {direction!r}")
    return (dx / norm, dy / norm, dz / norm)


def ray_sphere(
    origin: Vec3, direction: Vec3, center: Vec3, radius_m: float
) -> float | None:
    """Distance from ``origin`` to the sphere's SURFACE, or None for a miss.

    Geometric (projection / half-chord) form rather than the quadratic: it
    costs one square root instead of two and rejects the common cases -- past
    the target, wide of it -- with a comparison before any root is taken.

    Returning surface range and not centre range is the point.  A shot
    registers where it lands, and grading on the centre systematically
    over-reports how far every round travelled by the target's radius.
    """
    if radius_m < 0.0:
        raise ValueError(f"radius_m must be non-negative, got {radius_m!r}")
    ax, ay, az = _unit(direction)
    ox, oy, oz = origin
    cx, cy, cz = center

    vx, vy, vz = cx - ox, cy - oy, cz - oz
    # Projection of the muzzle->centre vector onto the aim.
    tca = vx * ax + vy * ay + vz * az
    dist2 = vx * vx + vy * vy + vz * vz
    r2 = radius_m * radius_m

    # A centre behind the muzzle is only reachable if the muzzle is INSIDE
    # the sphere; testing tca < 0 alone would refuse a point-blank shot.
    if tca < 0.0 and dist2 > r2:
        return None

    perp2 = dist2 - tca * tca
    if perp2 > r2:
        return None

    # Clamped at zero so a muzzle inside the target reports range 0 rather
    # than the negative near root, which would read as "behind me".
    return max(0.0, tca - math.sqrt(max(0.0, r2 - perp2)))


def ray_aabb(
    origin: Vec3, direction: Vec3, box_min: Vec3, box_max: Vec3
) -> float | None:
    """Distance from ``origin`` to the box's near face, or None for a miss.

    The slab method (Kay & Kajiya): clip the ray against each axis's pair of
    parallel planes and keep the running overlap.  Its one notorious trap is
    a direction component of exactly zero -- the naive ``(min - o) / 0``
    yields NaN, every subsequent comparison against NaN is false, and the
    miss silently reports as a HIT.  The parallel case is therefore branched
    out explicitly and decided by whether the origin lies between the planes.
    """
    ax, ay, az = _unit(direction)
    t_near = 0.0  # never report geometry behind the muzzle
    t_far = math.inf

    for o, d, lo, hi in zip(origin, (ax, ay, az), box_min, box_max):
        if lo > hi:
            raise ValueError(f"box bounds inverted on one axis: {lo} > {hi}")
        if abs(d) < _MIN_DIRECTION_NORM:
            if o < lo or o > hi:
                return None  # parallel to this slab and outside it
            continue  # parallel but within: this axis cannot clip the ray
        t1 = (lo - o) / d
        t2 = (hi - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)
        if t_near > t_far:
            return None

    return t_near


def _closest_approach(origin: Vec3, aim: Vec3, target: Target) -> float | None:
    """Surface distance from the ray to a target it missed, or None if behind.

    Only meaningful for spheres; a box's closest approach needs a different
    computation and a near-miss call on a wall is not a useful thing to say.
    """
    if not isinstance(target, SphereTarget):
        return None
    ox, oy, oz = origin
    cx, cy, cz = target.center()
    vx, vy, vz = cx - ox, cy - oy, cz - oz
    tca = vx * aim[0] + vy * aim[1] + vz * aim[2]
    if tca < 0.0:
        return None  # behind the muzzle: it was never in front to be missed
    perp2 = (vx * vx + vy * vy + vz * vz) - tca * tca
    return max(0.0, math.sqrt(max(0.0, perp2)) - target.radius_m)


# --- resolving a shot -----------------------------------------------------


@dataclass(frozen=True)
class ShotResult:
    """What one round did, in enough detail to re-grade it offline."""

    hit: bool
    muzzle: Muzzle
    max_range_m: float
    target_id: str | None = None
    range_m: float | None = None
    impact_east_m: float | None = None
    impact_north_m: float | None = None
    impact_up_m: float | None = None
    miss_distance_m: float | None = None

    def impact(self) -> Vec3 | None:
        if self.impact_east_m is None:
            return None
        return (self.impact_east_m, self.impact_north_m, self.impact_up_m)

    def to_dict(self) -> dict:
        """A JSON-safe record carrying the muzzle, not just the verdict.

        A trace of bare hit/miss booleans cannot be re-graded when the target
        list or the range gate later turns out to have been wrong, and every
        run that produced one has to be thrown away and re-flown.
        """
        return {
            "hit": self.hit,
            "target_id": self.target_id,
            "range_m": self.range_m,
            "impact": list(self.impact()) if self.impact() else None,
            "miss_distance_m": self.miss_distance_m,
            "max_range_m": self.max_range_m,
            "muzzle": self.muzzle.to_dict(),
            "aim": list(self.muzzle.direction()),
        }


def resolve_shot(
    muzzle: Muzzle, targets: list[Target], max_range_m: float
) -> ShotResult:
    """Fire one hitscan round and report the FIRST thing it reaches.

    Nearest-hit-wins is not an optimisation, it is the model: a round stops in
    the first body it enters, so a solver that reports every intersection lets
    a weapon shoot through the target standing in front of the one it wanted.

    The range gate is applied to the hit that actually occurred rather than
    used to pre-filter, so a distant target hiding behind a near one inside
    the gate cannot be reached by simply extending the range.
    """
    if max_range_m < 0.0:
        raise ValueError(f"max_range_m must be non-negative, got {max_range_m!r}")

    origin = muzzle.origin()
    aim = muzzle.direction()

    best_id: str | None = None
    best_t: float | None = None
    for target in targets:
        if isinstance(target, SphereTarget):
            t = ray_sphere(origin, aim, target.center(), target.radius_m)
        else:
            lo, hi = target.bounds()
            t = ray_aabb(origin, aim, lo, hi)
        if t is None:
            continue
        if best_t is None or t < best_t:
            best_t, best_id = t, target.target_id

    if best_t is not None and best_t <= max_range_m:
        return ShotResult(
            hit=True,
            muzzle=muzzle,
            max_range_m=max_range_m,
            target_id=best_id,
            range_m=best_t,
            impact_east_m=origin[0] + aim[0] * best_t,
            impact_north_m=origin[1] + aim[1] * best_t,
            impact_up_m=origin[2] + aim[2] * best_t,
        )

    approaches = [
        d for d in (_closest_approach(origin, aim, t) for t in targets) if d is not None
    ]
    return ShotResult(
        hit=False,
        muzzle=muzzle,
        max_range_m=max_range_m,
        miss_distance_m=min(approaches) if approaches else None,
    )
