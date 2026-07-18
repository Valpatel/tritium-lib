# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Collision geometry a convex-hull solver can actually chew.

Most modern physics backends — Newton's MuJoCo solver, PhysX's convex meshes,
Bullet — do not simulate an arbitrary triangle soup.  They first reduce each
collision shape to a convex hull, and a hull is built by picking four vertices
that bound a volume.  If every vertex of a shape lies on one plane there is no
such quartet, and the hull step fails.

That sounds like a corner case until you notice which shape hits it first: the
ground.  The obvious way to author a floor is a big flat quad — four corners,
all at the same height — and that shape is exactly rank 2.  Isaac Sim 6.0 with
Newton fails on it with qhull's "Initial simplex is flat (facet 1 is coplanar
with the interior point)", and it fails *quietly*: the exception escapes the
physics extension's initializer after an `_initializing` guard has been set and
before it is cleared, so every later attempt to initialize returns immediately
at that guard.  The result is a simulator that looks completely healthy — the
timeline plays, the frame counter climbs, sim time advances by dt every step —
while nothing whatsoever is integrated.  A 1 kg cube released from 3 m falls
0.0 m in 4 s of "simulation".  The only evidence is a single log line.

The lesson generalizes past that one bug: geometry that will break the solver
must be caught while it is still a list of vertices, before it is authored into
a stage, because afterwards the failure is both silent and global.  Catching it
there also means the check must not need the solver — no GPU, no USD, no Isaac,
no numpy.  This module is stdlib-only for that reason, and it imports on a bare
Jetson and inside a simulator's embedded Python alike.

Conventions:
  * Points are ``(x, y, z)`` in stage units.  The module is unit-agnostic; the
    tolerance is expressed in whatever unit the caller's points are in.
  * "Degenerate" is a property of the point SET (its rank), not of any single
    axis.  An axis-aligned extent test is a cheaper screen that misses tilted
    planes, so it is reported for context but never used as the verdict.
  * Two tiers, matching :mod:`tritium_lib.geo.camera_mount`: strict
    ``validate_*`` raises :class:`ValueError` for scene builders that should
    fail loudly, forgiving ``check_*`` returns a report and never raises, for
    bridges that would rather log and degrade.

The rank test mirrors how qhull picks its own initial simplex — span the widest
segment, then the point furthest off that line, then the point furthest off the
resulting plane — so a shape that passes here is a shape qhull can seed from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ShapeReport",
    "aabb_extent",
    "SlabSpec",
    "check_convex_hull_input",
    "ground_slab",
    "validate_convex_hull_input",
]

Point = tuple[float, float, float]

# Distances below this are treated as floating-point noise rather than real
# separation.  Chosen a few orders of magnitude above double-precision round-off
# on metre-scale scenes, and well below any thickness a person would author on
# purpose (a 1 mm shell is 1e-3, a thousand times this).
_DEFAULT_TOLERANCE = 1e-6


# --- reports -----------------------------------------------------------------


@dataclass(frozen=True)
class ShapeReport:
    """The verdict on one collision shape, plus enough context to fix it.

    ``rank`` is the dimension the vertices actually span: 0 for a single
    repeated point, 1 for a line, 2 for a plane, 3 for a volume.  Only rank 3
    can be hulled.  ``plane_normal`` is set only when the shape is degenerate
    and spans at least a line — it is the direction the shape has no thickness
    in, which is the direction to give it some.
    """

    is_hullable: bool
    rank: int
    extent: tuple[float, float, float]
    reason: str = ""
    plane_normal: Point | None = None


# --- axis-aligned extent -----------------------------------------------------


def aabb_extent(points: list[Point] | tuple[Point, ...]) -> tuple[float, float, float]:
    """Width, depth and height of the axis-aligned box bounding ``points``.

    Useful for logging and as a cheap first screen — a zero extent on any axis
    proves degeneracy — but a non-zero extent on all three proves nothing, so
    never branch on this alone.  See :func:`check_convex_hull_input`.
    """
    if not points:
        raise ValueError("points must contain at least one point, got an empty sequence")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


# --- rank ---------------------------------------------------------------------


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Point, b: Point) -> Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Point) -> float:
    return math.sqrt(_dot(a, a))


def _rank_and_normal(
    points: list[Point] | tuple[Point, ...], tolerance: float
) -> tuple[int, Point | None]:
    """Span-widest-first rank determination, the way qhull seeds a simplex.

    Returns the rank (0-3) and, when the set is exactly rank 2, the unit normal
    of the plane every vertex lies on.
    """
    origin = points[0]

    # Rank >= 1?  Find the vertex furthest from the first one.
    far_a, dist_a = None, 0.0
    for p in points:
        d = _norm(_sub(p, origin))
        if d > dist_a:
            far_a, dist_a = p, d
    if far_a is None or dist_a <= tolerance:
        return 0, None

    axis = _sub(far_a, origin)

    # Rank >= 2?  Find the vertex furthest off that line.  The cross product's
    # magnitude divided by the axis length IS that perpendicular distance, so
    # this needs no projection step.
    far_b, dist_b, normal = None, 0.0, None
    for p in points:
        n = _cross(axis, _sub(p, origin))
        d = _norm(n) / dist_a
        if d > dist_b:
            far_b, dist_b, normal = p, d, n
    if far_b is None or dist_b <= tolerance or normal is None:
        return 1, None

    # Rank == 3?  Find the vertex furthest off the plane those three span.
    length = _norm(normal)
    unit = (normal[0] / length, normal[1] / length, normal[2] / length)
    off_plane = max(abs(_dot(_sub(p, origin), unit)) for p in points)
    if off_plane <= tolerance:
        return 2, unit
    return 3, None


# --- the checks ---------------------------------------------------------------


def check_convex_hull_input(
    points: list[Point] | tuple[Point, ...],
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> ShapeReport:
    """Decide whether ``points`` can seed a convex hull.  Never raises.

    The forgiving half of the pair: a bridge handed geometry from an outside
    system wants to log a bad shape and carry on, not die on it.  Scene builders
    that own the geometry should prefer :func:`validate_convex_hull_input`.

    ``tolerance`` is the separation, in the caller's units, below which two
    points count as the same place.  Raise it to reject shapes that are
    technically three-dimensional but too thin for a particular solver to
    condition well.
    """
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance!r}")

    if len(points) < 4:
        return ShapeReport(
            is_hullable=False,
            rank=0 if not points else -1,
            extent=(0.0, 0.0, 0.0) if not points else aabb_extent(points),
            reason=(
                f"a convex hull needs at least 4 vertices to bound a volume, "
                f"got {len(points)}"
            ),
        )

    extent = aabb_extent(points)
    rank, normal = _rank_and_normal(points, tolerance)

    if rank == 3:
        return ShapeReport(is_hullable=True, rank=3, extent=extent)

    if rank == 2:
        reason = (
            "every vertex is coplanar (rank 2), so no convex hull can be built "
            "from it; give the shape thickness along its plane normal"
        )
    elif rank == 1:
        reason = "every vertex is collinear (rank 1), so the shape encloses no volume"
    else:
        reason = "every vertex is at the same position (rank 0); the shape has no size"

    return ShapeReport(
        is_hullable=False,
        rank=rank,
        extent=extent,
        reason=reason,
        plane_normal=normal,
    )


def validate_convex_hull_input(
    points: list[Point] | tuple[Point, ...],
    *,
    name: str = "<unnamed shape>",
    tolerance: float = _DEFAULT_TOLERANCE,
) -> None:
    """Raise :class:`ValueError` if ``points`` cannot seed a convex hull.

    ``name`` should be the prim path or mesh name, because the entire value of
    failing here rather than in the solver is that the author is told *which*
    shape to fix.  The alternative is a process whose physics is dead with no
    indication of the cause.
    """
    report = check_convex_hull_input(points, tolerance=tolerance)
    if report.is_hullable:
        return
    raise ValueError(
        f"collider {name!r} cannot be convex-hulled: {report.reason} "
        f"(rank {report.rank}, extent {report.extent}). A physics backend that "
        f"hulls its collision shapes will fail on this, and may do so silently."
    )


# --- ground slab -------------------------------------------------------------


@dataclass(frozen=True)
class SlabSpec:
    """A box a scene builder can author as ground, described how ground is used.

    The caller cares about one number the box form does not carry: the height of
    the surface bodies stand on.  Authoring a box by its centre is precisely the
    step where thickness silently sinks that surface by half a slab, so this
    type carries ``top_z`` and derives the centre from it rather than the other
    way round.

    ``vertices`` are the eight corners, ready to hand to
    :func:`check_convex_hull_input` — a slab always passes, and asserting that
    in the caller's own tests costs nothing.
    """

    center: Point
    half_extents: Point
    top_z: float

    @property
    def vertices(self) -> list[Point]:
        """The eight corners of the box, in stage units."""
        cx, cy, cz = self.center
        hx, hy, hz = self.half_extents
        return [
            (cx + sx * hx, cy + sy * hy, cz + sz * hz)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]


def ground_slab(
    *,
    size_m: float = 100.0,
    thickness_m: float = 1.0,
    top_z: float = 0.0,
) -> SlabSpec:
    """Ground geometry a hull solver accepts, with its surface at ``top_z``.

    This is the replacement for the flat quad described at the top of this
    module.  The footprint is unchanged from how a person would author a floor;
    the only difference is that the shape has thickness, which takes the vertex
    set from rank 2 to rank 3 and lets qhull seed its initial simplex.

    A default 1 m thickness is far more than any solver needs numerically.  It
    is chosen so that a fast-moving body cannot tunnel through the floor in one
    step, which is the failure that replaces the hull failure once the hull
    failure is fixed.
    """
    if thickness_m <= 0.0:
        raise ValueError(
            f"thickness_m must be positive, got {thickness_m!r}. A zero-thickness "
            f"ground is the coplanar quad this builder exists to replace."
        )
    if size_m <= 0.0:
        raise ValueError(f"size_m must be positive, got {size_m!r}")
    half = size_m / 2.0
    half_thickness = thickness_m / 2.0
    return SlabSpec(
        center=(0.0, 0.0, top_z - half_thickness),
        half_extents=(half, half, half_thickness),
        top_z=top_z,
    )
