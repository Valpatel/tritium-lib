# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Grade a ground-truth pose trace against a planned route.

This is the referee for "did the body actually walk the path?".  It takes
three things a controller cannot forge — where the body *was* (ground truth
poses), where it was *told* to go (the planner's polyline), and what was in
the way (scene boxes) — and returns a single verdict plus the numbers behind
it.

The design rule is that **nothing the controller says about itself is an
input**.  A follower publishes its own cross-track error every tick; that
number is computed from the follower's own idea of where it is, so a body
that is lost reports beautiful tracking right up until it walks into a wall.
Recomputing the same quantity from ground truth cannot lie in that direction.
The controller's telemetry is a debugging aid; this module is the grade.

Three properties are load-bearing:

**COLLIDED outranks REACHED.**  A run that arrives at the goal *through* a
wall is not a success with an asterisk, it is a failure — the same ranking
``score_trace`` uses when it puts TUMBLED above distance walked.  The
underlying facts stay honest (``reached_goal`` is still ``True``); only the
headline verdict is ordered.

**Progress is a monotone projection, not a path length.**  ``progress_ratio``
is the furthest arclength the body ever projected to along the polyline, over
the polyline's total length.  Integrating distance travelled instead would let
a body oscillate over the first two metres and score a completed route; taking
the final projection instead would let a shove backwards erase real progress.
A body that only ever moves sideways off the route projects to s = 0 and
accrues nothing, which is the case that catches a follower steering into open
ground.

**Cross-track is measured to the route SEGMENT, not the nearest waypoint.**
On a 10 m leg a body can be metres off the line while sitting close to a node,
so waypoint distance flatters long legs by an order of magnitude.  The
definition — and the geometry — is shared with
:mod:`tritium_lib.control.waypoint_follower` rather than re-derived here.

Obstacles are :class:`~tritium_lib.planning.scene_costmap.SceneObstacle` boxes
in world meters, filtered through the same two tests the costmap applies, so
the scorer and the planner agree on what is a wall and what is the world:

- the **body band** — a box only counts if its vertical span overlaps the slab
  the body sweeps.  A gantry overhead and the ground slab underfoot are both
  boxes, and neither is something a walking body collides with.
- the **footprint cap** — a box wider than ``max_footprint_m`` is terrain, not
  an obstacle.  A live stage's ground mesh came back with 1519 m half-extents
  spanning z=-24..+33: it passes the band test, so without the cap every
  sample is inside it and every run grades COLLIDED.  A metric that returns
  the same answer for every run measures nothing, and this is the failure mode
  that makes it happen silently.

Both filters can be opted out of explicitly (``max_footprint_m=None``) for a
caller that has already curated its obstacle list.

Pure stdlib — no simulator, no ROS, no numpy — so the same scorer runs in a
unit test, in an Isaac Sim harness, and on the robot's own brain.

Typical use::

    score = score_route_trace(
        pose_log,                       # [(x, y), ...] ground truth
        route,                          # the planner's polyline
        obstacles,                      # what was in the way
        goal_tolerance_m=0.35,
        clearance_m=0.10,               # inflate the footprint by the body
    )
    if score.verdict != "REACHED":
        raise AssertionError(score)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from tritium_lib.control.waypoint_follower import (
    Point,
    _distance,
    _segment_distance,
    cross_track_distance,
)
from tritium_lib.planning.scene_costmap import (
    DEFAULT_BODY_BAND,
    DEFAULT_MAX_FOOTPRINT_M,
    SceneObstacle,
)

__all__ = [
    "DEFAULT_CLEARANCE_M",
    "DEFAULT_GOAL_TOLERANCE_M",
    "DEFAULT_MAX_FOOTPRINT_M",
    "RouteScore",
    "score_route_trace",
]

#: A body is "there" once it is inside this radius of the final waypoint.
#: Roughly a Go2 body length — tighter than this and a legged gait's normal
#: stance sway decides the verdict.
DEFAULT_GOAL_TOLERANCE_M: float = 0.35

#: Footprint inflation, meters.  ``0.0`` means "graded on literal penetration
#: of the box" — the caller passes its own body radius to grade on the body's
#: hull instead of its centre point.
DEFAULT_CLEARANCE_M: float = 0.0

#: Verdicts, worst first.  The order *is* the ranking rule: the first one that
#: applies wins, so a collision can never be outvoted by an arrival.
VERDICT_ORDER: tuple[str, ...] = ("NO_TRACE", "COLLIDED", "REACHED", "SHORT")


@dataclass(frozen=True)
class RouteScore:
    """The grade for one run of a body along one planned route.

    Every field is recomputed from ground truth.  ``reached_goal`` and
    ``collided`` are independent facts; ``verdict`` is the ranked summary of
    them, and a caller that wants the unranked truth should read the booleans.

    Attributes:
        verdict: ``"NO_TRACE"``, ``"COLLIDED"``, ``"REACHED"`` or ``"SHORT"``.
        reached_goal: Final position within ``goal_tolerance_m`` of the last
            waypoint.  Stays true even when the verdict is ``COLLIDED``.
        final_gap_m: Distance from the last pose to the goal waypoint.
        max_cross_track_m: Worst distance from any sample to the route
            polyline.  The number that catches a corner cut.
        rms_cross_track_m: Quadratic mean of the same, over all samples — the
            number that catches a follower that is merely sloppy.
        min_clearance_m: Closest approach of any sample to any obstacle that
            blocks the body band.  Negative inside a box, ``inf`` when nothing
            blocks the band.
        collided: ``min_clearance_m < clearance_m``.
        progress_ratio: Monotone arclength progress along the route, in
            ``[0.0, 1.0]``.  Cannot exceed 1.0 and cannot be inflated by
            backtracking or oscillation.
        samples: Number of poses graded.
    """

    verdict: str
    reached_goal: bool
    final_gap_m: float
    max_cross_track_m: float
    rms_cross_track_m: float
    min_clearance_m: float
    collided: bool
    progress_ratio: float
    samples: int


def _as_point(pose: Sequence[float]) -> Point:
    """Take the planar part of a pose. ``(x, y)`` and ``(x, y, z)`` both work.

    Height is deliberately dropped: a ground body's tracking error is a
    2-D quantity, and folding z into it would score a body walking up a ramp
    as off-route.  Height still matters for obstacles, where it enters through
    the body band instead.
    """
    return (float(pose[0]), float(pose[1]))


def _blocking(
    obstacles: Sequence[SceneObstacle],
    body_band: tuple[float, float],
    max_footprint_m: float | None,
) -> list[SceneObstacle]:
    """Boxes that a walking body can actually hit.

    Two filters, both mirroring :func:`obstacles_to_feature_collection` so the
    scorer and the planner draw the same line between "obstacle" and "world":

    1. **Body band** — via :meth:`SceneObstacle.intersects_band`.  A gantry
       overhead and the ground slab underfoot are boxes a body walks under and
       on, not into.
    2. **Footprint cap** — world-scale geometry is terrain, not a wall.  This
       one is load-bearing on a live stage: a real ground mesh came back with
       1519 m half-extents spanning z=-24..+33, which *passes* the band test.
       Without the cap every sample sits inside it, every run scores COLLIDED,
       and the metric silently measures nothing.  Worse, the planner already
       rejects that box, so a route would be planned straight through geometry
       the scorer then fails you for.

    The cap rejects on ``edge > max_footprint_m``, so a box exactly at the cap
    is kept — identical to the planner's comparison.
    """
    band_min, band_max = body_band
    if band_min >= band_max:
        raise ValueError(
            f"body_band must be (min, max) with min < max, got {body_band!r}"
        )
    if max_footprint_m is not None and max_footprint_m < 0.0:
        raise ValueError(
            f"max_footprint_m must be >= 0 or None, got {max_footprint_m}"
        )

    kept = []
    for obstacle in obstacles:
        if not obstacle.intersects_band(band_min, band_max):
            continue
        if max_footprint_m is not None:
            hx, hy, _ = obstacle.half_extents
            if max(hx, hy) * 2.0 > max_footprint_m:
                continue
        kept.append(obstacle)
    return kept


def _box_distance(point: Point, obstacle: SceneObstacle) -> float:
    """Signed 2-D distance from ``point`` to a box footprint, negative inside.

    The point is rotated into the box's own frame so yaw is respected, then
    compared against the half-extents — the standard signed distance to an
    axis-aligned rectangle.  Only the XY footprint is used; the z band has
    already decided whether this box is relevant at all.
    """
    cx, cy, _ = obstacle.center
    hx, hy, _ = obstacle.half_extents
    theta = math.radians(obstacle.yaw_deg)
    dx_w, dy_w = point[0] - cx, point[1] - cy
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Rotate by -yaw into the box frame.
    local_x = dx_w * cos_t + dy_w * sin_t
    local_y = -dx_w * sin_t + dy_w * cos_t

    gap_x = abs(local_x) - hx
    gap_y = abs(local_y) - hy
    outside = math.hypot(max(gap_x, 0.0), max(gap_y, 0.0))
    inside = min(max(gap_x, gap_y), 0.0)
    return outside + inside


def _arclengths(route: Sequence[Point]) -> list[float]:
    """Cumulative arclength at each waypoint, starting at 0.0."""
    cumulative = [0.0]
    for i in range(len(route) - 1):
        cumulative.append(cumulative[-1] + _distance(route[i], route[i + 1]))
    return cumulative


def _projected_arclength(
    point: Point, route: Sequence[Point], cumulative: Sequence[float]
) -> float:
    """Arclength of ``point``'s projection onto the polyline.

    The projection is taken on the segment the point is *closest to*, which is
    what makes a sideways excursion score zero progress rather than the
    progress of whichever segment happens to be numerically first.  Each
    segment's parameter is clamped to ``[0, 1]``, so a body beyond the final
    waypoint projects to the end of the route and no further.
    """
    if len(route) < 2:
        return 0.0

    best_distance = math.inf
    best_s = 0.0
    for i in range(len(route) - 1):
        start, end = route[i], route[i + 1]
        distance = _segment_distance(point, start, end)
        if distance >= best_distance:
            continue
        dx, dy = end[0] - start[0], end[1] - start[1]
        span = dx * dx + dy * dy
        if span <= 0.0:
            t = 0.0
        else:
            t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / span
            t = max(0.0, min(1.0, t))
        best_distance = distance
        best_s = cumulative[i] + t * math.sqrt(span)
    return best_s


def score_route_trace(
    positions: Sequence[Sequence[float]],
    route: Sequence[Sequence[float]],
    obstacles: Sequence[SceneObstacle] = (),
    *,
    goal_tolerance_m: float = DEFAULT_GOAL_TOLERANCE_M,
    clearance_m: float = DEFAULT_CLEARANCE_M,
    body_band: tuple[float, float] = DEFAULT_BODY_BAND,
    max_footprint_m: float | None = DEFAULT_MAX_FOOTPRINT_M,
) -> RouteScore:
    """Grade a ground-truth pose trace against a planned route.

    Args:
        positions: Ground-truth body positions in world meters, in time order.
            ``(x, y)`` or ``(x, y, z)``; z is ignored for tracking.  These must
            come from the simulator's or the robot's state, never from the
            controller's own estimate — that is the entire point of the module.
        route: The planner's polyline, same frame.
        obstacles: Scene boxes.  Boxes outside ``body_band`` are ignored.
        goal_tolerance_m: Arrival radius around the final waypoint.
        clearance_m: Footprint inflation.  A sample closer than this to a box
            counts as a collision, so pass the body radius to grade the hull
            rather than the centre point.
        body_band: Vertical span ``(z_min, z_max)`` the body sweeps, shared
            with :mod:`tritium_lib.planning.scene_costmap`.
        max_footprint_m: Boxes whose larger footprint edge exceeds this are
            treated as terrain and excluded from both ``min_clearance_m`` and
            the collision verdict — the same cap the planner applies, so the
            two agree on what is a wall and what is the world.  Pass ``None``
            (or ``math.inf``) to grade every box handed in, which is what a
            caller with an already-curated obstacle list wants.

    Returns:
        A :class:`RouteScore`.  Fewer than two poses, or an empty route, gives
        ``NO_TRACE`` — a snapshot is not a walk, and grading one as ``SHORT``
        would make a harness that forgot to log look like a control failure.

    Raises:
        ValueError: On a negative ``goal_tolerance_m``, a negative
            ``clearance_m``, a negative ``max_footprint_m``, or an inverted
            ``body_band``.
    """
    if goal_tolerance_m < 0.0:
        raise ValueError(f"goal_tolerance_m must be >= 0, got {goal_tolerance_m}")
    if clearance_m < 0.0:
        raise ValueError(f"clearance_m must be >= 0, got {clearance_m}")
    blocking = _blocking(obstacles, body_band, max_footprint_m)

    samples = [_as_point(p) for p in positions]
    waypoints = [_as_point(p) for p in route]

    if len(samples) < 2 or not waypoints:
        return RouteScore(
            verdict="NO_TRACE",
            reached_goal=False,
            final_gap_m=math.inf,
            max_cross_track_m=math.inf,
            rms_cross_track_m=math.inf,
            min_clearance_m=math.inf,
            collided=False,
            progress_ratio=0.0,
            samples=len(samples),
        )

    goal = waypoints[-1]
    final_gap = _distance(samples[-1], goal)
    reached_goal = final_gap <= goal_tolerance_m

    errors = [cross_track_distance(p, waypoints) for p in samples]
    max_cross_track = max(errors)
    rms_cross_track = math.sqrt(sum(e * e for e in errors) / len(errors))

    min_clearance = math.inf
    for point in samples:
        for obstacle in blocking:
            min_clearance = min(min_clearance, _box_distance(point, obstacle))
    collided = min_clearance < clearance_m

    # Monotone projection: the furthest the body ever got, never the sum of
    # where it went.  Running-max rather than plain max to make the ratchet
    # explicit -- this is the property the metric lives or dies on.
    cumulative = _arclengths(waypoints)
    total_length = cumulative[-1]
    furthest = 0.0
    for point in samples:
        furthest = max(furthest, _projected_arclength(point, waypoints, cumulative))
    if total_length > 0.0:
        progress_ratio = max(0.0, min(1.0, furthest / total_length))
    else:
        # A one-waypoint "route" has no arclength to progress along; arrival
        # is the only meaningful statement about it.
        progress_ratio = 1.0 if reached_goal else 0.0

    if collided:
        verdict = "COLLIDED"
    elif reached_goal:
        verdict = "REACHED"
    else:
        verdict = "SHORT"

    return RouteScore(
        verdict=verdict,
        reached_goal=reached_goal,
        final_gap_m=final_gap,
        max_cross_track_m=max_cross_track,
        rms_cross_track_m=rms_cross_track,
        min_clearance_m=min_clearance,
        collided=collided,
        progress_ratio=progress_ratio,
        samples=len(samples),
    )
