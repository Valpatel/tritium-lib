# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Route following — the seam between a planner and a body.

A planner produces a polyline through free space.  A body cannot obey a
polyline; it can only obey a velocity.  This module is the piece in between,
and it is deliberately the *only* piece that knows both.

Two stages, kept apart on purpose:

  1. :class:`PurePursuitFollower` turns ``(pose, route)`` into a
     :class:`TwistCommand` — forward speed and yaw rate, the universal robot
     command.  This is body-agnostic in the strong sense: a quadruped, a
     rover and a boat differ in nothing it can observe.
  2. :func:`differential_stride` turns that twist into a left/right stride
     ratio for a body that steers by driving its two sides at different
     speeds.  The body enters as *data* (track width, nominal speed), never
     as a subclass.

Splitting there matters because stage 1 is what a rover and an aerial body
share and stage 2 is what a quadruped and a tracked vehicle share.  A body
with a steerable axle replaces stage 2 alone and keeps the follower.

The algorithm is pure pursuit (Coulter, CMU-RI-TR-92-01) — the standard
tracker, not an invented gain schedule.  It steers toward a point one
lookahead distance along the path rather than at the nearest waypoint, which
is what stops a body from sawing back and forth across its own route.

Stdlib only, so it imports on a bare Jetson next to the rest of the brain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "FollowState",
    "PurePursuitFollower",
    "StrideBias",
    "TwistCommand",
    "cross_track_distance",
    "differential_stride",
]

Point = tuple[float, float]

#: Lookahead below this is meaningless — curvature goes as 1/L and a body
#: chasing a point on its own hull oscillates instead of tracking.
MIN_LOOKAHEAD_M = 1e-3


@dataclass(frozen=True)
class TwistCommand:
    """A planar body velocity: forward speed and yaw rate.

    REP-103 conventions — +X forward, +Z up, yaw positive counter-clockwise,
    so a positive ``angular_rps`` turns to port (left).
    """

    linear_mps: float
    angular_rps: float

    @classmethod
    def stop(cls) -> "TwistCommand":
        """The command to issue when there is nothing safe to do."""
        return cls(linear_mps=0.0, angular_rps=0.0)


@dataclass(frozen=True)
class FollowState:
    """One tick of route following, including the metrics that grade it.

    ``cross_track_m`` is the honest tracking number: distance from the body to
    the route *segment* it is on, not to the nearest waypoint.  Distance to a
    waypoint flatters a follower on long legs, where a body can be metres off
    the line while still close to a node.
    """

    twist: TwistCommand
    target_index: int
    lookahead_point: Point
    cross_track_m: float
    distance_to_goal_m: float
    arrived: bool


@dataclass(frozen=True)
class StrideBias:
    """Per-side stride scaling for a skid-steered body.

    ``1.0`` is the nominal stride.  Negative means that side walks backwards,
    which is how a body spins in place.
    """

    left_scale: float
    right_scale: float


def _distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _segment_distance(point: Point, start: Point, end: Point) -> float:
    """Perpendicular distance to a segment, clamped to the endpoints."""
    sx, sy = start
    dx, dy = end[0] - sx, end[1] - sy
    span = dx * dx + dy * dy
    if span <= 0.0:  # duplicate points — degenerate to a node distance
        return _distance(point, start)
    t = ((point[0] - sx) * dx + (point[1] - sy) * dy) / span
    t = max(0.0, min(1.0, t))
    return _distance(point, (sx + t * dx, sy + t * dy))


def cross_track_distance(point: Point, route: Sequence[Point]) -> float:
    """Distance from ``point`` to the nearest *segment* of ``route``.

    The honest tracking metric, and deliberately a free function rather than a
    follower method: the offline grader in
    :mod:`tritium_lib.control.route_trace` recomputes this from ground-truth
    poses and has no follower to ask.  Making it standalone is what keeps the
    live number and the graded number the same definition instead of two
    implementations that drift.

    Measuring to the nearest *waypoint* instead would flatter a follower on
    long legs, where a body can sit metres off the line while close to a node.
    A single-waypoint route degenerates to plain point distance.
    """
    if not route:
        raise ValueError("cross_track_distance needs at least one waypoint")
    if len(route) == 1:
        return _distance(point, route[0])
    return min(
        _segment_distance(point, route[i], route[i + 1])
        for i in range(len(route) - 1)
    )


class PurePursuitFollower:
    """Steer a body along a polyline by chasing a point ahead of it.

    The follower is stateful in exactly one respect: it remembers how far
    along the route it has progressed, and never targets a waypoint behind
    that mark.  Without the ratchet a body shoved sideways can re-acquire a
    waypoint it already passed and loop.
    """

    def __init__(
        self,
        lookahead_m: float = 0.6,
        cruise_mps: float = 0.25,
        max_angular_rps: float = 0.8,
        goal_tolerance_m: float = 0.25,
        slow_radius_m: float = 0.0,
    ) -> None:
        if lookahead_m <= MIN_LOOKAHEAD_M:
            raise ValueError(
                f"lookahead_m must be > {MIN_LOOKAHEAD_M}, got {lookahead_m}"
            )
        if cruise_mps < 0.0:
            raise ValueError(f"cruise_mps must be >= 0, got {cruise_mps}")
        if max_angular_rps < 0.0:
            raise ValueError(
                f"max_angular_rps must be >= 0, got {max_angular_rps}"
            )
        self.lookahead_m = float(lookahead_m)
        self.cruise_mps = float(cruise_mps)
        self.max_angular_rps = float(max_angular_rps)
        self.goal_tolerance_m = float(goal_tolerance_m)
        self.slow_radius_m = float(slow_radius_m)
        self._progress_index = 0

    def reset(self) -> None:
        """Forget route progress — call between runs, not between ticks."""
        self._progress_index = 0

    def update(
        self, pose: tuple[float, float, float], route: Sequence[Point]
    ) -> FollowState:
        """Command the body toward ``route`` given its current ``pose``.

        ``pose`` is ``(x, y, heading_rad)`` in the same frame as the route.
        """
        position: Point = (float(pose[0]), float(pose[1]))
        heading = float(pose[2])

        if not route:
            # No route is not the same as "arrived" — refuse to invent one.
            return FollowState(
                twist=TwistCommand.stop(),
                target_index=0,
                lookahead_point=position,
                cross_track_m=0.0,
                distance_to_goal_m=0.0,
                arrived=False,
            )

        points = [(float(x), float(y)) for x, y in route]
        goal = points[-1]
        distance_to_goal = _distance(position, goal)
        cross_track = self._cross_track(position, points)

        if distance_to_goal <= self.goal_tolerance_m:
            return FollowState(
                twist=TwistCommand.stop(),
                target_index=len(points) - 1,
                lookahead_point=goal,
                cross_track_m=cross_track,
                distance_to_goal_m=distance_to_goal,
                arrived=True,
            )

        index, target = self._lookahead_target(position, points)

        # Pure pursuit: steer on the bearing to the lookahead point, expressed
        # in the body frame.  alpha is the angle the body must swing through.
        bearing = math.atan2(target[1] - position[1], target[0] - position[0])
        alpha = _wrap_to_pi(bearing - heading)

        linear = self._speed_for(distance_to_goal)
        curvature = 2.0 * math.sin(alpha) / self.lookahead_m
        angular = linear * curvature

        # A body that must turn nearly in place has little forward demand but
        # still needs authority, or it creeps off-route with the wheel over.
        if linear <= 0.0:
            angular = math.copysign(self.max_angular_rps, alpha) if alpha else 0.0
        angular = max(-self.max_angular_rps, min(self.max_angular_rps, angular))

        return FollowState(
            twist=TwistCommand(linear_mps=linear, angular_rps=angular),
            target_index=index,
            lookahead_point=target,
            cross_track_m=cross_track,
            distance_to_goal_m=distance_to_goal,
            arrived=False,
        )

    def _speed_for(self, distance_to_goal: float) -> float:
        """Cruise, tapering inside ``slow_radius_m`` so the body can stop."""
        if self.slow_radius_m <= 0.0:
            return self.cruise_mps
        ratio = min(1.0, distance_to_goal / self.slow_radius_m)
        return self.cruise_mps * ratio

    def _lookahead_target(
        self, position: Point, points: Sequence[Point]
    ) -> tuple[int, Point]:
        """First waypoint at least one lookahead away, ratcheting forward.

        The scan must start from where the body actually *is* on the route,
        not from the head of the list: a body near the end is far from
        waypoint 0, so a scan from index 0 would happily "acquire" the start
        point behind it and drive back down the route.
        """
        nearest = min(
            range(len(points)), key=lambda i: _distance(position, points[i])
        )
        self._progress_index = max(self._progress_index, nearest)
        start = min(self._progress_index, len(points) - 1)
        for index in range(start, len(points)):
            if _distance(position, points[index]) >= self.lookahead_m:
                self._progress_index = max(self._progress_index, index)
                return index, points[index]
        # Everything left is inside the lookahead circle: aim at the goal.
        last = len(points) - 1
        self._progress_index = max(self._progress_index, last)
        return last, points[last]

    def _cross_track(self, position: Point, points: Sequence[Point]) -> float:
        """Distance to the nearest segment of the route.

        Delegates to :func:`cross_track_distance` so the follower's live
        telemetry and the offline grade are the same computation.
        """
        return cross_track_distance(position, points)


def _wrap_to_pi(angle: float) -> float:
    """Fold an angle into (-pi, pi] so a turn takes the shorter way round."""
    return math.atan2(math.sin(angle), math.cos(angle))


def differential_stride(
    twist: TwistCommand,
    track_width_m: float,
    nominal_mps: float,
    max_scale: float = 2.0,
) -> StrideBias:
    """Mix a twist into left/right stride scales for a skid-steered body.

    The standard differential-drive law::

        v_left  = v - omega * W / 2
        v_right = v + omega * W / 2

    expressed as a ratio against the stride the gait already walks at, so a
    caller scales its existing trajectory rather than regenerating one.  A
    positive (port) yaw shortens the left side, which is what makes the body
    pivot left — the sign is the whole point, and mixing that drops it turns
    nothing while looking like an underpowered gait.
    """
    if track_width_m <= 0.0:
        raise ValueError(f"track_width_m must be > 0, got {track_width_m}")
    if nominal_mps <= 0.0:
        raise ValueError(f"nominal_mps must be > 0, got {nominal_mps}")

    half = twist.angular_rps * track_width_m / 2.0
    left = (twist.linear_mps - half) / nominal_mps
    right = (twist.linear_mps + half) / nominal_mps
    limit = abs(max_scale)
    return StrideBias(
        left_scale=max(-limit, min(limit, left)),
        right_scale=max(-limit, min(limit, right)),
    )
