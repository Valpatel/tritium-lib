# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""A LaserScan turned into obstacles the tactical map can draw.

A LiDAR does not report obstacles.  It reports a flat array of distances plus
the two numbers needed to know where each one was aimed, and everything past
that is the consumer's problem: convert polar to Cartesian, discard the beams
that hit nothing, carry the body's pose through, and decide which returns are
the SAME physical thing.  That last step is what turns 1,080 numbers into
"three obstacles", which is the only form an operator, a planner, or a
costmap can use.

This module is that pipeline and nothing else.  It knows no simulator, no ROS
node, and no map widget -- it takes arrays and a pose and returns points and
clusters, so an Isaac-simulated LiDAR, a real Unitree L1, and a fixed scanner
on a doorway all reach the map through one implementation.  Keeping it here
rather than in the Isaac addon where the first caller lives is what stops the
second sensor from re-deriving the trigonometry and mirroring its obstacles.

Conventions, matching :mod:`tritium_lib.control` and the rest of ``geo``:

* **Sensor/body frame** is ROS ``base_link`` / REP-103: +X forward out the
  nose, +Y out the port (left) side, yaw positive COUNTER-CLOCKWISE seen from
  above.  A ROS ``sensor_msgs/LaserScan``'s ``angle_min`` and
  ``angle_increment`` therefore transcribe directly, with no sign flip.
* **World frame** is the same planar convention, not compass bearings.  A
  caller holding compass headings converts once at its own boundary; doing it
  inside here would put two conventions in one file, which is where the sign
  bugs live.

Segmentation is adjacent-beam range-gap (Dietmayer et al., the standard
LaserScan method): walk the returns in BEAM ORDER and start a new cluster
wherever successive points are further apart than ``gap_m``.  It is chosen
over a general clustering algorithm because a scan is already sorted by
angle -- that ordering is free information, it makes segmentation O(n) with
no parameters beyond the gap, and it needs no sklearn on a Jetson.

numpy only, so it imports on a bare aarch64 brain next to the rest of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "Obstacle",
    "cluster_centroids",
    "cluster_extents",
    "cluster_points",
    "scan_obstacles",
    "scan_to_body_points",
    "scan_to_world_points",
]

Point = tuple[float, float]

#: An empty result still has to be an (N, 2) array.  Returning a bare ``[]``
#: or a (0,) array makes every caller's ``pts[:, 0]`` raise on the one input
#: that is most common in the field -- a scan into open air.
_EMPTY = np.empty((0, 2), dtype=float)


def scan_to_body_points(
    ranges: Sequence[float] | np.ndarray,
    angle_min: float,
    angle_increment: float,
    range_min: float = 0.0,
    range_max: float = float("inf"),
) -> np.ndarray:
    """Polar returns -> Cartesian points in the SENSOR's own frame.

    Args:
        ranges: one distance per beam, in metres, in beam order.
        angle_min: bearing of beam 0, radians, CCW positive from +X forward.
        angle_increment: bearing step between successive beams, radians.
        range_min: returns at or below this are discarded as self-hits.
        range_max: returns at or above this are discarded as no-returns.

    Returns:
        ``(N, 2)`` float array of ``(x, y)``.  N is the number of VALID
        returns, so it is generally smaller than ``len(ranges)`` and may be
        zero.  Order is preserved, which is what makes the output safe to
        hand to :func:`cluster_points`.

    Beams that hit nothing are dropped, not clamped.  A driver signals "no
    return" as NaN, as +inf, or as exactly ``range_max`` depending on who
    wrote it, and all three mean the same thing: there is nothing out there.
    Keeping them paints a phantom arc of obstacles at the sensor's own range,
    which reads on the map as a wall that follows the robot around.
    """
    r = np.asarray(ranges, dtype=float).ravel()
    if r.size == 0:
        return _EMPTY.copy()

    angles = angle_min + np.arange(r.size, dtype=float) * float(angle_increment)

    # NaN comparisons are all False, so NaN fails `valid` without a special
    # case; -inf is caught by the range_min bound and +inf by range_max.
    with np.errstate(invalid="ignore"):
        valid = (r > float(range_min)) & (r < float(range_max))
    if not valid.any():
        return _EMPTY.copy()

    r = r[valid]
    angles = angles[valid]
    return np.column_stack((r * np.cos(angles), r * np.sin(angles)))


def scan_to_world_points(
    ranges: Sequence[float] | np.ndarray,
    angle_min: float,
    angle_increment: float,
    sensor_x: float = 0.0,
    sensor_y: float = 0.0,
    sensor_yaw_deg: float = 0.0,
    range_min: float = 0.0,
    range_max: float = float("inf"),
) -> np.ndarray:
    """Polar returns -> Cartesian points in the WORLD frame.

    Args:
        sensor_x: world X of the sensor origin, metres.
        sensor_y: world Y of the sensor origin, metres.
        sensor_yaw_deg: sensor heading, degrees, CCW positive from world +X.
            This is the BODY's yaw when the LiDAR is bolted straight ahead;
            a rotated mount adds its own yaw here before calling.

    Returns:
        ``(N, 2)`` float array of world ``(x, y)``.

    Rotation happens BEFORE translation.  Adding the sensor's position to the
    body-frame point and rotating the sum is a transform of a different scene:
    it agrees with this one only while the sensor sits on the world origin,
    which is exactly the pose every quick test uses.
    """
    pts = scan_to_body_points(
        ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
    )
    if pts.size == 0:
        return pts

    yaw = math.radians(float(sensor_yaw_deg))
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    # Standard planar rotation, CCW positive -- the same matrix the pure
    # pursuit follower's yaw rate integrates toward.
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=float)
    return pts @ rot.T + np.array([float(sensor_x), float(sensor_y)], dtype=float)


def cluster_points(points: np.ndarray, gap_m: float) -> list[np.ndarray]:
    """Split scan points into obstacles by adjacent-beam range gap.

    Args:
        points: ``(N, 2)`` points IN BEAM ORDER, as returned by
            :func:`scan_to_body_points` or :func:`scan_to_world_points`.
        gap_m: successive points further apart than this begin a new cluster.

    Returns:
        A list of ``(M, 2)`` arrays, in beam order, together containing every
        input point exactly once.  An empty input gives an empty list.

    Beam order is the whole method.  A scan is already sorted by angle, so two
    returns from one surface are necessarily neighbours in the array; sorting
    or spatially grouping the points first throws that away and merges
    everything the robot can see into a single blob.

    The gap survives beam dropping without extra bookkeeping: because the
    comparison is Euclidean distance between kept points rather than beam
    index, a run of no-returns across a doorway naturally reads as a gap.

    Note that a fixed ``gap_m`` is angular-resolution dependent -- points on a
    far surface are further apart than points on a near one, so a very
    distant object can over-segment.  That is a known property of the classic
    method, and the honest fix is a caller that scales ``gap_m`` with range,
    not a hidden heuristic here.
    """
    if gap_m <= 0.0:
        raise ValueError(f"gap_m must be positive, got {gap_m!r}")

    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return []
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be shape (N, 2), got {pts.shape}")
    if pts.shape[0] == 1:
        return [pts.copy()]

    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    # +1 because a break at step i starts a new cluster at point i+1.
    breaks = np.flatnonzero(steps > float(gap_m)) + 1
    return [c for c in np.split(pts, breaks) if c.size]


def cluster_centroids(clusters: Iterable[np.ndarray]) -> list[Point]:
    """The mean position of each cluster -- where to put the map marker."""
    return [
        (float(np.mean(c[:, 0])), float(np.mean(c[:, 1])))
        for c in clusters
        if np.asarray(c).size
    ]


def cluster_extents(clusters: Iterable[np.ndarray]) -> list[float]:
    """Each cluster's radius about its own centroid, in metres.

    This is the farthest point from the centroid, not half the bounding box
    and not a standard deviation: it is the smallest circle centred on the
    drawn marker that actually CONTAINS every return, so a marker drawn at
    this radius never sits inside geometry the sensor saw.  A single-point
    cluster has extent 0.
    """
    out: list[float] = []
    for cluster in clusters:
        c = np.asarray(cluster, dtype=float)
        if c.size == 0:
            continue
        centroid = c.mean(axis=0)
        out.append(float(np.linalg.norm(c - centroid, axis=1).max()))
    return out


@dataclass(frozen=True)
class Obstacle:
    """One clustered obstacle, ready to draw or to feed a costmap.

    Attributes:
        x: centroid X in the frame the points were in (world, for
            :func:`scan_obstacles`).
        y: centroid Y in that same frame.
        radius_m: extent about the centroid -- see :func:`cluster_extents`.
        point_count: how many returns support it.  Carried through because it
            is the only evidence a consumer has for how much to trust the
            obstacle: a 40-return wall and a 1-return speckle are otherwise
            indistinguishable once reduced to a centroid.
    """

    x: float
    y: float
    radius_m: float
    point_count: int

    @property
    def position(self) -> Point:
        """Centroid as an ``(x, y)`` tuple."""
        return (self.x, self.y)


def scan_obstacles(
    ranges: Sequence[float] | np.ndarray,
    angle_min: float,
    angle_increment: float,
    sensor_x: float = 0.0,
    sensor_y: float = 0.0,
    sensor_yaw_deg: float = 0.0,
    range_min: float = 0.0,
    range_max: float = float("inf"),
    gap_m: float = 0.5,
    min_points: int = 1,
) -> list[Obstacle]:
    """Raw LaserScan + body pose -> world-frame obstacles, in one call.

    The convenience path the map consumer actually wants: it composes
    :func:`scan_to_world_points`, :func:`cluster_points`, and the centroid /
    extent reductions so no caller has to remember the order.

    Args:
        min_points: clusters with fewer returns than this are discarded.  One
            stray return is usually a dust mote or a range spike, and drawing
            it puts a phantom obstacle in front of a planner that then refuses
            to move.  Default 1 keeps everything -- filtering is opt-in, so a
            caller cannot silently lose a real thin obstacle it never asked to
            have filtered.

    Returns:
        Obstacles in beam order.  An empty scan returns an empty list.
    """
    pts = scan_to_world_points(
        ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        sensor_x=sensor_x,
        sensor_y=sensor_y,
        sensor_yaw_deg=sensor_yaw_deg,
        range_min=range_min,
        range_max=range_max,
    )
    clusters = cluster_points(pts, gap_m=gap_m)
    if min_points > 1:
        clusters = [c for c in clusters if c.shape[0] >= min_points]

    centroids = cluster_centroids(clusters)
    extents = cluster_extents(clusters)
    return [
        Obstacle(
            x=cx,
            y=cy,
            radius_m=radius,
            point_count=int(cluster.shape[0]),
        )
        for cluster, (cx, cy), radius in zip(clusters, centroids, extents)
    ]
