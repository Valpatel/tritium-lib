# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Project 3-D scene obstacles onto a 2-D planning costmap.

The planner in :mod:`tritium_lib.planning.astar` is 2-D: it wants a
:class:`~tritium_lib.planning.costmap.Costmap` of lethal and free cells.  A
simulator stage — or a real robot's obstacle detector — hands you *boxes in
3-D*.  This module is the seam between the two.

The important part is the **body band**.  Naively stamping every box in a
scene as lethal blocks the world, because two of the most common boxes are
things a legged robot does not collide with:

- the **ground slab** it is standing on (a box from z=-1 to z=0), and
- **overhead** structure — gantries, ceilings, bridge decks.

So a box only becomes lethal if its vertical span overlaps the band the body
actually sweeps through, ``body_band = (z_min, z_max)``.  This is the same
reduction Nav2 performs when it flattens a voxel layer into a 2-D costmap,
and it is why a quadruped can plan *under* a table but not *through* a wall.

The module is deliberately engine-agnostic: :class:`SceneObstacle` is a plain
box in world meters.  Nothing here imports Isaac, USD, or ROS — the caller
reads bounding boxes from whatever it has (a USD ``BBoxCache``, a LiDAR
clusterer, a floor plan) and hands them over.  Pure stdlib.

Typical use::

    from tritium_lib.planning import plan_route
    from tritium_lib.planning.scene_costmap import (
        SceneObstacle, costmap_from_scene,
    )

    costmap = costmap_from_scene(
        obstacles,                      # read off the live stage
        resolution=0.5,
        body_band=(0.10, 0.55),         # a Go2 standing at ~0.30 m
        ignore_prims=["/World/Go2"],    # the robot is not its own obstacle
    )
    route = plan_route(costmap, start_xy, goal_xy, clearance_m=0.4)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .costmap import Costmap, CostmapBuilder, CostmapWeights

__all__ = [
    "SceneObstacle",
    "costmap_from_scene",
    "footprint_polygon",
    "obstacles_to_feature_collection",
    "scene_bounds",
]

#: Default vertical band swept by a standing quadruped body, in meters above
#: the ground plane.  Below ``0.10`` is floor clutter a leg steps over; above
#: ``0.55`` clears the back of a Go2-sized body.
DEFAULT_BODY_BAND: tuple[float, float] = (0.10, 0.55)


@dataclass
class SceneObstacle:
    """An oriented bounding box in world meters.

    Attributes:
        prim_path: Stage path or any stable identifier.  Used for
            :func:`costmap_from_scene`'s ``ignore_prims`` matching and for
            operator introspection; it carries no geometric meaning.
        center: World ``(x, y, z)`` of the box center.
        half_extents: Half sizes ``(hx, hy, hz)``.  Must be non-negative.
        yaw_deg: Rotation about the world Z axis, degrees, counter-clockwise.
            Roll and pitch are deliberately not modelled — a tilted box is
            handled by passing its *axis-aligned* world bounds, which is what
            a ``BBoxCache`` returns anyway and which errs toward conservative.
    """

    prim_path: str
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    yaw_deg: float = 0.0
    kind: str = "obstacle"

    def __post_init__(self) -> None:
        if any(h < 0 for h in self.half_extents):
            raise ValueError(
                f"half_extents must be non-negative, got {self.half_extents!r} "
                f"for {self.prim_path!r}"
            )

    @property
    def z_min(self) -> float:
        return self.center[2] - self.half_extents[2]

    @property
    def z_max(self) -> float:
        return self.center[2] + self.half_extents[2]

    def intersects_band(self, band_min: float, band_max: float) -> bool:
        """True iff this box's vertical span overlaps ``[band_min, band_max]``.

        Touching exactly at an endpoint does not count as overlap: a slab whose
        top is exactly the band floor is the ground, not a wall.
        """
        return self.z_max > band_min and self.z_min < band_max


def footprint_polygon(obstacle: SceneObstacle) -> list[tuple[float, float]]:
    """The box's ground footprint as a closed CCW ring of ``(x, y)`` corners.

    The returned ring repeats its first point as its last, which is what
    GeoJSON polygons require and what :class:`CostmapBuilder` expects.
    """
    cx, cy, _ = obstacle.center
    hx, hy, _ = obstacle.half_extents
    theta = math.radians(obstacle.yaw_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    corners = []
    for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
        corners.append((cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t))
    corners.append(corners[0])
    return corners


def _is_ignored(prim_path: str, ignore_prims: list[str]) -> bool:
    """True if ``prim_path`` is, or is a descendant of, any ignored path.

    Descendant matching is on path segments, so ``/World/Go2`` ignores
    ``/World/Go2/base`` but never ``/World/Go2Target``.
    """
    for ignored in ignore_prims:
        if prim_path == ignored or prim_path.startswith(ignored.rstrip("/") + "/"):
            return True
    return False


def scene_bounds(
    obstacles: list[SceneObstacle],
    padding_m: float = 10.0,
    include: list[tuple[float, float]] | None = None,
) -> tuple[float, float, float, float]:
    """Axis-aligned ``(min_x, min_y, max_x, max_y)`` covering the scene.

    Args:
        obstacles: Boxes to cover.  Footprints are used, so yaw is respected.
        padding_m: Slack added on every side — the planner needs room to
            detour *around* the outermost obstacle, so a zero-padding bound
            would wall the route in.
        include: Extra world points that must fall inside the bounds, e.g. the
            robot's start and its goal.

    Raises:
        ValueError: If there is nothing to bound (no obstacles, no ``include``).
    """
    xs: list[float] = []
    ys: list[float] = []
    for obs in obstacles:
        for x, y in footprint_polygon(obs):
            xs.append(x)
            ys.append(y)
    for x, y in include or []:
        xs.append(x)
        ys.append(y)

    if not xs:
        raise ValueError(
            "cannot derive scene bounds from an empty scene — pass obstacles, "
            "an explicit include=[(x, y), ...], or an explicit bounds=..."
        )
    return (
        min(xs) - padding_m,
        min(ys) - padding_m,
        max(xs) + padding_m,
        max(ys) + padding_m,
    )


def obstacles_to_feature_collection(
    obstacles: list[SceneObstacle],
    body_band: tuple[float, float] = DEFAULT_BODY_BAND,
    ignore_prims: list[str] | None = None,
) -> dict:
    """GeoJSON FeatureCollection of the footprints that block the body band.

    Exposed separately from :func:`costmap_from_scene` so a caller can hand
    the same footprints to the operator UI (draw the obstacles the planner
    actually saw) without rebuilding a costmap.
    """
    band_min, band_max = body_band
    if band_min >= band_max:
        raise ValueError(f"body_band must be (min, max) with min < max, got {body_band!r}")
    ignored = ignore_prims or []

    features = []
    for obs in obstacles:
        if _is_ignored(obs.prim_path, ignored):
            continue
        if not obs.intersects_band(band_min, band_max):
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[list(p) for p in footprint_polygon(obs)]],
                },
                "properties": {"prim_path": obs.prim_path, "kind": obs.kind},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def costmap_from_scene(
    obstacles: list[SceneObstacle],
    *,
    bounds: tuple[float, float, float, float] | None = None,
    resolution: float = 0.5,
    body_band: tuple[float, float] = DEFAULT_BODY_BAND,
    ignore_prims: list[str] | None = None,
    padding_m: float = 10.0,
    include: list[tuple[float, float]] | None = None,
    weights: CostmapWeights | None = None,
) -> Costmap:
    """Build a planning costmap from 3-D scene boxes.

    Boxes are stamped lethal iff they overlap ``body_band`` vertically and are
    not excluded by ``ignore_prims``.  Everything else is free space at base
    cost — this projection carries no terrain or road layers, which callers
    add through :class:`CostmapBuilder` directly if they have them.

    Args:
        obstacles: Scene boxes in world meters.
        bounds: Explicit ``(min_x, min_y, max_x, max_y)``.  Derived via
            :func:`scene_bounds` when omitted.
        resolution: Cell size in meters.  Pick something well under the
            narrowest gap the body must fit through — a 0.5 m grid cannot
            represent a 0.4 m doorway.
        body_band: Vertical span ``(z_min, z_max)`` the body sweeps.
        ignore_prims: Paths (and their descendants) to skip — at minimum the
            robot itself, which would otherwise plan around its own body.
        padding_m: Passed to :func:`scene_bounds` when deriving bounds.
        include: Extra points the derived bounds must cover (start/goal).
        weights: Optional cost weights forwarded to :class:`CostmapBuilder`.

    Returns:
        A :class:`Costmap` ready for :func:`~tritium_lib.planning.plan_route`.
    """
    if bounds is None:
        bounds = scene_bounds(obstacles, padding_m=padding_m, include=include)

    collection = obstacles_to_feature_collection(
        obstacles, body_band=body_band, ignore_prims=ignore_prims
    )
    builder = CostmapBuilder(bounds, resolution=resolution, weights=weights)
    builder.add_obstacles(collection, kind="scene")
    return builder.build()
