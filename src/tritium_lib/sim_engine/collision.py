# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""2D collision system for the city3d simulation demo.

Pure-Python top-down collision detection with support for circles and
axis-aligned bounding boxes (AABB).  Uses collision layers and configurable
response rules so the city sim can express "car hits pedestrian at speed
> 8 m/s = kill" without hard-coding physics details.

Design goals:
- Pure Python, no NumPy — lightweight, easy to embed in JS later.
- Vec2 tuples from steering.py — same coordinate system.
- Spatial hash broad-phase for O(N) performance.
- Layer-based collision filtering with per-pair response rules.
- Raycast and area query for LOS / projectile / proximity checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from tritium_lib.sim_engine.ai.steering import Vec2, distance, magnitude, normalize

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ColliderType(Enum):
    """Shape of a collider."""
    CIRCLE = "circle"
    AABB = "aabb"


class ResponseType(Enum):
    """How to resolve a collision between two layers."""
    PUSH = "push"
    DAMAGE = "damage"
    KILL = "kill"
    STOP = "stop"
    IGNORE = "ignore"


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class Collider:
    """A collidable entity in the world."""
    entity_id: str
    collider_type: ColliderType
    position: Vec2 = (0.0, 0.0)
    velocity: Vec2 = (0.0, 0.0)
    radius: float = 0.0
    half_width: float = 0.0
    half_depth: float = 0.0
    mass: float = 1.0
    is_static: bool = False
    layer: str = "default"
    on_collision: str | None = None  # "damage", "push", "kill", "stop"

    # Derived helpers
    @property
    def min_x(self) -> float:
        if self.collider_type == ColliderType.CIRCLE:
            return self.position[0] - self.radius
        return self.position[0] - self.half_width

    @property
    def max_x(self) -> float:
        if self.collider_type == ColliderType.CIRCLE:
            return self.position[0] + self.radius
        return self.position[0] + self.half_width

    @property
    def min_y(self) -> float:
        if self.collider_type == ColliderType.CIRCLE:
            return self.position[1] - self.radius
        return self.position[1] - self.half_depth

    @property
    def max_y(self) -> float:
        if self.collider_type == ColliderType.CIRCLE:
            return self.position[1] + self.radius
        return self.position[1] + self.half_depth

    @property
    def bounding_radius(self) -> float:
        """Radius of a bounding circle that encloses the entire shape."""
        if self.collider_type == ColliderType.CIRCLE:
            return self.radius
        return math.hypot(self.half_width, self.half_depth)

    @property
    def speed(self) -> float:
        return magnitude(self.velocity)


@dataclass
class CollisionResult:
    """Describes one collision detected this tick."""
    entity_a: str
    entity_b: str
    overlap: float
    normal: Vec2  # push direction (a -> b)
    impact_speed: float


# ---------------------------------------------------------------------------
# Spatial hash grid (broad-phase)
# ---------------------------------------------------------------------------


class SpatialHashGrid:
    """Fixed-cell spatial hash for fast broad-phase queries."""

    __slots__ = ("cell_size", "_inv", "_grid")

    def __init__(self, cell_size: float = 10.0) -> None:
        self.cell_size = cell_size
        self._inv = 1.0 / cell_size
        self._grid: dict[tuple[int, int], list[str]] = {}

    def clear(self) -> None:
        self._grid.clear()

    def _cell_key(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor(x * self._inv)), int(math.floor(y * self._inv)))

    def insert(self, collider: Collider) -> None:
        """Insert a collider into all overlapping cells."""
        inv = self._inv
        cx_min = int(math.floor(collider.min_x * inv))
        cx_max = int(math.floor(collider.max_x * inv))
        cy_min = int(math.floor(collider.min_y * inv))
        cy_max = int(math.floor(collider.max_y * inv))
        eid = collider.entity_id
        grid = self._grid
        for cx in range(cx_min, cx_max + 1):
            for cy in range(cy_min, cy_max + 1):
                key = (cx, cy)
                if key in grid:
                    grid[key].append(eid)
                else:
                    grid[key] = [eid]

    def query(self, cell: tuple[int, int]) -> list[str]:
        """Return entity IDs in the given cell."""
        return list(self._grid.get(cell, []))

    def query_radius(self, center: Vec2, radius: float) -> set[str]:
        """Return all entity IDs whose cells overlap the query circle."""
        inv = self._inv
        cx_min = int(math.floor((center[0] - radius) * inv))
        cx_max = int(math.floor((center[0] + radius) * inv))
        cy_min = int(math.floor((center[1] - radius) * inv))
        cy_max = int(math.floor((center[1] + radius) * inv))
        result: set[str] = set()
        grid = self._grid
        for cx in range(cx_min, cx_max + 1):
            for cy in range(cy_min, cy_max + 1):
                bucket = grid.get((cx, cy))
                if bucket:
                    result.update(bucket)
        return result

    def candidate_pairs(self) -> set[tuple[str, str]]:
        """Return unique (id_a, id_b) pairs sharing at least one cell (a < b)."""
        pairs: set[tuple[str, str]] = set()
        for bucket in self._grid.values():
            n = len(bucket)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = bucket[i], bucket[j]
                    if a < b:
                        pairs.add((a, b))
                    else:
                        pairs.add((b, a))
        return pairs


# ---------------------------------------------------------------------------
# Narrow-phase intersection tests
# ---------------------------------------------------------------------------


def _circle_circle(a: Collider, b: Collider) -> CollisionResult | None:
    """Test two circles for overlap."""
    dx = b.position[0] - a.position[0]
    dy = b.position[1] - a.position[1]
    dist = math.hypot(dx, dy)
    min_dist = a.radius + b.radius
    if dist >= min_dist:
        return None
    if dist < 1e-9:
        normal: Vec2 = (1.0, 0.0)
        dist = 1e-9
    else:
        normal = (dx / dist, dy / dist)
    overlap = min_dist - dist
    # Impact speed: relative velocity projected onto collision normal.
    rel_vx = b.velocity[0] - a.velocity[0]
    rel_vy = b.velocity[1] - a.velocity[1]
    impact_speed = abs(rel_vx * normal[0] + rel_vy * normal[1])
    return CollisionResult(
        entity_a=a.entity_id,
        entity_b=b.entity_id,
        overlap=overlap,
        normal=normal,
        impact_speed=impact_speed,
    )


def _circle_aabb(circle: Collider, box: Collider) -> CollisionResult | None:
    """Test circle vs AABB overlap.  Returns normal pointing from box to circle."""
    cx, cy = circle.position
    bx, by = box.position
    # Closest point on AABB to circle center.
    closest_x = max(bx - box.half_width, min(cx, bx + box.half_width))
    closest_y = max(by - box.half_depth, min(cy, by + box.half_depth))
    dx = cx - closest_x
    dy = cy - closest_y
    dist = math.hypot(dx, dy)
    if dist >= circle.radius:
        return None
    if dist < 1e-9:
        # Circle center is inside the box — push along shortest axis.
        # Find shortest penetration axis.
        pen_left = (cx - (bx - box.half_width))
        pen_right = ((bx + box.half_width) - cx)
        pen_bottom = (cy - (by - box.half_depth))
        pen_top = ((by + box.half_depth) - cy)
        min_pen = min(pen_left, pen_right, pen_bottom, pen_top)
        if min_pen == pen_left:
            normal: Vec2 = (-1.0, 0.0)
            overlap = pen_left + circle.radius
        elif min_pen == pen_right:
            normal = (1.0, 0.0)
            overlap = pen_right + circle.radius
        elif min_pen == pen_bottom:
            normal = (0.0, -1.0)
            overlap = pen_bottom + circle.radius
        else:
            normal = (0.0, 1.0)
            overlap = pen_top + circle.radius
    else:
        normal = (dx / dist, dy / dist)
        overlap = circle.radius - dist
    # Impact speed.
    rel_vx = circle.velocity[0] - box.velocity[0]
    rel_vy = circle.velocity[1] - box.velocity[1]
    impact_speed = abs(rel_vx * normal[0] + rel_vy * normal[1])
    # entity_a = box, entity_b = circle, normal points box -> circle.
    return CollisionResult(
        entity_a=box.entity_id,
        entity_b=circle.entity_id,
        overlap=overlap,
        normal=normal,
        impact_speed=impact_speed,
    )


def _aabb_aabb(a: Collider, b: Collider) -> CollisionResult | None:
    """Test two AABBs for overlap using separating-axis."""
    dx = b.position[0] - a.position[0]
    dy = b.position[1] - a.position[1]
    overlap_x = (a.half_width + b.half_width) - abs(dx)
    overlap_y = (a.half_depth + b.half_depth) - abs(dy)
    if overlap_x <= 0 or overlap_y <= 0:
        return None
    if overlap_x < overlap_y:
        normal: Vec2 = (1.0 if dx >= 0 else -1.0, 0.0)
        overlap = overlap_x
    else:
        normal = (0.0, 1.0 if dy >= 0 else -1.0)
        overlap = overlap_y
    rel_vx = b.velocity[0] - a.velocity[0]
    rel_vy = b.velocity[1] - a.velocity[1]
    impact_speed = abs(rel_vx * normal[0] + rel_vy * normal[1])
    return CollisionResult(
        entity_a=a.entity_id,
        entity_b=b.entity_id,
        overlap=overlap,
        normal=normal,
        impact_speed=impact_speed,
    )


def _narrow_phase(a: Collider, b: Collider) -> CollisionResult | None:
    """Dispatch to the correct narrow-phase test."""
    ta, tb = a.collider_type, b.collider_type
    if ta == ColliderType.CIRCLE and tb == ColliderType.CIRCLE:
        return _circle_circle(a, b)
    if ta == ColliderType.AABB and tb == ColliderType.AABB:
        return _aabb_aabb(a, b)
    # Circle vs AABB (order matters for normal direction).
    if ta == ColliderType.CIRCLE and tb == ColliderType.AABB:
        return _circle_aabb(a, b)
    if ta == ColliderType.AABB and tb == ColliderType.CIRCLE:
        return _circle_aabb(b, a)
    return None


# ---------------------------------------------------------------------------
# Collision world
# ---------------------------------------------------------------------------

# Kill speed threshold for car-pedestrian collisions (m/s).
KILL_SPEED_THRESHOLD: float = 8.0


def _default_city_rules() -> dict[tuple[str, str], str]:
    """Default collision rules for the city simulation."""
    return {
        ("car", "car"): "push",
        ("car", "pedestrian"): "damage",  # damage at low speed, kill at high
        ("pedestrian", "car"): "damage",
        ("pedestrian", "pedestrian"): "push",
        ("pedestrian", "building"): "push",
        ("building", "pedestrian"): "push",
        ("car", "building"): "stop",
        ("building", "car"): "stop",
        ("projectile", "car"): "damage",
        ("projectile", "pedestrian"): "damage",
        ("projectile", "building"): "damage",
        ("car", "projectile"): "damage",
        ("pedestrian", "projectile"): "damage",
        ("building", "projectile"): "damage",
    }


class CollisionWorld:
    """Top-down 2D collision world with layers and response rules.

    Parameters
    ----------
    cell_size : float
        Spatial hash cell size.  Should be >= 2x the largest collider radius.
    kill_speed : float
        Speed threshold above which car-pedestrian collisions become kills.
    """

    def __init__(
        self,
        cell_size: float = 10.0,
        kill_speed: float = KILL_SPEED_THRESHOLD,
    ) -> None:
        self.colliders: dict[str, Collider] = {}
        self.static_colliders: list[Collider] = []
        self.collision_rules: dict[tuple[str, str], str] = _default_city_rules()
        self.kill_speed = kill_speed
        self._grid = SpatialHashGrid(cell_size)

    # -- Collider management -------------------------------------------------

    def add(self, collider: Collider) -> None:
        """Register a collider.  Static colliders also go in the fast list."""
        self.colliders[collider.entity_id] = collider
        if collider.is_static:
            self.static_colliders.append(collider)

    def remove(self, entity_id: str) -> None:
        """Remove a collider by entity ID."""
        col = self.colliders.pop(entity_id, None)
        if col and col.is_static:
            self.static_colliders = [
                c for c in self.static_colliders if c.entity_id != entity_id
            ]

    def update(
        self,
        entity_id: str,
        position: Vec2 | None = None,
        velocity: Vec2 | None = None,
    ) -> None:
        """Update a collider's position and/or velocity."""
        col = self.colliders.get(entity_id)
        if col is None:
            return
        if position is not None:
            col.position = position
        if velocity is not None:
            col.velocity = velocity

    # -- Rules ---------------------------------------------------------------

    def set_rule(self, layer_a: str, layer_b: str, response: str) -> None:
        """Set the collision response between two layers (symmetric)."""
        self.collision_rules[(layer_a, layer_b)] = response
        self.collision_rules[(layer_b, layer_a)] = response

    def get_rule(self, layer_a: str, layer_b: str) -> str:
        """Look up the response for a layer pair.  Default is 'push'."""
        return self.collision_rules.get((layer_a, layer_b), "push")

    # -- Detection -----------------------------------------------------------

    def check_all(self) -> list[CollisionResult]:
        """Run broad + narrow phase and return all collisions this tick.

        Broad phase uses a spatial hash grid rebuilt each call.
        Narrow phase dispatches to circle-circle, circle-AABB, or AABB-AABB.
        """
        # Rebuild spatial hash.
        self._grid.clear()
        for col in self.colliders.values():
            self._grid.insert(col)

        # Broad phase: candidate pairs.
        candidates = self._grid.candidate_pairs()

        # Narrow phase.
        results: list[CollisionResult] = []
        for id_a, id_b in candidates:
            a = self.colliders.get(id_a)
            b = self.colliders.get(id_b)
            if a is None or b is None:
                continue
            # Both static? Skip.
            if a.is_static and b.is_static:
                continue
            # Check layer rule — skip if "ignore".
            rule = self.get_rule(a.layer, b.layer)
            if rule == "ignore":
                continue
            result = _narrow_phase(a, b)
            if result is not None:
                results.append(result)
        return results

    # -- Resolution ----------------------------------------------------------

    def resolve(self, results: list[CollisionResult]) -> list[CollisionResult]:
        """Apply collision responses and return the results (for game logic).

        Modifies collider positions and velocities in place according to the
        layer rules.  Returns the same list for chaining / event dispatch.
        """
        for r in results:
            a = self.colliders.get(r.entity_a)
            b = self.colliders.get(r.entity_b)
            if a is None or b is None:
                continue
            rule = self.get_rule(a.layer, b.layer)
            if rule == "push":
                self._resolve_push(a, b, r)
            elif rule == "stop":
                self._resolve_stop(a, b, r)
            elif rule == "damage":
                # At high speed between car and pedestrian, escalate to kill.
                if self._is_car_pedestrian(a, b) and r.impact_speed > self.kill_speed:
                    self._resolve_kill(a, b, r)
                else:
                    self._resolve_push(a, b, r)
            elif rule == "kill":
                self._resolve_kill(a, b, r)
            # "ignore" — do nothing.
        return results

    def _resolve_push(self, a: Collider, b: Collider, r: CollisionResult) -> None:
        """Separate entities by overlap along normal, distribute by mass."""
        nx, ny = r.normal
        if a.is_static and not b.is_static:
            b.position = (
                b.position[0] + nx * r.overlap,
                b.position[1] + ny * r.overlap,
            )
            # Reflect velocity.
            dot = b.velocity[0] * nx + b.velocity[1] * ny
            if dot < 0:
                b.velocity = (
                    b.velocity[0] - 2.0 * dot * nx * 0.5,
                    b.velocity[1] - 2.0 * dot * ny * 0.5,
                )
        elif b.is_static and not a.is_static:
            a.position = (
                a.position[0] - nx * r.overlap,
                a.position[1] - ny * r.overlap,
            )
            dot = a.velocity[0] * (-nx) + a.velocity[1] * (-ny)
            if dot < 0:
                a.velocity = (
                    a.velocity[0] - 2.0 * dot * (-nx) * 0.5,
                    a.velocity[1] - 2.0 * dot * (-ny) * 0.5,
                )
        else:
            total_mass = a.mass + b.mass
            if total_mass < 1e-12:
                return
            ratio_a = b.mass / total_mass
            ratio_b = a.mass / total_mass
            a.position = (
                a.position[0] - nx * r.overlap * ratio_a,
                a.position[1] - ny * r.overlap * ratio_a,
            )
            b.position = (
                b.position[0] + nx * r.overlap * ratio_b,
                b.position[1] + ny * r.overlap * ratio_b,
            )
            # Simple elastic-ish velocity exchange along normal.
            rel_vn = (
                (b.velocity[0] - a.velocity[0]) * nx
                + (b.velocity[1] - a.velocity[1]) * ny
            )
            if rel_vn >= 0:
                return  # Already separating.
            impulse = rel_vn / total_mass
            a.velocity = (
                a.velocity[0] + impulse * b.mass * nx,
                a.velocity[1] + impulse * b.mass * ny,
            )
            b.velocity = (
                b.velocity[0] - impulse * a.mass * nx,
                b.velocity[1] - impulse * a.mass * ny,
            )

    def _resolve_stop(self, a: Collider, b: Collider, r: CollisionResult) -> None:
        """Halt the moving entity and push it out of overlap."""
        nx, ny = r.normal
        if not a.is_static:
            a.velocity = (0.0, 0.0)
            a.position = (
                a.position[0] - nx * r.overlap,
                a.position[1] - ny * r.overlap,
            )
        if not b.is_static:
            b.velocity = (0.0, 0.0)
            b.position = (
                b.position[0] + nx * r.overlap,
                b.position[1] + ny * r.overlap,
            )

    def _resolve_kill(self, a: Collider, b: Collider, r: CollisionResult) -> None:
        """Mark the pedestrian for removal (velocity zeroed, pushed out)."""
        ped = b if b.layer == "pedestrian" else a if a.layer == "pedestrian" else b
        ped.velocity = (0.0, 0.0)
        # Push the pedestrian out along normal.
        if ped is b:
            ped.position = (
                ped.position[0] + r.normal[0] * r.overlap,
                ped.position[1] + r.normal[1] * r.overlap,
            )
        else:
            ped.position = (
                ped.position[0] - r.normal[0] * r.overlap,
                ped.position[1] - r.normal[1] * r.overlap,
            )

    @staticmethod
    def _is_car_pedestrian(a: Collider, b: Collider) -> bool:
        layers = {a.layer, b.layer}
        return "car" in layers and "pedestrian" in layers

    # -- Spatial queries -----------------------------------------------------

    def raycast(
        self,
        origin: Vec2,
        direction: Vec2,
        max_dist: float,
    ) -> CollisionResult | None:
        """Cast a ray and return the first hit, or None.

        Uses a simple step-march through the spatial grid for efficiency,
        then does exact intersection against candidates.
        """
        d = normalize(direction)
        if d == (0.0, 0.0):
            return None

        # Gather candidates from cells along the ray.
        step = self._grid.cell_size * 0.5
        candidates: set[str] = set()
        t = 0.0
        while t <= max_dist:
            px = origin[0] + d[0] * t
            py = origin[1] + d[1] * t
            candidates.update(self._grid.query_radius((px, py), step))
            t += step

        # Test each candidate for ray intersection.
        best: CollisionResult | None = None
        best_t = max_dist + 1.0

        for eid in candidates:
            col = self.colliders.get(eid)
            if col is None:
                continue
            hit_t = self._ray_intersect(origin, d, max_dist, col)
            if hit_t is not None and hit_t < best_t:
                best_t = hit_t
                hit_point = (origin[0] + d[0] * hit_t, origin[1] + d[1] * hit_t)
                # Normal from collider center to hit point.
                hx = hit_point[0] - col.position[0]
                hy = hit_point[1] - col.position[1]
                hm = math.hypot(hx, hy)
                if hm < 1e-9:
                    normal = d
                else:
                    normal = (hx / hm, hy / hm)
                best = CollisionResult(
                    entity_a="__ray__",
                    entity_b=eid,
                    overlap=0.0,
                    normal=normal,
                    impact_speed=0.0,
                )
        return best

    @staticmethod
    def _ray_intersect(
        origin: Vec2, d: Vec2, max_dist: float, col: Collider,
    ) -> float | None:
        """Return parametric t of ray-collider intersection, or None."""
        if col.collider_type == ColliderType.CIRCLE:
            # Ray-circle intersection.
            ox = origin[0] - col.position[0]
            oy = origin[1] - col.position[1]
            a = d[0] * d[0] + d[1] * d[1]
            b = 2.0 * (ox * d[0] + oy * d[1])
            c = ox * ox + oy * oy - col.radius * col.radius
            disc = b * b - 4.0 * a * c
            if disc < 0:
                return None
            sqrt_disc = math.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            t = t1 if t1 >= 0 else t2
            if t < 0 or t > max_dist:
                return None
            return t
        else:
            # Ray-AABB (slab method).
            bmin_x = col.position[0] - col.half_width
            bmax_x = col.position[0] + col.half_width
            bmin_y = col.position[1] - col.half_depth
            bmax_y = col.position[1] + col.half_depth
            tmin = 0.0
            tmax = max_dist
            for axis in range(2):
                o = origin[axis]
                di = d[axis]
                bmin = bmin_x if axis == 0 else bmin_y
                bmax = bmax_x if axis == 0 else bmax_y
                if abs(di) < 1e-12:
                    if o < bmin or o > bmax:
                        return None
                else:
                    t1 = (bmin - o) / di
                    t2 = (bmax - o) / di
                    if t1 > t2:
                        t1, t2 = t2, t1
                    tmin = max(tmin, t1)
                    tmax = min(tmax, t2)
                    if tmin > tmax:
                        return None
            return tmin if tmin <= max_dist else None

    def query_area(self, center: Vec2, radius: float) -> list[Collider]:
        """Return all colliders within *radius* of *center*.

        Uses the spatial hash for a fast candidate set, then does an exact
        distance check using each collider's bounding radius.
        """
        candidate_ids = self._grid.query_radius(center, radius)
        result: list[Collider] = []
        for eid in candidate_ids:
            col = self.colliders.get(eid)
            if col is None:
                continue
            dist = distance(center, col.position)
            if dist <= radius + col.bounding_radius:
                result.append(col)
        return result


# ---------------------------------------------------------------------------
# Convenience: pre-configured city collision world
# ---------------------------------------------------------------------------


def create_city_world(cell_size: float = 10.0) -> CollisionWorld:
    """Create a CollisionWorld with default city-sim rules pre-loaded."""
    return CollisionWorld(cell_size=cell_size)
