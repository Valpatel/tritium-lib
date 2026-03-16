# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""2D collision detection and resolution with NumPy-vectorized broadphase.

Design goals:
- Struct-of-arrays layout matching steering_np for cache-friendly ticks.
- Spatial hash broadphase so tick() is O(N) not O(N^2).
- Circle-circle narrow phase (simple, fast, sufficient for top-down sim).
- Elastic collision with per-body restitution and mass-based momentum transfer.
- Static bodies (buildings, walls) participate in detection but don't move.
- Returns CollisionEvent list so game logic can react (damage, sounds, effects).

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RigidBody:
    """Simple 2D rigid body descriptor for creating bodies."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    mass: float = 1.0
    radius: float = 1.0
    width: float = 2.0
    length: float = 4.0
    heading: float = 0.0
    restitution: float = 0.3
    is_static: bool = False


@dataclass
class CollisionEvent:
    """Reported when two bodies collide in a tick."""

    body_a: int
    body_b: int
    point: np.ndarray
    normal: np.ndarray
    impulse: float
    relative_speed: float


# ---------------------------------------------------------------------------
# Spatial hash (broadphase)
# ---------------------------------------------------------------------------


class _SpatialHash:
    """Grid-based spatial hash for O(N) broadphase collision detection."""

    __slots__ = ("cell_size", "_inv", "_grid")

    def __init__(self, cell_size: float = 10.0) -> None:
        self.cell_size = cell_size
        self._inv = 1.0 / cell_size
        self._grid: dict[tuple[int, int], list[int]] = {}

    def rebuild(self, positions: np.ndarray, radii: np.ndarray, active: np.ndarray) -> None:
        """Bulk-insert all active bodies.  Uses radius to insert into overlapping cells."""
        self._grid.clear()
        grid = self._grid
        inv = self._inv
        for i in range(len(active)):
            if not active[i]:
                continue
            px, py = positions[i]
            r = radii[i]
            # Insert into every cell the body's bounding circle overlaps.
            cx_min = int((px - r) * inv)
            cx_max = int((px + r) * inv)
            cy_min = int((py - r) * inv)
            cy_max = int((py + r) * inv)
            for cx in range(cx_min, cx_max + 1):
                for cy in range(cy_min, cy_max + 1):
                    key = (cx, cy)
                    if key in grid:
                        grid[key].append(i)
                    else:
                        grid[key] = [i]

    def candidate_pairs(self) -> set[tuple[int, int]]:
        """Return unique (i, j) pairs where i < j sharing at least one cell."""
        pairs: set[tuple[int, int]] = set()
        for bucket in self._grid.values():
            n = len(bucket)
            for a_idx in range(n):
                for b_idx in range(a_idx + 1, n):
                    i, j = bucket[a_idx], bucket[b_idx]
                    if i < j:
                        pairs.add((i, j))
                    else:
                        pairs.add((j, i))
        return pairs


# ---------------------------------------------------------------------------
# Physics world
# ---------------------------------------------------------------------------


class PhysicsWorld:
    """Simple 2D physics simulation for N bodies.

    All state lives in flat NumPy arrays (struct-of-arrays) so broadphase and
    integration can be vectorized.  The narrow phase and resolution loop over
    candidate pairs from the spatial hash.

    Parameters
    ----------
    max_bodies : int
        Pre-allocated capacity.  Can grow via ``_grow()`` if exceeded.
    cell_size : float
        Spatial hash cell size in world units.  Should be >= 2x largest radius.
    """

    def __init__(self, max_bodies: int = 1024, cell_size: float = 10.0) -> None:
        self._cap = max_bodies
        self.positions = np.zeros((max_bodies, 2), dtype=np.float32)
        self.velocities = np.zeros((max_bodies, 2), dtype=np.float32)
        self.masses = np.ones(max_bodies, dtype=np.float32)
        self.radii = np.ones(max_bodies, dtype=np.float32)
        self.restitutions = np.full(max_bodies, 0.3, dtype=np.float32)
        self.static = np.zeros(max_bodies, dtype=bool)
        self.active = np.zeros(max_bodies, dtype=bool)
        self.count = 0
        self._hash = _SpatialHash(cell_size)
        # Accumulated forces for this tick (reset each tick).
        self._forces = np.zeros((max_bodies, 2), dtype=np.float32)

    # -- Capacity management -------------------------------------------------

    def _grow(self) -> None:
        """Double capacity."""
        old = self._cap
        new_cap = old * 2
        for attr in ("positions", "velocities", "_forces"):
            old_arr = getattr(self, attr)
            new_arr = np.zeros((new_cap, old_arr.shape[1]), dtype=old_arr.dtype)
            new_arr[:old] = old_arr
            setattr(self, attr, new_arr)
        for attr in ("masses", "radii", "restitutions"):
            old_arr = getattr(self, attr)
            new_arr = np.ones(new_cap, dtype=old_arr.dtype) if attr == "masses" else (
                np.full(new_cap, 0.3, dtype=old_arr.dtype) if attr == "restitutions"
                else np.ones(new_cap, dtype=old_arr.dtype)
            )
            new_arr[:old] = old_arr
            setattr(self, attr, new_arr)
        for attr in ("static", "active"):
            old_arr = getattr(self, attr)
            new_arr = np.zeros(new_cap, dtype=old_arr.dtype)
            new_arr[:old] = old_arr
            setattr(self, attr, new_arr)
        self._cap = new_cap

    # -- Body management -----------------------------------------------------

    def add_body(
        self,
        pos: tuple[float, float] | np.ndarray,
        vel: tuple[float, float] | np.ndarray = (0.0, 0.0),
        mass: float = 1.0,
        radius: float = 1.0,
        restitution: float = 0.3,
        static: bool = False,
    ) -> int:
        """Add a body and return its index."""
        if self.count >= self._cap:
            self._grow()
        idx = self.count
        self.positions[idx] = pos
        self.velocities[idx] = vel
        self.masses[idx] = mass
        self.radii[idx] = radius
        self.restitutions[idx] = restitution
        self.static[idx] = static
        self.active[idx] = True
        self.count += 1
        return idx

    def add_rigid_body(self, body: RigidBody) -> int:
        """Convenience — add from a RigidBody descriptor."""
        return self.add_body(
            pos=body.position,
            vel=body.velocity,
            mass=body.mass,
            radius=body.radius,
            restitution=body.restitution,
            static=body.is_static,
        )

    def remove_body(self, idx: int) -> None:
        """Deactivate a body (slot is reusable via compact or left inactive)."""
        self.active[idx] = False

    # -- Forces --------------------------------------------------------------

    def apply_force(self, body_id: int, force: np.ndarray | tuple[float, float]) -> None:
        """Accumulate a force on *body_id* for the current tick."""
        self._forces[body_id] += force

    def apply_explosion(
        self,
        center: tuple[float, float] | np.ndarray,
        radius: float,
        force: float,
    ) -> None:
        """Apply radial force to all active non-static bodies within radius."""
        center_arr = np.asarray(center, dtype=np.float32)
        mask = self.active[:self.count] & ~self.static[:self.count]
        if not np.any(mask):
            return
        indices = np.where(mask)[0]
        deltas = self.positions[indices] - center_arr
        dists = np.linalg.norm(deltas, axis=1)
        in_range = dists < radius
        if not np.any(in_range):
            return
        hit_indices = indices[in_range]
        hit_deltas = deltas[in_range]
        hit_dists = dists[in_range]
        # Avoid division by zero — clamp min distance.
        hit_dists = np.maximum(hit_dists, 0.01)
        # Force falls off linearly with distance.
        magnitudes = force * (1.0 - hit_dists / radius)
        normals = hit_deltas / hit_dists[:, np.newaxis]
        self._forces[hit_indices] += normals * magnitudes[:, np.newaxis]

    # -- Tick ----------------------------------------------------------------

    def tick(self, dt: float) -> List[CollisionEvent]:
        """Step physics forward by *dt* seconds.

        1. Apply accumulated forces -> velocity.
        2. Integrate positions.
        3. Broadphase (spatial hash).
        4. Narrow phase (circle-circle).
        5. Resolve collisions (impulse-based).
        6. Return collision events.
        """
        n = self.count
        if n == 0:
            return []

        # -- 1. Forces -> velocity (F = ma, a = F/m) --------------------------
        dynamic = self.active[:n] & ~self.static[:n]
        if np.any(dynamic):
            dyn_idx = np.where(dynamic)[0]
            accel = self._forces[dyn_idx] / self.masses[dyn_idx, np.newaxis]
            self.velocities[dyn_idx] += accel * dt

        # Reset accumulated forces.
        self._forces[:n] = 0.0

        # -- 2. Integrate positions --------------------------------------------
        if np.any(dynamic):
            self.positions[dyn_idx] += self.velocities[dyn_idx] * dt

        # -- 3. Broadphase -----------------------------------------------------
        self._hash.rebuild(self.positions[:n], self.radii[:n], self.active[:n])
        candidates = self._hash.candidate_pairs()

        # -- 4+5. Narrow phase + resolve ---------------------------------------
        events: list[CollisionEvent] = []
        for i, j in candidates:
            if not (self.active[i] and self.active[j]):
                continue
            # Both static? Skip.
            if self.static[i] and self.static[j]:
                continue

            delta = self.positions[j] - self.positions[i]
            dist = float(np.linalg.norm(delta))
            min_dist = self.radii[i] + self.radii[j]

            if dist >= min_dist:
                continue  # No collision.

            # Collision normal (i -> j).
            if dist < 1e-6:
                normal = np.array([1.0, 0.0], dtype=np.float32)
                dist = 1e-6
            else:
                normal = delta / dist

            # Relative velocity (j relative to i).
            rel_vel = self.velocities[j] - self.velocities[i]
            rel_speed_normal = float(np.dot(rel_vel, normal))

            # Already separating? Still report but don't resolve.
            if rel_speed_normal > 0:
                contact = self.positions[i] + normal * self.radii[i]
                events.append(CollisionEvent(
                    body_a=i, body_b=j,
                    point=contact.copy(),
                    normal=normal.copy(),
                    impulse=0.0,
                    relative_speed=abs(rel_speed_normal),
                ))
                continue

            # Combined restitution (average).
            e = 0.5 * (self.restitutions[i] + self.restitutions[j])

            # Inverse masses (static bodies have inv_mass = 0).
            inv_m_i = 0.0 if self.static[i] else 1.0 / self.masses[i]
            inv_m_j = 0.0 if self.static[j] else 1.0 / self.masses[j]
            inv_mass_sum = inv_m_i + inv_m_j

            if inv_mass_sum < 1e-12:
                continue  # Both effectively infinite mass.

            # Impulse magnitude (1D along normal).
            j_impulse = -(1.0 + e) * rel_speed_normal / inv_mass_sum

            # Apply impulse.
            impulse_vec = j_impulse * normal
            if not self.static[i]:
                self.velocities[i] -= inv_m_i * impulse_vec
            if not self.static[j]:
                self.velocities[j] += inv_m_j * impulse_vec

            # Positional correction (push apart to avoid sinking).
            overlap = min_dist - dist
            correction = normal * (overlap / inv_mass_sum * 0.8)
            if not self.static[i]:
                self.positions[i] -= inv_m_i * correction
            if not self.static[j]:
                self.positions[j] += inv_m_j * correction

            # Contact point.
            contact = self.positions[i] + normal * self.radii[i]

            events.append(CollisionEvent(
                body_a=i, body_b=j,
                point=contact.copy(),
                normal=normal.copy(),
                impulse=abs(j_impulse),
                relative_speed=abs(rel_speed_normal),
            ))

        return events
