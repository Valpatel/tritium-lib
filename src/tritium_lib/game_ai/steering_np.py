# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""NumPy-vectorized steering behaviors for 500+ agents at 10Hz.

Instead of calling seek()/flee()/arrive() per agent, this module operates on
arrays of ALL agent positions/velocities simultaneously.  A single call to
SteeringSystem.tick() updates every active agent in one vectorized pass.

Falls back gracefully: if numpy is not installed, import this module and you
get an ImportError at the top level so callers can catch it and use the pure
Python steering.py instead.

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Spatial hash for O(1) neighbor lookups
# ---------------------------------------------------------------------------

class SpatialHash:
    """Grid-based spatial hash for fast neighbor queries.

    Divides the world into cells of ``cell_size`` meters.  Inserting N agents
    is O(N).  Querying neighbors within a radius is O(k) where k is the
    number of agents in nearby cells, rather than O(N) brute force.
    """

    __slots__ = ("cell_size", "_inv_cell", "_grid")

    def __init__(self, cell_size: float = 10.0) -> None:
        self.cell_size = cell_size
        self._inv_cell = 1.0 / cell_size
        self._grid: dict[tuple[int, int], list[int]] = {}

    def clear(self) -> None:
        self._grid.clear()

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return int(x * self._inv_cell), int(y * self._inv_cell)

    def insert_all(self, positions: np.ndarray, active: np.ndarray) -> None:
        """Bulk-insert all active agents into the grid.

        Parameters
        ----------
        positions : ndarray, shape (N, 2)
        active : ndarray, shape (N,), dtype bool
        """
        self._grid.clear()
        grid = self._grid
        inv = self._inv_cell
        for i in range(len(active)):
            if not active[i]:
                continue
            cx = int(positions[i, 0] * inv)
            cy = int(positions[i, 1] * inv)
            key = (cx, cy)
            if key in grid:
                grid[key].append(i)
            else:
                grid[key] = [i]

    def query_radius(self, x: float, y: float, radius: float) -> list[int]:
        """Return indices of agents within *radius* of (x, y).

        Checks all cells that overlap the query circle.  Caller must do the
        actual distance check — this is a broad-phase filter.
        """
        r_cells = int(radius * self._inv_cell) + 1
        cx0 = int(x * self._inv_cell)
        cy0 = int(y * self._inv_cell)
        result: list[int] = []
        grid = self._grid
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                bucket = grid.get((cx0 + dx, cy0 + dy))
                if bucket:
                    result.extend(bucket)
        return result


# ---------------------------------------------------------------------------
# Vectorized helper functions
# ---------------------------------------------------------------------------

def _normalize_rows(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize an (N, 2) array of vectors.  Returns (unit_vectors, lengths)."""
    lengths = np.linalg.norm(v, axis=1, keepdims=True)
    safe = np.maximum(lengths, 1e-12)
    return v / safe, lengths.ravel()


def _truncate_rows(v: np.ndarray, max_len: np.ndarray) -> np.ndarray:
    """Clamp each row of (N,2) to the per-agent max_len (N,)."""
    lengths = np.linalg.norm(v, axis=1)
    over = lengths > max_len
    if over.any():
        scale = np.ones_like(lengths)
        scale[over] = max_len[over] / np.maximum(lengths[over], 1e-12)
        v = v * scale[:, np.newaxis]
    return v


# ---------------------------------------------------------------------------
# SteeringSystem — the core vectorized engine
# ---------------------------------------------------------------------------

class SteeringSystem:
    """Vectorized steering for N agents simultaneously.

    All state is stored as NumPy arrays of shape ``(max_agents, 2)`` for
    positions/velocities.  A single call to :meth:`tick` updates every active
    agent in one vectorized pass.
    """

    # Behavior bit flags
    SEEK = 1
    FLEE = 2
    ARRIVE = 4
    WANDER = 8
    SEPARATE = 16
    ALIGN = 32
    COHERE = 64
    AVOID = 128

    def __init__(self, max_agents: int = 1024) -> None:
        self.max_agents = max_agents
        self.count = 0

        # Core arrays — shape (max_agents, 2)
        self.positions = np.zeros((max_agents, 2), dtype=np.float32)
        self.velocities = np.zeros((max_agents, 2), dtype=np.float32)
        self.targets = np.zeros((max_agents, 2), dtype=np.float32)

        # Per-agent scalars
        self.max_speeds = np.full(max_agents, 1.4, dtype=np.float32)
        self.max_forces = np.full(max_agents, 2.0, dtype=np.float32)
        self.slow_radii = np.full(max_agents, 5.0, dtype=np.float32)
        self.separation_dist = np.full(max_agents, 3.0, dtype=np.float32)

        # Behavior config
        self.behavior_mask = np.zeros(max_agents, dtype=np.uint8)
        self.active = np.zeros(max_agents, dtype=bool)

        # Wander state (persistent angle per agent for smooth wandering)
        self._wander_angles = np.zeros(max_agents, dtype=np.float32)

        # Behavior weights
        self.w_seek = 1.0
        self.w_flee = 1.0
        self.w_arrive = 1.0
        self.w_wander = 0.5
        self.w_separate = 1.5
        self.w_align = 1.0
        self.w_cohere = 1.0

        # Spatial hash for neighbor queries
        self._spatial = SpatialHash(cell_size=10.0)

        # Free list for reuse of removed slots
        self._free: list[int] = []

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def add_agent(
        self,
        pos: tuple[float, float],
        vel: tuple[float, float] = (0.0, 0.0),
        max_speed: float = 1.4,
        max_force: float = 2.0,
        behavior: int = 1,  # SEEK
    ) -> int:
        """Add an agent and return its index."""
        if self._free:
            idx = self._free.pop()
        else:
            idx = self.count
            if idx >= self.max_agents:
                raise RuntimeError(f"SteeringSystem full ({self.max_agents} agents)")
            self.count = idx + 1

        self.positions[idx] = pos
        self.velocities[idx] = vel
        self.targets[idx] = pos  # default target = current position
        self.max_speeds[idx] = max_speed
        self.max_forces[idx] = max_force
        self.behavior_mask[idx] = behavior
        self.active[idx] = True
        self._wander_angles[idx] = np.random.uniform(0, 2 * np.pi)

        # Ensure count covers this index
        if idx >= self.count:
            self.count = idx + 1

        return idx

    def remove_agent(self, idx: int) -> None:
        """Deactivate agent at *idx* and add to free list."""
        self.active[idx] = False
        self.velocities[idx] = 0.0
        self._free.append(idx)

    # ------------------------------------------------------------------
    # Per-behavior vectorized computations
    # ------------------------------------------------------------------

    def _seek_vec(
        self, pos: np.ndarray, tgt: np.ndarray, vel: np.ndarray, speeds: np.ndarray,
    ) -> np.ndarray:
        """Vectorized seek: desired = normalize(target - pos) * max_speed."""
        desired = tgt - pos
        unit, _ = _normalize_rows(desired)
        return unit * speeds[:, np.newaxis] - vel

    def _flee_vec(
        self, pos: np.ndarray, tgt: np.ndarray, vel: np.ndarray, speeds: np.ndarray,
    ) -> np.ndarray:
        """Vectorized flee: steer away from target."""
        desired = pos - tgt
        unit, _ = _normalize_rows(desired)
        return unit * speeds[:, np.newaxis] - vel

    def _arrive_vec(
        self,
        pos: np.ndarray,
        tgt: np.ndarray,
        vel: np.ndarray,
        speeds: np.ndarray,
        slow_radii: np.ndarray,
    ) -> np.ndarray:
        """Vectorized arrive: seek with deceleration inside slow_radius."""
        to_target = tgt - pos
        unit, dist = _normalize_rows(to_target)
        # Ramp speed inside slow_radius
        ramped = np.where(dist >= slow_radii, speeds, speeds * (dist / np.maximum(slow_radii, 1e-12)))
        return unit * ramped[:, np.newaxis] - vel

    def _wander_vec(
        self, indices: np.ndarray, pos: np.ndarray, vel: np.ndarray, dt: float,
    ) -> np.ndarray:
        """Vectorized wander: smooth random meandering."""
        n = len(indices)
        # Jitter the persistent wander angle
        self._wander_angles[indices] += np.random.uniform(-0.5, 0.5, size=n).astype(np.float32)
        angles = self._wander_angles[indices]

        # Heading from current velocity (or default forward)
        _, v_len = _normalize_rows(vel)
        heading_angles = np.arctan2(vel[:, 1], vel[:, 0])

        # Project wander circle ahead
        wander_dist = 2.0
        wander_radius = 1.0
        circle_x = pos[:, 0] + np.cos(heading_angles) * wander_dist
        circle_y = pos[:, 1] + np.sin(heading_angles) * wander_dist
        target_x = circle_x + np.cos(angles) * wander_radius
        target_y = circle_y + np.sin(angles) * wander_radius

        wander_target = np.stack([target_x, target_y], axis=1)
        desired = wander_target - pos
        unit, _ = _normalize_rows(desired)
        speed = np.maximum(v_len, 0.5)
        return unit * speed[:, np.newaxis]

    def _separation_vec(self, n: int, pos: np.ndarray, sep_dist: np.ndarray) -> np.ndarray:
        """Vectorized separation using spatial hash."""
        force = np.zeros((n, 2), dtype=np.float32)
        self._spatial.clear()
        # Build a temporary active-subset mapping
        active_pos = pos
        # Create a full-index array mapping subset index -> subset index (identity for spatial hash)
        temp_active = np.ones(n, dtype=bool)
        self._spatial.insert_all(active_pos, temp_active)

        max_sep = float(sep_dist.max()) if n > 0 else 3.0

        for i in range(n):
            neighbors = self._spatial.query_radius(pos[i, 0], pos[i, 1], max_sep)
            if len(neighbors) <= 1:
                continue
            sep_r = sep_dist[i]
            fx, fy = 0.0, 0.0
            count = 0
            for j in neighbors:
                if j == i:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = (dx * dx + dy * dy) ** 0.5
                if 1e-6 < d < sep_r:
                    inv_d = 1.0 / d
                    fx += dx * inv_d * inv_d
                    fy += dy * inv_d * inv_d
                    count += 1
            if count > 0:
                inv_c = 1.0 / count
                force[i, 0] = fx * inv_c
                force[i, 1] = fy * inv_c
        return force

    def _alignment_vec(self, n: int, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Vectorized alignment using spatial hash (already built by separation)."""
        force = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            neighbors = self._spatial.query_radius(pos[i, 0], pos[i, 1], 10.0)
            if len(neighbors) <= 1:
                continue
            avg_vx, avg_vy = 0.0, 0.0
            count = 0
            for j in neighbors:
                if j == i:
                    continue
                avg_vx += vel[j, 0]
                avg_vy += vel[j, 1]
                count += 1
            if count > 0:
                force[i, 0] = avg_vx / count - vel[i, 0]
                force[i, 1] = avg_vy / count - vel[i, 1]
        return force

    def _cohesion_vec(self, n: int, pos: np.ndarray) -> np.ndarray:
        """Vectorized cohesion using spatial hash."""
        force = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            neighbors = self._spatial.query_radius(pos[i, 0], pos[i, 1], 10.0)
            if len(neighbors) <= 1:
                continue
            cx, cy = 0.0, 0.0
            count = 0
            for j in neighbors:
                if j == i:
                    continue
                cx += pos[j, 0]
                cy += pos[j, 1]
                count += 1
            if count > 0:
                force[i, 0] = cx / count - pos[i, 0]
                force[i, 1] = cy / count - pos[i, 1]
        return force

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        """Update ALL active agents in one vectorized pass."""
        n = self.count
        if n == 0:
            return

        active = self.active[:n]
        if not active.any():
            return

        pos = self.positions[:n]
        vel = self.velocities[:n]
        tgt = self.targets[:n]
        mask = self.behavior_mask[:n]
        speeds = self.max_speeds[:n]
        forces_max = self.max_forces[:n]
        slow_r = self.slow_radii[:n]
        sep_d = self.separation_dist[:n]

        forces = np.zeros((n, 2), dtype=np.float32)

        # --- Seek ---
        seek_mask = active & ((mask & self.SEEK) > 0)
        if seek_mask.any():
            idx = np.where(seek_mask)[0]
            f = self._seek_vec(pos[idx], tgt[idx], vel[idx], speeds[idx])
            forces[idx] += f * self.w_seek

        # --- Flee ---
        flee_mask = active & ((mask & self.FLEE) > 0)
        if flee_mask.any():
            idx = np.where(flee_mask)[0]
            f = self._flee_vec(pos[idx], tgt[idx], vel[idx], speeds[idx])
            forces[idx] += f * self.w_flee

        # --- Arrive ---
        arrive_mask = active & ((mask & self.ARRIVE) > 0)
        if arrive_mask.any():
            idx = np.where(arrive_mask)[0]
            f = self._arrive_vec(pos[idx], tgt[idx], vel[idx], speeds[idx], slow_r[idx])
            forces[idx] += f * self.w_arrive

        # --- Wander ---
        wander_mask = active & ((mask & self.WANDER) > 0)
        if wander_mask.any():
            idx = np.where(wander_mask)[0]
            f = self._wander_vec(idx, pos[idx], vel[idx], dt)
            forces[idx] += f * self.w_wander

        # --- Separation (needs spatial hash) ---
        sep_mask = active & ((mask & self.SEPARATE) > 0)
        if sep_mask.any():
            idx = np.where(sep_mask)[0]
            f = self._separation_vec(len(idx), pos[idx], sep_d[idx])
            forces[idx] += f * self.w_separate

        # --- Alignment ---
        align_mask = active & ((mask & self.ALIGN) > 0)
        if align_mask.any():
            idx = np.where(align_mask)[0]
            # Reuse spatial hash from separation if built, else build it
            if not sep_mask.any():
                self._spatial.clear()
                self._spatial.insert_all(pos[idx], np.ones(len(idx), dtype=bool))
            f = self._alignment_vec(len(idx), pos[idx], vel[idx])
            forces[idx] += f * self.w_align

        # --- Cohesion ---
        cohere_mask = active & ((mask & self.COHERE) > 0)
        if cohere_mask.any():
            idx = np.where(cohere_mask)[0]
            if not sep_mask.any() and not align_mask.any():
                self._spatial.clear()
                self._spatial.insert_all(pos[idx], np.ones(len(idx), dtype=bool))
            f = self._cohesion_vec(len(idx), pos[idx])
            forces[idx] += f * self.w_cohere

        # --- Truncate forces to max_force ---
        forces = _truncate_rows(forces, forces_max)

        # --- Integrate: vel += force * dt, clamp to max_speed, pos += vel * dt ---
        vel_new = vel + forces * dt
        vel_new = _truncate_rows(vel_new, speeds)

        # Only update active agents
        self.velocities[:n][active] = vel_new[active]
        self.positions[:n][active] = pos[active] + vel_new[active] * dt

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_positions(self) -> np.ndarray:
        """Return positions of active agents as (M, 2) array."""
        return self.positions[:self.count][self.active[:self.count]].copy()

    def get_headings(self) -> np.ndarray:
        """Return heading angles (radians) of active agents as (M,) array."""
        vel = self.velocities[:self.count][self.active[:self.count]]
        return np.arctan2(vel[:, 1], vel[:, 0])

    def get_all_positions(self) -> np.ndarray:
        """Return positions[:count] (includes inactive slots)."""
        return self.positions[:self.count]

    def get_all_active(self) -> np.ndarray:
        """Return active[:count] mask."""
        return self.active[:self.count]
