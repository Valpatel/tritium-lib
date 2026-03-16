# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""NumPy-vectorized ambient activity simulation for 500+ background entities.

Drop-in high-performance replacement for :mod:`ambient.AmbientSimulator`.
Uses :class:`steering_np.SteeringSystem` for all movement so the entire
population is updated in a single vectorized tick.

Falls back gracefully: import will raise ``ImportError`` if numpy is missing,
so callers can fall back to the pure Python ``ambient.py``.

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from __future__ import annotations

import math
import uuid

import numpy as np

from .steering_np import SteeringSystem


# ---------------------------------------------------------------------------
# Entity type constants (uint8 codes matching EntityType enum values)
# ---------------------------------------------------------------------------

PEDESTRIAN = 0
VEHICLE = 1
CYCLIST = 2
JOGGER = 3
DOG_WALKER = 4

_TYPE_NAMES = {
    PEDESTRIAN: "pedestrian",
    VEHICLE: "vehicle",
    CYCLIST: "cyclist",
    JOGGER: "jogger",
    DOG_WALKER: "dog_walker",
}

# State codes
STATE_MOVING = 0
STATE_STOPPED = 1
STATE_PARKED = 2
STATE_WAITING = 3

_STATE_NAMES = {
    STATE_MOVING: "moving",
    STATE_STOPPED: "stopped",
    STATE_PARKED: "parked",
    STATE_WAITING: "waiting",
}

# Speed ranges (m/s) per entity type
_SPEED_RANGES: dict[int, tuple[float, float]] = {
    PEDESTRIAN: (0.8, 1.6),
    VEHICLE: (5.0, 15.0),
    CYCLIST: (3.0, 6.0),
    JOGGER: (2.5, 3.5),
    DOG_WALKER: (0.5, 1.0),
}

# Stop probability per tick
_STOP_CHANCE: dict[int, float] = {
    PEDESTRIAN: 0.002,
    VEHICLE: 0.001,
    CYCLIST: 0.0005,
    JOGGER: 0.0003,
    DOG_WALKER: 0.008,
}

# Stop duration ranges (seconds)
_STOP_DURATION: dict[int, tuple[float, float]] = {
    PEDESTRIAN: (2.0, 8.0),
    VEHICLE: (2.0, 8.0),
    CYCLIST: (2.0, 8.0),
    JOGGER: (2.0, 8.0),
    DOG_WALKER: (3.0, 15.0),
}


# ---------------------------------------------------------------------------
# AmbientSimulatorNP
# ---------------------------------------------------------------------------

class AmbientSimulatorNP:
    """NumPy-vectorized ambient activity for 500+ background entities.

    Wraps a :class:`SteeringSystem` and adds per-entity metadata (type,
    state, stop timers, waypoints) while keeping all heavy math vectorized.

    Parameters
    ----------
    bounds : tuple[tuple[float, float], tuple[float, float]]
        ((min_x, min_y), (max_x, max_y)) in meters.
    max_entities : int
        Pre-allocated capacity.
    seed : int | None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        bounds: tuple[tuple[float, float], tuple[float, float]],
        max_entities: int = 1024,
        seed: int | None = None,
    ) -> None:
        self.bounds = bounds
        self.max_entities = max_entities
        self.steering = SteeringSystem(max_entities)

        # Per-entity metadata arrays
        self.entity_types = np.zeros(max_entities, dtype=np.uint8)
        self.states = np.zeros(max_entities, dtype=np.uint8)
        self.stop_timers = np.zeros(max_entities, dtype=np.float32)
        self.headings = np.zeros(max_entities, dtype=np.float32)
        self.base_speeds = np.zeros(max_entities, dtype=np.float32)

        # Waypoint system — each entity has a flat list of waypoints
        self._waypoints: dict[int, list[tuple[float, float]]] = {}
        self._wp_index: dict[int, int] = {}

        # Entity IDs for TargetTracker export
        self._entity_ids: dict[int, str] = {}

        self._rng = np.random.default_rng(seed)

        # Density profile (pedestrian/vehicle hour multipliers)
        self._ped_density = self._default_residential_ped()
        self._veh_density = self._default_residential_veh()
        self._target_pedestrians = 0
        self._target_vehicles = 0

    # ------------------------------------------------------------------
    # Density profiles
    # ------------------------------------------------------------------

    @staticmethod
    def _default_residential_ped() -> np.ndarray:
        return np.array([
            0.02, 0.01, 0.01, 0.01, 0.02, 0.05,
            0.15, 0.45, 0.55, 0.35, 0.25, 0.30,
            0.40, 0.35, 0.30, 0.40, 0.50, 0.60,
            0.55, 0.40, 0.25, 0.15, 0.08, 0.04,
        ], dtype=np.float32)

    @staticmethod
    def _default_residential_veh() -> np.ndarray:
        return np.array([
            0.03, 0.02, 0.02, 0.02, 0.05, 0.10,
            0.25, 0.60, 0.70, 0.40, 0.25, 0.30,
            0.35, 0.30, 0.25, 0.35, 0.55, 0.70,
            0.55, 0.35, 0.20, 0.12, 0.08, 0.05,
        ], dtype=np.float32)

    def _density_at(self, hour: float, density_table: np.ndarray) -> float:
        h0 = int(hour) % 24
        h1 = (h0 + 1) % 24
        frac = hour - int(hour)
        return float(density_table[h0] + (density_table[h1] - density_table[h0]) * frac)

    def set_density(self, pedestrians: int = 0, vehicles: int = 0) -> None:
        """Set base target counts before profile scaling."""
        self._target_pedestrians = pedestrians
        self._target_vehicles = vehicles

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    def _uid(self) -> str:
        return uuid.uuid4().hex[:8]

    def _random_pos(self) -> tuple[float, float]:
        lo, hi = self.bounds
        return (
            float(self._rng.uniform(lo[0], hi[0])),
            float(self._rng.uniform(lo[1], hi[1])),
        )

    def _random_edge_pos(self) -> tuple[float, float]:
        lo, hi = self.bounds
        edge = int(self._rng.integers(0, 4))
        if edge == 0:
            return (float(self._rng.uniform(lo[0], hi[0])), hi[1])
        elif edge == 1:
            return (float(self._rng.uniform(lo[0], hi[0])), lo[1])
        elif edge == 2:
            return (lo[0], float(self._rng.uniform(lo[1], hi[1])))
        else:
            return (hi[0], float(self._rng.uniform(lo[1], hi[1])))

    def _generate_path(self, start: tuple[float, float], end: tuple[float, float],
                       waypoints: int = 3) -> list[tuple[float, float]]:
        lo, hi = self.bounds
        path = [start]
        for i in range(1, waypoints + 1):
            frac = i / (waypoints + 1)
            mx = start[0] + (end[0] - start[0]) * frac + float(self._rng.normal(0, 15))
            my = start[1] + (end[1] - start[1]) * frac + float(self._rng.normal(0, 15))
            mx = max(lo[0], min(hi[0], mx))
            my = max(lo[1], min(hi[1], my))
            path.append((mx, my))
        path.append(end)
        return path

    def spawn(self, entity_type: int, pos: tuple[float, float] | None = None) -> int:
        """Spawn a single entity. Returns the agent index."""
        if pos is None:
            pos = self._random_edge_pos() if entity_type == VEHICLE else self._random_pos()

        speed_lo, speed_hi = _SPEED_RANGES.get(entity_type, (0.8, 1.6))
        speed = float(self._rng.uniform(speed_lo, speed_hi))

        behavior = SteeringSystem.SEEK
        idx = self.steering.add_agent(pos, max_speed=speed, behavior=behavior)

        self.entity_types[idx] = entity_type
        self.states[idx] = STATE_MOVING
        self.base_speeds[idx] = speed
        self._entity_ids[idx] = self._uid()

        # Generate a waypoint path
        dest = self._random_edge_pos() if entity_type == VEHICLE else self._random_pos()
        path = self._generate_path(pos, dest)
        self._waypoints[idx] = path
        self._wp_index[idx] = 0
        if path:
            self.steering.targets[idx] = path[0]

        return idx

    def spawn_batch(self, entity_type: int, count: int,
                    positions: np.ndarray | None = None) -> list[int]:
        """Spawn *count* entities of the given type.

        Parameters
        ----------
        entity_type : int
            One of PEDESTRIAN, VEHICLE, CYCLIST, JOGGER, DOG_WALKER.
        count : int
            Number to spawn.
        positions : ndarray, shape (count, 2), optional
            Explicit positions.  If None, random positions are generated.

        Returns
        -------
        list[int]
            Indices of the spawned agents.
        """
        indices: list[int] = []
        for i in range(count):
            p = (float(positions[i, 0]), float(positions[i, 1])) if positions is not None else None
            indices.append(self.spawn(entity_type, p))
        return indices

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, dt: float, current_hour: float = 12.0) -> None:
        """Advance the simulation by *dt* seconds at *current_hour*."""
        n = self.steering.count
        if n == 0:
            self._adjust_population(current_hour)
            return

        active = self.steering.active[:n]
        states = self.states[:n]
        timers = self.stop_timers[:n]
        types = self.entity_types[:n]

        # --- Handle stopped/waiting entities ---
        stopped = active & ((states == STATE_STOPPED) | (states == STATE_WAITING))
        if stopped.any():
            timers[stopped] -= dt
            resumed = stopped & (timers <= 0)
            if resumed.any():
                states[resumed] = STATE_MOVING
            # Zero velocity for still-stopped agents
            still_stopped = stopped & (timers > 0)
            if still_stopped.any():
                self.steering.velocities[:n][still_stopped] = 0.0

        # --- Random stops for moving entities ---
        moving = active & (states == STATE_MOVING)
        if moving.any():
            moving_idx = np.where(moving)[0]
            rolls = self._rng.random(len(moving_idx)).astype(np.float32)
            for k, i in enumerate(moving_idx):
                chance = _STOP_CHANCE.get(int(types[i]), 0.001)
                if rolls[k] < chance:
                    dur_lo, dur_hi = _STOP_DURATION.get(int(types[i]), (2.0, 8.0))
                    self.states[i] = STATE_STOPPED
                    self.stop_timers[i] = float(self._rng.uniform(dur_lo, dur_hi))
                    self.steering.velocities[i] = 0.0

        # --- Advance waypoints for moving entities ---
        moving = active & (states == STATE_MOVING)
        if moving.any():
            moving_idx = np.where(moving)[0]
            pos = self.steering.positions
            for i in moving_idx:
                wp_list = self._waypoints.get(int(i))
                if not wp_list:
                    continue
                wp_i = self._wp_index.get(int(i), 0)
                if wp_i >= len(wp_list):
                    continue

                target = wp_list[wp_i]
                dx = pos[i, 0] - target[0]
                dy = pos[i, 1] - target[1]
                dist = (dx * dx + dy * dy) ** 0.5

                threshold = self.base_speeds[i] * dt * 1.5
                if dist < max(threshold, 1.0):
                    wp_i += 1
                    self._wp_index[int(i)] = wp_i
                    if wp_i >= len(wp_list):
                        # Path complete
                        etype = int(types[i])
                        if etype == JOGGER:
                            self._wp_index[int(i)] = 0
                            self.steering.targets[i] = wp_list[0]
                        elif etype == VEHICLE:
                            self.states[i] = STATE_PARKED
                            self.steering.velocities[i] = 0.0
                        else:
                            self.states[i] = STATE_STOPPED
                            self.stop_timers[i] = float(self._rng.uniform(5.0, 30.0))
                            self.steering.velocities[i] = 0.0
                    else:
                        self.steering.targets[i] = wp_list[wp_i]

        # --- Vectorized steering tick ---
        self.steering.tick(dt)

        # --- Clamp to bounds ---
        lo, hi = self.bounds
        pos = self.steering.positions[:n]
        np.clip(pos[:, 0], lo[0], hi[0], out=pos[:, 0])
        np.clip(pos[:, 1], lo[1], hi[1], out=pos[:, 1])

        # --- Update headings from velocities ---
        vel = self.steering.velocities[:n]
        speed_sq = vel[:, 0] ** 2 + vel[:, 1] ** 2
        has_vel = speed_sq > 1e-6
        if has_vel.any():
            self.headings[:n][has_vel] = np.degrees(
                np.arctan2(vel[has_vel, 0], vel[has_vel, 1])
            ) % 360

        # --- Cull finished entities ---
        self._cull_finished()

        # --- Adjust population ---
        self._adjust_population(current_hour)

    # ------------------------------------------------------------------
    # Population management
    # ------------------------------------------------------------------

    def _adjust_population(self, current_hour: float) -> None:
        ped_mult = self._density_at(current_hour, self._ped_density)
        veh_mult = self._density_at(current_hour, self._veh_density)
        want_ped = max(0, int(self._target_pedestrians * ped_mult))
        want_veh = max(0, int(self._target_vehicles * veh_mult))

        n = self.steering.count
        active = self.steering.active[:n]
        types = self.entity_types[:n]
        cur_ped = int(np.sum(active & (types != VEHICLE)))
        cur_veh = int(np.sum(active & (types == VEHICLE)))

        while cur_ped < want_ped:
            roll = float(self._rng.random())
            if roll < 0.1:
                self.spawn(JOGGER)
            elif roll < 0.2:
                self.spawn(DOG_WALKER)
            elif roll < 0.25:
                self.spawn(CYCLIST)
            else:
                self.spawn(PEDESTRIAN)
            cur_ped += 1

        while cur_veh < want_veh:
            self.spawn(VEHICLE)
            cur_veh += 1

    def _cull_finished(self) -> None:
        n = self.steering.count
        active = self.steering.active[:n]
        states = self.states[:n]
        timers = self.stop_timers[:n]
        types = self.entity_types[:n]

        # Remove parked vehicles
        parked = active & (states == STATE_PARKED)
        # Remove stopped entities that have timed out and completed their path
        timed_out = active & (states == STATE_STOPPED) & (timers <= 0)
        to_remove = parked | timed_out

        if to_remove.any():
            for i in np.where(to_remove)[0]:
                ii = int(i)
                # For stopped: only remove if path is complete
                if states[i] == STATE_STOPPED:
                    wp_list = self._waypoints.get(ii, [])
                    wp_i = self._wp_index.get(ii, 0)
                    if wp_i < len(wp_list):
                        continue
                self.steering.remove_agent(ii)
                self._waypoints.pop(ii, None)
                self._wp_index.pop(ii, None)
                self._entity_ids.pop(ii, None)

    # ------------------------------------------------------------------
    # Export for TargetTracker
    # ------------------------------------------------------------------

    def to_target_dicts(self) -> list[dict]:
        """Export all active entities as dicts compatible with TargetTracker.

        Output format matches :meth:`AmbientEntity.to_dict` from the pure
        Python ``ambient.py`` module.
        """
        n = self.steering.count
        active = self.steering.active[:n]
        if not active.any():
            return []

        results: list[dict] = []
        pos = self.steering.positions[:n]
        types = self.entity_types[:n]
        states = self.states[:n]
        headings = self.headings[:n]
        speeds = self.base_speeds[:n]

        for i in np.where(active)[0]:
            ii = int(i)
            etype = int(types[i])
            etype_name = _TYPE_NAMES.get(etype, "pedestrian")
            state_name = _STATE_NAMES.get(int(states[i]), "moving")
            classification = "vehicle" if etype == VEHICLE else "person"
            eid = self._entity_ids.get(ii, f"np_{ii}")

            results.append({
                "target_id": f"amb_{eid}",
                "name": f"Ambient {etype_name}",
                "source": "ambient_sim",
                "asset_type": etype_name,
                "alliance": "neutral",
                "classification": classification,
                "position_x": float(pos[i, 0]),
                "position_y": float(pos[i, 1]),
                "heading": float(headings[i]),
                "speed": float(speeds[i]) if state_name == "moving" else 0.0,
                "state": state_name,
                "metadata": {
                    "entity_type": etype_name,
                    "simulated": True,
                },
            })
        return results

    @property
    def active_count(self) -> int:
        """Number of currently active entities."""
        return int(self.steering.active[:self.steering.count].sum())
