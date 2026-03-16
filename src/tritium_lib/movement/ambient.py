# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Ambient activity simulation for realistic background map entities.

Generates pedestrians, vehicles, cyclists, joggers, and dog walkers that
follow realistic daily patterns.  Entities move along paths, respond to
time-of-day density profiles, and export dicts compatible with
TargetTracker ingestion.

Usage::

    from tritium_lib.movement.ambient import AmbientSimulator, ActivityProfile

    sim = AmbientSimulator(
        bounds=((0.0, 0.0), (500.0, 500.0)),
        profile=ActivityProfile.residential(),
    )
    sim.set_density(pedestrians=20, vehicles=8)
    sim.tick(dt=1.0, current_hour=14.5)
    targets = sim.get_entities()  # list[dict] for TargetTracker
"""

from __future__ import annotations

import math
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Vec2 — lightweight 2D vector
# ---------------------------------------------------------------------------

@dataclass
class Vec2:
    """Simple 2D vector for local-meter coordinates."""

    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> Vec2:
        ln = self.length()
        if ln < 1e-9:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / ln, self.y / ln)

    def distance_to(self, other: Vec2) -> float:
        return (self - other).length()

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


# ---------------------------------------------------------------------------
# Entity states and types
# ---------------------------------------------------------------------------

class EntityState(str, Enum):
    MOVING = "moving"
    STOPPED = "stopped"
    PARKED = "parked"
    WAITING = "waiting"


class EntityType(str, Enum):
    PEDESTRIAN = "pedestrian"
    VEHICLE = "vehicle"
    CYCLIST = "cyclist"
    JOGGER = "jogger"
    DOG_WALKER = "dog_walker"


# Per-type speed ranges in m/s
_SPEED_RANGES: dict[str, tuple[float, float]] = {
    EntityType.PEDESTRIAN: (0.8, 1.6),
    EntityType.VEHICLE: (5.0, 15.0),
    EntityType.CYCLIST: (3.0, 6.0),
    EntityType.JOGGER: (2.5, 3.5),
    EntityType.DOG_WALKER: (0.5, 1.0),
}

# Probability of a random stop per tick (lower = less frequent)
_STOP_CHANCE: dict[str, float] = {
    EntityType.PEDESTRIAN: 0.002,
    EntityType.VEHICLE: 0.001,
    EntityType.CYCLIST: 0.0005,
    EntityType.JOGGER: 0.0003,
    EntityType.DOG_WALKER: 0.008,
}


# ---------------------------------------------------------------------------
# ActivityProfile — time-of-day density curves
# ---------------------------------------------------------------------------

class ActivityProfile:
    """Defines when and where different entity types are active.

    Maps hour (0-23) to a density multiplier (0.0-1.0).  The simulator
    scales its target population by these multipliers each tick.
    """

    def __init__(self) -> None:
        self.pedestrian_density: dict[int, float] = {h: 0.1 for h in range(24)}
        self.vehicle_density: dict[int, float] = {h: 0.1 for h in range(24)}

    # -- factory class methods ------------------------------------------------

    @classmethod
    def residential(cls) -> ActivityProfile:
        """Typical residential neighbourhood pattern.

        Morning rush 7-9, evening rush 5-7, quiet overnight.
        """
        p = cls()
        ped = {
            0: 0.02, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.02, 5: 0.05,
            6: 0.15, 7: 0.45, 8: 0.55, 9: 0.35, 10: 0.25, 11: 0.30,
            12: 0.40, 13: 0.35, 14: 0.30, 15: 0.40, 16: 0.50, 17: 0.60,
            18: 0.55, 19: 0.40, 20: 0.25, 21: 0.15, 22: 0.08, 23: 0.04,
        }
        veh = {
            0: 0.03, 1: 0.02, 2: 0.02, 3: 0.02, 4: 0.05, 5: 0.10,
            6: 0.25, 7: 0.60, 8: 0.70, 9: 0.40, 10: 0.25, 11: 0.30,
            12: 0.35, 13: 0.30, 14: 0.25, 15: 0.35, 16: 0.55, 17: 0.70,
            18: 0.55, 19: 0.35, 20: 0.20, 21: 0.12, 22: 0.08, 23: 0.05,
        }
        p.pedestrian_density = ped
        p.vehicle_density = veh
        return p

    @classmethod
    def commercial(cls) -> ActivityProfile:
        """Business district pattern — busy 8-18, dead at night."""
        p = cls()
        ped = {
            0: 0.02, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.02, 5: 0.03,
            6: 0.08, 7: 0.25, 8: 0.65, 9: 0.80, 10: 0.85, 11: 0.90,
            12: 0.95, 13: 0.90, 14: 0.85, 15: 0.80, 16: 0.70, 17: 0.55,
            18: 0.30, 19: 0.15, 20: 0.08, 21: 0.05, 22: 0.03, 23: 0.02,
        }
        veh = {
            0: 0.02, 1: 0.01, 2: 0.01, 3: 0.02, 4: 0.03, 5: 0.05,
            6: 0.15, 7: 0.45, 8: 0.75, 9: 0.65, 10: 0.50, 11: 0.55,
            12: 0.60, 13: 0.55, 14: 0.50, 15: 0.55, 16: 0.65, 17: 0.80,
            18: 0.50, 19: 0.25, 20: 0.10, 21: 0.05, 22: 0.03, 23: 0.02,
        }
        p.pedestrian_density = ped
        p.vehicle_density = veh
        return p

    @classmethod
    def school(cls) -> ActivityProfile:
        """School zone pattern — peaks at 8am and 3pm."""
        p = cls()
        ped = {
            0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.02,
            6: 0.05, 7: 0.40, 8: 0.90, 9: 0.20, 10: 0.10, 11: 0.10,
            12: 0.15, 13: 0.10, 14: 0.30, 15: 0.90, 16: 0.50, 17: 0.20,
            18: 0.10, 19: 0.05, 20: 0.03, 21: 0.02, 22: 0.01, 23: 0.01,
        }
        veh = {
            0: 0.02, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.02, 5: 0.05,
            6: 0.10, 7: 0.50, 8: 0.85, 9: 0.20, 10: 0.10, 11: 0.10,
            12: 0.15, 13: 0.10, 14: 0.35, 15: 0.85, 16: 0.45, 17: 0.20,
            18: 0.10, 19: 0.05, 20: 0.03, 21: 0.02, 22: 0.01, 23: 0.01,
        }
        p.pedestrian_density = ped
        p.vehicle_density = veh
        return p

    def density_at(self, hour: float, entity_type: str) -> float:
        """Interpolated density for a fractional hour (e.g. 14.5 = 2:30pm).

        Returns a float in [0, 1].
        """
        table = (
            self.vehicle_density
            if entity_type == EntityType.VEHICLE
            else self.pedestrian_density
        )
        h0 = int(hour) % 24
        h1 = (h0 + 1) % 24
        frac = hour - int(hour)
        d0 = table.get(h0, 0.1)
        d1 = table.get(h1, 0.1)
        return d0 + (d1 - d0) * frac


# ---------------------------------------------------------------------------
# AmbientEntity — a single simulated background entity
# ---------------------------------------------------------------------------

@dataclass
class AmbientEntity:
    """A simulated background entity (pedestrian, vehicle, cyclist, etc.)."""

    entity_id: str = ""
    entity_type: str = EntityType.PEDESTRIAN
    position: Vec2 = field(default_factory=Vec2)
    velocity: Vec2 = field(default_factory=Vec2)
    heading: float = 0.0  # degrees, 0=north, clockwise
    speed: float = 1.2
    state: EntityState = EntityState.MOVING
    path: list[Vec2] = field(default_factory=list)
    path_index: int = 0
    _stop_timer: float = 0.0
    _wander_offset: Vec2 = field(default_factory=Vec2)

    # -- movement -------------------------------------------------------------

    def tick(self, dt: float, road_network: list[Vec2] | None = None,
             walkable_area: tuple[Vec2, Vec2] | None = None) -> None:
        """Advance the entity by *dt* seconds."""
        if self.state == EntityState.PARKED:
            return

        # Handle temporary stops
        if self.state in (EntityState.STOPPED, EntityState.WAITING):
            self._stop_timer -= dt
            if self._stop_timer <= 0:
                self.state = EntityState.MOVING
            else:
                self.velocity = Vec2(0.0, 0.0)
                return

        # Random stops (phone check, dog sniffing, traffic light)
        stop_chance = _STOP_CHANCE.get(self.entity_type, 0.001)
        if random.random() < stop_chance:
            self._stop_timer = random.uniform(2.0, 8.0)
            if self.entity_type == EntityType.DOG_WALKER:
                self._stop_timer = random.uniform(3.0, 15.0)
            self.state = EntityState.STOPPED
            self.velocity = Vec2(0.0, 0.0)
            return

        # Follow path if one exists
        if self.path and self.path_index < len(self.path):
            target = self.path[self.path_index]
            # Dog walker wander offset
            if self.entity_type == EntityType.DOG_WALKER:
                offset = Vec2(
                    random.gauss(0, 0.3),
                    random.gauss(0, 0.3),
                )
                self._wander_offset = Vec2(
                    self._wander_offset.x * 0.95 + offset.x * 0.05,
                    self._wander_offset.y * 0.95 + offset.y * 0.05,
                )
                target = target + self._wander_offset

            direction = target - self.position
            dist = direction.length()

            if dist < self.speed * dt * 1.5:
                # Arrived at waypoint
                self.path_index += 1
                # Joggers loop back to start
                if self.path_index >= len(self.path):
                    if self.entity_type == EntityType.JOGGER:
                        self.path_index = 0
                    else:
                        # At destination — park vehicles, stop pedestrians
                        if self.entity_type == EntityType.VEHICLE:
                            self.state = EntityState.PARKED
                        else:
                            self.state = EntityState.STOPPED
                            self._stop_timer = random.uniform(5.0, 30.0)
                        self.velocity = Vec2(0.0, 0.0)
                        return
            else:
                norm = direction.normalized()
                self.velocity = norm * self.speed
                self.heading = math.degrees(math.atan2(norm.x, norm.y)) % 360

        # Apply velocity
        self.position = self.position + self.velocity * dt

        # Clamp to walkable area
        if walkable_area:
            lo, hi = walkable_area
            self.position.x = max(lo.x, min(hi.x, self.position.x))
            self.position.y = max(lo.y, min(hi.y, self.position.y))

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict:
        """Export as dict compatible with TargetTracker ingestion."""
        etype = self.entity_type
        classification = "person"
        if etype == EntityType.VEHICLE:
            classification = "vehicle"
        elif etype == EntityType.CYCLIST:
            classification = "person"

        return {
            "target_id": f"amb_{self.entity_id}",
            "name": f"Ambient {etype}",
            "source": "ambient_sim",
            "asset_type": etype,
            "alliance": "neutral",
            "classification": classification,
            "position_x": self.position.x,
            "position_y": self.position.y,
            "heading": self.heading,
            "speed": self.speed if self.state == EntityState.MOVING else 0.0,
            "state": self.state.value if isinstance(self.state, EntityState) else self.state,
            "metadata": {
                "entity_type": etype,
                "simulated": True,
            },
        }


# ---------------------------------------------------------------------------
# AmbientSimulator — manages a population of background entities
# ---------------------------------------------------------------------------

class AmbientSimulator:
    """Generates and manages background activity entities.

    Parameters
    ----------
    bounds : tuple[Vec2, Vec2]
        (min_corner, max_corner) in local meters.
    profile : ActivityProfile | None
        Time-of-day density curves.  Defaults to residential.
    seed : int | None
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        bounds: tuple[Vec2, Vec2],
        profile: ActivityProfile | None = None,
        seed: int | None = None,
    ) -> None:
        self.bounds = bounds
        self.profile = profile or ActivityProfile.residential()
        self.entities: dict[str, AmbientEntity] = {}
        self._target_pedestrians: int = 0
        self._target_vehicles: int = 0
        self._rng = random.Random(seed)

    # -- public API -----------------------------------------------------------

    def tick(self, dt: float, current_hour: float) -> None:
        """Advance simulation by *dt* seconds at *current_hour* (0-24)."""
        # Adjust population based on density profile
        self._adjust_population(current_hour)

        # Tick each entity
        for entity in list(self.entities.values()):
            entity.tick(dt, walkable_area=self.bounds)

        # Cull parked/stopped entities that have finished
        self._cull_finished()

    def get_entities(self) -> list[dict]:
        """Export all entities as dicts for TargetTracker."""
        return [e.to_dict() for e in self.entities.values()]

    def set_density(self, pedestrians: int = 0, vehicles: int = 0) -> None:
        """Set target entity counts (before profile scaling)."""
        self._target_pedestrians = pedestrians
        self._target_vehicles = vehicles

    # -- spawning -------------------------------------------------------------

    def spawn_pedestrian(self, start: Vec2 | None = None) -> AmbientEntity:
        """Spawn a pedestrian with a random walking path."""
        pos = start or self._random_edge_position()
        dest = self._random_position()
        path = self._generate_walking_path(pos, dest)
        speed = self._rng.uniform(*_SPEED_RANGES[EntityType.PEDESTRIAN])
        ent = AmbientEntity(
            entity_id=self._uid(),
            entity_type=EntityType.PEDESTRIAN,
            position=Vec2(pos.x, pos.y),
            speed=speed,
            path=path,
            state=EntityState.MOVING,
        )
        self.entities[ent.entity_id] = ent
        return ent

    def spawn_vehicle(self, road_network: list[Vec2] | None = None) -> AmbientEntity:
        """Spawn a vehicle that follows roads (or a random path)."""
        pos = self._random_edge_position()
        if road_network and len(road_network) >= 2:
            path = list(road_network)
        else:
            dest = self._random_edge_position()
            path = self._generate_road_path(pos, dest)
        speed = self._rng.uniform(*_SPEED_RANGES[EntityType.VEHICLE])
        ent = AmbientEntity(
            entity_id=self._uid(),
            entity_type=EntityType.VEHICLE,
            position=Vec2(pos.x, pos.y),
            speed=speed,
            path=path,
            state=EntityState.MOVING,
        )
        self.entities[ent.entity_id] = ent
        return ent

    def spawn_jogger(self, route: list[Vec2] | None = None) -> AmbientEntity:
        """Spawn a jogger that loops a route at ~3 m/s."""
        pos = route[0] if route else self._random_position()
        if not route:
            route = self._generate_loop_path(pos, radius=80.0)
        speed = self._rng.uniform(*_SPEED_RANGES[EntityType.JOGGER])
        ent = AmbientEntity(
            entity_id=self._uid(),
            entity_type=EntityType.JOGGER,
            position=Vec2(pos.x, pos.y),
            speed=speed,
            path=route,
            state=EntityState.MOVING,
        )
        self.entities[ent.entity_id] = ent
        return ent

    def spawn_dog_walker(self, start: Vec2 | None = None) -> AmbientEntity:
        """Spawn a dog walker — slow, frequent stops, slight wander."""
        pos = start or self._random_position()
        dest = self._random_position()
        path = self._generate_walking_path(pos, dest, waypoints=6)
        speed = self._rng.uniform(*_SPEED_RANGES[EntityType.DOG_WALKER])
        ent = AmbientEntity(
            entity_id=self._uid(),
            entity_type=EntityType.DOG_WALKER,
            position=Vec2(pos.x, pos.y),
            speed=speed,
            path=path,
            state=EntityState.MOVING,
        )
        self.entities[ent.entity_id] = ent
        return ent

    def spawn_cyclist(self, start: Vec2 | None = None) -> AmbientEntity:
        """Spawn a cyclist following a path across the area."""
        pos = start or self._random_edge_position()
        dest = self._random_edge_position()
        path = self._generate_road_path(pos, dest)
        speed = self._rng.uniform(*_SPEED_RANGES[EntityType.CYCLIST])
        ent = AmbientEntity(
            entity_id=self._uid(),
            entity_type=EntityType.CYCLIST,
            position=Vec2(pos.x, pos.y),
            speed=speed,
            path=path,
            state=EntityState.MOVING,
        )
        self.entities[ent.entity_id] = ent
        return ent

    # -- internal helpers -----------------------------------------------------

    def _uid(self) -> str:
        return uuid.uuid4().hex[:8]

    def _random_position(self) -> Vec2:
        lo, hi = self.bounds
        return Vec2(
            self._rng.uniform(lo.x, hi.x),
            self._rng.uniform(lo.y, hi.y),
        )

    def _random_edge_position(self) -> Vec2:
        """Return a position on one of the four edges of the bounds."""
        lo, hi = self.bounds
        edge = self._rng.randint(0, 3)
        if edge == 0:  # top
            return Vec2(self._rng.uniform(lo.x, hi.x), hi.y)
        elif edge == 1:  # bottom
            return Vec2(self._rng.uniform(lo.x, hi.x), lo.y)
        elif edge == 2:  # left
            return Vec2(lo.x, self._rng.uniform(lo.y, hi.y))
        else:  # right
            return Vec2(hi.x, self._rng.uniform(lo.y, hi.y))

    def _generate_walking_path(
        self, start: Vec2, end: Vec2, waypoints: int = 3
    ) -> list[Vec2]:
        """Generate a walking path with random intermediate waypoints."""
        path = [Vec2(start.x, start.y)]
        for i in range(1, waypoints + 1):
            frac = i / (waypoints + 1)
            mid_x = start.x + (end.x - start.x) * frac + self._rng.gauss(0, 15)
            mid_y = start.y + (end.y - start.y) * frac + self._rng.gauss(0, 15)
            # Clamp to bounds
            lo, hi = self.bounds
            mid_x = max(lo.x, min(hi.x, mid_x))
            mid_y = max(lo.y, min(hi.y, mid_y))
            path.append(Vec2(mid_x, mid_y))
        path.append(Vec2(end.x, end.y))
        return path

    def _generate_road_path(self, start: Vec2, end: Vec2) -> list[Vec2]:
        """Generate an L-shaped or Z-shaped road path (axis-aligned turns)."""
        path = [Vec2(start.x, start.y)]
        # One or two axis-aligned turns
        if self._rng.random() < 0.5:
            # L-shape: go horizontal first, then vertical
            path.append(Vec2(end.x, start.y))
        else:
            # Z-shape: go vertical partway, horizontal, then vertical
            mid_y = start.y + (end.y - start.y) * self._rng.uniform(0.3, 0.7)
            path.append(Vec2(start.x, mid_y))
            path.append(Vec2(end.x, mid_y))
        path.append(Vec2(end.x, end.y))
        return path

    def _generate_loop_path(self, center: Vec2, radius: float = 80.0) -> list[Vec2]:
        """Generate a roughly circular loop path for joggers."""
        points: list[Vec2] = []
        n = 8
        for i in range(n):
            angle = 2 * math.pi * i / n
            r = radius * self._rng.uniform(0.8, 1.2)
            px = center.x + r * math.cos(angle)
            py = center.y + r * math.sin(angle)
            # Clamp to bounds
            lo, hi = self.bounds
            px = max(lo.x, min(hi.x, px))
            py = max(lo.y, min(hi.y, py))
            points.append(Vec2(px, py))
        return points

    def _adjust_population(self, current_hour: float) -> None:
        """Spawn or allow despawn to match density-scaled targets."""
        ped_density = self.profile.density_at(current_hour, EntityType.PEDESTRIAN)
        veh_density = self.profile.density_at(current_hour, EntityType.VEHICLE)

        want_ped = max(0, int(self._target_pedestrians * ped_density))
        want_veh = max(0, int(self._target_vehicles * veh_density))

        # Count current by type (non-vehicle pedestrian-like types count as pedestrians)
        cur_ped = 0
        cur_veh = 0
        for e in self.entities.values():
            if e.entity_type == EntityType.VEHICLE:
                cur_veh += 1
            else:
                cur_ped += 1

        # Spawn to fill
        while cur_ped < want_ped:
            roll = self._rng.random()
            if roll < 0.1:
                self.spawn_jogger()
            elif roll < 0.2:
                self.spawn_dog_walker()
            elif roll < 0.25:
                self.spawn_cyclist()
            else:
                self.spawn_pedestrian()
            cur_ped += 1

        while cur_veh < want_veh:
            self.spawn_vehicle()
            cur_veh += 1

    def _cull_finished(self) -> None:
        """Remove entities that have completed their path and stopped."""
        to_remove: list[str] = []
        for eid, ent in self.entities.items():
            if ent.state == EntityState.PARKED:
                to_remove.append(eid)
            elif ent.state == EntityState.STOPPED and ent._stop_timer <= 0:
                if ent.path_index >= len(ent.path):
                    to_remove.append(eid)
        for eid in to_remove:
            del self.entities[eid]
