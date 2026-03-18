# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""AmbientSpawner — generates neutral neighborhood activity.

Spawns neighbors walking, cars driving, dogs roaming, delivery people,
and cats wandering to create a realistic "quiet phase" before any
threat escalation.  All targets use alliance='neutral' and auto-despawn
when they reach their destination.

Architecture
------------
AmbientSpawner is deliberately separate from the hostile spawner in
SimulationEngine.  The two serve different narrative purposes:

  - Hostile spawner (engine._random_hostile_spawner): creates *threats*
    at adaptive rates.  It is integral to the engine because hostile
    pressure drives the tactical loop — Amy must detect, classify, and
    dispatch against these.  The spawn rate adapts to current hostile
    count (back-pressure).

  - AmbientSpawner: creates *background noise* — neutral entities that
    test Amy's discrimination (is that person a threat or a neighbor?).
    It runs on its own thread with independent timing (15-45s intervals)
    and caps at MAX_NEUTRALS=8 to keep the map readable.

A unified SpawnerManager was considered and rejected: the two spawners
have no shared state, different timing models, different target profiles,
and different lifecycle rules.  Merging them would couple narrative pacing
(ambient) to tactical pressure (hostile) for no benefit.

Path generation is inline (_sidewalk_path, _road_path, _yard_wander,
_delivery_path) because each path type has unique topology.  A shared
pathfinding service would be warranted only if targets needed collision
avoidance or terrain-aware routing; for waypoint-following, inline
generators are simpler and more readable.
"""

from __future__ import annotations

import random
import threading
import time
import uuid
from datetime import datetime
from typing import Any

from tritium_lib.sim_engine.core.entity import SimulationTarget

# Default map bounds -- overridden by engine's actual bounds at runtime.
# These are only used if the spawner is somehow constructed without an engine.
_DEFAULT_MAP_BOUNDS = 200.0

# -- Neighborhood street grid ------------------------------------------------
# Street grid is generated dynamically based on map bounds.  Streets are
# placed every ~60m to form a realistic residential grid at any scale.
# The old hardcoded +-10 grid only covered a 60m map.  Now the grid scales
# with the simulation bounds (default 200m = 400m x 400m area).
_STREET_JITTER = 1.5            # Lateral jitter to avoid single-file lines
_STREET_SPACING = 60.0          # Meters between parallel streets


def _generate_street_grid(bounds: float) -> tuple[list[float], list[float]]:
    """Generate NS and EW street coordinates for a given map half-extent.

    Places streets every _STREET_SPACING meters from -bounds to +bounds.
    Always includes a street at 0.  Returns (ns_x_list, ew_y_list).
    """
    streets = []
    pos = 0.0
    while pos <= bounds:
        streets.append(pos)
        if pos > 0:
            streets.append(-pos)
        pos += _STREET_SPACING
    streets.sort()
    return streets, list(streets)  # NS and EW use same spacing


def _snap_to_nearest_street(
    x: float, y: float, streets_ns: list[float], streets_ew: list[float]
) -> tuple[float, float]:
    """Snap a point to the nearest street intersection or corridor."""
    best_ns = min(streets_ns, key=lambda sx: abs(sx - x))
    best_ew = min(streets_ew, key=lambda sy: abs(sy - y))
    if abs(best_ns - x) < abs(best_ew - y):
        return (best_ns + random.uniform(-_STREET_JITTER, _STREET_JITTER), y)
    else:
        return (x, best_ew + random.uniform(-_STREET_JITTER, _STREET_JITTER))


def _street_path(
    start: tuple[float, float],
    end: tuple[float, float],
    streets_ns: list[float],
    streets_ew: list[float],
) -> list[tuple[float, float]]:
    """Generate an L-shaped path following the street grid from start to end.

    Instead of cutting diagonally through yards, the path walks along the
    nearest north-south street, turns at an east-west street, then continues
    to the destination.  This produces the right-angle walking patterns you
    see in real neighborhoods.
    """
    sx, sy = start
    ex, ey = end

    # Pick the NS street closest to the start
    ns_x = min(streets_ns, key=lambda s: abs(s - sx))
    # Pick the EW street closest to the end
    ew_y = min(streets_ew, key=lambda s: abs(s - ey))

    jx = random.uniform(-_STREET_JITTER, _STREET_JITTER)
    jy = random.uniform(-_STREET_JITTER, _STREET_JITTER)

    # Path: start -> walk to NS street -> turn onto EW street -> end
    corner = (ns_x + jx, ew_y + jy)
    return [corner, end]


# -- Time-of-day activity scaling --------------------------------------------
# Real neighborhoods are quiet at 3am and busy at 5pm.  These multipliers
# scale spawn rates and type probabilities by hour of day.

def _hour_activity() -> tuple[float, float]:
    """Return (ambient_multiplier, hostile_multiplier) for the current hour.

    Ambient multiplier:  0.1 at 3am, 1.0 at 10am-6pm, 0.3 at midnight.
    Hostile multiplier:  0.3 during daytime, 1.0 at night (more intrusions
    after dark).
    """
    hour = datetime.now().hour
    if 6 <= hour < 22:
        # Daytime: full ambient, low hostile
        ambient = 1.0 if 8 <= hour < 20 else 0.6
        hostile = 0.3
    elif 22 <= hour or hour < 2:
        # Late evening: some ambient, rising hostile
        ambient = 0.3
        hostile = 0.8
    else:
        # Deep night (2am-6am): minimal ambient, peak hostile
        ambient = 0.1
        hostile = 1.0
    return ambient, hostile

# Predefined names for each target type
_NEIGHBOR_NAMES = [
    "Mrs. Henderson", "Mr. Kowalski", "Jenny", "Old Tom",
    "The Jogger", "Dog Walker", "Mail Carrier", "Teen on Bike",
    "Morning Walker", "Evening Stroller", "Couple", "Kid",
]

_CAR_NAMES = [
    "Red Sedan", "Blue SUV", "White Pickup", "Black Coupe",
    "Silver Minivan", "Green Hatchback", "Yellow Taxi", "Delivery Van",
]

_DOG_NAMES = [
    "Golden Retriever", "German Shepherd", "Beagle", "Husky",
    "Labrador", "Poodle", "Border Collie", "Corgi",
]

_CAT_NAMES = [
    "Orange Tabby", "Black Cat", "Calico", "Siamese",
    "Gray Cat", "White Cat", "Tuxedo Cat", "Maine Coon",
]

_DELIVERY_NAMES = [
    "FedEx Driver", "UPS Driver", "Amazon Delivery",
    "Pizza Delivery", "DoorDash", "Postman",
]


class AmbientSpawner:
    """Spawns neutral targets at random intervals to simulate neighborhood life.

    The engine parameter is duck-typed -- it must provide:
      - get_targets() -> list of targets with .alliance, .status
      - add_target(target) -> None
      - spawners_paused: bool
      - _map_bounds: float (optional, defaults to 200.0)
      - _obstacles (optional, for building collision)
    """

    MAX_NEUTRALS = 80
    SPAWN_MIN = 5.0  # seconds
    SPAWN_MAX = 15.0

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._running = False
        self._thread: threading.Thread | None = None
        self._used_names: set[str] = set()
        self._enabled = True
        # Derive map bounds from engine (defaults to _DEFAULT_MAP_BOUNDS)
        self._map_bounds = getattr(engine, '_map_bounds', _DEFAULT_MAP_BOUNDS)
        self._map_min = -self._map_bounds
        self._map_max = self._map_bounds
        # Generate street grid scaled to map bounds
        self._streets_ns, self._streets_ew = _generate_street_grid(self._map_bounds)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._spawn_loop, name="ambient-spawner", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _spawn_loop(self) -> None:
        while self._running:
            # Scale spawn interval by time of day — fewer spawns at 3am
            ambient_mult, _ = _hour_activity()
            base_delay = random.uniform(self.SPAWN_MIN, self.SPAWN_MAX)
            # Invert multiplier: low activity = longer delays
            delay = base_delay / max(ambient_mult, 0.1)

            elapsed = 0.0
            while elapsed < delay and self._running:
                time.sleep(0.5)
                elapsed += 0.5

            if not self._running or not self._enabled:
                continue

            # Respect engine-level spawner pause (LIVE mode)
            if self._engine.spawners_paused:
                continue

            # Count current neutrals
            neutral_count = sum(
                1 for t in self._engine.get_targets()
                if t.alliance == "neutral" and t.status not in ("despawned", "destroyed")
            )
            if neutral_count >= self.MAX_NEUTRALS:
                continue

            self._spawn_random()

    def _spawn_random(self) -> None:
        """Spawn a random neutral target type."""
        roll = random.random()
        if roll < 0.35:
            self._spawn_neighbor()
        elif roll < 0.55:
            self._spawn_car()
        elif roll < 0.70:
            self._spawn_dog()
        elif roll < 0.80:
            self._spawn_cat()
        else:
            self._spawn_delivery()

    def _pick_name(self, names: list[str]) -> str:
        """Pick an unused name, adding suffix if needed."""
        available = [n for n in names if n not in self._used_names]
        if not available:
            # All used -- pick random + suffix
            base = random.choice(names)
            suffix = 2
            name = f"{base} {suffix}"
            while name in self._used_names:
                suffix += 1
                name = f"{base} {suffix}"
        else:
            name = random.choice(available)
        self._used_names.add(name)
        return name

    def _random_edge(self) -> tuple[float, float]:
        """Random position on one of the four map edges."""
        edge = random.randint(0, 3)
        coord = random.uniform(self._map_min * 0.9, self._map_max * 0.9)
        if edge == 0:
            return (coord, self._map_max)
        elif edge == 1:
            return (coord, self._map_min)
        elif edge == 2:
            return (self._map_max, coord)
        else:
            return (self._map_min, coord)

    def _opposite_edge(self, pos: tuple[float, float]) -> tuple[float, float]:
        """Return a position on the opposite edge from the given position."""
        x, y = pos
        spread = self._map_bounds * 0.1  # 10% lateral spread
        if abs(y - self._map_max) < 2:
            return (self._clamp(x + random.uniform(-spread, spread)), self._map_min)
        elif abs(y - self._map_min) < 2:
            return (self._clamp(x + random.uniform(-spread, spread)), self._map_max)
        elif abs(x - self._map_max) < 2:
            return (self._map_min, self._clamp(y + random.uniform(-spread, spread)))
        else:
            return (self._map_max, self._clamp(y + random.uniform(-spread, spread)))

    def _clamp(self, v: float) -> float:
        return max(self._map_min, min(self._map_max, v))

    def _sidewalk_path(self, start: tuple[float, float]) -> list[tuple[float, float]]:
        """Generate a path along sidewalks following the street grid.

        People walk along streets, not through houses.  The path makes an
        L-shaped turn at a street corner, producing the right-angle walking
        pattern you see in real neighborhoods.  Corner points validated
        against building obstacles.
        """
        end = self._opposite_edge(start)
        path = _street_path(start, end, self._streets_ns, self._streets_ew)
        # Validate all waypoints against buildings
        return [self._safe_point(p[0], p[1]) for p in path]

    def _road_path(self, start: tuple[float, float]) -> list[tuple[float, float]]:
        """Generate a road path following the street grid.

        Cars follow streets.  Unlike the old straight-line path, this snaps
        to the nearest NS or EW street corridor so the car visibly drives
        along a road.  Corner points validated against building obstacles.
        """
        end = self._opposite_edge(start)
        path = _street_path(start, end, self._streets_ns, self._streets_ew)
        return [self._safe_point(p[0], p[1]) for p in path]

    def _point_in_building(self, x: float, y: float) -> bool:
        """Check if a point is inside any building (if obstacles loaded)."""
        obs = getattr(self._engine, '_obstacles', None)
        if obs is None:
            return False
        return obs.point_in_building(x, y)

    def _safe_point(self, x: float, y: float, max_attempts: int = 10) -> tuple[float, float]:
        """Jitter a point until it's outside all buildings, or return an edge point."""
        if not self._point_in_building(x, y):
            return (x, y)
        for _ in range(max_attempts):
            jx = x + random.uniform(-5, 5)
            jy = y + random.uniform(-5, 5)
            jx = max(self._map_min, min(self._map_max, jx))
            jy = max(self._map_min, min(self._map_max, jy))
            if not self._point_in_building(jx, jy):
                return (jx, jy)
        # Fallback: edge point (always safe)
        return self._random_edge()

    def _yard_wander(self) -> tuple[tuple[float, float], list[tuple[float, float]]]:
        """Generate a start + wander path within a yard area.

        All generated points are validated against building obstacles.
        Yard areas are scattered across the full map, not just near center.
        """
        # Pick a random yard area anywhere within 80% of map bounds
        yard_range = self._map_bounds * 0.8
        cx = random.uniform(-yard_range, yard_range)
        cy = random.uniform(-yard_range, yard_range)
        start = self._safe_point(cx + random.uniform(-3, 3), cy + random.uniform(-3, 3))
        points = []
        for _ in range(random.randint(3, 5)):
            pt = self._safe_point(
                self._clamp(cx + random.uniform(-5, 5)),
                self._clamp(cy + random.uniform(-5, 5)),
            )
            points.append(pt)
        # Exit toward nearest edge
        exit_point = self._random_edge()
        points.append(exit_point)
        return start, points

    def _delivery_path(self) -> tuple[tuple[float, float], list[tuple[float, float]]]:
        """Generate delivery path: road edge -> front door -> pause -> back.

        Front door position validated against building obstacles.
        Delivers to random locations across the full map area.
        """
        start = self._random_edge()
        # "Front door" somewhere in the interior — avoid buildings
        interior_range = self._map_bounds * 0.7
        door = self._safe_point(
            random.uniform(-interior_range, interior_range),
            random.uniform(-interior_range, interior_range),
        )
        # Path: approach -> door -> wait (same point, speed makes them pause) -> back to edge
        return_point = self._random_edge()
        return start, [door, door, return_point]

    def _spawn_neighbor(self) -> None:
        start = self._random_edge()
        waypoints = self._sidewalk_path(start)
        name = self._pick_name(_NEIGHBOR_NAMES)
        target = SimulationTarget(
            target_id=str(uuid.uuid4()),
            name=name,
            alliance="neutral",
            asset_type="person",
            position=start,
            speed=1.2,
            waypoints=waypoints,
        )
        self._engine.add_target(target)

    def _spawn_car(self) -> None:
        start = self._random_edge()
        waypoints = self._road_path(start)
        name = self._pick_name(_CAR_NAMES)
        target = SimulationTarget(
            target_id=str(uuid.uuid4()),
            name=name,
            alliance="neutral",
            asset_type="vehicle",
            position=start,
            speed=8.0,
            waypoints=waypoints,
        )
        self._engine.add_target(target)

    def _spawn_dog(self) -> None:
        start, waypoints = self._yard_wander()
        name = self._pick_name(_DOG_NAMES)
        target = SimulationTarget(
            target_id=str(uuid.uuid4()),
            name=name,
            alliance="neutral",
            asset_type="animal",
            position=start,
            speed=2.0,
            waypoints=waypoints,
        )
        self._engine.add_target(target)

    def _spawn_cat(self) -> None:
        start, waypoints = self._yard_wander()
        name = self._pick_name(_CAT_NAMES)
        target = SimulationTarget(
            target_id=str(uuid.uuid4()),
            name=name,
            alliance="neutral",
            asset_type="animal",
            position=start,
            speed=1.5,
            waypoints=waypoints,
        )
        self._engine.add_target(target)

    def _spawn_delivery(self) -> None:
        start, waypoints = self._delivery_path()
        name = self._pick_name(_DELIVERY_NAMES)
        target = SimulationTarget(
            target_id=str(uuid.uuid4()),
            name=name,
            alliance="neutral",
            asset_type="person",
            position=start,
            speed=1.0,
            waypoints=waypoints,
        )
        self._engine.add_target(target)

    def release_name(self, name: str) -> None:
        """Release a name when a target is removed."""
        self._used_names.discard(name)
