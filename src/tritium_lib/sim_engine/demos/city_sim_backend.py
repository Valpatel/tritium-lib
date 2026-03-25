# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Python-side city simulation backend using actual sim_engine modules.

Replaces the JavaScript-side simulation in city3d.html. The Python sim_engine
runs the simulation (all 60+ modules) and streams Three.js-compatible frame
data via WebSocket. city3d.html becomes a pure renderer.

Usage::

    from tritium_lib.sim_engine.demos.city_sim_backend import CitySim

    sim = CitySim(seed=42)
    sim.setup()
    for _ in range(100):
        frame = sim.tick(dt=0.05)
        # Send frame dict to city3d.html via WebSocket
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any

from tritium_lib.sim_engine.world import World, WorldConfig, WorldBuilder
from tritium_lib.sim_engine.crowd import CrowdSimulator, CrowdMood, CrowdEvent
from tritium_lib.sim_engine.units import Unit, Alliance, UnitType, create_unit
from tritium_lib.sim_engine.vehicles import (
    VehicleState,
    VehicleClass,
    VehiclePhysicsEngine,
    create_vehicle,
    VEHICLE_TEMPLATES,
)
from tritium_lib.sim_engine.environment import (
    Environment,
    TimeOfDay,
    Weather,
    WeatherSimulator,
)
from tritium_lib.sim_engine.destruction import (
    DestructionEngine,
    Structure,
    StructureType,
    MATERIAL_PROPERTIES,
)
from tritium_lib.sim_engine.fortifications import EngineeringEngine
from tritium_lib.sim_engine.detection import DetectionEngine
from tritium_lib.sim_engine.comms import CommsSimulator
from tritium_lib.sim_engine.medical import MedicalEngine
from tritium_lib.sim_engine.logistics import LogisticsEngine
from tritium_lib.sim_engine.scoring import ScoringEngine
from tritium_lib.sim_engine.factions import DiplomacyEngine, Faction
from tritium_lib.sim_engine.commander import BattleNarrator
from tritium_lib.sim_engine.territory import InfluenceMap, TerritoryControl
from tritium_lib.sim_engine.objectives import ObjectiveEngine
from tritium_lib.sim_engine.arsenal import ProjectileSimulator, ARSENAL
from tritium_lib.sim_engine.damage import DamageTracker
from tritium_lib.sim_engine.renderer import SimRenderer
from tritium_lib.sim_engine.telemetry import TelemetrySession
from tritium_lib.sim_engine.ai.tactics import TacticsEngine
from tritium_lib.sim_engine.ai.squad import Squad
from tritium_lib.sim_engine.ai.steering import Vec2, distance, normalize, _add, _sub, _scale
from tritium_lib.sim_engine.mapgen import MapGenerator, GeneratedMap


# ---------------------------------------------------------------------------
# Vehicle templates for civilian vehicles (not in the military templates)
# ---------------------------------------------------------------------------

_CIVILIAN_CAR_DEFAULTS = {
    "max_speed": 50.0 / 3.6,   # 50 km/h city speed
    "acceleration": 3.0,
    "turn_rate": 1.5,
    "max_health": 100.0,
    "armor": 0.0,
    "fuel_consumption": 0.005,
}

_TAXI_DEFAULTS = {
    "max_speed": 45.0 / 3.6,
    "acceleration": 2.5,
    "turn_rate": 1.5,
    "max_health": 100.0,
    "armor": 0.0,
    "fuel_consumption": 0.006,
}


# ---------------------------------------------------------------------------
# CityEntity — lightweight tracked entity for civilians/trees/etc.
# ---------------------------------------------------------------------------

@dataclass
class CityEntity:
    """A city entity not managed by the World's unit system (e.g. tree, bench)."""

    entity_id: str
    entity_type: str   # "tree", "bench", "lamp", "hydrant"
    position: Vec2
    size: tuple[float, float, float] = (2.0, 2.0, 4.0)  # w, d, h
    rotation: float = 0.0
    properties: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Daily routine system for civilians
# ---------------------------------------------------------------------------

# Roles and their distribution weights
CIVILIAN_ROLES = [
    ("resident", 0.35),
    ("worker", 0.20),
    ("student", 0.10),
    ("shopkeeper", 0.08),
    ("jogger", 0.10),
    ("dogwalker", 0.07),
    ("retired", 0.10),
]


@dataclass
class ScheduleGoal:
    """A single goal in a civilian's daily schedule.

    Mirrors the JS daily-routine.js RoutineGoal structure.
    """
    action: str          # 'go_to', 'stay_at', 'wander', 'idle'
    destination: str     # POI type: 'home', 'work', 'park', 'commercial', 'school'
    start_hour: float    # When this goal begins (0-24)
    duration: float = 1.0
    transport: str = "walk"  # 'walk', 'car'
    mood: str = "calm"       # 'calm', 'hurried', 'relaxed'


def _generate_routine(role: str, rng: random.Random) -> list[ScheduleGoal]:
    """Generate a daily routine for a civilian role.

    Mirrors the JS generateDailyRoutine() function.
    """
    r = rng.random
    if role == "worker":
        return _worker_routine(rng)
    elif role == "student":
        return _student_routine(rng)
    elif role == "shopkeeper":
        return _shopkeeper_routine(rng)
    elif role == "jogger":
        return _jogger_routine(rng)
    elif role == "dogwalker":
        return _dogwalker_routine(rng)
    elif role == "retired":
        return _retired_routine(rng)
    else:  # resident (default)
        return _resident_routine(rng)


def _resident_routine(rng: random.Random) -> list[ScheduleGoal]:
    wake = 6.5 + rng.random() * 2
    work_start = wake + 0.5 + rng.random() * 0.5
    lunch = 12 + rng.random() * 0.5
    work_end = 17 + rng.random()
    goals = [
        ScheduleGoal("stay_at", "home", 0, wake),
        ScheduleGoal("go_to", "work", wake, transport="walk", mood="hurried"),
        ScheduleGoal("stay_at", "work", work_start, lunch - work_start),
        ScheduleGoal("go_to", "commercial", lunch, transport="walk", mood="relaxed"),
        ScheduleGoal("stay_at", "commercial", lunch + 0.1, 0.7),
        ScheduleGoal("go_to", "work", lunch + 0.8, transport="walk"),
        ScheduleGoal("stay_at", "work", lunch + 1, work_end - lunch - 1),
        ScheduleGoal("go_to", "home", work_end, transport="walk"),
    ]
    if rng.random() < 0.4:
        dest = "park" if rng.random() < 0.5 else "commercial"
        goals.append(ScheduleGoal("go_to", dest, work_end + 0.5, transport="walk", mood="relaxed"))
        goals.append(ScheduleGoal("wander", dest, work_end + 0.7, 0.5 + rng.random()))
        goals.append(ScheduleGoal("go_to", "home", work_end + 1.5 + rng.random(), transport="walk"))
    goals.append(ScheduleGoal("stay_at", "home", 21, 10))
    return goals


def _worker_routine(rng: random.Random) -> list[ScheduleGoal]:
    goals = _resident_routine(rng)
    for g in goals:
        if g.transport != "walk":
            g.transport = "car"
    return goals


def _student_routine(rng: random.Random) -> list[ScheduleGoal]:
    wake = 7 + rng.random() * 0.5
    return [
        ScheduleGoal("stay_at", "home", 0, wake),
        ScheduleGoal("go_to", "school", wake, transport="walk", mood="calm"),
        ScheduleGoal("stay_at", "school", wake + 0.3, 7),
        ScheduleGoal("go_to", "park" if rng.random() < 0.5 else "home", 15, transport="walk", mood="relaxed"),
        ScheduleGoal("wander", "park", 15.3, 1 + rng.random()),
        ScheduleGoal("go_to", "home", 17, transport="walk"),
        ScheduleGoal("stay_at", "home", 17.5, 13),
    ]


def _shopkeeper_routine(rng: random.Random) -> list[ScheduleGoal]:
    open_h = 9 + rng.random() * 0.5
    close_h = 18 + rng.random() * 2
    return [
        ScheduleGoal("stay_at", "home", 0, open_h - 0.5),
        ScheduleGoal("go_to", "commercial", open_h - 0.5, transport="walk"),
        ScheduleGoal("stay_at", "commercial", open_h, close_h - open_h),
        ScheduleGoal("go_to", "home", close_h, transport="walk", mood="relaxed"),
        ScheduleGoal("stay_at", "home", close_h + 0.3, 24 - close_h),
    ]


def _jogger_routine(rng: random.Random) -> list[ScheduleGoal]:
    jog_h = 6 + rng.random() * 2
    return [
        ScheduleGoal("stay_at", "home", 0, jog_h),
        ScheduleGoal("go_to", "park", jog_h, transport="walk", mood="hurried"),
        ScheduleGoal("wander", "park", jog_h + 0.2, 0.5 + rng.random() * 0.5),
        ScheduleGoal("go_to", "home", jog_h + 1, transport="walk", mood="relaxed"),
        # Rest of day: stay home or go to work
        ScheduleGoal("stay_at", "home", jog_h + 1.5, 2),
        ScheduleGoal("go_to", "work", jog_h + 3.5, transport="walk"),
        ScheduleGoal("stay_at", "work", jog_h + 4, 17 - (jog_h + 4)),
        ScheduleGoal("go_to", "home", 17, transport="walk"),
        ScheduleGoal("stay_at", "home", 17.5, 13.5),
    ]


def _dogwalker_routine(rng: random.Random) -> list[ScheduleGoal]:
    walk_h = 7 + rng.random()
    eve_walk = 18 + rng.random()
    return [
        ScheduleGoal("stay_at", "home", 0, walk_h),
        ScheduleGoal("go_to", "park", walk_h, transport="walk", mood="calm"),
        ScheduleGoal("wander", "park", walk_h + 0.2, 0.4 + rng.random() * 0.3),
        ScheduleGoal("go_to", "home", walk_h + 0.7, transport="walk"),
        ScheduleGoal("stay_at", "home", walk_h + 1, eve_walk - walk_h - 1),
        ScheduleGoal("go_to", "park", eve_walk, transport="walk", mood="relaxed"),
        ScheduleGoal("wander", "park", eve_walk + 0.2, 0.3),
        ScheduleGoal("go_to", "home", eve_walk + 0.6, transport="walk"),
        ScheduleGoal("stay_at", "home", 21, 10),
    ]


def _retired_routine(rng: random.Random) -> list[ScheduleGoal]:
    return [
        ScheduleGoal("stay_at", "home", 0, 6),
        ScheduleGoal("go_to", "park", 6.5, transport="walk", mood="calm"),
        ScheduleGoal("wander", "park", 6.7, 0.8),
        ScheduleGoal("go_to", "home", 7.5, transport="walk"),
        ScheduleGoal("stay_at", "home", 8, 4),
        ScheduleGoal("go_to", "commercial", 12, transport="walk", mood="relaxed"),
        ScheduleGoal("stay_at", "commercial", 12.3, 0.7),
        ScheduleGoal("go_to", "home", 13, transport="walk"),
        ScheduleGoal("stay_at", "home", 13.5, 2.5),
        ScheduleGoal("go_to", "park", 16, transport="walk", mood="relaxed"),
        ScheduleGoal("wander", "park", 16.2, 0.8),
        ScheduleGoal("go_to", "home", 17, transport="walk"),
        ScheduleGoal("stay_at", "home", 17.5, 13.5),
    ]


def _pick_role(rng: random.Random) -> str:
    """Pick a civilian role based on weighted distribution."""
    r = rng.random()
    cumulative = 0.0
    for role, weight in CIVILIAN_ROLES:
        cumulative += weight
        if r < cumulative:
            return role
    return "resident"


def _current_goal(routine: list[ScheduleGoal], hour: float) -> ScheduleGoal | None:
    """Get the active goal for a given hour."""
    active: ScheduleGoal | None = None
    for goal in routine:
        if hour >= goal.start_hour:
            active = goal
        else:
            break
    return active


# ---------------------------------------------------------------------------
# CivilianAgent — a walking civilian with daily routine behavior
# ---------------------------------------------------------------------------

@dataclass
class CivilianAgent:
    """A civilian walking on sidewalks with daily-routine-driven behavior.

    Each civilian has a role (resident, worker, student, etc.) and a daily
    schedule that determines where they go at what time. The schedule is
    checked each tick against the sim clock to trigger goal transitions.
    """

    agent_id: str
    position: Vec2
    target: Vec2
    speed: float = 1.2   # m/s walking speed
    heading: float = 0.0
    state: str = "walking"  # walking, idle, waiting, at_poi
    waypoints: list[Vec2] = field(default_factory=list)
    _waypoint_idx: int = 0
    _idle_timer: float = 0.0

    # Daily routine fields
    role: str = "resident"
    routine: list[ScheduleGoal] = field(default_factory=list)
    activity: str = "idle"       # Current high-level activity description
    destination: str = "home"    # Current destination POI type
    _current_goal_idx: int = -1  # Index of current goal in routine
    _poi_locations: dict[str, Vec2] = field(default_factory=dict)  # POI type -> position

    def assign_schedule(
        self, role: str, routine: list[ScheduleGoal],
        poi_locations: dict[str, Vec2],
    ) -> None:
        """Assign a daily routine to this civilian."""
        self.role = role
        self.routine = routine
        self._poi_locations = poi_locations
        self._current_goal_idx = -1

    def tick(self, dt: float, rng: random.Random, bounds: tuple[float, float],
             sim_hour: float | None = None) -> None:
        """Advance the civilian by dt seconds.

        If sim_hour is provided and a routine is assigned, the civilian
        follows their daily schedule. Otherwise falls back to random walking.
        """
        # Check schedule transitions
        if sim_hour is not None and self.routine:
            self._check_schedule(sim_hour, rng, bounds)

        if self.state == "at_poi":
            # Staying at a point of interest — don't move
            return

        if self.state == "idle":
            self._idle_timer -= dt
            if self._idle_timer <= 0:
                self.state = "walking"
                # Pick a new random target if no routine drives us
                if not self.routine:
                    self.target = (
                        rng.uniform(10, bounds[0] - 10),
                        rng.uniform(10, bounds[1] - 10),
                    )
            return

        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        dist = math.hypot(dx, dy)

        if dist < 2.0:
            # Reached target — either advance to next waypoint or idle
            if self.waypoints and self._waypoint_idx < len(self.waypoints) - 1:
                self._waypoint_idx += 1
                self.target = self.waypoints[self._waypoint_idx]
            else:
                # Arrived at destination
                goal = _current_goal(self.routine, 0) if not self.routine else None
                if self.routine and self._current_goal_idx >= 0:
                    goal = self.routine[self._current_goal_idx]

                if goal and goal.action == "stay_at":
                    self.state = "at_poi"
                    self.activity = f"at_{goal.destination}"
                elif goal and goal.action == "wander":
                    # Wander: pick random nearby point
                    self.target = (
                        self.position[0] + rng.uniform(-30, 30),
                        self.position[1] + rng.uniform(-30, 30),
                    )
                    self.target = (
                        max(10, min(bounds[0] - 10, self.target[0])),
                        max(10, min(bounds[1] - 10, self.target[1])),
                    )
                    self.state = "walking"
                    self.activity = f"wandering_{goal.destination}"
                else:
                    self.state = "idle"
                    self._idle_timer = rng.uniform(2.0, 8.0)
                self.waypoints = []
                self._waypoint_idx = 0
            return

        # Move toward target
        nx, ny = dx / dist, dy / dist
        self.heading = math.atan2(ny, nx)
        move = self.speed * dt
        self.position = (
            self.position[0] + nx * move,
            self.position[1] + ny * move,
        )

    def _check_schedule(self, hour: float, rng: random.Random,
                        bounds: tuple[float, float]) -> None:
        """Check if the civilian should transition to a new goal."""
        goal = _current_goal(self.routine, hour)
        if goal is None:
            return

        # Find index of this goal
        try:
            goal_idx = self.routine.index(goal)
        except ValueError:
            return

        if goal_idx == self._current_goal_idx:
            return  # Already executing this goal

        # Transition to new goal
        self._current_goal_idx = goal_idx
        self.destination = goal.destination
        self.activity = f"{goal.action}_{goal.destination}"

        if goal.action in ("go_to", "wander"):
            # Navigate to the destination POI
            dest_pos = self._poi_locations.get(goal.destination)
            if dest_pos is None:
                # Fallback: random position
                dest_pos = (
                    rng.uniform(20, bounds[0] - 20),
                    rng.uniform(20, bounds[1] - 20),
                )
            if goal.action == "wander":
                # Wander near the POI
                dest_pos = (
                    dest_pos[0] + rng.uniform(-25, 25),
                    dest_pos[1] + rng.uniform(-25, 25),
                )
                dest_pos = (
                    max(10, min(bounds[0] - 10, dest_pos[0])),
                    max(10, min(bounds[1] - 10, dest_pos[1])),
                )
            self.target = dest_pos
            self.state = "walking"
            self.speed = 1.8 if goal.mood == "hurried" else (1.0 if goal.mood == "relaxed" else 1.2)
            self.waypoints = []
            self._waypoint_idx = 0

        elif goal.action == "stay_at":
            # If already near destination, stay put. Otherwise travel there.
            dest_pos = self._poi_locations.get(goal.destination)
            if dest_pos is not None:
                dist_to = math.hypot(
                    self.position[0] - dest_pos[0],
                    self.position[1] - dest_pos[1],
                )
                if dist_to > 5.0:
                    self.target = dest_pos
                    self.state = "walking"
                    self.waypoints = []
                    self._waypoint_idx = 0
                else:
                    self.state = "at_poi"
                    self.activity = f"at_{goal.destination}"
            else:
                self.state = "at_poi"
                self.activity = f"at_{goal.destination}"

        elif goal.action == "idle":
            self.state = "idle"
            self._idle_timer = goal.duration * 3600.0  # Convert hours to seconds
            self.activity = "idle"

    def to_dict(self) -> dict[str, Any]:
        """Three.js-compatible output."""
        return {
            "id": self.agent_id,
            "type": "civilian",
            "x": round(self.position[0], 2),
            "y": round(self.position[1], 2),
            "heading": round(self.heading, 3),
            "speed": self.speed if self.state == "walking" else 0.0,
            "state": self.state,
            "role": self.role,
            "activity": self.activity,
            "destination": self.destination,
        }


# ---------------------------------------------------------------------------
# CityVehicle — a vehicle driving on roads with simple route following
# ---------------------------------------------------------------------------

@dataclass
class CityVehicle:
    """A vehicle following a route on city roads."""

    vehicle_id: str
    vehicle_type: str   # "car", "taxi", "police_car"
    position: Vec2
    heading: float = 0.0
    speed: float = 0.0
    max_speed: float = 14.0  # m/s (~50 km/h)
    route: list[Vec2] = field(default_factory=list)
    _route_idx: int = 0
    _stopped_timer: float = 0.0
    color: str = "#888888"
    alliance: str = "neutral"

    def tick(self, dt: float, rng: random.Random, bounds: tuple[float, float]) -> None:
        """Advance the vehicle along its route."""
        if self._stopped_timer > 0:
            self._stopped_timer -= dt
            self.speed = 0.0
            return

        if not self.route or self._route_idx >= len(self.route):
            # Generate new route
            self._generate_route(rng, bounds)
            return

        target = self.route[self._route_idx]
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        dist = math.hypot(dx, dy)

        if dist < 5.0:
            self._route_idx += 1
            if self._route_idx >= len(self.route):
                # Route complete, stop briefly then generate new route
                self._stopped_timer = rng.uniform(1.0, 5.0)
                self.speed = 0.0
                return
            # Chance to stop at intersection
            if rng.random() < 0.15:
                self._stopped_timer = rng.uniform(2.0, 6.0)
                self.speed = 0.0
                return
            return

        # Accelerate / decelerate
        desired_speed = self.max_speed
        if dist < 20.0:
            desired_speed = max(2.0, self.max_speed * (dist / 20.0))

        if self.speed < desired_speed:
            self.speed = min(self.speed + 3.0 * dt, desired_speed)
        elif self.speed > desired_speed:
            self.speed = max(self.speed - 5.0 * dt, desired_speed)

        nx, ny = dx / dist, dy / dist
        self.heading = math.atan2(ny, nx)
        move = self.speed * dt
        self.position = (
            self.position[0] + nx * move,
            self.position[1] + ny * move,
        )

    def _generate_route(self, rng: random.Random, bounds: tuple[float, float]) -> None:
        """Generate a simple route within bounds."""
        self._route_idx = 0
        self.route = []
        current = self.position
        for _ in range(rng.randint(3, 8)):
            # Move along grid-like roads
            if rng.random() < 0.5:
                # Horizontal
                nx = rng.uniform(10, bounds[0] - 10)
                ny = current[1]
            else:
                # Vertical
                nx = current[0]
                ny = rng.uniform(10, bounds[1] - 10)
            # Clamp
            nx = max(10, min(bounds[0] - 10, nx))
            ny = max(10, min(bounds[1] - 10, ny))
            self.route.append((nx, ny))
            current = (nx, ny)

    def to_dict(self) -> dict[str, Any]:
        """Three.js-compatible output."""
        return {
            "id": self.vehicle_id,
            "type": self.vehicle_type,
            "x": round(self.position[0], 2),
            "y": round(self.position[1], 2),
            "heading": round(self.heading, 3),
            "speed": round(self.speed, 2),
            "color": self.color,
            "alliance": self.alliance,
        }


# ---------------------------------------------------------------------------
# CitySim — the main simulation class
# ---------------------------------------------------------------------------

class CitySim:
    """City simulation backend using actual sim_engine modules.

    Creates a living city with civilians, vehicles, police, crowds,
    buildings, trees, weather, and time-of-day. Ticks all subsystems
    and outputs Three.js-compatible frame data.

    Parameters
    ----------
    width : float
        Map width in meters. Default 500.
    height : float
        Map height in meters. Default 400.
    seed : int or None
        Random seed for reproducibility.
    hour : float
        Starting hour of day (0-24). Default 10.0 (10 AM).
    weather : Weather
        Starting weather condition. Default CLEAR.
    """

    def __init__(
        self,
        width: float = 500.0,
        height: float = 400.0,
        seed: int | None = None,
        hour: float = 10.0,
        weather: Weather = Weather.CLEAR,
    ) -> None:
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(0, 2**31)
        self._rng = random.Random(self.seed)
        self._next_id = 0

        # Core subsystems
        self.environment = Environment(
            time=TimeOfDay(hour),
            weather=WeatherSimulator(initial=weather, seed=self._rng.randint(0, 2**31)),
        )
        self.destruction = DestructionEngine(
            rng=random.Random(self._rng.randint(0, 2**31))
        )
        self.crowd: CrowdSimulator | None = None
        self.renderer = SimRenderer()
        self.telemetry = TelemetrySession(metadata={"type": "city_sim", "seed": self.seed})

        # Advanced subsystems
        self.detection = DetectionEngine()
        self.comms = CommsSimulator()
        self.medical = MedicalEngine()
        self.logistics = LogisticsEngine()
        self.scoring = ScoringEngine()
        self.diplomacy = DiplomacyEngine()
        self.narrator = BattleNarrator()
        self.influence_map = InfluenceMap(
            width=max(1, int(width / 10)),
            height=max(1, int(height / 10)),
            cell_size=10.0,
        )
        self.territory = TerritoryControl(influence_map=self.influence_map)
        self.objectives = ObjectiveEngine()
        self.engineering = EngineeringEngine()
        self.tactics = TacticsEngine()

        # Entity containers
        self.civilians: list[CivilianAgent] = []
        self.city_vehicles: list[CityVehicle] = []
        self.police_units: list[Unit] = []
        self.buildings: list[Structure] = []
        self.trees: list[CityEntity] = []
        self.props: list[CityEntity] = []  # benches, lamps, etc.

        # Map data (from MapGenerator)
        self.map_data: GeneratedMap | None = None
        self.roads: list[list[Vec2]] = []

        # Bookkeeping
        self.tick_count: int = 0
        self.sim_time: float = 0.0
        self.events: list[dict[str, Any]] = []
        self._is_setup: bool = False

    def _gen_id(self, prefix: str = "e") -> str:
        self._next_id += 1
        return f"{prefix}_{self._next_id}"

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize the city: generate map, spawn all entities."""
        self._generate_map()
        self._setup_factions()
        self._spawn_buildings()
        self._spawn_trees()
        self._spawn_civilians(25)
        self._spawn_cars(12)
        self._spawn_taxis(3)
        self._spawn_police(10)
        self._spawn_protestors(20)
        self._setup_weather()
        self._is_setup = True

    def _generate_map(self) -> None:
        """Generate a city map using MapGenerator."""
        gen = MapGenerator(
            width=self.width,
            height=self.height,
            cell_size=5.0,
            seed=self.seed,
        )
        gen.generate_terrain("flat")
        gen.add_city(
            center=(self.width / 2, self.height / 2),
            radius=min(self.width, self.height) * 0.35,
            density=0.6,
        )
        # Add some connecting roads
        gen.add_road((0, self.height / 2), (self.width, self.height / 2))
        gen.add_road((self.width / 2, 0), (self.width / 2, self.height))
        gen.place_spawn_points(["police", "civilian"], min_distance=100)

        self.map_data = gen.result()
        self.roads = self.map_data.roads

    def _setup_factions(self) -> None:
        """Set up city factions."""
        self.diplomacy.add_faction(Faction(
            faction_id="police",
            name="City Police",
            color="#0055ff",
            ideology="government",
            strength=0.6,
            wealth=0.5,
        ))
        self.diplomacy.add_faction(Faction(
            faction_id="civilians",
            name="Civilians",
            color="#00ff88",
            ideology="civilian",
            strength=0.1,
            wealth=0.3,
        ))
        self.diplomacy.add_faction(Faction(
            faction_id="protestors",
            name="Protestors",
            color="#ffaa00",
            ideology="rebel",
            strength=0.2,
            wealth=0.1,
        ))

    def _spawn_buildings(self) -> None:
        """Create Structure objects from the map features for the destruction engine."""
        if self.map_data is None:
            return
        for feat in self.map_data.features:
            if feat.feature_type != "building":
                continue
            mat = feat.properties.get("material", "concrete")
            mat_props = MATERIAL_PROPERTIES.get(mat, MATERIAL_PROPERTIES["concrete"])
            height = feat.properties.get("height", 8.0)
            structure = Structure(
                structure_id=feat.feature_id,
                structure_type=StructureType.BUILDING,
                position=feat.position,
                size=(feat.size[0], feat.size[1], height),
                material=mat,
                health=mat_props["health"],
                max_health=mat_props["health"],
            )
            self.destruction.add_structure(structure)
            self.buildings.append(structure)

    def _spawn_trees(self) -> None:
        """Spawn trees and props around the city."""
        if self.map_data is None:
            return
        for feat in self.map_data.features:
            if feat.feature_type == "forest":
                tree = CityEntity(
                    entity_id=feat.feature_id,
                    entity_type="tree",
                    position=feat.position,
                    size=(feat.size[0], feat.size[1], self._rng.uniform(4, 10)),
                    rotation=feat.rotation,
                    properties=feat.properties,
                )
                self.trees.append(tree)

        # Add street-side trees and props along roads
        for road in self.roads[:6]:  # Just a few roads
            for i in range(0, len(road) - 1, max(1, len(road) // 4)):
                pt = road[i]
                # Tree offset from road
                offset = self._rng.choice([-8, 8])
                tree_pos = (pt[0] + offset, pt[1] + offset * 0.3)
                if 5 < tree_pos[0] < self.width - 5 and 5 < tree_pos[1] < self.height - 5:
                    self.trees.append(CityEntity(
                        entity_id=self._gen_id("tree"),
                        entity_type="tree",
                        position=tree_pos,
                        size=(2, 2, self._rng.uniform(5, 8)),
                        rotation=self._rng.uniform(0, 360),
                        properties={"tree_type": self._rng.choice(["oak", "maple", "elm"])},
                    ))

    def _spawn_civilians(self, count: int) -> None:
        """Spawn civilians with daily routine schedules.

        Each civilian is assigned a random role and a full daily routine
        that drives their behavior throughout the sim day. POI locations
        (home, work, park, etc.) are assigned from the building list.
        """
        # Pre-compute POI zones from buildings for schedule destinations
        poi_zones = self._compute_poi_zones()

        for _ in range(count):
            role = _pick_role(self._rng)
            routine = _generate_routine(role, self._rng)

            # Assign personal POI locations (home, work, etc.)
            home_pos = (
                self._rng.uniform(20, self.width - 20),
                self._rng.uniform(20, self.height - 20),
            )
            poi_locs: dict[str, Vec2] = {"home": home_pos}

            # Assign other POIs from building zones or random positions
            for poi_type in ("work", "school", "commercial", "park"):
                if poi_type in poi_zones and poi_zones[poi_type]:
                    base = self._rng.choice(poi_zones[poi_type])
                    poi_locs[poi_type] = (
                        base[0] + self._rng.uniform(-10, 10),
                        base[1] + self._rng.uniform(-10, 10),
                    )
                else:
                    poi_locs[poi_type] = (
                        self._rng.uniform(20, self.width - 20),
                        self._rng.uniform(20, self.height - 20),
                    )

            agent = CivilianAgent(
                agent_id=self._gen_id("civ"),
                position=home_pos,
                target=home_pos,
                speed=self._rng.uniform(0.8, 1.5),
            )
            agent.assign_schedule(role, routine, poi_locs)
            self.civilians.append(agent)

    def _compute_poi_zones(self) -> dict[str, list[Vec2]]:
        """Extract POI positions from buildings for schedule destinations."""
        zones: dict[str, list[Vec2]] = {
            "work": [],
            "school": [],
            "commercial": [],
            "park": [],
        }
        if self.map_data is None:
            return zones
        for feat in self.map_data.features:
            if feat.feature_type != "building":
                continue
            btype = feat.properties.get("building_type", "")
            pos = feat.position
            if btype in ("office", "commercial", "industrial"):
                zones["work"].append(pos)
                zones["commercial"].append(pos)
            elif btype in ("school", "university"):
                zones["school"].append(pos)
            elif btype in ("retail", "shop", "restaurant"):
                zones["commercial"].append(pos)

        # Parks: use green spaces or fallback to random open areas
        for feat in self.map_data.features:
            if feat.feature_type in ("park", "forest", "green"):
                zones["park"].append(feat.position)

        # Ensure every zone has at least one location
        for key in zones:
            if not zones[key]:
                zones[key] = [(
                    self._rng.uniform(30, self.width - 30),
                    self._rng.uniform(30, self.height - 30),
                )]

        return zones

    def _spawn_cars(self, count: int) -> None:
        """Spawn civilian cars on roads."""
        colors = ["#444444", "#666666", "#993333", "#334499",
                  "#339933", "#999999", "#663399", "#996633"]
        for _ in range(count):
            pos = (
                self._rng.uniform(20, self.width - 20),
                self._rng.uniform(20, self.height - 20),
            )
            car = CityVehicle(
                vehicle_id=self._gen_id("car"),
                vehicle_type="car",
                position=pos,
                max_speed=self._rng.uniform(10.0, 16.0),
                color=self._rng.choice(colors),
            )
            car._generate_route(self._rng, (self.width, self.height))
            self.city_vehicles.append(car)

    def _spawn_taxis(self, count: int) -> None:
        """Spawn taxis on roads."""
        for _ in range(count):
            pos = (
                self._rng.uniform(20, self.width - 20),
                self._rng.uniform(20, self.height - 20),
            )
            taxi = CityVehicle(
                vehicle_id=self._gen_id("taxi"),
                vehicle_type="taxi",
                position=pos,
                max_speed=self._rng.uniform(8.0, 14.0),
                color="#ffcc00",  # Yellow taxi
            )
            taxi._generate_route(self._rng, (self.width, self.height))
            self.city_vehicles.append(taxi)

    def _spawn_police(self, count: int) -> None:
        """Spawn a police squad using the sim_engine unit system."""
        for i in range(count):
            uid = self._gen_id("police")
            pos = (
                self._rng.uniform(50, self.width - 50),
                self._rng.uniform(50, self.height - 50),
            )
            unit = create_unit("infantry", uid, f"Officer_{i+1}", Alliance.FRIENDLY, pos)
            unit.stats.speed = 3.5  # Patrol speed
            unit.stats.attack_range = 0.0  # Not in combat mode
            unit.stats.detection_range = 30.0
            self.police_units.append(unit)
            self.scoring.register_unit(uid, f"Officer_{i+1}", "police")

    def _spawn_protestors(self, count: int) -> None:
        """Spawn a protestor crowd using the CrowdSimulator."""
        self.crowd = CrowdSimulator(
            bounds=(0.0, 0.0, self.width, self.height),
            max_members=200,
        )
        # Protestors gather near center
        center = (self.width / 2 + self._rng.uniform(-30, 30),
                  self.height / 2 + self._rng.uniform(-30, 30))
        self.crowd.spawn_crowd(
            center=center,
            count=count,
            radius=25.0,
            mood=CrowdMood.AGITATED,
            leader_ratio=0.05,
        )

    def _setup_weather(self) -> None:
        """Configure initial weather."""
        # Weather already set in __init__, nothing extra needed
        pass

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, dt: float = 0.05) -> dict[str, Any]:
        """Advance the city simulation by dt seconds.

        Returns a Three.js-compatible frame dict.
        """
        if not self._is_setup:
            self.setup()

        self.events = []

        # 1. Environment (weather, time of day)
        self.environment.update(dt)

        # 2. Civilians follow daily routines
        sim_hour = self.environment.time.hour
        for civ in self.civilians:
            civ.tick(dt, self._rng, (self.width, self.height), sim_hour=sim_hour)

        # 3. Vehicles drive
        for veh in self.city_vehicles:
            veh.tick(dt, self._rng, (self.width, self.height))

        # 4. Police patrol — simple wander behavior
        self._tick_police(dt)

        # 5. Crowd simulation
        if self.crowd is not None:
            self.crowd.tick(dt)

        # 6. Destruction (fire spread, etc.)
        dest_events = self.destruction.tick(dt)
        if dest_events.get("fires_spread") or dest_events.get("structures_damaged"):
            self.events.append({"type": "destruction_tick", **dest_events})

        # 7. Detection engine
        # (sensors would detect units; for city sim we skip active detection)

        # 8. Diplomacy (no events to process in peaceful city)
        # self.diplomacy.tick(dt)  — DiplomacyEngine has no tick, it is event-driven

        # 9. Telemetry
        self.telemetry.record_frame(
            fps=int(1.0 / max(dt, 0.001)),
            frame_time=dt * 1000,
            draw_calls=0,
            triangles=0,
            entity_count=len(self.civilians) + len(self.city_vehicles) + len(self.police_units),
            particle_count=0,
            projectile_count=0,
        )

        # Update counters
        self.tick_count += 1
        self.sim_time += dt

        return self.to_frame()

    def _tick_police(self, dt: float) -> None:
        """Simple police patrol behavior — wander around the city."""
        for unit in self.police_units:
            if not unit.is_alive():
                continue
            # Simple random walk patrol
            if unit.state.status == "idle" or self._rng.random() < 0.01:
                # Pick a new patrol target
                angle = self._rng.uniform(0, 2 * math.pi)
                dist = self._rng.uniform(5, 30)
                tx = unit.position[0] + math.cos(angle) * dist
                ty = unit.position[1] + math.sin(angle) * dist
                tx = max(10, min(self.width - 10, tx))
                ty = max(10, min(self.height - 10, ty))
                unit._patrol_target = (tx, ty)  # type: ignore[attr-defined]
                unit.state.status = "moving"

            target = getattr(unit, '_patrol_target', unit.position)
            dx = target[0] - unit.position[0]
            dy = target[1] - unit.position[1]
            d = math.hypot(dx, dy)
            if d < 2.0:
                unit.state.status = "idle"
            elif d > 0.01:
                nx, ny = dx / d, dy / d
                speed = unit.stats.speed * dt
                unit.position = (
                    unit.position[0] + nx * speed,
                    unit.position[1] + ny * speed,
                )
                unit.heading = math.atan2(ny, nx)

    # ------------------------------------------------------------------
    # Frame output
    # ------------------------------------------------------------------

    def to_frame(self) -> dict[str, Any]:
        """Produce a Three.js-compatible frame dict from current state.

        This is the data that gets sent over WebSocket to city3d.html.
        """
        env_snap = self.environment.snapshot()

        # Civilians
        civilians_data = [civ.to_dict() for civ in self.civilians]

        # Vehicles
        vehicles_data = [veh.to_dict() for veh in self.city_vehicles]

        # Police
        police_data = []
        for unit in self.police_units:
            police_data.append({
                "id": unit.unit_id,
                "type": "police",
                "x": round(unit.position[0], 2),
                "y": round(unit.position[1], 2),
                "heading": round(unit.heading, 3),
                "health": unit.state.health / unit.stats.max_health,
                "status": unit.state.status,
                "alliance": "friendly",
            })

        # Crowd / protestors
        crowd_data: dict[str, Any] = {}
        if self.crowd is not None:
            crowd_data = self.crowd.to_three_js()

        # Buildings
        buildings_data = []
        for b in self.buildings:
            buildings_data.append({
                "id": b.structure_id,
                "type": "building",
                "x": round(b.position[0], 2),
                "y": round(b.position[1], 2),
                "width": b.size[0],
                "depth": b.size[1],
                "height": b.size[2],
                "material": b.material,
                "health": b.health / b.max_health if b.max_health > 0 else 1.0,
            })

        # Trees
        trees_data = []
        for t in self.trees:
            trees_data.append({
                "id": t.entity_id,
                "type": "tree",
                "x": round(t.position[0], 2),
                "y": round(t.position[1], 2),
                "height": t.size[2],
                "tree_type": t.properties.get("tree_type", "oak"),
            })

        # Roads from map
        roads_data = []
        for road in self.roads:
            roads_data.append([{"x": round(p[0], 2), "y": round(p[1], 2)} for p in road])

        # Destruction layer
        destruction_data = self.destruction.to_three_js()

        # Influence map (territory control)
        influence_data = {
            "factions": list(self.diplomacy.factions.keys()),
        }

        frame: dict[str, Any] = {
            "type": "city_frame",
            "tick": self.tick_count,
            "sim_time": round(self.sim_time, 3),
            "map": {
                "width": self.width,
                "height": self.height,
            },
            "environment": {
                "hour": env_snap.get("hour", 10.0),
                "is_day": env_snap.get("is_day", True),
                "is_night": env_snap.get("is_night", False),
                "light_level": env_snap.get("light_level", 1.0),
                "sun_angle": env_snap.get("sun_angle", 45.0),
                "weather": env_snap.get("weather", "clear"),
                "wind_speed": env_snap.get("wind_speed", 0.0),
                "wind_direction": env_snap.get("wind_direction", 0.0),
                "temperature": env_snap.get("temperature", 20.0),
                "visibility": env_snap.get("visibility", 1.0),
            },
            "civilians": civilians_data,
            "vehicles": vehicles_data,
            "police": police_data,
            "crowd": crowd_data,
            "buildings": buildings_data,
            "trees": trees_data,
            "roads": roads_data,
            "destruction": destruction_data,
            "influence": influence_data,
            "events": list(self.events),
            "stats": self.stats(),
        }
        return frame

    def stats(self) -> dict[str, Any]:
        """Quick simulation statistics."""
        walking_civs = sum(1 for c in self.civilians if c.state == "walking")
        idle_civs = sum(1 for c in self.civilians if c.state == "idle")
        at_poi_civs = sum(1 for c in self.civilians if c.state == "at_poi")
        moving_vehicles = sum(1 for v in self.city_vehicles if v.speed > 0.5)
        crowd_count = len(self.crowd.members) if self.crowd else 0
        crowd_mood = self.crowd.overall_mood.name.lower() if self.crowd else "none"

        # Role distribution
        role_counts: dict[str, int] = {}
        for c in self.civilians:
            role_counts[c.role] = role_counts.get(c.role, 0) + 1

        # Activity distribution
        activity_counts: dict[str, int] = {}
        for c in self.civilians:
            activity_counts[c.activity] = activity_counts.get(c.activity, 0) + 1

        return {
            "tick": self.tick_count,
            "sim_time": round(self.sim_time, 2),
            "total_civilians": len(self.civilians),
            "walking_civilians": walking_civs,
            "idle_civilians": idle_civs,
            "at_poi_civilians": at_poi_civs,
            "total_vehicles": len(self.city_vehicles),
            "moving_vehicles": moving_vehicles,
            "total_police": len(self.police_units),
            "crowd_count": crowd_count,
            "crowd_mood": crowd_mood,
            "total_buildings": len(self.buildings),
            "total_trees": len(self.trees),
            "environment": self.environment.describe(),
            "civilian_roles": role_counts,
            "civilian_activities": activity_counts,
        }

    def inject_crowd_event(self, event_type: str, position: Vec2,
                           radius: float = 20.0, intensity: float = 0.5) -> None:
        """Inject a crowd event (e.g. 'gunshot', 'speech', 'chant')."""
        if self.crowd is None:
            return
        event = CrowdEvent(
            event_type=event_type,
            position=position,
            radius=radius,
            intensity=intensity,
            timestamp=self.sim_time,
        )
        self.crowd.inject_event(event)
        self.events.append({
            "type": "crowd_event",
            "event_type": event_type,
            "position": position,
            "radius": radius,
            "intensity": intensity,
        })

    def set_weather(self, weather: Weather) -> None:
        """Force a weather change."""
        self.environment.weather.state.current = weather

    def set_time(self, hour: float) -> None:
        """Set the time of day."""
        self.environment.time.hour = hour % 24.0
