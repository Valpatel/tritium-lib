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
# CivilianAgent — a walking civilian with simple waypoint behavior
# ---------------------------------------------------------------------------

@dataclass
class CivilianAgent:
    """A civilian walking on sidewalks with simple waypoint navigation."""

    agent_id: str
    position: Vec2
    target: Vec2
    speed: float = 1.2   # m/s walking speed
    heading: float = 0.0
    state: str = "walking"  # walking, idle, waiting
    waypoints: list[Vec2] = field(default_factory=list)
    _waypoint_idx: int = 0
    _idle_timer: float = 0.0

    def tick(self, dt: float, rng: random.Random, bounds: tuple[float, float]) -> None:
        """Advance the civilian by dt seconds."""
        if self.state == "idle":
            self._idle_timer -= dt
            if self._idle_timer <= 0:
                self.state = "walking"
                # Pick a new random target
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
                # Go idle for a bit, then pick new destination
                self.state = "idle"
                self._idle_timer = rng.uniform(2.0, 8.0)
                # Prepare new waypoints for when we resume
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
        """Spawn civilians walking on sidewalks."""
        for _ in range(count):
            pos = (
                self._rng.uniform(20, self.width - 20),
                self._rng.uniform(20, self.height - 20),
            )
            target = (
                self._rng.uniform(20, self.width - 20),
                self._rng.uniform(20, self.height - 20),
            )
            agent = CivilianAgent(
                agent_id=self._gen_id("civ"),
                position=pos,
                target=target,
                speed=self._rng.uniform(0.8, 1.5),
            )
            self.civilians.append(agent)

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

        # 2. Civilians walk
        for civ in self.civilians:
            civ.tick(dt, self._rng, (self.width, self.height))

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
        moving_vehicles = sum(1 for v in self.city_vehicles if v.speed > 0.5)
        crowd_count = len(self.crowd.members) if self.crowd else 0
        crowd_mood = self.crowd.overall_mood.name.lower() if self.crowd else "none"

        return {
            "tick": self.tick_count,
            "sim_time": round(self.sim_time, 2),
            "total_civilians": len(self.civilians),
            "walking_civilians": walking_civs,
            "idle_civilians": idle_civs,
            "total_vehicles": len(self.city_vehicles),
            "moving_vehicles": moving_vehicles,
            "total_police": len(self.police_units),
            "crowd_count": crowd_count,
            "crowd_mood": crowd_mood,
            "total_buildings": len(self.buildings),
            "total_trees": len(self.trees),
            "environment": self.environment.describe(),
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
