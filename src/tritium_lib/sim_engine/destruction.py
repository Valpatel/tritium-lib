# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Destruction and fire propagation system for the Tritium sim engine.

Simulates building damage, fire spread, debris physics, and environmental
destruction.  All state is serialisable to Three.js-ready frame data via
``DestructionEngine.to_three_js()``.
"""

from __future__ import annotations

import enum
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Optional

from tritium_lib.sim_engine.ai.steering import Vec2, distance

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GRAVITY = 9.81  # m/s^2
_DEBRIS_LIFETIME = 3.0  # seconds before debris settles
_FIRE_IGNITE_RANGE = 5.0  # meters — fire can ignite structures within this
_FIRE_SPREAD_INTERVAL = 1.0  # how often fires try to spread (seconds)

# ---------------------------------------------------------------------------
# Material properties
# ---------------------------------------------------------------------------

MATERIAL_PROPERTIES: dict[str, dict] = {
    "concrete": {
        "health": 200.0,
        "fire_resistance": 0.9,
        "debris_size": "large",
        "color": "#888888",
        "burn_rate": 0.5,   # damage per second from fire (after resistance)
    },
    "wood": {
        "health": 80.0,
        "fire_resistance": 0.1,
        "debris_size": "medium",
        "color": "#8B4513",
        "burn_rate": 5.0,
    },
    "metal": {
        "health": 150.0,
        "fire_resistance": 0.7,
        "debris_size": "small",
        "color": "#C0C0C0",
        "burn_rate": 1.5,
    },
    "brick": {
        "health": 120.0,
        "fire_resistance": 0.6,
        "debris_size": "medium",
        "color": "#B22222",
        "burn_rate": 2.0,
    },
    "glass": {
        "health": 20.0,
        "fire_resistance": 0.8,
        "debris_size": "tiny",
        "color": "#88ccff",
        "burn_rate": 0.0,
    },
}

_DEBRIS_SIZE_MAP = {"tiny": 0.1, "small": 0.3, "medium": 0.6, "large": 1.0}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StructureType(enum.Enum):
    BUILDING = "building"
    WALL = "wall"
    BARRIER = "barrier"
    VEHICLE_WRECK = "vehicle_wreck"
    BRIDGE = "bridge"
    TOWER = "tower"
    FENCE = "fence"


class DamageLevel(enum.Enum):
    INTACT = "intact"
    LIGHT_DAMAGE = "light_damage"
    HEAVY_DAMAGE = "heavy_damage"
    CRITICAL = "critical"
    DESTROYED = "destroyed"
    COLLAPSED = "collapsed"


_DAMAGE_THRESHOLDS: list[tuple[float, DamageLevel]] = [
    (0.0, DamageLevel.COLLAPSED),
    (0.05, DamageLevel.DESTROYED),
    (0.25, DamageLevel.CRITICAL),
    (0.50, DamageLevel.HEAVY_DAMAGE),
    (0.75, DamageLevel.LIGHT_DAMAGE),
]


def _health_to_damage_level(health_pct: float) -> DamageLevel:
    """Map health percentage (0-1) to a DamageLevel."""
    for threshold, level in _DAMAGE_THRESHOLDS:
        if health_pct <= threshold:
            return level
    return DamageLevel.INTACT

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Structure:
    """A destructible structure in the simulation."""
    structure_id: str
    structure_type: StructureType
    position: Vec2
    size: tuple[float, float, float]  # width, depth, height (meters)
    rotation: float = 0.0  # radians
    health: float = 100.0
    max_health: float = 100.0
    damage_level: DamageLevel = DamageLevel.INTACT
    material: str = "concrete"
    is_on_fire: bool = False
    fire_intensity: float = 0.0
    provides_cover: bool = True
    debris: list[dict] = field(default_factory=list)
    holes: list[dict] = field(default_factory=list)


@dataclass
class Fire:
    """An active fire in the world."""
    fire_id: str
    position: Vec2
    radius: float
    intensity: float  # 0-1
    fuel_remaining: float  # seconds of burn time
    spread_rate: float = 0.5  # meters per second
    temperature: float = 800.0  # celsius
    smoke_height: float = 10.0  # meters


@dataclass
class Debris:
    """A physics-simulated debris chunk."""
    debris_id: str
    position: Vec2
    velocity: Vec2
    angular_velocity: float
    size: float
    material: str
    time_alive: float = 0.0
    is_active: bool = True
    z: float = 0.0  # height
    vz: float = 0.0  # vertical velocity


# ---------------------------------------------------------------------------
# DestructionEngine
# ---------------------------------------------------------------------------


class DestructionEngine:
    """Manages all destruction, fire, and debris in the simulation.

    Parameters
    ----------
    wind_direction:
        Angle in radians the wind blows *toward* (0 = east, pi/2 = north).
    wind_speed:
        Wind speed in m/s.  Affects fire spread direction and smoke drift.
    """

    def __init__(
        self,
        wind_direction: float = 0.0,
        wind_speed: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.wind_direction = wind_direction
        self.wind_speed = wind_speed
        self.structures: list[Structure] = []
        self.fires: list[Fire] = []
        self.debris_list: list[Debris] = []
        self._rng = rng or random.Random()
        self._structure_map: dict[str, Structure] = {}
        # Cache for the structures portion of to_three_js().
        # Structures geometry is static; only damage/health/fire fields
        # change, and only via damage_structure() or fire spread.
        self._structures_cache: list[dict] | None = None
        self._structures_version: int = 0
        self._structures_cache_version: int = -1

    # -- mutators -----------------------------------------------------------

    def add_structure(self, structure: Structure) -> None:
        """Register a structure with the engine."""
        self.structures.append(structure)
        self._structure_map[structure.structure_id] = structure
        self._structures_version += 1

    def damage_structure(
        self,
        structure_id: str,
        damage: float,
        damage_pos: Vec2,
        damage_type: str = "kinetic",
    ) -> dict:
        """Apply *damage* to a structure and return a Three.js event dict.

        Side-effects:
        - Health decreases, damage_level updates.
        - Debris spawned based on material type.
        - Wood at HEAVY_DAMAGE or worse has a chance to ignite.
        - DESTROYED or COLLAPSED → large debris field.
        """
        s = self._structure_map.get(structure_id)
        if s is None:
            return {"event": "damage", "structure_id": structure_id, "error": "not_found"}

        if damage <= 0:
            return {"event": "damage", "structure_id": structure_id, "damage": 0, "new_health": s.health}

        prev_level = s.damage_level

        s.health = max(0.0, s.health - damage)
        health_pct = s.health / s.max_health if s.max_health > 0 else 0.0
        s.damage_level = _health_to_damage_level(health_pct)
        self._structures_version += 1

        # Add a hole at the impact point
        hole_size = min(3.0, damage / 20.0)
        s.holes.append({"x": damage_pos[0], "y": damage_pos[1], "size": hole_size})

        # Generate debris
        new_debris = self._spawn_debris(s, damage_pos, damage)

        # Fire chance for wood at heavy damage or worse
        if (
            s.material == "wood"
            and s.damage_level in (DamageLevel.HEAVY_DAMAGE, DamageLevel.CRITICAL, DamageLevel.DESTROYED, DamageLevel.COLLAPSED)
            and not s.is_on_fire
        ):
            fire_chance = 0.4 if damage_type == "explosive" else 0.2
            if self._rng.random() < fire_chance:
                self._ignite_structure(s)

        # Large debris field on destruction
        if s.damage_level in (DamageLevel.DESTROYED, DamageLevel.COLLAPSED) and prev_level not in (DamageLevel.DESTROYED, DamageLevel.COLLAPSED):
            self._collapse_structure(s)

        # Cover lost at destroyed/collapsed
        if s.damage_level in (DamageLevel.DESTROYED, DamageLevel.COLLAPSED):
            s.provides_cover = False

        return {
            "event": "damage",
            "structure_id": structure_id,
            "damage": damage,
            "new_health": s.health,
            "health_pct": health_pct,
            "damage_level": s.damage_level.value,
            "prev_level": prev_level.value,
            "debris_spawned": len(new_debris),
            "on_fire": s.is_on_fire,
        }

    def start_fire(
        self,
        position: Vec2,
        radius: float = 2.0,
        intensity: float = 0.5,
        fuel: float = 60.0,
    ) -> Fire:
        """Ignite a new fire at *position*."""
        f = Fire(
            fire_id=f"fire_{uuid.uuid4().hex[:8]}",
            position=position,
            radius=max(0.1, radius),
            intensity=max(0.0, min(1.0, intensity)),
            fuel_remaining=max(0.0, fuel),
            temperature=400.0 + 600.0 * intensity,
            smoke_height=5.0 + 25.0 * intensity,
        )
        self.fires.append(f)
        return f

    # -- tick ---------------------------------------------------------------

    def tick(self, dt: float) -> dict:
        """Advance the destruction simulation by *dt* seconds.

        Returns a dict summarising all events that occurred this tick.
        """
        events: dict = {
            "fires_spread": [],
            "fires_died": [],
            "structures_ignited": [],
            "structures_damaged": [],
            "debris_settled": 0,
        }

        self._tick_fires(dt, events)
        self._tick_debris(dt, events)

        return events

    # -- Three.js export ----------------------------------------------------

    def to_three_js(self) -> dict:
        """Serialise the full destruction state to a Three.js-ready dict.

        The structures list is cached and only re-serialized when a
        structure is added, damaged, ignited, or extinguished.
        """
        # Rebuild structures cache only when the version has changed.
        if self._structures_cache_version != self._structures_version:
            structures_out: list[dict] = []
            for s in self.structures:
                destroyed = s.damage_level.value in ("destroyed", "collapsed")
                structures_out.append({
                    "id": s.structure_id,
                    "x": s.position[0],
                    "y": s.position[1],
                    # Full names used by the Three.js frontend (ensureBuildings)
                    "width": s.size[0],
                    "depth": s.size[1],
                    "height": s.size[2],
                    # Short aliases kept for backward compatibility
                    "w": s.size[0],
                    "d": s.size[1],
                    "h": s.size[2],
                    "rotation": s.rotation,
                    "material": s.material,
                    "damage": s.damage_level.value,
                    "destroyed": destroyed,
                    "health_pct": s.health / s.max_health if s.max_health > 0 else 0.0,
                    "on_fire": s.is_on_fire,
                    "holes": list(s.holes),
                })
            self._structures_cache = structures_out
            self._structures_cache_version = self._structures_version

        fires_out: list[dict] = []
        for f in self.fires:
            fires_out.append({
                "id": f.fire_id,
                "x": f.position[0],
                "y": f.position[1],
                "radius": f.radius,
                "intensity": f.intensity,
                "color": _fire_color(f.intensity),
                "smoke_height": f.smoke_height,
                "emitter": {
                    "rate": int(100 + 200 * f.intensity),
                    "lifetime": 1.0 + f.intensity,
                    "speed": 1.0 + 4.0 * f.intensity,
                    "colors": ["#ff4400", "#ff8800", "#ffcc00"],
                },
            })

        debris_out: list[dict] = []
        for d in self.debris_list:
            if d.is_active:
                debris_out.append({
                    "id": d.debris_id,
                    "x": d.position[0],
                    "y": d.position[1],
                    "z": d.z,
                    "vx": d.velocity[0],
                    "vy": d.velocity[1],
                    "vz": d.vz,
                    "size": d.size,
                    "material": d.material,
                    "rotation": d.angular_velocity * d.time_alive,
                })

        smoke_out: list[dict] = []
        for f in self.fires:
            if f.intensity > 0.05:
                wind_dx = math.cos(self.wind_direction) * self.wind_speed * 0.5
                wind_dy = math.sin(self.wind_direction) * self.wind_speed * 0.5
                smoke_out.append({
                    "x": f.position[0] + wind_dx,
                    "y": f.position[1] + wind_dy,
                    "radius": f.radius * 1.5 + self.wind_speed,
                    "height": f.smoke_height,
                    "opacity": min(0.8, 0.3 + 0.5 * f.intensity),
                    "color": "#333333",
                })

        return {
            "structures": self._structures_cache,
            "fires": fires_out,
            "debris": debris_out,
            "smoke": smoke_out,
        }

    # -- queries ------------------------------------------------------------

    def get_blocked_positions(self, cell_size: float = 1.0) -> set[tuple[int, int]]:
        """Return grid cells blocked by rubble from collapsed structures."""
        blocked: set[tuple[int, int]] = set()
        for s in self.structures:
            if s.damage_level not in (DamageLevel.DESTROYED, DamageLevel.COLLAPSED):
                continue
            # Rubble covers the structure footprint
            cx, cy = s.position
            hw = s.size[0] / 2.0
            hd = s.size[1] / 2.0
            x_min = int(math.floor((cx - hw) / cell_size))
            x_max = int(math.floor((cx + hw) / cell_size))
            y_min = int(math.floor((cy - hd) / cell_size))
            y_max = int(math.floor((cy + hd) / cell_size))
            for gx in range(x_min, x_max + 1):
                for gy in range(y_min, y_max + 1):
                    blocked.add((gx, gy))

        # Settled debris
        for d in self.debris_list:
            if not d.is_active and d.size >= 0.5:
                gx = int(math.floor(d.position[0] / cell_size))
                gy = int(math.floor(d.position[1] / cell_size))
                blocked.add((gx, gy))

        return blocked

    def get_los_blockers(self) -> list[dict]:
        """Return smoke columns that block line of sight."""
        blockers: list[dict] = []
        for f in self.fires:
            if f.intensity > 0.1 and f.smoke_height > 2.0:
                wind_dx = math.cos(self.wind_direction) * self.wind_speed * 0.5
                wind_dy = math.sin(self.wind_direction) * self.wind_speed * 0.5
                blockers.append({
                    "x": f.position[0] + wind_dx,
                    "y": f.position[1] + wind_dy,
                    "radius": f.radius * 1.5 + self.wind_speed * 0.5,
                    "height": f.smoke_height,
                    "opacity": min(0.8, 0.3 + 0.5 * f.intensity),
                })
        return blockers

    # -- internal helpers ---------------------------------------------------

    def _spawn_debris(self, s: Structure, impact_pos: Vec2, damage: float) -> list[Debris]:
        """Generate debris chunks from a damage event."""
        mat_props = MATERIAL_PROPERTIES.get(s.material, MATERIAL_PROPERTIES["concrete"])
        base_size = _DEBRIS_SIZE_MAP.get(mat_props["debris_size"], 0.5)
        count = max(1, int(damage / 15.0))
        new_debris: list[Debris] = []

        for _ in range(count):
            angle = self._rng.uniform(0, 2 * math.pi)
            speed = self._rng.uniform(2.0, 8.0) * (damage / 50.0)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            vz = self._rng.uniform(3.0, 10.0) * (damage / 50.0)
            size = base_size * self._rng.uniform(0.5, 1.5)

            d = Debris(
                debris_id=f"debris_{uuid.uuid4().hex[:8]}",
                position=impact_pos,
                velocity=(vx, vy),
                angular_velocity=self._rng.uniform(-5.0, 5.0),
                size=size,
                material=s.material,
                z=s.size[2] * self._rng.uniform(0.2, 0.8),
                vz=vz,
            )
            self.debris_list.append(d)
            new_debris.append(d)

        s.debris.extend([{"id": d.debris_id, "size": d.size} for d in new_debris])
        return new_debris

    def _collapse_structure(self, s: Structure) -> None:
        """Generate a large rubble field when a structure is destroyed."""
        count = max(5, int(s.size[0] * s.size[1] / 4.0))
        hw, hd = s.size[0] / 2.0, s.size[1] / 2.0

        for _ in range(count):
            ox = self._rng.uniform(-hw, hw)
            oy = self._rng.uniform(-hd, hd)
            pos = (s.position[0] + ox, s.position[1] + oy)
            angle = self._rng.uniform(0, 2 * math.pi)
            speed = self._rng.uniform(1.0, 4.0)

            d = Debris(
                debris_id=f"rubble_{uuid.uuid4().hex[:8]}",
                position=pos,
                velocity=(math.cos(angle) * speed, math.sin(angle) * speed),
                angular_velocity=self._rng.uniform(-3.0, 3.0),
                size=self._rng.uniform(0.5, 2.0),
                material=s.material,
                z=self._rng.uniform(0, s.size[2]),
                vz=self._rng.uniform(0.5, 3.0),
            )
            self.debris_list.append(d)

    def _ignite_structure(self, s: Structure) -> None:
        """Set a structure on fire."""
        s.is_on_fire = True
        s.fire_intensity = 0.5
        self._structures_version += 1
        f = Fire(
            fire_id=f"sfire_{s.structure_id}",
            position=s.position,
            radius=max(s.size[0], s.size[1]) / 2.0,
            intensity=0.5,
            fuel_remaining=60.0,
            temperature=600.0,
            smoke_height=s.size[2] * 2.0,
        )
        self.fires.append(f)

    def _tick_fires(self, dt: float, events: dict) -> None:
        """Update all fires: spread, damage, decay."""
        dead_fires: list[Fire] = []

        for f in self.fires:
            if f.fuel_remaining <= 0 or f.intensity <= 0:
                dead_fires.append(f)
                continue

            # Consume fuel
            f.fuel_remaining -= dt * f.intensity
            if f.fuel_remaining <= 0:
                f.fuel_remaining = 0.0
                f.intensity = 0.0
                dead_fires.append(f)
                events["fires_died"].append(f.fire_id)
                continue

            # Decay intensity as fuel runs low
            if f.fuel_remaining < 10.0:
                f.intensity *= (1.0 - dt * 0.1)

            # Spread: grow radius toward wind direction
            if self.wind_speed > 0:
                dx = math.cos(self.wind_direction) * self.wind_speed * 0.02 * dt
                dy = math.sin(self.wind_direction) * self.wind_speed * 0.02 * dt
                f.position = (f.position[0] + dx, f.position[1] + dy)

            f.radius += f.spread_rate * dt * 0.1 * f.intensity

            # Update temperature and smoke
            f.temperature = 400.0 + 600.0 * f.intensity
            f.smoke_height = 5.0 + 25.0 * f.intensity

            # Damage nearby structures
            for s in self.structures:
                dist = distance(f.position, s.position)
                if dist > f.radius + max(s.size[0], s.size[1]) / 2.0:
                    continue

                mat_props = MATERIAL_PROPERTIES.get(s.material, MATERIAL_PROPERTIES["concrete"])
                fire_res = mat_props["fire_resistance"]
                burn_rate = mat_props["burn_rate"]
                fire_dmg = burn_rate * f.intensity * (1.0 - fire_res) * dt

                if fire_dmg > 0 and s.health > 0:
                    s.health = max(0.0, s.health - fire_dmg)
                    s.damage_level = _health_to_damage_level(s.health / s.max_health if s.max_health > 0 else 0.0)
                    self._structures_version += 1
                    events["structures_damaged"].append(s.structure_id)

                # Ignite nearby wood structures
                if (
                    s.material == "wood"
                    and not s.is_on_fire
                    and dist < _FIRE_IGNITE_RANGE
                    and f.intensity > 0.3
                ):
                    if self._rng.random() < 0.1 * dt * f.intensity:
                        self._ignite_structure(s)
                        events["structures_ignited"].append(s.structure_id)

        # Remove dead fires
        for f in dead_fires:
            if f in self.fires:
                self.fires.remove(f)
                # Update structure fire status
                for s in self.structures:
                    if f.fire_id == f"sfire_{s.structure_id}":
                        s.is_on_fire = False
                        s.fire_intensity = 0.0
                        self._structures_version += 1

    def _tick_debris(self, dt: float, events: dict) -> None:
        """Update debris physics: ballistic trajectory, settle after timeout."""
        for d in self.debris_list:
            if not d.is_active:
                continue

            d.time_alive += dt

            # Ballistic trajectory
            d.position = (
                d.position[0] + d.velocity[0] * dt,
                d.position[1] + d.velocity[1] * dt,
            )
            d.z += d.vz * dt
            d.vz -= _GRAVITY * dt

            # Ground collision
            if d.z <= 0:
                d.z = 0.0
                d.vz = 0.0
                # Dampen horizontal velocity on bounce
                d.velocity = (d.velocity[0] * 0.3, d.velocity[1] * 0.3)

            # Settle after lifetime
            if d.time_alive >= _DEBRIS_LIFETIME:
                d.is_active = False
                d.velocity = (0.0, 0.0)
                d.vz = 0.0
                events["debris_settled"] += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fire_color(intensity: float) -> str:
    """Return a hex color for fire based on intensity (0-1)."""
    # Low intensity: orange-red, high intensity: bright yellow-white
    if intensity < 0.3:
        return "#ff4400"
    elif intensity < 0.6:
        return "#ff6600"
    elif intensity < 0.8:
        return "#ff8800"
    else:
        return "#ffcc00"
