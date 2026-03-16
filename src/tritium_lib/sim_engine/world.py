# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""World simulation integrator — the top-level sim engine.

Owns every subsystem (units, vehicles, terrain, weather, crowds,
destruction, projectiles, effects) and advances them all with a
single ``tick()`` call.  WorldBuilder provides a fluent API for
quick scenario setup, and WORLD_PRESETS offers ready-made scenarios.

Usage::

    from tritium_lib.sim_engine.world import World, WorldConfig, WorldBuilder, WORLD_PRESETS

    world = WORLD_PRESETS["urban_combat"]()
    for _ in range(100):
        frame = world.tick()
    print(world.stats())
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any

from tritium_lib.sim_engine.ai.steering import (
    Vec2,
    distance,
    normalize,
    _sub,
    _add,
    _scale,
    seek,
)
from tritium_lib.sim_engine.units import (
    Unit,
    Alliance,
    UnitType,
    UNIT_TEMPLATES,
    create_unit,
)
from tritium_lib.sim_engine.damage import (
    DamageType,
    DamageTracker,
    HitResult,
    resolve_attack,
    resolve_explosion,
)
from tritium_lib.sim_engine.arsenal import (
    ARSENAL,
    Weapon,
    Projectile,
    ProjectileSimulator,
    AreaEffect,
    AreaEffectManager,
    create_explosion_effect,
    create_smoke_effect,
    create_fire_effect,
)
from tritium_lib.sim_engine.vehicles import (
    VehicleState,
    VehiclePhysicsEngine,
    DroneController,
    ConvoySimulator,
    VEHICLE_TEMPLATES,
    create_vehicle,
)
from tritium_lib.sim_engine.terrain import HeightMap, LineOfSight
from tritium_lib.sim_engine.environment import (
    Environment,
    TimeOfDay,
    Weather,
    WeatherSimulator,
    WeatherEffects,
)
from tritium_lib.sim_engine.crowd import (
    CrowdSimulator,
    CrowdMood,
    CrowdEvent,
)
from tritium_lib.sim_engine.destruction import (
    DestructionEngine,
    Structure,
    StructureType,
    MATERIAL_PROPERTIES,
)
from tritium_lib.sim_engine.renderer import SimRenderer
from tritium_lib.sim_engine.ai.squad import Squad, SquadRole, SquadTactics, Order
from tritium_lib.sim_engine.ai.tactics import TacticsEngine, TacticalAction


# ---------------------------------------------------------------------------
# WorldConfig
# ---------------------------------------------------------------------------


@dataclass
class WorldConfig:
    """Configuration for a World instance."""

    map_size: tuple[float, float] = (500.0, 500.0)
    tick_rate: float = 20.0  # ticks/sec
    enable_weather: bool = True
    enable_destruction: bool = True
    enable_crowds: bool = False
    enable_vehicles: bool = True
    enable_los: bool = True
    gravity: float = 9.81
    seed: int | None = None


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


class World:
    """Top-level simulation world.  One ``tick()`` advances everything."""

    def __init__(self, config: WorldConfig | None = None) -> None:
        self.config = config or WorldConfig()
        self._rng = random.Random(self.config.seed)
        self._next_id = 0

        # --- sub-systems ---
        map_w, map_h = self.config.map_size
        grid_w = max(1, int(map_w))
        grid_h = max(1, int(map_h))

        self.heightmap = HeightMap(grid_w, grid_h, cell_size=1.0)
        self.los = LineOfSight(self.heightmap) if self.config.enable_los else None

        self.environment = Environment(
            time=TimeOfDay(12.0),
            weather=WeatherSimulator(seed=self._rng.randint(0, 2**31)),
        )

        self.destruction = (
            DestructionEngine(rng=random.Random(self._rng.randint(0, 2**31)))
            if self.config.enable_destruction
            else None
        )

        self.crowd: CrowdSimulator | None = None
        if self.config.enable_crowds:
            self.crowd = CrowdSimulator(
                bounds=(0.0, 0.0, map_w, map_h), max_members=500,
            )

        self.projectile_sim = ProjectileSimulator(
            gravity=self.config.gravity,
        )
        self.area_effects = AreaEffectManager()

        self.renderer = SimRenderer()

        self.tactics_engine = TacticsEngine()

        # --- entity containers ---
        self.units: dict[str, Unit] = {}
        self.squads: dict[str, Squad] = {}
        self.vehicles: dict[str, VehicleState] = {}
        self.drone_controllers: dict[str, DroneController] = {}

        # --- bookkeeping ---
        self.tick_count: int = 0
        self.sim_time: float = 0.0
        self.events: list[dict[str, Any]] = []
        self.damage_tracker = DamageTracker()

    # -- ID generation -------------------------------------------------------

    def _gen_id(self, prefix: str = "e") -> str:
        self._next_id += 1
        return f"{prefix}_{self._next_id}"

    # -- Spawn methods -------------------------------------------------------

    def spawn_unit(
        self,
        template: str,
        name: str,
        alliance: str | Alliance,
        position: Vec2,
        squad_id: str | None = None,
    ) -> Unit:
        """Create a unit from a template and add it to the world."""
        uid = self._gen_id("u")
        if isinstance(alliance, str):
            alliance = Alliance(alliance)
        unit = create_unit(template, uid, name, alliance, position)
        # Assign weapon from template if available
        if template in ("sniper",):
            unit.weapon = "m24"
        elif template in ("heavy",):
            unit.weapon = "m249_saw"
        elif template in ("infantry",):
            unit.weapon = "m4a1"
        elif template in ("scout",):
            unit.weapon = "mp5"
        else:
            unit.weapon = "m4a1"
        if squad_id is not None:
            unit.squad_id = squad_id
            if squad_id in self.squads:
                self.squads[squad_id].add_member(uid)
        self.units[uid] = unit
        self.events.append({"type": "spawn_unit", "id": uid, "name": name})
        return unit

    def spawn_vehicle(
        self,
        template: str,
        name: str,
        alliance: str,
        position: Vec2,
    ) -> VehicleState:
        """Create a vehicle from a template and add it to the world."""
        vid = self._gen_id("v")
        vehicle = create_vehicle(template, vid, alliance, position, name=name)
        self.vehicles[vid] = vehicle
        self.events.append({"type": "spawn_vehicle", "id": vid, "name": name})
        return vehicle

    def spawn_squad(
        self,
        name: str,
        alliance: str,
        unit_templates: list[str],
        positions: list[Vec2],
    ) -> Squad:
        """Create a squad with units placed at the given positions."""
        sid = self._gen_id("sq")
        squad = Squad(sid, name, alliance)
        self.squads[sid] = squad

        for i, tmpl in enumerate(unit_templates):
            pos = positions[i] if i < len(positions) else positions[-1]
            uname = f"{name}_{i}"
            unit = self.spawn_unit(tmpl, uname, alliance, pos, squad_id=sid)
            role = SquadRole.LEADER if i == 0 else SquadRole.RIFLEMAN
            squad.add_member(unit.unit_id, role)

        self.events.append({"type": "spawn_squad", "id": sid, "name": name, "size": len(unit_templates)})
        return squad

    def spawn_crowd(
        self,
        center: Vec2,
        count: int,
        radius: float,
        mood: CrowdMood = CrowdMood.CALM,
    ) -> list[str]:
        """Spawn crowd members.  Enables the crowd subsystem if needed."""
        if self.crowd is None:
            map_w, map_h = self.config.map_size
            self.crowd = CrowdSimulator(bounds=(0.0, 0.0, map_w, map_h), max_members=500)
            self.config.enable_crowds = True
        ids = self.crowd.spawn_crowd(center, count, radius, mood=mood)
        self.events.append({"type": "spawn_crowd", "center": center, "count": len(ids)})
        return ids

    def add_structure(self, structure: Structure) -> None:
        """Add a building/structure to the destruction engine."""
        if self.destruction is None:
            self.destruction = DestructionEngine(rng=random.Random(self._rng.randint(0, 2**31)))
            self.config.enable_destruction = True
        self.destruction.add_structure(structure)
        self.events.append({"type": "add_structure", "id": structure.structure_id})

    # -- Actions -------------------------------------------------------------

    def fire_weapon(self, unit_id: str, target_pos: Vec2) -> Projectile | None:
        """Fire the unit's weapon toward target_pos, creating a projectile."""
        unit = self.units.get(unit_id)
        if unit is None or not unit.is_alive():
            return None
        if not unit.can_attack(self.sim_time):
            return None

        weapon_id = unit.weapon
        weapon = ARSENAL.get(weapon_id)
        if weapon is None:
            # Fallback
            weapon = ARSENAL.get("m4a1")
        if weapon is None:
            return None

        # Apply environmental accuracy modifier
        acc_mod = self.environment.accuracy_modifier()
        proj = self.projectile_sim.fire(
            weapon, unit.position, target_pos,
            accuracy_modifier=1.0 / max(0.1, acc_mod),
            rng=random.Random(self._rng.randint(0, 2**31)),
        )

        unit.state.last_attack_time = self.sim_time
        unit.state.status = "attacking"
        if unit.state.ammo > 0:
            unit.state.ammo -= 1

        self.events.append({
            "type": "fire",
            "unit_id": unit_id,
            "weapon": weapon_id,
            "target": target_pos,
        })
        return proj

    # -- Tick ----------------------------------------------------------------

    def tick(self, dt: float | None = None) -> dict[str, Any]:
        """Advance the entire simulation by one step.

        Returns a rendered frame dict.
        """
        if dt is None:
            dt = 1.0 / self.config.tick_rate
        self.events = []

        # 1. Advance environment (weather, time)
        if self.config.enable_weather:
            self.environment.update(dt)

        # 2. Squad AI decisions
        self._tick_squads(dt)

        # 3-4. Unit AI: threat assessment, action selection, execution
        self._tick_units(dt)

        # 5. Advance projectiles
        impacts = self.projectile_sim.tick(dt)

        # 6. Resolve impacts -> damage
        self._resolve_impacts(impacts)

        # 7. Update destruction (fire spread, debris)
        if self.destruction is not None:
            dest_events = self.destruction.tick(dt)
            if dest_events.get("fires_spread") or dest_events.get("structures_damaged"):
                self.events.append({"type": "destruction_tick", **dest_events})

        # 8. Advance crowds
        if self.crowd is not None:
            self.crowd.tick(dt)

        # 9. Advance vehicles/drones
        if self.config.enable_vehicles:
            self._tick_vehicles(dt)

        # 10. Area effects tick
        expired = self.area_effects.tick(dt)
        if expired:
            self.events.append({"type": "effects_expired", "count": len(expired)})

        # 11. Apply weather modifiers (already integrated into fire_weapon accuracy)

        # Update counters
        self.tick_count += 1
        self.sim_time += dt

        # 12. Collect events (already collected inline)

        # 13. Return rendered frame
        return self.render()

    # -- Sub-tick methods ----------------------------------------------------

    def _tick_squads(self, dt: float) -> None:
        """Update squad state: cohesion, morale, orders."""
        unit_positions: dict[str, Vec2] = {
            uid: u.position for uid, u in self.units.items() if u.is_alive()
        }
        unit_morales: dict[str, float] = {
            uid: u.state.morale for uid, u in self.units.items() if u.is_alive()
        }

        for sid, squad in self.squads.items():
            living = [m for m in squad.members if m in unit_positions]
            if not living:
                continue
            squad.update_cohesion(unit_positions)
            squad.update_morale(unit_morales)

            # Auto-recommend orders if none set
            if squad.current_order is None or squad.should_retreat():
                # Gather threats
                threats: list[tuple[Vec2, str]] = []
                for uid, u in self.units.items():
                    if u.is_alive() and u.alliance.value != squad.alliance:
                        threats.append((u.position, uid))
                order = SquadTactics.recommend_order(squad, unit_positions, threats)
                squad.issue_order(order)

    def _tick_units(self, dt: float) -> None:
        """Per-unit AI: find targets, decide action, execute."""
        alive_units = {uid: u for uid, u in self.units.items() if u.is_alive()}

        for uid, unit in alive_units.items():
            # Recover suppression
            unit.recover_suppression(dt)

            # Find enemies
            enemies: list[dict] = []
            for oid, other in alive_units.items():
                if oid == uid:
                    continue
                if other.alliance == unit.alliance:
                    continue
                d = distance(unit.position, other.position)
                # Check detection range modified by environment
                det_range = unit.stats.detection_range * self.environment.detection_range_modifier()
                if d > det_range:
                    continue
                # LOS check
                if self.los is not None:
                    if not self.los.can_see(unit.position, other.position):
                        continue
                enemies.append({
                    "id": oid,
                    "pos": other.position,
                    "damage": other.stats.attack_damage,
                    "health": other.state.health / other.stats.max_health,
                    "facing": other.heading,
                })

            if not enemies:
                # No targets visible — follow squad order to advance toward enemy
                if unit.state.status not in ("dead",):
                    # Find this unit's squad and follow its order
                    moved = False
                    for _sid, squad in self.squads.items():
                        if uid in squad.members and squad.current_order is not None:
                            order = squad.current_order
                            if order.order_type in ("advance", "flank_left", "flank_right") and order.target_pos is not None:
                                direction = _sub(order.target_pos, unit.position)
                                d = distance(unit.position, order.target_pos)
                                if d > 5.0:  # don't jitter at destination
                                    direction = normalize(direction)
                                    move_speed = unit.effective_speed() * self.environment.movement_speed_modifier()
                                    dx = direction[0] * move_speed * dt
                                    dy = direction[1] * move_speed * dt
                                    unit.position = (unit.position[0] + dx, unit.position[1] + dy)
                                    unit.heading = math.atan2(direction[1], direction[0])
                                    unit.state.status = "moving"
                                    moved = True
                            break
                    if not moved:
                        unit.state.status = "idle"
                continue

            # Simple combat: engage nearest visible enemy
            nearest = min(enemies, key=lambda e: distance(unit.position, e["pos"]))
            nearest_dist = distance(unit.position, nearest["pos"])

            # Move toward if out of attack range
            if nearest_dist > unit.stats.attack_range:
                direction = _sub(nearest["pos"], unit.position)
                direction = normalize(direction)
                move_speed = unit.effective_speed() * self.environment.movement_speed_modifier()
                dx = direction[0] * move_speed * dt
                dy = direction[1] * move_speed * dt
                unit.position = (unit.position[0] + dx, unit.position[1] + dy)
                unit.heading = math.atan2(direction[1], direction[0])
                unit.state.status = "moving"
            else:
                # In range: fire
                if unit.can_attack(self.sim_time):
                    self.fire_weapon(uid, nearest["pos"])

    def _resolve_impacts(self, impacts: list[dict]) -> None:
        """Resolve projectile impacts against units and structures."""
        for impact in impacts:
            ix = impact.get("x", 0.0)
            iy = impact.get("y", 0.0)
            impact_pos: Vec2 = (ix, iy)
            damage = impact.get("damage", 0.0)
            proj_type = impact.get("type", "bullet")
            weapon_id = impact.get("weapon_id", "")

            # Check if any unit is near the impact
            hit_radius = 2.0  # bullets need to be close
            if proj_type in ("grenade", "rocket", "shell"):
                # Explosive: area damage
                targets = [
                    (u.position, uid)
                    for uid, u in self.units.items()
                    if u.is_alive()
                ]
                explosion_results = resolve_explosion(
                    impact_pos, radius=10.0, targets=targets,
                    base_damage=damage,
                )
                for result in explosion_results:
                    target_unit = self.units.get(result.target_id)
                    if target_unit and target_unit.is_alive():
                        actual = target_unit.take_damage(result.damage)
                        target_unit.apply_suppression(result.suppression_caused)
                        self.damage_tracker.record(result)
                        if not target_unit.is_alive():
                            self.events.append({
                                "type": "unit_killed",
                                "target_id": result.target_id,
                                "source_id": result.source_id,
                            })
                            self._handle_unit_death(result.target_id)

                # Damage structures
                if self.destruction is not None:
                    for s in self.destruction.structures:
                        if distance(impact_pos, s.position) < 15.0:
                            self.destruction.damage_structure(
                                s.structure_id, damage * 0.5, impact_pos,
                                damage_type="explosive",
                            )

                # Create explosion effect
                self.area_effects.add(create_explosion_effect(impact_pos, radius=8.0, duration=0.5))
            else:
                # Bullet: direct hit on nearest unit within hit_radius
                for uid, u in self.units.items():
                    if not u.is_alive():
                        continue
                    d = distance(impact_pos, u.position)
                    if d <= hit_radius:
                        actual = u.take_damage(damage)
                        u.apply_suppression(0.15)
                        hit_result = HitResult(
                            hit=True, damage=actual,
                            damage_type=DamageType.KINETIC,
                            target_id=uid,
                            source_id=weapon_id,
                            range_m=d,
                        )
                        self.damage_tracker.record(hit_result)
                        if not u.is_alive():
                            self.events.append({
                                "type": "unit_killed",
                                "target_id": uid,
                            })
                            self._handle_unit_death(uid)
                        break  # Only hit one unit per bullet

    def _handle_unit_death(self, unit_id: str) -> None:
        """Remove dead unit from squad and update bookkeeping."""
        unit = self.units.get(unit_id)
        if unit is None:
            return
        if unit.squad_id and unit.squad_id in self.squads:
            self.squads[unit.squad_id].remove_member(unit_id)

    def _tick_vehicles(self, dt: float) -> None:
        """Update vehicle positions via physics and drone controllers."""
        physics = VehiclePhysicsEngine()

        for vid, vehicle in self.vehicles.items():
            if vehicle.is_destroyed:
                continue

            # Drone controller
            if vid in self.drone_controllers:
                ctrl = self.drone_controllers[vid]
                throttle, steering, alt = ctrl.tick(dt)
                physics.update(vehicle, throttle, steering, dt, altitude_input=alt)
            else:
                # Non-drone vehicles: simple forward movement if they have speed
                # (ConvoySimulator or other higher-level AI would drive these)
                if abs(vehicle.speed) > 0.01:
                    physics.update(vehicle, 0.5, 0.0, dt)

    # -- Output methods ------------------------------------------------------

    def render(self) -> dict[str, Any]:
        """Produce a Three.js-ready frame from the current world state."""
        # Build sim_state dict for the renderer
        units_data: list[dict] = []
        for uid, u in self.units.items():
            units_data.append({
                "id": uid,
                "x": u.position[0],
                "y": u.position[1],
                "type": u.unit_type.value,
                "alliance": u.alliance.value,
                "health": u.state.health,
                "max_health": u.stats.max_health,
                "heading": u.heading,
                "status": u.state.status,
                "label": u.name,
            })

        proj_data = self.projectile_sim.to_three_js()
        effect_data = self.area_effects.to_three_js()

        env_snap = self.environment.snapshot()

        crowd_data: list[dict] = []
        if self.crowd is not None:
            crowd_js = self.crowd.to_three_js()
            crowd_data = crowd_js.get("members", [])

        sim_state: dict[str, Any] = {
            "tick": self.tick_count,
            "time": self.sim_time,
            "units": units_data,
            "projectiles": proj_data.get("projectiles", []),
            "effects": effect_data.get("effects", []),
            "weather": {
                "wind_speed": env_snap.get("wind_speed", 0.0),
                "wind_direction": env_snap.get("wind_direction", 0.0),
                "rain": 1.0 if env_snap.get("weather") in ("rain", "heavy_rain", "storm") else 0.0,
            },
            "time_of_day": {"hour": env_snap.get("hour", 12.0)},
            "crowd": crowd_data,
        }

        # Destruction layer
        if self.destruction is not None:
            dest_data = self.destruction.to_three_js()
            sim_state["destruction"] = dest_data

        frame = self.renderer.render_frame(sim_state)
        frame["events"] = list(self.events)
        frame["vehicles"] = [
            {
                "id": v.vehicle_id,
                "name": v.name,
                "x": v.position[0],
                "y": v.position[1],
                "z": v.altitude,
                "heading": v.heading,
                "speed": v.speed,
                "vehicle_class": v.vehicle_class.value,
                "alliance": v.alliance,
                "health": v.health / v.max_health if v.max_health > 0 else 0.0,
                "destroyed": v.is_destroyed,
            }
            for v in self.vehicles.values()
        ]
        return frame

    def snapshot(self) -> dict[str, Any]:
        """Full serializable state for save/restore."""
        return {
            "tick_count": self.tick_count,
            "sim_time": self.sim_time,
            "config": {
                "map_size": self.config.map_size,
                "tick_rate": self.config.tick_rate,
                "enable_weather": self.config.enable_weather,
                "enable_destruction": self.config.enable_destruction,
                "enable_crowds": self.config.enable_crowds,
                "enable_vehicles": self.config.enable_vehicles,
                "enable_los": self.config.enable_los,
                "gravity": self.config.gravity,
                "seed": self.config.seed,
            },
            "environment": self.environment.snapshot(),
            "units": {
                uid: {
                    "unit_id": u.unit_id,
                    "name": u.name,
                    "unit_type": u.unit_type.value,
                    "alliance": u.alliance.value,
                    "position": u.position,
                    "heading": u.heading,
                    "health": u.state.health,
                    "max_health": u.stats.max_health,
                    "morale": u.state.morale,
                    "is_alive": u.state.is_alive,
                    "status": u.state.status,
                    "weapon": u.weapon,
                    "squad_id": u.squad_id,
                    "kill_count": u.state.kill_count,
                    "damage_dealt": u.state.damage_dealt,
                    "damage_taken": u.state.damage_taken,
                }
                for uid, u in self.units.items()
            },
            "squads": {
                sid: {
                    "squad_id": sq.squad_id,
                    "name": sq.name,
                    "alliance": sq.alliance,
                    "members": list(sq.members),
                    "leader_id": sq.leader_id,
                    "morale": sq.state.morale,
                    "cohesion": sq.state.cohesion,
                    "casualties": sq.state.casualties,
                }
                for sid, sq in self.squads.items()
            },
            "vehicles": {
                vid: {
                    "vehicle_id": v.vehicle_id,
                    "name": v.name,
                    "vehicle_class": v.vehicle_class.value,
                    "alliance": v.alliance,
                    "position": v.position,
                    "health": v.health,
                    "is_destroyed": v.is_destroyed,
                }
                for vid, v in self.vehicles.items()
            },
            "crowd": self.crowd.snapshot() if self.crowd else None,
            "destruction": {
                "structures": len(self.destruction.structures) if self.destruction else 0,
                "fires": len(self.destruction.fires) if self.destruction else 0,
            },
            "damage_summary": self.damage_tracker.summary(),
        }

    def stats(self) -> dict[str, Any]:
        """Quick world statistics."""
        alive_friendly = sum(
            1 for u in self.units.values()
            if u.is_alive() and u.alliance == Alliance.FRIENDLY
        )
        alive_hostile = sum(
            1 for u in self.units.values()
            if u.is_alive() and u.alliance == Alliance.HOSTILE
        )
        dead_count = sum(1 for u in self.units.values() if not u.is_alive())
        total_units = len(self.units)

        destroyed_structures = 0
        if self.destruction:
            from tritium_lib.sim_engine.destruction import DamageLevel
            destroyed_structures = sum(
                1 for s in self.destruction.structures
                if s.damage_level in (DamageLevel.DESTROYED, DamageLevel.COLLAPSED)
            )

        active_fires = 0
        if self.destruction:
            active_fires = len(self.destruction.fires)

        destroyed_vehicles = sum(1 for v in self.vehicles.values() if v.is_destroyed)
        crowd_count = len(self.crowd.members) if self.crowd else 0

        return {
            "tick_count": self.tick_count,
            "sim_time": round(self.sim_time, 2),
            "total_units": total_units,
            "alive_friendly": alive_friendly,
            "alive_hostile": alive_hostile,
            "dead": dead_count,
            "total_vehicles": len(self.vehicles),
            "destroyed_vehicles": destroyed_vehicles,
            "total_squads": len(self.squads),
            "destroyed_structures": destroyed_structures,
            "active_fires": active_fires,
            "crowd_count": crowd_count,
            "active_projectiles": len(self.projectile_sim.projectiles),
            "active_effects": len(self.area_effects.effects),
            "environment": self.environment.describe(),
        }


# ---------------------------------------------------------------------------
# WorldBuilder — fluent API
# ---------------------------------------------------------------------------


class WorldBuilder:
    """Fluent builder for constructing World instances."""

    def __init__(self) -> None:
        self._config = WorldConfig()
        self._terrain_noise: dict | None = None
        self._weather: Weather | None = None
        self._hour: float | None = None
        self._squads: list[tuple[str, str, list[str], list[Vec2]]] = []
        self._vehicles: list[tuple[str, str, str, Vec2]] = []
        self._structures: list[tuple[Vec2, tuple[float, float, float], str]] = []
        self._crowds: list[tuple[Vec2, int, float, CrowdMood]] = []

    def set_map_size(self, width: float, height: float) -> WorldBuilder:
        self._config.map_size = (width, height)
        return self

    def set_tick_rate(self, rate: float) -> WorldBuilder:
        self._config.tick_rate = rate
        return self

    def set_seed(self, seed: int) -> WorldBuilder:
        self._config.seed = seed
        return self

    def set_gravity(self, gravity: float) -> WorldBuilder:
        self._config.gravity = gravity
        return self

    def enable_weather(self, enabled: bool = True) -> WorldBuilder:
        self._config.enable_weather = enabled
        return self

    def enable_destruction(self, enabled: bool = True) -> WorldBuilder:
        self._config.enable_destruction = enabled
        return self

    def enable_crowds(self, enabled: bool = True) -> WorldBuilder:
        self._config.enable_crowds = enabled
        return self

    def enable_vehicles(self, enabled: bool = True) -> WorldBuilder:
        self._config.enable_vehicles = enabled
        return self

    def enable_los(self, enabled: bool = True) -> WorldBuilder:
        self._config.enable_los = enabled
        return self

    def add_terrain_noise(self, octaves: int = 4, amplitude: float = 10.0, seed: int | None = None) -> WorldBuilder:
        self._terrain_noise = {"octaves": octaves, "amplitude": amplitude, "seed": seed}
        return self

    def set_weather(self, weather: Weather) -> WorldBuilder:
        self._weather = weather
        return self

    def set_time(self, hour: float) -> WorldBuilder:
        self._hour = hour
        return self

    def spawn_friendly_squad(self, name: str, templates: list[str], center: Vec2, spacing: float = 3.0) -> WorldBuilder:
        positions = self._spread_positions(center, len(templates), spacing)
        self._squads.append((name, "friendly", templates, positions))
        return self

    def spawn_hostile_squad(self, name: str, templates: list[str], center: Vec2, spacing: float = 3.0) -> WorldBuilder:
        positions = self._spread_positions(center, len(templates), spacing)
        self._squads.append((name, "hostile", templates, positions))
        return self

    def add_vehicle(self, template: str, name: str, alliance: str, position: Vec2) -> WorldBuilder:
        self._vehicles.append((template, name, alliance, position))
        return self

    def add_building(self, position: Vec2, size: tuple[float, float, float], material: str = "concrete") -> WorldBuilder:
        self._structures.append((position, size, material))
        return self

    def add_crowd(self, center: Vec2, count: int, radius: float, mood: CrowdMood = CrowdMood.CALM) -> WorldBuilder:
        self._crowds.append((center, count, radius, mood))
        self._config.enable_crowds = True
        return self

    def build(self) -> World:
        """Construct the World from accumulated configuration."""
        world = World(self._config)

        # Terrain noise
        if self._terrain_noise is not None:
            map_w, map_h = self._config.map_size
            grid_w = max(1, int(map_w))
            grid_h = max(1, int(map_h))
            seed = self._terrain_noise.get("seed") or (self._config.seed or 42)
            world.heightmap = HeightMap.from_noise(
                grid_w, grid_h, cell_size=1.0,
                octaves=self._terrain_noise["octaves"],
                seed=seed,
                amplitude=self._terrain_noise["amplitude"],
            )
            if self._config.enable_los:
                world.los = LineOfSight(world.heightmap)

        # Weather
        if self._weather is not None:
            world.environment.weather.state.current = self._weather

        # Time
        if self._hour is not None:
            world.environment.time.hour = self._hour

        # Squads
        for name, alliance, templates, positions in self._squads:
            world.spawn_squad(name, alliance, templates, positions)

        # Vehicles
        for template, name, alliance, position in self._vehicles:
            world.spawn_vehicle(template, name, alliance, position)

        # Structures
        for i, (position, size, material) in enumerate(self._structures):
            mat_props = MATERIAL_PROPERTIES.get(material, MATERIAL_PROPERTIES["concrete"])
            s = Structure(
                structure_id=f"bldg_{i}",
                structure_type=StructureType.BUILDING,
                position=position,
                size=size,
                material=material,
                health=mat_props["health"],
                max_health=mat_props["health"],
            )
            world.add_structure(s)

        # Crowds
        for center, count, radius, mood in self._crowds:
            world.spawn_crowd(center, count, radius, mood)

        return world

    @staticmethod
    def _spread_positions(center: Vec2, count: int, spacing: float) -> list[Vec2]:
        """Spread positions in a line centered on center."""
        positions: list[Vec2] = []
        for i in range(count):
            offset = (i - (count - 1) / 2.0) * spacing
            positions.append((center[0] + offset, center[1]))
        return positions


# ---------------------------------------------------------------------------
# WORLD_PRESETS
# ---------------------------------------------------------------------------


def _preset_urban_combat() -> World:
    """Urban combat: buildings, squads, vehicles, night time."""
    return (
        WorldBuilder()
        .set_map_size(300, 300)
        .set_seed(42)
        .set_time(hour=22.0)
        .enable_destruction(True)
        .add_building((100, 100), (20, 15, 10), "concrete")
        .add_building((120, 80), (10, 10, 5), "wood")
        .add_building((80, 130), (15, 10, 8), "brick")
        .add_building((150, 120), (12, 12, 6), "metal")
        .spawn_friendly_squad("Alpha", ["infantry"] * 4 + ["sniper"], (50, 50))
        .spawn_hostile_squad("Tango", ["infantry"] * 6, (200, 200))
        .add_vehicle("humvee", "Humvee-1", "friendly", (40, 40))
        .add_vehicle("technical", "Bandit-1", "hostile", (210, 210))
        .build()
    )


def _preset_open_field() -> World:
    """Open field: flat terrain, infantry squads, daytime."""
    return (
        WorldBuilder()
        .set_map_size(500, 500)
        .set_seed(7)
        .set_time(hour=14.0)
        .set_weather(Weather.CLEAR)
        .enable_destruction(False)
        .spawn_friendly_squad("Bravo", ["infantry"] * 6, (100, 250))
        .spawn_hostile_squad("Enemy", ["infantry"] * 8, (400, 250))
        .build()
    )


def _preset_riot_response() -> World:
    """Riot response: crowd + police units + teargas potential."""
    return (
        WorldBuilder()
        .set_map_size(200, 200)
        .set_seed(99)
        .set_time(hour=15.0)
        .enable_crowds(True)
        .enable_destruction(False)
        .add_crowd((100, 100), 150, 30.0, CrowdMood.AGITATED)
        .spawn_friendly_squad("Police", ["infantry"] * 4, (100, 30))
        .build()
    )


def _preset_convoy_ambush() -> World:
    """Convoy route + ambush positions."""
    return (
        WorldBuilder()
        .set_map_size(500, 200)
        .set_seed(55)
        .set_time(hour=10.0)
        .add_terrain_noise(octaves=3, amplitude=5.0)
        .add_vehicle("humvee", "Lead", "friendly", (50, 100))
        .add_vehicle("humvee", "Escort", "friendly", (35, 100))
        .add_vehicle("btr80", "APC", "friendly", (20, 100))
        .spawn_hostile_squad("Ambush-1", ["infantry"] * 4, (250, 50))
        .spawn_hostile_squad("Ambush-2", ["infantry"] * 3 + ["sniper"], (250, 150))
        .add_building((240, 80), (8, 8, 5), "concrete")
        .add_building((260, 120), (10, 6, 4), "wood")
        .build()
    )


def _preset_drone_strike() -> World:
    """Reaper drone + ground targets."""
    world = (
        WorldBuilder()
        .set_map_size(500, 500)
        .set_seed(77)
        .set_time(hour=13.0)
        .set_weather(Weather.CLEAR)
        .spawn_hostile_squad("Ground-Force", ["infantry"] * 6, (250, 250))
        .add_vehicle("reaper", "Reaper-1", "friendly", (250, 50))
        .build()
    )
    # Set up drone controller with orbit pattern
    for vid, v in world.vehicles.items():
        if v.name == "Reaper-1":
            v.altitude = 200.0
            v.speed = v.max_speed * 0.6
            ctrl = DroneController(v)
            ctrl.orbit((250, 250), radius=100.0, altitude=200.0)
            world.drone_controllers[vid] = ctrl
            break
    return world


WORLD_PRESETS: dict[str, Any] = {
    "urban_combat": _preset_urban_combat,
    "open_field": _preset_open_field,
    "riot_response": _preset_riot_response,
    "convoy_ambush": _preset_convoy_ambush,
    "drone_strike": _preset_drone_strike,
}
"""Preset name -> factory function() -> World."""
