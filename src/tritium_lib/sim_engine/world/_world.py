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
from tritium_lib.sim_engine.ai.formations import (
    FormationType,
    FormationConfig,
    FormationMover,
    formation_to_three_js,
    formation_mover_to_three_js,
)


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

        # --- geospatial terrain layer (optional) ---
        self.terrain_layer: Any = None  # TerrainLayer from intelligence.geospatial
        self.sidewalk_graph: Any = None  # SidewalkGraph for pedestrian nav

        # --- formation movers ---
        # squad_id -> FormationMover; updated each tick alongside squad AI
        self._formation_movers: dict[str, FormationMover] = {}

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

    def assign_squad_formation(
        self,
        squad_id: str,
        waypoints: list[Vec2],
        formation: FormationType = FormationType.WEDGE,
        spacing: float = 3.0,
        max_speed: float = 5.0,
    ) -> FormationMover | None:
        """Attach a FormationMover to a squad so it moves in formation.

        The mover is ticked each world tick. Formation slot positions are
        nudged onto each living squad member every tick, and the formation
        data is included in the render frame for Three.js to draw overlay
        lines.

        Args:
            squad_id: ID of the squad to attach the mover to.
            waypoints: Path for the formation leader to follow.
            formation: FormationType to use (default WEDGE).
            spacing: Metres between formation slots.
            max_speed: Leader move speed in m/s.

        Returns:
            The new FormationMover, or None if the squad doesn't exist.
        """
        if squad_id not in self.squads:
            return None
        if not waypoints:
            return None
        mover = FormationMover(
            waypoints=waypoints,
            formation=formation,
            spacing=spacing,
            max_speed=max_speed,
        )
        self._formation_movers[squad_id] = mover
        return mover

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

        # 5b. In-flight hit detection: check active projectiles against units
        hit_impacts = self._check_projectile_hits()
        impacts.extend(hit_impacts)

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
        """Update squad state: cohesion, morale, orders, and formation movers."""
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

            # Tick formation mover and nudge member positions toward formation slots
            mover = self._formation_movers.get(sid)
            if mover is not None and not mover.is_complete():
                member_pos: dict[str, Vec2] = {
                    uid: unit_positions[uid]
                    for uid in living
                    if uid in unit_positions
                }
                if member_pos:
                    targets = mover.tick(dt, member_pos)
                    # Gently move each living member toward their formation slot
                    for uid, target in targets.items():
                        unit = self.units.get(uid)
                        if unit is None or not unit.is_alive():
                            continue
                        # Skip units that are actively engaging — don't pull
                        # them out of a firefight into formation
                        if unit.state.status in ("attacking",):
                            continue
                        dx = target[0] - unit.position[0]
                        dy = target[1] - unit.position[1]
                        slot_dist = math.sqrt(dx * dx + dy * dy)
                        if slot_dist > 0.5:  # only nudge if meaningfully off-slot
                            # Converge at 20% of distance per tick, capped to speed
                            step = min(slot_dist * 0.2, unit.effective_speed() * dt)
                            frac = step / slot_dist
                            unit.position = (
                                unit.position[0] + dx * frac,
                                unit.position[1] + dy * frac,
                            )
                            if unit.state.status not in ("attacking",):
                                unit.state.status = "moving"

    # -- Cover-position helpers ----------------------------------------------

    def _get_cover_positions(self) -> list[Vec2]:
        """Return known cover positions from the destruction engine (buildings)."""
        if self.destruction is None:
            return []
        return [s.position for s in self.destruction.structures if s.health > 0]

    def _find_nearest_cover(self, unit_pos: Vec2, cover_positions: list[Vec2]) -> Vec2 | None:
        """Return the closest cover position, or None if none available."""
        if not cover_positions:
            return None
        return min(cover_positions, key=lambda cp: distance(unit_pos, cp))

    def _count_nearby_allies(self, uid: str, alive_units: dict, radius: float = 40.0) -> int:
        """Count friendly units within *radius* of *uid*."""
        unit = alive_units.get(uid)
        if unit is None:
            return 0
        return sum(
            1 for oid, other in alive_units.items()
            if oid != uid and other.alliance == unit.alliance
            and distance(unit.position, other.position) <= radius
        )

    def _compute_flank_position(self, unit_pos: Vec2, enemy_pos: Vec2) -> Vec2:
        """Compute a flanking position 90° to the side of the enemy."""
        to_enemy = _sub(enemy_pos, unit_pos)
        # Perpendicular: rotate 90°
        perp = (-to_enemy[1], to_enemy[0])
        perp_norm = normalize(perp)
        # Move 15m to the side of the enemy
        flank = _add(enemy_pos, _scale(perp_norm, 15.0))
        return flank

    # -- Main unit tick -------------------------------------------------------

    def _tick_units(self, dt: float) -> None:
        """Per-unit AI: find targets, decide action, execute.

        Behaviour priority (evaluated top-to-bottom, first match wins):
          1. Retreat if health < 30% AND morale < 0.30
          2. Seek cover if under heavy suppression (suppression > 0.6) AND damaged (health < 60%)
          3. Suppress enemy direction even without perfect LOS (fired-upon / nearby)
          4. Flank when 2+ friendlies outnumber a lone visible enemy
          5. Move-to-range and engage nearest visible enemy (standard)
          6. Follow squad order when no enemies visible
        """
        alive_units = {uid: u for uid, u in self.units.items() if u.is_alive()}
        cover_positions = self._get_cover_positions()

        for uid, unit in alive_units.items():
            # Recover suppression
            unit.recover_suppression(dt)

            health_pct = unit.state.health / max(unit.stats.max_health, 1.0)
            morale = unit.state.morale

            # --- BEHAVIOUR 1: Retreat when critically low ---
            if health_pct < 0.30 and morale < 0.30:
                # Find retreat direction: away from all visible enemies
                retreat_target = self._compute_retreat_pos(uid, alive_units)
                if retreat_target is not None:
                    self._move_unit_toward(unit, retreat_target, dt, status="retreating")
                else:
                    unit.state.status = "retreating"
                continue

            # --- BEHAVIOUR 2: Seek cover when under fire and damaged ---
            if unit.state.suppression > 0.6 and health_pct < 0.60:
                cover = self._find_nearest_cover(unit.position, cover_positions)
                if cover is not None and distance(unit.position, cover) > 2.0:
                    self._move_unit_toward(unit, cover, dt, status="moving")
                    # Still try to suppress enemies while moving to cover
                    # (below we allow firing without LOS)
                    self._try_suppress_without_los(uid, unit, alive_units, dt)
                    continue

            # Find enemies (LOS-confirmed)
            enemies_los: list[dict] = []
            # Track all nearby enemies even without LOS for suppression
            enemies_nearby: list[dict] = []
            det_range = unit.stats.detection_range * self.environment.detection_range_modifier()
            for oid, other in alive_units.items():
                if oid == uid:
                    continue
                if other.alliance == unit.alliance:
                    continue
                d = distance(unit.position, other.position)
                if d > det_range:
                    continue
                entry = {
                    "id": oid,
                    "pos": other.position,
                    "damage": other.stats.attack_damage,
                    "health": other.state.health / other.stats.max_health,
                    "facing": other.heading,
                    "dist": d,
                }
                enemies_nearby.append(entry)
                # LOS check
                if self.los is not None:
                    if not self.los.can_see(unit.position, other.position):
                        continue
                enemies_los.append(entry)

            # --- BEHAVIOUR 3: Suppress enemies without LOS ---
            # If suppressed and there are nearby enemies (heard but not seen),
            # fire toward their last known direction even without LOS
            if unit.state.suppression > 0.3 and enemies_nearby and not enemies_los:
                nearest_heard = min(enemies_nearby, key=lambda e: e["dist"])
                if unit.can_attack(self.sim_time):
                    # Add random scatter to simulate suppression fire accuracy
                    scatter_angle = self._rng.uniform(-0.3, 0.3)
                    to_enemy = _sub(nearest_heard["pos"], unit.position)
                    cos_s = math.cos(scatter_angle)
                    sin_s = math.sin(scatter_angle)
                    scattered = (
                        to_enemy[0] * cos_s - to_enemy[1] * sin_s,
                        to_enemy[0] * sin_s + to_enemy[1] * cos_s,
                    )
                    suppress_target = _add(unit.position, scattered)
                    self.fire_weapon(uid, suppress_target)
                    unit.state.status = "attacking"
                continue

            if not enemies_los:
                # No targets visible — follow squad order to advance toward enemy
                if unit.state.status not in ("dead",):
                    moved = False
                    for _sid, squad in self.squads.items():
                        if uid in squad.members and squad.current_order is not None:
                            order = squad.current_order
                            if order.order_type in ("advance", "flank_left", "flank_right") and order.target_pos is not None:
                                direction = _sub(order.target_pos, unit.position)
                                d = distance(unit.position, order.target_pos)
                                if d > 5.0:
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

            # --- BEHAVIOUR 4: Flank when 2+ friendlies vs 1 visible enemy ---
            if len(enemies_los) == 1:
                allies_nearby = self._count_nearby_allies(uid, alive_units, radius=35.0)
                if allies_nearby >= 2:
                    # This unit flanks; allies will engage directly
                    # Determine if this is the "flanker" (lowest morale — most
                    # aggressive) or the "suppressor" (stays and fires)
                    # Simple heuristic: units whose ID sorts last in the squad
                    # become the flanker
                    squad_mates = [
                        oid for oid, other in alive_units.items()
                        if oid != uid
                        and other.alliance == unit.alliance
                        and distance(unit.position, other.position) <= 35.0
                    ]
                    is_flanker = uid > max(squad_mates, default=uid)
                    if is_flanker:
                        enemy = enemies_los[0]
                        flank_pos = self._compute_flank_position(unit.position, enemy["pos"])
                        dist_to_flank = distance(unit.position, flank_pos)
                        # Move toward flank position until within 5 m of it;
                        # only switch to engaging once actually flanking
                        if dist_to_flank > 5.0:
                            self._move_unit_toward(unit, flank_pos, dt, status="moving")
                        else:
                            # Arrived at flank — fire at enemy
                            if unit.can_attack(self.sim_time):
                                self.fire_weapon(uid, enemy["pos"])
                        continue
                    # Non-flanker: suppress (fire at enemy) to cover the flanker
                    nearest = enemies_los[0]
                    if unit.can_attack(self.sim_time):
                        self.fire_weapon(uid, nearest["pos"])
                    else:
                        nearest_dist = distance(unit.position, nearest["pos"])
                        if nearest_dist > unit.stats.attack_range:
                            self._move_unit_toward(unit, nearest["pos"], dt, status="moving")
                    continue

            # --- BEHAVIOUR 5: Standard engage ---
            nearest = min(enemies_los, key=lambda e: e["dist"])
            nearest_dist = nearest["dist"]

            if nearest_dist > unit.stats.attack_range:
                self._move_unit_toward(unit, nearest["pos"], dt, status="moving")
            else:
                if unit.can_attack(self.sim_time):
                    self.fire_weapon(uid, nearest["pos"])

    # -- Movement helpers -----------------------------------------------------

    def _move_unit_toward(self, unit: Any, target: Vec2, dt: float, status: str = "moving") -> None:
        """Move *unit* one step toward *target* at its effective speed."""
        direction = _sub(target, unit.position)
        d = distance(unit.position, target)
        if d < 0.01:
            return
        direction = normalize(direction)
        move_speed = unit.effective_speed() * self.environment.movement_speed_modifier()
        step = min(move_speed * dt, d)
        unit.position = (
            unit.position[0] + direction[0] * step,
            unit.position[1] + direction[1] * step,
        )
        unit.heading = math.atan2(direction[1], direction[0])
        unit.state.status = status

    def _compute_retreat_pos(self, uid: str, alive_units: dict) -> Vec2 | None:
        """Compute a retreat position away from the centroid of all visible enemies."""
        unit = alive_units.get(uid)
        if unit is None:
            return None
        enemy_positions = [
            other.position
            for oid, other in alive_units.items()
            if oid != uid and other.alliance != unit.alliance
        ]
        if not enemy_positions:
            return None
        cx = sum(p[0] for p in enemy_positions) / len(enemy_positions)
        cy = sum(p[1] for p in enemy_positions) / len(enemy_positions)
        away = _sub(unit.position, (cx, cy))
        away_norm = normalize(away)
        mag = math.sqrt(away[0] ** 2 + away[1] ** 2)
        if mag < 1e-6:
            # Directly on top of enemy centroid — retreat north
            away_norm = (0.0, 1.0)
        return _add(unit.position, _scale(away_norm, 30.0))

    def _try_suppress_without_los(
        self, uid: str, unit: Any, alive_units: dict, dt: float
    ) -> None:
        """Fire suppression shots toward nearby enemies even without confirmed LOS."""
        det_range = unit.stats.detection_range * self.environment.detection_range_modifier()
        for oid, other in alive_units.items():
            if oid == uid or other.alliance == unit.alliance:
                continue
            if distance(unit.position, other.position) > det_range:
                continue
            if unit.can_attack(self.sim_time):
                scatter = self._rng.uniform(-0.4, 0.4)
                to_e = _sub(other.position, unit.position)
                cos_s, sin_s = math.cos(scatter), math.sin(scatter)
                scattered_target = _add(unit.position, (
                    to_e[0] * cos_s - to_e[1] * sin_s,
                    to_e[0] * sin_s + to_e[1] * cos_s,
                ))
                self.fire_weapon(uid, scattered_target)
            break  # suppress at most one target per tick

    def _check_projectile_hits(self) -> list[dict]:
        """Check active projectiles for hits using ray-segment vs circle test.

        Because projectiles travel fast (hundreds of m/s) and ticks are 0.1s,
        a simple proximity check misses.  Instead we check whether the line
        segment from prev_pos → current_pos passes within *hit_radius* of any
        enemy unit.
        """
        hits: list[dict] = []
        hit_radius = 3.0  # metres
        alive = {uid: u for uid, u in self.units.items() if u.is_alive()}
        if not alive:
            return hits

        for proj in list(self.projectile_sim.projectiles):
            if not proj.is_active:
                continue

            # Compute previous position from velocity
            vx, vy = proj.velocity
            px, py = proj.position
            prev_x = px - vx * 0.1  # dt assumed 0.1
            prev_y = py - vy * 0.1

            # Skip first 3m of travel (avoid hitting shooter)
            if proj.distance_traveled() < 5.0:
                continue

            for uid, unit in alive.items():
                ux, uy = unit.position
                # Point-to-segment distance
                seg_dx = px - prev_x
                seg_dy = py - prev_y
                seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
                if seg_len_sq < 0.001:
                    continue
                t = max(0.0, min(1.0,
                    ((ux - prev_x) * seg_dx + (uy - prev_y) * seg_dy) / seg_len_sq
                ))
                closest_x = prev_x + t * seg_dx
                closest_y = prev_y + t * seg_dy
                d = math.sqrt((ux - closest_x) ** 2 + (uy - closest_y) ** 2)
                if d <= hit_radius:
                    proj.is_active = False
                    hits.append({
                        "x": closest_x,
                        "y": closest_y,
                        "type": proj.projectile_type.value if hasattr(proj.projectile_type, 'value') else str(proj.projectile_type),
                        "damage": proj.damage,
                        "effect": "hit",
                        "weapon_id": proj.weapon_id,
                    })
                    break
        return hits

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

        # Formation layer — one entry per active squad with a mover
        formations_data: list[dict] = []
        alive_unit_positions: dict[str, Vec2] = {
            uid: u.position for uid, u in self.units.items() if u.is_alive()
        }
        for sid, mover in self._formation_movers.items():
            squad = self.squads.get(sid)
            if squad is None:
                continue
            living_members = [m for m in squad.members if m in alive_unit_positions]
            if not living_members:
                continue
            # Build a snapshot config from current leader position + mover state
            leader_uid = living_members[0]
            leader_pos = alive_unit_positions[leader_uid]
            config = FormationConfig(
                formation_type=mover.formation,
                spacing=mover.spacing,
                facing=mover._facing,
                leader_pos=leader_pos,
                num_members=len(living_members),
            )
            f_data = formation_to_three_js(config, living_members)
            f_data["squad_id"] = sid
            f_data["squad_name"] = getattr(squad, "name", sid)
            f_data["alliance"] = getattr(squad, "alliance", "unknown")
            f_data["progress"] = mover.progress()
            f_data["complete"] = mover.is_complete()
            formations_data.append(f_data)
        frame["formations"] = formations_data

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
        self._terrain_layer: Any = None
        self._terrain_layer_ao_id: str | None = None

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

    def load_terrain_layer(self, terrain_layer: Any) -> WorldBuilder:
        """Attach a geospatial TerrainLayer for terrain-aware pathfinding.

        The terrain layer provides classified terrain polygons (buildings,
        roads, water, vegetation, sidewalks) from satellite/aerial imagery.
        It builds a SidewalkGraph for pedestrian navigation automatically.
        """
        self._terrain_layer = terrain_layer
        return self

    def load_terrain_cache(self, ao_id: str, cache_dir: str = "data/cache/terrain") -> WorldBuilder:
        """Load a cached TerrainLayer by AO ID.

        Convenience method that creates a TerrainLayer and loads from cache.
        """
        self._terrain_layer_ao_id = ao_id
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

        # Geospatial terrain layer
        if self._terrain_layer is not None:
            world.terrain_layer = self._terrain_layer
            try:
                from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph
                sg = SidewalkGraph()
                sg.build_from_terrain_layer(self._terrain_layer)
                world.sidewalk_graph = sg
            except Exception:
                pass
            # Note: CoverSystem terrain loading is done in game_server.py
            # since CoverSystem is a GameState subsystem, not a World subsystem
        elif self._terrain_layer_ao_id is not None:
            try:
                from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
                from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph
                tl = TerrainLayer()
                if tl.load_cached(self._terrain_layer_ao_id):
                    world.terrain_layer = tl
                    sg = SidewalkGraph()
                    sg.build_from_terrain_layer(tl)
                    world.sidewalk_graph = sg
                    # CoverSystem terrain loading done in game_server.py
            except Exception:
                pass

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
