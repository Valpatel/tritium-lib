# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Standalone demo game server proving all sim_engine modules work together.

Runs a complete tactical simulation via FastAPI, streaming frames over
WebSocket to a Three.js frontend at 10 fps.

Usage::

    python3 -m tritium_lib.sim_engine.demos.game_server
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, Response
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

    class _NoOpDecorator:
        """Stub that silently ignores all route/websocket decorator calls."""
        title: str = ""
        version: str = ""

        def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
            pass

        def _noop(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            def decorator(fn: Any) -> Any:  # noqa: ANN401
                return fn
            return decorator

        get = post = put = delete = patch = websocket = on_event = _noop  # type: ignore[assignment]
        add_middleware = include_router = _noop  # type: ignore[assignment]

    FastAPI = _NoOpDecorator  # type: ignore[assignment,misc]
    WebSocket = object  # type: ignore[assignment,misc]
    WebSocketDisconnect = Exception  # type: ignore[assignment,misc]
    HTMLResponse = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# CitySim backend — real sim_engine city simulation
# ---------------------------------------------------------------------------

from tritium_lib.sim_engine.demos.city_sim_backend import CitySim

# ---------------------------------------------------------------------------
# Import EVERY sim_engine module
# ---------------------------------------------------------------------------

# --- Subpackage imports: effects, audio, game, debug ---
from tritium_lib.sim_engine.effects import (
    EffectsManager, explosion as fx_explosion, muzzle_flash as fx_muzzle_flash,
    tracer as fx_tracer, smoke as fx_smoke, fire as fx_fire,
    blood_splatter as fx_blood, debris as fx_debris, sparks as fx_sparks,
)
from tritium_lib.sim_engine.audio.spatial import (
    SoundEvent, distance_attenuation, stereo_pan, propagation_delay,
    gunshot_layers, explosion_parameters,
)
from tritium_lib.sim_engine.game.stats import StatsTracker
from tritium_lib.sim_engine.game.difficulty import DifficultyScaler
from tritium_lib.sim_engine.debug import DebugOverlay

# 1. world.py
from tritium_lib.sim_engine.world import (
    World, WorldConfig, WorldBuilder, WORLD_PRESETS,
)
# 2. scenario.py
from tritium_lib.sim_engine.scenario import (
    Scenario, ScenarioConfig, WaveConfig, Objective, PRESET_SCENARIOS,
)
# 3. units.py
from tritium_lib.sim_engine.units import (
    Unit, Alliance, UnitType, UNIT_TEMPLATES, create_unit,
)
# 4. vehicles.py
from tritium_lib.sim_engine.vehicles import (
    VehicleState, VehiclePhysicsEngine, DroneController, ConvoySimulator,
    VEHICLE_TEMPLATES, create_vehicle, VehicleClass,
)
# 5. arsenal.py
from tritium_lib.sim_engine.arsenal import (
    ARSENAL, Weapon, Projectile, ProjectileSimulator,
    AreaEffect, AreaEffectManager,
    create_explosion_effect, create_smoke_effect, create_fire_effect,
)
# 6. damage.py
from tritium_lib.sim_engine.damage import (
    DamageType, DamageTracker, HitResult, resolve_attack, resolve_explosion,
)
# 7. terrain.py
from tritium_lib.sim_engine.terrain import HeightMap, LineOfSight
# 8. environment.py
from tritium_lib.sim_engine.environment import (
    Environment, TimeOfDay, Weather, WeatherSimulator, WeatherEffects,
)
# 9. crowd.py
from tritium_lib.sim_engine.crowd import CrowdSimulator, CrowdMood, CrowdEvent
# 10. destruction.py
from tritium_lib.sim_engine.destruction import (
    DestructionEngine, Structure, StructureType, MATERIAL_PROPERTIES,
)
# 11. detection.py
from tritium_lib.sim_engine.detection import (
    DetectionEngine, Sensor, SensorType, SignatureProfile, SIGNATURE_PRESETS,
)
# 12. comms.py
from tritium_lib.sim_engine.comms import (
    CommsSimulator, RadioChannel, Radio, RadioType, RADIO_PRESETS,
)
# 13. medical.py
from tritium_lib.sim_engine.medical import (
    MedicalEngine, InjuryType, InjurySeverity, TriageCategory,
)
# 14. logistics.py
from tritium_lib.sim_engine.logistics import (
    LogisticsEngine, SupplyCache, SupplyType, cache_from_preset,
)
# 15. naval.py
from tritium_lib.sim_engine.naval import (
    NavalCombatEngine, ShipClass, ShipState, create_ship, NavalPhysics,
)
# 16. air_combat.py
from tritium_lib.sim_engine.air_combat import (
    AirCombatEngine, AircraftClass, AircraftState, AIRCRAFT_TEMPLATES, AntiAir,
)
# 17. fortifications.py
from tritium_lib.sim_engine.fortifications import (
    EngineeringEngine, FortificationType, Fortification,
)
# 18. asymmetric.py
from tritium_lib.sim_engine.asymmetric import (
    AsymmetricEngine, TrapType, Trap,
)
# 19. civilian.py
from tritium_lib.sim_engine.civilian import (
    CivilianSimulator, CivilianState, Civilian,
)
# 20. intel.py
from tritium_lib.sim_engine.intel import (
    IntelEngine, FogOfWar, IntelType,
)
# 20b. commander.py — battle narration and tactical AI
from tritium_lib.sim_engine.commander import (
    BattleNarrator, NarrationLog, NarrationEvent, TacticalAdvisor, PERSONALITIES,
)
# 21. scoring.py
from tritium_lib.sim_engine.scoring import (
    ScoringEngine, ScoreCategory, Achievement,
)
# 22. factions.py
from tritium_lib.sim_engine.factions import (
    DiplomacyEngine, Faction, Relation,
)
# 23. campaign.py
from tritium_lib.sim_engine.campaign import (
    Campaign, CAMPAIGNS,
)
# 24. renderer.py
from tritium_lib.sim_engine.renderer import SimRenderer, RenderLayer
# 25. ai/tactics.py
from tritium_lib.sim_engine.ai.tactics import TacticsEngine, TacticalAction
# 26. ai/squad.py
from tritium_lib.sim_engine.ai.squad import Squad, SquadRole, SquadTactics, Order
# AI behavior systems — behavior trees, steering, formations, combat AI
from tritium_lib.sim_engine.ai.behavior_tree import (
    Node as BTNode,
    Status as BTStatus,
    make_patrol_tree,
    make_friendly_tree,
    make_hostile_tree,
    make_civilian_tree,
)
from tritium_lib.sim_engine.ai.steering import (
    seek as steering_seek,
    flee as steering_flee,
    arrive as steering_arrive,
    wander as steering_wander,
    distance as steering_distance,
    magnitude as steering_magnitude,
    normalize as steering_normalize,
    truncate as steering_truncate,
)
from tritium_lib.sim_engine.ai.formations import (
    FormationMover,
    FormationType,
    FormationConfig,
    get_formation_positions,
)
from tritium_lib.sim_engine.ai.combat_ai import (
    find_cover,
    is_in_cover,
    should_engage,
    should_retreat as combat_should_retreat,
    formation_positions as combat_formation_positions,
    assign_targets,
    compute_flank_position,
)
# AI pathfinding — road network A* and pedestrian visibility-graph navigation
from tritium_lib.sim_engine.ai.pathfinding import (
    RoadNetwork,
    WalkableArea,
    plan_patrol_route,
)
# AI strategy — faction-level strategic planner
from tritium_lib.sim_engine.ai.strategy import (
    StrategicAI,
    StrategicGoal,
    StrategicPlan,
    STRATEGY_PROFILES,
)
# AI behavior profiles — per-unit personality archetypes
from tritium_lib.sim_engine.ai.behavior_profiles import (
    BehaviorEngine,
    BehaviorProfile,
    BehaviorTrait,
    PROFILES as BEHAVIOR_PROFILES,
)
# 27. morale.py
from tritium_lib.sim_engine.morale import MoraleEngine, MoraleEvent, MoraleEventType
# 28. electronic_warfare.py
from tritium_lib.sim_engine.electronic_warfare import (
    EWEngine, EWJammer, JammerType, CyberAttack,
)
# 29. supply_routes.py
from tritium_lib.sim_engine.supply_routes import (
    SupplyRouteEngine, SupplyLine, SupplyConvoy,
)
# 14. Objectives — mission chains with triggers
from tritium_lib.sim_engine.abilities import (
    AbilityEngine, Ability, ABILITIES,
)
# 15. Status effects (suppression, healing, burning)
from tritium_lib.sim_engine.status_effects import (
    StatusEffectEngine,
)
# 16. Collision detection
from tritium_lib.sim_engine.collision import (
    CollisionWorld, Collider, ColliderType,
)
# 17. Wave spawner — reinforcement waves
from tritium_lib.sim_engine.spawner import (
    SpawnerEngine, SpawnPoint, EnemyComposition, SpawnPattern, WAVE_PRESETS,
)
# 18. Procedural map generation
from tritium_lib.sim_engine.mapgen import MapGenerator, MAP_PRESETS
# 19. Weather visual effects
from tritium_lib.sim_engine.weather_fx import WeatherFXEngine
# 19. Artillery — fire support
from tritium_lib.sim_engine.artillery import (
    ArtilleryEngine, ArtilleryPiece, ArtilleryType, ARTILLERY_TEMPLATES,
)
# 17. Objectives — mission chains with triggers
from tritium_lib.sim_engine.objectives import (
    ObjectiveEngine, MissionObjective, ObjectiveType, ObjectiveStatus,
    OBJECTIVE_TEMPLATES,
)
# 15. Territory — influence maps and control points
from tritium_lib.sim_engine.territory import (
    InfluenceMap, TerritoryControl, ControlPoint, StrategicValue,
)
# 30. buildings.py — building interiors, floors, room-clearing CQB
from tritium_lib.sim_engine.buildings import (
    RoomClearingEngine, RoomType,
)
# 31. economy.py — resource management, build queues, tech trees
from tritium_lib.sim_engine.economy import (
    EconomyEngine, ECONOMY_PRESETS, UNIT_COSTS, TECH_TREE,
)
# 32. cyber.py — cyber/electronic warfare
from tritium_lib.sim_engine.cyber import (
    CyberWarfareEngine, create_asset_from_preset,
)
# 33. hud.py — minimap, compass, unit roster, kill feed
from tritium_lib.sim_engine.hud import (
    HUDEngine,
)
# 34. soundtrack.py — audio cue generation for Three.js Web Audio
from tritium_lib.sim_engine.soundtrack import SoundtrackEngine
# 35. event_bus.py — centralized sim event pub/sub + timeline
from tritium_lib.sim_engine.event_bus import SimEventBus, SimEvent, SimEventType
# 36. replay.py — replay recording for playback/analysis
from tritium_lib.sim_engine.replay import ReplayRecorder
# 37. physics/collision.py — NumPy-vectorized 2D collision detection
from tritium_lib.sim_engine.physics.collision import (
    PhysicsWorld as PhysicsWorld2D, CollisionEvent as PhysCollisionEvent,
)
# 38. physics/vehicle.py — bicycle-model vehicle dynamics
from tritium_lib.sim_engine.physics.vehicle import VehiclePhysics
# 39. behavior/unit_states.py — FSM-driven unit AI state machines
from tritium_lib.sim_engine.behavior.unit_states import (
    create_fsm_for_type, unit_state_to_three_js,
)
# 40. core/inventory.py — per-unit item inventory and loadouts
from tritium_lib.sim_engine.core.inventory import (
    build_loadout, UnitInventory, select_best_weapon,
)
# 41. telemetry.py — session performance telemetry
from tritium_lib.sim_engine.telemetry import TelemetrySession
# 42. animation.py — keyframe animation, easing, interpolation buffer
from tritium_lib.sim_engine.animation import (
    AnimationLibrary, EntityAnimation, InterpolationBuffer,
)
# 43. core/npc_thinker.py — NPC thought generation via local LLM
from tritium_lib.sim_engine.core.npc_thinker import NPCThinker, NPCThought
# 44. multiplayer.py — multiplayer session, fog-of-war views, commands
from tritium_lib.sim_engine.multiplayer import (
    MultiplayerEngine, Player, PlayerRole, GameCommand, CommandType,
    TurnBasedMode, MULTIPLAYER_PRESETS, create_from_preset,
)


# ---------------------------------------------------------------------------
# Economy / Cyber / CQB helpers — called from game_tick each frame
# ---------------------------------------------------------------------------

import random as _random

_AUTO_PURCHASE_INTERVAL = 10  # ticks between auto-purchase attempts per faction
_AUTO_CYBER_INTERVAL = 50     # ticks between random cyber attack launches

# Unit pools each faction cycles through when buying
_FACTION_UNIT_POOL: dict[str, list[str]] = {
    "friendly": ["infantry", "scout", "medic", "sniper"],
    "hostile": ["infantry", "infantry", "heavy", "scout"],
}

# Cyber capability map: asset_id -> list of capability_ids to try
# Populated lazily the first time _auto_cyber_attack is called.
_CYBER_ASSET_CAPS: dict[str, list[str]] = {}


def _auto_purchase_units(gs: "GameState", tick: int) -> list[str]:
    """Attempt to purchase one unit per faction every N ticks.

    Returns list of ``"faction:template"`` strings for orders placed.
    """
    if gs.economy is None or tick % _AUTO_PURCHASE_INTERVAL != 0:
        return []
    placed: list[str] = []
    for faction, pool in _FACTION_UNIT_POOL.items():
        if faction not in gs.economy.pools:
            continue
        # Round-robin through the pool based on tick count
        idx = (tick // _AUTO_PURCHASE_INTERVAL) % len(pool)
        template = pool[idx]
        if gs.economy.purchase_unit(faction, template):
            placed.append(f"{faction}:{template}")
    return placed


def _auto_cyber_attack(gs: "GameState", tick: int, rng: _random.Random) -> list[str]:
    """Periodically launch cyber attacks between opposing factions.

    Each asset fires its first ready capability at a random enemy unit
    position.  Returns list of attack type strings that were launched.
    """
    if gs.cyber is None or gs.world is None:
        return []
    if tick % _AUTO_CYBER_INTERVAL != 0:
        return []

    launched: list[str] = []

    # Build per-alliance position pools
    friendly_positions = [
        u.position for u in gs.world.units.values()
        if u.is_alive() and u.alliance.value == "friendly"
    ]
    hostile_positions = [
        u.position for u in gs.world.units.values()
        if u.is_alive() and u.alliance.value == "hostile"
    ]

    for asset in gs.cyber.assets.values():
        # Pick a target position from the opposing side
        if asset.alliance == "friendly":
            targets = hostile_positions
        else:
            targets = friendly_positions
        if not targets:
            continue
        target_pos = rng.choice(targets)

        # Try each capability in order until one fires
        for cap in asset.capabilities:
            if not cap.is_ready:
                continue
            effect = gs.cyber.launch_attack(asset.asset_id, cap.capability_id, target_pos)
            if effect is not None:
                launched.append(effect.attack_type.value)
                break  # one attack per asset per interval

    return launched


def _trigger_cqb(gs: "GameState", tick: int, rng: _random.Random) -> list[dict]:
    """Check whether any infantry units have entered a building zone and
    trigger CQB room clearing for the first uncleared room found.

    Buildings are checked every 20 ticks to keep overhead low.
    Returns a list of CQB result dicts (may be empty).
    """
    if gs.buildings is None or gs.world is None:
        return []
    if tick % 20 != 0:
        return []

    results: list[dict] = []

    for bld_id, layout in gs.buildings.buildings.items():
        if layout.is_fully_cleared:
            continue

        # Collect infantry near this building (within 30 m of building origin)
        bx, by = layout.position
        nearby_infantry: list[str] = []
        for uid, unit in gs.world.units.items():
            if not unit.is_alive():
                continue
            if unit.alliance.value != "friendly":
                continue
            ux, uy = unit.position
            if ((ux - bx) ** 2 + (uy - by) ** 2) ** 0.5 < 30.0:
                nearby_infantry.append(uid)

        if len(nearby_infantry) < 2:
            continue  # need at least 2 to clear

        # Find first uncleared room
        uncleared = gs.buildings.get_uncleared_rooms(bld_id)
        if not uncleared:
            continue
        room = uncleared[0]

        # Place hostile occupants probabilistically (50 % chance if building is_hostile)
        if layout.is_hostile and rng.random() < 0.5:
            hostile_unit_ids = [
                uid for uid, u in gs.world.units.items()
                if u.is_alive() and u.alliance.value == "hostile"
            ]
            if hostile_unit_ids:
                h_id = rng.choice(hostile_unit_ids)
                gs.buildings.hostile_ids.add(h_id)
                if h_id not in room.occupants:
                    room.occupants.append(h_id)
                    gs.buildings._unit_locations[h_id] = (bld_id, room.room_id)

        # Enter and clear
        for uid in nearby_infantry[:3]:
            gs.buildings.enter_building(uid, bld_id, entry_point=0)

        use_flashbang = rng.random() < 0.3
        result = gs.buildings.clear_room(
            nearby_infantry[:3], room.room_id, building_id=bld_id,
            flashbang=use_flashbang,
        )
        result["tick"] = tick
        results.append(result)
        break  # one building per interval

    return results


# ---------------------------------------------------------------------------
# Unit AI System — wires behavior trees, steering, formations, combat AI
# ---------------------------------------------------------------------------

import math as _math


class UnitAISystem:
    """Per-tick AI layer that runs behavior trees, steering, and formations.

    Runs *before* world.tick() so decisions influence the world's own unit AI.
    The world AI still fires weapons and resolves damage; this layer enriches
    movement decisions and exposes AI state to the frontend.

    Attributes:
        _trees: behavior tree per unit_id
        _bt_contexts: persisted context dicts (carry cooldown state between ticks)
        _wander_velocities: last wander velocity per unit (needed for smooth wander)
        _formation_movers: FormationMover per squad_id
        _squad_formation_type: current formation per squad
        _cover_cache: last-found cover position per unit_id
        _engage_timers: how long a unit has been engaging (for flank trigger)
    """

    # Formation types assigned to squads by alliance
    _FRIENDLY_FORMATION = FormationType.WEDGE
    _HOSTILE_FORMATION = FormationType.WEDGE

    # Steering constants
    _MAX_SPEED = 5.0        # m/s — matches UnitStats default
    _SLOW_RADIUS = 8.0      # arrive deceleration radius
    _WANDER_RADIUS = 3.0
    _WANDER_DIST = 6.0
    _WANDER_JITTER = 0.5
    _ENGAGE_RANGE = 30.0    # meters, mirrors attack_range default
    _DETECT_RANGE = 50.0    # meters, mirrors detection_range default
    _COVER_SEARCH_RANGE = 40.0
    _FLANK_STALL_TIME = 8.0  # seconds before triggering flank
    _FORMATION_SPACING = 5.0

    def __init__(self) -> None:
        self._trees: dict[str, BTNode] = {}
        self._bt_contexts: dict[str, dict] = {}
        self._wander_velocities: dict[str, tuple[float, float]] = {}
        self._formation_movers: dict[str, FormationMover] = {}
        self._squad_formation_type: dict[str, FormationType] = {}
        self._cover_cache: dict[str, tuple[float, float] | None] = {}
        self._engage_timers: dict[str, float] = {}
        # AI state exported to the frame: unit_id -> {decision, formation, in_cover, ...}
        self.ai_state: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def register_unit(self, uid: str, unit_type: str, alliance: str) -> None:
        """Create a behavior tree for this unit."""
        if uid in self._trees:
            return
        if alliance == "hostile":
            self._trees[uid] = make_hostile_tree()
        elif alliance == "friendly" and unit_type in ("infantry", "heavy", "sniper", "engineer", "scout"):
            self._trees[uid] = make_friendly_tree()
        elif unit_type in ("infantry", "heavy", "sniper", "engineer", "scout"):
            self._trees[uid] = make_patrol_tree()
        else:
            self._trees[uid] = make_civilian_tree()
        self._bt_contexts[uid] = {}
        self._wander_velocities[uid] = (1.0, 0.0)
        self._engage_timers[uid] = 0.0

    def register_squad(self, squad_id: str, alliance: str,
                       waypoints: list[tuple[float, float]]) -> None:
        """Create a FormationMover for this squad if it has waypoints."""
        if squad_id in self._formation_movers:
            return
        if len(waypoints) < 2:
            return
        formation = (self._FRIENDLY_FORMATION if alliance == "friendly"
                     else self._HOSTILE_FORMATION)
        mover = FormationMover(
            waypoints=waypoints,
            formation=formation,
            spacing=self._FORMATION_SPACING,
            max_speed=self._MAX_SPEED,
        )
        self._formation_movers[squad_id] = mover
        self._squad_formation_type[squad_id] = formation

    # ------------------------------------------------------------------
    # Per-tick update
    # ------------------------------------------------------------------

    def tick(
        self,
        dt: float,
        world: Any,
    ) -> dict[str, dict]:
        """Run one AI tick across all alive units.

        Mutates unit positions and headings via steering behaviors.
        Returns the ai_state dict (unit_id -> info) for frame export.
        """
        if world is None:
            return {}

        alive_units = {uid: u for uid, u in world.units.items() if u.is_alive()}
        sim_time = world.sim_time

        # Build obstacle list from destructible structures (cover objects)
        obstacles: list[tuple[tuple[float, float], float]] = []
        if world.destruction is not None:
            for s in world.destruction.structures:
                if s.health > 0:
                    obstacles.append((s.position, max(s.size[0], s.size[1]) / 2.0))

        # Tick formation movers and collect slot targets
        formation_targets: dict[str, tuple[float, float]] = {}
        for squad_id, mover in self._formation_movers.items():
            squad = world.squads.get(squad_id)
            if squad is None:
                continue
            living_members = [m for m in squad.members if m in alive_units]
            if not living_members:
                continue
            member_positions = {m: alive_units[m].position for m in living_members}
            targets = mover.tick(dt, member_positions)
            formation_targets.update(targets)

        new_ai_state: dict[str, dict] = {}

        for uid, unit in alive_units.items():
            # Ensure registered
            if uid not in self._trees:
                self.register_unit(uid, unit.unit_type.value, unit.alliance.value)

            # Build threat list (enemies within detection range)
            threats = []
            for oid, other in alive_units.items():
                if oid == uid or other.alliance == unit.alliance:
                    continue
                d = steering_distance(unit.position, other.position)
                if d <= unit.stats.detection_range:
                    threats.append({
                        "id": oid,
                        "pos": other.position,
                        "dist": d,
                        "vel": self._wander_velocities.get(oid, (0.0, 0.0)),
                        "health": other.state.health / max(other.stats.max_health, 1.0),
                    })

            threat_in_range = any(t["dist"] <= unit.stats.attack_range for t in threats)
            health_ratio = unit.state.health / max(unit.stats.max_health, 1.0)
            ammo_ratio = (unit.state.ammo / 60.0) if unit.state.ammo >= 0 else 1.0
            allies_nearby = sum(
                1 for oid, o in alive_units.items()
                if oid != uid and o.alliance == unit.alliance
                and steering_distance(unit.position, o.position) <= 40.0
            )

            # Cover check
            in_cover = False
            if obstacles and threats:
                nearest_threat = min(threats, key=lambda t: t["dist"])
                in_cover = is_in_cover(unit.position, nearest_threat["pos"], obstacles)
            self._cover_cache.setdefault(uid, None)

            # Engage timer update
            if unit.state.status == "attacking":
                self._engage_timers[uid] = self._engage_timers.get(uid, 0.0) + dt
            else:
                self._engage_timers[uid] = 0.0

            # Build BT context
            ctx = self._bt_contexts.setdefault(uid, {})
            ctx.update({
                "time": sim_time,
                "unit": unit,
                "threats": threats,
                "threat_in_range": threat_in_range,
                "health": health_ratio,
                "retreat_threshold": 0.3,
                "waypoints": bool(unit.squad_id),
                "at_destination": False,
                "recently_threatened": bool(threats),
                # combat_ai fields
                "enemies": threats,
                "enemy_in_range": threat_in_range,
                "in_cover": in_cover,
                "ammo_ratio": ammo_ratio,
                "enemies_visible": len(threats),
                "allies_nearby": allies_nearby,
                "is_flanking": False,
                "is_suppressed": unit.state.suppression > 0.4,
                "squad_members": allies_nearby > 0,
                "engage_duration": self._engage_timers.get(uid, 0.0),
                "stall_threshold": self._FLANK_STALL_TIME,
            })

            # Tick behavior tree
            tree = self._trees[uid]
            tree.tick(ctx)
            decision = ctx.get("decision", "idle")

            # Apply steering based on decision
            new_pos, new_heading = self._apply_steering(
                uid, unit, decision, threats, obstacles, formation_targets, dt,
            )

            # Only apply steering-driven movement for certain decisions where
            # it adds genuine value (flee, wander, seek cover).  For engage/patrol
            # the world._tick_units already handles it cleanly.
            if decision in ("flee", "hide", "retreat", "wander", "seek_cover", "regroup"):
                if new_pos != unit.position:
                    unit.position = new_pos
                    unit.heading = new_heading
                    if unit.state.status not in ("attacking", "dead"):
                        unit.state.status = "moving"

            # Formation slot nudge for non-combat units
            if uid in formation_targets and decision not in (
                "engage", "flee", "retreat", "seek_cover", "hide",
            ):
                slot = formation_targets[uid]
                d_slot = steering_distance(unit.position, slot)
                if d_slot > 1.5 and unit.state.status not in ("attacking", "dead"):
                    step_vec = steering_arrive(
                        unit.position, slot, unit.effective_speed(), self._SLOW_RADIUS,
                    )
                    speed = steering_magnitude(step_vec)
                    if speed > 1e-6:
                        direction = steering_normalize(step_vec)
                        move_dist = min(speed * dt, d_slot)
                        unit.position = (
                            unit.position[0] + direction[0] * move_dist,
                            unit.position[1] + direction[1] * move_dist,
                        )
                        unit.heading = _math.atan2(direction[1], direction[0])
                        if unit.state.status not in ("attacking", "dead"):
                            unit.state.status = "moving"

            # Determine squad formation label
            formation_label = "none"
            if unit.squad_id and unit.squad_id in self._squad_formation_type:
                formation_label = self._squad_formation_type[unit.squad_id].value

            new_ai_state[uid] = {
                "decision": decision,
                "formation": formation_label,
                "in_cover": in_cover,
                "threat_count": len(threats),
                "engage_timer": round(self._engage_timers.get(uid, 0.0), 1),
                "bt_status": ctx.get("state", ""),
                "in_formation_slot": uid in formation_targets,
            }

        self.ai_state = new_ai_state
        return new_ai_state

    # ------------------------------------------------------------------
    # Steering dispatcher
    # ------------------------------------------------------------------

    def _apply_steering(
        self,
        uid: str,
        unit: Any,
        decision: str,
        threats: list[dict],
        obstacles: list[tuple[tuple[float, float], float]],
        formation_targets: dict[str, tuple[float, float]],
        dt: float,
    ) -> tuple[tuple[float, float], float]:
        """Compute new position and heading for the given decision."""
        pos = unit.position
        vel = self._wander_velocities.get(uid, (1.0, 0.0))
        speed = unit.effective_speed()

        if decision in ("flee", "hide", "retreat") and threats:
            nearest = min(threats, key=lambda t: t["dist"])
            force = steering_flee(pos, nearest["pos"], speed)
            return self._integrate(pos, vel, force, dt, uid, speed)

        if decision == "wander":
            force = steering_wander(pos, vel, self._WANDER_RADIUS,
                                    self._WANDER_DIST, self._WANDER_JITTER)
            return self._integrate(pos, vel, force, dt, uid, speed)

        if decision == "seek_cover" and obstacles and threats:
            nearest_threat = min(threats, key=lambda t: t["dist"])
            cover_pos = find_cover(pos, nearest_threat["pos"], obstacles,
                                   max_range=self._COVER_SEARCH_RANGE)
            if cover_pos is not None:
                self._cover_cache[uid] = cover_pos
                force = steering_arrive(pos, cover_pos, speed, self._SLOW_RADIUS)
                return self._integrate(pos, vel, force, dt, uid, speed)

        if decision == "regroup" and uid in formation_targets:
            slot = formation_targets[uid]
            force = steering_arrive(pos, slot, speed, self._SLOW_RADIUS)
            return self._integrate(pos, vel, force, dt, uid, speed)

        # No steering override for other decisions
        return pos, unit.heading

    def _integrate(
        self,
        pos: tuple[float, float],
        vel: tuple[float, float],
        force: tuple[float, float],
        dt: float,
        uid: str,
        max_speed: float,
    ) -> tuple[tuple[float, float], float]:
        """Apply force to position for one tick; update stored velocity."""
        # Desired velocity is the force vector (steering returns desired vel)
        new_vel = steering_truncate(force, max_speed)
        speed = steering_magnitude(new_vel)
        if speed < 1e-9:
            return pos, _math.atan2(vel[1], vel[0])
        new_pos = (
            pos[0] + new_vel[0] * dt,
            pos[1] + new_vel[1] * dt,
        )
        new_heading = _math.atan2(new_vel[1], new_vel[0])
        self._wander_velocities[uid] = new_vel
        return new_pos, new_heading

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_three_js(self) -> dict:
        """Return AI state dict for WebSocket frame inclusion."""
        return {
            "units": self.ai_state,
            "formation_movers": {
                sid: {
                    "formation": self._squad_formation_type.get(sid, FormationType.WEDGE).value,
                    "progress": mover.progress(),
                    "complete": mover.is_complete(),
                }
                for sid, mover in self._formation_movers.items()
            },
        }


# ---------------------------------------------------------------------------
# Game state container
# ---------------------------------------------------------------------------

class GameState:
    """Holds all subsystem instances for a running game."""

    def __init__(self) -> None:
        self.world: World | None = None
        self.scoring: ScoringEngine | None = None
        self.detection: DetectionEngine | None = None
        self.comms: CommsSimulator | None = None
        self.medical: MedicalEngine | None = None
        self.logistics: LogisticsEngine | None = None
        self.naval: NavalCombatEngine | None = None
        self.air_combat: AirCombatEngine | None = None
        self.engineering: EngineeringEngine | None = None
        self.asymmetric: AsymmetricEngine | None = None
        self.civilians: CivilianSimulator | None = None
        self.intel: IntelEngine | None = None
        self.diplomacy: DiplomacyEngine | None = None
        self.campaign: Campaign | None = None
        self.morale: MoraleEngine | None = None
        self.ew: EWEngine | None = None
        self.supply_routes: SupplyRouteEngine | None = None
        self.generated_map: object | None = None
        self.weather_fx: WeatherFXEngine | None = None
        self.spawner: SpawnerEngine | None = None
        self.collision: CollisionWorld | None = None
        self.artillery: ArtilleryEngine | None = None
        self.narrator: BattleNarrator | None = None
        self.narration_log: NarrationLog | None = None
        self.tactical_advisor: TacticalAdvisor | None = None
        self.abilities: AbilityEngine | None = None
        self.status_effects: StatusEffectEngine | None = None
        self.objectives: ObjectiveEngine | None = None
        self.territory: TerritoryControl | None = None
        self.influence: InfluenceMap | None = None
        self.buildings: RoomClearingEngine | None = None
        self.economy: EconomyEngine | None = None
        self.cyber: CyberWarfareEngine | None = None
        self.hud: HUDEngine | None = None
        self.soundtrack: SoundtrackEngine | None = None
        self.event_bus: SimEventBus | None = None
        self.replay: ReplayRecorder | None = None
        self.unit_ai: UnitAISystem | None = None
        # --- Newly wired orphaned modules ---
        self.physics_2d: PhysicsWorld2D | None = None
        self.vehicle_physics: dict[str, VehiclePhysics] = {}
        self.vehicle_body_ids: dict[str, int] = {}
        self.unit_body_ids: dict[str, int] = {}
        self.unit_fsms: dict[str, object] = {}  # unit_id -> StateMachine
        self.inventories: dict[str, UnitInventory] = {}
        self.telemetry: TelemetrySession | None = None
        # --- AI strategy, pathfinding, behavior profiles ---
        self.road_network: RoadNetwork | None = None
        self.walkable_area: WalkableArea | None = None
        self.strategic_ai_friendly: StrategicAI | None = None
        self.strategic_ai_hostile: StrategicAI | None = None
        self.current_plan_friendly: StrategicPlan | None = None
        self.current_plan_hostile: StrategicPlan | None = None
        self.behavior_engine: BehaviorEngine | None = None
        # --- Animation system (keyframe + interpolation buffer) ---
        self.animations: dict[str, EntityAnimation] = {}  # entity_id -> active animation
        self.anim_start_times: dict[str, float] = {}      # entity_id -> anim start time
        self.interp_buffers: dict[str, InterpolationBuffer] = {}  # entity_id -> buffer
        # --- NPC thinker (LLM-driven NPC thoughts) ---
        self.npc_thinker: NPCThinker | None = None
        self.npc_thoughts: list[dict] = []  # recent thoughts for frame export
        # --- Multiplayer engine ---
        self.multiplayer: MultiplayerEngine | None = None
        # --- Subpackage systems: effects, audio, game stats, debug ---
        self.effects: EffectsManager | None = None
        self.combat_stats: StatsTracker | None = None
        self.difficulty: DifficultyScaler | None = None
        self.debug_overlay: DebugOverlay | None = None
        self.sound_events: list[SoundEvent] = []
        self.running: bool = False
        self.paused: bool = False
        self.tick_count: int = 0
        self.preset: str = ""
        self.start_time: float = 0.0
        # Deterministic RNG shared by economy/cyber/CQB helpers
        self._rng: _random.Random = _random.Random(42)
        # Accumulated CQB results for the current frame
        self.cqb_events: list[dict] = []
        # Cached static data — map_features and building metadata that
        # don't change between ticks (avoids ~9KB re-serialization).
        self._cached_map_features: list[dict] | None = None
        self._cached_building_ids: list[str] | None = None
        self._cached_destruction: dict | None = None
        self._destruction_version: int = 0


# ---------------------------------------------------------------------------
# Game builder — exercises every module
# ---------------------------------------------------------------------------

def build_full_game(preset: str = "urban_combat") -> GameState:
    """Create a GameState that exercises every sim_engine module."""
    gs = GameState()
    gs.preset = preset

    # --- 1-3. World + Scenario + Units ---
    builder = (
        WorldBuilder()
        .set_map_size(500, 500)
        .set_seed(42)
        .set_time(hour=10.0)  # daytime — visible scene
        .enable_destruction(True)
        .enable_crowds(True)
        .enable_los(True)
        .enable_vehicles(True)
        .add_terrain_noise(octaves=4, amplitude=8.0, seed=42)
        .set_weather(Weather.CLEAR)
        # Friendly squad Alpha: 4 infantry + 1 heavy + 1 sniper + 1 medic
        # Positioned near the center for quick engagement
        .spawn_friendly_squad(
            "Alpha",
            ["infantry", "infantry", "infantry", "infantry", "heavy", "sniper", "medic"],
            (-20.0, -10.0),
            spacing=4.0,
        )
        # Friendly fire-team Bravo: 3 infantry + 1 scout flanking from south
        .spawn_friendly_squad(
            "Bravo",
            ["infantry", "infantry", "infantry", "scout"],
            (-10.0, 20.0),
            spacing=4.0,
        )
        # Hostile squad: 4 infantry + 1 heavy (balanced against 11 friendlies)
        .spawn_hostile_squad(
            "Tango",
            ["infantry"] * 4 + ["heavy"],
            (70.0, 70.0),
            spacing=4.0,
        )
        # 4. Vehicles — humvee (friendly), technical (hostile)
        .add_vehicle("humvee", "Humvee-Alpha", "friendly", (-30.0, -20.0))
        .add_vehicle("technical", "Technical-1", "hostile", (80.0, 80.0))
        # 10. Destruction — 4 buildings centered around origin
        .add_building((0.0, 0.0), (20, 15, 10), "concrete")
        .add_building((30.0, -20.0), (15, 10, 8), "concrete")
        .add_building((-25.0, 15.0), (12, 8, 6), "wood")
        .add_building((50.0, 40.0), (10, 10, 5), "brick")
        # 9. Crowd — 50 civilians in market area
        .add_crowd((40.0, 30.0), 50, 30.0, CrowdMood.CALM)
    )

    # Try to load geospatial terrain layer if cached data exists
    try:
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from pathlib import Path

        # Search common cache locations (CWD-relative and absolute)
        cache_dirs = [
            Path("data/cache/terrain"),
            Path(__file__).parent.parent.parent.parent.parent / "data" / "cache" / "terrain",
        ]
        for cache_dir in cache_dirs:
            if not cache_dir.exists():
                continue
            tl = TerrainLayer(cache_dir=cache_dir)
            for ao_id in ["demo_area", "default"]:
                if tl.load_cached(ao_id):
                    builder.load_terrain_layer(tl)
                    break
            if builder._terrain_layer is not None:
                break
    except Exception:
        pass

    gs.world = builder.build()

    print(f"[SIM] Initialized: {len(gs.world.destruction.structures if gs.world.destruction else [])} buildings, "
          f"{len(gs.world.units)} units, {len(gs.world.crowd.members) if gs.world.crowd else 0} crowd")

    # 4b. Drone (friendly quadcopter)
    drone_v = gs.world.spawn_vehicle("quadcopter", "Recon-1", "friendly", (-40.0, -30.0))
    drone_v.altitude = 50.0
    drone_ctrl = DroneController(drone_v)
    drone_ctrl.orbit((25.0, 25.0), radius=80.0, altitude=50.0)
    gs.world.drone_controllers[drone_v.vehicle_id] = drone_ctrl

    # --- 21. Scoring ---
    gs.scoring = ScoringEngine()
    for uid, unit in gs.world.units.items():
        gs.scoring.register_unit(uid, unit.name, unit.alliance.value)

    # --- 11. Detection ---
    gs.detection = DetectionEngine()
    for uid, unit in gs.world.units.items():
        sig_key = "infantry"
        if unit.unit_type == UnitType.SNIPER:
            sig_key = "sniper_ghillie"
        gs.detection.set_signature(uid, SIGNATURE_PRESETS.get(sig_key, SIGNATURE_PRESETS["infantry"]))
        gs.detection.sensors.append(Sensor(
            sensor_id=f"vis_{uid}",
            sensor_type=SensorType.VISUAL,
            position=unit.position,
            heading=unit.heading,
            fov_deg=120.0,
            range_m=unit.stats.detection_range,
            sensitivity=0.7,
            owner_id=uid,
        ))

    # --- 12. Comms ---
    gs.comms = CommsSimulator()
    gs.comms.add_channel(RadioChannel("ch_friendly", 150.0, "Squad Net", encrypted=True, alliance="friendly"))
    gs.comms.add_channel(RadioChannel("ch_hostile", 160.0, "Enemy Net", encrypted=False, alliance="hostile"))
    for uid, unit in gs.world.units.items():
        channel = "ch_friendly" if unit.alliance == Alliance.FRIENDLY else "ch_hostile"
        preset_key = "squad_radio"
        gs.comms.add_radio(Radio(
            radio_id=f"radio_{uid}",
            radio_type=RadioType.HANDHELD,
            position=unit.position,
            current_channel=channel,
            **RADIO_PRESETS[preset_key],
        ))

    # --- 13. Medical ---
    gs.medical = MedicalEngine()

    # --- 14. Logistics ---
    gs.logistics = LogisticsEngine()
    gs.logistics.add_cache(cache_from_preset(
        "forward_cache", "cache_alpha", (95.0, 95.0), alliance="friendly",
    ))

    # --- 15. Naval (patrol boat if water preset) ---
    gs.naval = NavalCombatEngine(sea_state=0.3)
    if preset in ("naval", "urban_combat"):
        patrol = create_ship(ShipClass.PATROL_BOAT, "PB-1", "friendly", (50.0, 450.0))
        gs.naval.add_ship(patrol)
        gs.naval.set_ship_controls(patrol.ship_id, throttle=0.5, rudder=0.1)

    # --- 16. Air combat ---
    gs.air_combat = AirCombatEngine()
    # No aircraft in urban_combat, but spawn AA
    gs.air_combat.add_anti_air("stinger", "aa_1", "friendly", (100.0, 110.0))

    # --- 17. Fortifications ---
    gs.engineering = EngineeringEngine()
    gs.engineering.build("bunker", (95.0, 105.0))
    gs.engineering.build("sandbag", (105.0, 95.0))

    # --- 17b. Minefield between forces ---
    from tritium_lib.sim_engine.fortifications import Mine
    for i in range(8):
        mine = Mine(
            mine_id=f"mine_{i}",
            position=(200.0 + i * 12.0, 250.0 + (i % 3) * 5.0),
            mine_type="anti_personnel",
            damage=80.0,
            blast_radius=5.0,
            trigger_radius=2.0,
            alliance="friendly",
        )
        gs.engineering.minefields.append(mine)

    # --- 18. Asymmetric ---
    gs.asymmetric = AsymmetricEngine()
    gs.asymmetric.place_trap(
        TrapType.IED_ROADSIDE, (60.0, 60.0), "hostile",
        trigger_type="proximity", damage=120.0, blast_radius=8.0,
    )

    # --- 19. Civilian ---
    gs.civilians = CivilianSimulator()
    gs.civilians.spawn_population((25.0, 25.0), 50, 40.0, with_infrastructure=True)

    # --- 20. Intel ---
    gs.intel = IntelEngine(grid_size=(100, 100), cell_size=5.0)

    # --- 22. Factions ---
    gs.diplomacy = DiplomacyEngine()
    gs.diplomacy.add_faction(Faction(
        faction_id="gov", name="Government Forces", color="#05ffa1",
        ideology="government", strength=0.7, wealth=0.6,
    ))
    gs.diplomacy.add_faction(Faction(
        faction_id="reb", name="Rebel Militia", color="#ff2a6d",
        ideology="rebel", strength=0.4, wealth=0.2,
    ))
    gs.diplomacy.add_faction(Faction(
        faction_id="civ", name="Civilian Population", color="#fcee0a",
        ideology="civilian", strength=0.0, wealth=0.3,
    ))
    gs.diplomacy.declare_war("gov", "reb")

    # --- 23. Campaign ---
    gs.campaign = Campaign.from_preset("tutorial")

    # --- 27. Morale ---
    gs.morale = MoraleEngine()
    for uid, unit in gs.world.units.items():
        is_cmd = unit.unit_type in (UnitType.HEAVY,)  # first heavy is de facto leader
        gs.morale.register_unit(
            uid,
            alliance=unit.alliance.value,
            starting_morale=75.0,
            is_commander=is_cmd,
        )

    # --- 28. Electronic Warfare ---
    gs.ew = EWEngine(rng_seed=42)
    # Hostile jammer near the center of the map
    gs.ew.place_jammer(EWJammer(
        jammer_id="hostile_jammer_1",
        position=(120.0, 120.0),
        radius=80.0,
        jammer_type=JammerType.COMMUNICATIONS,
        alliance="hostile",
    ))

    # --- 29. Supply Routes ---
    gs.supply_routes = SupplyRouteEngine()
    gs.supply_routes.add_supply_line(SupplyLine(
        line_id="main_supply",
        waypoints=[(-80.0, -80.0), (-40.0, -40.0), (-20.0, -20.0)],
        source_cache_id="cache_alpha",
        alliance="friendly",
    ))
    for uid, unit in gs.world.units.items():
        if unit.alliance == Alliance.FRIENDLY:
            gs.supply_routes.register_unit(uid, alliance="friendly")

    # 14. Procedural map (roads, forest, river for visual richness)
    mg = MapGenerator(width=500, height=500, seed=42)
    mg.add_road((-250, 0), (250, 0), width=8.0)  # main east-west road
    mg.add_road((0, -250), (0, 250), width=6.0)  # main north-south road
    mg.add_forest((150, -150), radius=60, density=0.6)  # forest NE
    mg.add_river((-250, 150), (250, 100), width=15.0)  # river south
    gs.generated_map = mg.result()

    # 15. Weather visual effects (rain, fog, lightning)
    gs.weather_fx = WeatherFXEngine()

    # 15. Wave spawner — reinforcement spawn points for both sides
    gs.spawner = SpawnerEngine(seed=42)
    gs.spawner.add_spawn_point(SpawnPoint(position=(200.0, 0.0), alliance="hostile"))
    gs.spawner.add_spawn_point(SpawnPoint(position=(0.0, 200.0), alliance="hostile"))
    gs.spawner.add_spawn_point(SpawnPoint(position=(-200.0, 0.0), alliance="friendly"))
    gs.spawner.add_spawn_point(SpawnPoint(position=(0.0, -200.0), alliance="friendly"))

    # 15. Collision world — unit and vehicle collisions
    gs.collision = CollisionWorld(cell_size=5.0)
    for uid, unit in gs.world.units.items():
        gs.collision.add(Collider(
            entity_id=uid, position=unit.position,
            collider_type=ColliderType.CIRCLE, radius=1.0,
        ))
    for vid, veh in gs.world.vehicles.items():
        gs.collision.add(Collider(
            entity_id=vid, position=veh.position,
            collider_type=ColliderType.CIRCLE, radius=2.5,
        ))

    # 15. Artillery — friendly mortar battery
    gs.artillery = ArtilleryEngine()
    mortar_tmpl = ARTILLERY_TEMPLATES[ArtilleryType.MORTAR_60MM]
    gs.artillery.add_piece(ArtilleryPiece(
        piece_id="mortar_1", artillery_type=ArtilleryType.MORTAR_60MM,
        alliance="friendly", position=(-30.0, -30.0), heading=0.7,
        min_range=mortar_tmpl["min_range"], max_range=mortar_tmpl["max_range"],
        damage=mortar_tmpl["damage"], blast_radius=mortar_tmpl["blast_radius"],
        reload_time=mortar_tmpl["reload_time"], ammo=mortar_tmpl["max_ammo"],
        max_ammo=mortar_tmpl["max_ammo"], accuracy_cep=mortar_tmpl["accuracy_cep"],
        crew=mortar_tmpl["crew"],
    ))
    # Add a second mortar (hostile) for visual variety
    mortar_tmpl_81 = ARTILLERY_TEMPLATES[ArtilleryType.MORTAR_81MM]
    gs.artillery.add_piece(ArtilleryPiece(
        piece_id="mortar_hostile_1", artillery_type=ArtilleryType.MORTAR_81MM,
        alliance="hostile", position=(100.0, 90.0), heading=3.9,
        min_range=mortar_tmpl_81["min_range"], max_range=mortar_tmpl_81["max_range"],
        damage=mortar_tmpl_81["damage"], blast_radius=mortar_tmpl_81["blast_radius"],
        reload_time=mortar_tmpl_81["reload_time"], ammo=mortar_tmpl_81["max_ammo"],
        max_ammo=mortar_tmpl_81["max_ammo"], accuracy_cep=mortar_tmpl_81["accuracy_cep"],
        crew=mortar_tmpl_81["crew"],
    ))
    # Queue initial fire missions
    try:
        gs.artillery.request_fire_mission(
            "mortar_1", (70.0, 70.0), mission_type="barrage", rounds=5, interval=3.0,
        )
        gs.artillery.request_fire_mission(
            "mortar_hostile_1", (-20.0, -20.0), mission_type="area", rounds=3, interval=4.0,
        )
    except ValueError:
        pass  # out of range in some presets

    # Commander / narrator — "Mad Dog" personality for demo
    gs.narrator = BattleNarrator(personality=PERSONALITIES["mad_dog"])
    gs.narration_log = NarrationLog(max_events=200)
    gs.tactical_advisor = TacticalAdvisor(personality=PERSONALITIES["mad_dog"])
    # Seed narration log with mission start message
    hostile_count = sum(1 for u in gs.world.units.values() if u.alliance.value == "hostile")
    gs.narration_log.add(gs.narrator.narrate_wave_start(1, hostile_count))

    # 15. Abilities — grant special abilities to units
    gs.abilities = AbilityEngine()
    # Grant smoke grenade to snipers, heal to medics
    for uid, unit in gs.world.units.items():
        for ability in ABILITIES.values():
            if unit.unit_type.value in getattr(ability, 'allowed_types', []) or not hasattr(ability, 'allowed_types'):
                try:
                    gs.abilities.grant_ability(uid, ability)
                except Exception:
                    pass

    # 15. Status effects
    gs.status_effects = StatusEffectEngine()

    # 16. Objectives — assault mission chain
    gs.objectives = ObjectiveEngine()
    gs.objectives.load_template("assault_chain")

    # 15. Territory — influence map and control points
    # Map is 500×500 world units; cell_size=10 → 50×50 grid (2500 cells).
    # Previously width/height were set to 500 (world units not cell count) — 250K cells!
    gs.influence = InfluenceMap(width=50, height=50, cell_size=10.0)
    gs.territory = TerritoryControl()
    gs.territory.add_control_point(ControlPoint(
        point_id="cp_center", name="Central Objective",
        position=(25.0, 25.0), capture_radius=30.0,
    ))
    gs.territory.add_control_point(ControlPoint(
        point_id="cp_east", name="Eastern Approach",
        position=(100.0, 25.0), capture_radius=25.0,
    ))

    # --- 30. Buildings — procedural interior layouts for CQB ---
    gs.buildings = RoomClearingEngine()
    # Generate a house and an office block near the action
    gs.buildings.generate_layout(
        floors=1, rooms_per_floor=4,
        building_pos=(5.0, 5.0), template="house",
    )
    gs.buildings.generate_layout(
        floors=2, rooms_per_floor=4,
        building_pos=(60.0, 60.0), template="compound",
    )

    # --- 31. Economy — resource management for both factions ---
    gs.economy = EconomyEngine()
    gs.economy.setup_faction("friendly", ECONOMY_PRESETS["standard"])
    gs.economy.setup_faction("hostile", ECONOMY_PRESETS["insurgent"])
    gs.economy.register_unit_costs(UNIT_COSTS)
    import copy as _copy
    gs.economy.register_tech_tree("friendly", _copy.deepcopy(TECH_TREE))
    gs.economy.register_tech_tree("hostile", _copy.deepcopy(TECH_TREE))

    # --- 32. Cyber warfare — jammer + GPS spoofer ---
    gs.cyber = CyberWarfareEngine(rng_seed=42)
    # Friendly SIGINT post in the rear area
    sigint = create_asset_from_preset(
        "sigint_post", "sigint_friendly_1", (-50.0, -50.0), "friendly",
    )
    gs.cyber.deploy_asset(sigint)
    # Hostile GPS spoofer threatening drone routes
    spoofer = create_asset_from_preset(
        "gps_spoofer", "gps_spoofer_hostile_1", (130.0, 90.0), "hostile",
    )
    gs.cyber.deploy_asset(spoofer)
    # Activate the spoofer immediately so effects show in first frame
    gs.cyber.launch_attack(
        "gps_spoofer_hostile_1",
        "gps_spoofer_hostile_1_gps_spoof",
        (25.0, 25.0),
    )

    # --- 33. HUD — minimap, compass, unit roster, notifications ---
    gs.hud = HUDEngine(map_width=500.0, map_height=500.0, minimap_size=200)
    gs.hud.add_notification("Mission started — ALPHA team deploy", priority="high", icon="alert")

    # --- 34. Soundtrack — audio cue generation ---
    gs.soundtrack = SoundtrackEngine()

    # --- 35. Event bus — centralized event aggregation ---
    gs.event_bus = SimEventBus(max_log=5000)
    # Seed with a game-start event
    gs.event_bus.emit(SimEvent(
        event_type=SimEventType.WAVE_STARTED,
        tick=0,
        time=0.0,
        data={"wave": 1, "preset": preset},
    ))

    # --- 36. Replay recorder — frame-by-frame capture ---
    gs.replay = ReplayRecorder(
        metadata={"preset": preset, "units": len(gs.world.units)},
        max_frames=6000,  # ~10 min at 10fps
    )

    # --- AI behavior system — behavior trees, steering, formations, combat AI ---
    gs.unit_ai = UnitAISystem()
    for uid, unit in gs.world.units.items():
        gs.unit_ai.register_unit(uid, unit.unit_type.value, unit.alliance.value)
    # Register squads with patrol waypoints for formation movement
    for sid, squad in gs.world.squads.items():
        if not squad.members:
            continue
        # Determine alliance from first member
        first_uid = squad.members[0]
        first_unit = gs.world.units.get(first_uid)
        if first_unit is None:
            continue
        alliance = first_unit.alliance.value
        # Build patrol waypoints: squad start pos -> center -> opposite side
        leader_pos = first_unit.position
        center = (25.0, 25.0)
        opposite = (-leader_pos[0], -leader_pos[1])
        waypoints = [leader_pos, center, opposite, center, leader_pos]
        gs.unit_ai.register_squad(sid, alliance, waypoints)

    # --- Road network + walkable area (pathfinding) ---
    gs.road_network = RoadNetwork()
    # Build road graph from the procedural map roads.
    # Split long roads at intersections so A* can route through junctions.
    # Main E-W road split at origin
    gs.road_network.add_road((-250.0, 0.0), (0.0, 0.0), speed_limit=13.4)
    gs.road_network.add_road((0.0, 0.0), (250.0, 0.0), speed_limit=13.4)
    # Main N-S road split at origin
    gs.road_network.add_road((0.0, -250.0), (0.0, 0.0), speed_limit=11.2)
    gs.road_network.add_road((0.0, 0.0), (0.0, 250.0), speed_limit=11.2)
    # Cross-roads near objectives
    gs.road_network.add_road((-100.0, 0.0), (0.0, 100.0), speed_limit=8.9)
    gs.road_network.add_road((0.0, 0.0), (70.0, 70.0), speed_limit=8.9)
    gs.road_network.add_road((0.0, 0.0), (-25.0, 15.0), speed_limit=6.7)
    gs.road_network.add_road((70.0, 70.0), (100.0, 25.0), speed_limit=8.9)

    # Walkable area with building obstacles for pedestrian navigation
    gs.walkable_area = WalkableArea(bounds=((-250.0, -250.0), (250.0, 250.0)))
    if gs.world.destruction is not None:
        for structure in gs.world.destruction.structures:
            if structure.health > 0:
                sx, sy = structure.position
                hw, hh = structure.size[0] / 2.0, structure.size[1] / 2.0
                gs.walkable_area.add_obstacle([
                    (sx - hw, sy - hh), (sx + hw, sy - hh),
                    (sx + hw, sy + hh), (sx - hw, sy + hh),
                ])

    # --- Strategic AI — faction-level planning ---
    gs.strategic_ai_friendly = StrategicAI(profile="balanced")
    gs.strategic_ai_hostile = StrategicAI(profile="aggressive")

    # --- Behavior profiles — per-unit personality archetypes ---
    gs.behavior_engine = BehaviorEngine(profiles=BEHAVIOR_PROFILES)
    # Assign profiles based on unit type and alliance
    _UNIT_PROFILE_MAP = {
        ("sniper", "friendly"): "sniper_patient",
        ("sniper", "hostile"): "sniper_patient",
        ("medic", "friendly"): "medic_angel",
        ("heavy", "friendly"): "veteran_steady",
        ("heavy", "hostile"): "berserker",
        ("scout", "friendly"): "scout_ghost",
        ("scout", "hostile"): "guerrilla",
        ("engineer", "friendly"): "engineer_builder",
        ("infantry", "friendly"): "elite_operator",
        ("infantry", "hostile"): "conscript",
    }
    for uid, unit in gs.world.units.items():
        key = (unit.unit_type.value, unit.alliance.value)
        profile_id = _UNIT_PROFILE_MAP.get(key, "veteran_steady")
        gs.behavior_engine.assign_profile(uid, profile_id)

    # --- Effects (particle system) — muzzle flashes, explosions, smoke ---
    gs.effects = EffectsManager(max_emitters=256)
    # Seed with a few persistent smoke emitters at building positions
    for bld_id, layout in gs.buildings.buildings.items():
        if layout.is_hostile:
            gs.effects.add(fx_smoke(layout.position, duration=30.0))

    # --- Combat stats (StatsTracker from game subpackage) ---
    gs.combat_stats = StatsTracker()
    for uid, unit in gs.world.units.items():
        gs.combat_stats.register_unit(
            uid, unit.name, unit.alliance.value, unit.unit_type.value,
        )
    # Start wave tracking
    hostile_start = sum(1 for u in gs.world.units.values() if u.alliance.value == "hostile")
    gs.combat_stats.on_wave_start(1, "initial_engagement", hostile_start)

    # --- Difficulty scaler ---
    gs.difficulty = DifficultyScaler()

    # --- Debug overlay (collects all debug streams) ---
    gs.debug_overlay = DebugOverlay()
    gs.debug_overlay.register(gs.effects.debug)

    # --- 37. Physics 2D — NumPy collision world for unit/vehicle bodies ---
    gs.physics_2d = PhysicsWorld2D(max_bodies=256, cell_size=8.0)
    for uid, unit in gs.world.units.items():
        body_id = gs.physics_2d.add_body(
            pos=unit.position, vel=(0.0, 0.0),
            mass=80.0, radius=0.8, restitution=0.3, static=False,
        )
        gs.unit_body_ids[uid] = body_id

    # --- 38. Vehicle physics — bicycle-model dynamics per vehicle ---
    for vid, veh in gs.world.vehicles.items():
        vp = VehiclePhysics(
            mass=veh.mass if hasattr(veh, "mass") else 1500.0,
            max_speed=veh.max_speed if hasattr(veh, "max_speed") else 15.0,
        )
        vp.position[:] = veh.position
        vp.heading = veh.heading
        gs.vehicle_physics[vid] = vp
        body_id = gs.physics_2d.add_body(
            pos=veh.position, vel=(0.0, 0.0),
            mass=vp.mass, radius=2.5, restitution=0.2, static=False,
        )
        gs.vehicle_body_ids[vid] = body_id

    # --- 39. Unit FSMs — state machines per combatant unit type ---
    # Map sim_engine unit types to FSM asset types (person → hostile FSM,
    # infantry/sniper/medic/heavy/scout → rover FSM for mobile ground units,
    # drone → drone FSM)
    _FSM_TYPE_MAP: dict[str, str] = {
        "infantry": "rover", "heavy": "rover", "sniper": "rover",
        "medic": "rover", "scout": "rover", "engineer": "rover",
    }
    for uid, unit in gs.world.units.items():
        raw_type = unit.unit_type.value
        alliance = unit.alliance.value
        # Hostile ground units → hostile FSM
        if alliance == "hostile" and raw_type in _FSM_TYPE_MAP:
            from tritium_lib.sim_engine.behavior.unit_states import create_hostile_fsm
            fsm = create_hostile_fsm()
        else:
            mapped_type = _FSM_TYPE_MAP.get(raw_type, raw_type)
            fsm = create_fsm_for_type(mapped_type, alliance)
        if fsm is not None:
            gs.unit_fsms[uid] = fsm

    # --- 40. Inventories — per-unit loadouts (armor, weapons, devices) ---
    # Map sim_engine unit types to inventory asset types.
    # Person-class combatants (infantry, sniper, etc.) → "person" for loadout
    _INV_TYPE_MAP: dict[str, str] = {
        "infantry": "person", "heavy": "person", "sniper": "person",
        "medic": "person", "scout": "person", "engineer": "person",
    }
    for uid, unit in gs.world.units.items():
        inv_type = _INV_TYPE_MAP.get(unit.unit_type.value, unit.unit_type.value)
        inv = build_loadout(uid, inv_type, unit.alliance.value)
        gs.inventories[uid] = inv

    # --- 41. Telemetry — session performance recording ---
    gs.telemetry = TelemetrySession(
        metadata={"preset": preset, "units": len(gs.world.units)},
    )

    # --- 42. Animation system — pre-built animations + interpolation buffers ---
    # Assign looping animations to alive units based on their type/state:
    #   infantry/scouts → walk_cycle, vehicles → vehicle_bounce,
    #   drones/helicopters → helicopter_hover
    # Death animations are triggered in game_tick when units die.
    _ANIM_TYPE_MAP: dict[str, str] = {
        "infantry": "walk_cycle",
        "heavy": "walk_cycle",
        "sniper": "walk_cycle",
        "medic": "walk_cycle",
        "scout": "run_cycle",
        "engineer": "walk_cycle",
    }
    for uid, unit in gs.world.units.items():
        anim_name = _ANIM_TYPE_MAP.get(unit.unit_type.value, "walk_cycle")
        anim = AnimationLibrary.get(anim_name)
        if anim is not None:
            gs.animations[uid] = anim
            gs.anim_start_times[uid] = 0.0
        # InterpolationBuffer for smooth network-style position updates
        gs.interp_buffers[uid] = InterpolationBuffer(delay=0.1, max_samples=20)
    for vid, veh in gs.world.vehicles.items():
        veh_class = getattr(veh, "vehicle_class", None)
        if veh_class and veh_class.value in ("helicopter", "quadcopter"):
            anim = AnimationLibrary.get("helicopter_hover")
        else:
            anim = AnimationLibrary.get("vehicle_bounce")
        if anim is not None:
            gs.animations[vid] = anim
            gs.anim_start_times[vid] = 0.0

    # --- 43. NPC thinker — LLM-driven NPC inner thoughts (graceful degradation) ---
    # Creates the thinker but does NOT require a running LLM server.
    # If no server is found, thoughts are silently skipped.
    gs.npc_thinker = NPCThinker(cooldown_s=60.0, max_concurrent=2)

    # --- 44. Multiplayer engine — session management with fog-of-war ---
    # Set up a co-op vs AI preset: human player controls blue, AI controls red.
    gs.multiplayer = MultiplayerEngine()
    p1 = gs.multiplayer.add_player("player_1", "Commander", "friendly", PlayerRole.COMMANDER)
    p1.ready = True
    # AI player for hostile faction
    p_ai = gs.multiplayer.add_player("ai_red", "AI Hostiles", "hostile", PlayerRole.AI_CONTROLLED)
    p_ai.ready = True
    # Assign units to respective players
    friendly_uids = [uid for uid, u in gs.world.units.items() if u.alliance.value == "friendly"]
    hostile_uids = [uid for uid, u in gs.world.units.items() if u.alliance.value == "hostile"]
    gs.multiplayer.assign_units("player_1", friendly_uids)
    gs.multiplayer.assign_units("ai_red", hostile_uids)
    # Assign squads
    for sid, squad in gs.world.squads.items():
        if not squad.members:
            continue
        first_unit = gs.world.units.get(squad.members[0])
        if first_unit is None:
            continue
        if first_unit.alliance.value == "friendly":
            gs.multiplayer.assign_squads("player_1", [sid])
        else:
            gs.multiplayer.assign_squads("ai_red", [sid])

    gs.start_time = time.time()
    return gs


# ---------------------------------------------------------------------------
# Tick — advance all subsystems
# ---------------------------------------------------------------------------

def game_tick(gs: GameState, dt: float = 0.1) -> dict[str, Any]:
    """Advance all subsystems by dt, return a composite frame."""
    if gs.world is None:
        return {"error": "no_game"}

    gs.tick_count += 1

    # 0. AI behavior systems — run before world tick so decisions influence movement
    if gs.unit_ai is not None:
        gs.unit_ai.tick(dt, gs.world)

    # 1. World tick (units, squads, vehicles, projectiles, destruction, crowd)
    frame = gs.world.tick(dt)

    # Ensure destruction/building data is always in frame for Three.js
    if gs.world.destruction is not None and "destruction" not in frame:
        frame["destruction"] = gs.world.destruction.to_three_js()

    # 2. Detection tick
    if gs.detection is not None:
        entity_positions: dict[str, tuple[float, float]] = {}
        for uid, u in gs.world.units.items():
            if u.is_alive():
                entity_positions[uid] = u.position
                # Update sensor positions
                for s in gs.detection.sensors:
                    if s.owner_id == uid:
                        s.position = u.position
                        s.heading = u.heading
        env_snap = gs.world.environment.snapshot()
        det_env = {
            "weather": env_snap.get("weather", "clear"),
            "is_night": env_snap.get("hour", 12.0) < 6.0 or env_snap.get("hour", 12.0) > 20.0,
        }
        gs.detection.tick(dt, entity_positions, det_env)
        frame["detection"] = gs.detection.to_three_js()

    # 3. Comms tick
    if gs.comms is not None:
        for rid, radio in gs.comms.radios.items():
            uid = rid.replace("radio_", "")
            unit = gs.world.units.get(uid)
            if unit and unit.is_alive():
                radio.position = unit.position
        gs.comms.tick(dt)
        frame["comms"] = gs.comms.to_three_js()

    # 4. Medical tick
    if gs.medical is not None:
        med_events = gs.medical.tick(dt)
        frame["medical"] = gs.medical.to_three_js()
        frame["medical_events"] = med_events

    # 5. Logistics tick
    if gs.logistics is not None:
        unit_positions = {uid: u.position for uid, u in gs.world.units.items() if u.is_alive()}
        unit_alliances = {uid: u.alliance.value for uid, u in gs.world.units.items() if u.is_alive()}
        gs.logistics.tick(dt, unit_positions, unit_alliances)
        frame["logistics"] = gs.logistics.to_three_js()

    # 6. Naval tick
    if gs.naval is not None and gs.naval.ships:
        naval_result = gs.naval.tick(dt)
        frame["naval"] = gs.naval.to_three_js()
        frame["naval_events"] = naval_result.get("events", [])

    # 7. Air combat tick
    if gs.air_combat is not None:
        air_result = gs.air_combat.tick(dt)
        frame["air_combat"] = gs.air_combat.to_three_js()

    # 8. Intel tick
    if gs.intel is not None:
        observer_data: dict[str, list[tuple[tuple[float, float], float]]] = {"friendly": [], "hostile": []}
        for uid, u in gs.world.units.items():
            if u.is_alive():
                alliance_key = u.alliance.value
                if alliance_key in observer_data:
                    observer_data[alliance_key].append((u.position, u.stats.detection_range))
        entity_map = {uid: u.position for uid, u in gs.world.units.items() if u.is_alive()}
        gs.intel.tick(dt, observer_data=observer_data, entities=entity_map)
        # Export fog-of-war for the friendly alliance
        frame["intel"] = gs.intel.to_three_js("friendly")

    # 9. Scoring — record kills from world events
    if gs.scoring is not None:
        for ev in frame.get("events", []):
            if ev.get("type") == "unit_killed":
                killer = ev.get("source_id", "")
                victim = ev.get("target_id", "")
                if killer and victim:
                    gs.scoring.record_kill(killer, victim)
        gs.scoring.tick(dt)

    # 10. Diplomacy tick
    if gs.diplomacy is not None:
        gs.diplomacy.tick(dt)

    # 11. Morale tick
    if gs.morale is not None:
        unit_positions_morale = {
            uid: u.position for uid, u in gs.world.units.items() if u.is_alive()
        }
        # Feed kill events as morale events
        morale_events: list[MoraleEvent] = []
        for ev in frame.get("events", []):
            if ev.get("type") == "unit_killed":
                killer = ev.get("source_id", "")
                victim = ev.get("target_id", "")
                if killer:
                    morale_events.append(MoraleEvent(
                        unit_id=killer,
                        event_type=MoraleEventType.ENEMY_KILLED,
                    ))
                if victim:
                    gs.morale.mark_dead(victim)
                    # Nearby allies see ally die
                    victim_unit = gs.world.units.get(victim)
                    if victim_unit:
                        for uid, u in gs.world.units.items():
                            if (u.is_alive()
                                    and u.alliance == victim_unit.alliance
                                    and uid != victim):
                                from tritium_lib.sim_engine.ai.steering import distance as _dist
                                if _dist(u.position, victim_unit.position) < 80.0:
                                    morale_events.append(MoraleEvent(
                                        unit_id=uid,
                                        event_type=MoraleEventType.ALLY_KILLED,
                                    ))
        notifications = gs.morale.tick(dt, unit_positions=unit_positions_morale, events=morale_events)
        frame["morale"] = gs.morale.to_three_js()
        frame["morale_events"] = notifications

    # 12. Electronic Warfare tick
    if gs.ew is not None:
        ew_result = gs.ew.tick(dt)
        frame["electronic_warfare"] = gs.ew.to_three_js()
        frame["ew_events"] = ew_result.get("events", [])

    # 13. Supply Routes tick
    if gs.supply_routes is not None:
        unit_pos_supply = {
            uid: u.position for uid, u in gs.world.units.items() if u.is_alive()
        }
        enemy_pos_supply = {
            uid: u.position for uid, u in gs.world.units.items()
            if u.is_alive() and u.alliance == Alliance.HOSTILE
        }
        sr_result = gs.supply_routes.tick(dt, unit_positions=unit_pos_supply, enemy_positions=enemy_pos_supply)
        frame["supply_routes"] = gs.supply_routes.to_three_js()
        frame["supply_warnings"] = sr_result.get("warnings", [])

    # 14. Weather FX tick
    if gs.weather_fx is not None:
        env_snap = gs.world.environment.snapshot()
        wx_state = {
            "weather": env_snap.get("weather", "clear"),
            "intensity": env_snap.get("intensity", 0.5),
            "wind_speed": env_snap.get("wind_speed", 0.0),
            "wind_direction": env_snap.get("wind_direction", 0.0),
        }
        time_state = {"hour": env_snap.get("hour", 12.0)}
        wx_fx = gs.weather_fx.tick(dt, wx_state, time_state)
        if wx_fx:
            # Strip large position/velocity arrays — client generates rain
            # locally from params.  Only send drop_count and metadata.
            for key in ("rain", "snow"):
                if wx_fx.get(key) and isinstance(wx_fx[key], dict):
                    wx_fx[key] = {
                        k: v for k, v in wx_fx[key].items()
                        if k not in ("positions", "velocities", "sizes",
                                     "rotations", "splashes")
                    }
            # Strip fog grid data (large)
            if wx_fx.get("fog") and isinstance(wx_fx["fog"], dict):
                wx_fx["fog"] = {
                    k: v for k, v in wx_fx["fog"].items()
                    if k not in ("grid",)
                }
            frame["weather_fx"] = wx_fx

    # 15. Spawner tick
    if gs.spawner is not None:
        spawn_events = gs.spawner.tick(dt)
        frame["spawner"] = gs.spawner.to_three_js()

    # 15. Collision tick
    if gs.collision is not None:
        for uid, u in gs.world.units.items():
            if u.is_alive():
                gs.collision.update(uid, position=u.position)
        for vid, v in gs.world.vehicles.items():
            gs.collision.update(vid, position=v.position)
        collisions = gs.collision.check_all()
        if collisions:
            frame["collisions"] = [
                {"a": c.entity_a, "b": c.entity_b, "overlap": c.overlap}
                for c in collisions[:20]  # cap for frame size
            ]

    # 15. Artillery tick
    if gs.artillery is not None:
        arty_events = gs.artillery.tick(dt)
        frame["artillery"] = gs.artillery.to_three_js()

    # Commander / Narration — generate commentary from world events
    if gs.narrator is not None and gs.narration_log is not None:
        gs.narrator.set_tick(gs.tick_count, gs.world.sim_time)
        for ev in frame.get("events", []):
            etype = ev.get("type", "")
            if etype == "unit_killed":
                killer_id = ev.get("source_id", "unknown")
                victim_id = ev.get("target_id", "unknown")
                narr_ev = gs.narrator.narrate_kill(killer_id, victim_id)
                gs.narration_log.add(narr_ev)
            elif etype == "unit_hit":
                attacker_id = ev.get("source_id", "unknown")
                target_id = ev.get("target_id", "unknown")
                narr_ev = gs.narrator.narrate_engagement(attacker_id, target_id, "hit")
                gs.narration_log.add(narr_ev)
            elif etype == "explosion":
                pos = ev.get("position", (0.0, 0.0))
                radius = ev.get("radius", 5.0)
                casualties = ev.get("casualties", 0)
                narr_ev = gs.narrator.narrate_explosion(pos, radius, casualties)
                gs.narration_log.add(narr_ev)
        # Periodic tactical assessments (every 50 ticks)
        if gs.tick_count % 50 == 1 and gs.tactical_advisor is not None:
            friendly_alive = sum(
                1 for u in gs.world.units.values()
                if u.is_alive() and u.alliance.value == "friendly"
            )
            hostile_alive = sum(
                1 for u in gs.world.units.values()
                if u.is_alive() and u.alliance.value == "hostile"
            )
            friendly_dead = sum(
                1 for u in gs.world.units.values()
                if not u.is_alive() and u.alliance.value == "friendly"
            )
            recs = gs.tactical_advisor.assess_situation({
                "friendly_count": friendly_alive,
                "hostile_count": hostile_alive,
                "friendly_casualties": friendly_dead,
                "ammo_level": 0.7,
                "visibility": 0.8,
            })
            for rec_text in recs[:2]:  # max 2 per assessment
                gs.narration_log.add(NarrationEvent(
                    tick=gs.tick_count,
                    time=gs.world.sim_time,
                    category="tactical",
                    priority=2,
                    text=rec_text,
                    voice="commander",
                ))
        frame["narration"] = gs.narration_log.to_three_js()

    # 15. Abilities tick
    if gs.abilities is not None:
        ability_events = gs.abilities.tick(dt)
        # Ability data split: static catalog once on tick 1, dynamic state every tick.
        # Static fields (description, icon, color, type, cost, range, radius) are
        # per-ability-id and never change — send them once as ability_catalog.
        # Dynamic fields (current_cooldown, progress, ready, toggled_on, is_channeling)
        # are per-unit and change every tick — send compactly every frame.
        _static_keys = {"description", "icon", "color", "ability_type", "target_type",
                        "cooldown", "cost", "range", "radius", "name"}
        if gs.tick_count <= 1:
            # Build catalog: ability_id -> static fields (dedup across units)
            catalog: dict[str, dict[str, Any]] = {}
            for uid in gs.world.units:
                for ab_dict in gs.abilities.to_three_js(uid):
                    ab_id = ab_dict["ability_id"]
                    if ab_id not in catalog:
                        catalog[ab_id] = {k: v for k, v in ab_dict.items() if k in _static_keys}
                        catalog[ab_id]["ability_id"] = ab_id
            if catalog:
                frame["ability_catalog"] = catalog
        # Send compact dynamic state only for abilities that are not idle-ready.
        # An ability is "idle-ready" when: ready=True, progress=1.0, toggled_on=False,
        # is_channeling=False, current_cooldown=0.  These don't need per-tick updates.
        ability_state: dict[str, list[dict[str, Any]]] = {}
        for uid in gs.world.units:
            active = []
            for ab_dict in gs.abilities.to_three_js(uid):
                if (not ab_dict.get("ready", True)
                        or ab_dict.get("current_cooldown", 0) > 0
                        or ab_dict.get("toggled_on", False)
                        or ab_dict.get("is_channeling", False)):
                    active.append({k: v for k, v in ab_dict.items() if k not in _static_keys})
            if active:
                ability_state[uid] = active
        if ability_state:
            frame["abilities"] = ability_state

    # 15. Status effects tick
    if gs.status_effects is not None:
        se_events = gs.status_effects.tick(dt)
        se_fx = {}
        for uid in gs.world.units:
            fx = gs.status_effects.to_three_js(uid)
            if fx:
                se_fx[uid] = fx
        if se_fx:
            frame["status_effects"] = se_fx

    # 16. Objectives tick
    if gs.objectives is not None:
        world_state = {
            "friendly_positions": {
                uid: u.position for uid, u in gs.world.units.items()
                if u.is_alive() and u.alliance == Alliance.FRIENDLY
            },
            "hostile_positions": {
                uid: u.position for uid, u in gs.world.units.items()
                if u.is_alive() and u.alliance == Alliance.HOSTILE
            },
            "friendly_count": sum(
                1 for u in gs.world.units.values()
                if u.is_alive() and u.alliance == Alliance.FRIENDLY
            ),
            "hostile_count": sum(
                1 for u in gs.world.units.values()
                if u.is_alive() and u.alliance == Alliance.HOSTILE
            ),
        }
        obj_events = gs.objectives.tick(dt, world_state)
        frame["objectives"] = gs.objectives.to_three_js()

    # 15. Territory/Influence tick
    if gs.influence is not None:
        inf_units = {
            uid: (u.position, "friendly" if u.alliance == Alliance.FRIENDLY else "hostile")
            for uid, u in gs.world.units.items() if u.is_alive()
        }
        gs.influence.tick(dt, inf_units)
        # Only send summary, not the full 250K-cell grid (saves ~2.4MB/frame)
        infl_data = gs.influence.to_three_js()
        frame["influence"] = {
            "width": infl_data.get("width"),
            "height": infl_data.get("height"),
            "cell_size": infl_data.get("cell_size"),
            "frontlines": infl_data.get("frontlines", {}),
            # Omit heatmaps — too large for per-frame WebSocket
        }
    if gs.territory is not None:
        terr_units = {
            uid: (u.position, "friendly" if u.alliance == Alliance.FRIENDLY else "hostile")
            for uid, u in gs.world.units.items() if u.is_alive()
        }
        terr_events = gs.territory.tick(dt, terr_units)
        frame["territory"] = gs.territory.to_dict()

    # 30. Buildings tick — visibility/lighting + CQB room-clearing triggers
    if gs.buildings is not None:
        gs.buildings.tick(dt)
        # Trigger CQB when friendly infantry are near building zones
        cqb_results = _trigger_cqb(gs, gs.tick_count, gs._rng)
        if cqb_results:
            gs.cqb_events.extend(cqb_results)
        # Cache building IDs (layout doesn't change mid-game)
        if gs._cached_building_ids is None:
            gs._cached_building_ids = list(gs.buildings.buildings.keys())
        bld_ids = gs._cached_building_ids
        if bld_ids:
            bld_data = gs.buildings.to_three_js(bld_ids[0])
            # Keep frame small: strip room occupants lists, keep summary
            frame["buildings"] = {
                "count": len(bld_ids),
                "building_ids": bld_ids,
                "sample": {
                    "building_id": bld_data.get("building_id"),
                    "total_floors": bld_data.get("total_floors"),
                    "cleared_rooms": bld_data.get("cleared_rooms"),
                    "total_rooms": bld_data.get("total_rooms"),
                    "is_fully_cleared": bld_data.get("is_fully_cleared"),
                },
            }
        # Include recent CQB events (last 5) in frame
        if gs.cqb_events:
            frame["cqb"] = gs.cqb_events[-5:]

    # 31. Economy tick — income, upkeep, build queues, auto-purchase
    if gs.economy is not None:
        completed_units = gs.economy.tick(dt)
        # Auto-purchase: each faction queues a unit every N ticks
        auto_placed = _auto_purchase_units(gs, gs.tick_count)
        if auto_placed:
            completed_units = completed_units + [f"queued:{p}" for p in auto_placed]
        # Export both faction economies
        frame["economy"] = {
            "friendly": gs.economy.to_three_js("friendly"),
            "hostile": gs.economy.to_three_js("hostile"),
            "completed_units": completed_units,
        }

    # 32. Cyber warfare tick — effects + periodic auto-attacks
    if gs.cyber is not None:
        unit_pos_cyber = {
            uid: u.position for uid, u in gs.world.units.items() if u.is_alive()
        }
        drone_pos_cyber = {
            vid: v.position for vid, v in gs.world.vehicles.items()
            if not v.is_destroyed and getattr(v, "altitude", 0.0) > 0.0
        }
        cyber_events = gs.cyber.tick(dt, unit_pos_cyber, drone_pos_cyber)
        # Periodically launch new attacks from each asset
        auto_attacks = _auto_cyber_attack(gs, gs.tick_count, gs._rng)
        frame["cyber"] = gs.cyber.to_three_js()
        frame["cyber"]["auto_attacks_this_tick"] = auto_attacks
        all_cyber_events = cyber_events + gs.cyber.drain_event_log()
        if all_cyber_events:
            frame["cyber_events"] = all_cyber_events[:20]  # cap for frame size

    # 33. HUD tick
    if gs.hud is not None:
        # Assemble world_state dict for HUD from existing frame data
        hud_units = [
            {
                "unit_id": uid,
                "name": u.name,
                "alliance": u.alliance.value,
                "position": u.position,
                "health": u.state.health,
                "max_health": u.stats.max_health,
                "is_alive": u.is_alive(),
                "status": u.state.status if isinstance(u.state.status, str) else str(u.state.status),
            }
            for uid, u in gs.world.units.items()
        ]
        # Record kills in kill feed
        for ev in frame.get("events", []):
            if ev.get("type") == "unit_killed":
                killer = ev.get("source_id", "unknown")
                victim = ev.get("target_id", "unknown")
                killer_unit = gs.world.units.get(killer)
                victim_unit = gs.world.units.get(victim)
                gs.hud.add_kill(
                    killer=killer,
                    victim=victim,
                    killer_alliance=killer_unit.alliance.value if killer_unit else "unknown",
                    victim_alliance=victim_unit.alliance.value if victim_unit else "unknown",
                )
        hud_world_state = {
            "units": hud_units,
            "camera_pos": (25.0, 25.0),
            "camera_fov": 60.0,
            "player_heading": 0.0,
        }
        frame["hud"] = gs.hud.render_frame(hud_world_state, player_alliance="friendly", dt=dt)

    # 34. Civilian simulator tick — fleeing, fear, casualties
    if gs.civilians is not None:
        # Build threat list from hostile unit positions
        civ_threats: list[tuple[tuple[float, float], float]] = []
        civ_explosions: list[tuple[tuple[float, float], float]] = []
        for uid, u in gs.world.units.items():
            if u.is_alive() and u.alliance == Alliance.HOSTILE:
                civ_threats.append((u.position, 30.0))
        for ev in frame.get("events", []):
            if ev.get("type") == "explosion":
                pos = ev.get("position")
                radius = ev.get("radius", 5.0)
                if pos:
                    civ_explosions.append((tuple(pos), radius))
        civ_result = gs.civilians.tick(dt, threats=civ_threats, explosions=civ_explosions)
        frame["civilians"] = gs.civilians.to_three_js()
        if civ_result.get("casualties", 0) > 0:
            frame.setdefault("events", []).append({
                "type": "civilian_casualty",
                "casualties": civ_result["casualties"],
            })

    # 35. Campaign — mission chain state (no tick, event-driven)
    if gs.campaign is not None:
        frame["campaign"] = gs.campaign.to_three_js()

    # 36. Fortifications/Engineering tick — mines, construction progress
    if gs.engineering is not None:
        eng_unit_pos: dict[str, tuple[tuple[float, float], str]] = {}
        for uid, u in gs.world.units.items():
            if u.is_alive():
                eng_unit_pos[uid] = (u.position, u.alliance.value)
        eng_events = gs.engineering.tick(dt, eng_unit_pos)
        frame["fortifications"] = gs.engineering.to_three_js()
        if eng_events:
            frame["fortification_events"] = eng_events[:10]

    # 37. Soundtrack tick — music + audio cues for frontend
    if gs.soundtrack is not None:
        st_events = frame.get("events", [])
        hostile_count = sum(
            1 for u in gs.world.units.values()
            if u.is_alive() and u.alliance == Alliance.HOSTILE
        )
        combat_active = any(
            ev.get("type") in ("fire", "explosion", "unit_killed")
            for ev in st_events
        )
        env_snap = gs.world.environment.snapshot()
        st_world = {
            "hostiles_count": hostile_count,
            "combat_active": combat_active,
            "wave_cleared": hostile_count == 0,
            "game_over": False,
            "game_won": False,
            "weather": env_snap.get("weather", "clear"),
            "wind_speed": env_snap.get("wind_speed", 0.0),
            "rain_intensity": 0.5 if env_snap.get("weather") == "rain" else 0.0,
            "time_of_day": env_snap.get("hour", 12.0),
        }
        soundtrack_frame = gs.soundtrack.tick(st_events, st_world)
        if soundtrack_frame:
            frame["soundtrack"] = soundtrack_frame

    # 38. Event bus — aggregate sim events for timeline/HUD
    if gs.event_bus is not None:
        for ev in frame.get("events", []):
            etype = ev.get("type", "")
            ev_type_map = {
                "fire": SimEventType.SHOT_FIRED,
                "unit_killed": SimEventType.UNIT_KILLED,
                "unit_hit": SimEventType.UNIT_DAMAGED,
                "explosion": SimEventType.EXPLOSION,
            }
            if etype in ev_type_map:
                gs.event_bus.emit(SimEvent(
                    event_type=ev_type_map[etype],
                    tick=gs.tick_count,
                    time=gs.world.sim_time,
                    data=ev,
                ))
        # Export recent event timeline (last 20 events)
        frame["event_timeline"] = gs.event_bus.to_three_js(last_n=20)
        # Export event stats every 50 ticks
        if gs.tick_count % 50 == 0:
            frame["event_stats"] = gs.event_bus.stats()

    # 39. Replay recorder — capture frame for playback
    if gs.replay is not None:
        units_snap = [
            {
                "id": uid,
                "x": round(u.position[0], 2),
                "y": round(u.position[1], 2),
                "alive": u.is_alive(),
                "alliance": u.alliance.value,
            }
            for uid, u in gs.world.units.items()
        ]
        gs.replay.record_frame(
            tick=gs.tick_count,
            time=gs.world.sim_time,
            units=units_snap,
            events=frame.get("events", []),
            render_data={
                "unit_count": len(units_snap),
                "section_count": len(frame),
            },
        )
        frame["replay"] = {
            "recording": True,
            "frames_captured": len(gs.replay.frames),
            "max_frames": gs.replay.max_frames,
        }

    # 40. Physics 2D — sync positions into physics bodies, tick, read back
    if gs.physics_2d is not None:
        # Sync unit positions into physics bodies
        for uid, body_id in gs.unit_body_ids.items():
            unit = gs.world.units.get(uid)
            if unit and unit.is_alive():
                gs.physics_2d.positions[body_id] = unit.position
            else:
                gs.physics_2d.active[body_id] = False
        # Sync vehicle positions into physics bodies
        for vid, body_id in gs.vehicle_body_ids.items():
            veh = gs.world.vehicles.get(vid)
            if veh and not veh.is_destroyed:
                gs.physics_2d.positions[body_id] = veh.position
            else:
                gs.physics_2d.active[body_id] = False
        # Tick physics
        phys_events = gs.physics_2d.tick(dt)
        # Read corrected positions back (resolve overlaps)
        for uid, body_id in gs.unit_body_ids.items():
            unit = gs.world.units.get(uid)
            if unit and unit.is_alive() and gs.physics_2d.active[body_id]:
                unit.position = (
                    float(gs.physics_2d.positions[body_id][0]),
                    float(gs.physics_2d.positions[body_id][1]),
                )
        if phys_events:
            frame["physics_collisions"] = [
                {
                    "body_a": e.body_a,
                    "body_b": e.body_b,
                    "impulse": round(e.impulse, 2),
                    "speed": round(e.relative_speed, 2),
                }
                for e in phys_events[:15]
            ]

    # 41. Vehicle physics — bicycle-model tick for each vehicle
    if gs.vehicle_physics:
        vp_data: list[dict] = []
        for vid, vp in gs.vehicle_physics.items():
            veh = gs.world.vehicles.get(vid)
            if veh is None or veh.is_destroyed:
                continue
            # Set controls from vehicle state (throttle from speed ratio)
            vp.throttle = min(1.0, max(-1.0, getattr(veh, "throttle", 0.5)))
            vp.steering = min(1.0, max(-1.0, getattr(veh, "steering", 0.0)))
            vp.tick(dt)
            # Sync bicycle model back into vehicle state
            veh.position = (float(vp.position[0]), float(vp.position[1]))
            veh.heading = vp.heading
            # Also sync into physics body for collision
            body_id = gs.vehicle_body_ids.get(vid)
            if body_id is not None and gs.physics_2d is not None:
                vp.sync_to_world(gs.physics_2d, body_id)
            vp_data.append({
                "vehicle_id": vid,
                "speed": round(vp.speed, 2),
                "heading": round(vp.heading, 3),
                "throttle": round(vp.throttle, 2),
                "steering": round(vp.steering, 2),
            })
        if vp_data:
            frame["vehicle_physics"] = vp_data

    # 42. Unit FSMs — tick state machines and export state labels to frame
    if gs.unit_fsms:
        fsm_data: dict[str, dict] = {}
        for uid, fsm in gs.unit_fsms.items():
            unit = gs.world.units.get(uid)
            if unit is None or not unit.is_alive():
                continue
            # Build context for FSM transitions
            enemies_in_range = [
                oid for oid, o in gs.world.units.items()
                if oid != uid and o.is_alive() and o.alliance != unit.alliance
                and steering_distance(unit.position, o.position) <= unit.stats.detection_range
            ]
            enemy_in_weapon = any(
                steering_distance(unit.position, gs.world.units[oid].position) <= unit.stats.attack_range
                for oid in enemies_in_range
            )
            health_pct = unit.state.health / max(unit.stats.max_health, 1.0)
            fsm_ctx = {
                "enemies_in_range": enemies_in_range,
                "enemy_in_weapon_range": enemy_in_weapon,
                "aimed_at_target": enemy_in_weapon,
                "just_fired": unit.state.status == "attacking",
                "weapon_ready": True,
                "has_waypoints": bool(unit.squad_id),
                "health_pct": health_pct,
                "nearest_enemy_stationary": False,
                "degradation": 0.0,
            }
            fsm.tick(dt, fsm_ctx)
            fsm_data[uid] = unit_state_to_three_js(uid, fsm)
        if fsm_data:
            frame["unit_fsms"] = fsm_data

    # 43. Inventories — export loadout summaries + apply armor to damage
    if gs.inventories:
        inv_summary: dict[str, dict] = {}
        for uid, inv in gs.inventories.items():
            unit = gs.world.units.get(uid)
            if unit is None or not unit.is_alive():
                continue
            # Auto-switch weapon when out of ammo
            inv.auto_switch_weapon()
            active_weapon = inv.get_active_weapon()
            armor_dr = inv.total_damage_reduction()
            inv_summary[uid] = {
                "weapon": active_weapon.name if active_weapon else "unarmed",
                "ammo": active_weapon.ammo if active_weapon else 0,
                "armor_dr": round(armor_dr, 2),
                "item_count": len(inv.items),
            }
            # Apply armor damage reduction to recent hits
            for ev in frame.get("events", []):
                if ev.get("type") == "unit_hit" and ev.get("target_id") == uid:
                    if armor_dr > 0:
                        inv.damage_armor(1)
                        ev["armor_mitigated"] = round(armor_dr, 2)
        if inv_summary:
            frame["inventories"] = inv_summary

    # 44. Telemetry — record frame performance metrics
    if gs.telemetry is not None:
        gs.telemetry.set_tick(gs.tick_count, gs.world.sim_time)
        alive_count = sum(1 for u in gs.world.units.values() if u.is_alive())
        event_count = len(frame.get("events", []))
        effects_data = frame.get("effects", {})
        if isinstance(effects_data, dict):
            particle_count = len(effects_data.get("particles", []))
        elif isinstance(effects_data, list):
            particle_count = len(effects_data)
        else:
            particle_count = 0
        section_count = len(frame)
        gs.telemetry.record_frame(
            fps=10.0,  # target FPS of the game loop
            frame_time=dt * 1000.0,
            entity_count=alive_count,
            particle_count=particle_count,
        )
        for ev in frame.get("events", []):
            gs.telemetry.record_event(ev.get("type", "unknown"), ev)
        # Export telemetry summary every 100 ticks
        if gs.tick_count % 100 == 0:
            frame["telemetry"] = gs.telemetry.summary()

    # AI behavior state — decision, formation, cover status per unit
    if gs.unit_ai is not None:
        frame["ai_behaviors"] = gs.unit_ai.to_three_js()

    # --- Strategic AI tick (every 30 ticks to avoid overhead) ---
    if gs.tick_count % 30 == 1:
        for faction, strat_ai, plan_attr in [
            ("friendly", gs.strategic_ai_friendly, "current_plan_friendly"),
            ("hostile", gs.strategic_ai_hostile, "current_plan_hostile"),
        ]:
            if strat_ai is None:
                continue
            # Build world state for strategic assessment
            f_squads = []
            e_squads = []
            for sid, squad in gs.world.squads.items():
                living = [m for m in squad.members if m in gs.world.units and gs.world.units[m].is_alive()]
                if not living:
                    continue
                positions = [gs.world.units[m].position for m in living]
                cx = sum(p[0] for p in positions) / len(positions)
                cy = sum(p[1] for p in positions) / len(positions)
                first_alliance = gs.world.units[living[0]].alliance.value
                squad_data = {
                    "id": sid,
                    "position": (cx, cy),
                    "strength": float(len(living)),
                    "morale": 0.7,
                    "ammo": 0.6,
                }
                if first_alliance == faction:
                    f_squads.append(squad_data)
                else:
                    e_squads.append(squad_data)
            objectives = []
            if gs.territory is not None:
                for cp in gs.territory.control_points:
                    objectives.append({
                        "position": cp.position,
                        "owner": getattr(cp, "owner", None) or "contested",
                        "value": 1.0,
                    })
            world_state = {
                "friendly_squads": f_squads,
                "enemy_squads": e_squads,
                "objectives": objectives,
                "fog_of_war": True,
            }
            assessment = strat_ai.assess(world_state)
            plan = strat_ai.plan(faction, assessment)
            setattr(gs, plan_attr, plan)

    # Export strategy + pathfinding + behavior profile data in the frame
    strategy_data: dict[str, Any] = {}
    for faction, plan in [
        ("friendly", gs.current_plan_friendly),
        ("hostile", gs.current_plan_hostile),
    ]:
        if plan is not None:
            strategy_data[faction] = {
                "goal": plan.goal.value,
                "confidence": round(plan.confidence, 2),
                "reasoning": plan.reasoning,
                "primary_target": plan.primary_target,
                "priority": plan.priority,
            }
    if strategy_data:
        frame["strategy"] = strategy_data

    # Pathfinding data — send road network topology on first tick
    if gs.tick_count == 1 and gs.road_network is not None:
        frame["road_network"] = {
            "node_count": gs.road_network.node_count,
            "nodes": [list(n) for n in gs.road_network.nodes[:50]],
        }

    # Behavior profile decisions — per-unit personality-driven actions (every 10 ticks)
    if gs.behavior_engine is not None and gs.tick_count % 10 == 0:
        profile_decisions: dict[str, dict] = {}
        for uid, unit in gs.world.units.items():
            if not unit.is_alive():
                continue
            # Build situation dict from live unit state
            health_ratio = unit.state.health / max(unit.stats.max_health, 1.0)
            ammo_ratio = (unit.state.ammo / 60.0) if unit.state.ammo >= 0 else 1.0
            threats_nearby = sum(
                1 for oid, o in gs.world.units.items()
                if oid != uid and o.is_alive() and o.alliance != unit.alliance
                and steering_distance(unit.position, o.position) <= unit.stats.detection_range
            )
            allies_nearby = sum(
                1 for oid, o in gs.world.units.items()
                if oid != uid and o.is_alive() and o.alliance == unit.alliance
                and steering_distance(unit.position, o.position) <= 40.0
            )
            situation = {
                "health": health_ratio,
                "ammo": ammo_ratio,
                "threats": threats_nearby,
                "allies": allies_nearby,
                "in_cover": False,
                "suppressed": unit.state.suppression > 0.4,
                "enemy_distance": 50.0,
                "has_objective": True,
                "morale": 0.7,
            }
            decision = gs.behavior_engine.decide(uid, situation)
            profile = gs.behavior_engine.get_profile(uid)
            profile_decisions[uid] = {
                "action": decision["action"],
                "reasoning": decision["reasoning"],
                "profile": profile.name if profile else "unknown",
            }
        if profile_decisions:
            frame["behavior_profiles"] = profile_decisions

    # --- Effects (particle system) tick ---
    if gs.effects is not None:
        # Spawn particles from combat events this tick
        # Event types from World.tick(): "fire" (unit_id, weapon, target),
        # "unit_killed" (source_id, target_id), "explosion" (position, radius)
        for ev in frame.get("events", []):
            etype = ev.get("type", "")
            if etype == "fire":
                # Muzzle flash at shooter position
                shooter_id = ev.get("unit_id", "")
                shooter_unit = gs.world.units.get(shooter_id) if gs.world else None
                if shooter_unit and shooter_unit.is_alive():
                    gs.effects.add(fx_muzzle_flash(shooter_unit.position, shooter_unit.heading))
                # Sparks/impact at target position
                target_pos = ev.get("target", None)
                if target_pos:
                    gs.effects.add(fx_sparks(target_pos, direction=(0.0, 1.0)))
            elif etype == "unit_killed":
                # Blood splatter at victim position
                victim_id = ev.get("target_id", "")
                victim_unit = gs.world.units.get(victim_id) if gs.world else None
                if victim_unit:
                    gs.effects.add(fx_blood(victim_unit.position, direction=(0.0, 1.0)))
            elif etype == "explosion":
                pos = ev.get("position", None)
                if pos:
                    radius = ev.get("radius", 5.0)
                    gs.effects.add(fx_explosion(pos, radius=radius))
                    gs.effects.add(fx_debris(pos, num_pieces=15))
        gs.effects.tick(dt)
        frame["effects"] = gs.effects.to_three_js(max_particles=300)

    # --- Spatial audio events ---
    # Collect sound events from combat (gunshots, explosions) for this tick
    sounds: list[dict] = []
    listener_pos = (25.0, 25.0)  # camera / player position
    for ev in frame.get("events", []):
        etype = ev.get("type", "")
        if etype == "fire":
            # Gunshot sound at shooter position
            shooter_id = ev.get("unit_id", "")
            shooter_unit = gs.world.units.get(shooter_id) if gs.world else None
            if shooter_unit and shooter_unit.is_alive():
                weapon = ev.get("weapon", "rifle_shot")
                se = SoundEvent(weapon, shooter_unit.position, volume=0.8, category="weapon")
                computed = se.compute_for_listener(listener_pos)
                sounds.append(computed)
        elif etype == "explosion":
            pos = ev.get("position", None)
            if pos:
                se = SoundEvent("explosion", pos, volume=1.0, category="weapon")
                computed = se.compute_for_listener(listener_pos)
                sounds.append(computed)
    if sounds:
        frame["spatial_audio"] = sounds[:20]  # cap for frame size

    # --- Combat stats (game.StatsTracker) tick ---
    if gs.combat_stats is not None:
        for ev in frame.get("events", []):
            etype = ev.get("type", "")
            if etype == "fire":
                # Each "fire" event = one shot fired (may or may not hit)
                shooter_id = ev.get("unit_id", "")
                if shooter_id:
                    gs.combat_stats.on_shot_fired(shooter_id)
            elif etype == "unit_killed":
                killer = ev.get("source_id", "")
                victim = ev.get("target_id", "")
                if killer and victim:
                    # Record as both a hit and a kill
                    gs.combat_stats.on_shot_hit(killer, victim, 100.0)
                    gs.combat_stats.on_kill(killer, victim)
                victim_unit = gs.world.units.get(victim) if gs.world else None
                if victim_unit and victim_unit.alliance.value == "friendly":
                    gs.combat_stats.on_friendly_loss()
        gs.combat_stats.tick(dt)
        # Export compact stats summary + per-unit leaderboard
        frame["combat_stats"] = gs.combat_stats.get_summary()
        # Send full per-unit stats every 50 ticks (avoid bloating every frame)
        if gs.tick_count % 50 == 0:
            frame["combat_stats_detail"] = gs.combat_stats.to_dict()

    # --- Difficulty scaler state ---
    if gs.difficulty is not None:
        frame["difficulty"] = {
            "multiplier": round(gs.difficulty.get_multiplier(), 3),
            "waves_tracked": len(gs.difficulty.wave_history),
        }

    # --- 45. Animation system — evaluate active animations per entity ---
    if gs.animations:
        anim_data: dict[str, dict] = {}
        sim_t = gs.world.sim_time
        for eid, anim in gs.animations.items():
            # Check if entity is still alive/active
            unit = gs.world.units.get(eid)
            veh = gs.world.vehicles.get(eid)
            is_alive = (unit is not None and unit.is_alive()) or (veh is not None and not veh.is_destroyed)
            if not is_alive:
                # Trigger death animation for units that just died
                if unit is not None and anim.name != "death_fall":
                    death_anim = AnimationLibrary.get("death_fall")
                    if death_anim is not None:
                        gs.animations[eid] = death_anim
                        gs.anim_start_times[eid] = sim_t
                        anim = death_anim
                    else:
                        continue
                elif unit is None:
                    continue
            start_t = gs.anim_start_times.get(eid, 0.0)
            local_t = sim_t - start_t
            props = anim.evaluate(local_t)
            anim_data[eid] = {
                "name": anim.name,
                "loop": anim.loop,
                "time": round(local_t, 3),
                "properties": {k: round(v, 4) if isinstance(v, float) else v for k, v in props.items()},
            }
        if anim_data:
            frame["animations"] = anim_data

    # --- 45b. Interpolation buffers — push positions and export smoothed values ---
    if gs.interp_buffers:
        interp_data: dict[str, dict] = {}
        now = time.monotonic()
        for eid, buf in gs.interp_buffers.items():
            unit = gs.world.units.get(eid)
            if unit is None or not unit.is_alive():
                continue
            buf.push(unit.position, now)
            smoothed = buf.evaluate(now)
            if smoothed is not None:
                interp_data[eid] = {
                    "smoothed_x": round(smoothed[0], 2) if isinstance(smoothed, tuple) else round(smoothed, 2),
                    "smoothed_y": round(smoothed[1], 2) if isinstance(smoothed, tuple) else 0.0,
                    "samples": buf.sample_count,
                }
        if interp_data:
            frame["interpolation"] = interp_data

    # --- 46. NPC thinker — export recent thoughts (async generation happens in game_loop) ---
    if gs.npc_thinker is not None:
        recent = gs.npc_thinker.get_recent_thoughts(5)
        if recent:
            frame["npc_thoughts"] = recent

    # --- 47. Multiplayer — lobby state + per-player fog-of-war view ---
    if gs.multiplayer is not None:
        mp_data: dict[str, Any] = {
            "lobby": gs.multiplayer.get_lobby_state(),
            "turn": gs.multiplayer.turn_number,
        }
        # Export fog-of-war filtered view for the human player
        if "player_1" in gs.multiplayer.players:
            player_view = gs.multiplayer.get_player_view("player_1", gs.world)
            mp_data["player_view"] = {
                "visible_units": len(player_view.get("units", [])),
                "visible_vehicles": len(player_view.get("vehicles", [])),
                "controlled_units": player_view.get("controlled_units", []),
                "score": player_view.get("score", 0),
            }
        frame["multiplayer"] = mp_data

    # Add metadata
    frame["tick"] = gs.tick_count
    frame["sim_time"] = round(gs.world.sim_time, 2)
    frame["preset"] = gs.preset
    frame["stats"] = gs.world.stats()

    # Add terrain data on first tick (large payload, sent once)
    if gs.tick_count == 1 and gs.world.terrain_layer is not None:
        tl = gs.world.terrain_layer
        frame["terrain_geojson"] = tl.to_geojson()
        frame["terrain_brief"] = tl.terrain_brief()
        # Add mission data
        try:
            from tritium_lib.intelligence.geospatial.mission_generator import MissionGenerator
            gen = MissionGenerator()
            missions = gen.generate_missions(tl)
            frame["missions"] = [
                {
                    "id": m.id, "type": m.mission_type, "name": m.name,
                    "description": m.description, "position": m.position,
                    "waypoints": m.waypoints, "priority": m.priority,
                }
                for m in missions
            ]
        except Exception:
            pass

    # Add generated map features on first frame (cache for welcome reuse)
    if gs.tick_count == 1 and gs.generated_map is not None:
        if gs._cached_map_features is None:
            gs._cached_map_features = [
                {
                    "id": f.feature_id,
                    "type": f.feature_type,
                    "x": f.position[0],
                    "y": f.position[1],
                    "w": f.size[0],
                    "h": f.size[1],
                    "rotation": f.rotation,
                }
                for f in gs.generated_map.features
            ]
        frame["map_features"] = gs._cached_map_features

    return frame


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Tritium Sim Engine Demo", version="1.0.0")  # type: ignore[call-arg]
_game: GameState = GameState()
_ws_clients: list[Any] = []
_game_task: asyncio.Task | None = None

# Delta-frame state: previous frame cache for diff computation
_prev_game_frame: dict[str, Any] | None = None
_prev_city_frame: dict[str, Any] | None = None
_KEYFRAME_INTERVAL: int = 50  # Send full frame every N ticks

# ---------------------------------------------------------------------------
# CitySim state — separate from the tactical GameState
# ---------------------------------------------------------------------------
_city_sim: CitySim | None = None
_city_task: asyncio.Task | None = None
_city_mode: bool = False  # When True, WS streams city frames instead of game frames


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the game client HTML page. Auto-starts the game."""
    global _game, _game_task, _prev_game_frame
    if _game.world is None or not _game.running:
        import os
        preset = os.environ.get("SIM_PRESET", "urban_combat")
        _game = build_full_game(preset)
        _game.running = True
        _prev_game_frame = None  # Reset delta-frame cache
        if _game_task is not None and not _game_task.done():
            _game_task.cancel()
        _game_task = asyncio.create_task(_game_loop())
    # Always serve the inline GAME_HTML — it contains all rendering fixes
    # (camera centered on MAP_SIZE, effects dict handling, diff-frame guards).
    # The disk game.html is kept as a reference but is NOT served.
    return HTMLResponse(content=GAME_HTML, status_code=200)


@app.get("/favicon.ico")
async def favicon() -> Response:
    """Return an empty favicon to prevent 404 on page load."""
    return Response(content=b"", media_type="image/x-icon", status_code=204)


@app.get("/city", response_class=HTMLResponse)
async def city_view() -> HTMLResponse:
    """Serve the city3d.html demo with WebSocket bridge injected.

    Auto-starts the CitySim backend so the frontend receives real sim frames
    instead of running its own JS-side simulation.
    """
    global _city_sim, _city_task, _city_mode, _prev_city_frame

    # Auto-start CitySim when someone visits /city
    if _city_sim is None:
        _city_sim = CitySim(seed=42)
        _city_sim.setup()
        _city_mode = True
        _prev_city_frame = None  # Reset delta-frame cache
        if _city_task is not None and not _city_task.done():
            _city_task.cancel()
        _city_task = asyncio.create_task(_city_loop())

    import pathlib
    city_path = pathlib.Path(__file__).parent / "city3d.html"
    if not city_path.exists():
        return HTMLResponse("<h1>city3d.html not found</h1>", status_code=404)
    html = city_path.read_text()
    # Inject WebSocket bridge before closing </body>
    # This bridge receives city_frame data and exposes it to city3d for rendering
    ws_bridge = """
<script>
// WebSocket bridge — receives CitySim frames from Python backend
(function() {
  let ws;
  function connect() {
    ws = new WebSocket('ws://' + location.host + '/ws');
    ws.onopen = function() { console.log('[CitySim] Connected to backend'); };
    ws.onmessage = function(e) {
      try {
        var f = JSON.parse(e.data);
        if (f.type === 'city_frame') {
          // Expose full frame for city3d renderer to consume
          window.__cityFrame = f;
          window.__simFrame = {
            tick: f.tick, time: f.sim_time,
            civilians: f.civilians, vehicles: f.vehicles,
            police: f.police, crowd: f.crowd,
            buildings: f.buildings, trees: f.trees,
            environment: f.environment, events: f.events,
            stats: f.stats
          };
        } else {
          window.__simFrame = { tick: f.tick, time: f.time, phase: f.phase,
            unit_count: (f.units||[]).length, score: (f.ui||{}).score || 0 };
        }
      } catch(err) {}
    };
    ws.onerror = function() { console.log('[CitySim] Standalone mode'); };
    ws.onclose = function() { setTimeout(connect, 5000); };
  }
  setTimeout(connect, 1000);
})();
</script>
"""
    html = html.replace("</body>", ws_bridge + "</body>")
    return HTMLResponse(content=html, status_code=200)


@app.get("/api/status")
async def api_status() -> dict:
    """Current game state summary."""
    if _game.world is None:
        return {"running": False, "preset": "", "tick_count": 0}
    return {
        "running": _game.running,
        "paused": _game.paused,
        "preset": _game.preset,
        "tick_count": _game.tick_count,
        "sim_time": round(_game.world.sim_time, 2),
        "stats": _game.world.stats(),
        "factions": list(_game.diplomacy.factions.keys()) if _game.diplomacy else [],
        "modules_active": _count_active_modules(_game),
    }


@app.post("/api/start")
async def api_start(body: dict | None = None) -> dict:
    """Start a new game."""
    global _game, _game_task, _prev_game_frame
    if body is None:
        body = {}
    preset = body.get("preset", "urban_combat")
    _game = build_full_game(preset)
    _game.running = True
    _prev_game_frame = None  # Reset delta-frame cache for new game
    # Start the game loop
    if _game_task is not None and not _game_task.done():
        _game_task.cancel()
    _game_task = asyncio.create_task(_game_loop())
    return {"status": "started", "preset": preset, "modules": _count_active_modules(_game)}


@app.post("/api/pause")
async def api_pause() -> dict:
    """Pause or resume the game."""
    if _game.world is None and not _game.running:
        return {"error": "no_game"}
    _game.paused = not _game.paused
    return {"paused": _game.paused}


@app.post("/api/command")
async def api_command(body: dict) -> dict:
    """Issue a command to a unit."""
    if _game.world is None:
        return {"error": "no_game"}
    cmd_type = body.get("type", "")
    unit_id = body.get("unit_id", "")
    target = body.get("target", [0, 0])

    if cmd_type == "move":
        unit = _game.world.units.get(unit_id)
        if unit and unit.is_alive():
            unit.position = (float(target[0]), float(target[1]))
            return {"status": "moved", "unit_id": unit_id}
    elif cmd_type == "fire":
        proj = _game.world.fire_weapon(unit_id, (float(target[0]), float(target[1])))
        return {"status": "fired" if proj else "failed", "unit_id": unit_id}
    elif cmd_type == "weather":
        weather_name = body.get("weather", "")
        intensity = body.get("intensity", None)
        if intensity is not None:
            intensity = float(intensity)
        return _apply_weather_override(weather_name, intensity)

    return {"status": "unknown_command", "type": cmd_type}


@app.get("/api/presets")
async def api_presets() -> dict:
    """List available world presets."""
    return {
        "world_presets": list(WORLD_PRESETS.keys()),
        "scenario_presets": list(PRESET_SCENARIOS.keys()),
        "campaign_presets": list(CAMPAIGNS.keys()),
        "vehicle_templates": list(VEHICLE_TEMPLATES.keys()),
        "aircraft_templates": list(AIRCRAFT_TEMPLATES.keys()),
        "weapon_count": len(ARSENAL),
    }


@app.get("/api/stats")
async def api_stats() -> dict:
    """Current scoring and leaderboard."""
    if _game.scoring is None:
        return {"error": "no_game"}
    return {
        "leaderboard": _game.scoring.get_leaderboard(),
        "team_scores": {
            alliance: {
                "kills": ts.total_kills,
                "deaths": ts.total_deaths,
            }
            for alliance, ts in _game.scoring.team_scores.items()
        },
    }


@app.get("/api/aar")
async def api_aar() -> dict:
    """After-action report (when game is over or at any time)."""
    if _game.scoring is None:
        return {"error": "no_game"}
    winner = None
    if _game.world:
        stats = _game.world.stats()
        if stats["alive_friendly"] > 0 and stats["alive_hostile"] == 0:
            winner = "friendly"
        elif stats["alive_hostile"] > 0 and stats["alive_friendly"] == 0:
            winner = "hostile"
    aar = _game.scoring.generate_aar(winner_alliance=winner)
    # Sanitize for JSON serialization (remove surrogate chars)
    return json.loads(json.dumps(aar, default=str, ensure_ascii=True))


# ---------------------------------------------------------------------------
# CitySim API endpoints
# ---------------------------------------------------------------------------


@app.post("/api/city/start")
async def api_city_start(body: dict | None = None) -> dict:
    """Start the CitySim backend simulation."""
    global _city_sim, _city_task, _city_mode, _prev_city_frame
    if body is None:
        body = {}
    seed = body.get("seed", None)
    width = body.get("width", 500.0)
    height = body.get("height", 400.0)
    hour = body.get("hour", 10.0)

    _city_sim = CitySim(width=width, height=height, seed=seed, hour=hour)
    _city_sim.setup()
    _city_mode = True
    _prev_city_frame = None  # Reset delta-frame cache

    # Start the city loop
    if _city_task is not None and not _city_task.done():
        _city_task.cancel()
    _city_task = asyncio.create_task(_city_loop())

    return {
        "status": "started",
        "seed": _city_sim.seed,
        "width": _city_sim.width,
        "height": _city_sim.height,
        "civilians": len(_city_sim.civilians),
        "vehicles": len(_city_sim.city_vehicles),
        "police": len(_city_sim.police_units),
        "buildings": len(_city_sim.buildings),
    }


@app.post("/api/city/stop")
async def api_city_stop() -> dict:
    """Stop the CitySim backend simulation."""
    global _city_sim, _city_task, _city_mode
    if _city_sim is None:
        return {"error": "no_city_sim"}
    _city_mode = False
    if _city_task is not None and not _city_task.done():
        _city_task.cancel()
        _city_task = None
    _city_sim = None
    return {"status": "stopped"}


@app.get("/api/city/status")
async def api_city_status() -> dict:
    """Current CitySim state summary."""
    if _city_sim is None:
        return {"running": False, "tick_count": 0}
    return {
        "running": _city_mode,
        "tick_count": _city_sim.tick_count,
        "sim_time": round(_city_sim.sim_time, 2),
        "stats": _city_sim.stats(),
    }


@app.post("/api/city/event")
async def api_city_event(body: dict) -> dict:
    """Inject an event into the running CitySim.

    Body fields:
        event_type: str — e.g. "riot_start", "teargas", "gunshot", "chant"
        x: float — x position
        z: float — z/y position
        radius: float (optional, default 20.0)
        intensity: float (optional, default 0.5)
    """
    if _city_sim is None:
        return {"error": "no_city_sim"}
    event_type = body.get("event_type", "")
    if not event_type:
        return {"error": "missing_event_type"}
    x = body.get("x", _city_sim.width / 2)
    z = body.get("z", _city_sim.height / 2)
    radius = body.get("radius", 20.0)
    intensity = body.get("intensity", 0.5)

    _city_sim.inject_crowd_event(event_type, (x, z), radius=radius, intensity=intensity)
    return {
        "status": "injected",
        "event_type": event_type,
        "position": {"x": x, "z": z},
        "radius": radius,
        "intensity": intensity,
    }


# ---------------------------------------------------------------------------
# Weather override API — force weather for testing / Village Idiot audits
# ---------------------------------------------------------------------------

# Accepted shorthand names mapped to Weather enum values
_WEATHER_ALIASES: dict[str, Weather] = {
    "clear": Weather.CLEAR,
    "cloudy": Weather.CLOUDY,
    "fog": Weather.FOG,
    "rain": Weather.RAIN,
    "heavy_rain": Weather.HEAVY_RAIN,
    "snow": Weather.SNOW,
    "storm": Weather.STORM,
    "sandstorm": Weather.SANDSTORM,
}


def _apply_weather_override(weather_name: str, intensity: float | None = None) -> dict:
    """Apply a weather override to whichever sim is running.

    Returns a status dict.  Works for both the tactical game (_game) and
    CitySim (_city_sim).
    """
    weather_name = weather_name.strip().lower()
    weather = _WEATHER_ALIASES.get(weather_name)
    if weather is None:
        return {
            "error": "invalid_weather",
            "valid": list(_WEATHER_ALIASES.keys()),
        }

    applied_to: list[str] = []

    # Tactical game
    if _game.world is not None:
        _game.world.environment.weather.state.current = weather
        if intensity is not None:
            _game.world.environment.weather.state.intensity = max(0.0, min(1.0, intensity))
        applied_to.append("game")

    # CitySim
    if _city_sim is not None:
        _city_sim.environment.weather.state.current = weather
        if intensity is not None:
            _city_sim.environment.weather.state.intensity = max(0.0, min(1.0, intensity))
        applied_to.append("city")

    if not applied_to:
        return {"error": "no_sim_running"}

    result: dict[str, Any] = {
        "status": "weather_set",
        "weather": weather.value,
        "applied_to": applied_to,
    }
    if intensity is not None:
        result["intensity"] = round(max(0.0, min(1.0, intensity)), 3)
    return result


@app.post("/api/weather")
async def api_weather(body: dict) -> dict:
    """Force a weather condition override.

    Body::

        {"weather": "rain"}
        {"weather": "storm", "intensity": 0.8}

    Valid weather values: clear, cloudy, fog, rain, heavy_rain, snow, storm, sandstorm
    """
    weather_name = body.get("weather", "")
    intensity = body.get("intensity", None)
    if intensity is not None:
        intensity = float(intensity)
    return _apply_weather_override(weather_name, intensity)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """WebSocket for streaming frame data at 10 fps."""
    await ws.accept()

    # Send a welcome frame with one-time data that tick==1 already sent.
    # Late joiners would otherwise never receive map_features / destruction.
    try:
        welcome: dict[str, Any] = {"type": "welcome", "tick": _game.tick_count}
        if _game._cached_map_features is not None:
            welcome["map_features"] = _game._cached_map_features
        elif _game.generated_map is not None:
            _game._cached_map_features = [
                {
                    "id": f.feature_id,
                    "type": f.feature_type,
                    "x": f.position[0],
                    "y": f.position[1],
                    "w": f.size[0],
                    "h": f.size[1],
                    "rotation": f.rotation,
                }
                for f in _game.generated_map.features
            ]
            welcome["map_features"] = _game._cached_map_features
        if _game.world is not None and _game.world.destruction is not None:
            welcome["destruction"] = _game.world.destruction.to_three_js()
        await ws.send_text(json.dumps(welcome, default=str))
    except Exception:
        pass

    _ws_clients.append(ws)
    try:
        while True:
            # Keep connection alive; frames are pushed from game_loop
            data = await ws.receive_text()
            # Client can send commands via WS too
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
                elif msg.get("type") == "weather":
                    weather_name = msg.get("weather", "")
                    intensity = msg.get("intensity", None)
                    if intensity is not None:
                        intensity = float(intensity)
                    result = _apply_weather_override(weather_name, intensity)
                    result["type"] = "weather_ack"
                    await ws.send_text(json.dumps(result, default=str))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        _ws_clients.remove(ws)
    except Exception:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

async def _game_loop() -> None:
    """Background asyncio task: tick the game at 10 fps.

    Uses delta-frame encoding: most ticks send only the diff from the
    previous frame (~2-5 KB instead of ~32 KB).  Every ``_KEYFRAME_INTERVAL``
    ticks a full keyframe is sent so late-joining clients can sync.
    """
    global _prev_game_frame
    dt = 0.1  # 10 fps
    _npc_think_counter = 0
    while _game.running:
        if not _game.paused and _game.world is not None:
            frame = game_tick(_game, dt)

            # NPC thinker — generate thoughts for a random civilian every ~100 ticks
            _npc_think_counter += 1
            if (_game.npc_thinker is not None
                    and _game.civilians is not None
                    and _npc_think_counter % 100 == 0):
                try:
                    civs = list(_game.civilians.civilians.values())
                    if civs:
                        civ = civs[_npc_think_counter % len(civs)]
                        npc_dict = {
                            "target_id": getattr(civ, "civilian_id", "civ_0"),
                            "name": getattr(civ, "name", "Civilian"),
                            "status": getattr(civ, "state", CivilianState.NORMAL).value
                                if hasattr(getattr(civ, "state", None), "value") else "idle",
                            "asset_type": "person",
                        }
                        env_snap = _game.world.environment.snapshot()
                        situation = {
                            "time_of_day": "night" if env_snap.get("hour", 12) > 20 else "daytime",
                            "location": "the city center",
                        }
                        asyncio.create_task(_game.npc_thinker.think(npc_dict, situation))
                except Exception:
                    pass  # LLM unavailable — graceful degradation

            # Decide: keyframe or delta
            is_keyframe = (
                _prev_game_frame is None
                or _game.tick_count <= 1
                or _game.tick_count % _KEYFRAME_INTERVAL == 0
            )

            if is_keyframe:
                frame["frame_type"] = "keyframe"
                send_frame = frame
            else:
                diff = SimRenderer.render_diff(_prev_game_frame, frame)
                diff["frame_type"] = "diff"
                send_frame = diff

            _prev_game_frame = frame

            payload = json.dumps(send_frame, default=str)
            # Broadcast to all WS clients
            disconnected: list[WebSocket] = []
            for ws in _ws_clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    disconnected.append(ws)
            for ws in disconnected:
                if ws in _ws_clients:
                    _ws_clients.remove(ws)
        await asyncio.sleep(dt)


async def _city_loop() -> None:
    """Background asyncio task: tick the CitySim at 10 fps.

    Uses delta-frame encoding identical to ``_game_loop``.
    """
    global _prev_city_frame
    _city_tick_count = 0
    dt = 0.1  # 10 fps
    while _city_mode and _city_sim is not None:
        frame = _city_sim.tick(dt)
        _city_tick_count += 1

        is_keyframe = (
            _prev_city_frame is None
            or _city_tick_count <= 1
            or _city_tick_count % _KEYFRAME_INTERVAL == 0
        )

        if is_keyframe:
            frame["frame_type"] = "keyframe"
            send_frame = frame
        else:
            diff = SimRenderer.render_diff(_prev_city_frame, frame)
            diff["frame_type"] = "diff"
            send_frame = diff

        _prev_city_frame = frame

        payload = json.dumps(send_frame, default=str)
        disconnected: list[WebSocket] = []
        for ws in _ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            if ws in _ws_clients:
                _ws_clients.remove(ws)
        await asyncio.sleep(dt)


def _count_active_modules(gs: GameState) -> int:
    """Count how many subsystem modules are initialized."""
    count = 0
    for attr in (
        "world", "scoring", "detection", "comms", "medical",
        "logistics", "naval", "air_combat", "engineering",
        "asymmetric", "civilians", "intel", "diplomacy", "campaign",
        "morale", "ew", "supply_routes",
        "effects", "combat_stats", "difficulty", "debug_overlay",
        "physics_2d", "telemetry", "npc_thinker", "multiplayer",
    ):
        if getattr(gs, attr, None) is not None:
            count += 1
    # Dict-based modules: count if they have entries
    for attr in ("vehicle_physics", "unit_fsms", "inventories", "animations", "interp_buffers"):
        if getattr(gs, attr, None):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Inline game.html
# ---------------------------------------------------------------------------

GAME_HTML = r"""<!DOCTYPE html>
<!--
  Tritium Sim Engine — Three.js 3D Tactical Viewer
  Receives frame data via WebSocket from the Python game server.
  Copyright 2026 Valpatel Software LLC — AGPL-3.0
-->
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tritium Sim Engine — 3D Tactical</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a2030; overflow: hidden; font-family: 'Courier New', monospace; }
canvas { display: block; }

/* ---- HUD overlay ---- */
#hud {
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  pointer-events: none; color: #00f0ff; font-size: 13px;
}
#hud > div { pointer-events: auto; }

/* Top-left status panel */
#status-panel {
  position: absolute; top: 12px; left: 12px;
  background: rgba(10,10,15,0.88); padding: 12px 16px;
  border: 1px solid #00f0ff44; border-radius: 4px;
  min-width: 240px; line-height: 1.7;
}
#status-panel .title {
  font-size: 16px; font-weight: bold; color: #00f0ff;
  text-shadow: 0 0 8px #00f0ff66; margin-bottom: 6px;
}
#status-panel .row { display: flex; justify-content: space-between; }
#status-panel .label { color: #00f0ff99; }
#status-panel .val { font-weight: bold; }
#status-panel .val-friendly { color: #05ffa1; }
#status-panel .val-hostile { color: #ff2a6d; }
#status-panel .val-neutral { color: #fcee0a; }
#status-panel .val-cyan { color: #00f0ff; }

/* Controls bar */
#controls-bar {
  position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
  display: flex; gap: 4px; align-items: center;
}
#controls-bar button {
  background: #1a1a2e; color: #00f0ff; border: 1px solid #00f0ff66;
  padding: 6px 16px; cursor: pointer; font-family: inherit; font-size: 13px;
  border-radius: 3px; transition: all 0.15s;
}
#controls-bar button:hover { background: #00f0ff; color: #0a0a0f; }
#controls-bar select {
  background: #1a1a2e; color: #00f0ff; border: 1px solid #00f0ff66;
  padding: 6px 10px; font-family: inherit; font-size: 12px;
  border-radius: 3px; cursor: pointer;
}

/* Kill feed — top right */
#kill-feed {
  position: absolute; top: 12px; right: 12px;
  background: rgba(10,10,15,0.85); padding: 10px 14px;
  border: 1px solid #ff2a6d44; border-radius: 4px;
  max-height: 300px; overflow: hidden; min-width: 220px;
  font-size: 12px; line-height: 1.6;
}
#kill-feed .kf-title {
  color: #ff2a6d; font-weight: bold; font-size: 13px;
  margin-bottom: 4px; text-shadow: 0 0 6px #ff2a6d66;
}
.kf-entry { opacity: 0.9; }
.kf-entry.fade { opacity: 0.35; }

/* Unit roster — bottom left */
#roster {
  position: absolute; bottom: 50px; left: 12px;
  background: rgba(10,10,15,0.85); padding: 10px 14px;
  border: 1px solid #05ffa144; border-radius: 4px;
  max-height: 280px; overflow-y: auto; min-width: 260px;
  font-size: 11px; line-height: 1.5;
}
#roster .ros-title {
  color: #05ffa1; font-weight: bold; font-size: 13px;
  margin-bottom: 4px; text-shadow: 0 0 6px #05ffa166;
}
.ros-unit { display: flex; align-items: center; gap: 6px; margin: 2px 0; }
.ros-hp-bar {
  width: 60px; height: 6px; background: #1a1a2e; border-radius: 2px;
  overflow: hidden; flex-shrink: 0;
}
.ros-hp-fill { height: 100%; border-radius: 2px; transition: width 0.2s; }
.ros-label { min-width: 90px; }
.ros-dead { color: #555; text-decoration: line-through; }

/* Camera help — bottom center */
#camera-help {
  position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
  text-align: center; font-size: 11px; color: #00f0ff66;
  background: #0a0a0f88; padding: 5px 14px; border-radius: 3px;
  border: 1px solid #00f0ff22;
}
#camera-help span { color: #00f0ff; }

/* FPS counter */
#fps-counter {
  position: absolute; bottom: 10px; right: 12px;
  font-size: 11px; color: #05ffa188;
}

/* AAR overlay */
#aar-overlay {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  background: rgba(10,10,15,0.95); padding: 20px 28px;
  border: 1px solid #00f0ff; border-radius: 6px;
  max-width: 600px; max-height: 80vh; overflow-y: auto;
  font-size: 12px; white-space: pre-wrap; display: none; z-index: 20;
}
#aar-overlay .close-btn {
  position: absolute; top: 8px; right: 12px; cursor: pointer;
  color: #ff2a6d; font-size: 16px;
}

/* Waiting splash */
#splash {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  text-align: center; color: #00f0ff;
}
#splash .big { font-size: 32px; font-weight: bold; text-shadow: 0 0 16px #00f0ff66; }
#splash .sub { font-size: 14px; color: #00f0ff88; margin-top: 8px; }
</style>
</head>
<body>

<div id="hud">
  <!-- Status panel top-left -->
  <div id="status-panel">
    <div class="title">TRITIUM SIM ENGINE</div>
    <div class="row"><span class="label">Tick</span><span class="val val-cyan" id="hud-tick">0</span></div>
    <div class="row"><span class="label">Sim Time</span><span class="val val-cyan" id="hud-time">0.0s</span></div>
    <div class="row"><span class="label">Friendly</span><span class="val val-friendly" id="hud-friendly">0</span></div>
    <div class="row"><span class="label">Hostile</span><span class="val val-hostile" id="hud-hostile">0</span></div>
    <div class="row"><span class="label">Dead</span><span class="val" style="color:#555" id="hud-dead">0</span></div>
    <div class="row"><span class="label">Vehicles</span><span class="val val-cyan" id="hud-vehicles">0</span></div>
    <div class="row"><span class="label">Crowd</span><span class="val val-neutral" id="hud-crowd">0</span></div>
    <div class="row"><span class="label">Fires</span><span class="val val-hostile" id="hud-fires">0</span></div>
    <div class="row"><span class="label">Weather</span><span class="val val-cyan" id="hud-weather">-</span></div>
    <div class="row"><span class="label">Time of Day</span><span class="val val-neutral" id="hud-tod">-</span></div>
    <div class="row"><span class="label">Preset</span><span class="val val-cyan" id="hud-preset">-</span></div>
  </div>

  <!-- Controls bar top center -->
  <div id="controls-bar">
    <select id="preset-select">
      <option value="urban_combat">Urban Combat</option>
    </select>
    <button onclick="startGame()">START</button>
    <button onclick="pauseGame()">PAUSE</button>
    <button onclick="getAAR()">AAR</button>
  </div>

  <!-- Kill feed top-right -->
  <div id="kill-feed">
    <div class="kf-title">KILL FEED</div>
    <div id="kf-entries"></div>
  </div>

  <!-- Unit roster bottom-left -->
  <div id="roster">
    <div class="ros-title">UNIT ROSTER</div>
    <div id="roster-entries"></div>
  </div>

  <!-- Camera help bottom -->
  <div id="camera-help">
    <span>LMB</span> Rotate &nbsp;|&nbsp;
    <span>RMB</span> Pan &nbsp;|&nbsp;
    <span>Scroll</span> Zoom &nbsp;|&nbsp;
    <span>R</span> Reset Camera
  </div>

  <!-- FPS counter -->
  <div id="fps-counter">-- FPS</div>

  <!-- AAR overlay -->
  <div id="aar-overlay">
    <span class="close-btn" onclick="document.getElementById('aar-overlay').style.display='none'">&times;</span>
    <pre id="aar-content"></pre>
  </div>

  <!-- Splash -->
  <div id="splash">
    <div class="big">TRITIUM SIM ENGINE</div>
    <div class="sub">Select a preset and click START</div>
  </div>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// =========================================================================
// Constants
// =========================================================================
const CYAN    = 0x00f0ff;
const MAGENTA = 0xff2a6d;
const GREEN   = 0x05ffa1;
const YELLOW  = 0xfcee0a;
const VOID_BG = 0x1a2030;  // Brighter dark blue — scene stays visible

const MAP_SIZE = 500;

// =========================================================================
// State
// =========================================================================
let ws = null;
let lastFrame = null;
let frameCount = 0, fpsTimer = 0, fpsDisplay = 0;
let lastTime = performance.now();

// Kill feed entries: {text, age, color}
const killFeed = [];

// Object pools — keyed by id so we can reuse meshes across frames
const unitMeshes = {};    // id -> { cone, ring, label }
const vehicleMeshes = {}; // id -> { body, cabin }
const projectileMeshes = []; // array of line objects
const effectMeshes = [];     // array of sphere objects
// crowdInstCalm/Agitated/Rioting/Fleeing handle civilian rendering
const buildingMeshes = {};   // id/index -> mesh

// Reusable helpers
const _v3 = new THREE.Vector3();
const _color = new THREE.Color();
const _mat4 = new THREE.Matrix4();

// =========================================================================
// Scene setup
// =========================================================================
const scene = new THREE.Scene();
scene.background = new THREE.Color(VOID_BG);
scene.fog = new THREE.FogExp2(VOID_BG, 0.0015);

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.5, 2000);
// Units spawn near origin (-30..100), so centre camera on the action
camera.position.set(40, 120, 140);
camera.lookAt(30, 0, 30);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 2.0;
document.body.prepend(renderer.domElement);

// OrbitControls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(30, 0, 30);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 30;
controls.maxDistance = 900;
controls.maxPolarAngle = Math.PI / 2.1;
controls.update();

// =========================================================================
// Lights
// =========================================================================
scene.add(new THREE.AmbientLight(0x99aacc, 1.0));

const sunLight = new THREE.DirectionalLight(0xffeedd, 1.5);
sunLight.position.set(200, 400, 100);
sunLight.castShadow = true;
sunLight.shadow.mapSize.set(2048, 2048);
sunLight.shadow.camera.left = -MAP_SIZE;
sunLight.shadow.camera.right = MAP_SIZE;
sunLight.shadow.camera.top = MAP_SIZE;
sunLight.shadow.camera.bottom = -MAP_SIZE;
scene.add(sunLight);

scene.add(new THREE.HemisphereLight(0x00f0ff, 0x334466, 0.8));

// Cyan accent lights at corners
for (const [cx, cz] of [[50, 50], [450, 50], [50, 450], [450, 450]]) {
  const pl = new THREE.PointLight(0x00f0ff, 0.6, 200);
  pl.position.set(cx, 30, cz);
  scene.add(pl);
}

// =========================================================================
// Ground plane + grid
// =========================================================================
const groundGeo = new THREE.PlaneGeometry(MAP_SIZE + 100, MAP_SIZE + 100);
const groundMat = new THREE.MeshStandardMaterial({
  color: 0x3a3a48, roughness: 0.8, metalness: 0.05
});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.set(0, -0.05, 0);
ground.receiveShadow = true;
scene.add(ground);

// Grid lines
const gridHelper = new THREE.GridHelper(MAP_SIZE, 10, 0x3a3a55, 0x2a2a3a);
gridHelper.position.set(0, 0.01, 0);
scene.add(gridHelper);

// Fine grid
const fineGrid = new THREE.GridHelper(MAP_SIZE, 50, 0x222230, 0x222230);
fineGrid.position.set(0, 0.005, 0);
scene.add(fineGrid);

// =========================================================================
// Shared geometries & materials
// =========================================================================
// Units — cone body + ring marker
const unitConeGeo = new THREE.ConeGeometry(1.5, 4, 8);
const unitRingGeo = new THREE.RingGeometry(2.0, 2.6, 16);
const unitDeadGeo = new THREE.BoxGeometry(2, 0.3, 2);

// Vehicles — box body + smaller box cabin
const vehBodyGeo = new THREE.BoxGeometry(5, 2, 3);
const vehCabinGeo = new THREE.BoxGeometry(2.5, 1.2, 2.6);

// Projectile — small sphere
const projGeo = new THREE.SphereGeometry(0.3, 6, 6);

// Effect — wireframe sphere
const effectGeo = new THREE.SphereGeometry(1, 12, 12);

// Building — box (scaled per building)
const buildingGeo = new THREE.BoxGeometry(1, 1, 1);
const buildingRoofGeo = new THREE.BoxGeometry(1, 1, 1);

// Materials
const matFriendly = new THREE.MeshStandardMaterial({ color: GREEN, roughness: 0.5, metalness: 0.2 });
const matHostile = new THREE.MeshStandardMaterial({ color: MAGENTA, roughness: 0.5, metalness: 0.2 });
const matDead = new THREE.MeshStandardMaterial({ color: 0x333333, roughness: 0.9 });
const matFriendlyEmit = new THREE.MeshStandardMaterial({ color: GREEN, emissive: GREEN, emissiveIntensity: 0.3 });
const matHostileEmit = new THREE.MeshStandardMaterial({ color: MAGENTA, emissive: MAGENTA, emissiveIntensity: 0.3 });
const matVehFriendly = new THREE.MeshStandardMaterial({ color: 0x048a6a, roughness: 0.6, metalness: 0.4 });
const matVehHostile = new THREE.MeshStandardMaterial({ color: 0xa01848, roughness: 0.6, metalness: 0.4 });
const matVehDestroyed = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.9 });
const matVehCabin = new THREE.MeshStandardMaterial({ color: 0x1a1a2e, roughness: 0.7, metalness: 0.3 });
const matProjectile = new THREE.MeshBasicMaterial({ color: 0xffaa00 });
const matBuilding = new THREE.MeshStandardMaterial({ color: 0x6a6a80, roughness: 0.6, metalness: 0.05, emissive: 0x3a3a55, emissiveIntensity: 1.5 });
const matBuildingRoof = new THREE.MeshStandardMaterial({ color: 0x5a5a70, roughness: 0.7, emissive: 0x2a2a44, emissiveIntensity: 0.8 });
const matBuildingDestroyed = new THREE.MeshStandardMaterial({ color: 0x6a3030, roughness: 0.9, emissive: 0x3a1010, emissiveIntensity: 0.5 });
const matRing = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0.5 });

// Crowd: instanced cylinders so civilians are visible at any zoom level
const crowdMaxCount = 200;
const crowdGeoInst = new THREE.CylinderGeometry(0.4, 0.5, 1.8, 6, 1);
const crowdMatCalm      = new THREE.MeshStandardMaterial({ color: 0x05ffa1, emissive: 0x052a1a, emissiveIntensity: 0.5 });
const crowdMatAgitated  = new THREE.MeshStandardMaterial({ color: YELLOW,   emissive: 0x2a1a00, emissiveIntensity: 0.5 });
const crowdMatRioting   = new THREE.MeshStandardMaterial({ color: MAGENTA,  emissive: 0x2a0012, emissiveIntensity: 0.6 });
const crowdMatFleeing   = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x1a1a1a, emissiveIntensity: 0.3 });
const crowdInstCalm     = new THREE.InstancedMesh(crowdGeoInst, crowdMatCalm,     crowdMaxCount);
const crowdInstAgitated = new THREE.InstancedMesh(crowdGeoInst, crowdMatAgitated, crowdMaxCount);
const crowdInstRioting  = new THREE.InstancedMesh(crowdGeoInst, crowdMatRioting,  crowdMaxCount);
const crowdInstFleeing  = new THREE.InstancedMesh(crowdGeoInst, crowdMatFleeing,  crowdMaxCount);
crowdInstCalm.count     = 0;
crowdInstAgitated.count = 0;
crowdInstRioting.count  = 0;
crowdInstFleeing.count  = 0;
crowdInstCalm.castShadow     = true;
crowdInstAgitated.castShadow = true;
crowdInstRioting.castShadow  = true;
crowdInstFleeing.castShadow  = true;
scene.add(crowdInstCalm, crowdInstAgitated, crowdInstRioting, crowdInstFleeing);

// Building colors (cyberpunk palette)
const BLDG_COLORS = [0x6a6a88, 0x5a6d88, 0x706080, 0x5a7070, 0x706a5a, 0x806a6a];

// =========================================================================
// Projectile trail system
// =========================================================================
const trailLines = [];
const MAX_TRAILS = 50;

function getOrCreateTrail(idx) {
  if (trailLines[idx]) return trailLines[idx];
  const geo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0)
  ]);
  const mat = new THREE.LineBasicMaterial({ color: 0xffaa00, transparent: true, opacity: 0.7 });
  const line = new THREE.Line(geo, mat);
  scene.add(line);
  trailLines[idx] = line;
  return line;
}

// =========================================================================
// Explosion effect pool
// =========================================================================
const explosionPool = [];
const MAX_EXPLOSIONS = 20;

for (let i = 0; i < MAX_EXPLOSIONS; i++) {
  const mat = new THREE.MeshBasicMaterial({
    color: 0xff4400, wireframe: true, transparent: true, opacity: 0.6
  });
  const mesh = new THREE.Mesh(effectGeo, mat);
  mesh.visible = false;
  scene.add(mesh);
  explosionPool.push({ mesh, active: false, age: 0, maxAge: 0.5, targetRadius: 5 });
}

// Crowd instanced meshes already set up above with crowdInstCalm/Agitated/Rioting/Fleeing

// =========================================================================
// Map features (roads, forests, rivers) — rendered once
// =========================================================================
function renderMapFeatures(features) {
  if (!features || features.length === 0) return;
  const mapGroup = new THREE.Group();
  mapGroup.name = 'map-features';
  const roadMat = new THREE.MeshStandardMaterial({ color: 0x4a4a50, roughness: 0.85 });
  const roadLineMat = new THREE.LineBasicMaterial({ color: 0xffffaa, transparent: true, opacity: 0.25 });
  const riverMat = new THREE.MeshStandardMaterial({ color: 0x0a3a66, roughness: 0.1, transparent: true, opacity: 0.72 });
  const treeMat = new THREE.MeshStandardMaterial({ color: 0x1a5c1a, roughness: 0.9 });
  const trunkMat = new THREE.MeshStandardMaterial({ color: 0x3a2010, roughness: 0.95 });

  for (const f of features) {
    const fx = f.x || 0;
    const fz = f.y || 0;  // server y -> Three.js z
    const fw = f.w || 4;
    const fh = f.h || 4;
    const rot = f.rotation || 0;

    if (f.type === 'road') {
      const len = Math.max(fw, fh), wid = Math.min(fw, fh);
      const geo = new THREE.PlaneGeometry(len, wid);
      const mesh = new THREE.Mesh(geo, roadMat);
      mesh.rotation.x = -Math.PI / 2;
      mesh.rotation.z = rot;
      mesh.position.set(fx, 0.03, fz);
      mesh.receiveShadow = true;
      mapGroup.add(mesh);
    } else if (f.type === 'river' || f.type === 'water') {
      const len = Math.max(fw, fh), wid = Math.min(fw, fh);
      const geo = new THREE.PlaneGeometry(len, wid);
      const mesh = new THREE.Mesh(geo, riverMat);
      mesh.rotation.x = -Math.PI / 2;
      mesh.rotation.z = rot;
      mesh.position.set(fx, 0.04, fz);
      mapGroup.add(mesh);
    } else if (f.type === 'forest') {
      // Scatter tree cones
      const cr = Math.min(fw, fh) * 0.5;
      const count = Math.max(2, Math.floor(cr * cr * 0.15));
      let s = Math.abs((fx * 1000 + fz) | 0) || 1;
      const rng = () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
      for (let t = 0; t < count; t++) {
        const a = rng() * Math.PI * 2, r = rng() * cr * 0.9;
        const tx = fx + Math.cos(a) * r, tz = fz + Math.sin(a) * r;
        const th = 3 + rng() * 3, tr = 0.8 + rng() * 0.6;
        const trunk = new THREE.Mesh(new THREE.CylinderGeometry(tr*0.12, tr*0.18, th*0.35, 5), trunkMat);
        trunk.position.set(tx, th*0.175, tz);
        mapGroup.add(trunk);
        const cone = new THREE.Mesh(new THREE.ConeGeometry(tr, th*0.7, 6), treeMat);
        cone.position.set(tx, th*0.35 + th*0.35, tz);
        mapGroup.add(cone);
      }
    } else if (f.type === 'building' || f.type === 'village') {
      const bh = Math.max(fh, 2);
      const geo = new THREE.BoxGeometry(fw, bh, fw);
      const mat = new THREE.MeshStandardMaterial({ color: 0x444466, roughness: 0.7, emissive: 0x111122, emissiveIntensity: 0.2 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(fx, bh / 2, fz);
      mesh.castShadow = true;
      mapGroup.add(mesh);
    }
  }
  scene.add(mapGroup);
  console.log('[MAP] Rendered ' + features.length + ' map features');
}

// =========================================================================
// Building management
// =========================================================================
let buildingsCreated = false;

function ensureBuildings(structures) {
  if (!structures || structures.length === 0) return;

  for (let i = 0; i < structures.length; i++) {
    const s = structures[i];
    const key = s.id || ('bldg_' + i);

    if (!buildingMeshes[key]) {
      // Create building body
      const w = (s.width || 20);
      const d = (s.depth || 15);
      const h = (s.height || 10);

      const bodyMat = s.destroyed ? matBuildingDestroyed.clone() :
        matBuilding.clone();
      if (!s.destroyed) {
        bodyMat.color.set(BLDG_COLORS[i % BLDG_COLORS.length]);
      }

      const body = new THREE.Mesh(buildingGeo, bodyMat);
      body.scale.set(w, h, d);
      body.position.set(s.x, h / 2, s.y);
      body.castShadow = true;
      body.receiveShadow = true;
      scene.add(body);

      // Roof
      const roof = new THREE.Mesh(buildingRoofGeo, matBuildingRoof.clone());
      roof.scale.set(w + 0.5, 0.5, d + 0.5);
      roof.position.set(s.x, h + 0.25, s.y);
      roof.castShadow = true;
      scene.add(roof);

      // Emissive window strips on sides — bright so buildings visible at night
      const windowMat = new THREE.MeshBasicMaterial({
        color: 0xffdd88, transparent: true, opacity: 0.85
      });
      const floors = Math.floor(h / 4);
      for (let f = 1; f <= floors; f++) {
        const wy = f * 3.5;
        if (wy > h - 1) break;
        // Front face windows
        const numWins = Math.floor(w / 5);
        for (let wi = 0; wi < numWins; wi++) {
          if (Math.random() < 0.35) continue;
          const winGeo = new THREE.PlaneGeometry(1.5, 1.2);
          const win = new THREE.Mesh(winGeo, windowMat);
          win.position.set(
            s.x - w / 2 + 2.5 + wi * 5,
            wy,
            s.y + d / 2 + 0.05
          );
          scene.add(win);
        }
      }

      buildingMeshes[key] = { body, roof, destroyed: !!s.destroyed };
    } else {
      // Update destroyed state
      const bm = buildingMeshes[key];
      if (s.destroyed && !bm.destroyed) {
        bm.body.material = matBuildingDestroyed.clone();
        bm.destroyed = true;
        // Shrink building to show destruction
        const h = bm.body.scale.y;
        bm.body.scale.y = h * 0.3;
        bm.body.position.y = (h * 0.3) / 2;
      }
    }
  }
  buildingsCreated = true;
}

// =========================================================================
// Rain / weather particle system (client-side generation for efficiency)
// =========================================================================
const RAIN_MAX = 3000;
let rainPoints = null;
let rainActive = false;
let rainIntensity = 0;
let rainWindX = 0, rainWindZ = 0;
let rainFallSpeed = 9.0;

function initRainSystem() {
  const positions = new Float32Array(RAIN_MAX * 3);
  // Pre-seed random positions in the sky volume
  for (let i = 0; i < RAIN_MAX; i++) {
    positions[i * 3]     = (Math.random() - 0.5) * 200;  // x
    positions[i * 3 + 1] = Math.random() * 50;            // y (height)
    positions[i * 3 + 2] = (Math.random() - 0.5) * 200;  // z
  }
  const rainGeo = new THREE.BufferGeometry();
  rainGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const rainMat = new THREE.PointsMaterial({
    color: 0xaaccff, size: 0.5, transparent: true, opacity: 0.4,
    blending: THREE.AdditiveBlending, depthWrite: false
  });
  rainPoints = new THREE.Points(rainGeo, rainMat);
  rainPoints.frustumCulled = false;
  rainPoints.visible = false;
  scene.add(rainPoints);
}
initRainSystem();

function updateRain(weatherFx) {
  if (!rainPoints) return;
  const rainData = weatherFx ? weatherFx.rain : null;
  if (!rainData) {
    rainPoints.visible = false;
    rainActive = false;
    rainIntensity = 0;
    return;
  }
  // Server sends rain params: drop_count, and we can read wind from weatherFx
  rainActive = true;
  rainPoints.visible = true;
  const dropCount = Math.min(rainData.drop_count || 1000, RAIN_MAX);
  rainPoints.geometry.setDrawRange(0, dropCount);
  rainIntensity = dropCount / RAIN_MAX;
  rainPoints.material.opacity = Math.min(0.6, 0.2 + rainIntensity * 0.4);
  // Wind from weather_fx (vectors is array of [vx, vz] pairs)
  const wind = weatherFx.wind;
  if (wind && wind.vectors && wind.vectors.length > 0) {
    const w = wind.vectors[0];
    rainWindX = (w[0] || 0) * 0.3;
    rainWindZ = (w[1] || 0) * 0.3;
  } else {
    rainWindX = 0;
    rainWindZ = 0;
  }
  rainFallSpeed = 9.0 + rainIntensity * 3.0;
  // Center rain around camera position
  rainPoints.position.set(camera.position.x, 0, camera.position.z);
}

function tickRain(dt) {
  if (!rainActive || !rainPoints) return;
  const posAttr = rainPoints.geometry.getAttribute('position');
  const positions = posAttr.array;
  const count = rainPoints.geometry.drawRange.count;
  for (let i = 0; i < count; i++) {
    const idx = i * 3;
    positions[idx]     += rainWindX * dt;         // x drift
    positions[idx + 1] -= rainFallSpeed * dt;     // y fall
    positions[idx + 2] += rainWindZ * dt;         // z drift
    // Reset drops that fall below ground
    if (positions[idx + 1] < 0) {
      positions[idx]     = (Math.random() - 0.5) * 200;
      positions[idx + 1] = 40 + Math.random() * 15;
      positions[idx + 2] = (Math.random() - 0.5) * 200;
    }
  }
  posAttr.needsUpdate = true;
}

// =========================================================================
// Unit mesh management
// =========================================================================
function getUnitMesh(id) {
  if (unitMeshes[id]) return unitMeshes[id];

  // Cone for the unit
  const cone = new THREE.Mesh(unitConeGeo, matFriendly.clone());
  cone.castShadow = true;
  scene.add(cone);

  // Ring marker on ground
  const ring = new THREE.Mesh(unitRingGeo, matRing.clone());
  ring.rotation.x = -Math.PI / 2;
  scene.add(ring);

  // Health bar — thin box above unit
  const hpBgGeo = new THREE.BoxGeometry(3, 0.25, 0.3);
  const hpBgMat = new THREE.MeshBasicMaterial({ color: 0x1a1a2e });
  const hpBg = new THREE.Mesh(hpBgGeo, hpBgMat);
  scene.add(hpBg);

  const hpFillGeo = new THREE.BoxGeometry(3, 0.25, 0.3);
  const hpFillMat = new THREE.MeshBasicMaterial({ color: GREEN });
  const hpFill = new THREE.Mesh(hpFillGeo, hpFillMat);
  scene.add(hpFill);

  unitMeshes[id] = { cone, ring, hpBg, hpFill, lastAlliance: null };
  return unitMeshes[id];
}

function updateUnit(u) {
  const m = getUnitMesh(u.id);
  const isDead = u.status === 'dead';
  const isFriendly = u.alliance === 'friendly';

  // Update material color
  if (isDead) {
    m.cone.material.color.set(0x333333);
    m.cone.material.emissive.set(0x000000);
    m.ring.material.color.set(0x333333);
  } else if (isFriendly) {
    m.cone.material.color.set(GREEN);
    m.cone.material.emissive.set(GREEN);
    m.cone.material.emissiveIntensity = 0.15;
    m.ring.material.color.set(GREEN);
  } else {
    m.cone.material.color.set(MAGENTA);
    m.cone.material.emissive.set(MAGENTA);
    m.cone.material.emissiveIntensity = 0.15;
    m.ring.material.color.set(MAGENTA);
  }

  // Position — game uses x,y as ground plane; Three.js uses x,z
  if (isDead) {
    m.cone.position.set(u.x, 0.15, u.y);
    m.cone.rotation.set(0, 0, Math.PI / 2); // Fallen over
    m.cone.scale.set(0.6, 0.6, 0.6);
    m.ring.visible = false;
  } else {
    m.cone.position.set(u.x, 2, u.y);
    m.cone.rotation.set(0, (u.heading || 0), 0);
    m.cone.scale.set(1, 1, 1);
    m.ring.position.set(u.x, 0.05, u.y);
    m.ring.visible = true;
  }

  // Health bar
  const hp = u.hp || 0;
  const maxHp = u.max_hp || 100;
  const ratio = Math.max(0, Math.min(1, hp / maxHp));
  m.hpBg.position.set(u.x, isDead ? 1 : 5.5, u.y);
  m.hpFill.position.set(u.x - 1.5 * (1 - ratio), isDead ? 1.05 : 5.55, u.y);
  m.hpFill.scale.set(ratio, 1, 1);
  m.hpBg.visible = !isDead;
  m.hpFill.visible = !isDead && ratio > 0;

  // HP bar color
  if (ratio > 0.6) m.hpFill.material.color.set(GREEN);
  else if (ratio > 0.3) m.hpFill.material.color.set(YELLOW);
  else m.hpFill.material.color.set(MAGENTA);
}

// =========================================================================
// Vehicle mesh management
// =========================================================================
function getVehicleMesh(id) {
  if (vehicleMeshes[id]) return vehicleMeshes[id];

  const body = new THREE.Mesh(vehBodyGeo, matVehFriendly.clone());
  body.castShadow = true;
  scene.add(body);

  const cabin = new THREE.Mesh(vehCabinGeo, matVehCabin.clone());
  cabin.castShadow = true;
  scene.add(cabin);

  // Headlights
  const hlGeo = new THREE.BoxGeometry(0.3, 0.4, 0.5);
  const hlMat = new THREE.MeshBasicMaterial({ color: 0xffffee });
  const hl1 = new THREE.Mesh(hlGeo, hlMat);
  const hl2 = new THREE.Mesh(hlGeo, hlMat);
  scene.add(hl1);
  scene.add(hl2);

  vehicleMeshes[id] = { body, cabin, hl1, hl2 };
  return vehicleMeshes[id];
}

function updateVehicle(v) {
  const m = getVehicleMesh(v.id);
  const isFriendly = v.alliance === 'friendly';
  const heading = v.heading || 0;

  if (v.destroyed) {
    m.body.material.color.set(0x222222);
    m.body.material.emissive.set(0x110000);
    m.body.material.emissiveIntensity = 0.2;
  } else if (isFriendly) {
    m.body.material.color.set(0x048a6a);
    m.body.material.emissive.set(0x000000);
  } else {
    m.body.material.color.set(0xa01848);
    m.body.material.emissive.set(0x000000);
  }

  // Scale by vehicle class
  const isLarge = (v.vehicle_class === 'humvee' || v.vehicle_class === 'apc' ||
                   v.vehicle_class === 'tank');
  const sc = isLarge ? 1.3 : 1.0;

  m.body.position.set(v.x, 1.0 * sc, v.y);
  m.body.rotation.set(0, heading, 0);
  m.body.scale.set(sc, sc, sc);

  // Cabin offset
  const cosH = Math.cos(heading);
  const sinH = Math.sin(heading);
  const cabOffX = -0.5 * sinH * sc;
  const cabOffZ = -0.5 * cosH * sc;
  m.cabin.position.set(v.x + cabOffX, 2.2 * sc, v.y + cabOffZ);
  m.cabin.rotation.set(0, heading, 0);
  m.cabin.scale.set(sc, sc, sc);

  // Headlights
  const hlFwd = 2.5 * sc;
  const hlSide = 1.0 * sc;
  m.hl1.position.set(
    v.x + hlFwd * sinH + hlSide * cosH,
    0.7 * sc,
    v.y + hlFwd * cosH - hlSide * sinH
  );
  m.hl2.position.set(
    v.x + hlFwd * sinH - hlSide * cosH,
    0.7 * sc,
    v.y + hlFwd * cosH + hlSide * sinH
  );
  m.hl1.rotation.set(0, heading, 0);
  m.hl2.rotation.set(0, heading, 0);

  m.hl1.visible = !v.destroyed;
  m.hl2.visible = !v.destroyed;
}

// =========================================================================
// Projectile rendering
// =========================================================================
let activeProjectileCount = 0;
const projPool = [];
const MAX_PROJ = 50;

for (let i = 0; i < MAX_PROJ; i++) {
  const m = new THREE.Mesh(projGeo, matProjectile.clone());
  m.visible = false;
  scene.add(m);
  projPool.push(m);
}

// Trail lines for projectiles
const projTrailPool = [];
for (let i = 0; i < MAX_PROJ; i++) {
  const geo = new THREE.BufferGeometry();
  const positions = new Float32Array(6); // 2 points
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.LineBasicMaterial({
    color: 0xffaa00, transparent: true, opacity: 0.5
  });
  const line = new THREE.Line(geo, mat);
  line.visible = false;
  scene.add(line);
  projTrailPool.push(line);
}

function updateProjectiles(projectiles) {
  const count = Math.min(projectiles.length, MAX_PROJ);

  for (let i = 0; i < MAX_PROJ; i++) {
    if (i < count) {
      const p = projectiles[i];
      projPool[i].position.set(p.x, 1.5, p.y);
      projPool[i].visible = true;

      // Color from data
      if (p.color) {
        const c = p.color.startsWith('#') ? p.color : ('#' + p.color);
        projPool[i].material.color.set(c);
      }

      // Trail line
      const trail = projTrailPool[i];
      const posAttr = trail.geometry.attributes.position;
      // Current position
      posAttr.array[0] = p.x;
      posAttr.array[1] = 1.5;
      posAttr.array[2] = p.y;
      // Origin (if available) or slightly behind
      const ox = p.origin_x !== undefined ? p.origin_x : p.x - 2;
      const oy = p.origin_y !== undefined ? p.origin_y : p.y - 2;
      posAttr.array[3] = ox;
      posAttr.array[4] = 1.5;
      posAttr.array[5] = oy;
      posAttr.needsUpdate = true;
      trail.visible = true;
      if (p.color) {
        trail.material.color.set(p.color.startsWith('#') ? p.color : ('#' + p.color));
      }
    } else {
      projPool[i].visible = false;
      projTrailPool[i].visible = false;
    }
  }
}

// =========================================================================
// Effect rendering (explosions, smoke, fire)
// =========================================================================
function updateEffects(effects) {
  if (!effects) return;

  for (const e of effects) {
    // Find inactive explosion slot
    for (const exp of explosionPool) {
      if (!exp.active) {
        exp.active = true;
        exp.age = 0;
        exp.maxAge = 0.5;
        exp.targetRadius = (e.radius || 5);
        exp.mesh.position.set(e.x, 1, e.y);
        exp.mesh.visible = true;
        exp.mesh.scale.set(0.1, 0.1, 0.1);

        // Color
        const colorStr = e.color || '#ff4400';
        exp.mesh.material.color.set(
          colorStr.startsWith('#') ? colorStr : ('#' + colorStr)
        );
        break;
      }
    }
  }
}

function tickExplosions(dt) {
  for (const exp of explosionPool) {
    if (!exp.active) continue;
    exp.age += dt;
    const t = exp.age / exp.maxAge;
    if (t >= 1) {
      exp.active = false;
      exp.mesh.visible = false;
      continue;
    }
    // Expand then fade
    const r = exp.targetRadius * Math.sin(t * Math.PI);
    exp.mesh.scale.set(r, r, r);
    exp.mesh.material.opacity = 0.6 * (1 - t);
  }
}

// =========================================================================
// Crowd update — instanced cylinders coloured by mood
// =========================================================================
const _crowdDummy = new THREE.Object3D();
function updateCrowd(crowdData) {
  // Reset instance counts
  crowdInstCalm.count     = 0;
  crowdInstAgitated.count = 0;
  crowdInstRioting.count  = 0;
  crowdInstFleeing.count  = 0;
  if (!crowdData || crowdData.length === 0) return;

  const idxCalm = { v: 0 }, idxAgi = { v: 0 }, idxRiot = { v: 0 }, idxFlee = { v: 0 };

  for (let i = 0; i < crowdData.length; i++) {
    const c = crowdData[i];
    const mood = c.mood || 'calm';
    _crowdDummy.position.set(c.x, 0.9, c.y);
    _crowdDummy.rotation.y = c.heading || 0;
    _crowdDummy.updateMatrix();

    if (mood === 'rioting' && idxRiot.v < crowdMaxCount) {
      crowdInstRioting.setMatrixAt(idxRiot.v++, _crowdDummy.matrix);
    } else if ((mood === 'fleeing' || mood === 'panicked') && idxFlee.v < crowdMaxCount) {
      crowdInstFleeing.setMatrixAt(idxFlee.v++, _crowdDummy.matrix);
    } else if ((mood === 'agitated' || mood === 'uneasy') && idxAgi.v < crowdMaxCount) {
      crowdInstAgitated.setMatrixAt(idxAgi.v++, _crowdDummy.matrix);
    } else if (idxCalm.v < crowdMaxCount) {
      crowdInstCalm.setMatrixAt(idxCalm.v++, _crowdDummy.matrix);
    }
  }

  crowdInstCalm.count     = idxCalm.v;
  crowdInstAgitated.count = idxAgi.v;
  crowdInstRioting.count  = idxRiot.v;
  crowdInstFleeing.count  = idxFlee.v;
  crowdInstCalm.instanceMatrix.needsUpdate     = true;
  crowdInstAgitated.instanceMatrix.needsUpdate = true;
  crowdInstRioting.instanceMatrix.needsUpdate  = true;
  crowdInstFleeing.instanceMatrix.needsUpdate  = true;
}

// =========================================================================
// Delta-frame merge: apply a server diff onto the last full frame
// =========================================================================
const _DIFF_LIST_KEYS = new Set(['units', 'projectiles', 'effects', 'crowd']);

function mergeDiffFrame(base, diff) {
  const merged = Object.assign({}, base);
  for (const key of Object.keys(diff)) {
    if (key === 'frame_type' || key === 'is_diff') continue;
    const val = diff[key];
    if (_DIFF_LIST_KEYS.has(key) && val && typeof val === 'object' && !Array.isArray(val) && val.changed) {
      const prev = Array.isArray(merged[key]) ? merged[key] : [];
      const byId = {};
      for (const item of prev) { if (item && item.id != null) byId[item.id] = item; }
      for (const item of (val.changed || [])) { if (item && item.id != null) byId[item.id] = item; }
      const removedSet = new Set(val.removed || []);
      merged[key] = Object.values(byId).filter(item => !removedSet.has(String(item.id)));
    } else {
      merged[key] = val;
    }
  }
  return merged;
}

// =========================================================================
// Process a frame from the server
// =========================================================================
const seenUnits = new Set();
const seenVehicles = new Set();

function processFrame(f) {
  lastFrame = f;

  // Buildings (only need to create once, then update destruction)
  if (f.destruction && f.destruction.structures) {
    ensureBuildings(f.destruction.structures);
  }

  // Map features (roads, forests, rivers) — render once from first frame or welcome
  if (f.map_features && !window._mapFeaturesRendered) {
    renderMapFeatures(f.map_features);
    window._mapFeaturesRendered = true;
  }

  // Units
  seenUnits.clear();
  for (const u of (f.units || [])) {
    seenUnits.add(u.id);
    updateUnit(u);
  }
  // Hide units no longer in frame
  for (const id in unitMeshes) {
    if (!seenUnits.has(id)) {
      const m = unitMeshes[id];
      m.cone.visible = false;
      m.ring.visible = false;
      m.hpBg.visible = false;
      m.hpFill.visible = false;
    } else {
      const m = unitMeshes[id];
      m.cone.visible = true;
    }
  }

  // Vehicles
  seenVehicles.clear();
  for (const v of (f.vehicles || [])) {
    seenVehicles.add(v.id);
    updateVehicle(v);
  }
  for (const id in vehicleMeshes) {
    if (!seenVehicles.has(id)) {
      const m = vehicleMeshes[id];
      m.body.visible = false;
      m.cabin.visible = false;
      m.hl1.visible = false;
      m.hl2.visible = false;
    } else {
      const m = vehicleMeshes[id];
      m.body.visible = true;
      m.cabin.visible = true;
    }
  }

  // Projectiles
  updateProjectiles(f.projectiles || []);

  // Effects — server sends dict {type,count,positions,colors,sizes,ages}
  // or array of individual effect events; handle both formats
  if (f.effects) {
    if (Array.isArray(f.effects)) {
      if (f.effects.length > 0) updateEffects(f.effects);
    } else if (f.effects.positions && f.effects.positions.length > 0) {
      // Convert particle dict → array of point effects for updateEffects
      const pts = f.effects.positions;
      const cols = f.effects.colors || [];
      const szs = f.effects.sizes || [];
      const efxArr = [];
      for (let i = 0; i < pts.length; i++) {
        efxArr.push({
          x: pts[i][0], y: pts[i][2] != null ? pts[i][2] : pts[i][1],
          radius: (szs[i] || 1) * 0.5,
          color: cols[i] || '#ff4400'
        });
      }
      if (efxArr.length > 0) updateEffects(efxArr);
    }
  }

  // Crowd
  updateCrowd(f.crowd);

  // Weather FX — rain/snow particle system + fog + sky
  if (f.weather_fx) {
    updateRain(f.weather_fx);
    // Fog from weather effects
    const wxFog = f.weather_fx.fog;
    if (wxFog && wxFog.density > 0) {
      scene.fog.density = 0.0015 + wxFog.density * 0.004;
    } else {
      scene.fog.density = 0.0015;
    }
  }
  // Also handle weather dict from renderer (sky_color, ambient, rain, fog)
  if (f.weather && f.weather.fog_density > 0) {
    scene.fog.density = 0.0015 + f.weather.fog_density * 0.003;
  }

  // Kill feed from events
  for (const ev of (f.events || [])) {
    if (ev.type === 'unit_killed') {
      const src = ev.source_label || ev.source_id || '?';
      const tgt = ev.target_label || ev.target_id || '?';
      const alliance = ev.target_alliance || '';
      const color = alliance === 'friendly' ? '#05ffa1' : '#ff2a6d';
      killFeed.unshift({ text: src + ' killed ' + tgt, age: 0, color });
      if (killFeed.length > 12) killFeed.pop();
    } else if (ev.type === 'explosion') {
      killFeed.unshift({ text: 'Explosion at (' + Math.round(ev.x || 0) + ',' + Math.round(ev.y || 0) + ')', age: 0, color: '#ff4400' });
      if (killFeed.length > 12) killFeed.pop();
    }
  }

  // Update HUD
  updateHUD(f);

  // Auto-restart check
  checkAutoRestart(f.stats);
}

// =========================================================================
// HUD updates
// =========================================================================
function updateHUD(f) {
  const st = f.stats || {};
  // environment is a snapshot dict with weather, hour, temperature, etc.
  const env = (typeof st.environment === 'object' && st.environment) ? st.environment : {};

  document.getElementById('hud-tick').textContent = f.tick || 0;
  document.getElementById('hud-time').textContent = (f.sim_time || 0) + 's';
  document.getElementById('hud-friendly').textContent = st.alive_friendly || 0;
  document.getElementById('hud-hostile').textContent = st.alive_hostile || 0;
  document.getElementById('hud-dead').textContent = st.dead || 0;
  document.getElementById('hud-vehicles').textContent = st.total_vehicles || 0;
  document.getElementById('hud-crowd').textContent = st.crowd_count || 0;
  document.getElementById('hud-fires').textContent = st.active_fires || 0;

  // Weather from environment snapshot (e.g. "clear", "rain", "fog")
  const weatherVal = env.weather || '-';
  document.getElementById('hud-weather').textContent = weatherVal.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

  // Time of day from hour field (0-24 float)
  if (env.hour !== undefined) {
    const h = env.hour;
    let todLabel;
    if (h >= 6 && h < 12) todLabel = 'Morning';
    else if (h >= 12 && h < 17) todLabel = 'Afternoon';
    else if (h >= 17 && h < 20) todLabel = 'Evening';
    else todLabel = 'Night';
    const hh = Math.floor(h);
    const mm = Math.floor((h - hh) * 60);
    todLabel += ' ' + String(hh).padStart(2, '0') + ':' + String(mm).padStart(2, '0');
    document.getElementById('hud-tod').textContent = todLabel;
  } else {
    document.getElementById('hud-tod').textContent = '-';
  }

  document.getElementById('hud-preset').textContent = f.preset || '-';

  // Kill feed
  const kfEl = document.getElementById('kf-entries');
  kfEl.innerHTML = killFeed.map(k => {
    k.age += 0.1; // ~10fps
    const cls = k.age > 5 ? 'kf-entry fade' : 'kf-entry';
    return '<div class="' + cls + '" style="color:' + (k.color || '#fcee0a') + '">' + k.text + '</div>';
  }).join('');
  // Prune old
  while (killFeed.length > 0 && killFeed[killFeed.length - 1].age > 15) killFeed.pop();

  // Unit roster
  const rosterEl = document.getElementById('roster-entries');
  const units = f.units || [];
  // Sort: alive first, then by alliance
  const sorted = [...units].sort((a, b) => {
    if (a.status === 'dead' && b.status !== 'dead') return 1;
    if (a.status !== 'dead' && b.status === 'dead') return -1;
    if (a.alliance < b.alliance) return -1;
    if (a.alliance > b.alliance) return 1;
    return 0;
  });

  let rosterHTML = '';
  for (const u of sorted) {
    const hp = u.hp || 0;
    const maxHp = u.max_hp || 100;
    const ratio = Math.max(0, Math.min(1, hp / maxHp));
    const pct = Math.round(ratio * 100);
    const isDead = u.status === 'dead';
    const color = isDead ? '#555' : (u.alliance === 'friendly' ? '#05ffa1' : '#ff2a6d');
    const hpColor = ratio > 0.6 ? '#05ffa1' : (ratio > 0.3 ? '#fcee0a' : '#ff2a6d');
    const labelCls = isDead ? 'ros-label ros-dead' : 'ros-label';
    rosterHTML += '<div class="ros-unit">' +
      '<span class="' + labelCls + '" style="color:' + color + '">' + (u.label || u.id) + '</span>' +
      '<div class="ros-hp-bar"><div class="ros-hp-fill" style="width:' + pct + '%;background:' + hpColor + '"></div></div>' +
      '<span style="color:' + hpColor + ';font-size:10px">' + (isDead ? 'KIA' : pct + '%') + '</span>' +
      '</div>';
  }
  rosterEl.innerHTML = rosterHTML;

  // Hide splash once we have data
  const splash = document.getElementById('splash');
  if (splash && f.tick > 0) splash.style.display = 'none';
}

// =========================================================================
// API functions (exposed to global scope via window)
// =========================================================================
window.startGame = function() {
  const preset = document.getElementById('preset-select').value;
  fetch('/api/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ preset })
  }).then(r => r.json()).then(d => {
    console.log('Game started:', d);
    connectWS();
    const splash = document.getElementById('splash');
    if (splash) splash.style.display = 'none';
  });
};

window.pauseGame = function() {
  fetch('/api/pause', { method: 'POST' })
    .then(r => r.json())
    .then(d => console.log('Pause:', d));
};

window.getAAR = function() {
  fetch('/api/aar')
    .then(r => r.json())
    .then(d => {
      document.getElementById('aar-content').textContent = JSON.stringify(d, null, 2);
      document.getElementById('aar-overlay').style.display = 'block';
    });
};

// =========================================================================
// WebSocket connection
// =========================================================================
function connectWS() {
  if (ws) ws.close();
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(proto + '://' + location.host + '/ws');
  ws.onmessage = (ev) => {
    try {
      const frame = JSON.parse(ev.data);
      // Delta-frame handling: merge diffs into last known full frame
      if (frame.frame_type === 'diff' && lastFrame) {
        const merged = mergeDiffFrame(lastFrame, frame);
        processFrame(merged);
      } else {
        processFrame(frame);
      }
    } catch (e) {
      console.error('Frame parse error:', e);
    }
  };
  ws.onclose = () => console.log('WS disconnected');
  ws.onerror = (e) => console.error('WS error:', e);
}

// =========================================================================
// Fetch presets on load
// =========================================================================
fetch('/api/presets').then(r => r.json()).then(d => {
  const sel = document.getElementById('preset-select');
  sel.innerHTML = '';
  for (const p of (d.world_presets || ['urban_combat'])) {
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = p.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    sel.appendChild(opt);
  }
}).catch(() => {});

// =========================================================================
// Keyboard controls
// =========================================================================
window.addEventListener('keydown', (e) => {
  if (e.key === 'r' || e.key === 'R') {
    // Reset camera
    camera.position.set(40, 120, 140);
    controls.target.set(30, 0, 30);
    controls.update();
  }
});

// =========================================================================
// Resize handler
// =========================================================================
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// =========================================================================
// Animation loop
// =========================================================================
function animate() {
  requestAnimationFrame(animate);

  const now = performance.now();
  const dt = Math.min((now - lastTime) / 1000, 0.05);
  lastTime = now;

  // FPS
  frameCount++;
  fpsTimer += dt;
  if (fpsTimer >= 0.5) {
    fpsDisplay = Math.round(frameCount / fpsTimer);
    frameCount = 0;
    fpsTimer = 0;
    document.getElementById('fps-counter').textContent = fpsDisplay + ' FPS';
  }

  // Tick explosion animations
  tickExplosions(dt);

  // Animate rain particles between server frames
  tickRain(dt);

  // Make unit cones gently bob
  const bobTime = now * 0.002;
  for (const id in unitMeshes) {
    const m = unitMeshes[id];
    if (m.cone.visible && m.cone.scale.x > 0.5) {
      m.cone.position.y = 2 + Math.sin(bobTime + m.cone.position.x * 0.1) * 0.15;
    }
  }

  controls.update();
  renderer.render(scene, camera);
}

animate();

// =========================================================================
// Auto-connect WebSocket — game auto-starts on page load
// =========================================================================
(function autoConnect() {
  // Check if a game is already running, then immediately connect WS
  fetch('/api/status').then(r => r.json()).then(d => {
    if (d.running) {
      connectWS();
      const splash = document.getElementById('splash');
      if (splash) splash.style.display = 'none';
    }
  }).catch(() => {});
})();

// =========================================================================
// Auto-restart — when all friendlies die, restart after 5 seconds
// =========================================================================
let _restartScheduled = false;
function checkAutoRestart(stats) {
  if (!stats) return;
  const friendlyAlive = stats.alive_friendly || 0;
  const totalUnits = stats.total_units || 0;
  if (totalUnits > 0 && friendlyAlive === 0 && !_restartScheduled) {
    _restartScheduled = true;
    const msgEl = document.createElement('div');
    msgEl.id = 'restart-msg';
    msgEl.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);' +
      'background:rgba(10,10,15,0.92);border:1px solid #ff2a6d;color:#ff2a6d;' +
      'padding:24px 40px;font-family:monospace;font-size:20px;z-index:9999;text-align:center;' +
      'border-radius:4px;box-shadow:0 0 30px #ff2a6d44;';
    msgEl.innerHTML = '<div style="font-size:28px;font-weight:bold;margin-bottom:8px">ALL UNITS KIA</div>' +
      '<div style="color:#00f0ff;font-size:14px">Restarting in 5 seconds...</div>';
    document.body.appendChild(msgEl);
    setTimeout(() => {
      _restartScheduled = false;
      const existing = document.getElementById('restart-msg');
      if (existing) existing.remove();
      const preset = document.getElementById('preset-select').value;
      fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preset })
      }).then(r => r.json()).then(() => {
        connectWS();
      });
    }, 5000);
  }
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the game server."""
    import uvicorn
    import os
    port = int(os.environ.get("SIM_PORT", "9090"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
