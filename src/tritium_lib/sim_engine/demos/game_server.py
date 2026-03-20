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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ---------------------------------------------------------------------------
# CitySim backend — real sim_engine city simulation
# ---------------------------------------------------------------------------

from tritium_lib.sim_engine.demos.city_sim_backend import CitySim

# ---------------------------------------------------------------------------
# Import EVERY sim_engine module
# ---------------------------------------------------------------------------

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
# 17. Artillery — fire support
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
        self.collision: CollisionWorld | None = None
        self.artillery: ArtilleryEngine | None = None
        self.abilities: AbilityEngine | None = None
        self.status_effects: StatusEffectEngine | None = None
        self.objectives: ObjectiveEngine | None = None
        self.territory: TerritoryControl | None = None
        self.influence: InfluenceMap | None = None
        self.running: bool = False
        self.paused: bool = False
        self.tick_count: int = 0
        self.preset: str = ""
        self.start_time: float = 0.0


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
        .set_time(hour=2.0)  # night
        .enable_destruction(True)
        .enable_crowds(True)
        .enable_los(True)
        .enable_vehicles(True)
        .add_terrain_noise(octaves=4, amplitude=8.0, seed=42)
        .set_weather(Weather.RAIN)
        # Friendly squad: 4 infantry + 1 sniper + 1 medic
        # Positioned near the center for quick engagement
        .spawn_friendly_squad(
            "Alpha",
            ["infantry", "infantry", "infantry", "infantry", "sniper", "medic"],
            (200.0, 200.0),
            spacing=4.0,
        )
        # Hostile squad: 6 infantry + 2 heavy
        # Close enough for combat within ~10 seconds
        .spawn_hostile_squad(
            "Tango",
            ["infantry"] * 6 + ["heavy", "heavy"],
            (280.0, 280.0),
            spacing=4.0,
        )
        # 4. Vehicles — humvee (friendly), technical (hostile)
        .add_vehicle("humvee", "Humvee-Alpha", "friendly", (190.0, 190.0))
        .add_vehicle("technical", "Technical-1", "hostile", (290.0, 290.0))
        # 10. Destruction — 4 buildings
        .add_building((200.0, 200.0), (20, 15, 10), "concrete")
        .add_building((220.0, 180.0), (15, 10, 8), "concrete")
        .add_building((180.0, 220.0), (12, 8, 6), "wood")
        .add_building((240.0, 240.0), (10, 10, 5), "brick")
        # 9. Crowd — 50 civilians in market area
        .add_crowd((250.0, 250.0), 50, 30.0, CrowdMood.CALM)
    )

    # Try to load geospatial terrain layer if cached data exists
    try:
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from pathlib import Path
        tl = TerrainLayer()
        # Check common cache locations
        for ao_id in ["demo_area", "default"]:
            if tl.load_cached(ao_id):
                builder.load_terrain_layer(tl)
                break
    except Exception:
        pass

    gs.world = builder.build()

    # 4b. Drone (friendly quadcopter)
    drone_v = gs.world.spawn_vehicle("quadcopter", "Recon-1", "friendly", (105.0, 95.0))
    drone_v.altitude = 50.0
    drone_ctrl = DroneController(drone_v)
    drone_ctrl.orbit((250.0, 250.0), radius=80.0, altitude=50.0)
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
        TrapType.IED_ROADSIDE, (300.0, 300.0), "hostile",
        trigger_type="proximity", damage=120.0, blast_radius=8.0,
    )

    # --- 19. Civilian ---
    gs.civilians = CivilianSimulator()
    gs.civilians.spawn_population((250.0, 250.0), 50, 40.0, with_infrastructure=True)

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
        position=(350.0, 350.0),
        radius=80.0,
        jammer_type=JammerType.COMMUNICATIONS,
        alliance="hostile",
    ))

    # --- 29. Supply Routes ---
    gs.supply_routes = SupplyRouteEngine()
    gs.supply_routes.add_supply_line(SupplyLine(
        line_id="main_supply",
        waypoints=[(50.0, 50.0), (95.0, 95.0), (100.0, 100.0)],
        source_cache_id="cache_alpha",
        alliance="friendly",
    ))
    for uid, unit in gs.world.units.items():
        if unit.alliance == Alliance.FRIENDLY:
            gs.supply_routes.register_unit(uid, alliance="friendly")

    # 14. Collision world — unit and vehicle collisions
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
        alliance="friendly", position=(180.0, 180.0), heading=0.7,
        min_range=mortar_tmpl["min_range"], max_range=mortar_tmpl["max_range"],
        damage=mortar_tmpl["damage"], blast_radius=mortar_tmpl["blast_radius"],
        reload_time=mortar_tmpl["reload_time"], ammo=mortar_tmpl["max_ammo"],
        max_ammo=mortar_tmpl["max_ammo"], accuracy_cep=mortar_tmpl["accuracy_cep"],
        crew=mortar_tmpl["crew"],
    ))

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
    gs.influence = InfluenceMap(width=500, height=500, cell_size=10.0)
    gs.territory = TerritoryControl()
    gs.territory.add_control_point(ControlPoint(
        point_id="cp_center", name="Central Objective",
        position=(250.0, 250.0), capture_radius=30.0,
    ))
    gs.territory.add_control_point(ControlPoint(
        point_id="cp_east", name="Eastern Approach",
        position=(350.0, 250.0), capture_radius=25.0,
    ))

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

    # 1. World tick (units, squads, vehicles, projectiles, destruction, crowd)
    frame = gs.world.tick(dt)

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

    # 14. Collision tick
    if gs.collision is not None:
        for uid, u in gs.world.units.items():
            if u.is_alive():
                gs.collision.update(uid, position=u.position)
        for vid, v in gs.world.vehicles.items():
            gs.collision.update(vid, position=v.position)
        collisions = gs.collision.check_all()
        if collisions:
            frame["collisions"] = [
                {"a": c.entity_a, "b": c.entity_b, "type": c.collision_type}
                for c in collisions[:20]  # cap for frame size
            ]

    # 15. Artillery tick
    if gs.artillery is not None:
        arty_events = gs.artillery.tick(dt)
        frame["artillery"] = gs.artillery.to_three_js()

    # 15. Abilities tick
    if gs.abilities is not None:
        ability_events = gs.abilities.tick(dt)
        # Collect ability visual effects per unit
        ability_fx = {}
        for uid in gs.world.units:
            fx = gs.abilities.to_three_js(uid)
            if fx:
                ability_fx[uid] = fx
        if ability_fx:
            frame["abilities"] = ability_fx

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
        frame["influence"] = gs.influence.to_three_js()
    if gs.territory is not None:
        terr_units = {
            uid: (u.position, "friendly" if u.alliance == Alliance.FRIENDLY else "hostile")
            for uid, u in gs.world.units.items() if u.is_alive()
        }
        terr_events = gs.territory.tick(dt, terr_units)
        frame["territory"] = gs.territory.to_dict()

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

    return frame


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Tritium Sim Engine Demo", version="1.0.0")
_game: GameState = GameState()
_ws_clients: list[WebSocket] = []
_game_task: asyncio.Task | None = None

# ---------------------------------------------------------------------------
# CitySim state — separate from the tactical GameState
# ---------------------------------------------------------------------------
_city_sim: CitySim | None = None
_city_task: asyncio.Task | None = None
_city_mode: bool = False  # When True, WS streams city frames instead of game frames


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the game client HTML page."""
    return HTMLResponse(content=GAME_HTML, status_code=200)


@app.get("/city", response_class=HTMLResponse)
async def city_view() -> HTMLResponse:
    """Serve the city3d.html demo with WebSocket bridge injected.

    Auto-starts the CitySim backend so the frontend receives real sim frames
    instead of running its own JS-side simulation.
    """
    global _city_sim, _city_task, _city_mode

    # Auto-start CitySim when someone visits /city
    if _city_sim is None:
        _city_sim = CitySim(seed=42)
        _city_sim.setup()
        _city_mode = True
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
    global _game, _game_task
    if body is None:
        body = {}
    preset = body.get("preset", "urban_combat")
    _game = build_full_game(preset)
    _game.running = True
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
    global _city_sim, _city_task, _city_mode
    if body is None:
        body = {}
    seed = body.get("seed", None)
    width = body.get("width", 500.0)
    height = body.get("height", 400.0)
    hour = body.get("hour", 10.0)

    _city_sim = CitySim(width=width, height=height, seed=seed, hour=hour)
    _city_sim.setup()
    _city_mode = True

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


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """WebSocket for streaming frame data at 10 fps."""
    await ws.accept()
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
    """Background asyncio task: tick the game at 10 fps."""
    dt = 0.1  # 10 fps
    while _game.running:
        if not _game.paused and _game.world is not None:
            frame = game_tick(_game, dt)
            payload = json.dumps(frame, default=str)
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
    """Background asyncio task: tick the CitySim at 10 fps."""
    dt = 0.1  # 10 fps
    while _city_mode and _city_sim is not None:
        frame = _city_sim.tick(dt)
        payload = json.dumps(frame, default=str)
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
    ):
        if getattr(gs, attr, None) is not None:
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
body { background: #0a0a0f; overflow: hidden; font-family: 'Courier New', monospace; }
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
const VOID_BG = 0x0a0a0f;

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
const crowdPoints = null;    // single Points object
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
camera.position.set(MAP_SIZE * 0.5 + 80, 180, MAP_SIZE * 0.5 + 160);
camera.lookAt(MAP_SIZE / 2, 0, MAP_SIZE / 2);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
document.body.prepend(renderer.domElement);

// OrbitControls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(MAP_SIZE / 2, 0, MAP_SIZE / 2);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 30;
controls.maxDistance = 900;
controls.maxPolarAngle = Math.PI / 2.1;
controls.update();

// =========================================================================
// Lights
// =========================================================================
scene.add(new THREE.AmbientLight(0x8899bb, 0.45));

const sunLight = new THREE.DirectionalLight(0xffeedd, 1.0);
sunLight.position.set(200, 400, 100);
sunLight.castShadow = true;
sunLight.shadow.mapSize.set(2048, 2048);
sunLight.shadow.camera.left = -MAP_SIZE;
sunLight.shadow.camera.right = MAP_SIZE;
sunLight.shadow.camera.top = MAP_SIZE;
sunLight.shadow.camera.bottom = -MAP_SIZE;
scene.add(sunLight);

scene.add(new THREE.HemisphereLight(0x00f0ff, 0x112244, 0.4));

// Cyan accent lights at corners
for (const [cx, cz] of [[50, 50], [450, 50], [50, 450], [450, 450]]) {
  const pl = new THREE.PointLight(0x00f0ff, 0.3, 150);
  pl.position.set(cx, 30, cz);
  scene.add(pl);
}

// =========================================================================
// Ground plane + grid
// =========================================================================
const groundGeo = new THREE.PlaneGeometry(MAP_SIZE + 100, MAP_SIZE + 100);
const groundMat = new THREE.MeshStandardMaterial({
  color: 0x111118, roughness: 0.9, metalness: 0.1
});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.set(MAP_SIZE / 2, -0.05, MAP_SIZE / 2);
ground.receiveShadow = true;
scene.add(ground);

// Grid lines
const gridHelper = new THREE.GridHelper(MAP_SIZE, 10, 0x1a1a2e, 0x12121a);
gridHelper.position.set(MAP_SIZE / 2, 0.01, MAP_SIZE / 2);
scene.add(gridHelper);

// Fine grid
const fineGrid = new THREE.GridHelper(MAP_SIZE, 50, 0x0e0e14, 0x0e0e14);
fineGrid.position.set(MAP_SIZE / 2, 0.005, MAP_SIZE / 2);
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
const matBuilding = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0.05 });
const matBuildingRoof = new THREE.MeshStandardMaterial({ color: 0x1a1a2e, roughness: 0.9 });
const matBuildingDestroyed = new THREE.MeshStandardMaterial({ color: 0x3a1010, roughness: 0.9 });
const matRing = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0.5 });
const matCrowd = new THREE.PointsMaterial({ color: YELLOW, size: 1.2, sizeAttenuation: true });

// Building colors (cyberpunk palette)
const BLDG_COLORS = [0x2a2a3e, 0x1e2d3e, 0x2e1e3e, 0x1a2e2e, 0x2e2a1e, 0x3e2e2e];

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

// =========================================================================
// Crowd particle system
// =========================================================================
const crowdMaxCount = 200;
const crowdPositions = new Float32Array(crowdMaxCount * 3);
const crowdGeo = new THREE.BufferGeometry();
crowdGeo.setAttribute('position', new THREE.BufferAttribute(crowdPositions, 3));
const crowdMesh = new THREE.Points(crowdGeo, matCrowd);
scene.add(crowdMesh);

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

      // Emissive window strips on sides
      const windowMat = new THREE.MeshBasicMaterial({
        color: 0xffdd66, transparent: true, opacity: 0.6
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
// Crowd update
// =========================================================================
function updateCrowd(crowdData) {
  if (!crowdData) {
    crowdGeo.setDrawRange(0, 0);
    return;
  }
  const count = Math.min(crowdData.length, crowdMaxCount);
  for (let i = 0; i < count; i++) {
    crowdPositions[i * 3] = crowdData[i].x;
    crowdPositions[i * 3 + 1] = 0.8;
    crowdPositions[i * 3 + 2] = crowdData[i].y;
  }
  crowdGeo.attributes.position.needsUpdate = true;
  crowdGeo.setDrawRange(0, count);
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

  // Effects — only pass NEW effects each frame
  if (f.effects && f.effects.length > 0) {
    updateEffects(f.effects);
  }

  // Crowd
  updateCrowd(f.crowd);

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
}

// =========================================================================
// HUD updates
// =========================================================================
function updateHUD(f) {
  const st = f.stats || {};
  const env = st.environment || {};

  document.getElementById('hud-tick').textContent = f.tick || 0;
  document.getElementById('hud-time').textContent = (f.sim_time || 0) + 's';
  document.getElementById('hud-friendly').textContent = st.alive_friendly || 0;
  document.getElementById('hud-hostile').textContent = st.alive_hostile || 0;
  document.getElementById('hud-dead').textContent = st.dead || 0;
  document.getElementById('hud-vehicles').textContent = st.total_vehicles || 0;
  document.getElementById('hud-crowd').textContent = st.crowd_count || 0;
  document.getElementById('hud-fires').textContent = st.active_fires || 0;
  document.getElementById('hud-weather').textContent = env.weather || '-';
  document.getElementById('hud-tod').textContent = env.time_of_day || (env.hour !== undefined ? env.hour.toFixed(1) + 'h' : '-');
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
      processFrame(frame);
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
    camera.position.set(MAP_SIZE * 0.5 + 80, 180, MAP_SIZE * 0.5 + 160);
    controls.target.set(MAP_SIZE / 2, 0, MAP_SIZE / 2);
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
