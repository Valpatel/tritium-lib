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
        .spawn_friendly_squad(
            "Alpha",
            ["infantry", "infantry", "infantry", "infantry", "sniper", "medic"],
            (100.0, 100.0),
            spacing=4.0,
        )
        # Hostile squad: 6 infantry + 2 heavy
        .spawn_hostile_squad(
            "Tango",
            ["infantry"] * 6 + ["heavy", "heavy"],
            (400.0, 400.0),
            spacing=4.0,
        )
        # 4. Vehicles — humvee (friendly), technical (hostile)
        .add_vehicle("humvee", "Humvee-Alpha", "friendly", (90.0, 90.0))
        .add_vehicle("technical", "Technical-1", "hostile", (410.0, 410.0))
        # 10. Destruction — 4 buildings
        .add_building((200.0, 200.0), (20, 15, 10), "concrete")
        .add_building((220.0, 180.0), (15, 10, 8), "concrete")
        .add_building((180.0, 220.0), (12, 8, 6), "wood")
        .add_building((240.0, 240.0), (10, 10, 5), "brick")
        # 9. Crowd — 50 civilians in market area
        .add_crowd((250.0, 250.0), 50, 30.0, CrowdMood.CALM)
    )

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

    # Add metadata
    frame["tick"] = gs.tick_count
    frame["sim_time"] = round(gs.world.sim_time, 2)
    frame["preset"] = gs.preset
    frame["stats"] = gs.world.stats()

    return frame


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Tritium Sim Engine Demo", version="1.0.0")
_game: GameState = GameState()
_ws_clients: list[WebSocket] = []
_game_task: asyncio.Task | None = None


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the game client HTML page."""
    return HTMLResponse(content=GAME_HTML, status_code=200)


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


def _count_active_modules(gs: GameState) -> int:
    """Count how many subsystem modules are initialized."""
    count = 0
    for attr in (
        "world", "scoring", "detection", "comms", "medical",
        "logistics", "naval", "air_combat", "engineering",
        "asymmetric", "civilians", "intel", "diplomacy", "campaign",
    ):
        if getattr(gs, attr, None) is not None:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Inline game.html
# ---------------------------------------------------------------------------

GAME_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Tritium Sim Engine Demo</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0f; color: #00f0ff; font-family: 'Courier New', monospace; }
  #hud { position: fixed; top: 10px; left: 10px; z-index: 10; }
  #hud button { background: #1a1a2e; color: #00f0ff; border: 1px solid #00f0ff;
    padding: 6px 14px; margin: 2px; cursor: pointer; font-family: inherit; }
  #hud button:hover { background: #00f0ff; color: #0a0a0f; }
  #stats { position: fixed; top: 10px; right: 10px; z-index: 10;
    background: rgba(10,10,15,0.85); padding: 10px; border: 1px solid #00f0ff;
    max-width: 300px; font-size: 12px; white-space: pre-wrap; }
  #canvas-container { width: 100vw; height: 100vh; }
  canvas { width: 100%; height: 100%; display: block; }
  #log { position: fixed; bottom: 10px; left: 10px; z-index: 10;
    background: rgba(10,10,15,0.85); padding: 8px; border: 1px solid #1a1a2e;
    max-height: 150px; overflow-y: auto; font-size: 11px; width: 400px; }
</style>
</head>
<body>
<div id="hud">
  <button onclick="startGame()">START</button>
  <button onclick="pauseGame()">PAUSE</button>
  <button onclick="getAAR()">AAR</button>
</div>
<div id="stats">Waiting for game...</div>
<div id="canvas-container"><canvas id="c"></canvas></div>
<div id="log"></div>
<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const statsEl = document.getElementById('stats');
const logEl = document.getElementById('log');
let ws = null;
let lastFrame = null;

function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
resize();

function log(msg) {
  const d = document.createElement('div');
  d.textContent = msg;
  logEl.appendChild(d);
  logEl.scrollTop = logEl.scrollHeight;
}

function startGame() {
  fetch('/api/start', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({preset:'urban_combat'})})
    .then(r => r.json()).then(d => { log('Game started: ' + JSON.stringify(d)); connectWS(); });
}

function pauseGame() {
  fetch('/api/pause', {method:'POST'}).then(r=>r.json()).then(d=>log('Pause: '+JSON.stringify(d)));
}

function getAAR() {
  fetch('/api/aar').then(r=>r.json()).then(d=>{ statsEl.textContent = JSON.stringify(d, null, 1); });
}

function connectWS() {
  if (ws) ws.close();
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(proto + '://' + location.host + '/ws');
  ws.onmessage = (ev) => {
    try { lastFrame = JSON.parse(ev.data); renderFrame(lastFrame); } catch(e) {}
  };
  ws.onclose = () => { log('WS disconnected'); };
}

function renderFrame(f) {
  const W = canvas.width, H = canvas.height;
  const mapW = 500, mapH = 500;
  const sx = W / mapW, sy = H / mapH;
  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = '#1a1a2e';
  ctx.lineWidth = 0.5;
  for (let x = 0; x < mapW; x += 50) { ctx.beginPath(); ctx.moveTo(x*sx,0); ctx.lineTo(x*sx,H); ctx.stroke(); }
  for (let y = 0; y < mapH; y += 50) { ctx.beginPath(); ctx.moveTo(0,y*sy); ctx.lineTo(W,y*sy); ctx.stroke(); }

  // Buildings
  if (f.destruction) {
    (f.destruction.structures || []).forEach(s => {
      ctx.fillStyle = s.destroyed ? '#3a1010' : '#2a2a3e';
      ctx.fillRect(s.x*sx-10, s.y*sy-10, 20, 20);
    });
  }

  // Units
  (f.units || []).forEach(u => {
    ctx.fillStyle = u.alliance === 'friendly' ? '#05ffa1' : '#ff2a6d';
    if (u.status === 'dead') ctx.fillStyle = '#333';
    ctx.beginPath();
    ctx.arc(u.x*sx, u.y*sy, 4, 0, Math.PI*2);
    ctx.fill();
    ctx.fillStyle = '#00f0ff';
    ctx.font = '9px monospace';
    ctx.fillText(u.label || u.id, u.x*sx+6, u.y*sy-4);
  });

  // Vehicles
  (f.vehicles || []).forEach(v => {
    ctx.fillStyle = v.alliance === 'friendly' ? '#05ffa1' : '#ff2a6d';
    if (v.destroyed) ctx.fillStyle = '#333';
    ctx.fillRect(v.x*sx-5, v.y*sy-3, 10, 6);
  });

  // Projectiles
  (f.projectiles || []).forEach(p => {
    ctx.fillStyle = p.color || '#ffaa00';
    ctx.fillRect(p.x*sx-1, p.y*sy-1, 2, 2);
  });

  // Effects
  (f.effects || []).forEach(e => {
    ctx.strokeStyle = e.color || '#ff4400';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(e.x*sx, e.y*sy, e.radius*Math.max(sx,sy)*0.5, 0, Math.PI*2);
    ctx.stroke();
  });

  // Crowd
  (f.crowd || []).forEach(c => {
    ctx.fillStyle = '#fcee0a';
    ctx.fillRect(c.x*sx-1, c.y*sy-1, 2, 2);
  });

  // HUD
  const st = f.stats || {};
  statsEl.textContent = [
    'Tick: ' + (f.tick||0) + '  Time: ' + (f.sim_time||0) + 's',
    'Friendly: ' + (st.alive_friendly||0) + '  Hostile: ' + (st.alive_hostile||0),
    'Dead: ' + (st.dead||0) + '  Vehicles: ' + (st.total_vehicles||0),
    'Crowd: ' + (st.crowd_count||0) + '  Fires: ' + (st.active_fires||0),
    'Env: ' + JSON.stringify(st.environment||{}),
  ].join('\\n');
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
