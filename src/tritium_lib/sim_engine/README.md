# Tritium Simulation Engine

A pure-Python tactical simulation engine that computes the entire
battlespace -- units, weapons, terrain, weather, crowds, logistics,
medical, intel, economy, and more -- then streams JSON frames to a
Three.js frontend over WebSocket. No rendering or game-engine
dependencies on the backend; the server produces state, the browser
draws it.

## Subpackages

| Package | Files | Key Classes | Purpose |
|---------|------:|-------------|---------|
| `core/` | 6 | `SimulationTarget`, `StateMachine`, `SpatialIndex`, `MovementController` | Entity dataclass, state machines, spatial queries, movement |
| `ai/` | 14 | `SteeringSystem`, `BehaviorTree`, `CombatAI`, `StrategicAI`, `RoadNetwork` | Steering behaviors, pathfinding, behavior trees, squad tactics, formations |
| `unit_types/` | 18 | `UnitType`, `Drone`, `Rover`, `Tank`, `Person`, `Camera` | Type registry with `CombatStats`, movement category, perception cones |
| `behavior/` | 5 | `UnitBehaviors`, `UnitMissions`, `UnitStates`, `NPCBehavior` | Per-unit-type combat AI: turret tracking, drone strafing, flanking, cover |
| `game/` | 6 | `StatsTracker`, `DifficultyScaler`, `GameMode`, `CrowdDensity` | Game-mode rules, difficulty scaling, crowd density, morale |
| `world/` | 6 | `World`, `WorldBuilder`, `WorldConfig`, `CoverMap`, `VisionSystem` | Top-level tick loop, world presets, cover/vision, grid pathfinding |
| `combat/` | 3 | `CombatSystem`, `SquadManager`, `WeaponLoadout` | Hit resolution, squad coordination, weapon assignment |
| `effects/` | 2 | `EffectsManager`, `ParticleSystem` | Explosions, muzzle flash, tracers, smoke, fire, debris, sparks |
| `physics/` | 2 | `CollisionWorld`, `VehicleDynamics` | 2D collision detection, rigid-body vehicle physics |
| `audio/` | 1 | `SoundEvent`, `distance_attenuation`, `stereo_pan` | Spatial audio math for Web Audio |
| `debug/` | 1 | `DebugOverlay` | Frame inspection data streams |
| `demos/` | 42 | `game_server`, `CitySim`, demo scripts, HTML frontends | Runnable demos and performance tests |

There are also **46 top-level modules** covering domain-specific
subsystems (one file each):

`abilities`, `air_combat`, `animation`, `arsenal`, `artillery`,
`asymmetric`, `buildings`, `campaign`, `civilian`, `collision`,
`commander`, `comms`, `crowd`, `cyber`, `damage`, `destruction`,
`detection`, `economy`, `electronic_warfare`, `environment`,
`event_bus`, `factions`, `fortifications`, `hud`, `intel`,
`logistics`, `mapgen`, `medical`, `morale`, `multiplayer`, `naval`,
`objectives`, `renderer`, `replay`, `scenario`, `scoring`,
`soundtrack`, `spawner`, `status_effects`, `supply_routes`,
`telemetry`, `terrain`, `territory`, `units`, `vehicles`,
`weather_fx`

**Total: 152 Python modules** (excluding `__init__.py` files).

## Running the Demo

The primary demo is `game_server.py` -- a FastAPI app that streams
frames at 10 fps to a Three.js frontend.

```bash
cd tritium-lib
PYTHONPATH=src python3 -m tritium_lib.sim_engine.demos.game_server
```

Open `http://localhost:8090` in a browser. The server starts a full
tactical simulation and pushes JSON frames over a WebSocket.

### Other demos

| Demo | Command | What it does |
|------|---------|-------------|
| City life | `python3 -m tritium_lib.sim_engine.demos.demo_city` | NPC daily routines, traffic, pedestrians |
| Full city | `python3 -m tritium_lib.sim_engine.demos.demo_full` | GTA-style simulation with streets, buildings, traffic |
| Steering | `python3 -m tritium_lib.sim_engine.demos.demo_steering` | Steering behavior visualization |
| RF sigs | `python3 -m tritium_lib.sim_engine.demos.demo_rf` | RF signature visualization |
| Perf bench | `python3 -m tritium_lib.sim_engine.demos.demo_perf` | Performance benchmark for AI subsystems |
| City 3D | `python3 -m tritium_lib.sim_engine.demos.serve_city3d` | 3D city demo with Three.js |

All demos require `PYTHONPATH=src` (or an editable install of
tritium-lib). The game server and city3d demos need `fastapi` and
`uvicorn`.

## Frame Data Format

Every tick, `game_tick()` returns a JSON dict that the frontend
renders. The core keys come from `World.tick()`:

```json
{
  "tick": 42,
  "time": 4.2,
  "units": [
    {
      "id": "drone_1",
      "type": "drone",
      "x": 120.5,
      "y": 80.3,
      "z": 15.0,
      "heading": 1.57,
      "health": 0.85,
      "alliance": "friendly",
      "icon": "D",
      "alive": true
    }
  ],
  "projectiles": [
    { "x": 100, "y": 90, "dx": 1, "dy": 0, "type": "bullet", "color": "#ffaa00" }
  ],
  "effects": [
    { "type": "explosion", "x": 200, "y": 150, "radius": 5, "age": 0.1 }
  ],
  "weather": { "wind_speed": 3.2, "wind_direction": 0.8, "rain": 1.0 },
  "time_of_day": { "hour": 14.5 },
  "crowd": [ { "x": 50, "y": 60, "mood": "calm", "color": "#05ffa1" } ],
  "events": [ { "type": "unit_killed", "unit_id": "hostile_3", "killer_id": "turret_1" } ],
  "formations": [],
  "vehicles": []
}
```

The `game_server.py` layer adds many more keys from the subsystem
engines. Each subsystem provides a `to_three_js()` method that
returns its own JSON-serializable dict:

| Frame key | Source | Content |
|-----------|--------|---------|
| `destruction` | `DestructionEngine` | Damaged structures, fire, debris |
| `detection` | `DetectionEngine` | Sensor cones, detection events |
| `comms` | `CommsSimulator` | Radio channels, jamming zones |
| `medical` | `MedicalEngine` | Injury status, triage, evacuation |
| `logistics` | `LogisticsEngine` | Supply caches, routes, consumption |
| `naval` | `NavalCombatEngine` | Ships, torpedoes, formations |
| `air_combat` | `AirCombatEngine` | Aircraft, missiles, anti-air |
| `intel` | `IntelEngine` | Fog of war, recon, intelligence |
| `morale` | `MoraleEngine` | Unit morale levels, events |
| `electronic_warfare` | `EWEngine` | Jammers, cyber attacks |
| `supply_routes` | `SupplyRouteEngine` | Supply lines, convoys |
| `weather_fx` | `WeatherFXEngine` | Rain, snow, fog particles |
| `spawner` | `SpawnerEngine` | Wave spawn points, composition |
| `artillery` | `ArtilleryEngine` | Fire support, impacts |
| `narration` | `BattleNarrator` | Amy's battle commentary |
| `abilities` | `AbilityEngine` | Active abilities, cooldowns |
| `status_effects` | `StatusEffectEngine` | Suppression, burning, healing |
| `objectives` | `ObjectiveEngine` | Mission goals, progress |
| `influence` | `InfluenceMap` | Territorial control summary |
| `economy` | `EconomyEngine` | Resources, build queues, tech |
| `cyber` | `CyberWarfareEngine` | Cyber assets, attacks |
| `hud` | `HUDEngine` | Minimap, compass, kill feed |
| `buildings` | `RoomClearingEngine` | Interior rooms, CQB events |
| `soundtrack` | `SoundtrackEngine` | Audio cues for Web Audio |
| `event_timeline` | `SimEventBus` | Last N events for timeline UI |
| `campaign` | `Campaign` | Campaign progress, missions |
| `fortifications` | `EngineeringEngine` | Bunkers, barriers, minefields |
| `civilians` | `CivilianSimulator` | Civilian population state |

## How to Add a New Unit Type

1. Create a file under `unit_types/` in the appropriate category
   (`robots/`, `people/`, or `sensors/`).

2. Subclass `UnitType` from `unit_types/base.py` and set all
   `ClassVar` fields:

```python
# unit_types/robots/my_bot.py
from tritium_lib.sim_engine.unit_types.base import (
    CombatStats, MovementCategory, UnitType,
)

class MyBot(UnitType):
    type_id = "my_bot"
    display_name = "My Bot"
    icon = "B"
    category = MovementCategory.GROUND
    speed = 2.5
    drain_rate = 0.001
    vision_radius = 40.0
    placeable = True
    combat = CombatStats(
        health=100, max_health=100,
        weapon_range=30.0, weapon_cooldown=2.0, weapon_damage=15,
        is_combatant=True,
    )
```

3. Import the new class in `unit_types/__init__.py` so the registry
   discovers it.

4. Optionally add an entry to `behavior/behaviors.py`
   `_WEAPON_TYPES` dict if the unit needs a specific weapon mapping.

The `UnitType` base provides `MovementCategory` (STATIONARY, GROUND,
FOOT, AIR), `CombatStats` (health, weapon range/cooldown/damage),
perception fields (vision radius, cone angle, sweep RPM), and power
drain rate. The flat `SimulationTarget` entity carries the runtime
state; the `UnitType` class defines the archetype.

Existing categories and examples:
- `robots/`: drone, rover, tank, apc, turret, heavy_turret,
  missile_turret, scout_drone
- `people/`: person, hostile_person, hostile_leader,
  hostile_vehicle, animal, vehicle, swarm_drone
- `sensors/`: camera, motion_sensor

## How to Add a New Behavior

Unit behaviors live in `behavior/behaviors.py`. The `UnitBehaviors`
class runs each tick and decides what each combatant does based on
its `asset_type`.

1. Add a new method to `UnitBehaviors` (e.g., `_tick_my_bot()`).

2. Call it from the main `tick()` dispatch, keyed on the unit's
   `asset_type` string.

3. Behaviors sit ABOVE the waypoint/movement system. They can:
   - Set temporary engagement headings
   - Trigger weapon fire through `CombatSystem`
   - Apply dodge offsets to position
   - Initiate flanking, group rush, cover-seeking, or retreat

For more complex AI, use the behavior tree system in
`ai/behavior_tree.py`. Pre-built trees exist for patrol, friendly,
hostile, and civilian archetypes:

```python
from tritium_lib.sim_engine.ai.behavior_tree import make_patrol_tree

tree = make_patrol_tree()
tree.tick(context)  # returns BTStatus.SUCCESS / FAILURE / RUNNING
```

## Dependencies

- **Required:** None (pure Python, stdlib only)
- **Optional:** `numpy` for vectorized steering
  (`SteeringSystem`, `AmbientSimulatorNP`), `fastapi` + `uvicorn`
  for demo servers
