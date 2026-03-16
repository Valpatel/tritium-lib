# Tritium Simulation Engine

A comprehensive tactical simulation engine for the Tritium operating picture.
Covers land, sea, and air combat with terrain, weather, crowds, logistics,
medical, intelligence, communications, and scoring systems. Pure Python with
no rendering or game-engine dependencies -- the backend computes state and
the frontend renders it.

## Architecture

```
                         +------------------+
                         |      World       |
                         |  (tick loop)     |
                         +--------+---------+
                                  |
          +-----------+-----------+-----------+-----------+
          |           |           |           |           |
     +----+----+ +----+----+ +---+----+ +----+----+ +----+----+
     |  Units  | | Arsenal | | Terrain| |  Crowd  | | Scenario|
     | create_ | | ARSENAL | |HeightMap| |CrowdSim | | waves,  |
     | unit()  | | damage  | |  LOS   | |  mood   | |objectives|
     +---------+ +---------+ +--------+ +---------+ +---------+
          |           |           |           |           |
     +----+----+ +----+----+ +---+----+ +----+----+ +----+----+
     |Vehicles | |  Naval  | |  Air   | |Detection| |  Intel  |
     | drones, | | ships,  | |aircraft| | sensors,| |fog of   |
     | convoys | |torpedoes| |missiles| |signatures| war,recon|
     +---------+ +---------+ +--------+ +---------+ +---------+
          |           |           |           |           |
     +----+----+ +----+----+ +---+----+ +----+----+ +----+----+
     |Medical  | |Logistics| |Comms   | |Destruct.| |Scoring  |
     | triage, | | supply, | | radio, | | fire,   | |achieve- |
     | evac    | | routes  | |jamming | | debris  | |ments    |
     +---------+ +---------+ +--------+ +---------+ +---------+
          |           |           |
     +----+----+ +----+----+ +---+----+
     |Fortify  | |Asymmetric| |Civilian|
     | mines,  | | traps,  | |infra,  |
     | bunkers | |guerrilla| |collateral|
     +---------+ +---------+ +---------+

     Cross-cutting: AI (steering, combat, squads, tactics)
                    Effects (particles, muzzle flash, explosions)
                    Audio (spatial sound, attenuation, Doppler)
                    Physics (collision, rigid bodies, vehicle dynamics)
                    Environment (weather, time of day)
                    Renderer (layer descriptions for frontend)
                    Debug (frame inspection streams)
```

## Modules

| Module | File | Description |
|--------|------|-------------|
| **World** | `world.py` | Top-level container with tick loop, WorldBuilder, WORLD_PRESETS |
| **Scenario** | `scenario.py` | Scenario config, timed events, waves, objectives |
| **Renderer** | `renderer.py` | Render-layer descriptions consumed by frontend |
| **AI** | `ai/` | Steering behaviors, pathfinding, ambient NPCs, combat AI, squads, tactics |
| **Units** | `units.py` | Infantry/vehicle unit types, stats, UNIT_TEMPLATES |
| **Vehicles** | `vehicles.py` | Ground vehicles, drones, convoys |
| **Arsenal** | `arsenal.py` | 35+ weapons with realistic stats, projectile simulation |
| **Damage** | `damage.py` | Hit resolution, explosions, burst fire |
| **Naval** | `naval.py` | Ship combat, torpedoes, formations |
| **Air Combat** | `air_combat.py` | Aircraft dogfights, missiles, anti-air |
| **Terrain** | `terrain.py` | Procedural heightmaps, line of sight, cover, movement cost |
| **Environment** | `environment.py` | Weather simulation, time of day, environmental effects |
| **Crowd** | `crowd.py` | Civilian crowd dynamics, mood propagation, panic |
| **Destruction** | `destruction.py` | Structure damage, fire spread, debris physics |
| **Detection** | `detection.py` | Sensor types, signature profiles, detection probability |
| **Comms** | `comms.py` | Radio communications, channels, jamming |
| **Medical** | `medical.py` | Injury types, triage categories, casualty evacuation |
| **Logistics** | `logistics.py` | Supply types, caches, routes, consumption |
| **Fortifications** | `fortifications.py` | Bunkers, barriers, minefields |
| **Asymmetric** | `asymmetric.py` | Guerrilla cells, traps, IEDs |
| **Civilian** | `civilian.py` | Civilian population, infrastructure, collateral damage |
| **Intel** | `intel.py` | Intelligence gathering, fog of war, recon missions, fusion |
| **Scoring** | `scoring.py` | Score categories, 24 achievements, unit/team scorecards |
| **Effects** | `effects/` | Particle systems, explosions, muzzle flash, tracers |
| **Audio** | `audio/` | Spatial audio math, attenuation, Doppler, reverb |
| **Physics** | `physics/` | 2D collision detection, rigid bodies, vehicle dynamics |
| **Debug** | `debug/` | Debug frame streams and overlays |

## Quick Start

```python
from tritium_lib.sim_engine import (
    World, WorldBuilder, WORLD_PRESETS,
    Unit, create_unit, UNIT_TEMPLATES,
    ARSENAL, resolve_attack,
    HeightMap, LineOfSight,
    Environment, Weather,
)

# --- Option 1: Use a preset ---
world = WORLD_PRESETS["urban_combat"]()
world.tick(0.016)  # advance one frame at 60 Hz

# --- Option 2: Build from scratch ---
builder = WorldBuilder(width=500, height=500, name="My Scenario")
builder.add_heightmap(seed=42, octaves=4)
builder.add_weather(Weather.RAIN)

# Add units
friendly = create_unit("infantry", alliance="friendly", x=100, y=100)
hostile = create_unit("sniper", alliance="hostile", x=400, y=300)
builder.add_unit(friendly)
builder.add_unit(hostile)

world = builder.build()

# Simulate
for _ in range(600):  # 10 seconds at 60 Hz
    world.tick(1 / 60)

# Check results
for unit in world.units:
    print(f"{unit.template} [{unit.alliance}] hp={unit.hp}")
```

## Presets

### World Presets (`WORLD_PRESETS`)

| Preset | Description |
|--------|-------------|
| `urban_combat` | Dense urban environment with buildings, cover, and CQB engagement |
| `open_field` | Open terrain with minimal cover, long-range engagements |
| `riot_response` | Crowd control scenario with civilian crowds and response forces |
| `convoy_ambush` | Vehicle convoy attacked on a road, ambush tactics |
| `drone_strike` | Aerial drone operations with ground targets |

### Unit Templates (`UNIT_TEMPLATES`)

| Template | Type | Description |
|----------|------|-------------|
| `infantry` | Rifleman | Standard infantry with assault rifle |
| `sniper` | Marksman | Long-range precision shooter |
| `heavy` | Heavy | Machine gunner / heavy weapons |
| `medic` | Support | Combat medic with healing ability |
| `scout` | Recon | Fast, stealthy reconnaissance |
| `drone` | Aerial | Small quadcopter drone |
| `turret` | Static | Fixed defensive emplacement |
| `civilian` | Non-combatant | Unarmed civilian |

### Crowd Scenarios (`CROWD_SCENARIOS`)

| Scenario | Description |
|----------|-------------|
| `peaceful_protest` | Calm crowd that can be agitated |
| `riot` | Already-agitated crowd with high aggression |
| `stampede` | Panicked crowd fleeing a threat |
| `standoff` | Tense confrontation between crowd and forces |

## Import Examples

```python
# Everything from one import
from tritium_lib.sim_engine import *

# Or pick what you need by domain
from tritium_lib.sim_engine import World, WorldBuilder, WORLD_PRESETS
from tritium_lib.sim_engine import Unit, create_unit, Alliance, UNIT_TEMPLATES
from tritium_lib.sim_engine import Weapon, ARSENAL, ProjectileSimulator
from tritium_lib.sim_engine import resolve_attack, resolve_explosion, DamageTracker
from tritium_lib.sim_engine import HeightMap, LineOfSight, CoverMap, MovementCost
from tritium_lib.sim_engine import Weather, TimeOfDay, Environment
from tritium_lib.sim_engine import CrowdSimulator, CrowdMood, CROWD_SCENARIOS
from tritium_lib.sim_engine import DestructionEngine, Structure, Fire
from tritium_lib.sim_engine import DetectionEngine, Sensor, SignatureProfile
from tritium_lib.sim_engine import NavalCombatEngine, ShipState, Torpedo, create_ship
from tritium_lib.sim_engine import AirCombatEngine, AircraftState, Missile
from tritium_lib.sim_engine import CommsSimulator, Radio, Jammer
from tritium_lib.sim_engine import MedicalEngine, Injury, TriageCategory
from tritium_lib.sim_engine import LogisticsEngine, SupplyCache
from tritium_lib.sim_engine import EngineeringEngine, Fortification, Mine
from tritium_lib.sim_engine import AsymmetricEngine, Trap, GuerrillaCell
from tritium_lib.sim_engine import CivilianSimulator, Infrastructure, CollateralDamage
from tritium_lib.sim_engine import IntelEngine, FogOfWar, IntelReport
from tritium_lib.sim_engine import ScoringEngine, Achievement, ACHIEVEMENTS
from tritium_lib.sim_engine import EffectsManager, explosion, muzzle_flash
from tritium_lib.sim_engine import SoundEvent, stereo_pan, doppler_factor
from tritium_lib.sim_engine import Vec2, seek, flee, flock, Squad, TacticsEngine
```

## Dependencies

- **Required**: None (pure Python, stdlib only)
- **Optional**: `numpy` for vectorized steering (`SteeringSystem`, `AmbientSimulatorNP`)
