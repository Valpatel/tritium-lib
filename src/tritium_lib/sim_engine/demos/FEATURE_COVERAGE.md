# City3D / game.html — Feature Coverage Audit

**Updated**: Wave 183 (2026-03-21)
**Status**: 18 fully demonstrated, 22 partial, 22 missing (out of 62 modules)

## FULLY DEMONSTRATED (18 modules)

These modules have working visual output in `game.html` or the demo data pipeline.

| Module | What is shown |
|--------|---------------|
| `units.py` | Infantry + heavy units, instanced mesh, alliance colors, heading rotation |
| `vehicles.py` | APC/tank box mesh, separate instanced pool, scale parameter |
| `renderer.py` | Three.js scene, ACESFilmic tonemapping, shadow maps, fog |
| `terrain.py` | Ground plane, grid helper, axis markers |
| `mapgen.py` | Procedural road/forest/building features via `map_features` payload |
| `pathfinding.py` | Units move toward targets each tick (steering-based) |
| `ambient.py` | Background events injected via demo tick (supply drops, reinforcements) |
| `civilian.py` | Crowd instanced mesh (sphere), mood-color coding (calm/agitated/rioting/panicked) |
| `crowd.py` | Crowd positions driven from server frame `crowd` array |
| `effects/particles.py` | Explosion particle system (80 particles/pool, velocity+gravity) |
| `weather_fx.py` | Rain particle system (4000 pts), sky color, fog density, ambient light |
| `destruction.py` | Building damage color gradient (white → yellow → orange → red) |
| `buildings.py` | Dynamic building spawn + damage visualization |
| `scoring.py` | Wave counter, score, K/D ratio, accuracy, MVP in game-over screen |
| `medical.py` | Health bar per unit (color: green > 60%, yellow > 30%, red below) |
| `debug/streams.py` | FPS counter, connection status, phase display |
| `hud.py` | Kill feed, friendly roster, unit info panel, objectives panel |
| `gis_layers (terrain_geojson)` | Geospatial terrain polygon overlay via `terrain_geojson` payload |

## PARTIALLY DEMONSTRATED (22 modules — need enhancement)

Key gaps: steering behaviors (only seek/flee, not pursue/evade/flock), combat_ai
(no flanking/suppression visualization), squad formations (no formation geometry
drawn), status_effects (no buff/debuff indicators), morale (state tracked but not
displayed), animation (lerp used in camera but no easing library visible), replay
(no record/playback controls), territory (legend rendered but no heatmap polygons),
objectives (panel exists but hardcoded empty), damage (armor/critical hits computed
but not visualized), environment (day/night cycle triggers weather but no sky gradient),
soundtrack (no audio), spawner (wave spawning works but no spawn-point markers),
collision (computed server-side, not visualized), intel (no fog-of-war overlay),
campaign (single-phase only), multiplayer (WebSocket architecture ready but no
session sync), logistics (supply values tracked but no resource bars), morale
(UnitState.morale read but no on-screen indicator), electronic_warfare (no jamming
overlay), comms (no signal range circles), scenario (preset system works, no
in-game selector).

## NOT DEMONSTRATED (22 modules — need integration)

| Priority | Module | What to add |
|----------|--------|-------------|
| HIGH | `commander.py` | Amy narration text in kill feed / HUD panel |
| HIGH | `objectives.py` | Populated objective markers + progress bars |
| HIGH | `territory.py` | Control-point influence heatmap polygons |
| MEDIUM | `artillery.py` | Mortar barrage arc + impact zone indicator |
| MEDIUM | `economy.py` | Resource bars in HUD (police budget / supply points) |
| MEDIUM | `fortifications.py` | Police barricades + riot barrier objects |
| MEDIUM | `asymmetric.py` | IED / booby trap placement markers |
| MEDIUM | `replay.py` | Record button + playback scrubber |
| MEDIUM | `multiplayer.py` | Session sync, split-screen or networked control |
| MEDIUM | `soundtrack.py` | Background music + reactive intensity |
| MEDIUM | `status_effects.py` | Buff/debuff icon display above units |
| LOW | `cyber.py` | Drone camera feed + jamming overlay |
| LOW | `ai/behavior_tree.py` | Debug viz of AI decision tree |
| LOW | `ai/behavior_profiles.py` | Personality display on unit info panel |
| LOW | `ai/rf_signatures.py` | RF detection radius overlay |
| LOW | `naval.py` | River/harbor patrol boats |
| LOW | `air_combat.py` | Police helicopter unit type |
| LOW | `electronic_warfare.py` | Comm jamming visual during riot |
| LOW | `supply_routes.py` | Route lines between supply depots |
| LOW | `morale.py` | On-screen morale bar per unit |
| LOW | `campaign.py` | Multi-phase scenario progression UI |
| LOW | `intel.py` | Fog-of-war cell overlay |
