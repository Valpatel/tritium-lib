# sim_engine/world/

Top-level simulation integrator — wires every subsystem into one tick loop.

## Files

| File | Purpose |
|------|---------|
| `_world.py` | `World`, `WorldConfig`, `WorldBuilder`, `WORLD_PRESETS` — main sim orchestrator |
| `pathfinding.py` | `PathfindingSystem` — A* wrapper, path cache, unit steering integration |
| `grid_pathfinder.py` | Low-level grid A* with terrain-profile costs |
| `vision.py` | `VisionSystem` — line-of-sight queries, fog-of-war state per unit |
| `cover.py` | `CoverSystem` — cover positions, suppression zones |
| `sensors.py` | `SensorSystem` — unit detection radii, stealth modifier, reveal events |

## Key Concepts

- `World.tick(dt)` is the single entry point called by the game server each frame.
- It advances physics, AI, combat, pathfinding, and vision in a fixed order.
- `WorldBuilder` constructs a `World` from a `WorldConfig`/preset without needing
  to know individual subsystem constructors.
