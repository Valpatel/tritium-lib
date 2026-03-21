# sim_engine/debug/

Debug data streams and diagnostic overlays for the sim engine.

## Files

| File | Purpose |
|------|---------|
| `streams.py` | `DebugStreams` — named ring buffers for per-subsystem debug output |

## Key Concepts

- `DebugStreams` collects structured log events (pathfinding decisions, combat rolls,
  morale changes) into named ring buffers without blocking the sim tick.
- The game server exposes these via `/debug/streams` so the frontend can render
  overlays without modifying sim code.
- All streams are opt-in: subsystems check `streams.enabled(name)` before writing,
  so there is zero overhead in production mode.
