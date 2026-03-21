# Tritium-Lib Status

Current state of the shared library as of Wave 186.

## Wave 186 Baselines

| Metric | Count |
|--------|-------|
| Test files | 227 |
| Tests passing | 2,569 (Wave 165 baseline; full suite not re-run this wave) |
| Python source files (`src/`) | 343 |
| Sim engine Python files | 146 |
| Branch | `dev` |

## Sim Engine

146 Python files across `sim_engine/` covering:
- `ai/` — steering (pure Python + NumPy vectorized), pathfinding, combat AI, formations, behavior trees, squad tactics
- `behavior/` — NPC routines, unit missions, unit states
- `combat/` — combat loop, squads, weapons
- `core/` — entity, movement, spatial, state machine, NPC thinker
- `game/` — game modes, morale, difficulty, crowd density, ambient, stats
- `physics/` — collision, vehicle physics
- `world/` — cover, grid pathfinder, sensors, vision
- `effects/` — particles, weapons FX
- `audio/` — spatial audio
- `debug/` — debug streams
- `demos/` — game_server, city3d, perf tests, 30+ standalone test scripts

## Key Packages

| Package | Purpose |
|---------|---------|
| `models/` | 262+ Pydantic classes across 97+ files — canonical data contracts |
| `sim_engine/` | Full tactical simulation (146 files) |
| `store/` | 9 persistent data stores (all inherit BaseStore) |
| `intelligence/` | RL metrics, fusion metrics, position estimator |
| `classifier/` | Multi-signal BLE/WiFi device classifier |
| `data/` | 11 JSON fingerprint/lookup databases (933 BLE company IDs) |
| `sdk/` | Addon SDK — AddonBase, BaseRunner, GeoJSON layers |
| `graph/` | KuzuDB entity/relationship graph |

## Redundancy Notes (Wave 186 audit)

- `ai/steering.py` vs `ai/steering_np.py`: intentional split — pure Python for composable forces, NumPy variant for 500+ agents at 10Hz. Not redundant.
- No TODO/FIXME/HACK comments found anywhere in `sim_engine/`.
- `game_server.py`: clean, no dead code markers.
