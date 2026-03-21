# sim_engine/game/

Game-state management — wave controller, difficulty, scoring, and morale.

## Files

| File | Purpose |
|------|---------|
| `game_mode.py` | `GameMode` — setup/active/victory/defeat FSM, wave spawning, scoring |
| `morale.py` | `MoraleSystem` — per-unit morale, panic triggers, rally mechanics |
| `difficulty.py` | `DifficultyScaler` — dynamic enemy stat scaling based on player performance |
| `stats.py` | `SessionStats` — kill/death/accuracy tracking, K/D computation |
| `ambient.py` | `AmbientEventSystem` — background world events (supply drops, reinforcements) |
| `crowd_density.py` | `CrowdDensityTracker` — civilian density heat map by zone |

## Key Concepts

- `GameMode` is the top-level clock: it decides when waves spawn and when the game ends.
- Morale and difficulty are read each tick; they feed back into unit AI thresholds.
- `SessionStats` is serialized into the final game-over screen payload.
