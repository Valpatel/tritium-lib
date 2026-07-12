# sim_engine/demos/ — runnable proofs

**Parent:** [`../README.md`](../README.md) · **Family:** Simulation

Standalone scripts that prove the engine works — visually, over a browser, and
under load. Each imports only from `tritium_lib.sim_engine` (no tritium-sc
dependency), so this directory is the honest answer to "does the library
actually run on its own?" 11 runnable modules + a 33-file `tests/` suite.

Everything needs `PYTHONPATH=src`. The server demos additionally need
`fastapi` + `uvicorn`; the flocking/city/perf demos want `numpy`.

## The flagship: `game_server.py`

A FastAPI app that composes **every** sim_engine subsystem and streams frames
to a Three.js frontend over WebSocket — the fullest demonstration of the stack
(effects, audio, game rules, debug overlay, physics, and the full `ai/` toolkit).

```bash
cd tritium-lib
PYTHONPATH=src python3 -m tritium_lib.sim_engine.demos.game_server
# default port: SIM_PORT env, else 9090   (game_server.py:4688)
```

Its `CitySim` backend lives in `city_sim_backend.py` (`game_server.py:53`
imports it).

## Runnable demos

| Module | What it shows | Run |
|--------|---------------|-----|
| `game_server` | Full engine → Three.js over WebSocket (FastAPI) | `-m …demos.game_server` (port 9090) |
| `city_sim_backend` | `CitySim` — the Python city backend the server drives | imported by `game_server` |
| `demo_full` | GTA-style full city: real streets, buildings, traffic, pedestrians | `-m …demos.demo_full` |
| `integrated_demo` | City sim → **sensor-fusion pipeline** (the production tie-in) | `-m …demos.integrated_demo` |
| `serve_city3d` | Static HTTP server for the 3-D city page (`city3d.html`) | `-m …demos.serve_city3d` (port 8888) |
| `demo_steering` | 50 boids: separation / alignment / cohesion + obstacle avoidance | `-m …demos.demo_steering [--headless --duration N]` |
| `demo_city` | 50 residents on daily routines; cars drive & park; 10× clock | `-m …demos.demo_city [--headless]` |
| `demo_perf` | Scaling benchmark 50→1000 agents: tick time, FPS, memory | `-m …demos.demo_perf [--duration N]` |
| `demo_rf` | Every person/car emits BLE/WiFi/TPMS; MAC rotation vs persistent TPMS | `-m …demos.demo_rf [--headless]` |
| `perf_test` | Runs the server, starts a riot, records telemetry → HTML report | `-m …demos.perf_test` → `/tmp/tritium_perf_report.html` |
| `test_report` | Test-coverage report generator for sim_engine → HTML | `-m …demos.test_report` → `/tmp/tritium_test_report.html` |

Common flags for the visual demos: `--headless` (skip window, print stats),
`--duration N` (seconds, default 30).

## Static assets

`city3d.html`, `city3d-clean.html`, `game.html` + 13 JS modules are the
browser frontends `serve_city3d` / `game_server` serve. They render the JSON
frames the backend streams.

## `tests/` — 33 files

`demos/tests/` is a pytest suite that drives the city3d demo end-to-end and
asserts real behavior: `test_city3d_wave_spawning`, `_crowd_dynamics`,
`_traffic_lights`, `_building_damage`, `_line_of_sight`, `_fog_of_war`,
`_replay`, `_ew_jamming`, `_triage`, `_logistics_economy`, and more, plus
standalone unit tests (`test_behavior_tree_nodes`, `test_cyber_standalone`,
`test_economy_standalone`, `test_game_ai_behaviors`). This is where "the game
IS the test harness" is literally true — the demos are exercised as tests.

## Why this directory matters (North Star)

Per the North Star, the simulator *is* the validation harness. `integrated_demo`
is the clearest expression: it feeds a running city sim into the same sensor-
fusion pipeline the production stack uses. If the demo's fused picture is clean,
the fusion code is healthy.

## Dependencies

- **All demos:** `PYTHONPATH=src`.
- **Server demos** (`game_server`, `perf_test`): `fastapi` + `uvicorn`.
- **Flocking / city / perf:** `numpy` (optional `matplotlib` for plots).
