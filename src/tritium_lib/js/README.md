# tritium_lib/js — browser sim runtime (served, not imported)

The vanilla ES-module tree that the `sim_engine` **city demos** load in the
browser. This is JavaScript served over HTTP -- **not importable Python** -- so
it correctly carries no `__init__.py` and wheels exclude it by design.

**Where you are:** `tritium-lib/src/tritium_lib/js/`

> **Do not delete.** This looks orphaned but is load-bearing. The demos reach it
> through the git-tracked symlink `sim_engine/demos/js -> ../../js`. Deleting
> this tree breaks the symlink and the city3d demos. (Verified 2026-07-11.)

## Who loads it

| Consumer | Imports from here |
|----------|-------------------|
| `sim_engine/demos/city3d-clean.html` | `./js/sim/core/city-builder.js`, `sim/core/world.js`, `sim/weather.js` |
| `sim_engine/demos/city3d/inspect.js` | `../js/sim/identity.js` (loaded by `city3d.html`, served by `serve_city3d.py` on :8888) |

This is a second, textually-diverged copy of some `web/sim/` modules. `web/` is
the separate tree used by SC's frontend and tests; **this** tree is the runtime
for the lib-local city demos. Keep that distinction -- they are not the same
files and are not meant to be merged.

## Layout

| Path | Contents |
|------|----------|
| `sim/` | Simulation logic -- `idm.js` (car-following), `mobil.js` (lane changes), `movement.js`, `protest-engine.js` + `protest-scenario.js` (Epstein riot model), `daily-routine.js` + `schedule-executor.js` (NPC routines), `traffic-controller.js`, `road-network.js`, `weather.js`, `identity.js`, `inventory.js`, `entity.js`, `state.js`, `sensor-bridge.js` (bridge sim entities to the tracking pipeline) |
| `sim/core/` | World construction -- `world.js`, `city-builder.js`, `osm-city-builder.js`, `floor-plan.js`, `ground-unit.js`, `path.js`, `spatial-hash.js` (broad-phase neighbor queries) |
| `sim/units/` | Per-agent kinematics -- `pedestrian.js`, `vehicle.js` |
| `sim/rendering/` | `instanced-renderer.js` -- batched draw for large agent counts |
| `render/` | Scene rendering -- `buildings.js`, `vehicles.js`, `people.js`, `effects.js` |

## Run the demos

```bash
python -m tritium_lib.sim_engine.demos.serve_city3d   # serves city3d.html on :8888
```

Then open the served page; the browser pulls these modules through the symlink.

## Key algorithms (mirrors the Python sim_engine)

IDM car-following, MOBIL lane changes, Epstein protest/riot emergence, NPC
daily routines, spatial-hash neighbor lookup. Same models the production
tracking + fusion pipeline consumes -- the sim is the test harness.

**Parent:** [../README.md](../README.md)
