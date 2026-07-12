# Tritium-Lib — Shared Platform Library

> **North Star:** *Build a fun simulator for the purpose of testing and validating the software stack that drives independent machines.*

This library is the seam where the game and the production system are literally the same code. The simulation engine, target tracker, fusion correlator, and addon SDK ship together. When the in-game wave spawner calls them, they behave identically to when a real sensor fleet calls them. That is the point. See top-level [../CLAUDE.md](../CLAUDE.md) and `project_north_star.md`.

The foundation library for the entire Tritium ecosystem. Models, target tracking, sensor fusion, simulation engine, addon SDK, and shared JS frontend components. Python 3.12+ backend with vanilla JS modules for city simulation and UI.

SC imports directly from `tritium_lib` — no wrappers, no adapters, no shims.

## What belongs here (the bin boundary)

**Belongs in lib:** reusable models, algorithms, wire contracts, protocols, the
sim engine, geometry, planners, the addon SDK, shared JS — anything a *second*
caller reuses. **Litmus: it must import cleanly on a bare aarch64 Jetson with
only light deps (numpy / pydantic / opencv).**

**Never belongs in lib** — this is a hard invariant, not a preference:
`import isaacsim` / `pxr` / `rospy`, anything that *requires* torch or a
framework runtime, FastAPI routers, on-robot code. Those poison the "imports on
the robot brain" guarantee. Heavy simulator/tool runtimes → a **`tritium-addons`**
addon (e.g. Isaac Sim → `tritium-addons/isaac_sim`); on-robot ROS2 → **`tritium-edge/ros2`**.
When unsure where a file goes, see the copper-roof rule in
[`../CLAUDE.md`](../CLAUDE.md) → `docs/ARCHITECTURE.md` (parent repo).

**Parent context:** See [../CLAUDE.md](../CLAUDE.md) for full system architecture and conventions.

## Git Conventions

- **No co-authors on commits** — never add "Co-Authored-By" lines
- Remote: `git@github.com:Valpatel/tritium-lib.git`
- Copyright: Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC / AGPL-3.0

## Package Map (by capability area)

> Removed packages (12 zero-consumer packages trimmed 2026-04-29,
> ~18,500 LOC) are recorded in the parent repo's
> `docs/audits/REMOVED-PACKAGES.md` with the resurrection policy.
> This table lists only what exists on disk (verified 2026-06-11).

### Core Infrastructure
| Package | Purpose |
|---------|---------|
| `models` | 101 Pydantic model files — THE canonical data contracts for everything |
| `events` | Thread-safe and async pub/sub event bus |
| `mqtt` | Topic hierarchy `tritium/{site}/{domain}/{device}/{type}` |
| `auth` | JWT tokens, API key management |
| `config` | Pydantic base settings for service configuration |
| `store` | Persistent data stores (BLE, targets, async, time-series) |
| `sdk` | Addon SDK: AddonBase, AddonContext, DeviceRegistry, protocols, BaseRunner, GeoJSON layers |

### Intelligence & Fusion
| Package | Purpose |
|---------|---------|
| `intelligence` | Position estimator, RL metrics, fusion metrics, acoustic classifier, anomaly detection, behavior analysis |
| `fusion` | Multi-sensor target fusion (BLE + camera + WiFi -> unique UUID) |
| `inference` | ML inference pipelines, model management |
| `classifier` | Multi-signal BLE/WiFi device type classification with fingerprint databases |

### Tracking & Situational Awareness
| Package | Purpose |
|---------|---------|
| `tracking` | Target tracking, correlation, track management |
| `sitaware` | **Capstone module.** Situational awareness engine — fuses all subsystems into a unified operating picture |
| `incident` | Incident lifecycle: detected → investigating → responding → resolved (consumed by SC sitaware/forensics/geo routers) |
| `geo` | Coordinate transforms, camera projection, haversine |
| `indoor` | Indoor positioning, WiFi fingerprinting, floorplan mapping |

### Sensors & Signals
| Package | Purpose |
|---------|---------|
| `signals` | Signal processing, spectrum analysis |
| `sdr` | SDRDevice base, SDRInfo, SweepResult |
| `protocols` | Protocol decoders and handlers |
| `nodes` | Sensor node management |

### Simulation Engine
| Package | Purpose |
|---------|---------|
| `sim_engine` | Full city simulation: IDM car-following, MOBIL lane changes, Bezier intersection turns, NPC daily routines, Epstein protest/riot model, weather, traffic lights, sensor bridge |
| `synthetic` | Synthetic data generation for training and testing |
| `scenarios` | Pre-built simulation scenarios |

### JS Simulation (web/sim/)
Browser-side city simulation: IDM, MOBIL, vehicles, pedestrians, road networks, traffic control, procedural city, protest engine, daily routines, weather, spatial grid, identity system. (`src/tritium_lib/js/` is a SECOND, textually-diverged copy of some of these modules — **NOT orphaned**: it is the runtime browser-module tree for the `sim_engine/demos/` city demos, reached via the git-tracked symlink `sim_engine/demos/js -> ../../js`. Consumed by `demos/city3d-clean.html` (imports `./js/sim/core/city-builder.js`, `world.js`, `weather.js`) and by `demos/city3d/inspect.js` → `../js/sim/identity.js` (loaded by `city3d.html`, served by `serve_city3d.py` on :8888). It correctly carries no `__init__.py`/`package-data` — it is browser JS served over HTTP, not importable Python, so wheels exclude it by design. **Do NOT delete** — deletion breaks the symlink and the demos. `web/` is the SEPARATE tree used by SC's frontend/tests. Verified 2026-07-11 — supersedes the old "zero-consumer, delete-or-wire" claim.)

### JS UI & Map Framework (web/)
Shared frontend: layout manager, command palette, event bus, reactive store, WebSocket, plus panel system and tactical map (MapLibre GL, effects, asset types, 3D units, providers)

### Operations
| Package | Purpose |
|---------|---------|
| `mission` | Mission planning and execution |
| `fleet` | Fleet management |
| `scheduler` | Task scheduling |
| `actions` | Action definitions and execution |

### Data & Analysis
| Package | Purpose |
|---------|---------|
| `data` | JSON lookup databases (BLE fingerprints, OUI, WiFi SSID patterns, etc.) |
| `analytics` | Statistical analysis and reporting |
| `reporting` | Report generation |
| `recording` | Data recording and playback (un-deprecated Gap-fix G: AAR pipeline is its consumer) |
| `data_exchange` | Import/export formats |

### Security & Compliance
| Package | Purpose |
|---------|---------|
| `privacy` | PII handling, data redaction, retention policies |
| `audit` | Audit logging, compliance trails |
| `threat_intel` | Threat intelligence feeds |

### Infrastructure & Integration
| Package | Purpose |
|---------|---------|
| `cot` | Cursor on Target XML codec for TAK/ATAK |
| `graph` | TritiumGraph / KuzuDB ontology store — **shelfware**: tests + demos only, not wired to the live ontology API. `/api/v1/ontology/*` is an in-memory adapter over TargetTracker/DossierStore/BleStore. KuzuDB integration is aspirational future work. |
| `ontology` | Semantic type system — entity types, relationship types, schema validation |
| `comms` | Communication channel abstractions |
| `federation` | Multi-site federation |
| `notifications` | Notification model and NotificationManager |
| `firmware` | FirmwareFlasher, ESP32Flasher, MeshtasticFlasher |
| `pipeline` | Data processing pipelines |
| `rules` | Rule engine for automation |
| `alerting` | Alert routing and escalation |
| `monitoring` | System health monitoring |
| `visualization` | Rendering helpers |
| `web` | Cyberpunk HTML theme engine and dashboard components |
| `testing` | Visual regression, flicker detection, ESP32 automation |

## Directory Structure

```
tritium-lib/
├── src/tritium_lib/           Python packages
│   ├── models/                Pydantic data contracts — THE source of truth
│   ├── tracking/              Target tracking & correlation
│   ├── fusion/                Multi-sensor fusion
│   ├── intelligence/          ML, RL, acoustic, anomaly detection
│   ├── sitaware/              Capstone — unified operating picture
│   ├── sim_engine/            Tactical simulation (AI, combat, physics, world)
│   ├── sdk/                   Addon development kit
│   ├── classifier/            BLE/WiFi device classification
│   ├── graph/                 TritiumGraph / KuzuDB store (shelfware — tests + demos only)
│   └── ...                    Plus: alerting, auth, cot, events, fleet, geo,
│                              mqtt, protocols, signals, store, and more
├── web/                       Shared JS/CSS frontend
│   ├── sim/                   City sim (IDM, MOBIL, pedestrian, protest)
│   ├── map/                   Tactical map (MapLibre GL, effects, 3D units)
│   ├── panels/                Draggable panel system
│   └── css/                   Cyberpunk stylesheets
├── tests/                     Test suite (mirrors package structure)
└── pyproject.toml
```

## Build & Test

```bash
# Install
pip install -e .              # Core only
pip install -e ".[full]"      # All optional deps

# Run all tests
pytest tests/

# Quick smoke test
python -c "from tritium_lib.models import *; print('OK')"

# JS sim tests
cd web && npm test

# Run a demo
python -m tritium_lib.sitaware.demos.sitaware_demo
python -m tritium_lib.tracking.demos.tracking_demo
python -m tritium_lib.sim_engine.demos.demo_city
python -m tritium_lib.intelligence.demos.pipeline_demo
```

### Demos (22 standalone scripts)
Located inside their respective packages under `demos/` subdirectories:
- `sitaware` — Unified operating picture demo
- `tracking` — Target tracking pipeline
- `sim_engine` — City sim (10 demos: city, full, RF, steering, performance, backend, integrated, game server, serve city3d, test report)
- `intelligence` — ML pipeline demo
- `graph` — TritiumGraph / KuzuDB entity relationships (2 demos; backend is shelfware — not wired to live API)
- `cot` — TAK/ATAK interop
- `mqtt` — Topic hierarchy
- `auth` — JWT workflow
- `firmware` — OTA flashing
- `sdr` — SDR device interaction
- `notifications` — Alert pipeline

## Rules

1. **Models are the API contract.** Change a model here, update all consumers. Check with `grep -r "from tritium_lib" ../tritium-edge/ ../tritium-sc/`.
2. **MQTT topics: the canonical reference is the parent repo's `docs/MQTT-PROTOCOL.md`** (verified topic tables + drift register). Reality check (2026-06-11, drift D1): `TritiumTopics` is consumed only by its own demo/tests — SC, edge, and addons all build topic strings directly, and several `TritiumTopics` builders don't match real traffic (D2-D4). Until a chartered migration makes `TritiumTopics` true, match the protocol doc, not this builder.
3. **No framework dependencies.** Stay lightweight — Pydantic, PyJWT, paho-mqtt. No FastAPI, no SQLAlchemy.
4. **Type hints everywhere.** All public functions must have complete type annotations.
5. **Test after every change.** `pytest tests/` must pass before committing.
6. **Extract, don't duplicate.** If two submodules need it, it belongs here.
7. **SitAware is the capstone.** All new subsystems should feed into the situational awareness engine.

## Coding Conventions

- Python 3.12+, PEP 8, 4-space indent
- Pydantic v2 models (use `model_config` not `class Config`)
- Vanilla JS, no frameworks
- Cyberpunk aesthetic: cyan #00f0ff, magenta #ff2a6d, green #05ffa1, yellow #fcee0a
- C++17 (when C++ code is added)

## Key Algorithms

- **IDM** (Intelligent Driver Model) — car-following dynamics
- **MOBIL** — lane change decisions
- **Bezier curves** — intersection turn paths
- **Epstein model** — protest/riot emergence
- **MFCC KNN** — acoustic classification
- **Haversine** — distance calculations
- **Kalman filter** — position estimation

## Autonomous Iteration

This submodule is part of the Tritium system. For the full autonomous build loop, wave roadmap, and agent team composition, see [../CLAUDE.md](../CLAUDE.md).

1. Read `../CLAUDE.md` for the mission and loop
2. Read `~/.claude/projects/*/memory/project_iteration_queue.md` for the wave roadmap
3. Launch 6+ agents across ALL submodules (not just this one)
4. Every 3rd wave: documentation fractal pass + redundancy cleanup
5. Never stop. Never ask permission. Just build.
