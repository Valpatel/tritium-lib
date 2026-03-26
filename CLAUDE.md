# Tritium-Lib — Shared Platform Library

The foundation library for the entire Tritium ecosystem. What started as a small models + MQTT package is now a 63-package, 200K+ line platform powering tritium-sc (Command Center), tritium-edge (firmware + fleet), and tritium-addons. Python 3.12+ backend with 54 vanilla JS modules for city simulation and UI.

**Zero shims:** SC imports directly from `tritium_lib` — no wrappers, no adapters.

**Parent context:** See [../CLAUDE.md](../CLAUDE.md) for full system architecture and conventions.

## Git Conventions

- **No co-authors on commits** — never add "Co-Authored-By" lines
- Remote: `git@github.com:Valpatel/tritium-lib.git`
- Copyright: Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC / AGPL-3.0

## By the Numbers

| Metric | Count |
|--------|-------|
| Python packages | 63 |
| Python modules | 449 |
| Pydantic models | 101 files |
| JS modules (web/) | 54 |
| Test files (Python) | 401 |
| Tests collected | 15,574 |
| JSON lookup databases | 10 |
| Standalone demos | 22 |

## Package Map (by capability area)

### Core Infrastructure
| Package | Purpose |
|---------|---------|
| `models` | 101 Pydantic model files — THE canonical data contracts for everything |
| `events` | Thread-safe and async pub/sub event bus |
| `mqtt` | Topic hierarchy `tritium/{site}/{domain}/{device}/{type}` |
| `auth` | JWT tokens, API key management |
| `config` | Pydantic base settings for service configuration |
| `store` | Persistent data stores (BLE, targets, async, time-series) — 9 modules |
| `sdk` | Addon SDK: AddonBase, AddonContext, DeviceRegistry, protocols, BaseRunner, GeoJSON layers — 15 modules |
| `utils` | Shared helpers |

### Intelligence & Fusion (38+ modules)
| Package | Purpose |
|---------|---------|
| `intelligence` | Position estimator, RL metrics, fusion metrics, acoustic classifier, anomaly detection, behavior analysis — **39 modules** |
| `fusion` | Multi-sensor target fusion (BLE + camera + WiFi -> unique UUID) |
| `inference` | ML inference pipelines, model management |
| `classifier` | Multi-signal BLE/WiFi device type classification with fingerprint databases |
| `classification` | General-purpose classification framework |

### Tracking & Situational Awareness
| Package | Purpose |
|---------|---------|
| `tracking` | Target tracking, correlation, track management — 27 modules |
| `sitaware` | **Capstone module.** Situational awareness engine — fuses all subsystems into a unified operating picture |
| `tactical` | Tactical overlays, force disposition |
| `geo` | Coordinate transforms, camera projection, haversine |
| `indoor` | Indoor positioning, WiFi fingerprinting, floorplan mapping — 4 modules |
| `areas` | Geofence zones, area monitoring |

### Sensors & Signals
| Package | Purpose |
|---------|---------|
| `signals` | Signal processing, spectrum analysis — 4 modules |
| `sdr` | SDRDevice base, SDRInfo, SweepResult |
| `comint` | Communications intelligence |
| `protocols` | Protocol decoders and handlers — 7 modules |
| `nodes` | Sensor node management |

### Simulation Engine (153 modules)
| Package | Purpose |
|---------|---------|
| `sim_engine` | Full city simulation: IDM car-following, MOBIL lane changes, Bezier intersection turns, NPC daily routines, Epstein protest/riot model, weather, traffic lights, sensor bridge — **153 Python modules** |
| `synthetic` | Synthetic data generation for training and testing |
| `scenarios` | Pre-built simulation scenarios |

### JS Simulation (web/sim/ — 15 modules)
Browser-side city simulation: `idm.js`, `mobil.js`, `vehicle.js`, `pedestrian.js`, `road-network.js`, `traffic-controller.js`, `procedural-city.js`, `protest-engine.js`, `protest-scenario.js`, `daily-routine.js`, `schedule-executor.js`, `weather.js`, `spatial-grid.js`, `identity.js`, `index.js`

### JS UI & Map Framework (web/ — 39 modules)
Shared UI: `layout-manager.js`, `command-palette.js`, `events.js`, `store.js`, `utils.js`, `websocket.js`, plus `panels/` (panel-manager, tabbed-container), `map/` (31 modules: layer-manager, data-provider, draw-tools, overlays, coords, battle-hud, unit-markers, asset-types, effects, three-units, providers), and `css/`

### Operations & C2
| Package | Purpose |
|---------|---------|
| `c2` | Command and control abstractions |
| `mission` | Mission planning and execution |
| `fleet` | Fleet management |
| `deployment` | Deployment orchestration — 6 modules |
| `scheduler` | Task scheduling |
| `actions` | Action definitions and execution |

### Data & Analysis
| Package | Purpose |
|---------|---------|
| `data` | 10 JSON lookup databases (BLE fingerprints, OUI, WiFi SSID patterns, etc.) |
| `analytics` | Statistical analysis and reporting |
| `reporting` | Report generation |
| `recording` | Data recording and playback — 3 modules |
| `evidence` | Evidence collection and chain-of-custody — 6 modules |
| `map_data` | Map tile and geodata management |
| `geoint` | Geospatial intelligence |
| `data_exchange` | Import/export formats |

### Security & Compliance
| Package | Purpose |
|---------|---------|
| `privacy` | PII handling, data redaction, retention policies — 5 modules |
| `audit` | Audit logging, compliance trails — 4 modules |
| `threat_intel` | Threat intelligence feeds |

### Infrastructure & Integration
| Package | Purpose |
|---------|---------|
| `cot` | Cursor on Target XML codec for TAK/ATAK |
| `graph` | KuzuDB embedded graph database for entity/relationship storage |
| `ontology` | Semantic type system — entity types, relationship types, schema validation |
| `comms` | Communication channel abstractions |
| `federation` | Multi-site federation |
| `network` | Network topology and discovery |
| `notifications` | Notification model and NotificationManager |
| `firmware` | FirmwareFlasher, ESP32Flasher, MeshtasticFlasher |
| `interfaces` | Abstract interfaces for plugin integration — 4 modules |
| `pipeline` | Data processing pipelines |
| `rules` | Rule engine for automation |
| `alerting` | Alert routing and escalation |
| `monitoring` | System health monitoring |
| `quality` | Data quality checks |
| `visualization` | Rendering helpers — 5 modules |
| `web` | Cyberpunk HTML theme engine and dashboard components |
| `testing` | Visual regression, flicker detection, ESP32 automation — 7 modules |

## Directory Structure

```
tritium-lib/
├── src/tritium_lib/          # 63 Python packages, 449 modules
│   ├── models/               # 101 Pydantic model files (THE data contracts)
│   ├── sim_engine/           # 153 modules — full city simulation (largest package)
│   ├── intelligence/         # 39 modules — ML, RL, acoustic, anomaly
│   ├── tracking/             # 27 modules — target tracking & correlation
│   ├── sdk/                  # 15 modules — addon development kit
│   ├── store/                # 9 modules — persistence layer
│   ├── protocols/            # 7 modules — protocol decoders
│   ├── ... (55 more packages)
│   └── sitaware/             # Capstone — unified operating picture
├── web/                      # 54 JS modules
│   ├── sim/                  # 15 modules — city sim (IDM, MOBIL, pedestrian, protest)
│   ├── map/                  # 31 modules — map rendering, effects, asset types, 3D units
│   ├── panels/               # Panel manager, tabbed containers
│   ├── css/                  # Cyberpunk stylesheets
│   └── *.js                  # Layout, events, store, utils, websocket
├── tests/                    # 401 Python test files, 15,574 tests total
│   ├── sim_engine/           # Sim engine tests
│   ├── intelligence/         # Intelligence tests
│   ├── models/               # Model tests
│   └── test_*.py
└── pyproject.toml
```

## Build & Test

```bash
# Install
pip install -e .              # Core only
pip install -e ".[full]"      # All optional deps

# Run all tests (15,574 tests, ~2s collection)
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
- `graph` — KuzuDB entity relationships (2 demos)
- `cot` — TAK/ATAK interop
- `mqtt` — Topic hierarchy
- `auth` — JWT workflow
- `firmware` — OTA flashing
- `sdr` — SDR device interaction
- `notifications` — Alert pipeline

## Rules

1. **Models are the API contract.** Change a model here, update all consumers. Check with `grep -r "from tritium_lib" ../tritium-edge/ ../tritium-sc/`.
2. **MQTT topics are defined here, not in consumers.** Use `TritiumTopics` — never hardcode topic strings.
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
