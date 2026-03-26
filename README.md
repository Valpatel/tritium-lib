# tritium-lib

Shared Python + JavaScript library for the [Tritium](https://github.com/Valpatel/tritium) unified operating picture system. Provides models, tracking, inference, simulation, events, MQTT topics, auth, and reusable frontend components used by tritium-sc (Command Center), tritium-edge (ESP32 firmware/fleet server), and tritium-addons.

**449 Python modules | 92 packages | 54 JS modules | 425 test files (15,570+ tests) | 278 Pydantic models | 170 sim engine files | 13 standalone demos**

Copyright 2026 Matthew Valancy / Valpatel Software LLC / AGPL-3.0

---

## Install

```bash
pip install -e .                    # Core (models, events, MQTT, auth, tracking)
pip install -e ".[mqtt]"            # + paho-mqtt
pip install -e ".[graph]"           # + KuzuDB graph database
pip install -e ".[testing]"         # + OpenCV, numpy, requests
pip install -e ".[geospatial]"      # + rasterio, shapely, geopandas
pip install -e ".[full]"            # All optional deps
```

## Architecture

```
tritium-lib/
├── src/tritium_lib/           # Python packages (449 modules, 92 packages)
│   ├── models/                # 278 Pydantic v2 models -- THE canonical data contracts
│   ├── tracking/              # Target tracker, correlator, geofence, Kalman, dossiers
│   ├── inference/             # LLM client, fleet inference, model router
│   ├── intelligence/          # Acoustic classifier, anomaly, RL metrics, fusion, position
│   ├── events/                # Thread-safe + async pub/sub event bus
│   ├── comms/                 # Communication abstractions (speaker/TTS)
│   ├── mqtt/                  # MQTT topic hierarchy builder
│   ├── auth/                  # JWT + API key management
│   ├── store/                 # Persistent data stores (BLE sightings, targets)
│   ├── config/                # Pydantic base settings
│   ├── cot/                   # Cursor on Target XML codec (TAK/ATAK)
│   ├── geo/                   # Coordinate transforms, haversine, camera projection
│   ├── graph/                 # KuzuDB graph database wrapper
│   ├── ontology/              # Entity/relationship type system + registry
│   ├── classifier/            # Multi-signal BLE/WiFi device classifier
│   ├── data/                  # 11 JSON lookup databases (BLE, WiFi, OUI fingerprints)
│   ├── sdk/                   # Addon SDK -- AddonBase, DeviceRegistry, BaseRunner, GeoJSON
│   ├── interfaces/            # Plugin interfaces (camera, radar, SDR, sensor)
│   ├── nodes/                 # Node base classes
│   ├── firmware/              # Firmware flasher base (ESP32, Meshtastic)
│   ├── sdr/                   # SDR device abstractions
│   ├── notifications/         # Notification model + manager
│   ├── tactical/              # Tactical dossier generation
│   ├── synthetic/             # Synthetic test data generators
│   ├── actions/               # Formation control, Lua parser
│   ├── utils/                 # Feature extraction, memory helpers
│   ├── web/                   # Cyberpunk HTML theme engine + dashboard components
│   ├── testing/               # Visual regression checks, ESP32 device automation
│   ├── fusion/                # Multi-sensor fusion engine + sensor pipeline
│   ├── alerting/              # Alert rules engine and dispatch
│   ├── reporting/             # Situation report generation
│   ├── monitoring/            # System health monitoring and metrics
│   ├── recording/             # Record and replay sensor data streams
│   ├── pipeline/              # Configurable data pipeline orchestrator
│   ├── rules/                 # IF-THEN automation rules engine
│   ├── evidence/              # Evidence collection and chain-of-custody
│   ├── incident/              # Incident management lifecycle
│   ├── mission/               # Surveillance/security mission planning
│   ├── signals/               # RF signal analysis (RSSI, CSI, spectrum)
│   ├── protocols/             # Radio protocol parsers (ADS-B, AIS, NMEA, BLE, WiFi)
│   ├── classification/        # Multi-sensor target classification pipeline
│   ├── scenarios/             # Predefined scenario generator for training/demos
│   ├── indoor/                # Indoor positioning via WiFi/BLE fingerprinting
│   ├── privacy/               # Data retention, anonymization, compliance
│   ├── areas/                 # Named geographic area management
│   ├── comint/                # Communications intelligence (metadata analysis)
│   ├── threat_intel/          # Threat intelligence feeds (STIX parsing)
│   ├── c2/                    # Command and Control protocol for edge devices
│   ├── geoint/                # Geospatial intelligence (cover, LOS, routes)
│   ├── sitaware/              # Situational awareness engine (capstone module)
│   ├── analytics/             # Real-time statistics and trend analysis
│   ├── quality/               # Data quality monitoring for sensor feeds
│   ├── fleet/                 # Fleet device management and heartbeat
│   ├── visualization/         # Chart, timeline, heatmap, network graph data
│   ├── deployment/            # Deployment, backup, and health utilities
│   ├── network/               # Network topology discovery and analysis
│   ├── federation/            # Multi-site federation and target sync
│   ├── scheduler/             # Task scheduling and queue
│   ├── map_data/              # Tactical map data and GeoJSON export
│   ├── audit/                 # Persistent audit trail for compliance
│   ├── data_exchange/         # Import/export targets, dossiers, events
│   └── sim_engine/            # Tactical simulation engine (170 files)
│       ├── ai/                # Combat AI, behavior trees, pathfinding, steering
│       ├── behavior/          # NPC behaviors, unit states, missions
│       ├── combat/            # Combat resolution, squads, weapons
│       ├── core/              # Entity, movement, inventory, spatial, state machine
│       ├── effects/           # Particles, weapon effects
│       ├── game/              # Game modes, difficulty, morale, stats
│       ├── physics/           # Vehicle physics, collision
│       ├── world/             # Cover, pathfinding, sensors, vision
│       ├── unit_types/        # Unit base + people, robots, sensors
│       ├── audio/             # Spatial audio
│       ├── debug/             # Debug streams
│       └── demos/             # Demo apps + HTML frontends
├── web/                       # Shared JS/CSS frontend library (54 modules)
│   ├── css/                   # Cyberpunk themes (cybercore v1 + v2)
│   ├── map/                   # Tactical map components (MapLibre GL, 31 modules)
│   ├── sim/                   # City simulation (IDM, MOBIL, pedestrians, protest, 15 modules)
│   ├── panels/                # Draggable/resizable panel system
│   ├── events.js              # Frontend EventBus (pub/sub)
│   ├── store.js               # ReactiveStore (dot-path state, RAF-batched)
│   ├── websocket.js           # TritiumWebSocket (reconnect, ping, banner)
│   ├── command-palette.js     # Fuzzy search command palette (Ctrl+K)
│   ├── layout-manager.js      # Panel layout save/restore/import/export
│   └── utils.js               # _esc, _timeAgo, _badge, _fetchJson
└── tests/                     # 425 test files (15,570+ tests)
```

## Standalone Demos

Thirteen self-contained demos, each with its own HTTP server and cyberpunk UI.

| # | Demo | Command | Port | Description |
|---|------|---------|------|-------------|
| 1 | Tracking | `python -m tritium_lib.tracking.demos.tracking_demo` | 9091 | Target tracking pipeline -- BLE/WiFi/camera fusion, correlation, geofencing |
| 2 | Intelligence | `python -m tritium_lib.intelligence.demos.pipeline_demo` | 8090 | Sensor fusion, anomaly detection, acoustic classification, threat assessment |
| 3 | MQTT | `python -m tritium_lib.mqtt.demos.mqtt_demo` | 9092 | Sensor-to-fusion pipeline with mock or live MQTT broker |
| 4 | CoT/TAK | `python -m tritium_lib.cot.demos.cot_demo` | 9094 | MIL-STD-2045 Cursor on Target XML codec -- TAK/ATAK interoperability |
| 5 | Firmware | `python -m tritium_lib.firmware.demos.firmware_demo` | 8098 | Device discovery, OTA flash progress, fleet firmware management |
| 6 | Graph | `python -m tritium_lib.graph.demos.graph_demo` | 8099 | Entity-relationship storage, querying, SVG visualization |
| 7 | Notifications | `python -m tritium_lib.notifications.demos.notification_demo` | 9092 | Geofence alerts, threat scoring, sensor health, notification routing |
| 8 | SDR | `python -m tritium_lib.sdr.demos.sdr_demo` | 9092 | SDR spectrum analyzer simulation and signal analysis |
| 9 | City Sim | `python -m tritium_lib.sim_engine.demos.demo_city` | -- | City simulation with traffic (IDM/MOBIL) and pedestrians |
| 10 | Tactical Sim | `python -m tritium_lib.sim_engine.demos.demo_full` | 9090 | Full 3D combat sim -- squads, weapons, AI, morale, effects |
| 11 | Auth | `python -m tritium_lib.auth.demos.auth_demo` | 9097 | JWT login, refresh tokens, API keys, RBAC, cyberpunk login page |
| 12 | Sitaware | `python -m tritium_lib.sitaware.demos.sitaware_demo` | 9095 | Full operating picture -- tracking, fusion, intelligence, alerting, reporting |
| 13 | Integrated | `python -m tritium_lib.sim_engine.demos.integrated_demo` | 8099 | City sim to sensor fusion end-to-end pipeline with correlation and geofencing |

Additional sim demos: `./sim-demo.sh` (tactical), `demo_rf` (RF), `demo_steering` (AI), `perf_test`, `serve_city3d`, `game_server`. Presets: `urban_combat`, `open_field`, `riot_response`, `convoy_ambush`, `drone_strike`.

## Module Reference

### Core

| Module | Import | Description |
|--------|--------|-------------|
| `models` | `from tritium_lib.models import Device, BleSighting` | 278 Pydantic v2 models for devices, BLE, mesh, cameras, alerts, CoT, sensors, floorplans, dossiers, and more |
| `events` | `from tritium_lib.events import EventBus, AsyncEventBus` | Thread-safe and async pub/sub with wildcard topic matching |
| `mqtt` | `from tritium_lib.mqtt import TritiumTopics` | MQTT topic hierarchy builder (`tritium/{site}/{domain}/{device}/{type}`) |
| `auth` | `from tritium_lib.auth import create_token, decode_token` | JWT creation/decoding and API key management |
| `config` | `from tritium_lib.config import TritiumSettings` | Pydantic base settings for service configuration |
| `store` | `from tritium_lib.store import BleStore, TargetStore` | Persistent data stores for BLE sightings, targets, events, dossiers |
| `cot` | `from tritium_lib.cot import device_to_cot, parse_cot` | Cursor on Target XML codec for TAK/ATAK |

### Tracking and Intelligence

| Module | Import | Description |
|--------|--------|-------------|
| `tracking` | `from tritium_lib.tracking import TargetTracker, TargetCorrelator` | Target tracking, correlation strategies, geofencing, Kalman prediction, convoy detection, threat scoring, trilateration, dossiers |
| `inference` | `from tritium_lib.inference import LLMFleet` | Local LLM fleet inference (llama-server), model routing (`inference.model_router.ModelRouter`) |
| `intelligence` | `from tritium_lib.intelligence import AcousticClassifier, AnomalyDetector` | Acoustic classification, anomaly detection, RL metrics, fusion metrics, position estimation, pattern learning |
| `classifier` | `from tritium_lib.classifier import DeviceClassifier` | Multi-signal BLE/WiFi device type classifier with fingerprint databases |
| `geo` | `from tritium_lib.geo import local_to_latlng, haversine_distance` | Coordinate transforms (local meters to lat/lng), camera projection, haversine distance |
| `graph` | `from tritium_lib.graph import TritiumGraph` | KuzuDB embedded graph for entity/relationship storage |
| `ontology` | `from tritium_lib.ontology import OntologyRegistry` | Semantic type system for entities and relationships |

### Addon Development

| Module | Import | Description |
|--------|--------|-------------|
| `sdk` | `from tritium_lib.sdk import AddonBase, AddonContext, BaseRunner` | Addon SDK: base classes, device registry, transport, GeoJSON layers, config loader, subprocess manager |
| `interfaces` | `from tritium_lib.interfaces import CameraPlugin, RadarPlugin` | Plugin interface contracts for cameras, radar, SDR, sensors |
| `firmware` | `from tritium_lib.firmware import FirmwareFlasher` | Firmware flasher base for ESP32 and Meshtastic |
| `sdr` | `from tritium_lib.sdr import SDRDevice, SweepResult` | SDR device abstractions |
| `notifications` | `from tritium_lib.notifications import NotificationManager` | Thread-safe notification model and manager |

### Utilities

| Module | Import | Description |
|--------|--------|-------------|
| `data` | `from tritium_lib.data import load_ble_fingerprints` | 11 JSON lookup databases for BLE, WiFi, OUI fingerprinting |
| `web` | `from tritium_lib.web import CyberpunkTheme` | Cyberpunk HTML theme engine and dashboard components |
| `testing` | `from tritium_lib.testing import VisualCheck` | Visual regression checks and ESP32 device automation |
| `synthetic` | `from tritium_lib.synthetic import DataGenerators` | Synthetic test data generators |
| `tactical` | `from tritium_lib.tactical import TacticalDossier` | Tactical dossier generation |
| `actions` | `from tritium_lib.actions import FormationControl` | Formation control and Lua script parser |
| `utils` | `from tritium_lib.utils import FeatureExtractor` | Feature extraction and memory helpers |
| `nodes` | `from tritium_lib.nodes import BaseNode` | Node base classes |
| `comms` | `from tritium_lib.comms import Speaker` | Communication abstractions (TTS) |

### Simulation Engine (170 Python files + 15 JS modules)

| Subpackage | Description |
|------------|-------------|
| `sim_engine.ai` | Combat AI, behavior trees, pathfinding, steering, squad tactics, strategy, formations |
| `sim_engine.behavior` | NPC behaviors, unit states, unit missions, degradation |
| `sim_engine.combat` | Combat resolution, squad management, weapon systems |
| `sim_engine.core` | Entity model, movement, inventory, spatial hash, state machine, NPC thinker |
| `sim_engine.effects` | Particle systems, weapon visual effects |
| `sim_engine.game` | Game modes, difficulty, morale, crowd density, stats |
| `sim_engine.physics` | Vehicle physics, collision detection |
| `sim_engine.world` | Cover, pathfinding, sensors, vision/LOS |
| `sim_engine.unit_types` | Unit base + people, robots, sensors |
| `sim_engine.demos` | Demo apps (see Standalone Demos above) |

## Quick Start Examples

### MQTT topics

```python
from tritium_lib.mqtt import TritiumTopics

topics = TritiumTopics(site_id="home")
topics.edge_heartbeat("esp32-001")
# -> "tritium/home/edge/esp32-001/heartbeat"
topics.camera_detections("cam-front")
# -> "tritium/home/cameras/cam-front/detections"
```

### Event bus

```python
from tritium_lib.events import EventBus

bus = EventBus()
bus.subscribe("device.#", lambda e: print(e.topic, e.data))
bus.publish("device.heartbeat", {"id": "esp32-001"})
```

### Target tracking

```python
from tritium_lib.tracking import TargetTracker

tracker = TargetTracker()
tracker.update_from_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -65,
                          "name": "Phone", "observer_id": "node-1"})
targets = tracker.get_all()
# -> [TrackedTarget(target_id='ble_aabbccddeeff', ...)]
```

### Addon skeleton

```python
from tritium_lib.sdk import AddonBase, AddonContext

class MyAddon(AddonBase):
    name = "my-sensor"
    version = "0.1.0"

    async def start(self, ctx: AddonContext):
        ctx.mqtt.subscribe("tritium/+/my-sensor/#", self.on_message)

    async def on_message(self, topic, payload):
        ctx.events.publish("my-sensor.reading", payload)

    async def stop(self):
        pass
```

### Standalone runner

```python
from tritium_lib.sdk import BaseRunner

class MyRunner(BaseRunner):
    """Headless mode for Raspberry Pi deployment."""
    async def run(self):
        while True:
            reading = await self.collect()
            await self.publish(reading)
```

## JavaScript / Frontend Library

The `web/` directory contains 54 vanilla ES modules (no build step) used by tritium-sc and addons. See [web/README.md](web/README.md) for the full module list, import examples, and extension patterns.

| Package | Modules | Purpose |
|---------|---------|---------|
| `web/map/` | 31 | Tactical map: coords, layers, draw tools, battle HUD, asset types, effects, 3D units, providers |
| `web/sim/` | 15 | City simulation: IDM car-following, MOBIL lane changes, pedestrians, protest, traffic, weather |
| `web/panels/` | 2 | Draggable/resizable panel system with tabs |
| `web/css/` | 2 | Cyberpunk theme stylesheets (cybercore v1 + v2) |
| root | 6 | EventBus, ReactiveStore, TritiumWebSocket, CommandPalette, LayoutManager, utils |

SC serves these at `/lib/` via a StaticFiles mount or symlink.

## Tests

```bash
pytest tests/                          # Run all 425 test files (15,570+ tests)
pytest tests/ -x --tb=short           # Stop on first failure
pytest tests/test_models.py           # Single file
pytest tests/ -k "tracking"           # Pattern match
```

## Used By

- **[tritium-sc](https://github.com/Valpatel/tritium-sc)** -- Command Center (FastAPI + vanilla JS, 25 plugins, AI commander Amy)
- **[tritium-edge](https://github.com/Valpatel/tritium-edge)** -- ESP32-S3 firmware (Tritium-OS) + fleet server
- **[tritium-addons](https://github.com/Valpatel/tritium-addons)** -- HackRF SDR, Meshtastic LoRa, and future addons

## License
AGPL-3.0 -- Copyright 2026 Matthew Valancy / Valpatel Software LLC
