# tritium-lib

Shared Python + JavaScript library for the [Tritium](https://github.com/Valpatel/tritium) unified operating picture system. Provides models, tracking, inference, simulation, events, MQTT topics, auth, and reusable frontend components used by tritium-sc (Command Center), tritium-edge (ESP32 firmware/fleet server), and tritium-addons.

**358 Python modules | 270 test files | 54 JS modules | 101 Pydantic models | 152 sim engine files**

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
├── src/tritium_lib/           # Python packages (358 modules)
│   ├── models/                # 101 Pydantic v2 models — THE canonical data contracts
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
│   ├── sdk/                   # Addon SDK — AddonBase, DeviceRegistry, BaseRunner, GeoJSON
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
│   └── sim_engine/            # Tactical simulation engine (152 files)
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
│       └── demos/             # 10 demo apps + HTML frontends
├── web/                       # Shared JS/CSS frontend library (54 modules)
│   ├── css/                   # Cyberpunk themes (cybercore v1 + v2)
│   ├── map/                   # Tactical map components (MapLibre GL)
│   ├── sim/                   # City simulation (IDM, MOBIL, pedestrians, protest)
│   ├── panels/                # Draggable/resizable panel system
│   ├── events.js              # Frontend EventBus (pub/sub)
│   ├── store.js               # ReactiveStore (dot-path state, RAF-batched)
│   ├── websocket.js           # TritiumWebSocket (reconnect, ping, banner)
│   ├── command-palette.js     # Fuzzy search command palette (Ctrl+K)
│   ├── layout-manager.js      # Panel layout save/restore/import/export
│   └── utils.js               # _esc, _timeAgo, _badge, _fetchJson
└── tests/                     # 270 test files (pytest)
```

## Module Reference

### Core

| Module | Import | Description |
|--------|--------|-------------|
| `models` | `from tritium_lib.models import Device, BleSighting` | 101 Pydantic v2 models for devices, BLE, mesh, cameras, alerts, CoT, sensors, floorplans, dossiers, and more |
| `events` | `from tritium_lib.events import EventBus, AsyncEventBus` | Thread-safe and async pub/sub with wildcard topic matching |
| `mqtt` | `from tritium_lib.mqtt import TritiumTopics` | MQTT topic hierarchy builder (`tritium/{site}/{domain}/{device}/{type}`) |
| `auth` | `from tritium_lib.auth import create_token, decode_token` | JWT creation/decoding and API key management |
| `config` | `from tritium_lib.config import TritiumSettings` | Pydantic base settings for service configuration |
| `store` | `from tritium_lib.store import BleStore, TargetStore` | Persistent data stores for BLE sightings, targets, events, dossiers |
| `cot` | `from tritium_lib.cot import device_to_cot, parse_cot` | Cursor on Target XML codec for TAK/ATAK |

### Tracking & Intelligence

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

### Simulation Engine (152 Python files + 13 JS modules)

| Subpackage | Description |
|------------|-------------|
| `sim_engine.ai` | Combat AI, behavior trees, pathfinding, steering, squad tactics, strategy, formations |
| `sim_engine.behavior` | NPC behaviors, unit states, unit missions, degradation |
| `sim_engine.combat` | Combat resolution, squad management, weapon systems |
| `sim_engine.core` | Entity model, movement, inventory, spatial hash, state machine, NPC thinker |
| `sim_engine.effects` | Particle systems, weapon visual effects |
| `sim_engine.game` | Game modes, difficulty scaling, morale, crowd density, stats, ambient |
| `sim_engine.physics` | Vehicle physics, collision detection |
| `sim_engine.world` | Cover system, grid pathfinder, sensor simulation, vision/LOS |
| `sim_engine.unit_types` | Unit base + people, robots, sensors subtypes |
| `sim_engine.audio` | Spatial audio positioning |
| `sim_engine.debug` | Debug stream output |
| `sim_engine.demos` | 10 demo apps (see below) |

Top-level sim_engine modules include: air combat, artillery, buildings, campaign, civilian, crowd, cyber warfare, damage, destruction, detection, economy, electronic warfare, environment, and more.

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

### Visual regression testing

```python
from tritium_lib.testing import VisualCheck
import numpy as np

check = VisualCheck(width=1920, height=1080)
# Load a screenshot as numpy array (H x W x 3 BGR)
img = np.zeros((1080, 1920, 3), dtype=np.uint8)
issues = check.check_blank_screen(img)  # Returns list of issues
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

## Sim Engine Demos

Run demos from the repo root:

```bash
./sim-demo.sh                          # Tactical sim (default: urban_combat)
./sim-demo.sh --list                   # List presets
./sim-demo.sh --perf                   # Performance benchmark
./sim-demo.sh --coverage               # Module coverage report
```

Available demo apps:

| Demo | Command | Description |
|------|---------|-------------|
| Tactical Sim | `./sim-demo.sh` | Three.js 3D combat sim at `http://localhost:9090` |
| City Sim | `python -m tritium_lib.sim_engine.demos.demo_city` | City simulation with traffic and pedestrians |
| Full Demo | `python -m tritium_lib.sim_engine.demos.demo_full` | All systems combined |
| RF Demo | `python -m tritium_lib.sim_engine.demos.demo_rf` | RF signature simulation |
| Steering Demo | `python -m tritium_lib.sim_engine.demos.demo_steering` | AI steering behaviors |
| Perf Test | `python -m tritium_lib.sim_engine.demos.perf_test` | Performance benchmarks |
| City3D Server | `python -m tritium_lib.sim_engine.demos.serve_city3d` | City3D standalone server |
| Game Server | `python -m tritium_lib.sim_engine.demos.game_server` | Game mode server |
| Tracking Demo | `python -m tritium_lib.tracking.demos.tracking_demo` | Target tracking demo |

Sim controls: `SPACE`=riot `N`=night `R`=rain `F`=fog `D`=debug `I`=intel `M`=minimap `K`=record `L`=playback `A`=airstrike `S`=sound `2`=split-view `P`=skip-phase `C`=chase-cam `Click`=inspect `ESC`=deselect

Presets: `urban_combat`, `open_field`, `riot_response`, `convoy_ambush`, `drone_strike`

## JavaScript / Frontend Library

The `web/` directory contains 54 vanilla ES modules (no build step) used by tritium-sc and addons. See [web/README.md](web/README.md) for the full module list, import examples, and extension patterns.

Quick summary:

| Package | Modules | Purpose |
|---------|---------|---------|
| `web/map/` | 28 | Tactical map: coords, layers, draw tools, battle HUD, asset types, effects, 3D units, providers |
| `web/sim/` | 14 | City simulation: IDM car-following, MOBIL lane changes, pedestrians, protest, traffic, weather |
| `web/panels/` | 2 | Draggable/resizable panel system with tabs |
| `web/css/` | 2 | Cyberpunk theme stylesheets (cybercore v1 + v2) |
| root | 5 | EventBus, ReactiveStore, TritiumWebSocket, CommandPalette, LayoutManager, utils |

SC serves these at `/lib/` via a StaticFiles mount or symlink.

## Quick Start for Addon Developers

### 1. Install tritium-lib

```bash
cd tritium-lib
pip install -e ".[full]"
```

### 2. Use models for your data contracts

```python
from tritium_lib.models.device import DeviceInfo, DeviceStatus
from tritium_lib.models.sensor import SensorReading
from tritium_lib.models.alert import Alert, AlertLevel
```

### 3. Use MQTT topics (never hardcode strings)

```python
from tritium_lib.mqtt import TritiumTopics
topics = TritiumTopics(site_id="hq")
my_topic = topics.edge_heartbeat("my-device-001")
```

### 4. Use the event bus for internal pub/sub

```python
from tritium_lib.events import EventBus
bus = EventBus()
bus.subscribe("my-addon.#", handler)
bus.publish("my-addon.detection", {"target_id": "ble_aa:bb:cc"})
```

### 5. Extend the frontend (JS)

Register custom asset types, map data providers, or 3D unit models. Import from `/lib/`:

```javascript
import { BaseAssetType } from '/lib/map/asset-types/base.js';
import { assetTypeRegistry } from '/lib/map/asset-types/registry.js';
import { MapDataProvider, providerRegistry } from '/lib/map/data-provider.js';
```

See [web/README.md](web/README.md) for full examples.

### 6. Build a standalone runner

```python
from tritium_lib.sdk import BaseRunner

class MyRunner(BaseRunner):
    """Headless mode for Raspberry Pi deployment."""
    async def run(self):
        while True:
            reading = await self.collect()
            await self.publish(reading)
```

## Tests

```bash
pytest tests/                          # Run all 270 test files
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
