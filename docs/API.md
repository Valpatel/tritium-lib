# tritium_lib API Reference

Module-by-module reference for the Tritium shared library.

## tracking

Target tracking, identity resolution, and spatial analysis.

```python
from tritium_lib.tracking import *
```

| Class / Function | Module | Description |
|-----------------|--------|-------------|
| `TrackedTarget` | `target_tracker` | Dataclass for a tracked entity with unique ID, position, confidence, source |
| `TargetTracker` | `target_tracker` | Registry of all tracked targets, handles upsert and confidence decay |
| `TargetCorrelator` | `correlator` | Fuses multi-source sightings into unified target identities |
| `CorrelationRecord` | `correlator` | Record of a correlation decision between two targets |
| `CorrelationStrategy` | `correlation_strategies` | ABC for pluggable correlation scoring strategies |
| `SpatialStrategy` | `correlation_strategies` | Correlate by physical proximity |
| `TemporalStrategy` | `correlation_strategies` | Correlate by time co-occurrence |
| `SignalPatternStrategy` | `correlation_strategies` | Correlate by signal fingerprint similarity |
| `WiFiProbeStrategy` | `correlation_strategies` | Correlate by WiFi probe request patterns |
| `DossierStrategy` | `correlation_strategies` | Correlate using historical dossier data |
| `BLEClassifier` | `ble_classifier` | Classify BLE devices by type (phone, watch, beacon, etc.) |
| `BLEClassification` | `ble_classifier` | Classification result with device type and confidence |
| `GeofenceEngine` | `geofence` | Zone entry/exit detection for tracked targets |
| `GeoZone` | `geofence` | Polygon zone definition |
| `GeoEvent` | `geofence` | Entry/exit event |
| `TrilaterationEngine` | `trilateration` | Multi-anchor position estimation from RSSI |
| `PositionResult` | `trilateration` | Estimated position with uncertainty |
| `KalmanState` | `kalman_predictor` | Per-target Kalman filter state |
| `kalman_update` | `kalman_predictor` | Update Kalman state with new measurement |
| `predict_target_kalman` | `kalman_predictor` | Predict future position using Kalman state |
| `HeatmapEngine` | `heatmap` | Accumulate target sightings into spatial heatmaps |
| `MovementPatternAnalyzer` | `movement_patterns` | Detect loitering, pacing, circling from track history |
| `MovementPattern` | `movement_patterns` | Detected movement pattern type and confidence |
| `DossierStore` | `dossier` | Per-target dossier persistence (signal history, tags, notes) |
| `TargetDossier` | `dossier` | Complete dossier for a single target |
| `TargetHistory` | `target_history` | Time-series position history per target |
| `TargetReappearanceMonitor` | `target_reappearance` | Detect targets returning after absence |
| `predict_target` | `target_prediction` | Simple linear position prediction |
| `VehicleTrackingManager` | `vehicle_tracker` | Track vehicle-class targets with speed/heading |
| `ConvoyDetector` | `convoy_detector` | Detect groups of targets moving together |
| `DwellTracker` | `dwell_tracker` | Detect prolonged stationary presence |

## events

Thread-safe and async pub/sub event system.

```python
from tritium_lib.events import QueueEventBus, AsyncEventBus
```

| Class | Module | Description |
|-------|--------|-------------|
| `QueueEventBus` | `bus` | Thread-safe synchronous pub/sub using queue.Queue |
| `AsyncEventBus` | `bus` | Asyncio pub/sub for async consumers |
| `Event` | `bus` | Base event dataclass (topic, data, timestamp) |
| `EventBus` | `bus` | Alias for QueueEventBus (backward compat) |

## inference

LLM fleet management and model routing.

```python
from tritium_lib.inference import LLMFleet, ollama_chat, llama_server_chat
```

| Class / Function | Module | Description |
|-----------------|--------|-------------|
| `LLMFleet` | `fleet` | Multi-host LLM discovery and health checking |
| `FleetHost` | `fleet` | Single LLM host with URL and status |
| `OllamaFleet` | `fleet` | Ollama-specific fleet implementation |
| `ModelRouter` | `model_router` | Task-aware model selection with fallback |
| `TaskType` | `model_router` | Enum of task types (chat, vision, classify, etc.) |
| `ModelProfile` | `model_router` | Model metadata (name, capabilities, speed) |
| `ollama_chat` | `llm_client` | Direct Ollama API chat call |
| `llama_server_chat` | `llm_client` | Direct llama-server API chat call |

## comms

Communication modules.

```python
from tritium_lib.comms import Speaker
```

| Class | Module | Description |
|-------|--------|-------------|
| `Speaker` | `speaker` | TTS output via Piper (async-safe, queue-based) |

## actions

Lua action parsing for Amy and robots.

```python
from tritium_lib.actions import parse_motor_output, MotorOutput
```

| Class / Function | Module | Description |
|-----------------|--------|-------------|
| `MotorOutput` | `lua_parser` | Parsed Lua action (function name, args, raw text) |
| `parse_motor_output` | `lua_parser` | Parse LLM output into MotorOutput |
| `extract_lua_from_response` | `lua_parser` | Extract Lua code block from LLM text |
| `parse_function_call` | `lua_parser` | Parse a single Lua function call |
| `split_arguments` | `lua_parser` | Split Lua argument list respecting nesting |
| `formation` | `formation` | Squad formation calculations (line, wedge, column) |

## geo

Coordinate transforms between lat/lng and local meters.

```python
from tritium_lib.geo import init_reference, local_to_latlng, haversine_distance
```

| Class / Function | Module | Description |
|-----------------|--------|-------------|
| `GeoReference` | `__init__` | Dataclass anchoring local coords to real lat/lng |
| `CameraCalibration` | `__init__` | Camera projection parameters |
| `init_reference(lat, lng, alt)` | `__init__` | Set the map center reference point |
| `get_reference()` | `__init__` | Get current reference point |
| `is_initialized()` | `__init__` | Check if reference is set |
| `reset()` | `__init__` | Clear reference point |
| `local_to_latlng(x, y, z)` | `__init__` | Convert local meters to lat/lng dict |
| `latlng_to_local(lat, lng, alt)` | `__init__` | Convert lat/lng to local meters tuple |
| `local_to_latlng_2d(x, y)` | `__init__` | 2D variant returning (lat, lng) tuple |
| `camera_pixel_to_ground(cx, cy, calib)` | `__init__` | Project camera pixel to ground plane |
| `haversine_distance(lat1, lng1, lat2, lng2)` | `__init__` | Great-circle distance in meters |

## intelligence

ML scorers, anomaly detection, feature engineering, threat modeling.

```python
from tritium_lib.intelligence import BaseLearner, StaticScorer, AnomalyDetector
```

| Class / Function | Module | Description |
|-----------------|--------|-------------|
| `BaseLearner` | `base_learner` | ABC for ML learners (train, predict, save, load) |
| `ModelRegistry` | `model_registry` | SQLite-backed versioned ML model storage |
| `StaticScorer` | `scorer` | Hand-tuned weighted correlation scorer (baseline) |
| `LearnedScorer` | `scorer` | Trained logistic regression correlation scorer |
| `ScorerResult` | `scorer` | Score output (probability, confidence) |
| `AnomalyDetector` | `anomaly` | ABC for anomaly detectors |
| `SimpleThresholdDetector` | `anomaly` | Threshold-based anomaly detection |
| `AutoencoderDetector` | `anomaly` | Autoencoder-based anomaly detection (numpy) |
| `Anomaly` | `anomaly` | Anomaly dataclass |
| `build_extended_features` | `feature_engineering` | Build feature vector from two targets |
| `co_movement_score` | `feature_engineering` | Score co-movement between targets |
| `PatternLearner` | `pattern_learning` | Learn temporal patterns from target data |
| `RLMetrics` | `rl_metrics` | Track RL model accuracy and training progress |
| `ThreatModel` | `threat_model` | Multi-signal threat assessment |
| `ThreatAssessment` | `threat_model` | Assessment result with level and signals |
| `estimate_from_multiple_anchors` | `position_estimator` | Fuse RSSI from multiple sensors into position |

### intelligence.geospatial

Satellite imagery analysis, OSM enrichment, terrain classification.

| Class / Function | Module | Description |
|-----------------|--------|-------------|
| `SidewalkGraph` | `sidewalk_graph` | Pedestrian navigation graph from OSM |
| `TerrainLayer` | `terrain_layer` | Classified terrain grid (road, grass, building, etc.) |
| `RoadDetector` | `road_detector` | Detect roads in satellite imagery |
| `ChangeDetector` | `change_detector` | Detect changes between satellite images |
| `MissionGenerator` | `mission_generator` | Generate survey missions from terrain data |

## sim_engine

Tactical simulation engine -- combat, AI, physics, effects.

### sim_engine.core

```python
from tritium_lib.sim_engine.core import SimulationTarget, MovementController, StateMachine
```

| Class | Module | Description |
|-------|--------|-------------|
| `SimulationTarget` | `entity` | Simulation entity with position, health, alliance, type |
| `UnitIdentity` | `entity` | Identity metadata (name, callsign) |
| `MovementController` | `movement` | Smooth movement with acceleration and path following |
| `SpatialGrid` | `spatial` | Grid-based spatial index for efficient neighbor queries |
| `StateMachine` | `state_machine` | FSM for unit behavior states |
| `State` | `state_machine` | Single FSM state |
| `Transition` | `state_machine` | State transition rule |
| `UnitInventory` | `inventory` | Per-unit item/ammo inventory |
| `InventoryItem` | `inventory` | Single inventory item definition |
| `NPCThinker` | `npc_thinker` | LLM-powered NPC decision making |

### sim_engine.combat

```python
from tritium_lib.sim_engine.combat import CombatSystem, WeaponSystem, SquadManager
```

| Class | Module | Description |
|-------|--------|-------------|
| `CombatSystem` | `combat` | Projectile flight, hit detection, damage resolution |
| `Projectile` | `combat` | In-flight projectile with trajectory |
| `Weapon` | `weapons` | Weapon definition (range, damage, rate of fire) |
| `WeaponSystem` | `weapons` | Per-unit weapon management and firing |
| `WEAPON_CATALOG` | `weapons` | Predefined weapon templates |
| `Squad` | `squads` | Group of units operating together |
| `SquadManager` | `squads` | Create, dissolve, and command squads |

### sim_engine.behavior

```python
from tritium_lib.sim_engine.behavior import UnitBehaviors, create_fsm_for_type, NPCManager
```

| Class / Function | Module | Description |
|-----------------|--------|-------------|
| `UnitBehaviors` | `behaviors` | Tick-driven behavior logic for all unit types |
| `create_fsm_for_type` | `unit_states` | Create FSM for a unit type string |
| `create_turret_fsm` | `unit_states` | FSM for turret units |
| `create_rover_fsm` | `unit_states` | FSM for rover units |
| `create_drone_fsm` | `unit_states` | FSM for drone units |
| `create_hostile_fsm` | `unit_states` | FSM for hostile units |
| `UnitMissionSystem` | `unit_missions` | Assign and track unit missions |
| `NPCManager` | `npc` | Spawn and manage civilian NPC population |
| `NPCMission` | `npc` | NPC mission definition (commute, delivery, patrol) |

### sim_engine.game

```python
from tritium_lib.sim_engine.game import GameMode, AmbientSpawner, StatsTracker
```

| Class | Module | Description |
|-------|--------|-------------|
| `GameMode` | `game_mode` | Wave-based battle progression (10 waves) |
| `InfiniteWaveMode` | `game_mode` | Endless wave variant |
| `WaveConfig` | `game_mode` | Per-wave hostile count and types |
| `AmbientSpawner` | `ambient` | Spawn neutral street activity (cars, pedestrians, animals) |
| `StatsTracker` | `stats` | Kill/death/accuracy tracking per unit |
| `DifficultyScaler` | `difficulty` | Adaptive difficulty based on player performance |
| `MoraleSystem` | `morale` | Unit morale with suppression and rally |
| `CrowdDensityTracker` | `crowd_density` | Track civilian density for collateral rules |

### sim_engine.world

```python
from tritium_lib.sim_engine.world import VisionSystem, CoverSystem, plan_path
```

| Class / Function | Module | Description |
|-----------------|--------|-------------|
| `World` | `_world` | Top-level world container |
| `WorldBuilder` | `_world` | Fluent builder for World instances |
| `WORLD_PRESETS` | `_world` | Pre-built world configurations |
| `VisionSystem` | `vision` | Line-of-sight and visibility checks |
| `SightingReport` | `vision` | What a unit can see |
| `SensorSimulator` | `sensors` | Simulate sensor detection probability |
| `SensorDevice` | `sensors` | Simulated sensor hardware |
| `CoverSystem` | `cover` | Cover objects and concealment checks |
| `CoverObject` | `cover` | Single cover piece (wall, barrier, vehicle) |
| `plan_path` | `pathfinding` | A* pathfinding on nav mesh |
| `grid_find_path` | `grid_pathfinder` | Grid-based A* with movement profiles |
| `MovementProfile` | `grid_pathfinder` | Per-unit-type movement costs |

## store

Persistent storage backends (SQLite).

```python
from tritium_lib.store import EventStore, BleStore, TargetStore
```

| Class | Module | Description |
|-------|--------|-------------|
| `BaseStore` | `base` | ABC for SQLite-backed stores with migrations |
| `EventStore` | `event_store` | Tactical event log with severity levels |
| `TacticalEvent` | `event_store` | Single tactical event record |
| `BleStore` | `ble` | BLE sighting history and device records |
| `TargetStore` | `targets` | Persistent target registry |
| `DossierStore` | `dossiers` | Target dossier persistence |
| `ReIDStore` | `reid` | Re-identification embedding storage |
| `ConfigStore` | `config_store` | Key-value configuration persistence |
| `AuditStore` | `audit_log` | Security audit log |
| `ScreenshotStore` | `screenshot_store` | Screenshot storage for visual testing |

## models

Shared data models (dataclasses) -- the contract between SC and edge.

```python
from tritium_lib.models import Device, FleetNode, BleDevice, MeshNode
```

| Class | Module | Description |
|-------|--------|-------------|
| `Device` | `device` | Edge device with ID, capabilities, status |
| `DeviceHeartbeat` | `device` | Periodic device health report |
| `BleDevice` | `ble` | BLE device sighting with RSSI, name, services |
| `BleSighting` | `ble` | Single BLE scan result |
| `FleetNode` | `fleet` | Node in the device fleet |
| `FleetStatus` | `fleet` | Fleet-wide health summary |
| `MeshNode` | `mesh` | Meshtastic mesh radio node |
| `MeshMessage` | `mesh` | Mesh radio message |
| `Command` | `command` | Command sent to a device |
| `FirmwareMeta` | `firmware` | Firmware version and build info |
| `OTAJob` | `firmware` | Over-the-air update job |
| `SensorReading` | `sensor` | Generic sensor data point |
| `TileCoord` | `gis` | Map tile coordinate (z/x/y) |
| `MapLayer` | `gis` | GIS map layer definition |

100+ additional model files cover acoustics, analytics, cameras, convoys, correlations, dossiers, drones, forensics, LPR, radar, SDR, swarm, and more. Browse `tritium_lib/models/` for the full set.

## classifier

Multi-signal device classification.

```python
from tritium_lib.classifier import DeviceClassifier, DeviceClassification
```

| Class | Module | Description |
|-------|--------|-------------|
| `DeviceClassifier` | `device_classifier` | Classify device type from BLE/WiFi signals |
| `DeviceClassification` | `device_classifier` | Classification result (type, confidence, evidence) |

## cot

Cursor on Target XML codec (MIL-STD-2045).

```python
from tritium_lib.cot import device_to_cot, parse_cot
```

| Function | Module | Description |
|----------|--------|-------------|
| `device_to_cot` | `codec` | Convert Device to CoT XML string |
| `sensor_to_cot` | `codec` | Convert SensorReading to CoT XML |
| `parse_cot` | `codec` | Parse CoT XML into dict |

## sdk

Addon development SDK (Apache-2.0 licensed).

```python
from tritium_lib.sdk import AddonBase, AddonContext, SensorAddon
```

| Class | Module | Description |
|-------|--------|-------------|
| `AddonBase` | `addon_base` | Base class for all addons |
| `AddonInfo` | `addon_base` | Addon metadata (name, version, author) |
| `AddonContext` | `context` | Runtime context provided to addons (event bus, MQTT, tracker) |
| `AddonConfig` | `config_loader` | TOML manifest config loader |
| `AddonGeoLayer` | `geo_layer` | Addon-provided map layer |
| `SensorAddon` | `interfaces` | ABC for sensor data source addons |
| `ProcessorAddon` | `interfaces` | ABC for data processing addons |
| `CommanderAddon` | `interfaces` | ABC for commander personality addons |
| `IEventBus` | `protocols` | Event bus protocol interface |
| `IMQTTClient` | `protocols` | MQTT client protocol interface |
| `ITargetTracker` | `protocols` | Target tracker protocol interface |
| `AddonManifest` | `manifest` | Parsed addon.toml manifest |
| `load_manifest` | `manifest` | Load manifest from file path |

## auth

JWT tokens and API key management.

```python
from tritium_lib.auth import create_token, decode_token, generate_api_key
```

| Function | Module | Description |
|----------|--------|-------------|
| `create_token` | `jwt` | Create signed JWT |
| `decode_token` | `jwt` | Decode and validate JWT |
| `TokenType` | `jwt` | Enum (access, refresh, api_key) |
| `generate_api_key` | `jwt` | Generate random API key |
| `hash_api_key` | `jwt` | Hash API key for storage |
| `validate_api_key` | `jwt` | Validate key against hash |

## firmware

Firmware flashing for edge devices.

```python
from tritium_lib.firmware import ESP32Flasher, MeshtasticFlasher
```

| Class | Module | Description |
|-------|--------|-------------|
| `FirmwareFlasher` | `base` | ABC for firmware flashers |
| `ESP32Flasher` | `esp32` | esptool.py-based ESP32 flasher |
| `MeshtasticFlasher` | `meshtastic_flasher` | Meshtastic firmware download and flash |

## sdr

Software Defined Radio abstractions.

```python
from tritium_lib.sdr import SDRDevice, SweepResult
```

| Class | Module | Description |
|-------|--------|-------------|
| `SDRDevice` | `base` | ABC for SDR hardware (HackRF, RTL-SDR) |
| `SDRInfo` | `base` | Device info (serial, firmware version) |
| `SweepResult` | `base` | Broadband sweep data |
| `SweepPoint` | `base` | Single frequency/power measurement |

## graph

Entity relationship graph (KuzuDB).

```python
from tritium_lib.graph import TritiumGraph
```

| Class | Module | Description |
|-------|--------|-------------|
| `TritiumGraph` | `store` | Graph DB for entity relationships (carries, detected_with, traveled_with) |

## ontology

Formal type system and entity schema.

```python
from tritium_lib.ontology import OntologyRegistry, TRITIUM_ONTOLOGY
```

| Class | Module | Description |
|-------|--------|-------------|
| `OntologyRegistry` | `registry` | Runtime registry of entity and relationship types |
| `OntologySchema` | `schema` | Complete schema definition |
| `EntityType` | `schema` | Typed entity definition |
| `RelationshipType` | `schema` | Typed relationship definition |
| `TRITIUM_ONTOLOGY` | `schema` | Default Tritium ontology instance |

## nodes

Sensor node abstractions.

```python
from tritium_lib.nodes import SensorNode, Position
```

| Class | Module | Description |
|-------|--------|-------------|
| `SensorNode` | `base` | ABC for sensor hardware (camera, mic, PTZ) |
| `Position` | `base` | 3D position dataclass |

## notifications

Alert and notification system.

```python
from tritium_lib.notifications import Notification, NotificationManager
```

| Class | Module | Description |
|-------|--------|-------------|
| `Notification` | `__init__` | Single notification (title, message, severity, source) |
| `NotificationManager` | `__init__` | Thread-safe notification collector and broadcaster |

## synthetic

Test data generators.

```python
from tritium_lib.synthetic import BLEScanGenerator, CameraDetectionGenerator
```

| Class | Module | Description |
|-------|--------|-------------|
| `BLEScanGenerator` | `data_generators` | Generate realistic BLE scan data |
| `MeshtasticNodeGenerator` | `data_generators` | Generate mesh node telemetry |
| `CameraDetectionGenerator` | `data_generators` | Generate YOLO detection data |
| `TrilaterationDemoGenerator` | `data_generators` | Generate trilateration demo data |

## web

Reusable HTML components (cyberpunk theme).

```python
from tritium_lib.web import TritiumTheme, DashboardPage, full_page
```

| Class / Function | Module | Description |
|-----------------|--------|-------------|
| `TritiumTheme` | `theme` | Color palette, fonts, CSS variables |
| `DashboardPage` | `dashboard` | Full dashboard HTML generator |
| `StatusBadge` | `components` | Styled status indicator |
| `MetricCard` | `components` | Metric display card |
| `AlertBanner` | `components` | Alert banner component |
| `full_page` | `templates` | Complete HTML page template |

## testing

Visual testing and device automation.

```python
from tritium_lib.testing import VisualCheck, DeviceAPI, FlickerAnalyzer
```

| Class | Module | Description |
|-------|--------|-------------|
| `VisualCheck` | `visual` | OpenCV-based visual regression checks |
| `LayoutIssue` | `visual` | Detected layout problem |
| `FlickerAnalyzer` | `flicker` | Detect UI flicker in frame sequences |
| `DeviceAPI` | `device` | HTTP client for ESP32 device testing |
| `UITestRunner` | `runner` | Automated UI test harness |

## config

Shared configuration base.

```python
from tritium_lib.config import TritiumBaseSettings
```

| Class | Module | Description |
|-------|--------|-------------|
| `TritiumBaseSettings` | `__init__` | Pydantic BaseSettings with .env support |

## JS Modules

Located at `web/` (served by SC at `/lib/`). Additional JS sim modules at `src/tritium_lib/js/`.

### web/sim/

City simulation engine (15 modules):

| File | Description |
|------|-------------|
| `road-network.js` | Road graph from OSM data, lane geometry |
| `traffic-controller.js` | Traffic light state machine, signal phases |
| `idm.js` | Intelligent Driver Model (car following) |
| `mobil.js` | MOBIL lane change model |
| `vehicle.js` | Vehicle entity class |
| `pedestrian.js` | Pedestrian entity and ORCA model |
| `identity.js` | NPC identity generation |
| `daily-routine.js` | NPC daily schedule templates |
| `schedule-executor.js` | Execute NPC daily routines |
| `protest-engine.js` | Epstein civil unrest model |
| `protest-scenario.js` | Protest scenario configurations |
| `weather.js` | Weather simulation and effects |
| `spatial-grid.js` | Spatial hash grid for neighbor queries |
| `procedural-city.js` | Procedural city generation |
| `index.js` | Sim module re-exports |

### js/render/

3D rendering utilities for Three.js frontend.

### js/ui/

Shared UI components and panels.
