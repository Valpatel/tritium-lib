# Tritium-Lib Changelog

Changes tracked with verification status. All changes on `dev` branch.

## Verification Levels

| Level | Meaning |
|-------|---------|
| **Unit Tested** | Passes `pytest tests/` |
| **Consumer Tested** | Verified working in tritium-edge or tritium-sc imports |
| **Human Verified** | Manually reviewed by a human |

---

## 2026-03-14 — Wave 76: Behavioral Pattern Learning Models

### BehaviorPattern + PatternAnomaly + CoPresenceRelationship + PatternAlert (Unit Tested, 24 tests)
- New `models/pattern.py` — behavioral pattern learning data contracts
- `BehaviorPattern`: pattern_id, target_id, pattern_type, status, confidence, schedule, locations, observation_count
- `PatternAnomaly`: anomaly_id, deviation_type, deviation_score, expected/actual behavior descriptions
- `CoPresenceRelationship`: temporal/spatial correlation, co-occurrence count, confidence computation
- `PatternAlert`: alert_id, pattern_id, severity, deviation_threshold, cooldown, fire tracking
- `TimeSlot`: recurring schedule with day-of-week filtering and midnight wrap
- `LocationCluster`: spatial cluster center, radius, visit count, dwell time
- `compute_temporal_correlation()`: sliding-window co-occurrence correlation for two timestamp lists
- `detect_time_regularity()`: circular mean time-of-day with configurable tolerance
- 8 enums: PatternType (daily/weekly/commute/dwell/arrival/departure/co_presence/periodic), PatternStatus, DeviationType
- Exported in models/__init__.py as LearnedBehaviorPattern (avoids collision with behavior.py)

---

## 2026-03-14 — Wave 74: 3D Visualization Models

### Scene3DConfig + TrajectoryRibbon + CoverageVolume + TimelineConfig (Unit Tested, 18 tests)
- New `models/visualization.py` — 3D scene rendering parameter contracts
- `TrajectoryRibbon`: target_id, alliance, color, min/max width, opacity, fade_tail, time_height_scale, max_points, glow
- `CoverageVolume`: sensor_id, volume_type (cone/sphere/cylinder/frustum), range, FOV, heading, tilt, color, wireframe, pulse
- Factory methods: `for_camera()`, `for_ble()`, `for_wifi()` with sensible defaults
- `TimelineConfig`: start/end/current time, speed, loop, trail_duration, computed duration/progress properties
- `Scene3DConfig`: aggregates ribbons, volumes, timeline, plus scene settings (grid, fog, lighting, shadows, fps)
- `AllianceColor` enum: standard alliance-to-hex-color mapping
- `SensorVolumeType` enum: cone, sphere, cylinder, frustum
- All 6 types exported from `tritium_lib.models` top-level package

---

## 2026-03-14 — Wave 73: Spatial Intelligence + Indoor Mapping Models

### FloorPlan + Room + IndoorPosition + WiFiRSSIFingerprint (Unit Tested, 17 tests)
- New `models/floorplan.py` — indoor spatial intelligence data contracts
- `FloorPlan`: plan_id, name, building, floor_level, image_path, bounds, rooms, anchors, status, opacity, rotation
- `Room`: room_id, name, room_type (13 types), polygon (lat/lon), capacity, ray-casting contains_point()
- `FloorPlanBounds`: north/south/east/west with contains() and center properties
- `GeoAnchor`: pixel-to-lat/lon mapping for floor plan geo-referencing
- `IndoorPosition`: target_id, plan_id, room_id, floor_level, lat/lon, confidence, method
- `RoomOccupancy`: per-room person/device counts with occupancy_ratio
- `BuildingOccupancy`: whole-building occupancy summary
- `WiFiRSSIFingerprint`: BSSID->RSSI map at known position for fingerprint-based positioning
- `FloorPlanStatus` enum: draft, active, archived
- `RoomType` enum: office, conference, hallway, bathroom, kitchen, lobby, storage, server_room, stairwell, elevator, open_area, restricted, other
- All 11 types exported from `tritium_lib.models` top-level package

---

## 2026-03-14 — Wave 71: Edge-to-Cloud Intelligence Pipeline Models

### FeatureVector + ClassificationFeedback + EdgeIntelligenceMetrics (Unit Tested, 17 tests)
- New `models/feature_vector.py` — compact feature vectors for edge ML pipeline
- `FeatureVector`: source_id, mac, source_type (BLE/WiFi/acoustic/RF/camera), features dict, version, timestamp, feature_list() helper
- `AggregatedFeatures`: per-device aggregation across edge nodes with compute_mean()
- `ClassificationFeedback`: SC-to-edge classification result feedback (mac, predicted_type, confidence, confirmed_by)
- `EdgeIntelligenceMetrics`: per-node ML health (devices seen, classified, feedback received, accuracy rate)
- `FeatureSource` enum: BLE, WIFI, ACOUSTIC, RF, CAMERA
- All exported from `tritium_lib.models` top-level package

---

## 2026-03-14 — Wave 70: MILESTONE — Movement Analytics Models

### MovementAnalytics + FleetMetrics (Unit Tested, 14 tests)
- New `models/movement_analytics.py` — per-target movement analytics and fleet aggregates
- `MovementAnalytics`: target_id, avg_speed_mps, max_speed_mps, total_distance_m, dwell_times, direction_histogram (8 compass bins), activity_periods, current_speed/heading, is_stationary
- `FleetMetrics`: total/moving/stationary targets, avg/max fleet speed, total distance, busiest zone, dominant direction
- `FleetMetrics.from_analytics()` factory — computes aggregates from list of per-target analytics
- `ActivityPeriod`: start/end epoch, avg speed, distance, duration
- `DwellTime`: zone_id, zone_name, total_seconds, entry_count, last entry/exit
- All models with `to_dict()` / `from_dict()` roundtrip serialization
- Exported from `models/__init__.py`

---

## 2026-03-14 — Wave 67: Deployment Models

### Deployment Config Models (Unit Tested, 18 tests)
- New `models/deployment.py` — deployment configuration and service status tracking
- `ServiceName` enum: sc_server, mqtt_broker, meshtastic_bridge, ollama, edge_fleet_server, ros2_bridge, go2rtc
- `ServiceState` enum: running, stopped, error, starting, unknown
- `ServiceStatus`: name, display_name, state, pid, uptime_s, port, version, can_start/stop, start/stop commands
- `SystemRequirements`: python_version, system_packages, python_packages, optional_packages, min_ram_mb, ports_needed
- `DeployedService`: service + host + status + installed + autostart
- `DeploymentConfig`: site_id, hostname, services, requirements, edge_devices + helper methods (service_by_name, all_running, summary)
- Exported from `models/__init__.py`

---

## 2026-03-14 — Wave 64: Environment Sensor Model

### EnvironmentReading Model (Unit Tested, 10 tests)
- New `models/environment.py` — standardized environmental sensor data model
- `EnvironmentReading`: temperature_c, humidity_pct, pressure_hpa, air_quality_index, light_level_lux, noise_level_db, gas_resistance_ohm, uv_index, wind/rainfall
- `EnvironmentSnapshot`: aggregated readings from multiple sources with avg_temperature_c, avg_humidity_pct, avg_pressure_hpa
- `EnvironmentSource` enum: meshtastic, edge_device, weather_api, manual
- Properties: temperature_f (auto-conversion), has_data, summary_line()
- Exported from `models/__init__.py`

---

## 2026-03-14 — Wave 63: MeshNodeExtended Model

### MeshNodeExtended (Unit Tested, 18 tests)
- New `models/mesh_node_extended.py` — full Meshtastic node model for real hardware
- `MeshNodePosition`: GPS with sats_in_view, ground_speed, fix_quality
- `MeshNodeDeviceMetrics`: battery_level, voltage, channel_utilization, air_util_tx, uptime
- `MeshNodeEnvironment`: temperature, humidity, barometric_pressure, gas_resistance, IAQ
- `MeshNodeRadioMetrics`: snr, rssi, hop_limit, hop_start, computed hops_away
- `MeshNodeExtended.from_meshtastic_node()`: factory from meshtastic library dict format
- Properties: has_position, battery_percent, display_name, age_seconds
- Exported from `models/__init__.py`

---

## 2026-03-14 — Wave 59: Screenshot Store

### ScreenshotStore (Unit Tested, 9 tests)
- New `store/screenshot_store.py` — SQLite-backed tactical map screenshot persistence
- Save PNG binary with operator, description, dimensions, tags metadata
- List (paginated, filterable by operator), get (with binary), delete, count
- Exported from `store/__init__.py`

---

## 2026-03-14 — Wave 56: Anomaly Detection ABC

### AnomalyDetector ABC (Unit Tested, 18 tests)
- New `intelligence/anomaly.py` — AnomalyDetector ABC with `detect(current_metrics, baseline) -> list[Anomaly]`
- `SimpleThresholdDetector`: flags metrics > N sigma from baseline mean, severity levels, no external deps
- `AutoencoderDetector`: trains single-hidden-layer autoencoder, flags high reconstruction error, requires numpy
- `Anomaly` dataclass: metric_name, current_value, baseline stats, deviation sigma, severity, score
- Exported from `tritium_lib.intelligence` package

## 2026-03-14 — Wave 53: Intelligence Scorer ABC

### CorrelationScorer ABC (Unit Tested, 19 tests)
- New `intelligence/scorer.py` — CorrelationScorer ABC with predict(features) -> ScorerResult
- `StaticScorer`: hand-tuned weighted linear model with sigmoid, configurable weights and bias
- `LearnedScorer`: wraps trained sklearn LogisticRegression, falls back to StaticScorer on error
- Save/load trained models to pickle files via `LearnedScorer.from_file()` / `.save()`
- `ScorerResult` dataclass: probability, confidence, method, detail
- Canonical `FEATURE_NAMES`: distance, rssi_delta, co_movement, device_type_match, time_gap, signal_pattern
- Numerically stable sigmoid implementation
- New `intelligence/__init__.py` — exports all scorer classes

---

## 2026-03-14 — Wave 52: ML Training Data Models

### Training Data Models (Unit Tested, 11 tests)
- New `models/training.py` — TrainingExample, CorrelationTrainingData, ClassificationTrainingData, FeedbackRecord
- `DecisionType` enum: correlation, classification, threat_assessment, alliance_override
- `TrainingExample`: features dict, label, confidence (0-1 validated), source, timestamp, confirmed_by
- `CorrelationTrainingData`: target pair, features, score, decision, outcome for correlation pipeline
- `ClassificationTrainingData`: target_id, features, predicted/correct types and alliances
- `FeedbackRecord`: operator confirm/reject with notes for RL training
- Exported via models `__init__.py`, added to `__all__`

## 2026-03-14 — Wave 51: Map Sharing, Macros, Grid, Power Tracking, Templates

### Report Template Models (Unit Tested, 18 tests)
- New `models/template.py` — ReportTemplate, TemplateSection, TemplateVariable
- `ReportFormat` enum: plaintext, markdown, HTML, PDF, cot_xml
- `TemplateSectionType` enum: header, summary, findings, timeline, targets, recommendations, appendix, map_snapshot, sensor_data, custom
- `TemplateVariable`: name, label, type, default_value, required, source (auto-fill from tracker/fleet/dossier)
- `TemplateSection`: section_id, title, body_template with {{variable}} placeholders, ordering
- `ReportTemplate`: template_id, name, sections, variables, format, version, tags
- `render_preview()`: substitutes variables with provided/default values
- `get_required_variables()`, `get_section_order()` helper methods
- 3 built-in templates: SITREP, Mission Briefing, Investigation Report
- `BUILTIN_TEMPLATES` list for programmatic access
- Full Pydantic v2 serialization roundtrip verified

## 2026-03-14 — Wave 50: Multi-User & Operational Readiness

### User & Session Models (Unit Tested, 18 tests)
- New `models/user.py` — User, UserRole, Permission, UserSession, ROLE_PERMISSIONS
- `UserRole` enum: admin, commander, analyst, operator, observer
- `Permission` enum: 22 granular permissions covering targets, missions, fleet, intel, sensors, system, automation, briefings, Amy
- `ROLE_PERMISSIONS` mapping: default permission sets per role (admin=all, commander=tactical, analyst=intel, operator=fleet, observer=read-only)
- `User` dataclass: user_id, username, display_name, role, permissions, active_since, last_action, email, color
- `UserSession` dataclass: session tracking with cursor_lat/lng for real-time sharing
- `has_permission()`: checks explicit overrides then falls back to role defaults
- `get_effective_permissions()`: returns full effective permission set
- Roundtrip serialization via `to_dict()`/`from_dict()`
- `DeviceHeartbeat` gains `device_group` field for edge device group management

## 2026-03-14 — Wave 48: Network Topology Models

### NetworkNode, NodeRole, PeerQuality Models (Unit Tested, 12 tests)
- New `NetworkNode` model: node_id, name, role, position, health metrics, peer stats
- New `NodeRole` enum: gateway, relay, leaf, sensor
- New `PeerQuality` model: per-peer RSSI trend, packet loss, tx/rx counts
- `PeerQuality.quality_score` property: 0-100 computed from RSSI + loss penalty
- `NetworkLink` gains `packet_loss_pct` and `quality_score` fields
- `FleetTopology` gains optional `network_nodes` list for rich visualization
- All exported from `tritium_lib.models` — used by fleet dashboard and comm-link layer

## 2026-03-14 — Wave 45: Tactical Scenario Models

### TacticalScenario Model (Unit Tested, 12 tests)
- Added `models/scenario.py` — structured test scenarios and training exercises
- `TacticalScenario`: scenario_id, title, description, actors, events, timeline, objectives
- `ScenarioActor`: actor_id, name, type, alliance, position, BLE/WiFi properties, waypoints
- `ScenarioEvent`: event_type, time_offset_s, actor associations, expected results
- `ScenarioObjective`: description, priority, success_criteria, time_limit, score_value
- Enums: ScenarioStatus (7 states), ActorType (9 types), ActorAlliance (4 values), ScenarioEventType (13 types)
- Helper methods: computed_duration(), actor_by_id(), events_for_actor(), sorted_events(), completion_pct(), to_dict()
- Exported in `models/__init__.py` with full `__all__` entries

---

## 2026-03-14 — Wave 44: Communication Channel Models

### CommChannel Model (Unit Tested, 14 tests)
- Added `models/comms.py` — CommChannel, ChannelType, ChannelStatus, ChannelAuth
- ChannelType enum: MQTT, TAK, WebSocket, federation, serial, HTTP, ESP-NOW, LoRa
- ChannelStatus: disconnected, connecting, connected, error, disabled
- AuthType: none, basic, token, certificate, PSK
- ChannelHealth: uptime, latency, error rate, throughput summary
- ChannelInventory: aggregate summary with type/status counts
- `summarize_channels()` — build inventory from channel list
- `select_best_channel()` — pick best connected channel by priority and latency
- Exported in `models/__init__.py` with full `__all__` entries

---

## 2026-03-14 — Wave 43: Device Capability Advertisement Models

### Capability Advertisement (Unit Tested, 13 tests)
- Added `models/capability.py` — DeviceCapability, CapabilityAdvertisement, CapabilityType
- `DeviceCapability`: cap_type, version, enabled, config, description with to_summary()
- `CapabilityAdvertisement`: device_id, board, firmware_version, capabilities list
  - has_capability() / get_capability() for querying
  - capability_types() for listing enabled capabilities
  - to_heartbeat_list() for backward compatibility with DeviceCapabilities.from_list()
- `CapabilityType` enum: 26 standard types matching edge HAL names
- Added ble/wifi boolean fields to DeviceCapabilities model
- Added edge_capabilities() topic to TritiumTopics MQTT builder
- All 1,462 tests passing (no regressions)

---

## 2026-03-14 — Wave 41: Operational Period Models

### Operational Period (Unit Tested, 17 tests)
- Added `models/operational.py` — OperationalPeriod for structuring operations into defined time blocks
- `OperationalPeriod`: period_id, start, end, commander, objectives, weather, personnel_count, phase, site_id
- `OperationalPhase` enum: planned, briefing, active, transition, debriefing, completed, cancelled
- `OperationalObjective`: description, priority, completed status, assigned_to
- `WeatherInfo`: condition, temperature, wind speed/direction, visibility, humidity
- Lifecycle methods: `activate()`, `complete()`, `cancel()`, `complete_objective()`
- Properties: `progress` (0.0-1.0), `is_terminal`, `duration_seconds`
- Full `to_dict()` serialization for JSON transport
- Exported from `tritium_lib.models` namespace

---

## 2026-03-14 — Wave 39: Graph Store Fix, Test Baseline

- Fixed `TritiumGraph.__init__` — `mkdir(parents=True)` now inside try/except block so invalid paths (e.g., `/dev/null/...`) correctly raise RuntimeError instead of NotADirectoryError (Unit Tested)
- Test baseline: 1404 passed, 0 failures (up from 1357 in Wave 36)
- Model `__all__` exports verified: 250 symbols importable via `from tritium_lib.models import *`
- All 42 model files have explicit imports in `__init__.py`

---

## 2026-03-14 — Wave 38: Sensor Config, Multi-Camera, Target Merge, Power Saver

- Added `models/sensor_config.py` — SensorPlacement configuration model (Unit Tested, 19 tests)
  - `SensorPlacement`: sensor_id, position, height, fov, rotation, tilt, coverage_radius, sensor_type, mounting_type, status
  - `SensorPosition`: lat/lng/alt + local x/y/z coordinates
  - `SensorArray`: collection of sensors with filtering by type/status
  - `SensorType` enum: ble_radio, wifi_radio, camera, microphone, radar, lidar, pir, etc.
  - `MountingType` enum: wall, ceiling, pole, tripod, vehicle, drone, handheld, etc.
  - `SensorStatus` enum: online, offline, degraded, calibrating, error
  - Coverage area calculation (omni vs. sector), bearing containment check
- Exported all sensor_config types from `models/__init__.py`

---

## 2026-03-14 — Wave 35: Intelligence Reports

- Added `models/report.py` — IntelligenceReport model (Unit Tested)
  - `IntelligenceReport`: report_id, title, summary, entities, findings, recommendations, created_by, classification_level
  - `ReportFinding`: structured findings with confidence scores and evidence refs
  - `ReportRecommendation`: actionable recommendations with priority levels
  - `ClassificationLevel` enum: unclassified, fouo, confidential, secret
  - `ReportStatus` enum: draft, review, final, archived
  - `mark_final()`, `add_finding()`, `add_recommendation()` methods
  - Exported from `tritium_lib.models` package
  - 6 new tests passing

---

## 2026-03-14 — Wave 33: System Config + Mission Management

- Added `TritiumSystemConfig` model to `models/config.py` (Unit Tested)
  - System-level configuration: map defaults, scan intervals, notification prefs, theme
  - `MapDefaults`: center lat/lng, zoom, tilt, bearing, style
  - `ScanIntervals`: BLE, WiFi, probe, heartbeat, sighting intervals
  - `NotificationPrefs`: sound, geofence breach, threat escalation, suspicious device toggles
  - `to_dict()`, `from_dict()`, `save_to_store()`, `load_from_store()` for ConfigStore integration
  - All 4 classes exported from `tritium_lib.models`
  - 1351 tests passing

---

## 2026-03-14 — Wave 31: ConfigStore

- Added `store/config_store.py` — persistent system configuration store (Unit Tested)
  - Namespaced key-value pairs in SQLite WAL
  - `set/get/delete/clear_namespace/list_namespaces/count/set_many`
  - JSON serialization: `set_json/get_json` for complex values
  - Thread-safe via BaseStore locking
  - 16 tests passing in `tests/store/test_config_store.py`
  - Exported in `tritium_lib.store.__init__`

---

## 2026-03-14 — Wave 26: Event Schema System

### Event Schemas
| Change | Verification |
|--------|-------------|
| Added `models/event_schema.py` with 41 typed event schemas across 17 domains | Unit Tested (1273 tests) |
| EventDomain enum: simulation, combat, game, NPC, fleet, mesh, edge, TAK, sensor, target, dossier, federation, hazard, unit, mission, Amy, audio | Unit Tested |
| `validate_event_type()`, `get_event_schema()`, `list_event_types()` helpers | Unit Tested |

---

## 2026-03-14 — Wave 25: Maintenance & Quality

### BaseStore Migration
| Change | Verification |
|--------|-------------|
| BleStore, TargetStore, ReIDStore now inherit from BaseStore | Unit Tested (1273 tests) |
| Removed 85 lines of duplicated boilerplate (connect, WAL, lock, close) | Unit Tested |
| Dead `oui_lookup` import removed from DeviceClassifier | Unit Tested |
| READMEs added for `web/` and `config/` modules | Documented |

---

## 2026-03-14 — Wave 15: Federation Models

### Multi-Site Federation
| Change | Verification |
|--------|-------------|
| `models/federation.py` — FederatedSite, SiteConnection, SharedTarget, FederationMessage | Unit Tested (23 tests) |
| Enums: SiteRole, ConnectionState, SharePolicy, FederationMessageType | Unit Tested |
| Utilities: federation_topic() builder, is_message_expired() checker | Unit Tested |
| All models registered in models/__init__.py with proper __all__ exports | Unit Tested |

---

## 2026-03-13 — Wave 9: Graph Database & Ontology Schema

### KuzuDB Graph Store
| Change | Verification |
|--------|-------------|
| `graph/kuzu_store.py` — KuzuDB embedded graph database for ontology layer | Unit Tested |
| Node CRUD: create, read, update, delete typed entities | Unit Tested |
| Edge CRUD: typed relationships between entities (CARRIES, DETECTED_WITH, etc.) | Unit Tested |
| Cypher query interface for traversal and pattern matching | Unit Tested |

### Ontology Schema & Registry
| Change | Verification |
|--------|-------------|
| `ontology/schema.py` — formal ontology: 10 entity types, 12 relationships, 3 interfaces | Unit Tested |
| Entity types: Person, Device, Vehicle, Location, Network, Organization, Event, Alert, Asset, Zone | Unit Tested |
| Relationship types: OWNS, CARRIES, DETECTED_AT, CONNECTED_TO, MEMBER_OF, etc. | Unit Tested |
| Schema validation and type-safe entity/relationship construction | Unit Tested |

### DossierStore Enhancements
| Change | Verification |
|--------|-------------|
| `_update_json_field` helper for atomic tag/note updates in DossierStore | Unit Tested |

---

## 2026-03-13 — Wave 7: Dossiers & Target Intelligence

### Models — Dossier
| Change | Verification |
|--------|-------------|
| `models/dossier.py` — Target Dossier model for persistent entity intelligence | Unit Tested |

### Stores — DossierStore
| Change | Verification |
|--------|-------------|
| `store/dossier.py` — SQLite-backed DossierStore for persistent target intelligence | Unit Tested |

---

## 2026-03-13

### Models — New
| Change | Verification |
|--------|-------------|
| `models/meshtastic.py` — MeshtasticNode, MeshtasticMessage, MeshtasticWaypoint, MeshtasticStatus | Unit Tested |
| `models/camera.py` — CameraSource, CameraFrame, CameraDetection, BoundingBox | Unit Tested |
| All models exported from `models/__init__.py` | Unit Tested |

### MQTT Topics — New
| Change | Verification |
|--------|-------------|
| `meshtastic_nodes()`, `meshtastic_message()`, `meshtastic_command()` | Unit Tested |
| `camera_feed()`, `camera_snapshot()` | Unit Tested |
| `all_meshtastic()` wildcard subscription | Unit Tested |

### Infrastructure
| Change | Verification |
|--------|-------------|
| `testing/__init__.py` — lazy imports for cv2/numpy/requests deps | Unit Tested |
| `pyproject.toml` — `[testing]` optional dep group | Unit Tested |

### Documentation
| Change | Verification |
|--------|-------------|
| `CLAUDE.md` — submodule context with polyglot vision | N/A (docs) |
| `src/tritium_lib/README.md` — package reference with model categories | N/A (docs) |
| `README.md` — updated model table | N/A (docs) |
| `LICENSE` — AGPL-3.0 added | N/A (legal) |

---

## Test Baseline

| Suite | Count | Status | Date |
|-------|-------|--------|------|
| pytest tests/ | 833 | All passing | 2026-03-13 |
| Meshtastic model tests | 27 | All passing | 2026-03-13 |
| Camera model tests | 27 | All passing | 2026-03-13 |
| MQTT topic tests | 29 | All passing | 2026-03-13 |
