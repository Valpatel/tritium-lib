# Tritium-Lib Changelog

Changes tracked with verification status. All changes on `dev` branch.

## Verification Levels

| Level | Meaning |
|-------|---------|
| **Unit Tested** | Passes `pytest tests/` |
| **Consumer Tested** | Verified working in tritium-edge or tritium-sc imports |
| **Human Verified** | Manually reviewed by a human |

---

## 2026-03-15 — Wave 152: Maintenance

| Change | Verification |
|--------|-------------|
| Lib pytest: 2,527 passing (up from 2,516) | pytest verified |
| 97 model files, 131 test files, 262+ Pydantic classes | Verified |

---

## 2026-03-15 — Wave 151: AcousticFeatureVector Model

| Change | Verification |
|--------|-------------|
| AcousticFeatureVector model — compact MFCC feature transport for edge-to-SC serialization | 11 tests pass |
| Lib total: 2,527 tests passing, 97 model files, 131 test files | Verified |

---

## 2026-03-15 — Wave 147: KML Builder + TrailExport Models

| Change | Verification |
|--------|-------------|
| KmlDocument, KmlPlacemark, KmlStyle — KML document builder for target trail export | Unit Tested |
| TrailExport models — structured trail output with format selection (GPX/KML) | Unit Tested |
| Lib total: 2,516 tests passing, 96 model files, 130 test files | Verified |

---

## 2026-03-15 — Wave 146: Notification Template Enhancements

| Change | Verification |
|--------|-------------|
| NotificationTemplate: added ble:first_seen and convoy_detected built-in templates | 2477 tests pass |
| Lib total: 2,477 tests passing, 94 model files, 128 test files | Verified |

---

## 2026-03-15 — Wave 145: Convoy Model

| Change | Verification |
|--------|-------------|
| Convoy model — convoy_id, member_target_ids, speed_avg, heading_avg, formation, suspicious_score | 12 tests pass |
| ConvoyFormation enum — LINE, CLUSTER, SPREAD, UNKNOWN | Unit Tested |
| ConvoyStatus enum — ACTIVE, DISPERSED, STOPPED, MERGED | Unit Tested |
| ConvoySummary — aggregate stats for active convoys | Unit Tested |
| compute_suspicious_score — weighted scoring: heading/speed coordination, duration, member count | Unit Tested |

---

## 2026-03-15 — Wave 144: DailyPattern Model

| Change | Verification |
|--------|-------------|
| DailyPattern model — 24-bin hourly histogram with peak_hour, quiet_hours, regularity_score | 17 tests pass |
| compute_regularity_score — normalized entropy (1.0 = single spike, 0.0 = uniform) | Unit Tested |
| is_daytime_only / is_nighttime_only properties for activity classification | Unit Tested |
| add_sighting(hour), recompute(), active_hours property | Unit Tested |
| Registered in models/__init__.py __all__ | Consumer Tested |
| Lib total: 2,411 tests passing (29 skipped) | Unit Tested |

---

## 2026-03-15 — Wave 141: BehaviorCluster Model

| Change | Verification |
|--------|-------------|
| models/clustering.py — BehaviorCluster, ClusterSummary, CommonPattern, FormationType for behavioral target grouping | 7 tests passing |
| BehaviorCluster supports add/remove/merge targets, weighted centroid merging, formation detection | Unit Tested |
| FormationType enum: convoy, swarm, patrol, dispersed, stationary, unknown | Unit Tested |
| ClusterSummary.from_cluster() for lightweight API responses | Unit Tested |
| Added to models/__init__.py and __all__ | Import verified |

---

## 2026-03-15 — Wave 132: Transition Event Model

| Change | Verification |
|--------|-------------|
| models/transition.py — TransitionEvent, TransitionHistory, TransitionType for target state change tracking | 13 tests passing |
| Supports indoor/outdoor, zone crossing, speed change, classification change, visibility transitions | Unit Tested |
| Serialization: to_dict/from_dict with full roundtrip | Unit Tested |
| TransitionHistory: bounded history with count_by_type, last_transition queries | Unit Tested |
| Exported in models/__init__.py and __all__ | Consumer Tested |

---

## 2026-03-15 — Wave 131: Target Group Model

| Change | Verification |
|--------|-------------|
| models/target_group.py — TargetGroup and TargetGroupSummary models for operator-defined target collections | 7 tests passing |
| TargetGroup supports add/remove/has_target, color, icon, created_by | Unit Tested |
| Exported in models/__init__.py | Consumer Tested |

---

## 2026-03-15 — Wave 129: Vehicle Tracking Model

| Change | Verification |
|--------|-------------|
| models/vehicle.py — VehicleTrack Pydantic model for YOLO-detected vehicle behavior analysis | 27 tests passing |
| compute_speed_mph(): speed from consecutive frame positions | Unit Tested |
| compute_heading(): compass heading from position delta | Unit Tested |
| compute_suspicious_score(): loitering, unusual location, slow crawling, erratic speed/heading | Unit Tested |
| heading_to_label(): compass direction (N, NE, E, etc.) | Unit Tested |
| Exported from tritium_lib.models | Consumer Tested |

---

## 2026-03-15 — Wave 126: Feature Engineering for RL Correlation

| Change | Verification |
|--------|-------------|
| intelligence/feature_engineering.py — 5 reusable feature functions for correlation learners | Unit Tested (34 tests passing) |
| device_type_match(): semantic cross-sensor type compatibility (phone+person=1.0, watch+person=0.95) | Unit Tested |
| co_movement_score(): trail-based co-located movement with linear interpolation | Unit Tested |
| time_similarity(): circular time-of-day matching wrapping around midnight | Unit Tested |
| source_diversity(): cross-category sensor diversity scoring (RF+visual bonus) | Unit Tested |
| wifi_probe_temporal_correlation(): BLE + WiFi probe timing match with same_observer bonus | Unit Tested |
| EXTENDED_FEATURE_NAMES constant: all 10 features listed for correlation scoring | Unit Tested |
| build_extended_features(): helper to construct complete 10-feature dicts | Unit Tested |
| scorer.py FEATURE_NAMES expanded from 6 to 10, DEFAULT_WEIGHTS rebalanced | Unit Tested (121 total) |
| Exported from tritium_lib.intelligence — available to tritium-sc learners | Consumer Tested |

---

## 2026-03-15 — Wave 119: GeofenceEvent Model

| Change | Verification |
|--------|-------------|
| models/geofence_event.py — GeofenceEvent dataclass with target_id, zone_id, direction, timestamp, target_alliance, zone_type, zone_name, position | Unit Tested (7 tests passing) |
| to_dict/from_dict serialization with optional field omission (empty zone_name, null position) | Unit Tested |
| Roundtrip serialization test verified | Unit Tested |
| Exported from tritium_lib.models — available to tritium-sc and tritium-edge | Consumer Tested |

---

## 2026-03-14 — Wave 116: Commander Personality Model

| Change | Verification |
|--------|-------------|
| models/personality.py — CommanderPersonality dataclass with 5 traits: aggression, curiosity, verbosity, caution, initiative | Unit Tested (18 tests passing) |
| Preset profiles: PATROL, BATTLE, STEALTH, OBSERVER with tuned defaults for different operational contexts | Unit Tested |
| profile_label property for human-readable dominant trait description | Unit Tested |
| Clamping, serialization (to_dict/from_dict), roundtrip tested | Unit Tested |
| Exported from tritium_lib.models — available to tritium-sc Amy personality API | Consumer Tested |

---

## 2026-03-14 — Wave 111: Benchmark Result Models

| Change | Verification |
|--------|-------------|
| models/benchmark.py — BenchmarkResult, BenchmarkSuite, BenchmarkUnit for standardized performance reporting across the ecosystem | Unit Tested (2236 tests passing) |
| BenchmarkSuite.add() with auto pass/fail evaluation (higher_is_better support), report() for multi-line output, to_dict() for API serialization | Consumer Tested |

## 2026-03-14 — Wave 108: Proximity Alert Models

| Change | Verification |
|--------|-------------|
| models/proximity.py — ProximityAlert, ProximityRule, AlliancePair, ProximitySeverity for entity-to-entity distance monitoring | Unit Tested (17 tests) |
| classify_proximity_severity() — severity classification based on distance/threshold ratio | Unit Tested |
| ProximityRule.matches_alliance() — alliance pair matching including any_different mode | Unit Tested |
| DEFAULT_PROXIMITY_RULES — hostile_friendly at 10m, unknown_friendly at 15m | Unit Tested |
| Added all exports to models/__init__.py and __all__ | Unit Tested |

## 2026-03-14 — Wave 107: CoT Target Export Models

| Change | Verification |
|--------|-------------|
| models/tak_export.py — CoTExportEvent, CoTExportPoint for generating Cursor on Target XML from target data | Unit Tested (11 tests) |
| targets_to_cot_xml() and targets_to_cot_file() for batch CoT export | Unit Tested |
| Alliance-based type codes, asset type mapping, team colors, position fallback to x/y | Unit Tested |

## 2026-03-14 — Wave 104: GPX Export Models

| Change | Verification |
|--------|-------------|
| models/gpx.py — GPXDocument, GPXTrack, GPXRoute, GPXWaypoint for GPX 1.1 XML generation | Unit Tested (17 tests) |
| Builder pattern for creating valid GPX exports for ATAK, Google Earth, GIS tools | Unit Tested |
| Added to models/__init__.py exports | Unit Tested |

## 2026-03-14 — Wave 103: Intelligence Package Model

| Change | Verification |
|--------|-------------|
| models/intelligence_package.py — IntelligencePackage, PackageTarget, PackageEvent, PackageDossier, PackageEvidence, ChainOfCustody for portable inter-site intelligence sharing | Unit Tested (32 tests) |
| IntelClassification enum — unclassified, restricted, confidential, secret | Unit Tested |
| PackageStatus enum — draft, finalized, transmitted, received, imported, rejected lifecycle | Unit Tested |
| create_intelligence_package() factory with automatic chain of custody | Unit Tested |
| validate_package_import() pre-validation (expiration, classification, integrity) | Unit Tested |
| __init__.py updated to export all 12 new symbols | Unit Tested |

## 2026-03-14 — Wave 101: Quick Action Model

| Change | Verification |
|--------|-------------|
| models/quick_action.py — QuickAction, QuickActionLog, QuickActionType for logging tactical actions on targets | Unit Tested (10 tests) |
| QuickActionType enum — investigate, watch, classify, track, dismiss, escalate, annotate | Unit Tested |
| QuickActionLog — in-memory log with for_target(), by_type(), recent() queries | Unit Tested |
| Exported in models/__init__.py and __all__ | Unit Tested |

## 2026-03-14 — Wave 97: Sensor Health Monitoring Models

| Change | Verification |
|--------|-------------|
| models/sensor_health.py — SensorHealthMetrics, SensorArrayHealth, SensorBaseline, SensorAlert, SensorHealthStatus enum, classify_sensor_health() function | Unit Tested (20 tests) |
| SensorBaseline — learned baseline with mean/stddev/training window, deviation_from() method | Unit Tested |
| SensorAlert — alert model with severity, deviation %, sigma, recommended action | Unit Tested |
| Exported in models/__init__.py — SensorArrayHealth, SensorAlert, SensorBaseline, SensorHealthMetrics, SensorHealthStatus, classify_sensor_health | Import verified |
| tests/test_sensor_health.py — 20 tests covering classification thresholds, array health aggregation, roundtrip serialization | All passing |

## 2026-03-14 — Wave 95: Confidence Decay Model

| Change | Verification |
|--------|-------------|
| models/confidence.py — ConfidenceModel with exponential decay, SourceType enum, DEFAULT_HALF_LIVES (BLE 30s, WiFi 45s, YOLO 15s, mesh 120s, sim never), decay/is_stale/time_to_stale methods | Unit Tested (15 tests) |
| Exported in models/__init__.py — ConfidenceModel, DEFAULT_HALF_LIVES, SourceType | Import verified |
| tests/models/test_confidence.py — 15 tests covering decay math, staleness, serialization, source fallback | All passing |
| Wave 94 test_analytics_dashboard.py — verified 9/9 passing | Verified |

## 2026-03-14 — Wave 94: Analytics Dashboard Widgets

| Change | Verification |
|--------|-------------|
| models/analytics_dashboard.py — DashboardWidget, WidgetType (counter/chart/table/map/timeline), WidgetConfig, DEFAULT_WIDGETS (5 pre-configured widgets) | Unit Tested (import verified) |
| Exported in models/__init__.py — DashboardWidget, WidgetConfig, WidgetType, DEFAULT_WIDGETS | Import verified |
| tests/models/test_analytics_dashboard.py — 9 tests covering roundtrip, defaults, invalid type fallback, default widgets | Written |

## 2026-03-14 — Wave 92: Unified Threat Assessment Engine

| Change | Verification |
|--------|-------------|
| intelligence/threat_model.py — ThreatModel unified assessment engine with time decay, per-target signal buffers, weighted composite scores, and ThreatLevel derivation | Unit Tested (29 tests) |
| ThreatSignal with TTL expiry, ThreatAssessment with sub-scores, score_to_threat_level helper | Unit Tested |
| intelligence/__init__.py updated with new exports | Import Tested |

## 2026-03-14 — Wave 91: Autonomous Decision Models

| Change | Verification |
|--------|-------------|
| models/autonomous.py — AutonomousDecision, AutonomousDecisionLog, AutonomousDecisionType, AutonomousTrigger, EdgeAlertRule, OverrideState | Unit Tested (15 tests) |
| 6 decision types (alert, classify, escalate, lockdown, evade, report), 9 trigger types, 4 override states | Unit Tested |
| Registered in models/__init__.py with __all__ exports | Consumer Tested |

## 2026-03-14 — Wave 89: Forensic Reconstruction Models

| Change | Verification |
|--------|-------------|
| models/forensics.py — ForensicReconstruction, IncidentReport, GeoBounds, TimeRange, EvidenceItem, TargetTimeline, SensorCoverage, IncidentFinding, IncidentRecommendation, IncidentClassification, ReconstructionStatus | Unit Tested (15 tests) |
| Registered in models/__init__.py with __all__ exports | Consumer Tested |

## 2026-03-14 — Wave 88: Collaboration Models

| Change | Verification |
|--------|-------------|
| models/collaboration.py — SharedWorkspace, WorkspaceEvent, OperatorAction, MapDrawing, OperatorChatMessage, DrawingType, ChatMessageType enums | Unit Tested (13 tests) |
| Registered in models/__init__.py with full __all__ exports | Unit Tested |

## 2026-03-14 — Wave 86: Correlation Evidence Models

| Change | Verification |
|--------|-------------|
| `models/correlation_evidence.py` — CorrelationEvidence model with EvidenceType enum (spatial, temporal, signal, visual, handoff, behavioral, manual), make_pair_id(), compute_composite_confidence(), builder helpers for spatial/visual/handoff evidence | Unit tested (16 tests) |
| Registered in `models/__init__.py` for clean import access | Consumer tested |

---

## 2026-03-14 — Wave 82: EventStore for Tactical Event Persistence

| Change | Verification |
|--------|-------------|
| `store/event_store.py` — EventStore with SQLite WAL, time-range queries, type/severity/target/source filtering, batch insert, cleanup, stats | Unit tested (26 tests) |
| `TacticalEvent` dataclass with event_id, timestamp, event_type, severity, source, target_id, operator, summary, data, position, site_id | Unit tested |
| `SEVERITY_LEVELS` tuple for ordered severity filtering (debug < info < warning < error < critical) | Unit tested |
| Exported from `tritium_lib.store` package: EventStore, TacticalEvent, SEVERITY_LEVELS | Consumer tested |

## 2026-03-14 — Wave 79: Acoustic Intelligence Models

### AudioFeatureVector, SoundSignature, SoundClassification, AcousticTrilateration (Unit Tested, 16 tests)
- New `models/acoustic_intelligence.py` — acoustic ML classification and TDoA localization models
- `AudioFeatureVector`: 13 MFCCs + spectral centroid/bandwidth/rolloff/flatness + ZCR + RMS + peak
- `SoundSignature`: reference profiles for sound classes with `matches_features()` heuristic matcher
- `SoundClassification`: ML model output with top-N predictions and model version
- `AcousticObserver`: sensor node observation for TDoA (lat, lon, arrival_time, amplitude)
- `AcousticTrilateration`: multi-node localization result with confidence scoring
- `acoustic_trilaterate()`: TDoA weighted centroid localization from 2+ observers
- All models exported from `tritium_lib.models.__init__`

---

## 2026-03-14 — Wave 77: Fleet Operations Models

### FleetCommand + ConfigTemplate + FleetAnalyticsSnapshot (Unit Tested, 13 tests)
- New `models/fleet_ops.py` — fleet coordination data contracts
- `FleetCommand`: command_type, target_group, payload, status tracking, ack/fail counts
- `FleetCommandType`: reboot, scan_burst, increase_rate, decrease_rate, ota_update, apply_template, set_group, identify, sleep
- `ConfigTemplate`: named config templates with scan intervals, report rates, power mode
- `ConfigTemplateName`: perimeter_high_security, indoor_normal, power_saver_mobile, custom
- `BUILTIN_TEMPLATES`: 3 pre-built templates for common deployment scenarios
- `FleetAnalyticsSnapshot`: fleet-wide analytics with uptime records, sighting rates, coverage, groups
- `DeviceUptimeRecord`, `SightingRateRecord`, `CoveragePoint` supporting models
- All models exported from `tritium_lib.models` as `FLEET_BUILTIN_TEMPLATES`, etc.

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
