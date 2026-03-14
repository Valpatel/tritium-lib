# Tritium-Lib — Shared Foundation

Common code library for the entire Tritium ecosystem. Used by tritium-edge (firmware + fleet server), tritium-sc (command center), and any future components. Currently Python-only, but designed to grow into a polyglot library (Python, C++, JS/CSS/HTML) as reusable code is extracted from the other submodules.

**Guiding principle:** When code can be shared, pull it into tritium-lib. Test utilities, report generators, data models, theme assets, protocol codecs — if two submodules need it, it belongs here.

**Parent context:** See [../CLAUDE.md](../CLAUDE.md) for full system architecture and conventions.

## Git Conventions

- **No co-authors on commits** — never add "Co-Authored-By" lines
- Remote: `git@github.com:Valpatel/tritium-lib.git`
- Copyright: Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC / AGPL-3.0

## What This Does

| Package | What | Key Files |
|---------|------|-----------|
| `models` | 116+ Pydantic models (device, firmware, mesh, BLE, CoT, alerts, topology, diagnostics, sensor, transport) | `src/tritium_lib/models/` |
| `events` | Thread-safe and async pub/sub event bus | `src/tritium_lib/events/bus.py` |
| `mqtt` | MQTT topic hierarchy `tritium/{site}/{domain}/{device}/{type}` | `src/tritium_lib/mqtt/topics.py` |
| `auth` | JWT token creation/decoding, API key management | `src/tritium_lib/auth/jwt.py` |
| `store` | Persistent data stores (BLE sightings, targets) | `src/tritium_lib/store/` |
| `cot` | Cursor on Target XML codec for TAK/ATAK integration | `src/tritium_lib/cot/codec.py` |
| `config` | Pydantic base settings class for service configuration | `src/tritium_lib/config/` |
| `geo` | Coordinate transforms (local meters <-> lat/lng), camera projection, haversine | `src/tritium_lib/geo/__init__.py` |
| `notifications` | Notification model and thread-safe NotificationManager | `src/tritium_lib/notifications/__init__.py` |
| `graph` | KuzuDB embedded graph database for entity/relationship storage | `src/tritium_lib/graph/store.py` |
| `ontology` | Semantic type system — entity types, relationship types, schema validation | `src/tritium_lib/ontology/schema.py` |
| `classifier` | Multi-signal BLE/WiFi device type classification with fingerprint databases | `src/tritium_lib/classifier/device_classifier.py` |
| `data` | JSON lookup tables for BLE, WiFi, OUI fingerprinting (11 databases) | `src/tritium_lib/data/` |
| `web` | Cyberpunk HTML theme engine and dashboard components | `src/tritium_lib/web/` |
| `testing` | Visual regression checks and ESP32 device automation | `src/tritium_lib/testing/` |

## Directory Structure

```
tritium-lib/
├── src/tritium_lib/
│   ├── models/          # 116+ Pydantic models (THE canonical data contracts)
│   │   ├── device.py    # DeviceInfo, DeviceStatus, HeartbeatPayload
│   │   ├── firmware.py  # FirmwareVersion, OTARequest
│   │   ├── mesh.py      # MeshPeer, MeshMessage
│   │   ├── ble.py       # BLESighting, BLEDevice
│   │   ├── alert.py     # Alert, AlertLevel
│   │   ├── command.py   # DeviceCommand, CommandResponse
│   │   ├── cot.py       # CursorOnTarget models
│   │   ├── gis.py       # GeoPoint, MapTile
│   │   ├── topology.py  # NetworkTopology
│   │   ├── diagnostics.py # DiagnosticReport
│   │   ├── sensor.py    # SensorReading
│   │   ├── transport.py # TransportMessage
│   │   └── ...
│   ├── mqtt/
│   │   └── topics.py    # TritiumTopics — topic builder
│   ├── events/
│   │   └── bus.py       # EventBus, AsyncEventBus
│   ├── auth/
│   │   └── jwt.py       # JWT encode/decode
│   ├── store/
│   │   └── ble.py       # BLE sighting persistence
│   ├── cot/
│   │   └── codec.py     # CoT XML ↔ Pydantic
│   ├── config/
│   │   └── __init__.py  # Base settings
│   ├── graph/
│   │   └── store.py     # TritiumGraph — KuzuDB wrapper
│   ├── ontology/
│   │   ├── schema.py    # Entity/relationship type definitions
│   │   └── registry.py  # OntologyRegistry — runtime lookup
│   ├── classifier/
│   │   └── device_classifier.py  # Multi-signal BLE/WiFi classifier
│   ├── data/
│   │   ├── ble_fingerprints.json  # BLE device fingerprints
│   │   ├── ble_appearance_values.json  # GAP appearance codes
│   │   ├── ble_service_uuids.json     # Service UUID mapping
│   │   ├── oui_device_types.json      # OUI to device type
│   │   ├── wifi_ssid_patterns.json    # WiFi SSID classification
│   │   └── ...            # 11 JSON lookup databases
│   ├── web/
│   │   ├── theme.py     # Cyberpunk color palette
│   │   ├── components.py # Reusable HTML components
│   │   ├── templates.py # Page templates
│   │   └── dashboard.py # Dashboard generator
│   └── testing/
│       ├── visual.py    # VisualCheck (OpenCV validation)
│       ├── flicker.py   # Flicker detection
│       ├── device.py    # ESP32 device automation
│       └── runner.py    # Test runner utilities
├── tests/               # pytest tests
└── pyproject.toml       # Package config
```

## How To Work Here

```bash
# Install (editable)
pip install -e .              # Core only
pip install -e ".[full]"      # All optional deps

# Test
pytest tests/

# Quick check: import everything
python -c "from tritium_lib.models import *; print('OK')"
```

## Rules

1. **Models are the API contract.** If you change a model here, you must update both tritium-edge and tritium-sc consumers. Check imports with `grep -r "from tritium_lib" ../tritium-edge/ ../tritium-sc/`.
2. **MQTT topics are defined here, not in consumers.** Use `TritiumTopics` — never hardcode topic strings.
3. **No framework dependencies (Python).** Python packages must stay lightweight — Pydantic, PyJWT, and optionally paho-mqtt. No FastAPI, no SQLAlchemy.
4. **Type hints everywhere.** All public functions must have complete type annotations.
5. **Test after every change.** `pytest tests/` must pass before committing.
6. **Extract, don't duplicate.** If you find similar code in tritium-edge and tritium-sc, pull it here. Test utilities, report generators, protocol helpers, theme assets — if two submodules need it, it belongs in tritium-lib.

## Coding Conventions

- Python 3.12+, PEP 8
- C++17 (when C++ code is added)
- Vanilla JS, no frameworks (when web code is added)
- Cyberpunk aesthetic for shared web/CSS: cyan #00f0ff, magenta #ff2a6d, green #05ffa1, yellow #fcee0a
- 4-space indentation everywhere
- Type hints on all public Python functions
- Pydantic v2 models (use `model_config` not `class Config`)

## Future Structure

As C++ and web code gets extracted from the other submodules, the directory structure will grow:

```
tritium-lib/
├── src/tritium_lib/     # Python packages (current)
├── cpp/                 # Shared C++ headers and libraries (future)
├── web/                 # Shared JS/CSS/HTML assets (future)
├── tests/               # Python tests (current)
└── pyproject.toml       # Python package config
```
