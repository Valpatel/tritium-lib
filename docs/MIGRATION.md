# SC to Lib Migration Guide

## Overview

Tritium-lib (`tritium_lib`) is the shared library used by all Tritium components. Core logic that was originally written inside tritium-sc's `engine/` package has been extracted into tritium-lib so it can be reused by tritium-edge, tritium-addons, and standalone runners.

**What was moved:** Target tracking, correlation, geo transforms, event bus, inference fleet, simulation engine, combat system, actions parser, intelligence/ML, BLE classification, geofencing, heatmaps, movement analysis, threat scoring, and 90+ data models.

**What stays in SC:** Plugin system, Amy commander, perception/vision, scenarios, nodes (camera/mic hardware), audio pipeline, synthetic media, layers (import/export), and all FastAPI routers.

## Module Mapping

### Tracking (`engine.tactical.*` -> `tritium_lib.tracking.*`)

| SC Path | Lib Path |
|---------|----------|
| `engine.tactical.target_tracker` | `tritium_lib.tracking.target_tracker` |
| `engine.tactical.correlator` | `tritium_lib.tracking.correlator` |
| `engine.tactical.correlation_strategies` | `tritium_lib.tracking.correlation_strategies` |
| `engine.tactical.ble_classifier` | `tritium_lib.tracking.ble_classifier` |
| `engine.tactical.convoy_detector` | `tritium_lib.tracking.convoy_detector` |
| `engine.tactical.dossier` | `tritium_lib.tracking.dossier` |
| `engine.tactical.dwell_tracker` | `tritium_lib.tracking.dwell_tracker` |
| `engine.tactical.geofence` | `tritium_lib.tracking.geofence` |
| `engine.tactical.heatmap` | `tritium_lib.tracking.heatmap` |
| `engine.tactical.kalman_predictor` | `tritium_lib.tracking.kalman_predictor` |
| `engine.tactical.movement_patterns` | `tritium_lib.tracking.movement_patterns` |
| `engine.tactical.target_history` | `tritium_lib.tracking.target_history` |
| `engine.tactical.target_prediction` | `tritium_lib.tracking.target_prediction` |
| `engine.tactical.target_reappearance` | `tritium_lib.tracking.target_reappearance` |
| `engine.tactical.threat_scoring` | `tritium_lib.tracking.threat_scoring` |
| `engine.tactical.trilateration` | `tritium_lib.tracking.trilateration` |
| `engine.tactical.vehicle_tracker` | `tritium_lib.tracking.vehicle_tracker` |

### Geo (`engine.tactical.geo` -> `tritium_lib.geo`)

| SC Path | Lib Path |
|---------|----------|
| `engine.tactical.geo` | `tritium_lib.geo` |

Key exports: `GeoReference`, `CameraCalibration`, `init_reference`, `get_reference`, `local_to_latlng`, `latlng_to_local`, `haversine_distance`

### Events (`engine.comms.event_bus` -> `tritium_lib.events.bus`)

| SC Path | Lib Path | Notes |
|---------|----------|-------|
| `engine.comms.event_bus.EventBus` | `tritium_lib.events.bus.QueueEventBus` | SC re-exports as `EventBus` |

### Comms (`engine.comms.speaker` -> `tritium_lib.comms.speaker`)

| SC Path | Lib Path |
|---------|----------|
| `engine.comms.speaker` | `tritium_lib.comms.speaker` |

### Inference (`engine.inference.*` -> `tritium_lib.inference.*`)

| SC Path | Lib Path |
|---------|----------|
| `engine.inference.fleet` | `tritium_lib.inference.fleet` |
| `engine.inference.model_router` | `tritium_lib.inference.model_router` |

### Actions (`engine.actions.*` -> `tritium_lib.actions.*`)

| SC Path | Lib Path |
|---------|----------|
| `engine.actions.lua_motor` | `tritium_lib.actions.lua_parser` |
| `engine.actions.formation_actions` | `tritium_lib.actions.formation` |

### Simulation Engine (`engine.simulation.*` -> `tritium_lib.sim_engine.*`)

| SC Path | Lib Path |
|---------|----------|
| `engine.simulation.target` (SimulationTarget) | `tritium_lib.sim_engine.core.entity` |
| `engine.simulation.behaviors` (UnitBehaviors) | `tritium_lib.sim_engine.behavior.behaviors` |
| `engine.simulation.combat` (CombatSystem) | `tritium_lib.sim_engine.combat.combat` |
| `engine.simulation.game_mode` (GameMode) | `tritium_lib.sim_engine.game.game_mode` |
| `engine.simulation.ambient` (AmbientSpawner) | `tritium_lib.sim_engine.game.ambient` |
| `engine.simulation.stats` | `tritium_lib.sim_engine.game.stats` |
| `engine.simulation.difficulty` | `tritium_lib.sim_engine.game.difficulty` |
| `engine.simulation.crowd_density` | `tritium_lib.sim_engine.game.crowd_density` |
| `engine.simulation.morale` | `tritium_lib.sim_engine.game.morale` |

Additional sim_engine modules with no SC equivalent (lib-native):

- `tritium_lib.sim_engine.core.movement` -- MovementController
- `tritium_lib.sim_engine.core.spatial` -- SpatialGrid
- `tritium_lib.sim_engine.core.state_machine` -- State, StateMachine, Transition
- `tritium_lib.sim_engine.core.inventory` -- InventoryItem, UnitInventory, ITEM_CATALOG
- `tritium_lib.sim_engine.combat.squads` -- Squad, SquadManager
- `tritium_lib.sim_engine.combat.weapons` -- Weapon, WeaponSystem
- `tritium_lib.sim_engine.world.cover` -- CoverObject, CoverSystem
- `tritium_lib.sim_engine.world.vision` -- VisionSystem, SightingReport
- `tritium_lib.sim_engine.world.sensors` -- SensorDevice, SensorSimulator
- `tritium_lib.sim_engine.world.pathfinding` -- plan_path
- `tritium_lib.sim_engine.world.grid_pathfinder` -- grid_find_path, MovementProfile
- `tritium_lib.sim_engine.behavior.unit_states` -- create_fsm_for_type, create_turret_fsm, etc.
- `tritium_lib.sim_engine.behavior.unit_missions` -- UnitMissionSystem
- `tritium_lib.sim_engine.behavior.npc` -- NPCManager

### Intelligence (`engine.intelligence.*` -> `tritium_lib.intelligence.*`)

| SC Path | Lib Path |
|---------|----------|
| `engine.intelligence.base_learner` | `tritium_lib.intelligence.base_learner` |
| `engine.intelligence.anomaly_baseline` | `tritium_lib.intelligence.anomaly` |
| `engine.intelligence.feature_aggregator` | `tritium_lib.intelligence.feature_engineering` |
| `engine.intelligence.correlation_learner` | `tritium_lib.intelligence.scorer` |
| `engine.intelligence.reid_store` | `tritium_lib.store.reid` |

### Store (`engine.tactical.event_store` -> `tritium_lib.store.*`)

| SC Path | Lib Path |
|---------|----------|
| `engine.tactical.event_store` | `tritium_lib.store.event_store` |

### Classifier (`engine.tactical.ble_classifier` -> `tritium_lib.classifier`)

| SC Path | Lib Path |
|---------|----------|
| Device classifier logic | `tritium_lib.classifier.device_classifier` |

## The Shim Pattern

SC engine files that have been migrated to lib are replaced with **shim modules** -- thin files that re-export everything from the lib equivalent. This keeps all existing `from engine.X import Y` statements working.

### Pattern 1: Star re-export (simple)

```python
# engine/comms/speaker.py
"""Re-export from tritium-lib. SC shim for backward compatibility."""
from tritium_lib.comms.speaker import *  # noqa: F401,F403
```

### Pattern 2: Star + explicit names (common)

```python
# engine/tactical/ble_classifier.py
"""Shim -- canonical implementation lives in tritium_lib.tracking.ble_classifier."""
from tritium_lib.tracking.ble_classifier import *  # noqa: F401,F403
from tritium_lib.tracking.ble_classifier import (  # noqa: F401
    BLEClassification,
    BLEClassifier,
    CLASSIFICATION_LEVELS,
    DEFAULT_SUSPICIOUS_RSSI,
)
```

The explicit re-imports after the star import exist because `*` only exports names listed in `__all__` (or names without a leading underscore). Constants and private helpers needed by tests require the second import block.

### Pattern 3: Alias re-export (EventBus)

```python
# engine/comms/event_bus.py
"""EventBus shim -- delegates to tritium-lib's QueueEventBus."""
from tritium_lib.events.bus import QueueEventBus as EventBus
__all__ = ["EventBus"]
```

SC code uses `EventBus` everywhere. Lib's canonical name is `QueueEventBus`. The shim aliases it.

### Pattern 4: Fallback with local implementation (geo)

```python
# engine/tactical/geo.py
_USE_LIB = False
try:
    from tritium_lib.geo import (
        GeoReference, init_reference, get_reference, ...
    )
    _USE_LIB = True
except ImportError:
    pass

if not _USE_LIB:
    # Full local fallback implementation
    ...
```

This is used only for `geo.py` because SC must be able to start even without tritium-lib installed (development convenience). Other shims assume tritium-lib is present.

### Pattern 5: Named re-export with `__all__` (sim engine)

```python
# engine/simulation/game_mode.py
from tritium_lib.sim_engine.game.game_mode import (
    GameMode,
    InfiniteWaveMode,
    InstigatorDetector,
    WaveConfig,
    WAVE_CONFIGS,
    _SPAWN_STAGGER,
    _WAVE_ADVANCE_DELAY,
    _COUNTDOWN_DURATION,
    _STALEMATE_TIMEOUT,
)
__all__ = ["GameMode", "InfiniteWaveMode", "InstigatorDetector", "WaveConfig", "WAVE_CONFIGS"]
```

Used for simulation modules where tests need private constants (`_SPAWN_STAGGER`, etc.) but public API is controlled via `__all__`.

## Step-by-Step: Converting an Import

### 1. Find the lib equivalent

Check the mapping table above, or search:

```bash
grep -r "class MyClass" tritium-lib/src/tritium_lib/ --include="*.py"
```

### 2. Update the import

Before:
```python
from engine.tactical.target_tracker import TargetTracker, TrackedTarget
```

After:
```python
from tritium_lib.tracking.target_tracker import TargetTracker, TrackedTarget
```

### 3. Run tests

```bash
cd tritium-sc && .venv/bin/python3 -m pytest tests/path/to/relevant_test.py -v
```

### 4. Check if the shim can be removed

```bash
# Count remaining importers of the SC shim path
grep -r "from engine.tactical.target_tracker" tritium-sc/src/ --include="*.py" | wc -l
```

If zero, the shim file can be deleted. If non-zero, leave it -- other SC modules still depend on the old path.

## Common Gotchas

### EventBus naming

SC uses `EventBus`. Lib exports `QueueEventBus` (synchronous, thread-safe) and `AsyncEventBus` (asyncio). The shim aliases `QueueEventBus as EventBus`. If you import directly from lib, use the correct name:

```python
# SC-style (via shim):
from engine.comms.event_bus import EventBus

# Direct lib import:
from tritium_lib.events.bus import QueueEventBus  # same class

# If you need async:
from tritium_lib.events.bus import AsyncEventBus
```

Both use `publish(topic, data)` and `subscribe(topic, callback)`.

### TrackedTarget.to_dict() geo converter

`TrackedTarget.to_dict()` accepts an optional `converter` parameter for coordinate transforms. SC passes the geo module's `local_to_latlng`:

```python
from tritium_lib.geo import local_to_latlng
target.to_dict(converter=local_to_latlng)
```

Without the converter, lat/lng fields will be zero.

### JS file paths

SC serves lib JS files at `/static/lib/`. When importing in frontend JS:

```javascript
// Before (SC-local):
import { IDMModel } from '../sim/idm.js';

// After (from lib):
import { IDMModel } from '/static/lib/js/sim/idm.js';
```

### sim_engine sub-package structure

Lib reorganized the flat `engine.simulation.*` into a nested hierarchy:

- `sim_engine.core.*` -- entity, movement, spatial, state machine, inventory
- `sim_engine.behavior.*` -- unit behaviors, NPC, FSM states, missions
- `sim_engine.combat.*` -- combat system, squads, weapons
- `sim_engine.game.*` -- game mode, ambient, stats, difficulty, morale, crowd density
- `sim_engine.world.*` -- cover, vision, sensors, pathfinding

### Intelligence feature engineering

SC's `engine.intelligence.feature_aggregator` maps to `tritium_lib.intelligence.feature_engineering`. The function names changed:

```python
# Lib canonical names:
from tritium_lib.intelligence.feature_engineering import (
    build_extended_features,
    co_movement_score,
    device_type_match,
    source_diversity,
    time_similarity,
)
```

### Models are in lib, not tracked in SC

All shared data models (`Device`, `FleetNode`, `BleDevice`, `MeshNode`, etc.) live in `tritium_lib.models.*`. SC's `src/app/models.py` contains only SQLAlchemy ORM models for the database, not data transfer objects.

## Verification Commands

After any import change:

```bash
# Quick syntax check
cd tritium-sc && python3 -c "import engine.tactical.target_tracker"

# Run targeted tests
cd tritium-sc && .venv/bin/python3 -m pytest tests/engine/tactical/ -v --timeout=30

# Run lib tests for the equivalent module
cd tritium-lib && pytest tests/tracking/ -v

# Full fast check
cd tritium-sc && ./test.sh fast
```

## Files That Are NOT Shims

These SC `engine/` files contain SC-specific logic and have no lib equivalent:

- `engine/simulation/engine.py` -- SimulationEngine (10Hz tick loop, SC-specific)
- `engine/simulation/behavior/*` -- SC-specific behavior coordinator
- `engine/simulation/npc_intelligence/*` -- LLM-powered NPC AI
- `engine/comms/mqtt_bridge.py` -- SC MQTT bridge
- `engine/comms/listener.py` -- Audio VAD + recording
- `engine/comms/tak_bridge.py` -- TAK server bridge
- `engine/perception/*` -- Vision, extraction, prompts
- `engine/nodes/*` -- Camera, mic, robot hardware nodes
- `engine/plugins/*` -- Plugin system (manager, base, data provider)
- `engine/scenarios/*` -- Behavioral test framework
- `engine/layers/*` -- GIS import/export (GPX, KML, GeoJSON, CoT)
- `engine/synthetic/*` -- Demo mode, video generation
- `engine/units/*` -- Unit type registry (16 types)
- `engine/audio/*` -- Sound effects, acoustic classifier
- `engine/tactical/escalation.py` -- ThreatClassifier + AutoDispatcher
- `engine/tactical/geo_protocols.py` -- Geo protocol interfaces
- `engine/tactical/street_graph.py` -- OSM road graph + A*
- `engine/tactical/obstacles.py` -- Building obstacles
