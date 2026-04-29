# tritium_lib.fusion

Multi-sensor fusion engine -- combines BLE, WiFi, camera, acoustic, mesh, ADS-B, and RF motion into correlated target identities.

**Where you are:** `tritium-lib/src/tritium_lib/fusion/`

## How It Works

```mermaid
graph TD
    subgraph "Sensor Ingestion"
        BLE[ingest_ble] --> FE[FusionEngine]
        WIFI[ingest_wifi] --> FE
        CAM[ingest_camera] --> FE
        ACOU[ingest_acoustic] --> FE
        MESH[ingest_mesh] --> FE
        ADSB[ingest_adsb] --> FE
        RF[ingest_rf_motion] --> FE
    end

    FE --> TK[TargetTracker]
    FE --> HM[HeatmapEngine]
    FE --> NA[NetworkAnalyzer]
    FE --> SR[SensorRecord store]

    FE --> |run_correlation| TC[TargetCorrelator]
    TC --> DS[DossierStore]
    TC --> FM[FusionMetrics]

    FE --> |query| FT[FusedTarget]
    FE --> |query| FS[FusionSnapshot]
    FE --> |query| TD2[Target Dossier]

    subgraph "Event-driven mode"
        EB[EventBus] --> SP[SensorPipeline]
        SP --> FE
    end
```

`FusionEngine` composes `TargetTracker`, `TargetCorrelator`, `GeofenceEngine`, `HeatmapEngine`, `DossierStore`, `NetworkAnalyzer`, and `FusionMetrics` into a single ingest-then-query API. It does not reimplement these -- it orchestrates them.

`SensorPipeline` adds event-bus-driven operation: subscribe to `sensor.*` topics and auto-route to the engine.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Exports `FusionEngine`, `FusionSnapshot`, `FusedTarget`, `SensorRecord`, `SensorPipeline` |
| `engine.py` | `FusionEngine` -- unified ingest API for 7 sensor types, correlation control, zone management, query methods |
| `pipeline.py` | `SensorPipeline` -- event-bus bridge that auto-routes `sensor.*` topics to `FusionEngine.ingest_*()` |

## Key Types

- **`FusionEngine`** -- the orchestrator. Call `ingest_ble()`, `ingest_camera()`, etc., then `get_fused_targets()`.
- **`FusedTarget`** -- a `TrackedTarget` enriched with sensor records, dossier, zones, and correlations.
- **`FusionSnapshot`** -- point-in-time snapshot of all targets, dossiers, correlations, zones, and metrics.
- **`SensorPipeline`** -- subscribes to EventBus sensor topics and dispatches to the engine automatically.

## Usage

```python
from tritium_lib.fusion import FusionEngine

engine = FusionEngine(auto_correlate=True)
engine.ingest_ble({"mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
engine.ingest_camera({"class_name": "person", "confidence": 0.9, "center_x": 10.0, "center_y": 5.0})
targets = engine.get_fused_targets()  # returns list[FusedTarget]
```

**Parent:** [../README.md](../README.md)
