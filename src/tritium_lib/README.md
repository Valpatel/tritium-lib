# tritium_lib — Package Reference

**Where you are:** `tritium-lib/src/tritium_lib/` — the shared Python packages for the Tritium ecosystem.

**Parent:** [../../CLAUDE.md](../../CLAUDE.md) | [../../../CLAUDE.md](../../../CLAUDE.md) (tritium root)

## Packages

| Package | What | Key Exports |
|---------|------|-------------|
| `models/` | 116+ Pydantic data models | `Device`, `BLESighting`, `MeshNode`, `Alert`, `Command`, `FleetNode`, ... |
| `mqtt/` | MQTT topic hierarchy | `TritiumTopics`, `ParsedTopic`, `parse_topic()` |
| `events/` | Pub/sub event bus | `EventBus`, `AsyncEventBus`, `Event` |
| `auth/` | JWT and API keys | `create_token()`, `decode_token()`, `generate_api_key()` |
| `config/` | Base settings | `TritiumBaseSettings` (Pydantic BaseSettings) |
| `store/` | Data persistence | `BleStore` (SQLite-backed) |
| `cot/` | Cursor on Target codec | `device_to_cot()`, `sensor_to_cot()`, `parse_cot()` |
| `web/` | HTML theme engine | `TritiumTheme`, `DashboardPage`, UI components |
| `testing/` | Visual regression | `VisualCheck`, `DeviceAPI`, `FlickerAnalyzer` |

## MQTT Topic Structure

Topics follow the pattern: `tritium/{site}/{domain}/{device_id}/{data_type}`

```python
from tritium_lib.mqtt import TritiumTopics

topics = TritiumTopics(site_id="home")
topics.edge_heartbeat("esp32-001")     # → "tritium/home/edge/esp32-001/heartbeat"
topics.camera_detections("cam-01")     # → "tritium/home/cameras/cam-01/detections"
topics.robot_command("rover-01")       # → "tritium/home/robots/rover-01/command"
topics.all_edge()                      # → "tritium/home/edge/#"
```

## Event Bus

```python
from tritium_lib.events import EventBus, AsyncEventBus

bus = EventBus()
bus.subscribe("device.#", lambda e: print(e.topic, e.data))
bus.publish("device.heartbeat", {"id": "esp32-001"})
```

Supports wildcard patterns: `.*` (single-level), `.#` (multi-level).

## Model Categories

| Category | Models | Files |
|----------|--------|-------|
| Device | DeviceInfo, DeviceHeartbeat, DeviceCapabilities, DeviceGroup | `device.py` |
| BLE | BleDevice, BleSighting, BlePresence | `ble.py` |
| Mesh | MeshNode, MeshRoute, MeshTopology, MeshMessage | `mesh.py` |
| Fleet | FleetNode, FleetStatus, NodeEvent | `fleet.py` |
| Commands | Command, CommandType, CommandStatus | `command.py` |
| Alerts | Alert, AlertSeverity, AlertDelivery | `alert.py` |
| GIS | TileCoord, MapLayer, OfflineRegion | `gis.py` |
| Diagnostics | CrashInfo, DiagLogEntry, HeapTrend | `diagnostics.py` |
| CoT/TAK | CotEvent, CotPoint, CotDetail | `cot.py` |
| Transport | TransportType, TransportMetrics | `transport.py` |
| Topology | NetworkLink, FleetTopology | `topology.py` |
| Firmware | FirmwareMeta, OTAJob, OTAStatus | `firmware.py` |
| Sensor | SensorReading | `sensor.py` |
| Timeseries | TimeSeries, FleetTimeSeries, PagedResult | `timeseries.py` |
| Trilateration | AnchorPoint, PositionEstimate, rssi_to_distance() | `trilateration.py` |
| Acoustic | AcousticFrame, AcousticConfig, ModulationType | `acoustic_modem.py` |
| Provision | ProvisionData, ProvisionRecord | `provision.py` |
| Correlation | CorrelationEvent, classify_correlation_severity() | `correlation.py` |
| Seed | SeedFile, SeedManifest, SeedPackage | `seed.py` |
| Config | ConfigDrift, DeviceConfig | `config.py` |

## Web Theme

The `web/` package generates cyberpunk-themed HTML dashboards:

| Color | Hex | Use |
|-------|-----|-----|
| Background | `#0a0a0a` | Page background |
| Accent | `#00ffd0` | Primary accent (cyan/green) |
| Danger | `#ff3366` | Alerts, errors |
| Warning | `#ffaa00` | Caution states |
| Text | `#c0c0c0` | Body text |

## Testing

```bash
pytest tests/          # All tests
pytest tests/ -k mqtt  # Just MQTT tests
```

## Consumers

- **tritium-edge** (fleet server) — imports models, MQTT topics, auth, BLE stores
- **tritium-sc** (command center) — imports models, events, auth, web theme
