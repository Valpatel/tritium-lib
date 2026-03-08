# tritium-lib

Shared library for the Tritium ecosystem. Provides reusable components across
[tritium-sc](https://github.com/Valpatel/tritium-sc) and
[tritium-edge](https://github.com/Valpatel/tritium-edge).

## Modules

| Module | Purpose |
|--------|---------|
| `tritium_lib.models` | Shared data models (see below) |
| `tritium_lib.mqtt` | MQTT topic conventions and builders |
| `tritium_lib.auth` | JWT token create/decode utilities |
| `tritium_lib.events` | Thread-safe pub/sub event bus |
| `tritium_lib.config` | Pydantic settings base class |
| `tritium_lib.cot` | CoT (Cursor on Target) XML codec for TAK integration |
| `tritium_lib.web` | Cyberpunk HTML/CSS theme and dashboard components |

### Models

| Model Module | Key Types | Purpose |
|---|---|---|
| `models.device` | Device, DeviceHeartbeat, DeviceCapabilities | Core device identity and telemetry |
| `models.command` | Command, CommandType, CommandStatus | Remote command dispatch |
| `models.firmware` | FirmwareMeta, OTAJob, OTAStatus | OTA firmware management |
| `models.sensor` | SensorReading | Generic sensor data |
| `models.fleet` | FleetNode, FleetStatus, NodeEvent | Fleet status and health scoring |
| `models.ble` | BleDevice, BlePresence, BlePresenceMap | BLE presence detection and triangulation |
| `models.topology` | FleetTopology, NetworkLink, ConnectivityReport | Multi-transport network graph with BFS reachability, connected components, and average path length |
| `models.diagnostics` | HealthSnapshot, Anomaly, HeapTrend, DiagLogEntry | Health monitoring, heap trend analysis (`analyze_heap_trends`), fleet anomaly correlation (`detect_fleet_anomalies`) |
| `models.mesh` | MeshNode, MeshRoute, MeshTopology, MeshMessage | ESP-NOW multi-hop mesh networking |
| `models.gis` | TileCoord, TileBounds, MapRegion, OfflineRegion | Offline GIS tile management |
| `models.seed` | SeedManifest, SeedPackage, SeedTransfer | Self-replicating firmware distribution |
| `models.acoustic_modem` | AcousticFrame, AcousticConfig | FSK data-over-audio channel |
| `models.cot` | CotEvent, CotPoint, CotDetail | MIL-STD-2045 Cursor-on-Target XML |
| `models.config` | DeviceConfig, ConfigDrift, FleetConfigStatus | Config sync and drift detection |
| `models.provision` | ProvisionData, ProvisionRecord, FleetProvisionStatus | Device provisioning pipeline |
| `models.alert` | Alert, AlertHistory, WebhookConfig | Alert routing and webhook delivery |

## Install

```bash
pip install -e .                  # Core
pip install -e ".[mqtt]"          # With MQTT support
pip install -e ".[full]"          # Everything
```

## Usage

```python
from tritium_lib.models import Device, DeviceHeartbeat, Command
from tritium_lib.mqtt import TritiumTopics
from tritium_lib.auth import create_token, decode_token, TokenType
from tritium_lib.events import EventBus
from tritium_lib.cot import device_to_cot

# MQTT topics
topics = TritiumTopics(site_id="home")
print(topics.edge_heartbeat("my-device"))
# → tritium/home/edge/my-device/heartbeat

# CoT XML for TAK
xml = device_to_cot("esp32-001", lat=37.7159, lng=-121.896,
                     capabilities=["camera", "imu"])
# → <event uid="tritium-edge-esp32-001" type="a-f-G-E-S-C" ...>

# Event bus
bus = EventBus()
bus.subscribe("device.#", lambda e: print(e.topic, e.data))
bus.publish("device.heartbeat", {"id": "esp32-001"})

# Fleet topology analysis
from tritium_lib.models import NetworkLink, build_topology, analyze_connectivity
links = [
    NetworkLink(source_id="node-1", target_id="node-2", transport="wifi"),
    NetworkLink(source_id="node-2", target_id="node-3", transport="espnow"),
]
topo = build_topology(links)
print(topo.reachable("node-1", "node-3"))  # True (multi-hop)
report = analyze_connectivity(topo)
print(report.num_components, report.transports_used)  # 1, ['espnow', 'wifi']

# Heap trend / memory leak detection
from tritium_lib.models import analyze_heap_trends
snapshots = [
    {"device_id": "esp32-001", "free_heap": 280000, "uptime_s": 0},
    {"device_id": "esp32-001", "free_heap": 260000, "uptime_s": 3600},
]
trends = analyze_heap_trends(snapshots)
print(trends[0].delta_per_hour, trends[0].leak_suspected)  # -20000.0, True
```

## Part of Tritium

[**Tritium**](https://github.com/Valpatel/tritium) is a distributed cybernetic
operating system. It turns heterogeneous hardware — microcontrollers, single-board
computers, robots, cameras, servers, radios — into a unified mesh that observes,
thinks, and acts. Every device is a node. The network is the computer.

tritium-lib is the **spine**: the shared contract that lets every node in the
mesh speak the same language. It works alongside
[tritium-sc](https://github.com/Valpatel/tritium-sc) (the brain — command,
vision, models) and [tritium-edge](https://github.com/Valpatel/tritium-edge)
(the nervous system — fleet management, OTA, heartbeat).

## License

AGPL-3.0 — Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC
