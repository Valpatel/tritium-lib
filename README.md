# tritium-lib

Shared Python library for models, events, auth, MQTT, and testing across
the [Tritium](https://github.com/Valpatel/tritium) ecosystem.

## What's inside

| Package | Description |
|---------|-------------|
| `models` | 130+ Pydantic models covering devices, firmware, mesh, BLE, CoT, alerts, topology, diagnostics, camera, Meshtastic, and more |
| `events` | Thread-safe and async pub/sub event bus (`EventBus`, `AsyncEventBus`) |
| `mqtt` | MQTT topic hierarchy (`tritium/{site}/{domain}/{device}/{type}`) and parsers |
| `auth` | JWT token creation/decoding and API key management |
| `store` | Persistent data stores (BLE sightings, targets, node positions) |
| `cot` | Cursor on Target XML codec for TAK/ATAK integration |
| `config` | Pydantic base settings class for service configuration |
| `web` | Cyberpunk HTML theme engine and dashboard components |
| `testing` | Visual regression checks and ESP32 device automation |
| `sim_engine` | **Tactical simulation engine** — 110 files, 57K lines: combat AI, weapons, vehicles, naval, air, terrain, weather, crowds, destruction, medical, logistics, intel, scoring, campaigns, multiplayer, replay, economy, artillery, cyber warfare. Three.js-compatible. |

## Sim Engine Demo

Run a complete tactical simulation with 3D visualization:

```bash
./sim-demo.sh                    # Start demo (opens http://localhost:9090)
./sim-demo.sh --list             # See all presets
./sim-demo.sh --perf             # Run performance benchmark
./sim-demo.sh --coverage         # Run module coverage report
```

Controls: `SPACE`=riot `N`=night `R`=rain `F`=fog `D`=debug `I`=intel `M`=minimap `K`=record `L`=playback `A`=airstrike `S`=sound `2`=split-view `P`=skip-phase `C`=chase-cam `Click`=inspect `ESC`=deselect

Opens `http://localhost:8888` with a Three.js 3D view of the battle.

Presets: `urban_combat`, `open_field`, `riot_response`, `convoy_ambush`, `drone_strike`

## Install

```bash
pip install -e .              # Core
pip install -e ".[mqtt]"      # With MQTT support
pip install -e ".[full]"      # Everything
```

## Quick examples

```python
from tritium_lib.mqtt import TritiumTopics

topics = TritiumTopics(site_id="home")
topics.edge_heartbeat("esp32-001")
# → "tritium/home/edge/esp32-001/heartbeat"
```

```python
from tritium_lib.events import EventBus

bus = EventBus()
bus.subscribe("device.#", lambda e: print(e.topic, e.data))
bus.publish("device.heartbeat", {"id": "esp32-001"})
```

```python
from tritium_lib.testing import VisualCheck

check = VisualCheck(screenshot_path="screenshot.png")
issues = check.run()
for issue in issues:
    print(issue.severity, issue.description)
```

## Used by

- **[tritium-edge](https://github.com/Valpatel/tritium-edge)** — IoT fleet
  server for ESP32-S3 boards; imports models, MQTT topics, auth, and BLE stores.
- **[tritium-sc](https://github.com/Valpatel/tritium-sc)** — Command center
  with plugin system; imports models, events, auth, and the web theme engine.

## License

AGPL-3.0 — Copyright 2026 Valpatel Software LLC
