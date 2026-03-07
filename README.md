# tritium-lib

Shared library for the Tritium ecosystem. Provides reusable components across
[tritium-sc](https://github.com/Valpatel/tritium-sc) and
[tritium-edge](https://github.com/Valpatel/tritium-edge).

## Modules

| Module | Purpose |
|--------|---------|
| `tritium_lib.models` | Shared data models: Device, Command, Firmware, SensorReading |
| `tritium_lib.mqtt` | MQTT topic conventions and builders |
| `tritium_lib.auth` | JWT token create/decode utilities |
| `tritium_lib.events` | Thread-safe pub/sub event bus |
| `tritium_lib.config` | Pydantic settings base class |
| `tritium_lib.cot` | CoT (Cursor on Target) XML codec for TAK integration |

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
```

## License

AGPL-3.0 — Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC
