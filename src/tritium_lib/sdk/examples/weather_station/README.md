# Weather Station — Example Tritium Addon

A complete reference implementation of a Tritium sensor addon. Use this as a
template when building your own addon.

## What This Demonstrates

| SDK Feature | Where |
|-------------|-------|
| `SensorAddon` subclass with `gather()` | `__init__.py` |
| `AddonInfo` metadata | `__init__.py` |
| `AddonEventBus` publishing | `gather()` method |
| `ITargetTracker` integration | `gather()` method |
| `IEventBus` integration | `gather()` method |
| `AddonContext` dependency injection | `register()` method |
| `AddonConfig` runtime configuration | `register()` method |
| `AddonGeoLayer` map layers | `get_geojson_layers()` |
| Panels, layers, shortcuts | `get_panels()`, etc. |
| Health checks | `health_check()` |
| `BaseRunner` headless mode | `runner.py` |
| TOML manifest | `tritium_addon.toml` |
| Alert threshold checking | `_check_alerts()` |
| GeoJSON export | `to_geojson()` |

## Quick Start

```python
from tritium_lib.sdk.examples.weather_station import WeatherStationAddon
from tritium_lib.sdk import AddonContext, AddonEventBus

# Create addon
addon = WeatherStationAddon(station_id="my-station", lat=40.0, lng=-105.0)

# Create context with services
context = AddonContext(
    event_bus=my_event_bus,
    target_tracker=my_tracker,
    addon_event_bus=AddonEventBus(),
)

# Register (wires up services)
await addon.register(context=context)

# Gather weather data
targets = await addon.gather()
print(targets[0]["conditions"])  # "clear", "light_rain", etc.

# Query
print(addon.latest_reading.temperature_c)
print(addon.get_summary())
print(addon.to_geojson())

# Cleanup
await addon.unregister()
```

## Headless Runner (Raspberry Pi / Edge Device)

```bash
# Default: simulated station, publishes to localhost MQTT
python -m tritium_lib.sdk.examples.weather_station.runner

# Custom station on remote MQTT broker
python -m tritium_lib.sdk.examples.weather_station.runner \
    --station wx-roof-01 \
    --interval 30 \
    --lat 40.0150 --lng -105.2705 \
    --mqtt-host 192.168.1.100 \
    --site field-office
```

## Creating Your Own Addon From This Template

1. Copy this directory to your addon location
2. Rename the class and update `AddonInfo`
3. Replace `WeatherSimulator` with your real hardware/API reader
4. Update `gather()` to produce your target dicts
5. Update `tritium_addon.toml` with your panels, layers, permissions
6. Update the runner for your device discovery and polling
7. Write tests (see `tests/sdk/test_weather_station.py`)

## File Structure

```
weather_station/
    __init__.py            WeatherStationAddon, WeatherReading, WeatherSimulator
    tritium_addon.toml     Addon manifest (metadata, permissions, UI, config)
    runner.py              Standalone MQTT runner (BaseRunner subclass)
    README.md              This file
```
