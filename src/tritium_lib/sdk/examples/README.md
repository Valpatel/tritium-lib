# tritium_lib.sdk.examples

Reference addons that exercise the full Tritium SDK lifecycle. Copy one as the
starting template when building a new sensor integration.

**Where you are:** `tritium-lib/src/tritium_lib/sdk/examples/`

**License:** Apache-2.0 (matches `sdk/`) -- private/proprietary addons can copy
freely.

## Examples

| Example | What it demonstrates |
|---------|----------------------|
| [`weather_station/`](weather_station/README.md) | The full lifecycle: `AddonBase` subclass, `AddonContext` DI, publishing events on the `AddonEventBus`, registering targets with the `TargetTracker`, exposing a GeoJSON map layer, plus a standalone headless `BaseRunner` (`runner.py`) and a `tritium_addon.toml` manifest. |

## Run it

```python
# In-process (inside a Command Center or a test harness)
from tritium_lib.sdk.examples.weather_station import WeatherStationAddon
addon = WeatherStationAddon()
await addon.register(context=my_context)
readings = await addon.gather()
```

```bash
# Standalone headless (Raspberry Pi, etc.) -- publishes to a Command Center over MQTT
python -m tritium_lib.sdk.examples.weather_station.runner
python -m tritium_lib.sdk.examples.weather_station.runner --station wx-roof-01 --interval 30
```

## How it fits

These live *inside* lib so the SDK ships with a runnable, tested reference. The
canonical addon-authoring walkthrough (manifest -> loader -> context DI ->
target emission -> panels/layers -> headless runner, each claim file:line-cited)
is `tritium-addons/DEVELOPER-GUIDE.md`; these examples are the working code that
guide points at.

**Parent:** [../README.md](../README.md)
