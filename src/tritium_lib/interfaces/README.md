# tritium_lib.interfaces

Parent: [tritium_lib](../README.md)

Abstract base classes for SC plugins. Plugins implement these interfaces to integrate with the plugin system.

## Key Files

| File | Purpose |
|------|---------|
| `sensor_plugin.py` | Base interface for all sensor plugins |
| `camera_plugin.py` | Camera feed plugin interface |
| `radar_plugin.py` | Radar plugin interface |
| `sdr_plugin.py` | SDR plugin interface |

## Related

- SC plugin loader: `tritium-sc/src/engine/plugins/`
- Plugin directories: `tritium-sc/plugins/`
