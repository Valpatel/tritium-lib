# Geo Library

**Where you are:** `tritium-lib/src/tritium_lib/geo/`

**Parent:** [../](../) | [../../../CLAUDE.md](../../../CLAUDE.md)

## What This Is

Coordinate transform library shared between tritium-sc and tritium-edge. Converts between local meters (used by physics, tracking, and simulation) and WGS84 lat/lng (used by APIs and map display). Also provides camera ground-plane projection for converting pixel coordinates to real-world positions, and haversine distance calculation.

A geo-reference point (map center) anchors all local coordinates to real-world lat/lng. Local origin (0,0,0) corresponds to the reference point. Convention: +X = East, +Y = North, +Z = Up, 1 unit = 1 meter.

## Key Files

| File | Purpose |
|------|---------|
| `__init__.py` | All code lives here — GeoReference, CameraCalibration, coordinate transforms, haversine |

## API

| Function | Purpose |
|----------|---------|
| `init_reference(lat, lng, alt)` | Set the geo-reference point (call once at startup) |
| `get_reference()` | Get current geo-reference point |
| `local_to_latlng(x, y, z)` | Convert local meters to lat/lng/alt dict |
| `latlng_to_local(lat, lng, alt)` | Convert lat/lng/alt to local meters tuple |
| `local_to_latlng_2d(x, y)` | Convenience 2D conversion to (lat, lng) tuple |
| `camera_pixel_to_ground(cx, cy, calib)` | Project normalized image coordinates to ground plane |
| `haversine_distance(lat1, lng1, lat2, lng2)` | Great-circle distance in meters |

## Related

- [../../../../tritium-sc/src/engine/tactical/geo.py](../../../../tritium-sc/src/engine/tactical/geo.py) — SC-side geo module (uses this library)
- [../../../../tritium-sc/src/engine/tactical/trilateration.py](../../../../tritium-sc/src/engine/tactical/trilateration.py) — Multi-node position estimation
- [../models/gis.py](../models/gis.py) — GeoPoint and MapTile models
