# tritium-lib/web — Shared Frontend Library

Reusable JavaScript modules, CSS themes, and demo apps for the Tritium ecosystem.
Used by tritium-sc (Command Center), addons, and any future Tritium frontend.

**54 JS modules** across 5 packages. No build step -- vanilla ES modules served directly.

Copyright 2026 Matthew Valancy / Valpatel Software LLC / AGPL-3.0

## Architecture

```
web/
├── css/                    # Cyberpunk theme stylesheets
│   ├── cybercore.css       # CYBERCORE v1 — base cyberpunk theme
│   └── cybercore-v2.css    # CYBERCORE v2 — refined, smaller
├── map/                    # Tactical map components (MapLibre GL)
│   ├── coords.js           # Coordinate transforms: game <-> lngLat <-> Mercator
│   ├── layer-manager.js    # GeoJSON layer CRUD with hash-based caching
│   ├── draw-tools.js       # Polygon (geofence) and polyline (patrol) drawing
│   ├── battle-hud.js       # Battle HUD overlay (wave counter, kill feed, banners)
│   ├── overlays.js         # Tactical overlays (ranges, FOV cones, grids)
│   ├── unit-markers.js     # Unit marker renderer with alliance colors
│   ├── data-provider.js    # Extensible map data source base + registry
│   ├── index.js            # Barrel export — all map components
│   ├── asset-types/        # Extensible sensor/asset type registry
│   │   ├── base.js         # BaseAssetType — override for custom sensor types
│   │   ├── registry.js     # AssetTypeRegistry singleton
│   │   ├── camera.js       # Camera asset type
│   │   ├── ble-sensor.js   # BLE sensor asset type
│   │   ├── motion-sensor.js # Motion sensor asset type
│   │   └── mesh-radio.js   # Meshtastic mesh radio asset type
│   ├── effects/            # Combat visual effects (no map dependency)
│   │   ├── base.js         # CombatEffects manager + weapon VFX config
│   │   ├── projectile.js   # Projectile trail effect
│   │   ├── explosion.js    # Explosion effect
│   │   ├── particles.js    # Particle burst effect
│   │   ├── flash.js        # Screen flash effect
│   │   ├── floating-text.js # Floating damage/status text
│   │   └── index.js        # Barrel export
│   ├── three-units/        # Three.js 3D unit models
│   │   ├── base.js         # Base3DUnit — override build() for custom models
│   │   ├── index.js        # Model registry + barrel export
│   │   ├── turret.js       # Turret model (rotating barrel)
│   │   ├── drone.js        # Drone model (spinning rotors)
│   │   ├── rover.js        # Rover model (wheeled ground unit)
│   │   ├── person.js       # Person model (standing figure)
│   │   └── tank.js         # Tank model (treads + turret)
│   ├── providers/          # Built-in map data providers
│   │   ├── index.js        # Provider barrel export
│   │   ├── satellite.js    # Esri satellite + OSM tile providers
│   │   └── terrain.js      # Terrain segmentation provider
│   └── demos/              # Interactive demo pages
│       ├── index.html       # Demo index
│       ├── coords-demo.html
│       ├── layer-manager-demo.html
│       ├── effects-demo.html
│       ├── asset-types-demo.html
│       ├── draw-tools-demo.html
│       └── data-providers-demo.html
├── sim/                    # City simulation modules
│   ├── idm.js              # IDM car-following model
│   ├── mobil.js             # MOBIL lane-change model
│   ├── vehicle.js           # SimVehicle agent
│   ├── pedestrian.js        # SimPedestrian agent
│   ├── road-network.js      # Road graph with Bezier intersections
│   ├── traffic-controller.js # Traffic light controller
│   ├── spatial-grid.js      # Spatial hash grid for neighbor queries
│   ├── protest-engine.js    # Epstein civil violence model
│   ├── protest-scenario.js  # Protest scenario phases
│   ├── daily-routine.js     # NPC daily routine generator
│   ├── schedule-executor.js # SimClock + schedule-driven behavior
│   ├── weather.js           # Weather simulation
│   ├── identity.js          # NPC/vehicle identity generator
│   ├── procedural-city.js   # Procedural city road generator
│   ├── index.js             # Barrel export — all sim modules
│   └── demos/
│       └── sim-demo.html    # City simulation demo
├── panels/                 # Panel/window management
│   ├── panel-manager.js     # Draggable, resizable panel system
│   ├── tabbed-container.js  # Tabbed container widget
│   └── demos/
│       └── panel-demo.html  # Panel system demo
├── events.js               # EventBus — frontend pub/sub (on/off/emit)
├── store.js                # ReactiveStore — dot-path state with RAF-batched updates
├── websocket.js            # TritiumWebSocket — reconnect, ping, disconnected banner
├── command-palette.js      # Command palette — fuzzy search, Ctrl+K trigger
├── layout-manager.js       # Panel layout save/restore/import/export
└── utils.js                # Shared utilities (_esc, _timeAgo, _badge, _fetchJson)
```

## Serving Setup

tritium-lib/web is served to browsers in two ways:

### 1. Symlink (development)

tritium-sc creates a symlink so the frontend can import lib modules directly:

```
tritium-sc/src/frontend/lib -> tritium-lib/web
```

This means `import { MapCoords } from '/lib/map/coords.js'` resolves to
`tritium-lib/web/map/coords.js` at runtime.

### 2. FastAPI static mount (production)

In `tritium-sc/src/app/main.py`, the lib is mounted at `/lib/`:

```python
_lib_web = _sc_root.parent / "tritium-lib" / "web"
if _lib_web.exists():
    app.mount("/lib", StaticFiles(directory=_lib_web, follow_symlink=True), name="lib-static")
```

Both approaches serve the same files at the same URL path (`/lib/...`).

## Import Examples

### Map components

```javascript
// Individual imports (tree-shakeable)
import { MapCoords, haversineDistance } from '/lib/map/coords.js';
import { GeoJSONLayerManager } from '/lib/map/layer-manager.js';
import { DrawTools } from '/lib/map/draw-tools.js';
import { BattleHUD } from '/lib/map/battle-hud.js';
import { BaseAssetType } from '/lib/map/asset-types/base.js';
import { assetTypeRegistry } from '/lib/map/asset-types/registry.js';
import { MapDataProvider, providerRegistry } from '/lib/map/data-provider.js';

// Barrel import (everything at once)
import {
    MapCoords, GeoJSONLayerManager, DrawTools, BattleHUD,
    BaseAssetType, assetTypeRegistry,
    MapDataProvider, providerRegistry,
    Base3DUnit, registerModel, getModel,
    CombatEffects, ProjectileEffect, ExplosionEffect,
} from '/lib/map/index.js';
```

### Simulation

```javascript
import { IDM_DEFAULTS, idmStep } from '/lib/sim/idm.js';
import { decideLaneChange } from '/lib/sim/mobil.js';
import { SimVehicle } from '/lib/sim/vehicle.js';
import { SimPedestrian } from '/lib/sim/pedestrian.js';
import { RoadNetwork } from '/lib/sim/road-network.js';
import { ProtestEngine } from '/lib/sim/protest-engine.js';

// Or barrel import
import { IDM_DEFAULTS, idmStep, decideLaneChange, SimVehicle } from '/lib/sim/index.js';
```

### Core UI

```javascript
import { EventBus } from '/lib/events.js';
import { ReactiveStore } from '/lib/store.js';
import { TritiumWebSocket } from '/lib/websocket.js';
import { initCommandPalette } from '/lib/command-palette.js';
import { LayoutManager } from '/lib/layout-manager.js';
import { _esc, _timeAgo, _badge, _fetchJson } from '/lib/utils.js';
```

### CSS

```html
<link rel="stylesheet" href="/lib/css/cybercore.css">
<!-- or the v2 variant -->
<link rel="stylesheet" href="/lib/css/cybercore-v2.css">
```

## Demo Apps

Open these in a browser while the server is running at `:8000`:

| Demo | URL | What it shows |
|------|-----|---------------|
| **Demo Index** | `/lib/map/demos/` | Links to all map demos |
| Coords | `/lib/map/demos/coords-demo.html` | Game/lngLat/Mercator transforms, FOV cones, haversine |
| Layer Manager | `/lib/map/demos/layer-manager-demo.html` | GeoJSON layer CRUD with hash caching |
| Effects | `/lib/map/demos/effects-demo.html` | Combat effects: floating text, flash, explosions |
| Asset Types | `/lib/map/demos/asset-types-demo.html` | Asset type registry, register custom types |
| Draw Tools | `/lib/map/demos/draw-tools-demo.html` | Polygon and polyline drawing with undo |
| Data Providers | `/lib/map/demos/data-providers-demo.html` | Satellite, OSM, terrain, custom providers |
| Panels | `/lib/panels/demos/panel-demo.html` | Draggable/resizable panel system |
| Sim | `/lib/sim/demos/sim-demo.html` | City simulation (IDM, MOBIL, pedestrians) |

## Quick Start for Addon Developers

### Create a custom asset type (10 lines)

Register a new sensor type that appears in the map's asset placement menu:

```javascript
import { BaseAssetType } from '/lib/map/asset-types/base.js';
import { assetTypeRegistry } from '/lib/map/asset-types/registry.js';

class AcousticSensorType extends BaseAssetType {
    static typeId = 'acoustic';
    static label = 'Acoustic Sensor';
    static icon = 'A';
    static color = '#ff9900';
    static defaultRange = 100;
    static coverageShape = 'circle';
}

assetTypeRegistry.register(AcousticSensorType);
// Done — the sensor now appears in the asset type list with its icon, color, and coverage ring.
```

### Create a custom map data provider

Add a new data layer to the map:

```javascript
import { MapDataProvider, providerRegistry } from '/lib/map/data-provider.js';

class WaterPipeProvider extends MapDataProvider {
    static providerId = 'city-water';
    static label = 'City Water Pipes';
    static category = 'municipal';
    static icon = 'W';

    getSourceConfig() {
        return { type: 'geojson', data: { type: 'FeatureCollection', features: [] } };
    }
    getLayerConfigs() {
        return [{ id: 'water-pipes', type: 'line',
                  paint: { 'line-color': '#00aaff', 'line-width': 2 } }];
    }
    async fetchData() {
        const res = await fetch('/api/gis/water-pipes');
        return res.ok ? res.json() : null;
    }
}

providerRegistry.register(new WaterPipeProvider());
```

### Create a custom 3D unit model

Add a new Three.js model for the 3D tactical view:

```javascript
import { Base3DUnit, registerModel } from '/lib/map/three-units/index.js';

class GraphlingModel extends Base3DUnit {
    static typeId = 'graphling';

    build(THREE, opts = {}) {
        const g = new THREE.Group();
        const body = this.sphere(THREE, 0.4, opts.color || 0x05ffa1);
        body.position.y = 0.5;
        g.add(body);
        this.group = g;
        return g;
    }

    animate(dt) {
        if (this.group) this.group.rotation.y += dt * 0.5;
    }
}

registerModel(GraphlingModel);
```

## How SC Extends These Modules

tritium-sc imports and extends lib modules rather than forking them. The extension pattern:

**ReactiveStore** -- SC creates a store instance and adds domain-specific keys:

```javascript
import { ReactiveStore } from '/lib/store.js';
const store = new ReactiveStore();
store.set('targets', {});        // SC-specific state
store.set('plugins.loaded', []); // SC-specific state
// Lib modules can also read/write the same store instance
```

**TritiumWebSocket** -- SC wraps with application-specific message routing:

```javascript
import { TritiumWebSocket } from '/lib/websocket.js';
const ws = new TritiumWebSocket('/ws');
ws.onMessage('target.update', (data) => { /* SC-specific handler */ });
ws.onMessage('sim.tick', (data) => { /* route to sim modules */ });
```

**EventBus** -- SC uses a single shared bus for plugin-to-plugin communication:

```javascript
import { EventBus } from '/lib/events.js';
const bus = new EventBus();
// Plugins subscribe to events from other plugins or from lib sim modules
bus.on('sim.vehicle.spawned', (e) => { /* update map markers */ });
bus.on('tracking.target.new', (e) => { /* show notification */ });
```

**Map components** -- SC composes lib map modules into its main map view. Plugins register additional asset types, data providers, and 3D models through the lib registries. The lib owns the base classes and registries; SC and plugins provide the concrete implementations.

**Panel system** -- SC uses PanelManager from lib for all floating panels. Plugins create panels by calling the manager API; layout persistence is handled by LayoutManager.

## Module Counts

| Package | JS Modules | HTML Demos | Description |
|---------|-----------|------------|-------------|
| `map/` | 28 | 6 | Tactical map components |
| `sim/` | 14 | 1 | City simulation |
| `panels/` | 2 | 1 | Panel/window management |
| `css/` | 2 (CSS) | — | Cyberpunk themes |
| root | 5 | — | EventBus, Store, WS, Utils, Layout |
| **Total** | **54** | **8** | |
