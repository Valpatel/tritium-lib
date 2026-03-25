// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * tritium-lib/web/map — Reusable tactical map components.
 *
 * Modules:
 *   coords          — Coordinate system (game↔lngLat↔Mercator)
 *   layer-manager   — GeoJSON layer CRUD + caching
 *   effects/        — Combat visual effects (projectile, explosion, particles, flash, text)
 *   asset-types/    — Extensible asset type registry (camera, BLE, motion, mesh, custom)
 */

export { MapCoords, buildFovConePolygon, buildCirclePolygon, haversineDistance } from './coords.js';
export { GeoJSONLayerManager } from './layer-manager.js';
export { DrawTools } from './draw-tools.js';
export { BattleHUD } from './battle-hud.js';
export { TacticalOverlays } from './overlays.js';
export { MapDataProvider, MapDataProviderRegistry, providerRegistry } from './data-provider.js';
export { EsriSatelliteProvider, OSMTilesProvider } from './providers/satellite.js';
export { TerrainSegmentationProvider } from './providers/terrain.js';
export { UnitMarkerRenderer, abbreviateName, ALLIANCE_COLORS } from './unit-markers.js';
export { Base3DUnit, registerModel, getModel, allModelTypes } from './three-units/index.js';
export { TurretModel } from './three-units/turret.js';
export { DroneModel } from './three-units/drone.js';
export { RoverModel } from './three-units/rover.js';
export { PersonModel } from './three-units/person.js';
export { TankModel } from './three-units/tank.js';
export { CombatEffects, DEFAULT_WEAPON_VFX } from './effects/base.js';
export { ProjectileEffect } from './effects/projectile.js';
export { ExplosionEffect } from './effects/explosion.js';
export { ParticleBurst } from './effects/particles.js';
export { FlashEffect } from './effects/flash.js';
export { FloatingText } from './effects/floating-text.js';
export { BaseAssetType } from './asset-types/base.js';
export { assetTypeRegistry } from './asset-types/registry.js';
export { CameraAssetType } from './asset-types/camera.js';
export { BLESensorAssetType } from './asset-types/ble-sensor.js';
export { MotionSensorAssetType } from './asset-types/motion-sensor.js';
export { MeshRadioAssetType } from './asset-types/mesh-radio.js';
