// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * AssetTypeRegistry — runtime registry for extensible asset types.
 *
 * Built-in types are registered on import. Addons register custom types
 * at runtime. The map layer and context menu read from this registry
 * to render coverage, generate popups, and offer placement options.
 *
 * Usage:
 *   import { assetTypeRegistry } from '/lib/map/asset-types/registry.js';
 *
 *   // Get a type
 *   const camType = assetTypeRegistry.get('camera');
 *   console.log(camType.color, camType.defaultRange);
 *
 *   // Register a custom addon type
 *   assetTypeRegistry.register(MyLidarAssetType);
 *
 *   // List all types (for context menu)
 *   for (const type of assetTypeRegistry.all()) {
 *       console.log(type.typeId, type.label);
 *   }
 */

import { CameraAssetType } from './camera.js';
import { BLESensorAssetType } from './ble-sensor.js';
import { MotionSensorAssetType } from './motion-sensor.js';
import { MeshRadioAssetType } from './mesh-radio.js';

class AssetTypeRegistry {
    constructor() {
        this._types = new Map();
    }

    /**
     * Register an asset type class.
     * @param {typeof BaseAssetType} TypeClass
     */
    register(TypeClass) {
        if (!TypeClass.typeId) throw new Error('Asset type must have a static typeId');
        this._types.set(TypeClass.typeId, TypeClass);
    }

    /**
     * Get an asset type class by typeId.
     * @param {string} typeId
     * @returns {typeof BaseAssetType | undefined}
     */
    get(typeId) {
        return this._types.get(typeId);
    }

    /**
     * Get all registered type classes.
     * @returns {Array<typeof BaseAssetType>}
     */
    all() {
        return [...this._types.values()];
    }

    /**
     * Get all type IDs.
     * @returns {Array<string>}
     */
    typeIds() {
        return [...this._types.keys()];
    }

    /**
     * Look up the type for an asset based on its asset_class or asset_type field.
     * Falls back to matching by capabilities if no direct match.
     * @param {Object} asset — asset from /api/assets
     * @returns {typeof BaseAssetType | undefined}
     */
    resolveForAsset(asset) {
        // Direct match by asset_class
        for (const T of this._types.values()) {
            if (T.assetClass === asset.asset_class) return T;
        }
        // Match by typeId in asset fields
        const typeHint = asset.asset_class || asset.asset_type || '';
        return this._types.get(typeHint) || undefined;
    }

    /** Number of registered types. */
    get size() {
        return this._types.size;
    }
}

// Singleton registry with built-in types
export const assetTypeRegistry = new AssetTypeRegistry();
assetTypeRegistry.register(CameraAssetType);
assetTypeRegistry.register(BLESensorAssetType);
assetTypeRegistry.register(MotionSensorAssetType);
assetTypeRegistry.register(MeshRadioAssetType);
