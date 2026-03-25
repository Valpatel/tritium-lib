// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * BaseAssetType — extensible base class for map asset types.
 *
 * Each asset type (camera, BLE sensor, motion detector, mesh radio, etc.)
 * extends this class and overrides properties + methods. Addons register
 * custom types at runtime via the AssetTypeRegistry.
 *
 * Properties define how the asset renders on the map (color, icon, coverage
 * shape). Methods define behavior (popup content, marker style, data generation).
 *
 * Usage:
 *   class LidarAssetType extends BaseAssetType {
 *       static typeId = 'lidar';
 *       static label = 'LIDAR Scanner';
 *       static icon = 'L';
 *       static color = '#aa44ff';
 *       static defaultRange = 50;
 *       static coverageShape = 'cone';
 *       static defaultFov = 60;
 *   }
 *   registry.register(LidarAssetType);
 */

export class BaseAssetType {
    // --- Override these in subclasses ---

    /** Unique type identifier (e.g., 'camera', 'ble_sensor'). */
    static typeId = 'generic';

    /** Human-readable label. */
    static label = 'Generic Asset';

    /** Single-character icon for map markers. */
    static icon = '?';

    /** Primary color (hex string). */
    static color = '#888888';

    /** Default detection/coverage range in meters. */
    static defaultRange = 10;

    /** Coverage shape: 'circle' (omni) or 'cone' (directional). */
    static coverageShape = 'circle';

    /** Default FOV in degrees (only used if coverageShape = 'cone'). */
    static defaultFov = 360;

    /** Default mounting height in meters. */
    static defaultHeight = 2.0;

    /** Default mounting type. */
    static defaultMounting = 'wall';

    /** Default capabilities list. */
    static defaultCapabilities = [];

    /** Asset class for the /api/assets backend. */
    static assetClass = 'sensor';

    // --- Methods (override for custom behavior) ---

    /**
     * Generate popup HTML for this asset when clicked on the map.
     * @param {Object} asset — asset properties from the API
     * @returns {string} — HTML string
     */
    static getPopupHtml(asset) {
        const range = asset.coverageRadius || asset.coverage_radius_meters || this.defaultRange;
        const mount = asset.mounting || asset.mounting_type || this.defaultMounting;
        return `
            <div style="font-family:monospace;font-size:11px;color:${this.color}">
                <div style="font-weight:bold;margin-bottom:3px">${asset.name || this.label}</div>
                <div style="color:#888;font-size:9px">${this.label} · ${mount} mount</div>
                <div style="color:#aaa;font-size:9px">${range}m range · ${asset.height || asset.height_meters || this.defaultHeight}m height</div>
            </div>
        `;
    }

    /**
     * Get the MapLibre paint properties for coverage circle.
     * @returns {Object} — MapLibre paint config
     */
    static getCoveragePaint() {
        return {
            fillColor: this.color.replace(')', ', 0.08)').replace('rgb', 'rgba').replace('#', '') ?
                `rgba(${parseInt(this.color.slice(1,3),16)}, ${parseInt(this.color.slice(3,5),16)}, ${parseInt(this.color.slice(5,7),16)}, 0.08)` :
                'rgba(136, 136, 136, 0.06)',
            strokeColor: this.color,
            strokeWidth: 1,
            strokeOpacity: 0.4,
        };
    }

    /**
     * Get default values for creating a new asset of this type.
     * @returns {Object} — default asset properties
     */
    static getDefaults() {
        return {
            asset_type: 'fixed',
            asset_class: this.assetClass,
            height_meters: this.defaultHeight,
            mounting_type: this.defaultMounting,
            coverage_radius_meters: this.defaultRange,
            coverage_cone_angle: this.coverageShape === 'cone' ? this.defaultFov : null,
            capabilities: this.defaultCapabilities,
            connection_url: 'simulated://demo',
        };
    }
}
