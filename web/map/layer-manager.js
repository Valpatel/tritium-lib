// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * GeoJSONLayerManager — abstracts MapLibre source/layer CRUD with caching.
 *
 * Instead of each overlay manually calling map.getSource(), checking hashes,
 * and managing source+layer lifecycle, callers use:
 *
 *   manager.update('patrol-routes', geojson, layerConfigs);
 *   manager.setVisibility('patrol-routes', true);
 *   manager.clear('patrol-routes');
 *
 * Handles:
 * - Create-or-update sources (idempotent)
 * - Hash-based change detection (skip updates if data unchanged)
 * - Multi-layer support per source (fill + line + circle from same GeoJSON)
 * - Visibility toggling for all layers in a group
 * - Cleanup on destroy
 *
 * Usage:
 *   import { GeoJSONLayerManager } from '/lib/map/layer-manager.js';
 *   const layers = new GeoJSONLayerManager(map);
 *   layers.update('hazards', geojson, [
 *       { id: 'hazard-fill', type: 'fill', paint: { 'fill-color': '#ff0000', 'fill-opacity': 0.2 } },
 *       { id: 'hazard-line', type: 'line', paint: { 'line-color': '#ff0000', 'line-width': 2 } },
 *   ]);
 */

/**
 * Fast hash for GeoJSON change detection.
 * Not cryptographic — just needs to detect when data changed.
 */
function hashGeoJSON(geojson) {
    if (!geojson?.features) return '0';
    const features = geojson.features;
    if (features.length === 0) return 'empty';
    // Hash based on feature count + first/last coordinates + property keys
    let h = features.length;
    for (let i = 0; i < Math.min(features.length, 5); i++) {
        const f = features[i];
        const coords = f.geometry?.coordinates;
        if (Array.isArray(coords)) {
            const flat = coords.flat?.(3) || coords;
            for (let j = 0; j < Math.min(flat.length, 4); j++) {
                h = (h * 31 + (typeof flat[j] === 'number' ? Math.round(flat[j] * 1e6) : 0)) | 0;
            }
        }
        if (f.properties) {
            for (const k of Object.keys(f.properties)) {
                for (let c = 0; c < k.length; c++) h = (h * 31 + k.charCodeAt(c)) | 0;
            }
        }
    }
    return String(h);
}

export class GeoJSONLayerManager {
    /**
     * @param {maplibregl.Map} map — MapLibre GL JS map instance
     */
    constructor(map) {
        this._map = map;
        this._sources = new Map();   // sourceId → { hash, layerIds }
        this._visible = new Map();   // sourceId → boolean
    }

    /**
     * Update a GeoJSON source and its layers. Creates source + layers if they
     * don't exist yet. Skips update if data hash matches (no change).
     *
     * @param {string} sourceId — unique source identifier
     * @param {Object} geojson — GeoJSON FeatureCollection
     * @param {Array<Object>} [layerConfigs] — layer definitions (only used on first call)
     *   Each: { id, type, paint, layout?, filter?, minzoom?, maxzoom? }
     * @returns {boolean} — true if data was actually updated
     */
    update(sourceId, geojson, layerConfigs = []) {
        const newHash = hashGeoJSON(geojson);
        const entry = this._sources.get(sourceId);

        if (entry) {
            // Source exists — check if data changed
            if (entry.hash === newHash) return false;
            entry.hash = newHash;
            const src = this._map.getSource(sourceId);
            if (src) src.setData(geojson);
            return true;
        }

        // First call — create source + layers
        this._map.addSource(sourceId, { type: 'geojson', data: geojson });
        const layerIds = [];
        for (const cfg of layerConfigs) {
            const layerDef = {
                id: cfg.id,
                type: cfg.type,
                source: sourceId,
                paint: cfg.paint || {},
            };
            if (cfg.layout) layerDef.layout = cfg.layout;
            if (cfg.filter) layerDef.filter = cfg.filter;
            if (cfg.minzoom != null) layerDef.minzoom = cfg.minzoom;
            if (cfg.maxzoom != null) layerDef.maxzoom = cfg.maxzoom;
            this._map.addLayer(layerDef);
            layerIds.push(cfg.id);
        }
        this._sources.set(sourceId, { hash: newHash, layerIds });
        this._visible.set(sourceId, true);
        return true;
    }

    /**
     * Set visibility for all layers in a source group.
     */
    setVisibility(sourceId, visible) {
        const entry = this._sources.get(sourceId);
        if (!entry) return;
        const vis = visible ? 'visible' : 'none';
        for (const layerId of entry.layerIds) {
            if (this._map.getLayer(layerId)) {
                this._map.setLayoutProperty(layerId, 'visibility', vis);
            }
        }
        this._visible.set(sourceId, visible);
    }

    /**
     * Toggle visibility for a source group.
     * @returns {boolean} — new visibility state
     */
    toggle(sourceId) {
        const current = this._visible.get(sourceId) ?? true;
        this.setVisibility(sourceId, !current);
        return !current;
    }

    /** Check if a source group is visible. */
    isVisible(sourceId) {
        return this._visible.get(sourceId) ?? false;
    }

    /**
     * Clear data from a source (set empty FeatureCollection).
     */
    clear(sourceId) {
        const entry = this._sources.get(sourceId);
        if (!entry) return;
        entry.hash = 'empty';
        const src = this._map.getSource(sourceId);
        if (src) src.setData({ type: 'FeatureCollection', features: [] });
    }

    /**
     * Remove a source and all its layers from the map.
     */
    remove(sourceId) {
        const entry = this._sources.get(sourceId);
        if (!entry) return;
        for (const layerId of entry.layerIds) {
            if (this._map.getLayer(layerId)) this._map.removeLayer(layerId);
        }
        if (this._map.getSource(sourceId)) this._map.removeSource(sourceId);
        this._sources.delete(sourceId);
        this._visible.delete(sourceId);
    }

    /**
     * Remove all managed sources and layers.
     */
    destroy() {
        for (const sourceId of [...this._sources.keys()]) {
            this.remove(sourceId);
        }
    }

    /** Get all managed source IDs. */
    get sourceIds() {
        return [...this._sources.keys()];
    }
}
