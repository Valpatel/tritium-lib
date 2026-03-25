// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * TacticalOverlays — GeoJSON-based tactical map overlays.
 *
 * Each overlay is a named layer group that renders GeoJSON data on the map.
 * Uses GeoJSONLayerManager for efficient create/update/toggle.
 *
 * Built-in overlays:
 *   - Patrol routes (polylines with arrows)
 *   - Dispatch arrows (animated dashed lines)
 *   - Weapon range circles
 *   - Combat heatmap
 *   - Swarm hull (convex hull around drone group)
 *   - Squad formation hulls
 *   - Hazard zones
 *   - Hostile objectives
 *   - Crowd density
 *   - Cover points
 *   - Unit signals (comm range circles)
 *   - Engagement lines (who's shooting whom)
 *
 * Usage:
 *   import { TacticalOverlays } from '/lib/map/overlays.js';
 *   const overlays = new TacticalOverlays(layerManager);
 *   overlays.updatePatrolRoutes(geojson);
 *   overlays.toggleHazardZones();
 */

export class TacticalOverlays {
    /**
     * @param {import('./layer-manager.js').GeoJSONLayerManager} layerManager
     */
    constructor(layerManager) {
        this._lm = layerManager;
        this._registered = new Set();
    }

    /**
     * Register and update a named overlay.
     * @param {string} name — overlay identifier
     * @param {Object} geojson — GeoJSON FeatureCollection
     * @param {Array} layerConfigs — layer definitions (used on first call)
     */
    update(name, geojson, layerConfigs) {
        if (!this._registered.has(name)) {
            this._registered.add(name);
        }
        this._lm.update(name, geojson, layerConfigs);
    }

    /** Toggle visibility of a named overlay. */
    toggle(name) { return this._lm.toggle(name); }

    /** Set visibility of a named overlay. */
    setVisible(name, visible) { this._lm.setVisibility(name, visible); }

    /** Clear data from a named overlay. */
    clear(name) { this._lm.clear(name); }

    /** Check if an overlay is visible. */
    isVisible(name) { return this._lm.isVisible(name); }

    /** Get all registered overlay names. */
    get names() { return [...this._registered]; }

    // ── Convenience methods for common overlays ──────────────────

    /** Update patrol route lines. */
    updatePatrolRoutes(geojson) {
        this.update('patrol-routes', geojson, [
            { id: 'patrol-routes-line', type: 'line', paint: { 'line-color': '#05ffa1', 'line-width': 2, 'line-dasharray': [6, 3] } },
            { id: 'patrol-routes-dots', type: 'circle', paint: { 'circle-radius': 4, 'circle-color': '#05ffa1', 'circle-stroke-color': '#000', 'circle-stroke-width': 1 }, filter: ['==', '$type', 'Point'] },
        ]);
    }

    /** Update weapon range circle. */
    updateWeaponRange(geojson) {
        this.update('weapon-range', geojson, [
            { id: 'weapon-range-fill', type: 'fill', paint: { 'fill-color': 'rgba(0, 240, 255, 0.06)', 'fill-opacity': 0.5 } },
            { id: 'weapon-range-line', type: 'line', paint: { 'line-color': '#00f0ff', 'line-width': 1, 'line-opacity': 0.4, 'line-dasharray': [4, 2] } },
        ]);
    }

    /** Update hazard zone polygons. */
    updateHazardZones(geojson) {
        this.update('hazard-zones', geojson, [
            { id: 'hazard-zones-fill', type: 'fill', paint: { 'fill-color': ['get', 'color'], 'fill-opacity': 0.15 } },
            { id: 'hazard-zones-line', type: 'line', paint: { 'line-color': ['get', 'color'], 'line-width': 2, 'line-dasharray': [4, 4] } },
        ]);
    }

    /** Update crowd density heatmap. */
    updateCrowdDensity(geojson) {
        this.update('crowd-density', geojson, [
            { id: 'crowd-density-heat', type: 'heatmap', paint: {
                'heatmap-weight': ['get', 'weight'],
                'heatmap-radius': 20,
                'heatmap-color': ['interpolate', ['linear'], ['heatmap-density'],
                    0, 'rgba(0,0,0,0)', 0.3, '#05ffa1', 0.6, '#fcee0a', 1, '#ff2a6d'],
                'heatmap-opacity': 0.5,
            }},
        ]);
    }

    /** Update cover point markers. */
    updateCoverPoints(geojson) {
        this.update('cover-points', geojson, [
            { id: 'cover-points-circle', type: 'circle', paint: {
                'circle-radius': 5, 'circle-color': '#05ffa1',
                'circle-stroke-color': '#000', 'circle-stroke-width': 1, 'circle-opacity': 0.6,
            }},
        ]);
    }

    /** Update engagement lines (shooter → target). */
    updateEngagementLines(geojson) {
        this.update('engagement-lines', geojson, [
            { id: 'engagement-lines-line', type: 'line', paint: {
                'line-color': '#ff2a6d', 'line-width': 1, 'line-opacity': 0.5, 'line-dasharray': [2, 2],
            }},
        ]);
    }

    /** Update dispatch arrows. */
    updateDispatchArrows(geojson) {
        this.update('dispatch-arrows', geojson, [
            { id: 'dispatch-arrows-line', type: 'line', paint: {
                'line-color': '#00f0ff', 'line-width': 2, 'line-opacity': 0.6, 'line-dasharray': [8, 4],
            }},
        ]);
    }

    /** Clean up all overlays. */
    destroy() {
        for (const name of this._registered) {
            this._lm.remove(name);
        }
        this._registered.clear();
    }
}
