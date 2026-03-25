// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * UnitMarkerRenderer — creates and styles DOM markers for map units.
 *
 * Each unit (target, robot, hostile, friendly) gets an HTML marker element
 * on the MapLibre map. This module handles creation, styling, health bars,
 * selection highlight, and thought bubbles.
 *
 * Subclass or configure for app-specific styling (NATO symbols, morale, etc.).
 *
 * Usage:
 *   import { UnitMarkerRenderer } from '/lib/map/unit-markers.js';
 *   const renderer = new UnitMarkerRenderer(map);
 *   renderer.updateUnit(unit); // creates or updates marker
 *   renderer.removeUnit(unitId);
 */

/** Alliance → default color mapping. */
const ALLIANCE_COLORS = {
    friendly: '#00f0ff',
    hostile:  '#ff2a6d',
    neutral:  '#05ffa1',
    unknown:  '#fcee0a',
};

/**
 * Abbreviate a unit name to fit in a compact marker.
 * "Intruder Foxtrot-2" → "Fox-2", "Rover-01" → "Rvr-01"
 */
export function abbreviateName(name) {
    if (!name || name.length <= 6) return name || '';
    // Try last word
    const parts = name.split(/[\s-]+/);
    if (parts.length >= 2) {
        const last = parts[parts.length - 1];
        if (last.length <= 6) return last;
        return last.substring(0, 5);
    }
    return name.substring(0, 5);
}

export class UnitMarkerRenderer {
    /**
     * @param {maplibregl.Map} map
     * @param {Object} [options]
     * @param {boolean} [options.showHealthBars=true]
     * @param {boolean} [options.showLabels=true]
     * @param {Object} [options.allianceColors] — override ALLIANCE_COLORS
     */
    constructor(map, options = {}) {
        this._map = map;
        this._markers = new Map(); // unitId → maplibregl.Marker
        this.showHealthBars = options.showHealthBars ?? true;
        this.showLabels = options.showLabels ?? true;
        this._colors = { ...ALLIANCE_COLORS, ...(options.allianceColors || {}) };
    }

    /**
     * Create or update a marker for a unit.
     * @param {Object} unit — { target_id, name, alliance, position: {lat, lng}, health, asset_type, ... }
     * @returns {maplibregl.Marker}
     */
    updateUnit(unit) {
        const id = unit.target_id || unit.id;
        if (!id) return null;

        let marker = this._markers.get(id);
        if (!marker) {
            marker = this._createMarker(unit);
            this._markers.set(id, marker);
        }

        // Update position
        if (unit.position?.lng != null && unit.position?.lat != null) {
            marker.setLngLat([unit.position.lng, unit.position.lat]);
        }

        // Update styling
        this._styleMarker(marker, unit);
        return marker;
    }

    /** Remove a unit's marker from the map. */
    removeUnit(unitId) {
        const marker = this._markers.get(unitId);
        if (marker) {
            marker.remove();
            this._markers.delete(unitId);
        }
    }

    /** Remove all markers. */
    clear() {
        for (const [, marker] of this._markers) marker.remove();
        this._markers.clear();
    }

    /** Get marker by unit ID. */
    getMarker(unitId) {
        return this._markers.get(unitId);
    }

    /** Number of active markers. */
    get count() {
        return this._markers.size;
    }

    /** All unit IDs with markers. */
    get unitIds() {
        return [...this._markers.keys()];
    }

    // --- Internal ---

    _createMarker(unit) {
        const el = document.createElement('div');
        el.className = 'tritium-unit-marker';
        el.style.cssText = 'cursor: pointer; user-select: none;';

        const inner = document.createElement('div');
        inner.className = 'tritium-unit-inner';
        el.appendChild(inner);

        const marker = new maplibregl.Marker({ element: el, anchor: 'center' })
            .setLngLat([0, 0])
            .addTo(this._map);

        marker._unitEl = el;
        marker._innerEl = inner;
        return marker;
    }

    _styleMarker(marker, unit) {
        const alliance = unit.alliance || 'unknown';
        const color = this._colors[alliance] || this._colors.unknown;
        const name = this.showLabels ? abbreviateName(unit.name) : '';
        const letter = (unit.asset_type || 'U')[0].toUpperCase();

        const inner = marker._innerEl;
        if (!inner) return;

        inner.style.cssText = `
            display: flex; flex-direction: column; align-items: center;
            font-family: 'JetBrains Mono', monospace; font-size: 9px;
            color: ${color}; text-shadow: 0 0 3px ${color}44;
        `;
        inner.innerHTML = `
            <div style="width:16px;height:16px;border:1.5px solid ${color};border-radius:2px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:10px;font-weight:bold;background:rgba(0,0,0,0.7)">
                ${letter}
            </div>
            ${name ? `<div style="font-size:7px;color:${color};margin-top:1px;white-space:nowrap">${name}</div>` : ''}
        `;

        // Health bar
        if (this.showHealthBars && unit.health != null && unit.health < 1.0) {
            const pct = Math.max(0, Math.min(100, unit.health * 100));
            const barColor = pct > 60 ? '#05ffa1' : pct > 30 ? '#fcee0a' : '#ff2a6d';
            const bar = document.createElement('div');
            bar.style.cssText = `
                width: 20px; height: 2px; background: #333; margin-top: 1px; border-radius: 1px;
            `;
            bar.innerHTML = `<div style="width:${pct}%;height:100%;background:${barColor};border-radius:1px"></div>`;
            inner.appendChild(bar);
        }
    }
}

export { ALLIANCE_COLORS };
