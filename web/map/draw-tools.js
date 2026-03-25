// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * DrawTools — polygon and polyline drawing tools for MapLibre GL maps.
 *
 * Provides two modes:
 *   - Polygon draw: click vertices to draw a closed polygon (geofence)
 *   - Polyline draw: click waypoints to draw an open path (patrol route)
 *
 * Both use rubber-band preview lines, vertex markers, and support
 * ESC to cancel, Enter to finish, and Backspace to undo last vertex.
 *
 * Usage:
 *   import { DrawTools } from '/lib/map/draw-tools.js';
 *   const draw = new DrawTools(map);
 *   draw.startPolygon({ onFinish: (coords) => { ... } });
 *   draw.startPolyline({ onFinish: (coords) => { ... } });
 */

export class DrawTools {
    /**
     * @param {maplibregl.Map} map
     * @param {Object} [options]
     * @param {string} [options.lineColor='#00f0ff']
     * @param {string} [options.fillColor='rgba(0, 240, 255, 0.1)']
     * @param {string} [options.vertexColor='#00f0ff']
     * @param {number} [options.lineWidth=2]
     */
    constructor(map, options = {}) {
        this._map = map;
        this._color = options.lineColor || '#00f0ff';
        this._fillColor = options.fillColor || 'rgba(0, 240, 255, 0.1)';
        this._vertexColor = options.vertexColor || '#00f0ff';
        this._lineWidth = options.lineWidth || 2;

        this._mode = null;          // 'polygon' | 'polyline' | null
        this._vertices = [];        // [lng, lat] pairs
        this._markers = [];         // maplibregl.Marker instances for vertices
        this._mouseLngLat = null;   // current mouse position
        this._onFinish = null;      // callback
        this._onCancel = null;      // callback
        this._sourceId = 'draw-tools-preview';
        this._layerFillId = 'draw-tools-fill';
        this._layerLineId = 'draw-tools-line';

        this._boundKeydown = this._onKeydown.bind(this);
        this._boundMousemove = this._onMousemove.bind(this);
        this._boundClick = this._onClick.bind(this);
    }

    /** Is a draw operation in progress? */
    get active() { return this._mode !== null; }

    /** Current mode: 'polygon', 'polyline', or null. */
    get mode() { return this._mode; }

    /** Number of vertices placed so far. */
    get vertexCount() { return this._vertices.length; }

    /**
     * Start polygon drawing mode (for geofences).
     * @param {Object} opts
     * @param {Function} opts.onFinish — called with Array<[lng,lat]> when Enter pressed
     * @param {Function} [opts.onCancel] — called when ESC pressed
     */
    startPolygon(opts = {}) {
        this._start('polygon', opts);
    }

    /**
     * Start polyline drawing mode (for patrol routes).
     * @param {Object} opts
     * @param {Function} opts.onFinish — called with Array<[lng,lat]> when Enter pressed
     * @param {Function} [opts.onCancel] — called when ESC pressed
     */
    startPolyline(opts = {}) {
        this._start('polyline', opts);
    }

    /** Cancel current draw operation. */
    cancel() {
        if (!this._mode) return;
        const cb = this._onCancel;
        this._cleanup();
        if (cb) cb();
    }

    /** Finish current draw operation (same as pressing Enter). */
    finish() {
        if (!this._mode || this._vertices.length < 2) return;
        const coords = [...this._vertices];
        const cb = this._onFinish;
        this._cleanup();
        if (cb) cb(coords);
    }

    /** Undo last vertex. */
    undoVertex() {
        if (this._vertices.length === 0) return;
        this._vertices.pop();
        const marker = this._markers.pop();
        if (marker) marker.remove();
        this._updatePreview();
    }

    // --- Internal ---

    _start(mode, opts) {
        if (this._mode) this.cancel(); // cancel any existing draw
        this._mode = mode;
        this._vertices = [];
        this._markers = [];
        this._onFinish = opts.onFinish || null;
        this._onCancel = opts.onCancel || null;
        this._mouseLngLat = null;

        // Add preview source + layers
        if (!this._map.getSource(this._sourceId)) {
            this._map.addSource(this._sourceId, {
                type: 'geojson',
                data: { type: 'FeatureCollection', features: [] },
            });
            this._map.addLayer({
                id: this._layerFillId,
                type: 'fill',
                source: this._sourceId,
                paint: { 'fill-color': this._fillColor, 'fill-opacity': 0.3 },
                filter: ['==', '$type', 'Polygon'],
            });
            this._map.addLayer({
                id: this._layerLineId,
                type: 'line',
                source: this._sourceId,
                paint: {
                    'line-color': this._color,
                    'line-width': this._lineWidth,
                    'line-dasharray': [3, 2],
                },
            });
        }

        // Wire events
        this._map.getCanvas().style.cursor = 'crosshair';
        this._map.on('mousemove', this._boundMousemove);
        this._map.on('click', this._boundClick);
        document.addEventListener('keydown', this._boundKeydown);
    }

    _cleanup() {
        this._mode = null;
        this._onFinish = null;
        this._onCancel = null;

        // Remove markers
        for (const m of this._markers) m.remove();
        this._markers = [];
        this._vertices = [];

        // Clear preview
        const src = this._map.getSource(this._sourceId);
        if (src) src.setData({ type: 'FeatureCollection', features: [] });

        // Unwire events
        this._map.getCanvas().style.cursor = '';
        this._map.off('mousemove', this._boundMousemove);
        this._map.off('click', this._boundClick);
        document.removeEventListener('keydown', this._boundKeydown);
    }

    _onClick(e) {
        const coord = [e.lngLat.lng, e.lngLat.lat];
        this._vertices.push(coord);

        // Add vertex marker
        const el = document.createElement('div');
        el.style.cssText = `
            width: 10px; height: 10px; border-radius: 50%;
            background: ${this._vertexColor}; border: 2px solid #fff;
            box-shadow: 0 0 4px ${this._vertexColor};
        `;
        const marker = new maplibregl.Marker({ element: el })
            .setLngLat(coord)
            .addTo(this._map);
        this._markers.push(marker);

        this._updatePreview();
    }

    _onMousemove(e) {
        this._mouseLngLat = [e.lngLat.lng, e.lngLat.lat];
        if (this._vertices.length > 0) this._updatePreview();
    }

    _onKeydown(e) {
        if (e.key === 'Escape') {
            e.preventDefault();
            this.cancel();
        } else if (e.key === 'Enter') {
            e.preventDefault();
            this.finish();
        } else if (e.key === 'Backspace' && this._vertices.length > 0) {
            e.preventDefault();
            this.undoVertex();
        }
    }

    _updatePreview() {
        const src = this._map.getSource(this._sourceId);
        if (!src) return;

        const features = [];
        const coords = [...this._vertices];
        if (this._mouseLngLat) coords.push(this._mouseLngLat);

        if (coords.length >= 2) {
            if (this._mode === 'polygon' && coords.length >= 3) {
                // Closed polygon preview
                features.push({
                    type: 'Feature',
                    geometry: { type: 'Polygon', coordinates: [[...coords, coords[0]]] },
                });
            }
            // Line preview (always shown)
            features.push({
                type: 'Feature',
                geometry: { type: 'LineString', coordinates: coords },
            });
        }

        src.setData({ type: 'FeatureCollection', features });
    }
}
