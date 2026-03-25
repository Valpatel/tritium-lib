// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

import { MapDataProvider } from '../data-provider.js';

/** Tritium terrain segmentation layer (from /api/terrain/layer). */
export class TerrainSegmentationProvider extends MapDataProvider {
    static providerId = 'terrain-segmentation';
    static label = 'Terrain (Segmented)';
    static category = 'intel';
    static icon = 'T';
    static refreshInterval = 60000; // refresh every 60s

    constructor(apiBase = '') {
        super();
        this._apiBase = apiBase;
    }

    getSourceConfig() {
        return { type: 'geojson', data: { type: 'FeatureCollection', features: [] } };
    }

    getLayerConfigs() {
        return [
            { id: 'terrain-fill', type: 'fill', paint: { 'fill-color': ['get', 'color'], 'fill-opacity': 0.2 } },
            { id: 'terrain-outline', type: 'line', paint: { 'line-color': ['get', 'color'], 'line-width': 1, 'line-opacity': 0.4 } },
        ];
    }

    async fetchData() {
        try {
            const r = await fetch(`${this._apiBase}/api/terrain/layer`);
            if (!r.ok) return null;
            return await r.json();
        } catch { return null; }
    }
}
