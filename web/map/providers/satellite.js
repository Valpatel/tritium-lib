// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

import { MapDataProvider } from '../data-provider.js';

/** Esri World Imagery satellite tiles. */
export class EsriSatelliteProvider extends MapDataProvider {
    static providerId = 'esri-satellite';
    static label = 'Satellite (Esri)';
    static category = 'basemap';
    static icon = 'S';
    static defaultVisible = true;
    static attribution = '© Esri, Maxar, Earthstar';

    getSourceConfig() {
        return {
            type: 'raster',
            tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
            tileSize: 256,
            maxzoom: 19,
        };
    }

    getLayerConfigs() {
        return [{ id: 'satellite-tiles', type: 'raster', paint: { 'raster-opacity': 1 } }];
    }
}

/** OpenStreetMap standard tiles. */
export class OSMTilesProvider extends MapDataProvider {
    static providerId = 'osm-tiles';
    static label = 'OpenStreetMap';
    static category = 'basemap';
    static icon = 'O';
    static attribution = '© OpenStreetMap contributors';

    getSourceConfig() {
        return {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            maxzoom: 19,
        };
    }

    getLayerConfigs() {
        return [{ id: 'osm-tiles', type: 'raster', paint: { 'raster-opacity': 1 } }];
    }
}
