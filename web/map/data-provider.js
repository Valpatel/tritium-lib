// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * MapDataProvider — base class for extensible map layer data sources.
 *
 * Any data source that provides geographic features for the map extends
 * this class. Built-in providers: satellite tiles, OSM roads, city water,
 * terrain segmentation. Addons can register custom providers at runtime.
 *
 * A provider:
 *   1. Declares its layer type (raster tiles, vector tiles, GeoJSON, etc.)
 *   2. Returns a MapLibre source + layer configuration
 *   3. Optionally fetches/refreshes data on a schedule
 *   4. Can be toggled on/off by the user
 *
 * Usage:
 *   class CityWaterProvider extends MapDataProvider {
 *       static providerId = 'city-water';
 *       static label = 'City Water Infrastructure';
 *       static category = 'municipal';
 *       getSourceConfig() { return { type: 'geojson', data: '/api/gis/water' }; }
 *       getLayerConfigs() { return [{ id: 'water-pipes', type: 'line', ... }]; }
 *   }
 *   providerRegistry.register(new CityWaterProvider());
 */

export class MapDataProvider {
    // --- Override these in subclasses ---

    /** Unique provider identifier. */
    static providerId = 'generic';

    /** Human-readable label for the layer menu. */
    static label = 'Generic Layer';

    /** Category for grouping in the UI (e.g., 'satellite', 'infrastructure', 'intel'). */
    static category = 'overlay';

    /** Icon character for menus. */
    static icon = '?';

    /** Default visibility when first loaded. */
    static defaultVisible = false;

    /** Whether this provider needs periodic refresh (e.g., live data). */
    static refreshInterval = 0; // ms, 0 = no auto-refresh

    /** Attribution text for the map. */
    static attribution = '';

    // --- Methods to override ---

    /**
     * Return MapLibre source configuration.
     * @returns {Object} — { type: 'geojson'|'raster'|'vector', data?, tiles?, ... }
     */
    getSourceConfig() {
        return { type: 'geojson', data: { type: 'FeatureCollection', features: [] } };
    }

    /**
     * Return MapLibre layer configurations for this source.
     * @returns {Array<Object>} — [{ id, type, paint, layout?, filter?, minzoom?, maxzoom? }]
     */
    getLayerConfigs() {
        return [];
    }

    /**
     * Fetch fresh data. Called on init and on refresh interval.
     * Override to load from API, file, or service.
     * @returns {Object|null} — GeoJSON FeatureCollection, or null for non-GeoJSON sources
     */
    async fetchData() {
        return null;
    }

    /**
     * Called when the provider is added to the map.
     * @param {maplibregl.Map} map
     */
    onAdd(map) {}

    /**
     * Called when the provider is removed from the map.
     */
    onRemove() {}
}

/**
 * MapDataProviderRegistry — manages all map data providers.
 */
export class MapDataProviderRegistry {
    constructor() {
        this._providers = new Map(); // providerId → { instance, active, refreshTimer }
    }

    /**
     * Register a data provider.
     * @param {MapDataProvider} provider — provider instance
     */
    register(provider) {
        const id = provider.constructor.providerId;
        this._providers.set(id, { instance: provider, active: false, refreshTimer: null });
    }

    /**
     * Activate a provider on the map.
     * @param {string} providerId
     * @param {maplibregl.Map} map
     * @param {import('./layer-manager.js').GeoJSONLayerManager} layerManager
     */
    async activate(providerId, map, layerManager) {
        const entry = this._providers.get(providerId);
        if (!entry || entry.active) return;

        const provider = entry.instance;
        const Cls = provider.constructor;

        // Add source config
        const srcConfig = provider.getSourceConfig();
        const layerConfigs = provider.getLayerConfigs();

        // Fetch initial data if GeoJSON
        if (srcConfig.type === 'geojson') {
            const data = await provider.fetchData();
            if (data) srcConfig.data = data;
        }

        // Use layer manager for GeoJSON, direct addSource for raster/vector
        if (srcConfig.type === 'geojson') {
            layerManager.update(Cls.providerId, srcConfig.data, layerConfigs);
        } else {
            if (!map.getSource(Cls.providerId)) {
                map.addSource(Cls.providerId, srcConfig);
                for (const cfg of layerConfigs) {
                    map.addLayer({ ...cfg, source: Cls.providerId });
                }
            }
        }

        provider.onAdd(map);
        entry.active = true;

        // Auto-refresh
        if (Cls.refreshInterval > 0 && srcConfig.type === 'geojson') {
            entry.refreshTimer = setInterval(async () => {
                const data = await provider.fetchData();
                if (data) layerManager.update(Cls.providerId, data, []);
            }, Cls.refreshInterval);
        }
    }

    /**
     * Deactivate a provider.
     * @param {string} providerId
     * @param {import('./layer-manager.js').GeoJSONLayerManager} layerManager
     */
    deactivate(providerId, layerManager) {
        const entry = this._providers.get(providerId);
        if (!entry || !entry.active) return;

        if (entry.refreshTimer) {
            clearInterval(entry.refreshTimer);
            entry.refreshTimer = null;
        }
        entry.instance.onRemove();
        layerManager.remove(providerId);
        entry.active = false;
    }

    /** Toggle a provider on/off. */
    async toggle(providerId, map, layerManager) {
        const entry = this._providers.get(providerId);
        if (!entry) return;
        if (entry.active) this.deactivate(providerId, layerManager);
        else await this.activate(providerId, map, layerManager);
    }

    /** Get all registered providers. */
    all() {
        return [...this._providers.entries()].map(([id, e]) => ({
            providerId: id,
            label: e.instance.constructor.label,
            category: e.instance.constructor.category,
            icon: e.instance.constructor.icon,
            active: e.active,
        }));
    }

    /** Get providers by category. */
    byCategory(category) {
        return this.all().filter(p => p.category === category);
    }

    /** Check if a provider is active. */
    isActive(providerId) {
        return this._providers.get(providerId)?.active || false;
    }

    /** Deactivate all and clean up timers. */
    destroy(layerManager) {
        for (const id of this._providers.keys()) {
            this.deactivate(id, layerManager);
        }
    }
}

// Singleton registry
export const providerRegistry = new MapDataProviderRegistry();
