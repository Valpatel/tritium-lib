// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * MapCoords — coordinate transforms for tactical map rendering.
 *
 * Converts between three coordinate systems:
 *   - Game coords: meters from a reference point (x=east, y=north)
 *   - LngLat: [longitude, latitude] in WGS84 degrees
 *   - Mercator: Web Mercator projection coordinates
 *
 * Also provides geometry helpers: FOV cone polygon, circle polygon,
 * bearing/distance calculations.
 *
 * Usage:
 *   const coords = new MapCoords();
 *   coords.setReference(37.7749, -122.4194);
 *   const [lng, lat] = coords.gameToLngLat(100, 200);
 *   const { x, y } = coords.lngLatToGame(-122.42, 37.78);
 */

const EARTH_RADIUS = 6378137; // WGS84 equatorial radius in meters
const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;
const METERS_PER_DEG_LAT = 111320; // approximate

export class MapCoords {
    constructor() {
        this.refLat = 0;
        this.refLng = 0;
        this._cosLat = 1;
    }

    /**
     * Set the geographic reference point. All game coordinates are meters
     * relative to this point (x=east, y=north).
     */
    setReference(lat, lng) {
        this.refLat = lat;
        this.refLng = lng;
        this._cosLat = Math.cos(lat * DEG_TO_RAD);
    }

    /** Game meters → [longitude, latitude] */
    gameToLngLat(gx, gy) {
        const dLng = gx / (EARTH_RADIUS * this._cosLat) * RAD_TO_DEG;
        const dLat = gy / EARTH_RADIUS * RAD_TO_DEG;
        return [this.refLng + dLng, this.refLat + dLat];
    }

    /** [longitude, latitude] → { x, y } in game meters */
    lngLatToGame(lng, lat) {
        const gx = (lng - this.refLng) * DEG_TO_RAD * EARTH_RADIUS * this._cosLat;
        const gy = (lat - this.refLat) * DEG_TO_RAD * EARTH_RADIUS;
        return { x: gx, y: gy };
    }

    /**
     * Game meters → Three.js position object.
     * In the MapLibre custom layer, objects are positioned in game meters
     * (x=east, y=north, z=up).
     */
    gameToMercator(gx, gy, altMeters = 0) {
        return {
            x: gx,
            y: gy,
            z: altMeters,
            meterInMercatorCoordinateUnits() { return 1.0; },
        };
    }

    /** Whether a reference point has been set (non-zero). */
    get hasReference() {
        return this.refLat !== 0 || this.refLng !== 0;
    }
}

// ── Geometry helpers ─────────────────────────────────────────────

/**
 * Build a GeoJSON polygon for a FOV cone (sector/wedge shape).
 *
 * @param {number} lng — center longitude
 * @param {number} lat — center latitude
 * @param {number} heading — degrees clockwise from north
 * @param {number} fovAngle — total FOV in degrees
 * @param {number} rangeMeters — how far the cone extends
 * @param {number} [steps=24] — arc resolution
 * @returns {Array<[number, number]>} — GeoJSON polygon coordinate ring
 */
export function buildFovConePolygon(lng, lat, heading, fovAngle, rangeMeters, steps = 24) {
    const halfFov = fovAngle / 2;
    const startBearing = heading - halfFov;
    const endBearing = heading + halfFov;
    const coords = [[lng, lat]];
    const latRad = lat * DEG_TO_RAD;
    const cosLat = Math.cos(latRad);

    for (let i = 0; i <= steps; i++) {
        const bearing = startBearing + (endBearing - startBearing) * (i / steps);
        const bearingRad = bearing * DEG_TO_RAD;
        const dLat = (rangeMeters * Math.cos(bearingRad)) / METERS_PER_DEG_LAT;
        const dLng = (rangeMeters * Math.sin(bearingRad)) / (METERS_PER_DEG_LAT * cosLat);
        coords.push([lng + dLng, lat + dLat]);
    }
    coords.push([lng, lat]); // close polygon
    return coords;
}

/**
 * Build a GeoJSON circle polygon from center + radius.
 *
 * @param {number} lng — center longitude
 * @param {number} lat — center latitude
 * @param {number} radiusMeters — circle radius
 * @param {number} [steps=32] — number of vertices
 * @returns {Array<[number, number]>} — GeoJSON polygon coordinate ring
 */
export function buildCirclePolygon(lng, lat, radiusMeters, steps = 32) {
    return buildFovConePolygon(lng, lat, 0, 360, radiusMeters, steps);
}

/**
 * Haversine distance between two points in meters.
 */
export function haversineDistance(lng1, lat1, lng2, lat2) {
    const dLat = (lat2 - lat1) * DEG_TO_RAD;
    const dLng = (lng2 - lng1) * DEG_TO_RAD;
    const a = Math.sin(dLat / 2) ** 2 +
              Math.cos(lat1 * DEG_TO_RAD) * Math.cos(lat2 * DEG_TO_RAD) *
              Math.sin(dLng / 2) ** 2;
    return 2 * EARTH_RADIUS * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}
