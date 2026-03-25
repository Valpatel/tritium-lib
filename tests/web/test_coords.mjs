// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/map/coords.js — MapCoords + geometry helpers
// Run: node tests/web/test_coords.mjs

import { MapCoords, buildFovConePolygon, buildCirclePolygon, haversineDistance } from '../../web/map/coords.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

function approxEqual(a, b, tolerance = 0.01) {
    return Math.abs(a - b) < tolerance;
}

console.log('=== MapCoords Tests ===');

// ── Constructor / setReference ────────────────────────────────────

test('constructor initializes to zero reference', () => {
    const coords = new MapCoords();
    assert.strictEqual(coords.refLat, 0);
    assert.strictEqual(coords.refLng, 0);
    assert.strictEqual(coords.hasReference, false);
});

test('setReference stores coordinates and computes cosLat', () => {
    const coords = new MapCoords();
    coords.setReference(37.7749, -122.4194);
    assert.strictEqual(coords.refLat, 37.7749);
    assert.strictEqual(coords.refLng, -122.4194);
    assert.ok(coords._cosLat > 0 && coords._cosLat < 1, 'cosLat should be between 0 and 1 for mid-latitudes');
    assert.strictEqual(coords.hasReference, true);
});

test('hasReference is true after setting non-zero reference', () => {
    const coords = new MapCoords();
    coords.setReference(0.001, 0);
    assert.strictEqual(coords.hasReference, true);
});

// ── gameToLngLat / lngLatToGame round-trip ────────────────────────

test('gameToLngLat at origin returns reference point', () => {
    const coords = new MapCoords();
    coords.setReference(37.7749, -122.4194);
    const [lng, lat] = coords.gameToLngLat(0, 0);
    assert.ok(approxEqual(lng, -122.4194, 0.0001), `lng: expected ~-122.4194, got ${lng}`);
    assert.ok(approxEqual(lat, 37.7749, 0.0001), `lat: expected ~37.7749, got ${lat}`);
});

test('gameToLngLat 1000m east shifts longitude', () => {
    const coords = new MapCoords();
    coords.setReference(37.7749, -122.4194);
    const [lng, lat] = coords.gameToLngLat(1000, 0);
    assert.ok(lng > -122.4194, 'eastward should increase longitude');
    assert.ok(approxEqual(lat, 37.7749, 0.0001), 'latitude should not change');
});

test('gameToLngLat 1000m north shifts latitude', () => {
    const coords = new MapCoords();
    coords.setReference(37.7749, -122.4194);
    const [lng, lat] = coords.gameToLngLat(0, 1000);
    assert.ok(approxEqual(lng, -122.4194, 0.0001), 'longitude should not change');
    assert.ok(lat > 37.7749, 'northward should increase latitude');
});

test('round-trip: game -> lnglat -> game preserves position', () => {
    const coords = new MapCoords();
    coords.setReference(37.7749, -122.4194);
    const gx = 500, gy = 300;
    const [lng, lat] = coords.gameToLngLat(gx, gy);
    const result = coords.lngLatToGame(lng, lat);
    assert.ok(approxEqual(result.x, gx, 0.1), `x: expected ~${gx}, got ${result.x}`);
    assert.ok(approxEqual(result.y, gy, 0.1), `y: expected ~${gy}, got ${result.y}`);
});

test('lngLatToGame converts reference point to origin', () => {
    const coords = new MapCoords();
    coords.setReference(37.7749, -122.4194);
    const result = coords.lngLatToGame(-122.4194, 37.7749);
    assert.ok(approxEqual(result.x, 0, 0.01));
    assert.ok(approxEqual(result.y, 0, 0.01));
});

// ── gameToMercator ────────────────────────────────────────────────

test('gameToMercator returns coordinate object', () => {
    const coords = new MapCoords();
    const m = coords.gameToMercator(100, 200, 50);
    assert.strictEqual(m.x, 100);
    assert.strictEqual(m.y, 200);
    assert.strictEqual(m.z, 50);
    assert.strictEqual(typeof m.meterInMercatorCoordinateUnits, 'function');
    assert.strictEqual(m.meterInMercatorCoordinateUnits(), 1.0);
});

// ── haversineDistance ─────────────────────────────────────────────

test('haversineDistance: same point returns zero', () => {
    const d = haversineDistance(-122.4194, 37.7749, -122.4194, 37.7749);
    assert.ok(approxEqual(d, 0, 0.01), `expected 0, got ${d}`);
});

test('haversineDistance: known distance SF to NYC ~4130 km', () => {
    // SF: 37.7749, -122.4194  NYC: 40.7128, -74.0060
    const d = haversineDistance(-122.4194, 37.7749, -74.0060, 40.7128);
    const km = d / 1000;
    assert.ok(km > 4000 && km < 4300, `expected ~4130 km, got ${km} km`);
});

test('haversineDistance: short distance ~100m', () => {
    // Two points approximately 100m apart at equator
    // 1 degree of longitude at equator ~ 111,320m, so 100m ~ 0.000898 degrees
    const d = haversineDistance(0, 0, 0.000898, 0);
    assert.ok(d > 90 && d < 110, `expected ~100m, got ${d}m`);
});

test('haversineDistance: symmetric', () => {
    const d1 = haversineDistance(-122, 37, -74, 40);
    const d2 = haversineDistance(-74, 40, -122, 37);
    assert.ok(approxEqual(d1, d2, 0.01), 'distance should be symmetric');
});

// ── buildFovConePolygon ───────────────────────────────────────────

test('buildFovConePolygon returns closed polygon', () => {
    const coords = buildFovConePolygon(-122, 37, 0, 90, 100, 8);
    assert.ok(coords.length >= 3, 'should have at least 3 points');
    // First point is center, last point should close back to center
    const first = coords[0];
    const last = coords[coords.length - 1];
    assert.ok(approxEqual(first[0], last[0], 0.0001), 'polygon should close');
    assert.ok(approxEqual(first[1], last[1], 0.0001), 'polygon should close');
});

test('buildFovConePolygon center point is the origin', () => {
    const coords = buildFovConePolygon(-122.5, 37.5, 90, 60, 200);
    assert.ok(approxEqual(coords[0][0], -122.5, 0.0001));
    assert.ok(approxEqual(coords[0][1], 37.5, 0.0001));
});

test('buildFovConePolygon arc points extend to range', () => {
    const range = 500; // meters
    const coords = buildFovConePolygon(-122, 37, 0, 90, range, 12);
    // Arc points (indices 1 to length-2) should be roughly 'range' meters from center
    const arcPt = coords[Math.floor(coords.length / 2)];
    const d = haversineDistance(-122, 37, arcPt[0], arcPt[1]);
    assert.ok(d > range * 0.9 && d < range * 1.1, `arc point should be ~${range}m, got ${d}m`);
});

// ── buildCirclePolygon ────────────────────────────────────────────

test('buildCirclePolygon returns closed polygon', () => {
    const coords = buildCirclePolygon(-122, 37, 100, 16);
    const first = coords[0];
    const last = coords[coords.length - 1];
    assert.ok(approxEqual(first[0], last[0], 0.0001), 'circle polygon should close');
});

test('buildCirclePolygon all points are equidistant from center', () => {
    const radius = 200;
    const coords = buildCirclePolygon(-122, 37, radius, 16);
    // Skip first (center) and last (closing center) points
    for (let i = 1; i < coords.length - 1; i++) {
        const d = haversineDistance(-122, 37, coords[i][0], coords[i][1]);
        assert.ok(d > radius * 0.9 && d < radius * 1.1,
            `point ${i}: expected ~${radius}m, got ${d}m`);
    }
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
