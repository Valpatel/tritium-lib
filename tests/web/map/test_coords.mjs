// Tests for tritium-lib/web/map/coords.js
// Run: node tests/web/map/test_coords.mjs

import { MapCoords, buildFovConePolygon, buildCirclePolygon, haversineDistance } from '../../../web/map/coords.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

console.log('=== MapCoords Tests ===');

// Constructor
test('default reference is 0,0', () => {
    const c = new MapCoords();
    assert.strictEqual(c.refLat, 0);
    assert.strictEqual(c.refLng, 0);
});

test('hasReference false initially', () => {
    const c = new MapCoords();
    assert.strictEqual(c.hasReference, false);
});

// setReference
test('setReference stores lat/lng', () => {
    const c = new MapCoords();
    c.setReference(37.7749, -122.4194);
    assert.strictEqual(c.refLat, 37.7749);
    assert.strictEqual(c.refLng, -122.4194);
    assert.strictEqual(c.hasReference, true);
});

// gameToLngLat
test('gameToLngLat(0,0) returns reference point', () => {
    const c = new MapCoords();
    c.setReference(37.7749, -122.4194);
    const [lng, lat] = c.gameToLngLat(0, 0);
    assert(Math.abs(lng - (-122.4194)) < 0.0001);
    assert(Math.abs(lat - 37.7749) < 0.0001);
});

test('gameToLngLat positive X goes east (higher lng)', () => {
    const c = new MapCoords();
    c.setReference(37.7749, -122.4194);
    const [lng] = c.gameToLngLat(100, 0); // 100m east
    assert(lng > -122.4194, `Expected ${lng} > -122.4194`);
});

test('gameToLngLat positive Y goes north (higher lat)', () => {
    const c = new MapCoords();
    c.setReference(37.7749, -122.4194);
    const [, lat] = c.gameToLngLat(0, 100); // 100m north
    assert(lat > 37.7749, `Expected ${lat} > 37.7749`);
});

// lngLatToGame
test('lngLatToGame roundtrip', () => {
    const c = new MapCoords();
    c.setReference(37.7749, -122.4194);
    const [lng, lat] = c.gameToLngLat(50, 75);
    const { x, y } = c.lngLatToGame(lng, lat);
    assert(Math.abs(x - 50) < 0.01, `x=${x}, expected ~50`);
    assert(Math.abs(y - 75) < 0.01, `y=${y}, expected ~75`);
});

test('lngLatToGame reference point returns 0,0', () => {
    const c = new MapCoords();
    c.setReference(37.7749, -122.4194);
    const { x, y } = c.lngLatToGame(-122.4194, 37.7749);
    assert(Math.abs(x) < 0.01);
    assert(Math.abs(y) < 0.01);
});

// gameToMercator
test('gameToMercator returns position object', () => {
    const c = new MapCoords();
    const pos = c.gameToMercator(10, 20, 5);
    assert.strictEqual(pos.x, 10);
    assert.strictEqual(pos.y, 20);
    assert.strictEqual(pos.z, 5);
    assert.strictEqual(typeof pos.meterInMercatorCoordinateUnits, 'function');
    assert.strictEqual(pos.meterInMercatorCoordinateUnits(), 1.0);
});

// buildFovConePolygon
test('FOV cone returns closed polygon', () => {
    const coords = buildFovConePolygon(-122.42, 37.77, 0, 90, 30);
    assert(coords.length > 10, `Expected >10 vertices, got ${coords.length}`);
    // First and last should be the center point
    assert.deepStrictEqual(coords[0], [-122.42, 37.77]);
    assert.deepStrictEqual(coords[coords.length - 1], [-122.42, 37.77]);
});

test('FOV cone points extend north for heading=0', () => {
    const coords = buildFovConePolygon(-122.42, 37.77, 0, 90, 100);
    // Middle arc point should be north (higher lat)
    const mid = coords[Math.floor(coords.length / 2)];
    assert(mid[1] > 37.77, `Expected mid lat ${mid[1]} > 37.77`);
});

// buildCirclePolygon
test('circle polygon is closed and has many vertices', () => {
    const coords = buildCirclePolygon(-122.42, 37.77, 50);
    assert(coords.length > 20);
    assert.deepStrictEqual(coords[0], [-122.42, 37.77]);
});

// haversineDistance
test('haversine distance is 0 for same point', () => {
    const d = haversineDistance(-122.42, 37.77, -122.42, 37.77);
    assert(Math.abs(d) < 0.01);
});

test('haversine ~111km per degree latitude', () => {
    const d = haversineDistance(0, 0, 0, 1);
    assert(Math.abs(d - 111320) < 500, `Expected ~111320m, got ${d}`);
});

test('haversine San Francisco to Oakland ~13km', () => {
    const d = haversineDistance(-122.4194, 37.7749, -122.2711, 37.8044);
    assert(d > 10000 && d < 20000, `Expected 10-20km, got ${d}m`);
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
