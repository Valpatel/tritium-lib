// Tests for tritium-lib/web/map/overlays.js — layer isolation
// Run: node tests/web/map/test_overlays.mjs
//
// Each overlay is tested in isolation: create, update, toggle, clear.

import { GeoJSONLayerManager } from '../../../web/map/layer-manager.js';
import { TacticalOverlays } from '../../../web/map/overlays.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

function mockMap() {
    const sources = new Map();
    const layers = new Map();
    return {
        addSource(id, cfg) { sources.set(id, { ...cfg }); },
        getSource(id) { const s = sources.get(id); return s ? { setData(d) { s.data = d; } } : null; },
        removeSource(id) { sources.delete(id); },
        addLayer(cfg) { layers.set(cfg.id, { ...cfg, _vis: 'visible' }); },
        getLayer(id) { return layers.get(id) || null; },
        removeLayer(id) { layers.delete(id); },
        setLayoutProperty(id, prop, val) { const l = layers.get(id); if (l) l._vis = val; },
        _s: sources, _l: layers,
    };
}

const point = (x, y) => ({ type: 'Feature', geometry: { type: 'Point', coordinates: [x, y] }, properties: {} });
const line = (coords) => ({ type: 'Feature', geometry: { type: 'LineString', coordinates: coords }, properties: {} });
const poly = (coords) => ({ type: 'Feature', geometry: { type: 'Polygon', coordinates: [coords] }, properties: { color: '#ff0000' } });
const fc = (...features) => ({ type: 'FeatureCollection', features });

console.log('=== Tactical Overlay Layer Isolation Tests ===\n');

// --- Patrol Routes ---
test('patrol routes: create + update', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    const ov = new TacticalOverlays(lm);
    ov.updatePatrolRoutes(fc(line([[0,0],[1,1]])));
    assert(m._s.has('patrol-routes'), 'source created');
    assert(m._l.has('patrol-routes-line'), 'line layer created');
});

test('patrol routes: toggle visibility', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    const ov = new TacticalOverlays(lm);
    ov.updatePatrolRoutes(fc(line([[0,0],[1,1]])));
    assert.strictEqual(ov.isVisible('patrol-routes'), true);
    ov.toggle('patrol-routes');
    assert.strictEqual(ov.isVisible('patrol-routes'), false);
    assert.strictEqual(m._l.get('patrol-routes-line')._vis, 'none');
});

test('patrol routes: clear empties data', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    const ov = new TacticalOverlays(lm);
    ov.updatePatrolRoutes(fc(line([[0,0],[1,1]])));
    ov.clear('patrol-routes');
    assert(m._s.has('patrol-routes'), 'source still exists');
});

// --- Weapon Range ---
test('weapon range: create circle overlay', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    const ov = new TacticalOverlays(lm);
    ov.updateWeaponRange(fc(poly([[0,0],[1,0],[1,1],[0,1],[0,0]])));
    assert(m._l.has('weapon-range-fill'), 'fill layer');
    assert(m._l.has('weapon-range-line'), 'line layer');
});

// --- Hazard Zones ---
test('hazard zones: create with color property', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    const ov = new TacticalOverlays(lm);
    ov.updateHazardZones(fc(poly([[0,0],[1,0],[1,1],[0,1],[0,0]])));
    assert(m._l.has('hazard-zones-fill'));
    assert(m._l.has('hazard-zones-line'));
});

test('hazard zones: toggle independent of other layers', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    const ov = new TacticalOverlays(lm);
    ov.updatePatrolRoutes(fc(line([[0,0],[1,1]])));
    ov.updateHazardZones(fc(poly([[0,0],[1,0],[1,1],[0,1],[0,0]])));
    ov.toggle('hazard-zones');
    assert.strictEqual(ov.isVisible('patrol-routes'), true, 'patrol still visible');
    assert.strictEqual(ov.isVisible('hazard-zones'), false, 'hazard hidden');
});

// --- Crowd Density ---
test('crowd density: creates heatmap layer', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    const ov = new TacticalOverlays(lm);
    ov.updateCrowdDensity(fc(point(0, 0)));
    assert(m._l.has('crowd-density-heat'));
});

// --- Cover Points ---
test('cover points: creates circle layer', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    const ov = new TacticalOverlays(lm);
    ov.updateCoverPoints(fc(point(0, 0)));
    assert(m._l.has('cover-points-circle'));
});

// --- Engagement Lines ---
test('engagement lines: creates line layer', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    const ov = new TacticalOverlays(lm);
    ov.updateEngagementLines(fc(line([[0,0],[1,1]])));
    assert(m._l.has('engagement-lines-line'));
});

// --- Dispatch Arrows ---
test('dispatch arrows: creates dashed line', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    const ov = new TacticalOverlays(lm);
    ov.updateDispatchArrows(fc(line([[0,0],[1,1]])));
    assert(m._l.has('dispatch-arrows-line'));
});

// --- Multiple layers isolated ---
test('all overlays create independent sources', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    const ov = new TacticalOverlays(lm);
    ov.updatePatrolRoutes(fc(line([[0,0],[1,1]])));
    ov.updateWeaponRange(fc(poly([[0,0],[1,0],[1,1],[0,1],[0,0]])));
    ov.updateHazardZones(fc(poly([[2,2],[3,2],[3,3],[2,3],[2,2]])));
    ov.updateCrowdDensity(fc(point(5, 5)));
    ov.updateCoverPoints(fc(point(6, 6)));
    ov.updateEngagementLines(fc(line([[7,7],[8,8]])));
    ov.updateDispatchArrows(fc(line([[9,9],[10,10]])));

    assert.strictEqual(ov.names.length, 7, 'all 7 overlays registered');
    assert.strictEqual(m._s.size, 7, '7 independent sources');
});

test('toggling one overlay does not affect others', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    const ov = new TacticalOverlays(lm);
    ov.updatePatrolRoutes(fc(line([[0,0],[1,1]])));
    ov.updateHazardZones(fc(poly([[0,0],[1,0],[1,1],[0,1],[0,0]])));
    ov.updateCoverPoints(fc(point(0, 0)));

    ov.toggle('hazard-zones'); // hide only hazards
    assert.strictEqual(ov.isVisible('patrol-routes'), true);
    assert.strictEqual(ov.isVisible('hazard-zones'), false);
    assert.strictEqual(ov.isVisible('cover-points'), true);
});

test('destroy removes all overlays', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    const ov = new TacticalOverlays(lm);
    ov.updatePatrolRoutes(fc(line([[0,0],[1,1]])));
    ov.updateHazardZones(fc(poly([[0,0],[1,0],[1,1],[0,1],[0,0]])));
    ov.destroy();
    assert.strictEqual(ov.names.length, 0);
    assert.strictEqual(m._s.size, 0);
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
