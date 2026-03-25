// Tests for tritium-lib/web/map/layer-manager.js
// Run: node tests/web/map/test_layer_manager.mjs

import { GeoJSONLayerManager } from '../../../web/map/layer-manager.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

console.log('=== GeoJSONLayerManager Tests ===');

// Mock MapLibre map
function mockMap() {
    const sources = new Map();
    const layers = new Map();
    return {
        addSource(id, config) { sources.set(id, { ...config }); },
        getSource(id) {
            const s = sources.get(id);
            return s ? { setData(d) { s.data = d; } } : null;
        },
        removeSource(id) { sources.delete(id); },
        addLayer(config) { layers.set(config.id, { ...config, _layout: { visibility: 'visible' } }); },
        getLayer(id) { return layers.get(id) || null; },
        removeLayer(id) { layers.delete(id); },
        setLayoutProperty(id, prop, val) {
            const l = layers.get(id);
            if (l) l._layout[prop] = val;
        },
        _sources: sources,
        _layers: layers,
    };
}

const emptyGeo = { type: 'FeatureCollection', features: [] };
const oneFeature = { type: 'FeatureCollection', features: [
    { type: 'Feature', geometry: { type: 'Point', coordinates: [0, 0] }, properties: { id: 'a' } }
]};
const twoFeatures = { type: 'FeatureCollection', features: [
    { type: 'Feature', geometry: { type: 'Point', coordinates: [0, 0] }, properties: { id: 'a' } },
    { type: 'Feature', geometry: { type: 'Point', coordinates: [1, 1] }, properties: { id: 'b' } },
]};

test('constructor accepts map', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    assert(lm);
});

test('update creates source and layers on first call', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    const result = lm.update('test', oneFeature, [
        { id: 'test-circle', type: 'circle', paint: { 'circle-color': '#fff' } },
    ]);
    assert.strictEqual(result, true);
    assert(m._sources.has('test'));
    assert(m._layers.has('test-circle'));
});

test('update returns false when data unchanged', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    lm.update('test', oneFeature, [{ id: 'test-c', type: 'circle', paint: {} }]);
    const result = lm.update('test', oneFeature, []);
    assert.strictEqual(result, false);
});

test('update returns true when data changes', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    lm.update('test', oneFeature, [{ id: 'test-c', type: 'circle', paint: {} }]);
    const result = lm.update('test', twoFeatures, []);
    assert.strictEqual(result, true);
});

test('setVisibility hides all layers in group', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    lm.update('test', oneFeature, [
        { id: 'test-fill', type: 'fill', paint: {} },
        { id: 'test-line', type: 'line', paint: {} },
    ]);
    lm.setVisibility('test', false);
    assert.strictEqual(m._layers.get('test-fill')._layout.visibility, 'none');
    assert.strictEqual(m._layers.get('test-line')._layout.visibility, 'none');
});

test('toggle flips visibility', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    lm.update('test', emptyGeo, [{ id: 'test-c', type: 'circle', paint: {} }]);
    assert.strictEqual(lm.isVisible('test'), true);
    lm.toggle('test');
    assert.strictEqual(lm.isVisible('test'), false);
    lm.toggle('test');
    assert.strictEqual(lm.isVisible('test'), true);
});

test('clear empties source data', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    lm.update('test', oneFeature, [{ id: 'test-c', type: 'circle', paint: {} }]);
    lm.clear('test');
    // Source still exists but data should be empty
    assert(m._sources.has('test'));
});

test('remove deletes source and layers', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    lm.update('test', oneFeature, [
        { id: 'test-fill', type: 'fill', paint: {} },
        { id: 'test-line', type: 'line', paint: {} },
    ]);
    lm.remove('test');
    assert(!m._sources.has('test'));
    assert(!m._layers.has('test-fill'));
    assert(!m._layers.has('test-line'));
    assert.strictEqual(lm.sourceIds.length, 0);
});

test('destroy removes all sources', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    lm.update('a', emptyGeo, [{ id: 'a-c', type: 'circle', paint: {} }]);
    lm.update('b', emptyGeo, [{ id: 'b-c', type: 'circle', paint: {} }]);
    assert.strictEqual(lm.sourceIds.length, 2);
    lm.destroy();
    assert.strictEqual(lm.sourceIds.length, 0);
    assert.strictEqual(m._sources.size, 0);
});

test('sourceIds returns all managed sources', () => {
    const m = mockMap();
    const lm = new GeoJSONLayerManager(m);
    lm.update('alpha', emptyGeo, [{ id: 'a-c', type: 'circle', paint: {} }]);
    lm.update('beta', emptyGeo, [{ id: 'b-c', type: 'circle', paint: {} }]);
    const ids = lm.sourceIds.sort();
    assert.deepStrictEqual(ids, ['alpha', 'beta']);
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
