// Tests for tritium-lib/web/map/data-provider.js
// Run: node tests/web/map/test_data_provider.mjs

import { MapDataProvider, MapDataProviderRegistry } from '../../../web/map/data-provider.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

console.log('=== MapDataProvider Tests ===');

test('base provider has defaults', () => {
    assert.strictEqual(MapDataProvider.providerId, 'generic');
    assert.strictEqual(MapDataProvider.category, 'overlay');
});

test('base getSourceConfig returns empty GeoJSON', () => {
    const p = new MapDataProvider();
    const src = p.getSourceConfig();
    assert.strictEqual(src.type, 'geojson');
    assert.strictEqual(src.data.features.length, 0);
});

test('base getLayerConfigs returns empty array', () => {
    const p = new MapDataProvider();
    assert.deepStrictEqual(p.getLayerConfigs(), []);
});

// Custom provider
class TestProvider extends MapDataProvider {
    static providerId = 'test-layer';
    static label = 'Test Layer';
    static category = 'test';
    static defaultVisible = true;

    getSourceConfig() {
        return { type: 'geojson', data: { type: 'FeatureCollection', features: [
            { type: 'Feature', geometry: { type: 'Point', coordinates: [0, 0] }, properties: {} }
        ]}};
    }
    getLayerConfigs() {
        return [{ id: 'test-circle', type: 'circle', paint: { 'circle-color': '#f00' } }];
    }
}

test('custom provider overrides', () => {
    const p = new TestProvider();
    assert.strictEqual(p.constructor.providerId, 'test-layer');
    assert.strictEqual(p.getSourceConfig().data.features.length, 1);
    assert.strictEqual(p.getLayerConfigs().length, 1);
});

// Registry
test('registry register and all', () => {
    const reg = new MapDataProviderRegistry();
    reg.register(new TestProvider());
    const all = reg.all();
    assert.strictEqual(all.length, 1);
    assert.strictEqual(all[0].providerId, 'test-layer');
    assert.strictEqual(all[0].active, false);
});

test('registry isActive starts false', () => {
    const reg = new MapDataProviderRegistry();
    reg.register(new TestProvider());
    assert.strictEqual(reg.isActive('test-layer'), false);
});

test('registry byCategory filters', () => {
    const reg = new MapDataProviderRegistry();
    reg.register(new TestProvider());
    class OtherProvider extends MapDataProvider {
        static providerId = 'other';
        static category = 'other-cat';
    }
    reg.register(new OtherProvider());
    const testCat = reg.byCategory('test');
    assert.strictEqual(testCat.length, 1);
    assert.strictEqual(testCat[0].providerId, 'test-layer');
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
