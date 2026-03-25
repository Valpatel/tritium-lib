// Tests for tritium-lib/web/map/draw-tools.js
// Run: node tests/web/map/test_draw_tools.mjs

import { DrawTools } from '../../../web/map/draw-tools.js';
import assert from 'node:assert';

// Mock DOM globals
globalThis.document = {
    addEventListener() {},
    removeEventListener() {},
    createElement(tag) { return { style: {}, cssText: '', textContent: '', remove() {} }; },
};

// Mock maplibregl globally (DrawTools uses maplibregl.Marker)
globalThis.maplibregl = {
    Marker: class {
        constructor() { this._lngLat = [0,0]; this._el = { remove() {} }; }
        setLngLat(ll) { this._lngLat = ll; return this; }
        addTo() { return this; }
        remove() {}
        getElement() { return this._el; }
    },
};

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

function mockMap() {
    const sources = new Map();
    const layers = new Map();
    const handlers = {};
    return {
        addSource(id, cfg) { sources.set(id, { ...cfg }); },
        getSource(id) { const s = sources.get(id); return s ? { setData(d) { s.data = d; } } : null; },
        addLayer(cfg) { layers.set(cfg.id, cfg); },
        getLayer(id) { return layers.get(id); },
        on(event, fn) { handlers[event] = fn; },
        off(event, fn) { if (handlers[event] === fn) delete handlers[event]; },
        getCanvas() { return { style: {} }; },
        _fire(event, data) { if (handlers[event]) handlers[event](data); },
    };
}

console.log('=== DrawTools Tests ===');

test('not active initially', () => {
    const dt = new DrawTools(mockMap());
    assert.strictEqual(dt.active, false);
    assert.strictEqual(dt.mode, null);
});

test('startPolygon enters polygon mode', () => {
    const dt = new DrawTools(mockMap());
    dt.startPolygon({ onFinish: () => {} });
    assert.strictEqual(dt.active, true);
    assert.strictEqual(dt.mode, 'polygon');
});

test('startPolyline enters polyline mode', () => {
    const dt = new DrawTools(mockMap());
    dt.startPolyline({ onFinish: () => {} });
    assert.strictEqual(dt.mode, 'polyline');
});

test('cancel exits mode', () => {
    let cancelled = false;
    const dt = new DrawTools(mockMap());
    dt.startPolygon({ onCancel: () => { cancelled = true; } });
    dt.cancel();
    assert.strictEqual(dt.active, false);
    assert.strictEqual(cancelled, true);
});

test('vertexCount starts at 0', () => {
    const dt = new DrawTools(mockMap());
    dt.startPolygon({ onFinish: () => {} });
    assert.strictEqual(dt.vertexCount, 0);
});

test('starting new draw cancels previous', () => {
    let cancelCount = 0;
    const dt = new DrawTools(mockMap());
    dt.startPolygon({ onCancel: () => { cancelCount++; } });
    dt.startPolyline({ onFinish: () => {} }); // should cancel polygon
    assert.strictEqual(cancelCount, 1);
    assert.strictEqual(dt.mode, 'polyline');
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
