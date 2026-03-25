// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/layout-manager.js — LayoutManager
// Run: node tests/web/test_layout_manager.mjs

import { LayoutManager } from '../../web/layout-manager.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

console.log('=== LayoutManager Tests ===');

test('constructor with no builtins', () => {
    const lm = new LayoutManager();
    assert.deepStrictEqual(lm.listAll(), []);
});

test('constructor with builtins shows them in listAll', () => {
    const lm = new LayoutManager({
        'default': { panels: { status: { x: 0, y: 0, w: 300, h: 200, visible: true } } },
    });
    const list = lm.listAll();
    assert.strictEqual(list.length, 1);
    assert.strictEqual(list[0].name, 'default');
    assert.strictEqual(list[0].builtin, true);
});

test('apply returns builtin layout', () => {
    const panels = { status: { x: 0, y: 0, w: 300, h: 200, visible: true } };
    const lm = new LayoutManager({ 'default': { panels } });
    const layout = lm.apply('default');
    assert.deepStrictEqual(layout.panels, panels);
});

test('apply returns null for missing layout', () => {
    const lm = new LayoutManager();
    assert.strictEqual(lm.apply('nope'), null);
});

test('saveCurrent + apply returns custom layout', () => {
    const lm = new LayoutManager();
    const panels = { map: { x: 10, y: 20, w: 400, h: 300, visible: true } };
    lm.saveCurrent('my-layout', panels);
    const layout = lm.apply('my-layout');
    assert.deepStrictEqual(layout.panels, panels);
    assert.strictEqual(typeof layout.savedAt, 'number');
});

test('custom overrides builtin with same name', () => {
    const lm = new LayoutManager({
        'shared': { panels: { a: { x: 0 } } },
    });
    lm.saveCurrent('shared', { b: { x: 1 } });
    const layout = lm.apply('shared');
    assert.deepStrictEqual(layout.panels, { b: { x: 1 } });
});

test('delete removes custom layout', () => {
    const lm = new LayoutManager();
    lm.saveCurrent('temp', { a: {} });
    assert.strictEqual(lm.delete('temp'), true);
    assert.strictEqual(lm.apply('temp'), null);
});

test('delete returns false for missing or builtin', () => {
    const lm = new LayoutManager({ 'builtin': { panels: {} } });
    assert.strictEqual(lm.delete('builtin'), false);
    assert.strictEqual(lm.delete('nonexistent'), false);
});

test('listAll includes both builtin and custom', () => {
    const lm = new LayoutManager({ 'alpha': { panels: {} } });
    lm.saveCurrent('beta', {});
    const list = lm.listAll();
    assert.strictEqual(list.length, 2);
    const names = list.map(l => l.name).sort();
    assert.deepStrictEqual(names, ['alpha', 'beta']);
    const alpha = list.find(l => l.name === 'alpha');
    const beta = list.find(l => l.name === 'beta');
    assert.strictEqual(alpha.builtin, true);
    assert.strictEqual(beta.builtin, false);
});

test('exportJSON returns valid JSON', () => {
    const lm = new LayoutManager();
    lm.saveCurrent('export-test', { panel1: { x: 5, y: 10 } });
    const json = lm.exportJSON('export-test');
    assert.notStrictEqual(json, null);
    const parsed = JSON.parse(json);
    assert.strictEqual(parsed.name, 'export-test');
    assert.deepStrictEqual(parsed.panels, { panel1: { x: 5, y: 10 } });
});

test('exportJSON returns null for missing layout', () => {
    const lm = new LayoutManager();
    assert.strictEqual(lm.exportJSON('nope'), null);
});

test('importJSON adds layout', () => {
    const lm = new LayoutManager();
    const json = JSON.stringify({ name: 'imported', panels: { p: { x: 1 } } });
    const name = lm.importJSON(json);
    assert.strictEqual(name, 'imported');
    const layout = lm.apply('imported');
    assert.deepStrictEqual(layout.panels, { p: { x: 1 } });
});

test('importJSON returns null for bad JSON', () => {
    const lm = new LayoutManager();
    assert.strictEqual(lm.importJSON('not json'), null);
});

test('importJSON returns null if missing name or panels', () => {
    const lm = new LayoutManager();
    assert.strictEqual(lm.importJSON('{"name":"x"}'), null);
    assert.strictEqual(lm.importJSON('{"panels":{}}'), null);
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
