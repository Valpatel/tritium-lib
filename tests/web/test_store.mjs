// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/store.js — ReactiveStore
// Run: node tests/web/test_store.mjs

import { ReactiveStore } from '../../web/store.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

console.log('=== ReactiveStore Tests ===');

test('set and get a simple value', () => {
    const store = new ReactiveStore();
    store.set('name', 'tritium');
    assert.strictEqual(store.get('name'), 'tritium');
});

test('get returns default for missing path', () => {
    const store = new ReactiveStore();
    assert.strictEqual(store.get('missing.path', 42), 42);
});

test('set creates nested objects', () => {
    const store = new ReactiveStore();
    store.set('game.score', 100);
    assert.strictEqual(store.get('game.score'), 100);
    assert.strictEqual(store.get('game.missing', 'nope'), 'nope');
});

test('on fires listener on set (sync, no RAF)', () => {
    const store = new ReactiveStore();
    let received = null;
    store.on('game.phase', (val, old) => { received = { val, old }; });
    store.set('game.phase', 'combat');
    // In Node.js there's no RAF, so notification fires synchronously
    assert.deepStrictEqual(received, { val: 'combat', old: undefined });
});

test('on returns unsubscribe function', () => {
    const store = new ReactiveStore();
    let count = 0;
    const unsub = store.on('x', () => { count++; });
    store.set('x', 1);
    unsub();
    store.set('x', 2);
    assert.strictEqual(count, 1);
});

test('set with same value does not notify', () => {
    const store = new ReactiveStore();
    let count = 0;
    store.set('a', 5);
    store.on('a', () => { count++; });
    store.set('a', 5); // same value
    assert.strictEqual(count, 0);
});

test('multiple listeners on same path', () => {
    const store = new ReactiveStore();
    let a = 0, b = 0;
    store.on('x', () => { a++; });
    store.on('x', () => { b++; });
    store.set('x', 'hello');
    assert.strictEqual(a, 1);
    assert.strictEqual(b, 1);
});

test('listener error does not break other listeners', () => {
    const store = new ReactiveStore();
    let secondCalled = false;
    store.on('y', () => { throw new Error('boom'); });
    store.on('y', () => { secondCalled = true; });
    store.set('y', 1);
    assert.strictEqual(secondCalled, true);
});

test('flushNotify fires pending notifications', () => {
    const store = new ReactiveStore();
    let val = null;
    store.on('z', (v) => { val = v; });
    // Directly manipulate _pendingNotify to test flushNotify
    store._state.z = 99;
    store._pendingNotify.add('z');
    store._pendingValues = new Map();
    store._pendingValues.set('z', { value: 99, oldValue: undefined });
    store.flushNotify();
    assert.strictEqual(val, 99);
});

test('destroy clears everything', () => {
    const store = new ReactiveStore();
    let count = 0;
    store.on('a', () => { count++; });
    store.set('a', 1);
    store.destroy();
    assert.strictEqual(store.get('a'), undefined);
    // Listeners gone — this shouldn't increment count
    store.set('a', 2);
    assert.strictEqual(count, 1);
});

test('deeply nested paths', () => {
    const store = new ReactiveStore();
    store.set('a.b.c.d', 'deep');
    assert.strictEqual(store.get('a.b.c.d'), 'deep');
    assert.strictEqual(store.get('a.b.c.missing', 'nope'), 'nope');
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
