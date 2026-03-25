// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/events.js — EventBus
// Run: node tests/web/test_events.mjs

import { EventBus } from '../../web/events.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

// Reset handlers between tests
function reset() { EventBus._handlers.clear(); }

console.log('=== EventBus Tests ===');

reset();
test('on + emit fires handler', () => {
    let called = false;
    EventBus.on('test:a', () => { called = true; });
    EventBus.emit('test:a');
    assert.strictEqual(called, true);
});

reset();
test('emit passes data to handler', () => {
    let received = null;
    EventBus.on('test:b', (d) => { received = d; });
    EventBus.emit('test:b', { x: 42 });
    assert.deepStrictEqual(received, { x: 42 });
});

reset();
test('off removes handler', () => {
    let count = 0;
    const handler = () => { count++; };
    EventBus.on('test:c', handler);
    EventBus.emit('test:c');
    EventBus.off('test:c', handler);
    EventBus.emit('test:c');
    assert.strictEqual(count, 1);
});

reset();
test('on returns unsubscribe function', () => {
    let count = 0;
    const unsub = EventBus.on('test:d', () => { count++; });
    EventBus.emit('test:d');
    unsub();
    EventBus.emit('test:d');
    assert.strictEqual(count, 1);
});

reset();
test('multiple handlers on same event', () => {
    let a = 0, b = 0;
    EventBus.on('test:e', () => { a++; });
    EventBus.on('test:e', () => { b++; });
    EventBus.emit('test:e');
    assert.strictEqual(a, 1);
    assert.strictEqual(b, 1);
});

reset();
test('wildcard * handler receives all events', () => {
    const events = [];
    EventBus.on('*', (eventName, data) => { events.push(eventName); });
    EventBus.emit('test:wild1');
    EventBus.emit('test:wild2');
    assert(events.includes('test:wild1'));
    assert(events.includes('test:wild2'));
});

reset();
test('handler error does not break other handlers', () => {
    let secondCalled = false;
    EventBus.on('test:err', () => { throw new Error('boom'); });
    EventBus.on('test:err', () => { secondCalled = true; });
    EventBus.emit('test:err');
    assert.strictEqual(secondCalled, true);
});

reset();
test('emit with no listeners does not throw', () => {
    EventBus.emit('nonexistent:event', { data: 1 });
    assert(true); // no throw
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
