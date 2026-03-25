// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/websocket.js — TritiumWebSocket
// Run: node tests/web/test_websocket.mjs

import { TritiumWebSocket } from '../../web/websocket.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

console.log('=== TritiumWebSocket Tests ===');

test('constructor sets defaults', () => {
    const ws = new TritiumWebSocket('ws://localhost:8000/ws');
    assert.strictEqual(ws._url, 'ws://localhost:8000/ws');
    assert.strictEqual(ws._initialDelay, 1000);
    assert.strictEqual(ws._maxDelay, 16000);
    assert.strictEqual(ws._PING_INTERVAL_MS, 25000);
    assert.strictEqual(ws._destroyed, false);
    assert.strictEqual(ws._ws, null);
});

test('constructor accepts custom options', () => {
    const ws = new TritiumWebSocket('ws://x', {
        initialDelay: 500,
        maxDelay: 8000,
        pingInterval: 10000,
    });
    assert.strictEqual(ws._initialDelay, 500);
    assert.strictEqual(ws._maxDelay, 8000);
    assert.strictEqual(ws._PING_INTERVAL_MS, 10000);
});

test('disconnect sets destroyed flag', () => {
    const ws = new TritiumWebSocket('ws://x');
    ws.disconnect();
    assert.strictEqual(ws._destroyed, true);
    assert.strictEqual(ws._ws, null);
});

test('connect does nothing when destroyed', () => {
    const ws = new TritiumWebSocket('ws://x');
    ws.disconnect();
    // connect after destroy should not throw or create ws
    ws.connect();
    assert.strictEqual(ws._ws, null);
});

test('isConnected is false when no ws', () => {
    const ws = new TritiumWebSocket('ws://x');
    assert.strictEqual(ws.isConnected, false);
});

test('send does not throw when no ws', () => {
    const ws = new TritiumWebSocket('ws://x');
    // Should not throw even with no active connection
    ws.send({ type: 'test' });
    assert.strictEqual(ws._ws, null);
});

test('_handleMessage responds to ping', () => {
    const ws = new TritiumWebSocket('ws://x');
    let sentData = null;
    // Mock send to capture output
    ws.send = (data) => { sentData = data; };
    ws._handleMessage({ type: 'ping' });
    assert.deepStrictEqual(sentData, { type: 'pong' });
});

test('_handleMessage ignores non-ping messages', () => {
    const ws = new TritiumWebSocket('ws://x');
    let sentData = null;
    ws.send = (data) => { sentData = data; };
    ws._handleMessage({ type: 'game_state', data: {} });
    assert.strictEqual(sentData, null);
});

test('multiple disconnect calls are safe', () => {
    const ws = new TritiumWebSocket('ws://x');
    ws.disconnect();
    ws.disconnect();
    assert.strictEqual(ws._destroyed, true);
});

test('_stopPingKeepalive clears timer', () => {
    const ws = new TritiumWebSocket('ws://x');
    ws._pingTimer = setInterval(() => {}, 100000);
    ws._stopPingKeepalive();
    assert.strictEqual(ws._pingTimer, null);
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
