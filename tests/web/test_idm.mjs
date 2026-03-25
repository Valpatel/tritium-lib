// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/sim/idm.js — Intelligent Driver Model
// Run: node tests/web/test_idm.mjs

import { idmAcceleration, idmFreeFlow, idmStep, IDM_DEFAULTS, ROAD_SPEEDS } from '../../web/sim/idm.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

console.log('=== IDM Tests ===');

// ── IDM_DEFAULTS ──────────────────────────────────────────────────

test('IDM_DEFAULTS has all required fields', () => {
    assert.strictEqual(typeof IDM_DEFAULTS.v0, 'number');
    assert.strictEqual(typeof IDM_DEFAULTS.a, 'number');
    assert.strictEqual(typeof IDM_DEFAULTS.b, 'number');
    assert.strictEqual(typeof IDM_DEFAULTS.s0, 'number');
    assert.strictEqual(typeof IDM_DEFAULTS.T, 'number');
    assert.strictEqual(typeof IDM_DEFAULTS.delta, 'number');
    assert.strictEqual(IDM_DEFAULTS.delta, 4);
});

test('ROAD_SPEEDS has expected road classes', () => {
    assert.strictEqual(typeof ROAD_SPEEDS.motorway, 'number');
    assert.strictEqual(typeof ROAD_SPEEDS.residential, 'number');
    assert.strictEqual(typeof ROAD_SPEEDS.service, 'number');
    assert.ok(ROAD_SPEEDS.motorway > ROAD_SPEEDS.residential, 'motorway faster than residential');
    assert.ok(ROAD_SPEEDS.residential > ROAD_SPEEDS.service, 'residential faster than service');
});

// ── idmFreeFlow ───────────────────────────────────────────────────

test('idmFreeFlow: zero speed gives max acceleration', () => {
    const acc = idmFreeFlow(0, IDM_DEFAULTS);
    // At v=0: a * (1 - 0) = a
    assert.ok(Math.abs(acc - IDM_DEFAULTS.a) < 0.001, `expected ~${IDM_DEFAULTS.a}, got ${acc}`);
});

test('idmFreeFlow: at desired speed, acceleration is zero', () => {
    const acc = idmFreeFlow(IDM_DEFAULTS.v0, IDM_DEFAULTS);
    // At v=v0: a * (1 - 1) = 0
    assert.ok(Math.abs(acc) < 0.001, `expected ~0, got ${acc}`);
});

test('idmFreeFlow: above desired speed, acceleration is negative', () => {
    const acc = idmFreeFlow(IDM_DEFAULTS.v0 * 1.5, IDM_DEFAULTS);
    assert.ok(acc < 0, `expected negative, got ${acc}`);
});

test('idmFreeFlow: half desired speed gives positive acceleration', () => {
    const acc = idmFreeFlow(IDM_DEFAULTS.v0 / 2, IDM_DEFAULTS);
    assert.ok(acc > 0, `expected positive, got ${acc}`);
    assert.ok(acc < IDM_DEFAULTS.a, 'should be less than max acceleration');
});

// ── idmAcceleration ───────────────────────────────────────────────

test('idmAcceleration: large gap and same speed behaves like free flow', () => {
    const freeAcc = idmFreeFlow(5, IDM_DEFAULTS);
    const acc = idmAcceleration(5, 1000, 5, IDM_DEFAULTS);
    // With a 1000m gap and no speed difference, interaction term is negligible
    assert.ok(Math.abs(acc - freeAcc) < 0.1, `expected ~${freeAcc}, got ${acc}`);
});

test('idmAcceleration: very small gap causes heavy braking', () => {
    const acc = idmAcceleration(10, 1.0, 0, IDM_DEFAULTS);
    assert.ok(acc < -2.0, `expected strong braking, got ${acc}`);
});

test('idmAcceleration: clamped to minimum -9 m/s^2', () => {
    const acc = idmAcceleration(20, 0.5, 0, IDM_DEFAULTS);
    assert.ok(acc >= -9.0, `expected >= -9, got ${acc}`);
});

test('idmAcceleration: approaching faster than leader causes braking', () => {
    const acc = idmAcceleration(15, 10, 5, IDM_DEFAULTS);
    // Big speed difference (15 vs 5) at close range = braking
    assert.ok(acc < 0, `expected negative (braking), got ${acc}`);
});

test('idmAcceleration: leader faster than ego reduces desired gap', () => {
    // When leader is faster, dv is negative, which reduces s*
    const accFaster = idmAcceleration(5, 10, 10, IDM_DEFAULTS);
    const accSame = idmAcceleration(5, 10, 5, IDM_DEFAULTS);
    assert.ok(accFaster >= accSame, 'faster leader should mean less braking');
});

test('idmAcceleration: gap cannot be zero (clamped to 0.5)', () => {
    // Should not throw or produce NaN/Infinity
    const acc = idmAcceleration(5, 0, 5, IDM_DEFAULTS);
    assert.ok(Number.isFinite(acc), `expected finite, got ${acc}`);
});

// ── idmStep ───────────────────────────────────────────────────────

test('idmStep: positive acceleration increases speed', () => {
    const { v, ds } = idmStep(5, 1.0, 1.0);
    assert.strictEqual(v, 6);
    assert.ok(ds > 5, 'distance should be more than v*dt due to acceleration');
});

test('idmStep: speed does not go below zero', () => {
    const { v, ds } = idmStep(1, -5, 1.0);
    assert.strictEqual(v, 0);
    assert.ok(ds >= 0, 'distance should be non-negative');
});

test('idmStep: zero acceleration, constant speed', () => {
    const { v, ds } = idmStep(10, 0, 0.5);
    assert.strictEqual(v, 10);
    assert.ok(Math.abs(ds - 5.0) < 0.001, `expected 5.0, got ${ds}`);
});

test('idmStep: small timestep', () => {
    const { v, ds } = idmStep(10, 1.0, 0.01);
    assert.ok(Math.abs(v - 10.01) < 0.001, `expected ~10.01, got ${v}`);
    assert.ok(ds > 0, 'should move forward');
});

test('idmStep: distance uses kinematic equation', () => {
    // ds = v*dt + 0.5*a*dt^2 = 10*1 + 0.5*2*1 = 11
    const { ds } = idmStep(10, 2, 1.0);
    assert.ok(Math.abs(ds - 11) < 0.001, `expected 11, got ${ds}`);
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
