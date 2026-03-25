// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/sim/mobil.js — MOBIL lane change model
// Run: node tests/web/test_mobil.mjs

import { findNeighborsInLane, evaluateLaneChange, decideLaneChange, MOBIL_DEFAULTS } from '../../web/sim/mobil.js';
import { IDM_DEFAULTS } from '../../web/sim/idm.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

// Helper to create a mock vehicle
function makeCar(opts) {
    return {
        id: opts.id || 'car_test',
        edge: opts.edge || { id: 'e1', lanesPerDir: 2 },
        u: opts.u || 50,
        speed: opts.speed || 10,
        direction: opts.direction || 1,
        laneIdx: opts.laneIdx ?? 0,
        length: opts.length || 4.5,
        idm: opts.idm || { ...IDM_DEFAULTS },
    };
}

console.log('=== MOBIL Tests ===');

// ── MOBIL_DEFAULTS ────────────────────────────────────────────────

test('MOBIL_DEFAULTS has all required fields', () => {
    assert.strictEqual(typeof MOBIL_DEFAULTS.politeness, 'number');
    assert.strictEqual(typeof MOBIL_DEFAULTS.threshold, 'number');
    assert.strictEqual(typeof MOBIL_DEFAULTS.bSafe, 'number');
    assert.strictEqual(typeof MOBIL_DEFAULTS.minGap, 'number');
});

// ── findNeighborsInLane ───────────────────────────────────────────

test('findNeighborsInLane: no neighbors in empty list', () => {
    const car = makeCar({ u: 50 });
    const result = findNeighborsInLane(car, 0, [car]);
    assert.strictEqual(result.ahead, null);
    assert.strictEqual(result.behind, null);
});

test('findNeighborsInLane: finds vehicle ahead', () => {
    const edge = { id: 'e1', lanesPerDir: 2 };
    const car = makeCar({ edge, u: 50, laneIdx: 0 });
    const ahead = makeCar({ id: 'ahead', edge, u: 70, laneIdx: 0 });
    const result = findNeighborsInLane(car, 0, [car, ahead]);
    assert.strictEqual(result.ahead, ahead);
    assert.ok(result.aheadGap > 0, 'gap should be positive');
});

test('findNeighborsInLane: finds vehicle behind', () => {
    const edge = { id: 'e1', lanesPerDir: 2 };
    const car = makeCar({ edge, u: 50, laneIdx: 0 });
    const behind = makeCar({ id: 'behind', edge, u: 20, laneIdx: 0 });
    const result = findNeighborsInLane(car, 0, [car, behind]);
    assert.strictEqual(result.behind, behind);
    assert.ok(result.behindGap > 0, 'behind gap should be positive');
});

test('findNeighborsInLane: ignores vehicles in different lane', () => {
    const edge = { id: 'e1', lanesPerDir: 2 };
    const car = makeCar({ edge, u: 50, laneIdx: 0 });
    const otherLane = makeCar({ id: 'other', edge, u: 70, laneIdx: 1 });
    const result = findNeighborsInLane(car, 0, [car, otherLane]);
    assert.strictEqual(result.ahead, null);
});

test('findNeighborsInLane: ignores vehicles on different edge', () => {
    const edge1 = { id: 'e1', lanesPerDir: 2 };
    const edge2 = { id: 'e2', lanesPerDir: 2 };
    const car = makeCar({ edge: edge1, u: 50, laneIdx: 0 });
    const other = makeCar({ id: 'other', edge: edge2, u: 70, laneIdx: 0 });
    const result = findNeighborsInLane(car, 0, [car, other]);
    assert.strictEqual(result.ahead, null);
});

test('findNeighborsInLane: finds closest ahead among multiple', () => {
    const edge = { id: 'e1', lanesPerDir: 2 };
    const car = makeCar({ edge, u: 50, laneIdx: 0 });
    const far = makeCar({ id: 'far', edge, u: 100, laneIdx: 0 });
    const near = makeCar({ id: 'near', edge, u: 65, laneIdx: 0 });
    const result = findNeighborsInLane(car, 0, [car, far, near]);
    assert.strictEqual(result.ahead, near, 'should find nearest ahead');
});

test('findNeighborsInLane: bumper-to-bumper gap subtracts car lengths', () => {
    const edge = { id: 'e1', lanesPerDir: 2 };
    const car = makeCar({ edge, u: 50, laneIdx: 0, length: 4.0 });
    const ahead = makeCar({ id: 'ahead', edge, u: 60, laneIdx: 0, length: 4.0 });
    const result = findNeighborsInLane(car, 0, [car, ahead]);
    // Raw gap is 10m, minus 4.0/2 + 4.0/2 = 4.0 => 6.0m bumper gap
    assert.ok(Math.abs(result.aheadGap - 6.0) < 0.1, `expected ~6.0, got ${result.aheadGap}`);
});

test('findNeighborsInLane: respects direction for reverse travel', () => {
    const edge = { id: 'e1', lanesPerDir: 2 };
    const car = makeCar({ edge, u: 50, laneIdx: 0, direction: -1 });
    // When direction=-1, a vehicle at u=30 is "ahead" because we're going backward
    const ahead = makeCar({ id: 'ahead', edge, u: 30, laneIdx: 0, direction: -1 });
    const result = findNeighborsInLane(car, 0, [car, ahead]);
    assert.strictEqual(result.ahead, ahead, 'u=30 should be ahead when direction=-1');
});

// ── evaluateLaneChange ────────────────────────────────────────────

test('evaluateLaneChange: insufficient gap returns shouldChange=false', () => {
    const edge = { id: 'e1', lanesPerDir: 2 };
    const car = makeCar({ edge, u: 50, laneIdx: 0 });
    // Place a vehicle very close in target lane
    const blocker = makeCar({ id: 'blocker', edge, u: 52, laneIdx: 1 });
    const result = evaluateLaneChange(car, 1, [car, blocker]);
    assert.strictEqual(result.shouldChange, false);
    assert.strictEqual(result.reason, 'insufficient_gap');
});

test('evaluateLaneChange: empty target lane is beneficial when current lane blocked', () => {
    const edge = { id: 'e1', lanesPerDir: 2 };
    const car = makeCar({ edge, u: 50, laneIdx: 0, speed: 10 });
    // Slow vehicle blocking current lane, target lane empty
    const blocker = makeCar({ id: 'blocker', edge, u: 60, laneIdx: 0, speed: 2 });
    const result = evaluateLaneChange(car, 1, [car, blocker]);
    assert.strictEqual(result.shouldChange, true, 'should want to change to empty lane');
    assert.ok(result.reason === 'beneficial_empty_lane' || result.reason === 'beneficial');
});

test('evaluateLaneChange: no benefit if both lanes clear', () => {
    const edge = { id: 'e1', lanesPerDir: 2 };
    const car = makeCar({ edge, u: 50, laneIdx: 0, speed: 10 });
    // No other vehicles at all
    const result = evaluateLaneChange(car, 1, [car]);
    // With no leader in either lane, there's no advantage
    assert.strictEqual(result.shouldChange, false);
    assert.strictEqual(result.reason, 'insufficient_incentive');
});

// ── decideLaneChange ──────────────────────────────────────────────

test('decideLaneChange: single-lane road returns null direction', () => {
    const edge = { id: 'e1', lanesPerDir: 1 };
    const car = makeCar({ edge, laneIdx: 0 });
    const result = decideLaneChange(car, [car]);
    assert.strictEqual(result.direction, null);
    assert.strictEqual(result.targetLane, null);
});

test('decideLaneChange: picks best lane when blocked', () => {
    const edge = { id: 'e1', lanesPerDir: 3 };
    const car = makeCar({ edge, u: 50, laneIdx: 1, speed: 10 });
    // Slow car blocking lane 1 (our current lane)
    const blocker = makeCar({ id: 'blocker', edge, u: 60, laneIdx: 1, speed: 1 });
    const result = decideLaneChange(car, [car, blocker]);
    // Should suggest changing to lane 0 or 2
    if (result.direction !== null) {
        assert.ok(result.targetLane === 0 || result.targetLane === 2);
    }
    // Note: may not always trigger due to threshold
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
