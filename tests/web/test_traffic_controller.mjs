// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/sim/traffic-controller.js — TrafficController
// Run: node tests/web/test_traffic_controller.mjs

import { TrafficController, TrafficControllerManager } from '../../web/sim/traffic-controller.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

// Helper: create mock edges
function makeEdge(id) {
    return { id, from: 'n0', to: 'n1', length: 100 };
}

console.log('=== TrafficController Tests ===');

// ── Single edge (dead end) ────────────────────────────────────────

test('dead-end intersection is always green', () => {
    const ctrl = new TrafficController('n0', [makeEdge('e1')], { x: 0, z: 0, degree: 1 });
    assert.strictEqual(ctrl.phases.length, 1);
    assert.strictEqual(ctrl.isGreen('e1'), true);
    assert.strictEqual(ctrl.getSignalColor('e1'), 'green');
});

// ── Two-edge intersection ─────────────────────────────────────────

test('two-edge intersection creates correct phase cycle', () => {
    const edges = [makeEdge('e1'), makeEdge('e2')];
    const ctrl = new TrafficController('n0', edges, { x: 0, z: 0, degree: 2 });
    // 6 phases: A-green, yellow, allred, B-green, yellow, allred
    assert.strictEqual(ctrl.phases.length, 6);
    assert.strictEqual(ctrl.phases[0].type, 'green');
    assert.strictEqual(ctrl.phases[1].type, 'yellow');
    assert.strictEqual(ctrl.phases[2].type, 'allred');
    assert.strictEqual(ctrl.phases[3].type, 'green');
    assert.strictEqual(ctrl.phases[4].type, 'yellow');
    assert.strictEqual(ctrl.phases[5].type, 'allred');
});

test('isGreen returns true only for phase A edges during phase A', () => {
    const edges = [makeEdge('e1'), makeEdge('e2')];
    const ctrl = new TrafficController('n0', edges, { x: 0, z: 0, degree: 2 });
    ctrl.currentPhase = 0; // Phase A green
    assert.strictEqual(ctrl.isGreen('e1'), true, 'e1 should be green in phase A');
    assert.strictEqual(ctrl.isGreen('e2'), false, 'e2 should be red in phase A');
});

test('isGreen flips when phase advances', () => {
    const edges = [makeEdge('e1'), makeEdge('e2')];
    const ctrl = new TrafficController('n0', edges, { x: 0, z: 0, degree: 2 });
    ctrl.currentPhase = 3; // Phase B green
    assert.strictEqual(ctrl.isGreen('e1'), false, 'e1 should be red in phase B');
    assert.strictEqual(ctrl.isGreen('e2'), true, 'e2 should be green in phase B');
});

// ── Signal color ──────────────────────────────────────────────────

test('getSignalColor returns correct colors per phase type', () => {
    const edges = [makeEdge('e1'), makeEdge('e2')];
    const ctrl = new TrafficController('n0', edges, { x: 0, z: 0, degree: 2 });

    ctrl.currentPhase = 0; // green for e1
    assert.strictEqual(ctrl.getSignalColor('e1'), 'green');
    assert.strictEqual(ctrl.getSignalColor('e2'), 'red');

    ctrl.currentPhase = 1; // yellow for e1
    assert.strictEqual(ctrl.getSignalColor('e1'), 'yellow');

    ctrl.currentPhase = 2; // allred
    assert.strictEqual(ctrl.getSignalColor('e1'), 'red');
    assert.strictEqual(ctrl.getSignalColor('e2'), 'red');
});

test('isYellow detects yellow phase', () => {
    const edges = [makeEdge('e1'), makeEdge('e2')];
    const ctrl = new TrafficController('n0', edges, { x: 0, z: 0, degree: 2 });
    ctrl.currentPhase = 0;
    assert.strictEqual(ctrl.isYellow(), false);
    ctrl.currentPhase = 1;
    assert.strictEqual(ctrl.isYellow(), true);
});

// ── Tick ──────────────────────────────────────────────────────────

test('tick advances phase when timer exceeds duration', () => {
    const edges = [makeEdge('e1'), makeEdge('e2')];
    const ctrl = new TrafficController('n0', edges, { x: 0, z: 0, degree: 2 });
    ctrl.phaseTimer = 0;
    const greenDuration = ctrl.phases[0].duration;
    // Tick past the green duration
    ctrl.tick(greenDuration + 1);
    assert.strictEqual(ctrl.currentPhase, 1, 'should advance to yellow phase');
});

test('tick wraps around phases', () => {
    const edges = [makeEdge('e1'), makeEdge('e2')];
    const ctrl = new TrafficController('n0', edges, { x: 0, z: 0, degree: 2 });
    ctrl.phaseTimer = 0;
    ctrl.currentPhase = 5; // last phase (allred)
    const duration = ctrl.phases[5].duration;
    ctrl.tick(duration + 0.01);
    assert.strictEqual(ctrl.currentPhase, 0, 'should wrap to first phase');
});

test('tick does nothing in RL mode', () => {
    const edges = [makeEdge('e1'), makeEdge('e2')];
    const ctrl = new TrafficController('n0', edges, { x: 0, z: 0, degree: 2 });
    ctrl.mode = 'rl';
    ctrl.currentPhase = 0;
    ctrl.phaseTimer = 0;
    ctrl.tick(1000);
    assert.strictEqual(ctrl.currentPhase, 0, 'RL mode should not advance phases');
});

// ── Adaptive mode ─────────────────────────────────────────────────

test('adaptive mode extends green based on queue count', () => {
    const edges = [makeEdge('e1'), makeEdge('e2')];
    const ctrl = new TrafficController('n0', edges, { x: 0, z: 0, degree: 2 });
    ctrl.mode = 'adaptive';
    ctrl.currentPhase = 0;
    ctrl.phaseTimer = 0;
    ctrl.updateQueueCounts({ e1: 5 }); // 5 vehicles queued on green edge
    const adaptiveDuration = ctrl._getAdaptiveDuration();
    const baseDuration = ctrl.phases[0].duration;
    assert.ok(adaptiveDuration > baseDuration, 'adaptive should extend green for queued vehicles');
    assert.ok(adaptiveDuration <= ctrl.maxGreen, 'should not exceed maxGreen');
});

test('adaptive mode returns min green for empty phase with demand elsewhere', () => {
    const edges = [makeEdge('e1'), makeEdge('e2')];
    const ctrl = new TrafficController('n0', edges, { x: 0, z: 0, degree: 2 });
    ctrl.mode = 'adaptive';
    ctrl.currentPhase = 0;
    ctrl.phaseTimer = 0;
    // No vehicles on current green edge, but demand on other phase
    ctrl.updateQueueCounts({ e2: 3 });
    const adaptiveDuration = ctrl._getAdaptiveDuration();
    assert.strictEqual(adaptiveDuration, ctrl.minGreen, 'should use minGreen for empty phase with demand elsewhere');
});

test('updateQueueCounts stores counts correctly', () => {
    const edges = [makeEdge('e1'), makeEdge('e2')];
    const ctrl = new TrafficController('n0', edges, { x: 0, z: 0, degree: 2 });
    ctrl.updateQueueCounts({ e1: 3, e2: 7 });
    assert.strictEqual(ctrl._queueCounts.e1, 3);
    assert.strictEqual(ctrl._queueCounts.e2, 7);
});

// ── TrafficControllerManager ──────────────────────────────────────

test('manager initializes controllers for 3+ degree nodes', () => {
    const mgr = new TrafficControllerManager();
    // Build a simple T-intersection network
    const roadNetwork = {
        nodes: {
            n0: { id: 'n0', x: 0, z: 0, degree: 1 },
            n1: { id: 'n1', x: 100, z: 0, degree: 3 },
            n2: { id: 'n2', x: 200, z: 0, degree: 1 },
            n3: { id: 'n3', x: 100, z: 100, degree: 1 },
        },
        adjList: {
            n0: [0], n1: [0, 1, 2], n2: [1], n3: [2],
        },
        edges: [
            { id: 'e0', from: 'n0', to: 'n1', length: 100 },
            { id: 'e1', from: 'n1', to: 'n2', length: 100 },
            { id: 'e2', from: 'n1', to: 'n3', length: 100 },
        ],
    };
    mgr.initFromNetwork(roadNetwork);
    assert.strictEqual(Object.keys(mgr.controllers).length, 1, 'should create controller for n1 only');
    assert.ok(mgr.controllers.n1, 'n1 should have a controller');
});

test('manager isGreen returns true when no controller exists', () => {
    const mgr = new TrafficControllerManager();
    assert.strictEqual(mgr.isGreen('nonexistent', 'e1'), true);
});

test('manager setMode applies to all controllers', () => {
    const mgr = new TrafficControllerManager();
    const roadNetwork = {
        nodes: { n0: { id: 'n0', x: 0, z: 0, degree: 3 } },
        adjList: { n0: [0, 1, 2] },
        edges: [
            { id: 'e0', from: 'n0', to: 'n1', length: 100 },
            { id: 'e1', from: 'n0', to: 'n2', length: 100 },
            { id: 'e2', from: 'n0', to: 'n3', length: 100 },
        ],
    };
    mgr.initFromNetwork(roadNetwork);
    mgr.setMode('adaptive');
    assert.strictEqual(mgr.controllers.n0.mode, 'adaptive');
});

test('manager getSignalColor returns green when no controller', () => {
    const mgr = new TrafficControllerManager();
    assert.strictEqual(mgr.getSignalColor('nonexistent', 'e1'), 'green');
});

test('manager tick advances all controllers', () => {
    const mgr = new TrafficControllerManager();
    const roadNetwork = {
        nodes: { n0: { id: 'n0', x: 0, z: 0, degree: 3 } },
        adjList: { n0: [0, 1, 2] },
        edges: [
            { id: 'e0', from: 'n0', to: 'n1', length: 100 },
            { id: 'e1', from: 'n0', to: 'n2', length: 100 },
            { id: 'e2', from: 'n0', to: 'n3', length: 100 },
        ],
    };
    mgr.initFromNetwork(roadNetwork);
    const initialTimer = mgr.controllers.n0.phaseTimer;
    mgr.tick(1.0);
    assert.ok(mgr.controllers.n0.phaseTimer > initialTimer || mgr.controllers.n0.currentPhase > 0,
        'timer should advance or phase should change');
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
