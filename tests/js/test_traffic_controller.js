#!/usr/bin/env node
// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * TrafficController + TrafficControllerManager tests.
 *
 * Tests: phase cycling, virtual obstacles, signal colors, adaptive timing,
 * RL mode, manager init, queue updates, stagger offsets.
 *
 * Run: node tests/js/test_traffic_controller.js
 */

const fs = require('fs');
const vm = require('vm');

let passed = 0, failed = 0;

function assert(condition, msg) {
    if (condition) { passed++; console.log(`PASS: ${msg}`); }
    else { failed++; console.log(`FAIL: ${msg}`); }
}

function assertEqual(a, b, msg) {
    assert(a === b, msg + ` (got ${JSON.stringify(a)}, expected ${JSON.stringify(b)})`);
}

function assertApprox(a, b, tol, msg) {
    const ok = Math.abs(a - b) <= tol;
    if (ok) { passed++; console.log(`PASS: ${msg} (${a} ~= ${b})`); }
    else { failed++; console.log(`FAIL: ${msg} (${a} != ${b} +/-${tol})`); }
}

// Load traffic-controller.js, stripping ES module syntax
const code = fs.readFileSync(__dirname + '/../../web/sim/traffic-controller.js', 'utf8');
const plainCode = code
    .replace(/^export\s+/gm, '')
    .replace(/^import\s+.*$/gm, '');

function createContext() {
    const sandbox = {
        Math, console, Object, Array, Number, String, Set, Map, Infinity, NaN,
        parseInt, parseFloat, JSON, undefined, Boolean,
    };
    const ctx = vm.createContext(sandbox);
    vm.runInContext(plainCode, ctx);
    return ctx;
}

// Helper: create a TrafficController for a standard 4-way intersection
function create4Way(ctx) {
    const edges = [
        { id: 'e_north', from: 'n0', to: 'center' },
        { id: 'e_south', from: 'center', to: 'n1' },
        { id: 'e_east', from: 'n2', to: 'center' },
        { id: 'e_west', from: 'center', to: 'n3' },
    ];
    const node = { x: 50, z: 50, degree: 4 };
    vm.runInContext(`
        var _edges = ${JSON.stringify(edges)};
        var _node = ${JSON.stringify(node)};
        var _ctrl = new TrafficController('center', _edges, _node);
    `, ctx);
    return ctx;
}

// Helper: create a 3-way intersection
function create3Way(ctx) {
    const edges = [
        { id: 'e_a', from: 'n0', to: 'center' },
        { id: 'e_b', from: 'center', to: 'n1' },
        { id: 'e_c', from: 'n2', to: 'center' },
    ];
    const node = { x: 50, z: 50, degree: 3 };
    vm.runInContext(`
        var _edges = ${JSON.stringify(edges)};
        var _node = ${JSON.stringify(node)};
        var _ctrl = new TrafficController('center', _edges, _node);
    `, ctx);
    return ctx;
}

// Helper: create a dead-end (1 edge)
function createDeadEnd(ctx) {
    const edges = [{ id: 'e_only', from: 'n0', to: 'center' }];
    const node = { x: 50, z: 50, degree: 1 };
    vm.runInContext(`
        var _edges = ${JSON.stringify(edges)};
        var _node = ${JSON.stringify(node)};
        var _ctrl = new TrafficController('center', _edges, _node);
    `, ctx);
    return ctx;
}

// ============================================================
// Constructor — 4-way
// ============================================================

console.log('\n--- TrafficController Constructor (4-way) ---');

(function() {
    const ctx = createContext();
    create4Way(ctx);
    const nodeId = vm.runInContext('_ctrl.nodeId', ctx);
    assertEqual(nodeId, 'center', 'nodeId set correctly');
})();

(function() {
    const ctx = createContext();
    create4Way(ctx);
    const numPhases = vm.runInContext('_ctrl.phases.length', ctx);
    assertEqual(numPhases, 6, '4-way intersection has 6 phases (2x green + 2x yellow + 2x allred)');
})();

(function() {
    const ctx = createContext();
    create4Way(ctx);
    const currentPhase = vm.runInContext('_ctrl.currentPhase', ctx);
    assertEqual(currentPhase, 0, 'Starts at phase 0');
})();

(function() {
    const ctx = createContext();
    create4Way(ctx);
    const mode = vm.runInContext('_ctrl.mode', ctx);
    assertEqual(mode, 'fixed', 'Default mode is fixed');
})();

// ============================================================
// Phase structure — 4-way
// ============================================================

console.log('\n--- Phase structure (4-way) ---');

(function() {
    const ctx = createContext();
    create4Way(ctx);
    // Phase 0: green for group A
    const type0 = vm.runInContext('_ctrl.phases[0].type', ctx);
    assertEqual(type0, 'green', 'Phase 0 is green');
    // Phase 1: yellow
    const type1 = vm.runInContext('_ctrl.phases[1].type', ctx);
    assertEqual(type1, 'yellow', 'Phase 1 is yellow');
    // Phase 2: allred
    const type2 = vm.runInContext('_ctrl.phases[2].type', ctx);
    assertEqual(type2, 'allred', 'Phase 2 is allred');
    // Phase 3: green for group B
    const type3 = vm.runInContext('_ctrl.phases[3].type', ctx);
    assertEqual(type3, 'green', 'Phase 3 is green');
    // Phase 4: yellow
    const type4 = vm.runInContext('_ctrl.phases[4].type', ctx);
    assertEqual(type4, 'yellow', 'Phase 4 is yellow');
    // Phase 5: allred
    const type5 = vm.runInContext('_ctrl.phases[5].type', ctx);
    assertEqual(type5, 'allred', 'Phase 5 is allred');
})();

(function() {
    const ctx = createContext();
    create4Way(ctx);
    // 4-way: green duration should be 20
    const dur0 = vm.runInContext('_ctrl.phases[0].duration', ctx);
    assertEqual(dur0, 20, '4-way green phase duration is 20s');
    // Yellow is 2s
    const dur1 = vm.runInContext('_ctrl.phases[1].duration', ctx);
    assertEqual(dur1, 2, 'Yellow phase duration is 2s');
    // Allred is 1s
    const dur2 = vm.runInContext('_ctrl.phases[2].duration', ctx);
    assertEqual(dur2, 1, 'Allred phase duration is 1s');
})();

// ============================================================
// Phase structure — 3-way
// ============================================================

console.log('\n--- Phase structure (3-way) ---');

(function() {
    const ctx = createContext();
    create3Way(ctx);
    const numPhases = vm.runInContext('_ctrl.phases.length', ctx);
    assertEqual(numPhases, 6, '3-way has 6 phases');
    // 3-way: green duration should be 15
    const dur = vm.runInContext('_ctrl.phases[0].duration', ctx);
    assertEqual(dur, 15, '3-way green phase duration is 15s');
})();

// ============================================================
// Dead end — always green
// ============================================================

console.log('\n--- Dead end ---');

(function() {
    const ctx = createContext();
    createDeadEnd(ctx);
    const numPhases = vm.runInContext('_ctrl.phases.length', ctx);
    assertEqual(numPhases, 1, 'Dead end has 1 phase');
    const type = vm.runInContext('_ctrl.phases[0].type', ctx);
    assertEqual(type, 'green', 'Dead end phase is green');
    const dur = vm.runInContext('_ctrl.phases[0].duration', ctx);
    assertEqual(dur, 999, 'Dead end green duration is 999s');
})();

(function() {
    const ctx = createContext();
    createDeadEnd(ctx);
    const isGreen = vm.runInContext('_ctrl.isGreen("e_only")', ctx);
    assertEqual(isGreen, true, 'Dead end edge is always green');
})();

// ============================================================
// Green edge check — group A vs group B
// ============================================================

console.log('\n--- Green edge check ---');

(function() {
    const ctx = createContext();
    create4Way(ctx);
    // Phase 0: group A has green (first half of edges: e_north, e_south)
    const greenNorth = vm.runInContext('_ctrl.isGreen("e_north")', ctx);
    const greenSouth = vm.runInContext('_ctrl.isGreen("e_south")', ctx);
    const greenEast = vm.runInContext('_ctrl.isGreen("e_east")', ctx);
    const greenWest = vm.runInContext('_ctrl.isGreen("e_west")', ctx);
    assert(greenNorth === true || greenSouth === true, 'Group A has at least one green edge in phase 0');
    // Group B should be red in phase 0
    // (Group A = first half = e_north, e_south; Group B = second half = e_east, e_west)
    assertEqual(greenEast, false, 'Group B edge not green in phase 0');
    assertEqual(greenWest, false, 'Group B edge not green in phase 0');
})();

// ============================================================
// Tick — phase cycling
// ============================================================

console.log('\n--- Phase cycling ---');

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext('_ctrl.phaseTimer = 0;', ctx);
    // Phase 0 is green with duration 20s
    // Tick 20s => should advance to phase 1
    vm.runInContext('_ctrl.tick(20);', ctx);
    const phase = vm.runInContext('_ctrl.currentPhase', ctx);
    assertEqual(phase, 1, 'After 20s, advances from green to yellow');
})();

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext('_ctrl.phaseTimer = 0;', ctx);
    // Full cycle: 20 + 2 + 1 + 20 + 2 + 1 = 46s
    // Tick through phases 0-5, should wrap to 0
    vm.runInContext(`
        _ctrl.tick(20); // 0->1
        _ctrl.tick(2);  // 1->2
        _ctrl.tick(1);  // 2->3
        _ctrl.tick(20); // 3->4
        _ctrl.tick(2);  // 4->5
        _ctrl.tick(1);  // 5->0
    `, ctx);
    const phase = vm.runInContext('_ctrl.currentPhase', ctx);
    assertEqual(phase, 0, 'Full cycle returns to phase 0');
})();

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext('_ctrl.phaseTimer = 0;', ctx);
    // Small ticks should not advance phase
    vm.runInContext('_ctrl.tick(5);', ctx);
    const phase = vm.runInContext('_ctrl.currentPhase', ctx);
    assertEqual(phase, 0, '5s tick does not change 20s green phase');
    const timer = vm.runInContext('_ctrl.phaseTimer', ctx);
    assertApprox(timer, 5, 0.01, 'Phase timer accumulates');
})();

// ============================================================
// getSignalColor
// ============================================================

console.log('\n--- getSignalColor ---');

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext('_ctrl.phaseTimer = 0; _ctrl.currentPhase = 0;', ctx);
    const color = vm.runInContext('_ctrl.getSignalColor("e_north")', ctx);
    assertEqual(color, 'green', 'Group A edge shows green in phase 0');
})();

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext('_ctrl.currentPhase = 0;', ctx);
    const color = vm.runInContext('_ctrl.getSignalColor("e_east")', ctx);
    assertEqual(color, 'red', 'Group B edge shows red in phase 0');
})();

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext('_ctrl.currentPhase = 1;', ctx);
    // Phase 1 is yellow for group A
    const color = vm.runInContext('_ctrl.getSignalColor("e_north")', ctx);
    assertEqual(color, 'yellow', 'Group A edge shows yellow in yellow phase');
})();

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext('_ctrl.currentPhase = 2;', ctx);
    // Phase 2 is allred
    const color = vm.runInContext('_ctrl.getSignalColor("e_north")', ctx);
    assertEqual(color, 'red', 'All edges red in allred phase');
    const color2 = vm.runInContext('_ctrl.getSignalColor("e_east")', ctx);
    assertEqual(color2, 'red', 'Group B also red in allred phase');
})();

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext('_ctrl.currentPhase = 3;', ctx);
    // Phase 3 is green for group B
    const colorB = vm.runInContext('_ctrl.getSignalColor("e_east")', ctx);
    assertEqual(colorB, 'green', 'Group B edge shows green in phase 3');
    const colorA = vm.runInContext('_ctrl.getSignalColor("e_north")', ctx);
    assertEqual(colorA, 'red', 'Group A edge shows red in phase 3');
})();

// ============================================================
// isYellow
// ============================================================

console.log('\n--- isYellow ---');

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext('_ctrl.currentPhase = 0;', ctx);
    assertEqual(vm.runInContext('_ctrl.isYellow()', ctx), false, 'Not yellow in green phase');
    vm.runInContext('_ctrl.currentPhase = 1;', ctx);
    assertEqual(vm.runInContext('_ctrl.isYellow()', ctx), true, 'Yellow in yellow phase');
    vm.runInContext('_ctrl.currentPhase = 2;', ctx);
    assertEqual(vm.runInContext('_ctrl.isYellow()', ctx), false, 'Not yellow in allred phase');
})();

// ============================================================
// RL mode — tick does nothing
// ============================================================

console.log('\n--- RL mode ---');

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext('_ctrl.mode = "rl"; _ctrl.phaseTimer = 0;', ctx);
    vm.runInContext('_ctrl.tick(100);', ctx);
    const timer = vm.runInContext('_ctrl.phaseTimer', ctx);
    assertEqual(timer, 0, 'RL mode: tick does not advance timer');
    const phase = vm.runInContext('_ctrl.currentPhase', ctx);
    assertEqual(phase, 0, 'RL mode: phase does not change');
})();

// ============================================================
// Adaptive mode — extends green with queued vehicles
// ============================================================

console.log('\n--- Adaptive mode ---');

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext(`
        _ctrl.mode = 'adaptive';
        _ctrl.currentPhase = 0;
        _ctrl.phaseTimer = 0;
    `, ctx);
    // No queue counts — should use fixed duration
    vm.runInContext('_ctrl.updateQueueCounts({});', ctx);
    vm.runInContext('_ctrl.tick(20);', ctx);
    const phase = vm.runInContext('_ctrl.currentPhase', ctx);
    assertEqual(phase, 1, 'Adaptive with no queues: standard phase advance at 20s');
})();

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext(`
        _ctrl.mode = 'adaptive';
        _ctrl.currentPhase = 0;
        _ctrl.phaseTimer = 0;
    `, ctx);
    // Queue 5 vehicles on a green-phase edge
    vm.runInContext('_ctrl.updateQueueCounts({ "e_north": 5 });', ctx);
    // 20s (base) + 5*3 (extension) = 35s needed to advance
    vm.runInContext('_ctrl.tick(20);', ctx);
    const phase20 = vm.runInContext('_ctrl.currentPhase', ctx);
    assertEqual(phase20, 0, 'Adaptive: green extended beyond 20s with 5 queued vehicles');
    vm.runInContext('_ctrl.tick(15);', ctx);
    const phase35 = vm.runInContext('_ctrl.currentPhase', ctx);
    assertEqual(phase35, 1, 'Adaptive: phase advances after extended green (35s)');
})();

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext(`
        _ctrl.mode = 'adaptive';
        _ctrl.currentPhase = 0;
        _ctrl.phaseTimer = 0;
        _ctrl.maxGreen = 30;
    `, ctx);
    // Queue 20 vehicles => extension would be 20 + 20*3 = 80, but capped at maxGreen=30
    vm.runInContext('_ctrl.updateQueueCounts({ "e_north": 20 });', ctx);
    vm.runInContext('_ctrl.tick(30);', ctx);
    const phase = vm.runInContext('_ctrl.currentPhase', ctx);
    assertEqual(phase, 1, 'Adaptive: green capped at maxGreen');
})();

// ============================================================
// Adaptive mode — skip to next green if no demand
// ============================================================

console.log('\n--- Adaptive skip on no demand ---');

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext(`
        _ctrl.mode = 'adaptive';
        _ctrl.currentPhase = 0;
        _ctrl.phaseTimer = 0;
        _ctrl.minGreen = 5;
    `, ctx);
    // No vehicles on group A edges, but vehicles waiting on group B
    vm.runInContext('_ctrl.updateQueueCounts({ "e_east": 3 });', ctx);
    // Should use minGreen=5 instead of full duration
    vm.runInContext('_ctrl.tick(5);', ctx);
    const phase = vm.runInContext('_ctrl.currentPhase', ctx);
    assertEqual(phase, 1, 'Adaptive: skips to yellow with minGreen when no demand on current phase');
})();

// ============================================================
// updateQueueCounts
// ============================================================

console.log('\n--- updateQueueCounts ---');

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext('_ctrl.updateQueueCounts({ "e_north": 3, "e_east": 1 });', ctx);
    const counts = vm.runInContext('JSON.stringify(_ctrl._queueCounts)', ctx);
    const parsed = JSON.parse(counts);
    assertEqual(parsed.e_north, 3, 'Queue count set for e_north');
    assertEqual(parsed.e_east, 1, 'Queue count set for e_east');
})();

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext('_ctrl.updateQueueCounts(null);', ctx);
    const counts = vm.runInContext('JSON.stringify(_ctrl._queueCounts)', ctx);
    assertEqual(counts, '{}', 'null updates to empty counts');
})();

// ============================================================
// TrafficControllerManager
// ============================================================

console.log('\n--- TrafficControllerManager ---');

(function() {
    const ctx = createContext();
    vm.runInContext('var _mgr = new TrafficControllerManager();', ctx);
    const numControllers = vm.runInContext('Object.keys(_mgr.controllers).length', ctx);
    assertEqual(numControllers, 0, 'New manager has 0 controllers');
})();

// ============================================================
// Manager initFromNetwork
// ============================================================

console.log('\n--- Manager initFromNetwork ---');

(function() {
    const ctx = createContext();
    // Create a mock road network with a 4-way and 2-way intersections
    vm.runInContext(`
        var _mgr = new TrafficControllerManager();
        var _rn = {
            nodes: {
                n0: { id: 'n0', x: 0, z: 0, degree: 1 },
                n1: { id: 'n1', x: 50, z: 0, degree: 4 },
                n2: { id: 'n2', x: 100, z: 0, degree: 2 },
                n3: { id: 'n3', x: 50, z: 50, degree: 1 },
                n4: { id: 'n4', x: 50, z: -50, degree: 1 },
            },
            edges: [
                { id: 'e0', from: 'n0', to: 'n1', length: 50 },
                { id: 'e1', from: 'n1', to: 'n2', length: 50 },
                { id: 'e2', from: 'n1', to: 'n3', length: 50 },
                { id: 'e3', from: 'n1', to: 'n4', length: 50 },
            ],
            adjList: {
                n0: [0],
                n1: [0, 1, 2, 3],
                n2: [1],
                n3: [2],
                n4: [3],
            },
        };
        _mgr.initFromNetwork(_rn);
    `, ctx);
    const numControllers = vm.runInContext('Object.keys(_mgr.controllers).length', ctx);
    assertEqual(numControllers, 1, 'Only 1 controller for the 4-way (degree >= 3)');
    const hasN1 = vm.runInContext('"n1" in _mgr.controllers', ctx);
    assertEqual(hasN1, true, 'Controller created for n1 (4-way)');
    const hasN2 = vm.runInContext('"n2" in _mgr.controllers', ctx);
    assertEqual(hasN2, false, 'No controller for n2 (2-way)');
})();

// ============================================================
// Manager setMode
// ============================================================

console.log('\n--- Manager setMode ---');

(function() {
    const ctx = createContext();
    vm.runInContext(`
        var _mgr = new TrafficControllerManager();
        var _rn = {
            nodes: {
                n0: { id: 'n0', x: 0, z: 0, degree: 1 },
                n1: { id: 'n1', x: 50, z: 0, degree: 3 },
                n2: { id: 'n2', x: 100, z: 0, degree: 1 },
                n3: { id: 'n3', x: 50, z: 50, degree: 1 },
            },
            edges: [
                { id: 'e0', from: 'n0', to: 'n1', length: 50 },
                { id: 'e1', from: 'n1', to: 'n2', length: 50 },
                { id: 'e2', from: 'n1', to: 'n3', length: 50 },
            ],
            adjList: { n0: [0], n1: [0, 1, 2], n2: [1], n3: [2] },
        };
        _mgr.initFromNetwork(_rn);
        _mgr.setMode('adaptive');
    `, ctx);
    const ctrlMode = vm.runInContext('_mgr.controllers.n1.mode', ctx);
    assertEqual(ctrlMode, 'adaptive', 'setMode propagates to all controllers');
})();

// ============================================================
// Manager isGreen
// ============================================================

console.log('\n--- Manager isGreen ---');

(function() {
    const ctx = createContext();
    vm.runInContext(`
        var _mgr = new TrafficControllerManager();
        var _rn = {
            nodes: {
                n0: { id: 'n0', x: 0, z: 0, degree: 1 },
                n1: { id: 'n1', x: 50, z: 0, degree: 3 },
                n2: { id: 'n2', x: 100, z: 0, degree: 1 },
                n3: { id: 'n3', x: 50, z: 50, degree: 1 },
            },
            edges: [
                { id: 'e0', from: 'n0', to: 'n1', length: 50 },
                { id: 'e1', from: 'n1', to: 'n2', length: 50 },
                { id: 'e2', from: 'n1', to: 'n3', length: 50 },
            ],
            adjList: { n0: [0], n1: [0, 1, 2], n2: [1], n3: [2] },
        };
        _mgr.initFromNetwork(_rn);
    `, ctx);
    // No controller for unmanaged node => always green
    const isGreen = vm.runInContext('_mgr.isGreen("n0", "e0")', ctx);
    assertEqual(isGreen, true, 'Uncontrolled node always returns green');
})();

// ============================================================
// Manager getSignalColor for uncontrolled node
// ============================================================

console.log('\n--- Manager getSignalColor ---');

(function() {
    const ctx = createContext();
    vm.runInContext(`
        var _mgr = new TrafficControllerManager();
        _mgr.controllers = {};
    `, ctx);
    const color = vm.runInContext('_mgr.getSignalColor("nonexistent", "e0")', ctx);
    assertEqual(color, 'green', 'Uncontrolled node signal color is green');
})();

// ============================================================
// Manager tick
// ============================================================

console.log('\n--- Manager tick ---');

(function() {
    const ctx = createContext();
    vm.runInContext(`
        var _mgr = new TrafficControllerManager();
        var _rn = {
            nodes: {
                n0: { id: 'n0', x: 0, z: 0, degree: 1 },
                n1: { id: 'n1', x: 50, z: 0, degree: 3 },
                n2: { id: 'n2', x: 100, z: 0, degree: 1 },
                n3: { id: 'n3', x: 50, z: 50, degree: 1 },
            },
            edges: [
                { id: 'e0', from: 'n0', to: 'n1', length: 50 },
                { id: 'e1', from: 'n1', to: 'n2', length: 50 },
                { id: 'e2', from: 'n1', to: 'n3', length: 50 },
            ],
            adjList: { n0: [0], n1: [0, 1, 2], n2: [1], n3: [2] },
        };
        _mgr.initFromNetwork(_rn);
        _mgr.controllers.n1.phaseTimer = 0;
    `, ctx);
    vm.runInContext('_mgr.tick(5);', ctx);
    const timer = vm.runInContext('_mgr.controllers.n1.phaseTimer', ctx);
    assertApprox(timer, 5, 0.01, 'Manager tick advances controller timers');
})();

// ============================================================
// Manager updateQueues
// ============================================================

console.log('\n--- Manager updateQueues ---');

(function() {
    const ctx = createContext();
    vm.runInContext(`
        var _mgr = new TrafficControllerManager();
        var _rn = {
            nodes: {
                n0: { id: 'n0', x: 0, z: 0, degree: 1 },
                n1: { id: 'n1', x: 50, z: 0, degree: 3 },
                n2: { id: 'n2', x: 100, z: 0, degree: 1 },
                n3: { id: 'n3', x: 50, z: 50, degree: 1 },
            },
            edges: [
                { id: 'e0', from: 'n0', to: 'n1', length: 50 },
                { id: 'e1', from: 'n1', to: 'n2', length: 50 },
                { id: 'e2', from: 'n1', to: 'n3', length: 50 },
            ],
            adjList: { n0: [0], n1: [0, 1, 2], n2: [1], n3: [2] },
        };
        _mgr.initFromNetwork(_rn);
        // Simulate vehicles near stop line
        var _vehicles = [
            { edge: { id: 'e0', from: 'n0', to: 'n1', length: 50 }, direction: 1, u: 40, parked: false },
            { edge: { id: 'e0', from: 'n0', to: 'n1', length: 50 }, direction: 1, u: 35, parked: false },
        ];
        _mgr.updateQueues(_vehicles);
    `, ctx);
    const counts = vm.runInContext('JSON.stringify(_mgr.controllers.n1._queueCounts)', ctx);
    const parsed = JSON.parse(counts);
    assertEqual(parsed.e0, 2, 'Queue counts updated: 2 vehicles near stop line on e0');
})();

(function() {
    const ctx = createContext();
    vm.runInContext(`
        var _mgr = new TrafficControllerManager();
        var _rn = {
            nodes: {
                n0: { id: 'n0', x: 0, z: 0, degree: 1 },
                n1: { id: 'n1', x: 50, z: 0, degree: 3 },
                n2: { id: 'n2', x: 100, z: 0, degree: 1 },
                n3: { id: 'n3', x: 50, z: 50, degree: 1 },
            },
            edges: [
                { id: 'e0', from: 'n0', to: 'n1', length: 50 },
                { id: 'e1', from: 'n1', to: 'n2', length: 50 },
                { id: 'e2', from: 'n1', to: 'n3', length: 50 },
            ],
            adjList: { n0: [0], n1: [0, 1, 2], n2: [1], n3: [2] },
        };
        _mgr.initFromNetwork(_rn);
        // Parked vehicle should not count
        var _vehicles = [
            { edge: { id: 'e0', from: 'n0', to: 'n1', length: 50 }, direction: 1, u: 40, parked: true },
        ];
        _mgr.updateQueues(_vehicles);
    `, ctx);
    const counts = vm.runInContext('JSON.stringify(_mgr.controllers.n1._queueCounts)', ctx);
    const parsed = JSON.parse(counts);
    assert(!parsed.e0, 'Parked vehicles not counted in queue');
})();

// ============================================================
// Manager getSignalStates
// ============================================================

console.log('\n--- Manager getSignalStates ---');

(function() {
    const ctx = createContext();
    vm.runInContext(`
        var _mgr = new TrafficControllerManager();
        var _rn = {
            nodes: {
                n0: { id: 'n0', x: 0, z: 0, degree: 1 },
                n1: { id: 'n1', x: 50, z: 0, degree: 3 },
                n2: { id: 'n2', x: 100, z: 0, degree: 1 },
                n3: { id: 'n3', x: 50, z: 50, degree: 1 },
            },
            edges: [
                { id: 'e0', from: 'n0', to: 'n1', length: 50 },
                { id: 'e1', from: 'n1', to: 'n2', length: 50 },
                { id: 'e2', from: 'n1', to: 'n3', length: 50 },
            ],
            adjList: { n0: [0], n1: [0, 1, 2], n2: [1], n3: [2] },
        };
        _mgr.initFromNetwork(_rn);
        var _states = _mgr.getSignalStates(_rn);
    `, ctx);
    const numStates = vm.runInContext('_states.length', ctx);
    assertEqual(numStates, 3, 'getSignalStates returns one entry per edge per controller');
    // Each state should have nodeId, x, z, edgeId, color
    const state0 = vm.runInContext('JSON.stringify(_states[0])', ctx);
    const s = JSON.parse(state0);
    assert(s.nodeId !== undefined, 'Signal state has nodeId');
    assert(s.x !== undefined, 'Signal state has x');
    assert(s.z !== undefined, 'Signal state has z');
    assert(s.edgeId !== undefined, 'Signal state has edgeId');
    assert(['green', 'yellow', 'red'].includes(s.color), 'Signal state has valid color');
})();

// ============================================================
// Stagger offset
// ============================================================

console.log('\n--- Stagger offset ---');

(function() {
    const ctx = createContext();
    create4Way(ctx);
    const stagger = vm.runInContext('_ctrl.staggerOffset', ctx);
    assertEqual(stagger, 0, 'Direct constructor has 0 stagger');
})();

// ============================================================
// _findNextGreenPhase
// ============================================================

console.log('\n--- _findNextGreenPhase ---');

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext('_ctrl.currentPhase = 0;', ctx);
    const next = vm.runInContext('_ctrl._findNextGreenPhase()', ctx);
    assertEqual(next, 3, 'From phase 0 (green A), next green is phase 3 (green B)');
})();

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext('_ctrl.currentPhase = 3;', ctx);
    const next = vm.runInContext('_ctrl._findNextGreenPhase()', ctx);
    assertEqual(next, 0, 'From phase 3 (green B), next green wraps to phase 0 (green A)');
})();

// ============================================================
// Adaptive — non-green phases use fixed duration
// ============================================================

console.log('\n--- Adaptive non-green phases ---');

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext(`
        _ctrl.mode = 'adaptive';
        _ctrl.currentPhase = 1; // yellow
        _ctrl.phaseTimer = 0;
    `, ctx);
    const dur = vm.runInContext('_ctrl._getAdaptiveDuration()', ctx);
    assertEqual(dur, 2, 'Adaptive: yellow phase uses fixed 2s duration');
})();

(function() {
    const ctx = createContext();
    create4Way(ctx);
    vm.runInContext(`
        _ctrl.mode = 'adaptive';
        _ctrl.currentPhase = 2; // allred
        _ctrl.phaseTimer = 0;
    `, ctx);
    const dur = vm.runInContext('_ctrl._getAdaptiveDuration()', ctx);
    assertEqual(dur, 1, 'Adaptive: allred phase uses fixed 1s duration');
})();

// ============================================================
// Summary
// ============================================================

console.log(`\n${'='.repeat(50)}`);
console.log(`Traffic controller tests: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
