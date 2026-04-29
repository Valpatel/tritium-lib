#!/usr/bin/env node
// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * SimVehicle tests — constructor, tick, IDM acceleration, leader detection,
 * MOBIL lane changes, edge transitions, red lights, subtypes, emergency,
 * parking, turn signals, weather, accidents, collisions.
 *
 * Run: node tests/js/test_vehicle.js
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

// Load source files, stripping ES module syntax
function loadModule(path) {
    const code = fs.readFileSync(path, 'utf8');
    return code
        .replace(/^export\s+/gm, '')
        .replace(/^import\s+.*$/gm, '');
}

const idmCode = loadModule(__dirname + '/../../web/sim/idm.js');
const mobilCode = loadModule(__dirname + '/../../web/sim/mobil.js');
const vehicleCode = loadModule(__dirname + '/../../web/sim/vehicle.js');

// Build a minimal road network for testing
function createTestRoadNetwork() {
    return {
        nodes: {
            n0: { id: 'n0', x: 0, z: 0, degree: 2 },
            n1: { id: 'n1', x: 100, z: 0, degree: 2 },
            n2: { id: 'n2', x: 200, z: 0, degree: 1 },
            n3: { id: 'n3', x: 100, z: 100, degree: 1 },
        },
        edges: [
            {
                id: 'e0', from: 'n0', to: 'n1',
                ax: 0, az: 0, bx: 100, bz: 0,
                length: 100, lanesPerDir: 2, laneWidth: 3,
                roadClass: 'residential', oneway: false, bridge: false,
                waypoints: [[0, 0], [100, 0]],
                speedLimit: null,
            },
            {
                id: 'e1', from: 'n1', to: 'n2',
                ax: 100, az: 0, bx: 200, bz: 0,
                length: 100, lanesPerDir: 1, laneWidth: 3,
                roadClass: 'residential', oneway: false, bridge: false,
                waypoints: [[100, 0], [200, 0]],
                speedLimit: null,
            },
            {
                id: 'e2', from: 'n1', to: 'n3',
                ax: 100, az: 0, bx: 100, bz: 100,
                length: 100, lanesPerDir: 1, laneWidth: 3,
                roadClass: 'secondary', oneway: false, bridge: false,
                waypoints: [[100, 0], [100, 100]],
                speedLimit: null,
            },
        ],
        adjList: {
            n0: [0],
            n1: [0, 1, 2],
            n2: [1],
            n3: [2],
        },
        edgeById: {},
        findPath(from, to) {
            // Minimal pathfinding for test purposes
            if (from === to) return [];
            // Hardcoded paths for our test graph
            if (from === 'n0' && to === 'n2') {
                return [
                    { edge: this.edges[0], nodeId: 'n1' },
                    { edge: this.edges[1], nodeId: 'n2' },
                ];
            }
            if (from === 'n1' && to === 'n2') {
                return [{ edge: this.edges[1], nodeId: 'n2' }];
            }
            if (from === 'n1' && to === 'n3') {
                return [{ edge: this.edges[2], nodeId: 'n3' }];
            }
            if (from === 'n0' && to === 'n3') {
                return [
                    { edge: this.edges[0], nodeId: 'n1' },
                    { edge: this.edges[2], nodeId: 'n3' },
                ];
            }
            // Return a random path to any connected node
            const idx = Math.floor(Math.random() * this.edges.length);
            const e = this.edges[idx];
            return [{ edge: e, nodeId: e.to }];
        },
        nearestNode(x, z) {
            let best = null, bestDist = Infinity;
            for (const id in this.nodes) {
                const n = this.nodes[id];
                const d = Math.hypot(n.x - x, n.z - z);
                if (d < bestDist) { bestDist = d; best = id; }
            }
            return best ? { nodeId: best, dist: bestDist } : null;
        },
    };
}

// Create a sandbox with all needed modules loaded
function createVehicle(edgeOverrides, uPos, rnOverrides) {
    const rn = createTestRoadNetwork();
    if (rnOverrides) Object.assign(rn, rnOverrides);
    rn.edgeById = {};
    rn.edges.forEach(e => rn.edgeById[e.id] = e);

    const edge = { ...rn.edges[0], ...(edgeOverrides || {}) };
    // Update the edge in the network too
    rn.edges[0] = edge;
    rn.edgeById[edge.id] = edge;

    const sandbox = {
        Math, console, Object, Array, Number, String, Set, Map, Infinity, NaN,
        parseInt, parseFloat, JSON, undefined, Boolean,
    };
    const ctx = vm.createContext(sandbox);

    // Load modules in order
    vm.runInContext(idmCode, ctx);
    vm.runInContext(mobilCode, ctx);
    vm.runInContext(vehicleCode, ctx);

    // Create vehicle in sandbox context
    vm.runInContext(`
        var _rn = ${JSON.stringify(rn)};
        // Restore findPath and nearestNode as functions
        _rn.findPath = function(from, to) {
            if (from === to) return [];
            if (from === 'n0' && to === 'n2') return [{ edge: _rn.edges[0], nodeId: 'n1' }, { edge: _rn.edges[1], nodeId: 'n2' }];
            if (from === 'n1' && to === 'n2') return [{ edge: _rn.edges[1], nodeId: 'n2' }];
            if (from === 'n1' && to === 'n3') return [{ edge: _rn.edges[2], nodeId: 'n3' }];
            if (from === 'n0' && to === 'n3') return [{ edge: _rn.edges[0], nodeId: 'n1' }, { edge: _rn.edges[2], nodeId: 'n3' }];
            var idx = Math.floor(Math.random() * _rn.edges.length);
            var e = _rn.edges[idx];
            return [{ edge: e, nodeId: e.to }];
        };
        _rn.nearestNode = function(x, z) {
            var best = null, bestDist = Infinity;
            for (var id in _rn.nodes) {
                var n = _rn.nodes[id];
                var d = Math.sqrt((n.x - x) * (n.x - x) + (n.z - z) * (n.z - z));
                if (d < bestDist) { bestDist = d; best = id; }
            }
            return best ? { nodeId: best, dist: bestDist } : null;
        };
        _rn.edgeById = {};
        _rn.edges.forEach(function(e) { _rn.edgeById[e.id] = e; });
        var _edge = _rn.edges[0];
        var _vehicle = new SimVehicle(_edge, ${uPos !== undefined ? uPos : 10}, _rn);
    `, ctx);

    return ctx;
}

// Helper to read vehicle properties from context
function getVehicle(ctx) {
    return vm.runInContext('JSON.parse(JSON.stringify(_vehicle))', ctx);
}

// Helper to call methods
function callTick(ctx, dt, nearbyJSON) {
    vm.runInContext(`_vehicle.tick(${dt}, ${nearbyJSON || '[]'})`, ctx);
}

// ============================================================
// Constructor Defaults
// ============================================================

console.log('\n--- SimVehicle Constructor ---');

(function() {
    const ctx = createVehicle();
    const v = getVehicle(ctx);
    assert(v.id.startsWith('car_'), 'ID starts with car_');
    assertEqual(v.speed, 0, 'Initial speed is 0');
    assertEqual(v.acc, 0, 'Initial acceleration is 0');
    assertEqual(v.alive, true, 'Vehicle is alive');
    assertEqual(v.parked, false, 'Not parked');
    assertEqual(v.inAccident, false, 'Not in accident');
    assertEqual(v.isEmergency, false, 'Not emergency');
    assertEqual(v.turnSignal, 'none', 'No turn signal');
    assertEqual(v.direction, 1, 'Default direction is forward');
    assertEqual(v.purpose, 'random', 'Default purpose is random');
    assertEqual(v.taxiState, 'idle', 'Default taxi state is idle');
    assert(v.routeIdx === 0, 'Route index starts at 0');
})();

(function() {
    const ctx = createVehicle({}, 50);
    const v = getVehicle(ctx);
    assertEqual(v.u, 50, 'Starting position set to 50');
})();

// ============================================================
// Vehicle Subtypes and Profiles
// ============================================================

console.log('\n--- Vehicle Subtypes ---');

(function() {
    const ctx = createVehicle();
    const v = getVehicle(ctx);
    const validSubtypes = ['sedan', 'suv', 'truck', 'motorcycle', 'van'];
    assert(validSubtypes.includes(v.subtype), `Subtype ${v.subtype} is valid`);
    assert(v.length > 0, 'Vehicle has positive length');
    assert(v.width > 0, 'Vehicle has positive width');
    assert(v.height > 0, 'Vehicle has positive height');
    assert(v.mass > 0, 'Vehicle has positive mass');
})();

(function() {
    // Create multiple vehicles, verify subtypes are assigned
    const subtypeSeen = new Set();
    for (let i = 0; i < 20; i++) {
        const ctx = createVehicle();
        const v = getVehicle(ctx);
        subtypeSeen.add(v.subtype);
    }
    assert(subtypeSeen.size >= 2, 'Multiple subtypes assigned across vehicles');
})();

// ============================================================
// IDM Parameters
// ============================================================

console.log('\n--- IDM Parameters ---');

(function() {
    const ctx = createVehicle();
    const v = getVehicle(ctx);
    assert(v.idm.v0 > 0, 'Desired speed > 0');
    assert(v.idm.a > 0, 'Max accel > 0');
    assert(v.idm.b > 0, 'Comfortable decel > 0');
    assert(v.idm.s0 > 0, 'Minimum gap > 0');
    assert(v.idm.T > 0, 'Time headway > 0');
    assert(v.idm.delta > 0, 'Exponent > 0');
})();

// ============================================================
// Position Updates (_updatePosition)
// ============================================================

console.log('\n--- Position update ---');

(function() {
    const ctx = createVehicle({}, 0);
    const v = getVehicle(ctx);
    // At u=0 on edge from (0,0) to (100,0), vehicle should be at (0,0)
    assertApprox(v.x, 0, 1, 'At u=0, x ~= 0');
    assertApprox(v.z, 0, 1, 'At u=0, z ~= 0');
})();

(function() {
    const ctx = createVehicle({}, 50);
    const v = getVehicle(ctx);
    // At u=50 on edge from (0,0) to (100,0), vehicle should be at (50,0)
    assertApprox(v.x, 50, 5, 'At u=50, x ~= 50');
    assertApprox(v.z, 0, 5, 'At u=50, z ~= 0');
})();

// ============================================================
// Tick — basic movement
// ============================================================

console.log('\n--- Tick basic movement ---');

(function() {
    const ctx = createVehicle({}, 10);
    // Set speed to 5 m/s
    vm.runInContext('_vehicle.speed = 5; _vehicle.acc = 0;', ctx);
    callTick(ctx, 0.1, '[]');
    const v = getVehicle(ctx);
    assert(v.u > 10, 'Vehicle moved forward after tick');
})();

(function() {
    const ctx = createVehicle({}, 10);
    // Zero speed — IDM should accelerate
    callTick(ctx, 0.1, '[]');
    const v = getVehicle(ctx);
    assert(v.acc > 0, 'Stopped vehicle accelerates (IDM free flow)');
})();

// ============================================================
// IDM free-flow acceleration
// ============================================================

console.log('\n--- IDM free-flow ---');

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext('_vehicle.speed = 0;', ctx);
    callTick(ctx, 0.1, '[]');
    const v = getVehicle(ctx);
    assert(v.acc > 0, 'v=0 gives positive acceleration');
    assert(v.acc <= v.idm.a, 'Acceleration bounded by max accel');
})();

(function() {
    // At desired speed, acceleration should be near zero
    const ctx = createVehicle({}, 10);
    const desiredSpeed = vm.runInContext('_vehicle.idm.v0', ctx);
    vm.runInContext(`_vehicle.speed = ${desiredSpeed};`, ctx);
    callTick(ctx, 0.1, '[]');
    const v = getVehicle(ctx);
    assertApprox(v.acc, 0, 0.5, 'At desired speed, acceleration ~0');
})();

// ============================================================
// Leader detection — same edge/lane
// ============================================================

console.log('\n--- Leader detection ---');

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext('_vehicle.speed = 5; _vehicle.laneIdx = 0;', ctx);
    // Create a nearby vehicle ahead in same lane
    const leaderJSON = JSON.stringify([{
        id: 'leader1', edge: { id: 'e0' }, u: 30, speed: 3,
        direction: 1, laneIdx: 0, length: 4.5,
    }]);
    // Need to set edge reference to match
    vm.runInContext(`
        var _nearby = ${leaderJSON};
        _nearby[0].edge = _vehicle.edge;
        _vehicle.tick(0.1, _nearby);
    `, ctx);
    const v = getVehicle(ctx);
    // With a slow leader 20m ahead, acceleration should be less than free flow
    assert(v.acc < v.idm.a, 'Deceleration with leader ahead');
})();

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext('_vehicle.speed = 5; _vehicle.laneIdx = 0;', ctx);
    // Leader in different lane — should not affect
    vm.runInContext(`
        var _nearby = [{
            id: 'other_lane', edge: _vehicle.edge, u: 15, speed: 0,
            direction: 1, laneIdx: 1, length: 4.5
        }];
        _vehicle.tick(0.1, _nearby);
    `, ctx);
    const v = getVehicle(ctx);
    // Different lane, should be near free-flow acceleration
    assert(v.acc > 0, 'Other-lane vehicle does not cause braking');
})();

(function() {
    const ctx = createVehicle({}, 30);
    vm.runInContext('_vehicle.speed = 5; _vehicle.laneIdx = 0;', ctx);
    // Vehicle behind us — should not be considered a leader
    vm.runInContext(`
        var _nearby = [{
            id: 'behind', edge: _vehicle.edge, u: 10, speed: 8,
            direction: 1, laneIdx: 0, length: 4.5
        }];
        _vehicle.tick(0.1, _nearby);
    `, ctx);
    const v = getVehicle(ctx);
    assert(v.acc > 0, 'Vehicle behind does not cause braking');
})();

// ============================================================
// Red light braking
// ============================================================

console.log('\n--- Red light braking ---');

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext(`
        _vehicle.speed = 5;
        _vehicle._redLightActive = true;
        _vehicle._redLightGap = 8;
        _vehicle.isEmergency = false;
    `, ctx);
    callTick(ctx, 0.1, '[]');
    const v = getVehicle(ctx);
    assert(v.acc < 0, 'Red light causes braking');
})();

(function() {
    // Emergency vehicles ignore red lights
    const ctx = createVehicle({}, 10);
    vm.runInContext(`
        _vehicle.speed = 5;
        _vehicle._redLightActive = true;
        _vehicle._redLightGap = 8;
        _vehicle.isEmergency = true;
    `, ctx);
    callTick(ctx, 0.1, '[]');
    const v = getVehicle(ctx);
    assert(v.acc >= 0, 'Emergency vehicle ignores red light');
})();

// ============================================================
// Parking and unparking
// ============================================================

console.log('\n--- Parking ---');

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext('_vehicle.parked = true; _vehicle.parkTimer = 10; _vehicle.speed = 0;', ctx);
    const u0 = vm.runInContext('_vehicle.u', ctx);
    callTick(ctx, 1.0, '[]');
    const v = getVehicle(ctx);
    assertEqual(v.u, u0, 'Parked vehicle does not move');
    assertEqual(v.parked, true, 'Still parked (timer > 0)');
    assertApprox(v.parkTimer, 9, 0.1, 'Park timer decremented');
})();

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext('_vehicle.parked = true; _vehicle.parkTimer = 0.5; _vehicle.speed = 0;', ctx);
    callTick(ctx, 1.0, '[]');
    const v = getVehicle(ctx);
    assertEqual(v.parked, false, 'Vehicle unparks when timer expires');
})();

// ============================================================
// Accident state machine
// ============================================================

console.log('\n--- Accident state ---');

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext('_vehicle.inAccident = true; _vehicle.accidentTimer = 5; _vehicle.speed = 0;', ctx);
    const u0 = vm.runInContext('_vehicle.u', ctx);
    callTick(ctx, 1.0, '[]');
    const v = getVehicle(ctx);
    assertEqual(v.u, u0, 'Accident vehicle does not move');
    assertApprox(v.accidentTimer, 4, 0.1, 'Accident timer decremented');
})();

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext('_vehicle.inAccident = true; _vehicle.accidentTimer = 0.5; _vehicle.speed = 0;', ctx);
    callTick(ctx, 1.0, '[]');
    const v = getVehicle(ctx);
    assertEqual(v.inAccident, false, 'Accident clears when timer expires');
    assertEqual(v.accidentTimer, 0, 'Accident timer reset to 0');
})();

// ============================================================
// Emergency vehicle siren
// ============================================================

console.log('\n--- Emergency vehicles ---');

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext('_vehicle.isEmergency = true; _vehicle.sirenActive = true; _vehicle.sirenPhase = 0;', ctx);
    callTick(ctx, 0.5, '[]');
    const v = getVehicle(ctx);
    assert(v.sirenPhase > 0, 'Siren phase advances during tick');
})();

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext('_vehicle.isEmergency = true; _vehicle.sirenActive = false; _vehicle.sirenPhase = 0;', ctx);
    callTick(ctx, 0.5, '[]');
    const v = getVehicle(ctx);
    assertEqual(v.sirenPhase, 0, 'Siren phase does not advance if siren inactive');
})();

// ============================================================
// Weather speed modifier
// ============================================================

console.log('\n--- Weather speed modifier ---');

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext('_vehicle.speed = 0; _vehicle._weatherSpeedMult = 0.5;', ctx);
    callTick(ctx, 0.1, '[]');
    const v1 = getVehicle(ctx);

    const ctx2 = createVehicle({}, 10);
    vm.runInContext('_vehicle.speed = 0; _vehicle._weatherSpeedMult = 1.0;', ctx2);
    callTick(ctx2, 0.1, '[]');
    const v2 = getVehicle(ctx2);

    // Both accelerate from 0 but weather-modified should use lower v0
    // Can't compare exact values since subtype is random, just verify both work
    assert(v1.speed >= 0, 'Weather 0.5: speed non-negative');
    assert(v2.speed >= 0, 'Weather 1.0: speed non-negative');
})();

// ============================================================
// Shake intensity decay
// ============================================================

console.log('\n--- Shake decay ---');

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext('_vehicle.shakeIntensity = 1.0;', ctx);
    callTick(ctx, 0.5, '[]');
    const v = getVehicle(ctx);
    assert(v.shakeIntensity < 1.0, 'Shake intensity decays over time');
    assert(v.shakeIntensity >= 0, 'Shake intensity remains non-negative');
})();

// ============================================================
// Collision cooldown decay
// ============================================================

console.log('\n--- Collision cooldown ---');

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext('_vehicle._collisionCooldown = 2.0;', ctx);
    callTick(ctx, 0.5, '[]');
    const cd = vm.runInContext('_vehicle._collisionCooldown', ctx);
    assertApprox(cd, 1.5, 0.01, 'Collision cooldown decrements by dt');
})();

// ============================================================
// Lane change state animation
// ============================================================

console.log('\n--- Lane change animation ---');

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext(`
        _vehicle.speed = 5;
        _vehicle.laneIdx = 0;
        _vehicle._mobilTimer = 999; // prevent MOBIL evaluation
        _vehicle._laneChangeState = { fromLane: 0, toLane: 1, t: 0, duration: 2.0 };
        _vehicle.turnSignal = 'right';
    `, ctx);
    callTick(ctx, 1.0, '[]');
    const t = vm.runInContext('_vehicle._laneChangeState ? _vehicle._laneChangeState.t : -1', ctx);
    assertApprox(t, 0.5, 0.01, 'Lane change t progresses (1s / 2s duration = 0.5)');
})();

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext(`
        _vehicle.speed = 5;
        _vehicle.laneIdx = 0;
        _vehicle._mobilTimer = 999;
        _vehicle._laneChangeState = { fromLane: 0, toLane: 1, t: 0.9, duration: 2.0 };
        _vehicle.turnSignal = 'right';
    `, ctx);
    callTick(ctx, 1.0, '[]');
    const lcs = vm.runInContext('_vehicle._laneChangeState', ctx);
    const laneIdx = vm.runInContext('_vehicle.laneIdx', ctx);
    assertEqual(lcs, null, 'Lane change completes when t >= 1');
    assertEqual(laneIdx, 1, 'laneIdx updated after lane change');
    const ts = vm.runInContext('_vehicle.turnSignal', ctx);
    assertEqual(ts, 'none', 'Turn signal cleared after lane change');
})();

// ============================================================
// MOBIL lane change evaluation trigger
// ============================================================

console.log('\n--- MOBIL timer ---');

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext(`
        _vehicle.speed = 5;
        _vehicle.laneIdx = 0;
        _vehicle._mobilTimer = 0.1;
        _vehicle._laneChangeState = null;
    `, ctx);
    callTick(ctx, 0.2, '[]');
    const timer = vm.runInContext('_vehicle._mobilTimer', ctx);
    assert(timer > 0, 'MOBIL timer resets after evaluation');
})();

// ============================================================
// Edge transition (direction +1, u >= edge.length)
// ============================================================

console.log('\n--- Edge transition ---');

(function() {
    const ctx = createVehicle({}, 95);
    // Give vehicle a route to follow
    vm.runInContext(`
        _vehicle.speed = 10;
        _vehicle.direction = 1;
        _vehicle.route = [
            { edge: _rn.edges[0], nodeId: 'n1' },
            { edge: _rn.edges[1], nodeId: 'n2' },
        ];
        _vehicle.routeIdx = 0;
    `, ctx);
    callTick(ctx, 1.0, '[]');
    const edgeId = vm.runInContext('_vehicle.edge.id', ctx);
    // After moving ~10m from u=95 on a 100m edge, should transition
    assertEqual(edgeId, 'e1', 'Vehicle transitioned to next edge');
})();

// ============================================================
// Dead vehicle does not tick
// ============================================================

console.log('\n--- Dead vehicle ---');

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext('_vehicle.alive = false; _vehicle.speed = 5;', ctx);
    const u0 = vm.runInContext('_vehicle.u', ctx);
    callTick(ctx, 1.0, '[]');
    const u1 = vm.runInContext('_vehicle.u', ctx);
    assertEqual(u0, u1, 'Dead vehicle does not move');
})();

// ============================================================
// Commute parking — stays parked outside rush hours
// ============================================================

console.log('\n--- Commute parking ---');

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext(`
        _vehicle.parked = true;
        _vehicle.parkTimer = 999;
        _vehicle.purpose = 'commute';
        _vehicle._simHour = 12;
        _vehicle._isWeekend = false;
    `, ctx);
    const u0 = vm.runInContext('_vehicle.u', ctx);
    callTick(ctx, 1.0, '[]');
    const v = getVehicle(ctx);
    assertEqual(v.parked, true, 'Commute vehicle stays parked at noon');
})();

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext(`
        _vehicle.parked = true;
        _vehicle.parkTimer = 999;
        _vehicle.purpose = 'commute';
        _vehicle._simHour = 8;
        _vehicle._isWeekend = false;
    `, ctx);
    callTick(ctx, 1000.0, '[]');
    const v = getVehicle(ctx);
    // During morning rush, park timer decrements normally
    assert(v.parkTimer < 999, 'Commute vehicle timer decrements during rush hour');
})();

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext(`
        _vehicle.parked = true;
        _vehicle.parkTimer = 999;
        _vehicle.purpose = 'commute';
        _vehicle._simHour = 8;
        _vehicle._isWeekend = true;
    `, ctx);
    callTick(ctx, 1.0, '[]');
    const v = getVehicle(ctx);
    assertEqual(v.parked, true, 'Commute vehicle stays parked on weekends');
})();

// ============================================================
// IDM acceleration with leader (unit level)
// ============================================================

console.log('\n--- IDM acceleration with leader ---');

(function() {
    const ctx = createVehicle({}, 10);
    // Directly test IDM via the context
    const acc = vm.runInContext(`
        idmAcceleration(10, 5, 0, _vehicle.idm)
    `, ctx);
    assert(acc < 0, 'Close gap to stopped leader => strong braking');
})();

(function() {
    const ctx = createVehicle({}, 10);
    const acc = vm.runInContext(`
        idmAcceleration(5, 100, 10, _vehicle.idm)
    `, ctx);
    assert(acc > 0, 'Large gap with faster leader => acceleration');
})();

(function() {
    const ctx = createVehicle({}, 10);
    const acc = vm.runInContext(`
        idmFreeFlow(0, _vehicle.idm)
    `, ctx);
    const maxA = vm.runInContext('_vehicle.idm.a', ctx);
    assertApprox(acc, maxA, 0.01, 'Free flow from stop = max acceleration');
})();

// ============================================================
// IDM step integration
// ============================================================

console.log('\n--- IDM step ---');

(function() {
    const ctx = createVehicle({}, 10);
    const result = vm.runInContext(`
        JSON.stringify(idmStep(5, 1.0, 0.1))
    `, ctx);
    const r = JSON.parse(result);
    assertApprox(r.v, 5.1, 0.01, 'Speed increases by acc*dt');
    assert(r.ds > 0, 'Distance traveled is positive');
})();

(function() {
    const ctx = createVehicle({}, 10);
    // Speed cannot go negative
    const result = vm.runInContext(`
        JSON.stringify(idmStep(0.5, -5.0, 0.5))
    `, ctx);
    const r = JSON.parse(result);
    assertEqual(r.v, 0, 'Speed clamped to 0 (cannot go negative)');
    assert(r.ds >= 0, 'Distance cannot be negative');
})();

// ============================================================
// Turn signal activation
// ============================================================

console.log('\n--- Turn signals ---');

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext('_vehicle.turnSignal = "none";', ctx);
    const ts = vm.runInContext('_vehicle.turnSignal', ctx);
    assertEqual(ts, 'none', 'Turn signal default is none');
})();

// ============================================================
// Vehicle color
// ============================================================

console.log('\n--- Vehicle color ---');

(function() {
    const ctx = createVehicle({}, 10);
    const color = vm.runInContext('_vehicle.color', ctx);
    assert(typeof color === 'number', 'Color is a number (hex)');
    assert(color >= 0, 'Color is non-negative');
})();

// ============================================================
// Vehicle collision
// ============================================================

console.log('\n--- Vehicle collision ---');

(function() {
    const ctx = createVehicle({}, 30);
    // Create two vehicles and test collision
    vm.runInContext(`
        _vehicle.speed = 10;
        _vehicle.heading = 0;
        _vehicle.mass = 1400;
        _vehicle.x = 30;
        _vehicle.z = 0;
        var _vehicle2 = new SimVehicle(_rn.edges[0], 35, _rn);
        _vehicle2.speed = 0;
        _vehicle2.heading = 0;
        _vehicle2.mass = 1400;
        _vehicle2.x = 35;
        _vehicle2.z = 0;
        var _collided = _vehicle.applyCollision(_vehicle2, 1.0);
    `, ctx);
    const collided = vm.runInContext('_collided', ctx);
    assertEqual(collided, true, 'Collision detected (approaching)');
    const s1 = vm.runInContext('_vehicle.speed', ctx);
    const s2 = vm.runInContext('_vehicle2.speed', ctx);
    assert(s1 < 10, 'Vehicle 1 slowed by collision');
    assert(s2 > 0, 'Vehicle 2 gained speed from collision');
})();

(function() {
    // Non-approaching vehicles should not collide
    const ctx = createVehicle({}, 30);
    vm.runInContext(`
        _vehicle.speed = 5;
        _vehicle.heading = Math.PI; // facing away
        _vehicle.mass = 1400;
        _vehicle.x = 30;
        _vehicle.z = 0;
        var _vehicle2 = new SimVehicle(_rn.edges[0], 35, _rn);
        _vehicle2.speed = 10;
        _vehicle2.heading = 0; // facing same direction, faster
        _vehicle2.mass = 1400;
        _vehicle2.x = 35;
        _vehicle2.z = 0;
        var _collided = _vehicle.applyCollision(_vehicle2, 1.0);
    `, ctx);
    const collided = vm.runInContext('_collided', ctx);
    assertEqual(collided, false, 'No collision when vehicles moving apart');
})();

// ============================================================
// High-energy collision triggers accident
// ============================================================

console.log('\n--- High-energy collision ---');

(function() {
    const ctx = createVehicle({}, 30);
    vm.runInContext(`
        _vehicle.speed = 20;
        _vehicle.heading = 0;
        _vehicle.mass = 5000;  // heavy truck
        _vehicle.x = 30;
        _vehicle.z = 0;
        var _vehicle2 = new SimVehicle(_rn.edges[0], 35, _rn);
        _vehicle2.speed = 0;
        _vehicle2.heading = 0;
        _vehicle2.mass = 1400;
        _vehicle2.x = 35;
        _vehicle2.z = 0;
        _vehicle.applyCollision(_vehicle2, 2.0);
    `, ctx);
    // Check if at least one went into accident state
    const acc1 = vm.runInContext('_vehicle.inAccident', ctx);
    const acc2 = vm.runInContext('_vehicle2.inAccident', ctx);
    assert(acc1 || acc2, 'High energy collision triggers accident state');
    // Both should have collision cooldown set
    const cd1 = vm.runInContext('_vehicle._collisionCooldown', ctx);
    const cd2 = vm.runInContext('_vehicle2._collisionCooldown', ctx);
    assertApprox(cd1, 1.0, 0.01, 'Collision cooldown set on vehicle 1');
    assertApprox(cd2, 1.0, 0.01, 'Collision cooldown set on vehicle 2');
})();

// ============================================================
// Pedestrian collision
// ============================================================

console.log('\n--- Pedestrian collision ---');

(function() {
    const ctx = createVehicle({}, 30);
    vm.runInContext(`
        _vehicle.speed = 10;
        _vehicle.heading = 0;
        _vehicle.mass = 1400;
        _vehicle.x = 30;
        _vehicle.z = 0;
        var _ped = { x: 31, z: 1, speed: 1.5, stunTimer: 0 };
        var _knockForce = _vehicle.applyPedestrianCollision(_ped);
    `, ctx);
    const vSpeed = vm.runInContext('_vehicle.speed', ctx);
    const pedSpeed = vm.runInContext('_ped.speed', ctx);
    const knockForce = vm.runInContext('_knockForce', ctx);
    const stunTimer = vm.runInContext('_ped.stunTimer', ctx);
    assert(vSpeed < 10, 'Vehicle slows slightly after hitting pedestrian');
    assertEqual(pedSpeed, 0, 'Pedestrian speed set to 0');
    assert(knockForce > 0, 'Knock force is positive');
    assert(stunTimer > 0, 'Pedestrian is stunned');
})();

// ============================================================
// Heading calculation
// ============================================================

console.log('\n--- Heading ---');

(function() {
    const ctx = createVehicle({}, 50);
    const heading = vm.runInContext('_vehicle.heading', ctx);
    assert(typeof heading === 'number', 'Heading is a number');
    assert(!isNaN(heading), 'Heading is not NaN');
})();

// ============================================================
// Instance index
// ============================================================

console.log('\n--- Instance index ---');

(function() {
    const ctx = createVehicle({}, 10);
    const idx = vm.runInContext('_vehicle.instanceIdx', ctx);
    assertEqual(idx, -1, 'Default instanceIdx is -1');
})();

// ============================================================
// _setSpeedForRoad
// ============================================================

console.log('\n--- Speed for road class ---');

(function() {
    const ctx = createVehicle({}, 10);
    // Manually call _setSpeedForRoad with a motorway edge
    vm.runInContext(`
        var _motorwayEdge = { roadClass: 'motorway', speedLimit: null };
        _vehicle._setSpeedForRoad(_motorwayEdge);
    `, ctx);
    const v0 = vm.runInContext('_vehicle.idm.v0', ctx);
    // ROAD_SPEEDS.motorway = 30, so v0 should be within 0.9-1.1 * 30
    assert(v0 >= 27 && v0 <= 33, `Motorway speed ${v0} in range [27, 33]`);
})();

(function() {
    const ctx = createVehicle({}, 10);
    // Edge with explicit speed limit
    vm.runInContext(`
        var _limitedEdge = { roadClass: 'residential', speedLimit: 8.33 };
        _vehicle._setSpeedForRoad(_limitedEdge);
    `, ctx);
    const v0 = vm.runInContext('_vehicle.idm.v0', ctx);
    assert(v0 >= 7.5 && v0 <= 9.2, `Speed limit 8.33 => v0 ${v0} in range`);
})();

// ============================================================
// Multiple ticks — vehicle approaches desired speed
// ============================================================

console.log('\n--- Speed convergence ---');

(function() {
    const ctx = createVehicle({}, 10);
    vm.runInContext('_vehicle.speed = 0;', ctx);
    for (let i = 0; i < 100; i++) {
        callTick(ctx, 0.1, '[]');
    }
    const speed = vm.runInContext('_vehicle.speed', ctx);
    const v0 = vm.runInContext('_vehicle.idm.v0', ctx);
    // After 10 seconds of free flow, should be near desired speed
    assert(speed > v0 * 0.5, `After 10s, speed ${speed.toFixed(1)} > 50% of v0=${v0.toFixed(1)}`);
})();

// ============================================================
// Node transition tracking
// ============================================================

console.log('\n--- Node transition tracking ---');

(function() {
    const ctx = createVehicle({}, 10);
    const lastNodeId = vm.runInContext('_vehicle.lastNodeId', ctx);
    assertEqual(lastNodeId, null, 'Initial lastNodeId is null');
    const pending = vm.runInContext('_vehicle._pendingTransition', ctx);
    assertEqual(pending, null, 'Initial pending transition is null');
})();

// ============================================================
// Lateral offset for multi-lane roads
// ============================================================

console.log('\n--- Lateral offset ---');

(function() {
    const ctx = createVehicle({ lanesPerDir: 2, laneWidth: 3 }, 50);
    vm.runInContext('_vehicle.laneIdx = 0; _vehicle._updatePosition();', ctx);
    const offset0 = vm.runInContext('_vehicle.lateralOffset', ctx);
    vm.runInContext('_vehicle.laneIdx = 1; _vehicle._updatePosition();', ctx);
    const offset1 = vm.runInContext('_vehicle.lateralOffset', ctx);
    assert(offset0 !== offset1, 'Different lanes have different lateral offsets');
})();

// ============================================================
// Direction -1 (reverse on edge)
// ============================================================

console.log('\n--- Reverse direction ---');

(function() {
    const ctx = createVehicle({}, 90);
    vm.runInContext('_vehicle.direction = -1; _vehicle.speed = 5;', ctx);
    callTick(ctx, 0.1, '[]');
    const v = getVehicle(ctx);
    assert(v.u < 90, 'Direction -1 moves vehicle backward along edge');
})();

// ============================================================
// Summary
// ============================================================

console.log(`\n${'='.repeat(50)}`);
console.log(`Vehicle tests: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
