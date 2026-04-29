#!/usr/bin/env node
// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * RoadNetwork tests — buildFromOSM, node merging, edge creation,
 * adjacency, Dijkstra pathfinding, nearestNode, randomEdge, stats.
 *
 * Run: node tests/js/test_road_network.js
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

// Load road-network.js into sandbox, stripping ES module syntax
const code = fs.readFileSync(__dirname + '/../../web/sim/road-network.js', 'utf8');
const plainCode = code
    .replace(/^export\s+/gm, '')
    .replace(/^import\s+.*$/gm, '');

function createCtx() {
    const sandbox = {
        Math, console, Object, Array, Number, String, Set, Map, Infinity, NaN,
        parseInt, parseFloat, JSON, undefined,
    };
    const ctx = vm.createContext(sandbox);
    vm.runInContext(plainCode, ctx);
    vm.runInContext('var _rn = new RoadNetwork();', ctx);
    return ctx;
}

// Helper to run buildFromOSM inside VM
function build(ctx, roads, mergeRadius) {
    vm.runInContext(`_rn.buildFromOSM(${JSON.stringify(roads)}${mergeRadius !== undefined ? ', ' + mergeRadius : ''});`, ctx);
}

// Helper to read serializable RN state
function q(ctx, expr) {
    return vm.runInContext(expr, ctx);
}

function qj(ctx, expr) {
    return JSON.parse(vm.runInContext(`JSON.stringify(${expr})`, ctx));
}

// Grid road data
const GRID = [
    { points: [[0, 0], [100, 0]], class: 'residential', width: 6, lanes: 2, oneway: false },
    { points: [[0, 0], [0, 100]], class: 'residential', width: 6, lanes: 2, oneway: false },
    { points: [[100, 0], [100, 100]], class: 'residential', width: 6, lanes: 2, oneway: false },
    { points: [[0, 100], [100, 100]], class: 'residential', width: 6, lanes: 2, oneway: false },
];

// ============================================================
// Constructor
// ============================================================

console.log('\n--- RoadNetwork Constructor ---');

(function() {
    const ctx = createCtx();
    assertEqual(q(ctx, 'Object.keys(_rn.nodes).length'), 0, 'New network has 0 nodes');
    assertEqual(q(ctx, '_rn.edges.length'), 0, 'New network has 0 edges');
    assertEqual(q(ctx, 'Object.keys(_rn.adjList).length'), 0, 'New network has empty adjList');
    assertEqual(q(ctx, 'Object.keys(_rn.edgeById).length'), 0, 'New network has empty edgeById');
})();

// ============================================================
// buildFromOSM — empty/null input
// ============================================================

console.log('\n--- buildFromOSM empty/null ---');

(function() {
    const ctx = createCtx();
    vm.runInContext('_rn.buildFromOSM(null)', ctx);
    assertEqual(q(ctx, '_rn.edges.length'), 0, 'null input => 0 edges');
    assertEqual(q(ctx, 'Object.keys(_rn.nodes).length'), 0, 'null input => 0 nodes');
})();

(function() {
    const ctx = createCtx();
    vm.runInContext('_rn.buildFromOSM([])', ctx);
    assertEqual(q(ctx, '_rn.edges.length'), 0, 'empty array => 0 edges');
})();

(function() {
    const ctx = createCtx();
    vm.runInContext('_rn.buildFromOSM(undefined)', ctx);
    assertEqual(q(ctx, '_rn.edges.length'), 0, 'undefined input => 0 edges');
})();

// ============================================================
// buildFromOSM — non-vehicle roads filtered out
// ============================================================

console.log('\n--- buildFromOSM filters non-vehicle roads ---');

(function() {
    const ctx = createCtx();
    build(ctx, [
        { points: [[0, 0], [100, 0]], class: 'footway', width: 2 },
        { points: [[0, 0], [0, 100]], class: 'cycleway', width: 2 },
        { points: [[0, 0], [100, 100]], class: 'path', width: 1 },
    ]);
    assertEqual(q(ctx, '_rn.edges.length'), 0, 'footway/cycleway/path all filtered');
    assertEqual(q(ctx, 'Object.keys(_rn.nodes).length'), 0, 'no nodes from non-vehicle roads');
})();

// ============================================================
// buildFromOSM — single road
// ============================================================

console.log('\n--- buildFromOSM single road ---');

(function() {
    const ctx = createCtx();
    build(ctx, [{ points: [[0, 0], [100, 0]], class: 'residential', width: 6, lanes: 2 }]);
    assertEqual(q(ctx, 'Object.keys(_rn.nodes).length'), 2, 'Single road creates 2 nodes');
    assertEqual(q(ctx, '_rn.edges.length'), 1, 'Single road creates 1 edge');
    assertApprox(q(ctx, '_rn.edges[0].length'), 100, 0.1, 'Edge length is 100');
    assertEqual(q(ctx, '_rn.edges[0].roadClass'), 'residential', 'Edge has correct roadClass');
})();

// ============================================================
// buildFromOSM — grid network (node merging)
// ============================================================

console.log('\n--- buildFromOSM grid network ---');

(function() {
    const ctx = createCtx();
    build(ctx, GRID);
    assertEqual(q(ctx, 'Object.keys(_rn.nodes).length'), 4, 'Grid creates 4 nodes (corners merged)');
    assertEqual(q(ctx, '_rn.edges.length'), 4, 'Grid creates 4 edges');
})();

// ============================================================
// Node merging — nearby endpoints
// ============================================================

console.log('\n--- Node merging ---');

(function() {
    const ctx = createCtx();
    build(ctx, [
        { points: [[0, 0], [100, 0]], class: 'residential', width: 6 },
        { points: [[2, 2], [100, 100]], class: 'residential', width: 6 },
    ]);
    assertEqual(q(ctx, 'Object.keys(_rn.nodes).length'), 3, 'Close endpoints (~2.8m) merge into one node');
})();

(function() {
    const ctx = createCtx();
    build(ctx, [
        { points: [[0, 0], [100, 0]], class: 'residential', width: 6 },
        { points: [[10, 10], [100, 100]], class: 'residential', width: 6 },
    ]);
    assertEqual(q(ctx, 'Object.keys(_rn.nodes).length'), 4, 'Distant endpoints (~14m) remain separate');
})();

(function() {
    const ctx = createCtx();
    build(ctx, [
        { points: [[0, 0], [100, 0]], class: 'residential', width: 6 },
        { points: [[8, 8], [100, 100]], class: 'residential', width: 6 },
    ], 15);
    assertEqual(q(ctx, 'Object.keys(_rn.nodes).length'), 3, 'Custom merge radius 15m merges ~11m apart');
})();

// ============================================================
// Edge properties
// ============================================================

console.log('\n--- Edge properties ---');

(function() {
    const ctx = createCtx();
    build(ctx, [{ points: [[0, 0], [50, 0], [100, 0]], class: 'primary', width: 12, lanes: 4 }]);
    assertEqual(q(ctx, '_rn.edges[0].roadClass'), 'primary', 'Edge roadClass is primary');
    assertEqual(q(ctx, '_rn.edges[0].lanesPerDir'), 2, '4 total lanes, not oneway => 2 per dir');
    assertApprox(q(ctx, '_rn.edges[0].length'), 100, 0.1, 'Straight edge length computed correctly');
    assertEqual(q(ctx, '_rn.edges[0].waypoints.length'), 3, 'Waypoints preserved');
    assertEqual(q(ctx, '_rn.edges[0].oneway'), false, 'oneway defaults to false');
    assertEqual(q(ctx, '_rn.edges[0].bridge'), false, 'bridge defaults to false');
})();

(function() {
    const ctx = createCtx();
    build(ctx, [{ points: [[0, 0], [100, 0]], class: 'motorway', width: 14, lanes: 3, oneway: true, bridge: true }]);
    assertEqual(q(ctx, '_rn.edges[0].lanesPerDir'), 3, 'Oneway 3 lanes => 3 per dir');
    assertEqual(q(ctx, '_rn.edges[0].oneway'), true, 'oneway set to true');
    assertEqual(q(ctx, '_rn.edges[0].bridge'), true, 'bridge set to true');
})();

(function() {
    const ctx = createCtx();
    build(ctx, [{ points: [[0, 0], [100, 0]], class: 'residential', maxspeed: '30' }]);
    assertApprox(q(ctx, '_rn.edges[0].speedLimit'), 30 / 3.6, 0.1, 'maxspeed 30 km/h => ~8.33 m/s');
})();

// ============================================================
// Edge length — diagonal
// ============================================================

console.log('\n--- Edge length computation ---');

(function() {
    const ctx = createCtx();
    build(ctx, [{ points: [[0, 0], [30, 40]], class: 'residential' }]);
    assertApprox(q(ctx, '_rn.edges[0].length'), 50, 0.1, '3-4-5 triangle gives length 50');
})();

(function() {
    const ctx = createCtx();
    build(ctx, [{ points: [[0, 0], [30, 40], [60, 80]], class: 'residential' }]);
    assertApprox(q(ctx, '_rn.edges[0].length'), 100, 0.1, 'Multi-segment length sums correctly');
})();

// ============================================================
// Adjacency list
// ============================================================

console.log('\n--- Adjacency list ---');

(function() {
    const ctx = createCtx();
    build(ctx, GRID);
    const nodeIds = qj(ctx, 'Object.keys(_rn.nodes)');
    for (const nodeId of nodeIds) {
        assertEqual(q(ctx, `_rn.adjList["${nodeId}"].length`), 2, `Node ${nodeId} has degree 2`);
    }
})();

(function() {
    const ctx = createCtx();
    build(ctx, [
        { points: [[0, 50], [50, 50]], class: 'residential' },
        { points: [[50, 50], [100, 50]], class: 'residential' },
        { points: [[50, 50], [50, 100]], class: 'residential' },
    ]);
    const centralCount = q(ctx, `(function() {
        var cnt = 0;
        for (var id in _rn.nodes) { if (_rn.adjList[id].length === 3) cnt++; }
        return cnt;
    })()`);

    assertEqual(centralCount, 1, 'T-intersection has one node with degree 3');
})();

// ============================================================
// edgeById
// ============================================================

console.log('\n--- edgeById ---');

(function() {
    const ctx = createCtx();
    build(ctx, [
        { points: [[0, 0], [100, 0]], class: 'residential' },
        { points: [[0, 0], [0, 100]], class: 'residential' },
    ]);
    assertEqual(q(ctx, '_rn.edgeById["e0"] !== undefined'), true, 'edgeById has e0');
    assertEqual(q(ctx, '_rn.edgeById["e1"] !== undefined'), true, 'edgeById has e1');
    assertEqual(q(ctx, '_rn.edgeById["e0"] === _rn.edges[0]'), true, 'edgeById e0 matches edges[0]');
})();

// ============================================================
// Node types / degrees
// ============================================================

console.log('\n--- Node types ---');

(function() {
    const ctx = createCtx();
    build(ctx, [{ points: [[0, 0], [100, 0]], class: 'residential' }]);
    const allDeadEnd = q(ctx, `
        (function() {
            var types = [];
            for (var id in _rn.nodes) types.push(_rn.nodes[id].type);
            return types.every(function(t) { return t === 'dead-end'; });
        })()
    `);
    assert(allDeadEnd, 'All endpoints of single road are dead-end');
})();

(function() {
    const ctx = createCtx();
    build(ctx, [
        { points: [[0, 50], [50, 50]], class: 'residential' },
        { points: [[50, 50], [100, 50]], class: 'residential' },
        { points: [[50, 0], [50, 50]], class: 'residential' },
        { points: [[50, 50], [50, 100]], class: 'residential' },
    ]);
    const found = q(ctx, `(function() {
        for (var id in _rn.nodes) { if (_rn.nodes[id].type === '4+-way') return true; }
        return false;
    })()`);
    assert(found, '4-way intersection detected');
})();

// ============================================================
// Dijkstra pathfinding
// ============================================================

console.log('\n--- Dijkstra pathfinding ---');

(function() {
    const ctx = createCtx();
    build(ctx, GRID);
    const selfLen = q(ctx, `(function() {
        var ids = Object.keys(_rn.nodes);
        return _rn.findPath(ids[0], ids[0]).length;
    })()`);

    assertEqual(selfLen, 0, 'Path to self is empty');
})();

(function() {
    const ctx = createCtx();
    build(ctx, GRID);
    const pathLen = q(ctx, `(function() {
        var ids = Object.keys(_rn.nodes);
        return _rn.findPath(ids[0], ids[ids.length - 1]).length;
    })()`);
    assert(pathLen > 0, 'Path exists between corners of grid');
})();

(function() {
    const ctx = createCtx();
    build(ctx, GRID);
    const hasEdge = q(ctx, `(function() {
        var ids = Object.keys(_rn.nodes);
        var path = _rn.findPath(ids[0], ids[ids.length - 1]);
        return path.every(function(s) { return s.edge !== undefined && s.nodeId !== undefined; });
    })()`);
    assert(hasEdge, 'All path steps have edge and nodeId');
})();

(function() {
    const ctx = createCtx();
    build(ctx, [
        { points: [[0, 0], [100, 0]], class: 'residential' },
        { points: [[500, 500], [600, 500]], class: 'residential' },
    ]);
    const pathLen = q(ctx, `(function() {
        var ids = Object.keys(_rn.nodes);
        return _rn.findPath(ids[0], ids[ids.length - 1]).length;
    })()`);
    assertEqual(pathLen, 0, 'No path between disconnected roads');
})();

(function() {
    const ctx = createCtx();
    const pathLen = q(ctx, '_rn.findPath("nonexistent1", "nonexistent2").length');
    assertEqual(pathLen, 0, 'Non-existent nodes return empty path');
})();

// ============================================================
// Dijkstra — shortest path by distance
// ============================================================

console.log('\n--- Dijkstra shortest path ---');

(function() {
    const ctx = createCtx();
    build(ctx, [
        { points: [[0, 0], [10, 0]], class: 'residential' },
        { points: [[0, 0], [3, 5.196]], class: 'residential' },
        { points: [[10, 0], [3, 5.196]], class: 'residential' },
    ]);
    const directLen = q(ctx, '_rn.findPath(_rn.edges[0].from, _rn.edges[0].to).length');
    assertEqual(directLen, 1, 'Direct A-B path is 1 edge');
})();

// ============================================================
// Dijkstra — one-way roads
// ============================================================

console.log('\n--- One-way road pathfinding ---');

(function() {
    const ctx = createCtx();
    build(ctx, [
        { points: [[0, 0], [100, 0]], class: 'residential', oneway: true },
        { points: [[100, 0], [50, 50]], class: 'residential' },
        { points: [[50, 50], [0, 0]], class: 'residential' },
    ]);
    const fwdLen = q(ctx, '_rn.findPath(_rn.edges[0].from, _rn.edges[0].to).length');
    assert(fwdLen > 0, 'Forward path on one-way exists');
    const revLen = q(ctx, '_rn.findPath(_rn.edges[0].to, _rn.edges[0].from).length');
    assert(revLen > 0, 'Reverse path exists via return route');
})();

// ============================================================
// nearestNode
// ============================================================

console.log('\n--- nearestNode ---');

(function() {
    const ctx = createCtx();
    build(ctx, GRID);
    const dist = q(ctx, '_rn.nearestNode(1, 1).dist');
    assert(dist < 5, 'Near (0,0) finds node within 5m');
    const hasId = q(ctx, '_rn.nearestNode(1, 1).nodeId !== undefined');
    assert(hasId, 'Result has nodeId');
})();

(function() {
    const ctx = createCtx();
    build(ctx, GRID);
    const dist = q(ctx, '_rn.nearestNode(99, 99).dist');
    assert(dist < 5, 'Near (100,100) finds close node');
})();

(function() {
    const ctx = createCtx();
    const result = q(ctx, '_rn.nearestNode(50, 50)');
    assertEqual(result, null, 'Empty network returns null');
})();

// ============================================================
// randomEdge
// ============================================================

console.log('\n--- randomEdge ---');

(function() {
    const ctx = createCtx();
    build(ctx, GRID);
    const hasId = q(ctx, '_rn.randomEdge().id !== undefined');
    assert(hasId, 'randomEdge returns an edge with id');
    const length = q(ctx, '_rn.randomEdge().length');
    assert(length > 0, 'Edge has positive length');
})();

(function() {
    const ctx = createCtx();
    const result = q(ctx, '_rn.randomEdge()');
    assertEqual(result, null, 'Empty network randomEdge returns null');
})();

// ============================================================
// stats
// ============================================================

console.log('\n--- stats ---');

(function() {
    const ctx = createCtx();
    build(ctx, GRID);
    const s = qj(ctx, '_rn.stats()');
    assertEqual(s.nodes, 4, 'Stats: 4 nodes');
    assertEqual(s.edges, 4, 'Stats: 4 edges');
    assertEqual(s.roadClasses.residential, 4, 'Stats: 4 residential roads');
    assertApprox(s.totalLengthM, 400, 1, 'Stats: total length ~400m');
})();

// ============================================================
// buildFromOSM — roads with < 2 points are skipped
// ============================================================

console.log('\n--- Edge cases ---');

(function() {
    const ctx = createCtx();
    build(ctx, [
        { points: [[0, 0]], class: 'residential' },
        { points: [[0, 0], [100, 0]], class: 'residential' },
    ]);
    assertEqual(q(ctx, '_rn.edges.length'), 1, 'Road with 1 point is skipped');
})();

(function() {
    const ctx = createCtx();
    build(ctx, [
        { points: null, class: 'residential' },
        { points: [[0, 0], [100, 0]], class: 'residential' },
    ]);
    assertEqual(q(ctx, '_rn.edges.length'), 1, 'Road with null points is skipped');
})();

// ============================================================
// buildFromOSM chaining
// ============================================================

console.log('\n--- Chaining ---');

(function() {
    const ctx = createCtx();
    const same = q(ctx, `_rn.buildFromOSM(${JSON.stringify(GRID)}) === _rn`);
    assertEqual(same, true, 'buildFromOSM returns this for chaining');
})();

// ============================================================
// Mixed road classes
// ============================================================

console.log('\n--- Mixed road classes ---');

(function() {
    const ctx = createCtx();
    build(ctx, [
        { points: [[0, 0], [100, 0]], class: 'motorway', width: 14, lanes: 4, oneway: true },
        { points: [[100, 0], [200, 0]], class: 'secondary', width: 8, lanes: 2 },
        { points: [[200, 0], [300, 0]], class: 'service', width: 4, lanes: 1 },
    ]);
    assertEqual(q(ctx, '_rn.edges.length'), 3, '3 roads of different classes');
    const s = qj(ctx, '_rn.stats()');
    assertEqual(s.roadClasses.motorway, 1, '1 motorway');
    assertEqual(s.roadClasses.secondary, 1, '1 secondary');
    assertEqual(s.roadClasses.service, 1, '1 service');
})();

// ============================================================
// Lane width computation
// ============================================================

console.log('\n--- Lane width ---');

(function() {
    const ctx = createCtx();
    build(ctx, [{ points: [[0, 0], [100, 0]], class: 'residential', width: 12, lanes: 4 }]);
    assertEqual(q(ctx, '_rn.edges[0].laneWidth'), 3, 'Lane width computed correctly');
})();

(function() {
    const ctx = createCtx();
    build(ctx, [{ points: [[0, 0], [100, 0]], class: 'service', width: 3, lanes: 2 }]);
    // width=3, lanesPerDir=1, laneWidth = max(2, 3 / (1 * 2)) = max(2, 1.5) = 2
    assertEqual(q(ctx, '_rn.edges[0].laneWidth'), 2, 'Lane width clamped to minimum 2');
})();

// ============================================================
// Rebuild clears old data
// ============================================================

console.log('\n--- Rebuild clears old data ---');

(function() {
    const ctx = createCtx();
    build(ctx, GRID);
    assertEqual(q(ctx, '_rn.edges.length'), 4, 'First build: 4 edges');
    build(ctx, [{ points: [[0, 0], [100, 0]], class: 'residential' }]);
    assertEqual(q(ctx, '_rn.edges.length'), 1, 'Second build replaces with 1 edge');
    assertEqual(q(ctx, 'Object.keys(_rn.nodes).length'), 2, 'Second build: 2 nodes');
})();

// ============================================================
// Summary
// ============================================================

console.log(`\n${'='.repeat(50)}`);
console.log(`Road network tests: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
