// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/sim/road-network.js — RoadNetwork
// Run: node tests/web/test_road_network.mjs

import { RoadNetwork } from '../../web/sim/road-network.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

// Helper: build a simple road with two endpoints
function makeRoad(x1, z1, x2, z2, cls = 'residential', opts = {}) {
    return {
        points: [[x1, z1], [x2, z2]],
        class: cls,
        width: opts.width || 6,
        lanes: opts.lanes || 2,
        oneway: opts.oneway || false,
        bridge: opts.bridge || false,
    };
}

console.log('=== RoadNetwork Tests ===');

// ── Constructor ───────────────────────────────────────────────────

test('empty RoadNetwork has no nodes or edges', () => {
    const rn = new RoadNetwork();
    assert.strictEqual(Object.keys(rn.nodes).length, 0);
    assert.strictEqual(rn.edges.length, 0);
});

// ── buildFromOSM ──────────────────────────────────────────────────

test('buildFromOSM with null/empty input', () => {
    const rn = new RoadNetwork();
    rn.buildFromOSM(null);
    assert.strictEqual(rn.edges.length, 0);
    rn.buildFromOSM([]);
    assert.strictEqual(rn.edges.length, 0);
});

test('buildFromOSM with single road creates 2 nodes and 1 edge', () => {
    const rn = new RoadNetwork();
    const roads = [makeRoad(0, 0, 100, 0)];
    rn.buildFromOSM(roads);
    assert.strictEqual(Object.keys(rn.nodes).length, 2);
    assert.strictEqual(rn.edges.length, 1);
});

test('buildFromOSM computes edge length correctly', () => {
    const rn = new RoadNetwork();
    const roads = [makeRoad(0, 0, 30, 40)]; // 3-4-5 triangle: length = 50
    rn.buildFromOSM(roads);
    assert.ok(Math.abs(rn.edges[0].length - 50) < 0.1, `expected 50, got ${rn.edges[0].length}`);
});

test('buildFromOSM merges nearby endpoints into intersection nodes', () => {
    const rn = new RoadNetwork();
    // Two roads meeting at approximately (100, 0)
    const roads = [
        makeRoad(0, 0, 100, 0),
        makeRoad(101, 0, 200, 0),  // within default mergeRadius=5
    ];
    rn.buildFromOSM(roads);
    // Should create 3 nodes (left, merged middle, right), 2 edges
    assert.strictEqual(Object.keys(rn.nodes).length, 3);
    assert.strictEqual(rn.edges.length, 2);
});

test('buildFromOSM filters non-vehicle road types', () => {
    const rn = new RoadNetwork();
    const roads = [
        makeRoad(0, 0, 100, 0, 'residential'),
        makeRoad(0, 50, 100, 50, 'footway'),     // should be filtered
        makeRoad(0, 100, 100, 100, 'cycleway'),   // should be filtered
    ];
    rn.buildFromOSM(roads);
    assert.strictEqual(rn.edges.length, 1, 'should only build residential road');
});

test('buildFromOSM sets lanes per direction', () => {
    const rn = new RoadNetwork();
    const roads = [makeRoad(0, 0, 100, 0, 'primary', { lanes: 4 })];
    rn.buildFromOSM(roads);
    assert.strictEqual(rn.edges[0].lanesPerDir, 2); // 4 total / 2 = 2 per direction
});

test('buildFromOSM oneway road keeps all lanes in one direction', () => {
    const rn = new RoadNetwork();
    const roads = [makeRoad(0, 0, 100, 0, 'primary', { lanes: 3, oneway: true })];
    rn.buildFromOSM(roads);
    assert.strictEqual(rn.edges[0].lanesPerDir, 3);
    assert.strictEqual(rn.edges[0].oneway, true);
});

test('buildFromOSM sets roadClass', () => {
    const rn = new RoadNetwork();
    const roads = [makeRoad(0, 0, 100, 0, 'motorway')];
    rn.buildFromOSM(roads);
    assert.strictEqual(rn.edges[0].roadClass, 'motorway');
});

test('buildFromOSM updates node degree and type', () => {
    const rn = new RoadNetwork();
    // T-intersection: 3 roads meeting at (100, 0)
    const roads = [
        makeRoad(0, 0, 100, 0),
        makeRoad(101, 0, 200, 0),    // merged endpoint near (100,0)
        makeRoad(100, -1, 100, 100),  // merged endpoint near (100,0)
    ];
    rn.buildFromOSM(roads);
    // Find the intersection node (highest degree)
    let maxDegree = 0;
    let intersectionType = '';
    for (const id in rn.nodes) {
        if (rn.nodes[id].degree > maxDegree) {
            maxDegree = rn.nodes[id].degree;
            intersectionType = rn.nodes[id].type;
        }
    }
    assert.ok(maxDegree >= 3, `expected degree >= 3, got ${maxDegree}`);
    assert.strictEqual(intersectionType, '3-way');
});

// ── findPath (Dijkstra) ───────────────────────────────────────────

test('findPath returns empty for same node', () => {
    const rn = new RoadNetwork();
    rn.buildFromOSM([makeRoad(0, 0, 100, 0)]);
    const nodeIds = Object.keys(rn.nodes);
    const path = rn.findPath(nodeIds[0], nodeIds[0]);
    assert.deepStrictEqual(path, []);
});

test('findPath finds direct path on single edge', () => {
    const rn = new RoadNetwork();
    rn.buildFromOSM([makeRoad(0, 0, 100, 0)]);
    const nodeIds = Object.keys(rn.nodes);
    const path = rn.findPath(nodeIds[0], nodeIds[1]);
    assert.strictEqual(path.length, 1);
    assert.ok(path[0].edge, 'path step should have edge');
});

test('findPath finds multi-hop path', () => {
    const rn = new RoadNetwork();
    // Chain: A --100m-- B --100m-- C
    const roads = [
        makeRoad(0, 0, 100, 0),
        makeRoad(101, 0, 200, 0),
    ];
    rn.buildFromOSM(roads);
    const nodeIds = Object.keys(rn.nodes);
    // Find the two endpoint nodes (degree 1)
    const endpoints = nodeIds.filter(id => rn.nodes[id].degree === 1);
    assert.strictEqual(endpoints.length, 2, 'should have 2 endpoints');
    const path = rn.findPath(endpoints[0], endpoints[1]);
    assert.strictEqual(path.length, 2, 'should traverse 2 edges');
});

test('findPath returns empty for unreachable node', () => {
    const rn = new RoadNetwork();
    rn.buildFromOSM([makeRoad(0, 0, 100, 0)]);
    const path = rn.findPath('n0', 'nonexistent');
    assert.deepStrictEqual(path, []);
});

test('findPath respects oneway streets', () => {
    const rn = new RoadNetwork();
    const roads = [
        makeRoad(0, 0, 100, 0, 'residential', { oneway: true }),
    ];
    rn.buildFromOSM(roads);
    const nodeIds = Object.keys(rn.nodes);
    // Forward path should work
    const fromNode = rn.edges[0].from;
    const toNode = rn.edges[0].to;
    const fwd = rn.findPath(fromNode, toNode);
    assert.strictEqual(fwd.length, 1, 'forward path should exist');
    // Reverse should fail (one-way)
    const rev = rn.findPath(toNode, fromNode);
    assert.deepStrictEqual(rev, [], 'reverse path should be blocked by one-way');
});

// ── nearestNode ───────────────────────────────────────────────────

test('nearestNode finds closest node', () => {
    const rn = new RoadNetwork();
    rn.buildFromOSM([makeRoad(0, 0, 100, 0)]);
    const result = rn.nearestNode(5, 0);
    assert.ok(result, 'should find a node');
    assert.ok(result.nodeId, 'should have nodeId');
    assert.ok(result.dist < 10, `expected close distance, got ${result.dist}`);
});

test('nearestNode returns null for empty network', () => {
    const rn = new RoadNetwork();
    const result = rn.nearestNode(50, 50);
    assert.strictEqual(result, null);
});

// ── randomEdge ────────────────────────────────────────────────────

test('randomEdge returns an edge', () => {
    const rn = new RoadNetwork();
    rn.buildFromOSM([makeRoad(0, 0, 100, 0)]);
    const edge = rn.randomEdge();
    assert.ok(edge, 'should return an edge');
    assert.ok(edge.id, 'edge should have id');
});

test('randomEdge returns null for empty network', () => {
    const rn = new RoadNetwork();
    const edge = rn.randomEdge();
    assert.strictEqual(edge, null);
});

// ── stats ─────────────────────────────────────────────────────────

test('stats reports correct counts', () => {
    const rn = new RoadNetwork();
    rn.buildFromOSM([makeRoad(0, 0, 100, 0), makeRoad(101, 0, 200, 0)]);
    const s = rn.stats();
    assert.strictEqual(s.edges, 2);
    assert.strictEqual(s.nodes, 3);
    assert.ok(s.totalLengthM > 0, 'total length should be positive');
    assert.ok(s.roadClasses.residential > 0, 'should have residential count');
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
