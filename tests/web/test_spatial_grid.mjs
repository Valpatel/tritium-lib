// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/sim/spatial-grid.js — SpatialGrid
// Run: node tests/web/test_spatial_grid.mjs

import { SpatialGrid } from '../../web/sim/spatial-grid.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

console.log('=== SpatialGrid Tests ===');

// ── Constructor ───────────────────────────────────────────────────

test('constructor sets default cell size', () => {
    const grid = new SpatialGrid();
    assert.strictEqual(grid.cellSize, 20);
});

test('constructor accepts custom cell size', () => {
    const grid = new SpatialGrid(50);
    assert.strictEqual(grid.cellSize, 50);
});

// ── Insert and getNearby ──────────────────────────────────────────

test('insert and retrieve a single entity', () => {
    const grid = new SpatialGrid(10);
    const entity = { id: 'a', x: 5, z: 5 };
    grid.insert(entity);
    const nearby = grid.getNearby(5, 5);
    assert.ok(nearby.includes(entity), 'should find the inserted entity');
});

test('getNearby returns entities from same cell', () => {
    const grid = new SpatialGrid(10);
    const e1 = { id: 'a', x: 5, z: 5 };
    const e2 = { id: 'b', x: 7, z: 3 };
    grid.insert(e1);
    grid.insert(e2);
    const nearby = grid.getNearby(5, 5);
    assert.ok(nearby.includes(e1));
    assert.ok(nearby.includes(e2));
});

test('getNearby returns entities from adjacent cells', () => {
    const grid = new SpatialGrid(10);
    // Place entities in adjacent cells
    const e1 = { id: 'a', x: 5, z: 5 };     // cell (0,0)
    const e2 = { id: 'b', x: 15, z: 5 };    // cell (1,0)
    grid.insert(e1);
    grid.insert(e2);
    // Query from cell (0,0) — should find both since cell (1,0) is adjacent
    const nearby = grid.getNearby(5, 5);
    assert.ok(nearby.includes(e1));
    assert.ok(nearby.includes(e2));
});

test('getNearby does not return entities from distant cells', () => {
    const grid = new SpatialGrid(10);
    const close = { id: 'close', x: 5, z: 5 };
    const far = { id: 'far', x: 100, z: 100 };
    grid.insert(close);
    grid.insert(far);
    const nearby = grid.getNearby(5, 5);
    assert.ok(nearby.includes(close), 'should include close entity');
    assert.ok(!nearby.includes(far), 'should not include far entity');
});

test('getNearby returns empty array for empty grid', () => {
    const grid = new SpatialGrid(10);
    const nearby = grid.getNearby(50, 50);
    assert.deepStrictEqual(nearby, []);
});

// ── Clear ─────────────────────────────────────────────────────────

test('clear removes all entities', () => {
    const grid = new SpatialGrid(10);
    grid.insert({ id: 'a', x: 5, z: 5 });
    grid.insert({ id: 'b', x: 15, z: 15 });
    grid.clear();
    const nearby = grid.getNearby(5, 5);
    assert.deepStrictEqual(nearby, []);
});

// ── getOnEdge ─────────────────────────────────────────────────────

test('getOnEdge filters vehicles by edge ID', () => {
    const grid = new SpatialGrid(10);
    const v1 = { id: 'v1', edge: { id: 'e1' } };
    const v2 = { id: 'v2', edge: { id: 'e2' } };
    const v3 = { id: 'v3', edge: { id: 'e1' } };
    const result = grid.getOnEdge('e1', [v1, v2, v3]);
    assert.strictEqual(result.length, 2);
    assert.ok(result.includes(v1));
    assert.ok(result.includes(v3));
});

test('getOnEdge handles vehicles with null edge', () => {
    const grid = new SpatialGrid(10);
    const v1 = { id: 'v1', edge: null };
    const v2 = { id: 'v2', edge: { id: 'e1' } };
    const result = grid.getOnEdge('e1', [v1, v2]);
    assert.strictEqual(result.length, 1);
    assert.ok(result.includes(v2));
});

// ── Stats ─────────────────────────────────────────────────────────

test('stats reports correct counts', () => {
    const grid = new SpatialGrid(10);
    grid.insert({ id: 'a', x: 5, z: 5 });
    grid.insert({ id: 'b', x: 5, z: 5 });
    grid.insert({ id: 'c', x: 50, z: 50 });
    const s = grid.stats();
    assert.strictEqual(s.entities, 3);
    assert.strictEqual(s.cells, 2);
    assert.strictEqual(s.maxCellSize, 2);
});

test('stats on empty grid', () => {
    const grid = new SpatialGrid(10);
    const s = grid.stats();
    assert.strictEqual(s.entities, 0);
    assert.strictEqual(s.cells, 0);
    assert.strictEqual(s.maxCellSize, 0);
    assert.strictEqual(s.avgCellSize, 0);
});

// ── Cell key correctness ──────────────────────────────────────────

test('entities at negative coordinates are handled', () => {
    const grid = new SpatialGrid(10);
    const e1 = { id: 'a', x: -5, z: -5 };
    grid.insert(e1);
    const nearby = grid.getNearby(-5, -5);
    assert.ok(nearby.includes(e1), 'should find entity at negative coordinates');
});

test('entities right on cell boundary are assigned correctly', () => {
    const grid = new SpatialGrid(10);
    const e1 = { id: 'a', x: 10, z: 10 };  // exactly on boundary between cells (0,0) and (1,1)
    grid.insert(e1);
    const nearby = grid.getNearby(10, 10);
    assert.ok(nearby.includes(e1), 'should find entity on cell boundary');
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
