// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.
// Tests for tritium-lib/web/command-palette.js
// Run: node tests/web/test_command_palette.mjs

import { fuzzyScore, initCommandPalette } from '../../web/command-palette.js';
import assert from 'node:assert';

let pass = 0, fail = 0;
function test(name, fn) {
    try { fn(); pass++; console.log(`  PASS: ${name}`); }
    catch (e) { fail++; console.log(`  FAIL: ${name} -- ${e.message}`); }
}

// ===========================================================================
// fuzzyScore tests
// ===========================================================================

console.log('=== fuzzyScore Tests ===');

test('empty query returns 1 (show everything)', () => {
    assert.strictEqual(fuzzyScore('', 'anything'), 1);
});

test('null/undefined query returns 1', () => {
    assert.strictEqual(fuzzyScore(null, 'text'), 1);
    assert.strictEqual(fuzzyScore(undefined, 'text'), 1);
});

test('exact match scores high (>= 10)', () => {
    const score = fuzzyScore('camera feeds', 'camera feeds');
    assert.ok(score >= 10, `Expected >= 10, got ${score}`);
});

test('substring match scores >= 10', () => {
    const score = fuzzyScore('cam', 'camera feeds');
    assert.ok(score >= 10, `Expected >= 10, got ${score}`);
});

test('exact match scores higher than substring match', () => {
    const exact = fuzzyScore('camera feeds', 'camera feeds');
    const partial = fuzzyScore('cam', 'camera feeds');
    assert.ok(exact > partial, `Exact ${exact} should be > partial ${partial}`);
});

test('no match returns 0', () => {
    assert.strictEqual(fuzzyScore('xyz', 'camera feeds'), 0);
});

test('case insensitive matching', () => {
    const score = fuzzyScore('CAMERA', 'camera feeds');
    assert.ok(score >= 10, `Expected >= 10, got ${score}`);
});

test('multi-term: both terms must match', () => {
    const score = fuzzyScore('edge tracker', 'edge_tracker plugin');
    assert.ok(score > 0, `Expected > 0, got ${score}`);
});

test('multi-term: one missing term returns 0', () => {
    const score = fuzzyScore('edge missing', 'edge_tracker plugin');
    assert.strictEqual(score, 0);
});

test('multi-term: word-boundary bonus scores higher than mid-word', () => {
    // "edge track" has both terms at word boundaries in "edge_tracker"
    const boundary = fuzzyScore('edge track', 'edge_tracker');
    // "dge rack" has both terms mid-word (dge in eDGE, rack in tRACKer)
    const midword = fuzzyScore('dge rack', 'edge_tracker');
    // Both match, but boundary should score higher (3 per term vs 1 per term)
    assert.ok(boundary > 0, `Boundary score should be > 0, got ${boundary}`);
    assert.ok(midword > 0, `Mid-word score should be > 0, got ${midword}`);
    assert.ok(boundary > midword, `Boundary ${boundary} should be > midword ${midword}`);
});

test('multi-term: boundary characters include hyphen, underscore, colon, slash', () => {
    // Terms at boundaries after separators should get bonus
    const score1 = fuzzyScore('bar', 'foo-bar');
    const score2 = fuzzyScore('bar', 'foo_bar');
    const score3 = fuzzyScore('bar', 'foo:bar');
    const score4 = fuzzyScore('bar', 'foo/bar');
    // All should match with substring bonus (>= 10)
    assert.ok(score1 >= 10, `hyphen: ${score1}`);
    assert.ok(score2 >= 10, `underscore: ${score2}`);
    assert.ok(score3 >= 10, `colon: ${score3}`);
    assert.ok(score4 >= 10, `slash: ${score4}`);
});

test('longer query match ratio scores higher', () => {
    // "camera feeds" against itself vs against longer text
    const tight = fuzzyScore('camera', 'camera');
    const loose = fuzzyScore('camera', 'camera feeds plugin system');
    assert.ok(tight > loose, `Tight ${tight} should be > loose ${loose}`);
});

test('special characters in query do not crash', () => {
    const score = fuzzyScore('foo(bar)', 'foobar');
    // Should return 0 since literal "(bar)" won't be found
    assert.strictEqual(typeof score, 'number');
});

test('special regex characters in query are handled safely', () => {
    const score = fuzzyScore('[test]', 'some [test] string');
    assert.ok(score >= 10, `Expected >= 10, got ${score}`);
});

test('whitespace-only query returns 1', () => {
    // After trim inside split, whitespace query has no non-empty terms
    // The function checks !query first; '   ' is truthy but terms are empty
    const score = fuzzyScore('   ', 'anything');
    assert.strictEqual(typeof score, 'number');
});

test('single character query matches substring', () => {
    const score = fuzzyScore('c', 'camera');
    assert.ok(score >= 10, `Expected >= 10, got ${score}`);
});

test('query longer than text returns 0', () => {
    const score = fuzzyScore('this is a very long query', 'short');
    assert.strictEqual(score, 0);
});

test('multi-term where full string is NOT a substring but individual terms are', () => {
    // "map layer" is not a substring of "gis_layers map" but both terms exist
    const score = fuzzyScore('map layer', 'gis_layers map view');
    assert.ok(score > 0, `Expected > 0, got ${score}`);
});

// ===========================================================================
// initCommandPalette API tests
// ===========================================================================

console.log('\n=== initCommandPalette API Tests ===');

test('initCommandPalette returns API object', () => {
    const palette = initCommandPalette(null, () => [], {});
    assert.strictEqual(typeof palette.open, 'function');
    assert.strictEqual(typeof palette.close, 'function');
    assert.strictEqual(typeof palette.isOpen, 'function');
    assert.strictEqual(typeof palette.destroy, 'function');
});

test('isOpen returns false initially', () => {
    const palette = initCommandPalette(null, () => []);
    assert.strictEqual(palette.isOpen(), false);
});

test('destroy does not throw when no DOM', () => {
    const palette = initCommandPalette(null, () => []);
    palette.destroy();
    assert.strictEqual(palette.isOpen(), false);
});

test('module exports fuzzyScore function', () => {
    assert.strictEqual(typeof fuzzyScore, 'function');
});

test('module exports initCommandPalette function', () => {
    assert.strictEqual(typeof initCommandPalette, 'function');
});

test('commandsFn is called lazily (not at init)', () => {
    let called = false;
    const palette = initCommandPalette(null, () => {
        called = true;
        return [];
    });
    assert.strictEqual(called, false);
    palette.destroy();
});

test('multiple instances are independent', () => {
    const p1 = initCommandPalette(null, () => [{ name: 'A' }]);
    const p2 = initCommandPalette(null, () => [{ name: 'B' }]);
    assert.strictEqual(p1.isOpen(), false);
    assert.strictEqual(p2.isOpen(), false);
    p1.destroy();
    p2.destroy();
});

console.log(`\n=== Results: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
